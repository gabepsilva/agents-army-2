"""OpenCode CLI backend implementation."""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path

from backends.base import (
    DEFAULT_TURN_TIMEOUT,
    AgentBackend,
    OutputSchema,
    TurnError,
    TurnResult,
    describe_command,
    structured_reply,
)

log = logging.getLogger(__name__)


class OpenCodeTurnError(TurnError):
    """Raised when the OpenCode CLI returns something unusable."""


def _events(stdout: str) -> list[dict]:
    """Parse OpenCode's newline-delimited event stream, ignoring noise."""
    events: list[dict] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _error_detail(event: dict) -> str | None:
    """Return the useful detail from one OpenCode error event, if any."""
    error = event.get("error")
    if not isinstance(error, dict):
        return None
    data = error.get("data")
    if isinstance(data, dict):
        message = data.get("message")
        if isinstance(message, str) and message:
            return message
    name = error.get("name")
    if isinstance(name, str) and name:
        return name
    return None


def _reported_error(events: list[dict]) -> str | None:
    """Prefer the last OpenCode error event's actionable detail."""
    for event in reversed(events):
        if event.get("type") != "error":
            continue
        detail = _error_detail(event)
        if detail is not None:
            return detail
    return None


def _failed_turn_message(returncode: int, events: list[dict], stderr: str) -> str:
    """Prefer OpenCode's error detail, then the exit code and stderr tail."""
    detail = _reported_error(events)
    if detail is not None:
        return f"opencode reported an error: {detail}"
    return f"opencode exited {returncode}\nstderr: {stderr[-2000:]}"


class OpenCodeBackend(AgentBackend):
    """Backend for OpenCode 1.18.21 and later."""

    name = "opencode"
    enforces_schema = False

    def run_turn(
        self,
        prompt: str,
        session_id: str | None,
        cwd: Path,
        timeout: int = DEFAULT_TURN_TIMEOUT,
        schema: OutputSchema | None = None,
    ) -> TurnResult:
        cwd = cwd.absolute()
        # OpenCode resolves its project directory from PWD, so --dir remains
        # explicit even though subprocess.run also receives the same cwd.
        args = [
            "opencode",
            "run",
            "--format",
            "json",
            "--auto",
            "--dir",
            str(cwd),
        ]
        if self.model is not None:
            args += ["--model", self.model]
        if self.reasoning_effort is not None:
            args += ["--variant", self.reasoning_effort]
        if session_id:
            args += ["--session", session_id]

        log.debug(
            "opencode turn: cwd=%s resume=%s prompt_chars=%d timeout=%ds",
            cwd,
            bool(session_id),
            len(prompt),
            timeout,
        )
        log.debug("opencode turn: invoking %s", describe_command(args, prompt))
        started = time.monotonic()
        proc = subprocess.run(
            args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            # OpenCode joins positional arguments before sending them to the
            # model. Omitting that argument makes its no-value fallback read
            # stdin verbatim, preserving spaces, quotes, and newlines.
            input=prompt,
        )
        log.debug(
            "opencode turn: exited %d after %.1fs with %d chars of stdout",
            proc.returncode,
            time.monotonic() - started,
            len(proc.stdout),
        )
        events = _events(proc.stdout)
        if proc.returncode != 0:
            raise OpenCodeTurnError(
                _failed_turn_message(proc.returncode, events, proc.stderr)
            )
        return self._parse(proc.stdout, proc.stderr, schema, events)

    def _parse(
        self,
        stdout: str,
        stderr: str,
        schema: OutputSchema | None,
        events: list[dict],
    ) -> TurnResult:
        if any(event.get("type") == "error" for event in events):
            detail = _reported_error(events)
            if detail is None:
                raise OpenCodeTurnError("opencode reported an error event")
            raise OpenCodeTurnError(f"opencode reported an error: {detail}")

        session_id: str | None = None
        part_order: list[str] = []
        part_text: dict[str, str] = {}
        for event in events:
            reported_session = event.get("sessionID")
            if not isinstance(reported_session, str) or not reported_session.strip():
                raise OpenCodeTurnError(
                    f"opencode did not report a sessionID\n"
                    f"stdout: {stdout[-2000:]}\nstderr: {stderr[-2000:]}"
                )
            session_id = reported_session
            if event.get("type") != "text":
                continue
            part = event.get("part")
            if not isinstance(part, dict):
                continue
            part_id = part.get("id")
            text = part.get("text")
            if not isinstance(part_id, str) or not isinstance(text, str):
                continue
            if part_id not in part_text:
                part_order.append(part_id)
            part_text[part_id] = text

        if session_id is None:
            raise OpenCodeTurnError(
                f"opencode did not report a sessionID\n"
                f"stdout: {stdout[-2000:]}\nstderr: {stderr[-2000:]}"
            )
        reply = "".join(part_text[part_id] for part_id in part_order)
        log.debug(
            "opencode turn: parsed session=%s parts=%d reply_chars=%d",
            session_id,
            len(part_order),
            len(reply),
        )
        return TurnResult(
            session_id=session_id,
            reply=reply,
            raw=stdout,
            structured=structured_reply(schema, reply),
        )
