"""OpenCode CLI backend implementation."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from backends.base import (
    DEFAULT_TURN_TIMEOUT,
    AgentBackend,
    OutputSchema,
    TurnError,
    TurnResult,
    json_objects,
    run_cli_turn,
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


def _embedded_object(reply: str) -> dict | None:
    """The JSON object an unconstrained reply wrapped in prose or a fence.

    The other three CLIs are handed the schema and return the object as the
    whole reply, so `structured_reply` can parse the text as it stands. This
    one is not: the schema reaches the model as a line of prompt, and a model
    that has been *asked* for JSON still answers the way it answers everything
    else -- a ```json fence, or a sentence before the object.

    Measured against opencode 1.18.21 on 2026-08-22: every attempt came back
    fenced, `json.loads` of the whole reply failed each time, and the
    orchestrator's repair loop could not converge because re-asking produced
    the same fence. Without this scan the validate-and-repair fallback never
    reaches the validator at all, and every schema turn fails.

    The last object wins: a reply that reasons before answering puts the
    answer last, and the schema check that follows is what decides whether it
    is the right one.
    """
    candidates = json_objects(reply)
    return candidates[-1] if candidates else None


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

        proc = run_cli_turn(
            self.name,
            args,
            prompt=prompt,
            session_id=session_id,
            cwd=cwd,
            timeout=timeout,
            prompt_on_stdin=True,
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
        # Newline-joined, not concatenated: each part is a *completed*
        # block, so a turn that speaks either side of a tool call emits two
        # of them. Gluing them produces run-on text ('...directory.rhubarb'
        # in a live 1.18.21 turn) in a reply that is read by a human in a
        # pull-request comment and re-fed to the next agent.
        reply = "\n".join(part_text[part_id] for part_id in part_order)
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
            structured=structured_reply(schema, reply, _embedded_object(reply)),
        )
