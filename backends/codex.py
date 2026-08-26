"""OpenAI Codex CLI backend implementation."""

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


class CodexTurnError(TurnError):
    """Raised when the Codex CLI returns something unusable."""


# Takes a *path* to the schema document, not the document itself — one of the
# places these CLIs disagree about more than a flag name.
SCHEMA_FLAG = "--output-schema"

# The events a failed turn arrives as, on stdout. `error` carries the message
# directly; `turn.failed` nests it under `error`.
ERROR_EVENT = "error"
FAILED_TURN_EVENT = "turn.failed"


def _reported_error_message(stdout: str) -> str | None:
    """Codex's own words for a failed turn, or None if it said none.

    A failure lands on *stdout* as an error event, while stderr at that moment
    holds only the CLI's "Reading additional input from stdin..." notice — so
    reporting the exit code and the stderr tail tells the user nothing about
    what actually went wrong. A rejected `--output-schema` is the case that
    made this matter: the API says which part of the schema it refused, and
    that sentence is the whole of the fix.

    Read last-first: the last thing the CLI said before giving up is the
    proximate failure, and an earlier event may be a warning it recovered
    from.
    """
    for event in reversed(json_objects(stdout)):
        etype = event.get("type")
        if etype == ERROR_EVENT:
            message = event.get("message")
        elif etype == FAILED_TURN_EVENT:
            nested = event.get(ERROR_EVENT)
            message = nested.get("message") if isinstance(nested, dict) else None
        else:
            continue
        if isinstance(message, str) and message:
            return message
    return None


def _failed_turn_message(returncode: int, stdout: str, stderr: str) -> str:
    """Prefer Codex's reported message; fall back to the exit and stderr tail."""
    message = _reported_error_message(stdout)
    if message is None:
        return f"codex exited {returncode}\nstderr: {stderr[-2000:]}"
    return f"codex reported an error: {message}"


class CodexBackend(AgentBackend):
    """Backend for OpenAI's Codex CLI (`codex`)."""

    name = "codex"

    def run_turn(
        self,
        prompt: str,
        session_id: str | None,
        cwd: Path,
        timeout: int = DEFAULT_TURN_TIMEOUT,
        schema: OutputSchema | None = None,
    ) -> TurnResult:
        args = ["codex", "exec"]
        if self.model is not None:
            args += ["--model", self.model]
        if self.reasoning_effort is not None:
            args += ["--config", f'model_reasoning_effort="{self.reasoning_effort}"']
        if session_id:
            args += ["resume", session_id]
        args += [prompt, "--json", "--skip-git-repo-check"]
        if schema is not None:
            args += [SCHEMA_FLAG, str(schema.path)]

        proc = run_cli_turn(self.name, args, prompt, session_id, cwd, timeout)
        if proc.returncode != 0:
            raise CodexTurnError(
                _failed_turn_message(proc.returncode, proc.stdout, proc.stderr)
            )
        return self._parse(proc.stdout, proc.stderr, schema)

    def _parse(
        self, stdout: str, stderr: str, schema: OutputSchema | None
    ) -> TurnResult:
        session_id = None
        reply_parts: list[str] = []
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = event.get("type")
            if etype == "thread.started":
                session_id = event.get("thread_id")
            elif etype == "item.completed":
                item = event.get("item") or {}
                if item.get("type") == "agent_message":
                    text = item.get("text")
                    if text:
                        reply_parts.append(text)
        if session_id is None:
            raise CodexTurnError(
                f"codex did not report a thread_id\nstdout: {stdout[-2000:]}"
                f"\nstderr: {stderr[-2000:]}"
            )
        reply = "\n".join(reply_parts)
        log.debug(
            "codex turn: parsed session=%s messages=%d reply_chars=%d",
            session_id,
            len(reply_parts),
            len(reply),
        )
        return TurnResult(
            session_id=session_id,
            reply=reply,
            raw=stdout,
            # No pre-parsed field of its own: codex publishes the object only
            # as the agent message's text.
            structured=structured_reply(schema, reply),
        )
