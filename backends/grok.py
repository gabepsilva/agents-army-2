"""Grok CLI backend implementation."""

from __future__ import annotations

import logging
from pathlib import Path

from backends.base import (
    DEFAULT_TURN_TIMEOUT,
    AgentBackend,
    OutputSchema,
    TurnError,
    TurnResult,
    json_objects,
    reply_text,
    run_cli_turn,
    stdout_for_error,
    structured_reply,
)

log = logging.getLogger(__name__)


class GrokTurnError(TurnError):
    """Raised when the Grok CLI returns something unusable."""


# Headless mode otherwise waits for tool approval, which a captured
# subprocess can never give. --always-approve is Grok's native name for
# the same opt-in Claude spells --permission-mode bypassPermissions.
ALWAYS_APPROVE_FLAG = "--always-approve"

# Attached as `--single=<prompt>` rather than passed as its own argument:
# grok's parser reads a bare argument beginning with `-` as a flag, so a
# prompt like "--fix the parser" fails the run before the model sees it.
PROMPT_FLAG = "--single"

# Takes the schema document inline, as its own argument. The --single= gluing
# above is not needed here: the value starts with "{", which no argument
# parser reads as a flag.
SCHEMA_FLAG = "--json-schema"

# Grok's own parse of a schema-constrained reply, alongside the same object as
# a JSON string in `text`. camelCase, like the rest of this envelope.
STRUCTURED_FIELD = "structuredOutput"


def _pick_grok_object(candidates: list[dict]) -> dict:
    """Prefer a Grok envelope over surrounding noise, strongest marker first.

    Success uses camelCase sessionId/text; failure uses type=error. Only
    those two identify an envelope on their own. A bare `text` is a weak
    marker — a trailing tip or warning line carries one too — so it decides
    nothing until the stream turns out to hold no real envelope at all.
    """
    for obj in reversed(candidates):
        if "sessionId" in obj or obj.get("type") == "error":
            return obj
    for obj in reversed(candidates):
        if "text" in obj:
            return obj
    return candidates[-1]


def parse_grok_stdout(stdout: str) -> dict:
    """Return the Grok result object from `--output-format json` stdout.

    The documented format is one object. A text prefix still happens, so
    a load of the whole buffer then fails even though a valid envelope
    is sitting in the stream.
    """
    stripped = stdout.strip()
    if not stripped:
        raise GrokTurnError("grok output was not JSON\nstdout: ")
    candidates = json_objects(stripped)
    if not candidates:
        raise GrokTurnError(
            f"grok output was not JSON\nstdout: {stdout_for_error(stdout)}"
        )
    return _pick_grok_object(candidates)


def _reported_error(payload: dict) -> str:
    """The one wording for a Grok-reported failure, whatever the exit code."""
    return f"grok reported an error: {payload.get('message')}"


def _failed_turn_message(returncode: int, stdout: str, stderr: str) -> str:
    """Prefer Grok's JSON error object; fall back to the exit and stderr tail.

    Failure emits `{"type":"error","message":...}` on stdout and a non-zero
    exit. The message is the part a caller can act on.
    """
    try:
        payload = parse_grok_stdout(stdout)
    except GrokTurnError:
        payload = {}
    if payload.get("type") == "error":
        return _reported_error(payload)
    return f"grok exited {returncode}\nstderr: {stderr[-2000:]}"


class GrokBackend(AgentBackend):
    """Backend for the Grok CLI (`grok`)."""

    name = "grok"

    def run_turn(
        self,
        prompt: str,
        session_id: str | None,
        cwd: Path,
        timeout: int = DEFAULT_TURN_TIMEOUT,
        schema: OutputSchema | None = None,
    ) -> TurnResult:
        # --session-id names a *new* session only and errors if that id
        # already exists. Resume is --resume.
        args = ["grok", "--output-format", "json", ALWAYS_APPROVE_FLAG]
        if self.model is not None:
            args += ["--model", self.model]
        if self.reasoning_effort is not None:
            args += ["--reasoning-effort", self.reasoning_effort]
        if schema is not None:
            args += [SCHEMA_FLAG, schema.text]
        if session_id:
            args += ["--resume", session_id]
        args.append(f"{PROMPT_FLAG}={prompt}")

        proc = run_cli_turn(
            self.name,
            args,
            prompt=prompt,
            session_id=session_id,
            cwd=cwd,
            timeout=timeout,
        )
        if proc.returncode != 0:
            raise GrokTurnError(
                _failed_turn_message(proc.returncode, proc.stdout, proc.stderr)
            )
        payload = parse_grok_stdout(proc.stdout)

        if payload.get("type") == "error":
            raise GrokTurnError(_reported_error(payload))
        reported_session = payload.get("sessionId")
        # No session id means the turn cannot be resumed. Returning None here
        # would be written over the id already on file, so the next turn would
        # start a new conversation instead of continuing this agent's.
        if not isinstance(reported_session, str) or not reported_session:
            raise GrokTurnError(
                f"grok did not report a sessionId\n"
                f"stdout: {stdout_for_error(proc.stdout)}"
            )
        reply = reply_text(payload, "text")
        result = TurnResult(
            session_id=reported_session,
            reply=reply,
            raw=proc.stdout,
            structured=structured_reply(schema, reply, payload.get(STRUCTURED_FIELD)),
        )
        log.debug(
            "grok turn: parsed session=%s reply_chars=%d",
            result.session_id,
            len(result.reply),
        )
        return result
