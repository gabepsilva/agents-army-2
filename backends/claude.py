"""Claude Code CLI backend implementation."""

from __future__ import annotations

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
    json_objects,
    reply_text,
    stdout_for_error,
    structured_reply,
)

log = logging.getLogger(__name__)


class ClaudeTurnError(TurnError):
    """Raised when the Claude CLI returns something unusable."""


# Print mode denies tools (gh, Bash, WebFetch) unless a permission mode
# is set. bypassPermissions is the non-interactive opt-in; without it
# Claude still exits 0 and the result JSON carries reason=sdk_opt_in_required.
PERMISSION_MODE = "bypassPermissions"

# The marker that opt-in carries when it did not happen. The CLI keeps exit 0
# and is_error false, so a turn that ran with no tools at all is otherwise
# indistinguishable from one that worked: if PERMISSION_MODE ever stops being
# honoured, only this check turns the degraded reply into a failure.
OPT_IN_REQUIRED_REASON = "sdk_opt_in_required"

# Takes the schema document inline, as one argument. It is passed on its own
# rather than glued to the flag (grok's --single= lesson does not apply): the
# value starts with "{", which no argument parser reads as a flag.
SCHEMA_FLAG = "--json-schema"

# Claude's own parse of a schema-constrained reply. `result` carries the same
# object as a JSON *string*, so this field only saves a parse — it is not a
# second source of truth.
STRUCTURED_FIELD = "structured_output"


def _pick_result_object(candidates: list[dict]) -> dict:
    for obj in reversed(candidates):
        if obj.get("type") == "result":
            return obj
    return candidates[-1]


def parse_claude_stdout(stdout: str) -> dict:
    """Return the Claude result object from `--output-format json` stdout.

    Print mode sometimes writes a text reply (or a stream of objects) before
    the envelope. json.loads of the whole buffer then fails at column 1 even
    though a valid result object is sitting at the end.
    """
    stripped = stdout.strip()
    if not stripped:
        raise ClaudeTurnError("claude output was not JSON\nstdout: ")
    candidates = json_objects(stripped)
    if not candidates:
        raise ClaudeTurnError(
            f"claude output was not JSON\nstdout: {stdout_for_error(stdout)}"
        )
    return _pick_result_object(candidates)


class ClaudeBackend(AgentBackend):
    """Backend for Anthropic's Claude Code CLI (`claude`)."""

    name = "claude"

    def run_turn(
        self,
        prompt: str,
        session_id: str | None,
        cwd: Path,
        timeout: int = DEFAULT_TURN_TIMEOUT,
        schema: OutputSchema | None = None,
    ) -> TurnResult:
        args = [
            "claude",
            "--print",
            "--output-format",
            "json",
            "--permission-mode",
            PERMISSION_MODE,
        ]
        if self.model is not None:
            args += ["--model", self.model]
        if self.reasoning_effort is not None:
            args += ["--effort", self.reasoning_effort]
        if schema is not None:
            args += [SCHEMA_FLAG, schema.text]
        if session_id:
            args += ["--resume", session_id]
        args += ["-p", prompt]

        log.debug(
            "claude turn: cwd=%s resume=%s prompt_chars=%d timeout=%ds",
            cwd,
            bool(session_id),
            len(prompt),
            timeout,
        )
        log.debug("claude turn: invoking %s", describe_command(args, prompt))
        started = time.monotonic()
        proc = subprocess.run(
            args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            # A CLI reading a non-tty stdin blocks until it is killed, so a run
            # from cron, CI or any host script that is not a terminal would
            # spend the whole timeout and return nothing.
            stdin=subprocess.DEVNULL,
        )
        log.debug(
            "claude turn: exited %d after %.1fs with %d chars of stdout",
            proc.returncode,
            time.monotonic() - started,
            len(proc.stdout),
        )
        if proc.returncode != 0:
            raise ClaudeTurnError(
                f"claude exited {proc.returncode}\nstderr: {proc.stderr[-2000:]}"
            )
        payload = parse_claude_stdout(proc.stdout)

        if payload.get("is_error"):
            raise ClaudeTurnError(f"claude reported an error: {payload.get('result')}")
        if payload.get("reason") == OPT_IN_REQUIRED_REASON:
            raise ClaudeTurnError(
                f"claude ran without tools: reason={OPT_IN_REQUIRED_REASON}. "
                f"--permission-mode {PERMISSION_MODE} did not take effect."
            )
        session_id = payload.get("session_id")
        # No session id means the turn cannot be resumed. Returning None here
        # would be written over the id already on file, so the next turn would
        # start a new conversation instead of continuing this agent's.
        if not isinstance(session_id, str) or not session_id:
            raise ClaudeTurnError(
                f"claude did not report a session_id\n"
                f"stdout: {stdout_for_error(proc.stdout)}"
            )
        reply = reply_text(payload, "result")
        result = TurnResult(
            session_id=session_id,
            reply=reply,
            raw=proc.stdout,
            structured=structured_reply(schema, reply, payload.get(STRUCTURED_FIELD)),
        )
        log.debug(
            "claude turn: parsed session=%s reply_chars=%d",
            result.session_id,
            len(result.reply),
        )
        return result
