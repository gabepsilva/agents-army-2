"""Claude Code CLI backend implementation."""

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
    reply_text,
    run_cli_turn,
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

# Resumes into a copy of the named session instead of continuing it. Only
# meaningful alongside --resume, which is the only way this adapter emits it.
FORK_FLAG = "--fork-session"

# The dialect declaration, which the CLI cannot resolve. `--json-schema` is
# checked by an ajv instance that only knows draft-07, so a document declaring
# a newer dialect is refused before the turn starts:
#   Error: --json-schema is not a valid JSON Schema: no schema with key or ref
#   "https://json-schema.org/draft/2020-12/schema"
# The keyword names the dialect and constrains nothing, and the orchestrator
# has already checked the document against the dialect the file declared, so
# dropping it for this one argument costs no validation. Every other keyword
# is passed through exactly as loaded.
DIALECT_KEYWORD = "$schema"

# Claude's own parse of a schema-constrained reply. `result` carries the same
# object as a JSON *string*, so this field only saves a parse — it is not a
# second source of truth.
STRUCTURED_FIELD = "structured_output"


def schema_argument(schema: OutputSchema) -> str:
    """The schema document as `--json-schema` accepts it: no dialect keyword.

    A document that never declared one is passed through as loaded, so the
    rewrite below is reached only by the schemas the CLI would otherwise
    reject.
    """
    document = json.loads(schema.text)
    if DIALECT_KEYWORD not in document:
        return schema.text
    del document[DIALECT_KEYWORD]
    return json.dumps(document, separators=(",", ":"), sort_keys=True)


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


def _failed_turn_message(returncode: int, stdout: str, stderr: str) -> str:
    """Prefer claude's own error envelope; fall back to the exit and both pipes.

    A print-mode failure does not always write to stderr. One observed run
    (2026-08-22, a griller turn in the development workflow) exited 1 with
    stderr empty, so the only wording the orchestrator could log was
    "claude exited 1\nstderr: " -- nothing to act on, and nothing to tell a
    transient API failure from a rejected argument. Whatever the CLI did say
    went to stdout and was dropped on the floor.
    """
    try:
        payload = parse_claude_stdout(stdout)
    except ClaudeTurnError:
        payload = {}
    if payload.get("is_error"):
        return f"claude reported an error: {payload.get('result')}"
    return (
        f"claude exited {returncode}\n"
        f"stderr: {stderr[-2000:]}\n"
        f"stdout: {stdout_for_error(stdout)}"
    )


class ClaudeBackend(AgentBackend):
    """Backend for Anthropic's Claude Code CLI (`claude`)."""

    name = "claude"
    supports_fork = True
    # A2 verdict (2026-08-27, Claude Code 2.1.240): PASS. `claude --resume
    # <id>` is the interactive resume spelling and does not request a fork.
    supports_chat = True

    def chat_argv(self, session_id: str, cwd: Path) -> list[str]:
        """Resume the stored Claude session in its interactive terminal UI."""
        log.debug("claude chat: cwd=%s session=%s", cwd, session_id)
        return ["claude", "--resume", session_id]

    def run_turn(  # noqa: PLR0913 - mirrors AgentBackend's flat interface
        self,
        prompt: str,
        session_id: str | None,
        cwd: Path,
        timeout: int = DEFAULT_TURN_TIMEOUT,
        schema: OutputSchema | None = None,
        *,
        resume_as_fork: bool = False,
        stream: bool = False,
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
            args += [SCHEMA_FLAG, schema_argument(schema)]
        if session_id:
            args += ["--resume", session_id]
            if resume_as_fork:
                args.append(FORK_FLAG)
        args += ["-p", prompt]

        proc = run_cli_turn(
            self.name,
            args,
            prompt=prompt,
            session_id=session_id,
            cwd=cwd,
            timeout=timeout,
            stream=stream,
        )
        if proc.returncode != 0:
            raise ClaudeTurnError(
                _failed_turn_message(proc.returncode, proc.stdout, proc.stderr)
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
