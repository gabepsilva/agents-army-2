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

# A turn runs unattended, so codex must neither stop to ask nor sandbox what
# it runs. `--yolo` is its alias for --dangerously-bypass-approvals-and-sandbox,
# the counterpart to the claude backend's bypassPermissions. Without it a turn
# cannot commit in a linked worktree - the git directory it must write sits
# outside the sandboxed workspace - nor reach the network to push.
YOLO_FLAG = "--yolo"

# `codex exec resume <id> <prompt>` and `codex exec fork <id> <prompt>` take
# the same slot and the same trailing flags, so a forked turn is the ordinary
# resume argv with this token in place of the other.
RESUME_COMMAND = "resume"
FORK_COMMAND = "fork"

# The events a failed turn arrives as, on stdout. `error` carries the message
# directly; `turn.failed` nests it under `error`.
ERROR_EVENT = "error"
FAILED_TURN_EVENT = "turn.failed"


def _stream_value(value: object) -> str:
    """Keep a streamed value readable on one stderr line."""
    if isinstance(value, str):
        return value.replace("\r", "\\r").replace("\n", "\\n")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _format_error_event(event: dict) -> str:
    """Format a Codex parser failure without rendering usage metadata."""
    error = event.get("error")
    message = error.get("message") if isinstance(error, dict) else event.get("message")
    if not isinstance(message, str) or not message:
        message = "Codex reported an error"
    return f"Error: {_stream_value(message)}"


def _mcp_name(item: dict) -> str:
    """Return the readable server/tool name used by an MCP item."""
    name = item.get("tool")
    if not isinstance(name, str):
        name = item.get("name")
    if not isinstance(name, str):
        name = "unknown"
    server = item.get("server")
    if isinstance(server, str) and server:
        return f"{server}/{name}"
    return name


def _item_output(item: dict) -> object:
    """Find the result field used by a completed Codex item."""
    for key in ("aggregated_output", "output", "result", "error"):
        if key in item:
            return item[key]
    return None


def _format_item(event_type: object, item: dict) -> str | None:
    """Format one Codex item event."""
    item_type = item.get("type")
    if item_type == "reasoning":
        return "Thinking..."
    if item_type == "agent_message":
        text = item.get("text")
        return f"Assistant: {_stream_value(text)}" if isinstance(text, str) else None
    if item_type not in {"command_execution", "mcp_tool_call"}:
        return None

    is_result = event_type == "item.completed" or (
        event_type == "item.updated"
        and (
            item.get("status") == "completed"
            or any(key in item for key in ("aggregated_output", "output", "result"))
        )
    )
    if is_result:
        output = _item_output(item)
        label = "MCP result" if item_type == "mcp_tool_call" else "Tool result"
        return f"{label}: {_stream_value(output)}" if output is not None else None

    if item_type == "mcp_tool_call":
        return (
            f"MCP call: {_mcp_name(item)} "
            f"{_stream_value(item.get('arguments', item.get('input', {})))}"
        )
    return f"Tool call: command {_stream_value(item.get('command', ''))}"


def format_event(event: dict) -> str | None:
    """Render one Codex JSONL event for the live stderr display."""
    event_type = event.get("type")
    if event_type in {ERROR_EVENT, FAILED_TURN_EVENT}:
        return _format_error_event(event)
    item = event.get("item")
    if not isinstance(item, dict):
        if event_type == "mcp_tool_call":
            item = event
        else:
            return None
    return _format_item(event_type, item)


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
    supports_fork = True
    # A2 verdict (2026-08-27, codex-cli 0.149.0): PASS. `codex resume
    # <id>` resumes the previous interactive session without forking it.
    supports_chat = True

    def chat_argv(self, session_id: str, cwd: Path) -> list[str]:
        """Resume the stored Codex session in its interactive terminal UI."""
        log.debug("codex chat: cwd=%s session=%s", cwd, session_id)
        return ["codex", "resume", session_id]

    def run_turn(  # noqa: PLR0913 - flat backend turn arguments are the public seam
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
        args = ["codex", "exec", YOLO_FLAG]
        if self.model is not None:
            args += ["--model", self.model]
        if self.reasoning_effort is not None:
            args += ["--config", f'model_reasoning_effort="{self.reasoning_effort}"']
        if session_id:
            args += [FORK_COMMAND if resume_as_fork else RESUME_COMMAND, session_id]
        args += [prompt, "--json", "--skip-git-repo-check"]
        if schema is not None:
            args += [SCHEMA_FLAG, str(schema.path)]

        proc = run_cli_turn(
            self.name,
            args,
            prompt=prompt,
            session_id=session_id,
            cwd=cwd,
            timeout=timeout,
            stream=stream,
            format_event=format_event if stream else None,
        )
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
