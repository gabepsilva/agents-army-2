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

# Forks whatever session the same command line names, so it is only ever
# emitted next to the `--session` a resume already carries; opencode itself
# rejects it with neither `--session` nor `--continue`.
FORK_FLAG = "--fork"


class OpenCodeTurnError(TurnError):
    """Raised when the OpenCode CLI returns something unusable."""


def _stream_value(value: object) -> str:
    """Keep a streamed value readable on one stderr line."""
    if isinstance(value, str):
        return value.replace("\r", "\\r").replace("\n", "\\n")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _format_error_event(event: dict) -> str:
    """Format an OpenCode parser failure without rendering metadata."""
    detail = _error_detail(event)
    if detail is None:
        message = event.get("message")
        if isinstance(message, str) and message:
            detail = message
    if detail is None:
        error = event.get("error")
        if isinstance(error, str) and error:
            detail = error
    if detail is None:
        detail = "OpenCode reported an error"
    return f"Error: {_stream_value(detail)}"


def _tool_name(part: dict) -> str:
    """Return the provider/tool name from an OpenCode tool part."""
    for key in ("tool", "name"):
        name = part.get(key)
        if isinstance(name, str) and name:
            return name
    return "unknown"


def _is_mcp_tool(part: dict, name: str) -> bool:
    """Recognize OpenCode's MCP tool names and explicit server metadata."""
    return bool(
        any(
            isinstance(part.get(key), str) and part[key]
            for key in ("server", "mcp_server")
        )
        or name.startswith(("mcp__", "mcp_"))
        or part.get("type") == "mcp"
    )


def _tool_result(state: dict, mcp: bool) -> str | None:
    """Format a tool result when the part carries one."""
    if "error" in state:
        label = "MCP result (error)" if mcp else "Tool result (error)"
        return f"{label}: {_stream_value(state['error'])}"
    for key in ("output", "result"):
        if key in state:
            label = "MCP result" if mcp else "Tool result"
            return f"{label}: {_stream_value(state[key])}"
    return None


def _format_tool_event(part: dict) -> str | None:
    """Format an OpenCode tool call, result, or both in one line."""
    state = part.get("state")
    if not isinstance(state, dict):
        return None
    name = _tool_name(part)
    mcp = _is_mcp_tool(part, name)
    label = "MCP call" if mcp else "Tool call"
    lines: list[str] = []
    if "input" in state:
        lines.append(f"{label}: {name} {_stream_value(state['input'])}")
    result = _tool_result(state, mcp)
    if result is not None:
        lines.append(result)
    return " | ".join(lines) or None


def format_event(event: dict) -> str | None:
    """Render one OpenCode JSON event for the live stderr display."""
    event_type = event.get("type")
    if event_type == "error":
        return _format_error_event(event)

    part = event.get("part")
    if not isinstance(part, dict):
        return None
    part_type = part.get("type")
    if event_type == "reasoning" or part_type == "reasoning":
        return "Thinking..."
    if event_type == "text" and part_type in {None, "text"}:
        text = part.get("text")
        return f"Assistant: {_stream_value(text)}" if isinstance(text, str) else None
    if event_type == "tool_use" and part_type in {None, "tool", "mcp"}:
        return _format_tool_event(part)
    return None


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
    supports_fork = True
    # A2 verdict (2026-08-27, OpenCode 1.18.21): PASS. `--session <id>` is
    # the interactive continuation spelling; `--fork` is the separate fork.
    supports_chat = True

    def chat_argv(self, session_id: str, cwd: Path) -> list[str]:
        """Resume the stored OpenCode session in its interactive terminal UI."""
        log.debug("opencode chat: cwd=%s session=%s", cwd, session_id)
        return ["opencode", "--session", session_id]

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
            if resume_as_fork:
                args.append(FORK_FLAG)

        proc = run_cli_turn(
            self.name,
            args,
            prompt=prompt,
            session_id=session_id,
            cwd=cwd,
            timeout=timeout,
            prompt_on_stdin=True,
            stream=stream,
            format_event=format_event if stream else None,
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
