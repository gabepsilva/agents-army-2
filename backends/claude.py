"""Claude Code CLI backend implementation."""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path

from backends.base import AgentBackend, TurnResult, describe_command

log = logging.getLogger(__name__)


class ClaudeTurnError(RuntimeError):
    """Raised when the Claude CLI returns something unusable."""


# Print mode denies tools (gh, Bash, WebFetch) unless a permission mode
# is set. bypassPermissions is the non-interactive opt-in; without it
# Claude still exits 0 and the result JSON carries reason=sdk_opt_in_required.
PERMISSION_MODE = "bypassPermissions"


def _stdout_for_error(stdout: str) -> str:
    """Keep both ends of a long dump: the parse error is at char 0, the
    result envelope is usually at the tail."""
    if len(stdout) <= 2000:
        return stdout
    return f"{stdout[:400]}\n…\n{stdout[-1600:]}"


def _json_objects(text: str) -> list[dict]:
    """Scan `text` for every top-level JSON object, in order of appearance."""
    decoder = json.JSONDecoder()
    found: list[dict] = []
    idx = 0
    while idx < len(text):
        start = text.find("{", idx)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            idx = start + 1
            continue
        found.append(obj)
        idx = end
    return found


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
    candidates = _json_objects(stripped)
    if not candidates:
        raise ClaudeTurnError(
            f"claude output was not JSON\nstdout: {_stdout_for_error(stdout)}"
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
        timeout: int = 1800,
    ) -> TurnResult:
        args = [
            "claude",
            "--print",
            "--output-format",
            "json",
            "--permission-mode",
            PERMISSION_MODE,
        ]
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
        result = TurnResult(
            session_id=payload.get("session_id"),
            reply=payload.get("result", ""),
            raw=proc.stdout,
        )
        log.debug(
            "claude turn: parsed session=%s reply_chars=%d",
            result.session_id,
            len(result.reply),
        )
        return result
