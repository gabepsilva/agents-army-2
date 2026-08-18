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
        args = ["claude", "--print", "--output-format", "json"]
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
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise ClaudeTurnError(
                f"claude output was not JSON\nstdout: {proc.stdout[-2000:]}"
            ) from exc

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
