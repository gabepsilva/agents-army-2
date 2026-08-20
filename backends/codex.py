"""OpenAI Codex CLI backend implementation."""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path

from backends.base import AgentBackend, TurnError, TurnResult, describe_command

log = logging.getLogger(__name__)


class CodexTurnError(TurnError):
    """Raised when the Codex CLI returns something unusable."""


class CodexBackend(AgentBackend):
    """Backend for OpenAI's Codex CLI (`codex`)."""

    name = "codex"

    def run_turn(
        self,
        prompt: str,
        session_id: str | None,
        cwd: Path,
        timeout: int = 1800,
    ) -> TurnResult:
        args = ["codex", "exec"]
        if session_id:
            args += ["resume", session_id]
        args += [prompt, "--json", "--skip-git-repo-check"]

        log.debug(
            "codex turn: cwd=%s resume=%s prompt_chars=%d timeout=%ds",
            cwd,
            bool(session_id),
            len(prompt),
            timeout,
        )
        log.debug("codex turn: invoking %s", describe_command(args, prompt))
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
            "codex turn: exited %d after %.1fs with %d chars of stdout",
            proc.returncode,
            time.monotonic() - started,
            len(proc.stdout),
        )
        if proc.returncode != 0:
            raise CodexTurnError(
                f"codex exited {proc.returncode}\nstderr: {proc.stderr[-2000:]}"
            )
        return self._parse(proc.stdout, proc.stderr)

    def _parse(self, stdout: str, stderr: str) -> TurnResult:
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
        )
