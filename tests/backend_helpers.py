"""Shared helpers for backend and orchestrator tests."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from backends.base import (
    DEFAULT_TURN_TIMEOUT,
    AgentBackend,
    OutputSchema,
    TurnResult,
)


def _messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Log lines are asserted verbatim: a wrong one is a wrong diagnostic."""
    return [record.getMessage() for record in caplog.records]


def _reported_seconds(message: str, pattern: str) -> float:
    """Return the duration a log line claims, so an implausible one fails.

    Matching the shape of the number is not enough: computing the elapsed time
    with the wrong sign still prints a well-formed float. Only its magnitude —
    process uptime rather than a turn duration — distinguishes the two.
    """
    match = re.fullmatch(pattern, message)
    assert match is not None, f"unexpected log line: {message!r}"
    return float(match.group(1))


class EchoBackend(AgentBackend):
    """A backend that answers without a CLI, for tests about everything else."""

    @property
    def name(self) -> str:
        return "echo"

    def run_turn(  # noqa: PLR0913 - test doubles mirror AgentBackend.run_turn public seam
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
        return TurnResult(session_id="echo-sid", reply=f"echo:{prompt}", raw="")


def _assert_subprocess_kwargs(
    kwargs: dict,
    cwd: Path,
    expected_stdin: object = subprocess.DEVNULL,
    expected_input: str | None = None,
    *,
    expected_timeout: int = DEFAULT_TURN_TIMEOUT,
) -> None:
    """Every backend must run its subprocess the same disciplined way."""
    assert kwargs["cwd"] == str(cwd)
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert kwargs["check"] is False
    assert kwargs["timeout"] == expected_timeout
    # Not a detail: a CLI whose stdin is an inherited pipe rather than a tty
    # blocks until it is killed. `codex exec "reply ok" --json` under a pipe
    # returns nothing after 25s and exits 124, and claude and grok are given
    # no chance to do the same. Asserted for every backend, in the one helper
    # every backend test already calls, so a new backend cannot skip it.
    if expected_input is None:
        assert kwargs["stdin"] == expected_stdin
    else:
        assert kwargs["input"] == expected_input


def _completed(returncode: int, stdout: str, stderr: str = "") -> Callable:
    """A `subprocess.run` stand-in for a test that only cares what came back."""

    def run(args, **kwargs):
        return subprocess.CompletedProcess(
            args, returncode, stdout=stdout, stderr=stderr
        )

    return run


def _subprocess_recorder(
    result: Callable,
) -> tuple[Callable, list[tuple[list[str], dict]]]:
    """Record subprocess calls while returning a supplied canned result."""
    calls: list[tuple[list[str], dict]] = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return result(args, **kwargs)

    return fake_run, calls


# A schema as the adapters receive it: already loaded, in both spellings. The
# text is what claude and grok take inline; the path is what codex is handed.
SCHEMA = OutputSchema(
    text='{"type":"object","additionalProperties":false,"properties":{}}',
    path=Path("/schemas/reply.json"),
)
