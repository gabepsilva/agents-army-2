"""Tests for the agent metadata `orchestrator list agents` now prints:
`created_at`/`last_turn_at`/`turns` persistence, the `registry:` header, and
the `busy` probe. See PR #99 (issue #97)."""

from __future__ import annotations

import fcntl
import json
import re
from pathlib import Path
from typing import TextIO

import pytest

import orchestrator
from backends.base import (
    DEFAULT_TURN_TIMEOUT,
    AgentBackend,
    OutputSchema,
    TurnResult,
    structured_reply,
)
from backends.registry import register_backend
from orchestrator import Orchestrator
from orchestrator.schema import load_schema


class EchoBackend(AgentBackend):
    """A backend that answers without a CLI, for tests about everything else."""

    @property
    def name(self) -> str:
        return "echo"

    def run_turn(
        self,
        prompt: str,
        session_id: str | None,
        cwd: Path,
        timeout: int = DEFAULT_TURN_TIMEOUT,
        schema: OutputSchema | None = None,
    ) -> TurnResult:
        return TurnResult(session_id="echo-sid", reply=f"echo:{prompt}", raw="")


@pytest.fixture(autouse=True)
def register_echo_backend() -> None:
    register_backend("echo", EchoBackend)


# A stage/verdict schema a scripted backend can violate once and then satisfy,
# matching the shape tests/test_schema.py already uses for the same purpose.
_STRICT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["stage", "verdict"],
    "properties": {
        "stage": {"type": "string"},
        "verdict": {"type": "string", "enum": ["pass", "fail"]},
    },
}
_CONFORMING = {"stage": "build", "verdict": "pass"}


def _strict_schema(tmp_path: Path) -> OutputSchema:
    path = tmp_path / "schema.json"
    path.write_text(json.dumps(_STRICT_SCHEMA), encoding="utf-8")
    return load_schema(path)


def _scripted(replies: list[str], name: str = "scripted") -> list[dict]:
    """Register a backend answering `replies` in order; return its call log."""
    calls: list[dict] = []
    queued = list(replies)

    class ScriptedBackend(AgentBackend):
        @property
        def name(self) -> str:
            return name

        def run_turn(
            self,
            prompt: str,
            session_id: str | None,
            cwd: Path,
            timeout: int = DEFAULT_TURN_TIMEOUT,
            schema: OutputSchema | None = None,
        ) -> TurnResult:
            calls.append({"prompt": prompt, "session_id": session_id})
            reply = queued.pop(0)
            return TurnResult(
                session_id="sid-1",
                reply=reply,
                raw=reply,
                structured=structured_reply(schema, reply),
            )

    register_backend(name, ScriptedBackend)
    return calls


def _flock_probe(path: Path, mode: int) -> TextIO:
    """Take and hold `mode` on `path` outside the orchestrator's own locking,
    standing in for a concurrent process."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    fcntl.flock(handle.fileno(), mode)
    return handle


class TestUtcNow:
    def test_matches_iso8601_utc_seconds_with_a_z_suffix(self) -> None:
        assert re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", orchestrator._utcnow()
        )

    def test_reads_utc_not_local_time(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Distinct from the format test: a machine whose local clock happens
        to be UTC would pass that one even reading local time by mistake."""
        real_datetime = orchestrator.datetime
        seen_tz = []

        class FrozenDatetime(real_datetime):
            @classmethod
            def now(cls, tz=None):
                seen_tz.append(tz)
                return real_datetime(2026, 8, 25, 23, 27, 32, tzinfo=tz)

        monkeypatch.setattr(orchestrator, "datetime", FrozenDatetime)
        assert orchestrator._utcnow() == "2026-08-25T23:27:32Z"
        assert seen_tz == [orchestrator.UTC]


class TestAgentDefaults:
    def test_a_freshly_constructed_agent_has_no_metadata_yet(self) -> None:
        agent = orchestrator.Agent("a", EchoBackend())
        assert agent.created_at is None
        assert agent.last_turn_at is None
        assert agent.turns is None


class TestNewFieldsRoundTrip:
    def test_round_trip_through_a_fresh_orchestrator(self, tmp_path: Path) -> None:
        """Created, talk, then re-read from a *new* Orchestrator instance —
        never the in-memory Agent already held, which is exactly what would
        still pass under the `agent.turns += 1` bug the spec calls out."""
        state_file = tmp_path / "state.json"
        orch = Orchestrator(state_file=state_file)
        orch.spawn("a", "echo")
        orch.talk("a", "hi")

        reloaded = Orchestrator(state_file=state_file)
        agent = reloaded.agents["a"]
        assert agent.created_at is not None
        assert agent.last_turn_at is not None
        assert agent.turns == 1

    def test_a_pre_change_registry_loads_and_prints_dashes(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        state_file = tmp_path / "state.json"
        state_file.write_text(
            json.dumps({"a": {"backend": "echo", "session_id": None}}),
            encoding="utf-8",
        )
        orch = Orchestrator(state_file=state_file)
        agent = orch.agents["a"]
        assert agent.created_at is None
        assert agent.last_turn_at is None
        assert agent.turns is None

        orchestrator._print_agents(orch)
        line = capsys.readouterr().out.splitlines()[1]
        assert "turns=-" in line
        assert "created=-" in line
        assert "last=-" in line
        assert "session=-" in line

    def test_turns_counts_every_validation_attempt_not_every_call(
        self, tmp_path: Path
    ) -> None:
        state_file = tmp_path / "state.json"
        bad = '{"stage":"build","verdict":"banana"}'
        _scripted([bad, json.dumps(_CONFORMING)])
        orch = Orchestrator(state_file=state_file)
        orch.spawn("a", "scripted")
        result = orch.talk("a", "go", schema=_strict_schema(tmp_path), retries=1)
        assert result.structured == _CONFORMING

        reloaded = Orchestrator(state_file=state_file)
        assert reloaded.agents["a"].turns == 2

    def test_a_never_talked_agent_shows_zero_turns_and_a_real_created_stamp(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        state_file = tmp_path / "state.json"
        orch = Orchestrator(state_file=state_file)
        orch.spawn("a", "echo")

        orchestrator._print_agents(orch)
        line = capsys.readouterr().out.splitlines()[1]
        assert "turns=0" in line
        assert "last=-" in line
        assert "session=-" in line
        assert orch.agents["a"].created_at is not None
        assert f"created={orch.agents['a'].created_at}" in line


class TestBusyProbe:
    def test_an_exclusively_locked_agent_is_reported_busy(self, tmp_path: Path) -> None:
        state_file = tmp_path / "state.json"
        orch = Orchestrator(state_file=state_file)
        orch.spawn("a", "echo")
        path = orch._agent_lock_path("a")

        holder = _flock_probe(path, fcntl.LOCK_EX)
        try:
            assert orch._agent_is_busy("a") is True
        finally:
            fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
            holder.close()

    def test_a_shared_locked_agent_is_reported_idle(self, tmp_path: Path) -> None:
        """A second `list agents` probing the same agent concurrently must not
        make each other report `busy` — only a real turn's exclusive hold
        does."""
        state_file = tmp_path / "state.json"
        orch = Orchestrator(state_file=state_file)
        orch.spawn("a", "echo")
        path = orch._agent_lock_path("a")

        holder = _flock_probe(path, fcntl.LOCK_SH)
        try:
            assert orch._agent_is_busy("a") is False
        finally:
            fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
            holder.close()

    def test_an_agent_that_never_talked_is_idle_with_no_lock_file(
        self, tmp_path: Path
    ) -> None:
        state_file = tmp_path / "state.json"
        orch = Orchestrator(state_file=state_file)
        orch.spawn("a", "echo")
        assert orch._agent_is_busy("a") is False

    def test_listing_creates_no_lock_files(self, tmp_path: Path) -> None:
        """Regression test for probing without first checking `path.exists()`:
        `_flock` creates the file it opens, and an unguarded probe would
        scatter lock files as a side effect of listing."""
        state_file = tmp_path / "state.json"
        orch = Orchestrator(state_file=state_file)
        orch.spawn("a", "echo")

        orchestrator._print_agents(orch)

        locks_dir = orch._locks_dir()
        assert not locks_dir.exists() or list(locks_dir.iterdir()) == []


class TestListingAlignment:
    def test_session_column_starts_at_the_same_offset(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        state_file = tmp_path / "state.json"
        orch = Orchestrator(state_file=state_file)
        long_name = "a-name-well-past-twenty-characters"
        orch.spawn(long_name, "echo")
        orch.spawn("short", "echo")

        path = orch._agent_lock_path(long_name)
        holder = _flock_probe(path, fcntl.LOCK_EX)
        try:
            orchestrator._print_agents(orch)
        finally:
            fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
            holder.close()

        header, *lines = capsys.readouterr().out.splitlines()
        assert header == f"registry: {orch.state_file}"
        assert len(lines) == 2
        offsets = {line.index("session=") for line in lines}
        assert len(offsets) == 1
        # Exactly one line: the marker must read the literal word "busy",
        # not merely contain it, and a wrong-width idle marker would have
        # already failed the offsets assertion above.
        assert sum(" busy  session=" in line for line in lines) == 1
