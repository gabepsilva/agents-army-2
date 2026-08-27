"""Unit tests for agent backend interface, implementations, and orchestrator."""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

import orchestrator.core as core
from backends.base import (
    DEFAULT_TURN_TIMEOUT,
    AgentBackend,
    OutputSchema,
    TurnResult,
)
from backends.registry import (
    UnknownBackendError,
    register_backend,
)
from orchestrator.core import (
    Agent,
    AgentBusyError,
    Orchestrator,
)
from orchestrator.schema import compose_schema_prompt
from tests.backend_helpers import (
    SCHEMA,
    _assert_subprocess_kwargs,
    _messages,
)
from tests.path_helpers import runtime_paths


def _flock_is_held(path: Path) -> bool:
    """Is someone holding this lock file?

    flock is owned by the open file description, not the process, so a second
    handle here contends with the orchestrator's exactly as another process
    would — the lock this asks about is the real one, not a stand-in.
    """
    with path.open("a+", encoding="utf-8") as probe:
        try:
            fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(probe.fileno(), fcntl.LOCK_UN)
        return False


# What each tool prints for `--version` on a machine that has all of them.
# The spellings are the real ones: uv leads with its own name, jq glues its
# name on with a hyphen, claude prints a bare number, and codex names the
# package rather than the command.


def _gate_backend(
    name: str, entered: threading.Event, release: threading.Event
) -> type[AgentBackend]:
    """A backend class whose turn signals `entered`, then blocks on `release`.

    A factory rather than one shared class: each caller closes over its own
    pair of events, and `.name` must equal the string it gets registered
    under — `_persist`/`_reload` round-trip a spawned agent's backend through
    `agent.backend.name`, not through the registry key `spawn` was called
    with, so the two have to match.
    """

    class Gate(AgentBackend):
        @property
        def name(self) -> str:
            return name

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
            entered.set()
            release.wait(timeout=5)
            return TurnResult(session_id="s1", reply="ok", raw="")

    return Gate


def _pause_first_flock(
    monkeypatch: pytest.MonkeyPatch, thread_name: str
) -> tuple[threading.Event, threading.Event]:
    """Pause `thread_name`'s first `fcntl.flock()` call just before it runs.

    Drives a real `open()`-then-`flock()` TOCTOU window on demand: the
    caller can act between a thread's `_flock` opening a path and that same
    call's `flock()` actually being granted. Returns `(opened, proceed)` —
    `opened` fires once the call is paused there, and the caller sets
    `proceed` to let it continue. Only the first call is paused, so the
    same thread's later `flock()` calls (e.g. the matching unlock) run
    unpatched.
    """
    opened = threading.Event()
    proceed = threading.Event()
    real_flock = fcntl.flock
    paused = False

    def patched(fd: int, op: int) -> None:
        nonlocal paused
        if threading.current_thread().name == thread_name and not paused:
            paused = True
            opened.set()
            proceed.wait(timeout=5)
        real_flock(fd, op)

    monkeypatch.setattr(fcntl, "flock", patched)
    return opened, proceed


class _OverlapState:
    """Results from `_overlap_recorder`, read after every thread joins."""

    def __init__(self) -> None:
        self.overlapped = False
        self.inodes: dict[str, int] = {}


def _overlap_recorder(
    lock_path: Path,
) -> tuple[Callable[..., None], _OverlapState]:
    """Build a `record(who, entered=None)` that marks `who` active for a
    beat, capturing which inode `lock_path` currently names.

    Shared by the two concurrency tests below, each of which calls `record`
    from more than one thread contending for the same agent lock:
    `state.overlapped` is true if two callers were ever active at once, and
    `state.inodes[who]` is what each one saw. A closure rather than a class
    with one method, so each test gets its own `active`/guard/`inodes`
    without instantiating anything.
    """
    active: list[str] = []
    active_guard = threading.Lock()
    state = _OverlapState()

    def record(who: str, entered: threading.Event | None = None) -> None:
        with active_guard:
            active.append(who)
            if len(active) > 1:
                state.overlapped = True
        if entered is not None:
            entered.set()
        state.inodes[who] = os.stat(lock_path).st_ino
        time.sleep(0.05)
        with active_guard:
            active.remove(who)

    return record, state


# The same schema as the orchestrator loads it from a file that declares its
# dialect. Its keys are deliberately unsorted, so a re-serialisation that lost
# the canonical ordering would be visible.


class TestOrchestrator:
    def test_an_orchestrator_requires_explicit_runtime_paths(self) -> None:
        constructor = cast(Callable[..., object], Orchestrator)
        with pytest.raises(TypeError, match="runtime_paths"):
            constructor()

    def test_runtime_paths_are_a_construction_snapshot(self, tmp_path: Path) -> None:
        workdir = tmp_path / "workdir"
        skills_dir = tmp_path / "skills"
        snapshot = runtime_paths(
            tmp_path,
            state_file=tmp_path / "state.json",
            workdir=workdir,
            skills_dir=skills_dir,
        )
        orch = Orchestrator(snapshot)

        assert orch.runtime_paths is snapshot
        assert orch.spawn("agent", "echo").workdir == workdir

    def test_an_explicit_state_file_overrides_the_runtime_snapshot(
        self, tmp_path: Path
    ) -> None:
        runtime_state_file = tmp_path / "runtime-state.json"
        explicit_state_file = tmp_path / "explicit-state.json"
        orch = Orchestrator(
            runtime_paths(tmp_path, state_file=runtime_state_file),
            state_file=explicit_state_file,
        )

        orch.spawn("agent", "echo")

        assert explicit_state_file.is_file()
        assert not runtime_state_file.exists()

    def test_validated_opencode_turn_warns_once_about_cli_schema_enforcement(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        class AdvisoryBackend(AgentBackend):
            name = "advisory"
            enforces_schema = False

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
                return TurnResult(
                    session_id="s1",
                    reply="{}",
                    raw="",
                    structured={},
                )

        register_backend("advisory", AdvisoryBackend)
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "state.json"))
        orch.spawn("agent", "advisory")

        with caplog.at_level(logging.WARNING, logger="orchestrator"):
            result = orch.talk("agent", "hello", schema=SCHEMA, retries=0)

        assert result.structured == {}
        assert _messages(caplog) == [
            "backend advisory: schema is enforced via validation/repair, not the CLI"
        ]

    def test_a_non_enforcing_backend_is_sent_the_schema_document(
        self, tmp_path: Path
    ) -> None:
        """The CLI has no flag to carry it, so the prompt has to. Without this
        the model is told to conform to a schema it was never shown."""
        seen: list[str] = []

        class AdvisoryBackend(AgentBackend):
            name = "advisory-prompt"
            enforces_schema = False

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
                seen.append(prompt)
                return TurnResult(session_id="s1", reply="{}", raw="", structured={})

        register_backend("advisory-prompt", AdvisoryBackend)
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "state.json"))
        orch.spawn("agent", "advisory-prompt")
        orch.talk("agent", "hello", schema=SCHEMA, retries=0)

        assert seen == [compose_schema_prompt("hello", SCHEMA)]
        assert SCHEMA.text in seen[0]

    def test_an_enforcing_backend_is_not_sent_the_schema_document(
        self, tmp_path: Path
    ) -> None:
        seen: list[str] = []

        class EnforcingBackend(AgentBackend):
            name = "enforcing-prompt"

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
                seen.append(prompt)
                return TurnResult(session_id="s1", reply="{}", raw="", structured={})

        register_backend("enforcing-prompt", EnforcingBackend)
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "state.json"))
        orch.spawn("agent", "enforcing-prompt")
        orch.talk("agent", "hello", schema=SCHEMA, retries=0)

        assert seen == [compose_schema_prompt("hello")]
        assert SCHEMA.text not in seen[0]

    def test_schema_warning_is_absent_without_schema_or_for_enforcing_backend(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        class EnforcingBackend(AgentBackend):
            name = "enforcing"

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
                return TurnResult(
                    session_id="s1",
                    reply="{}",
                    raw="",
                    structured={},
                )

        register_backend("enforcing", EnforcingBackend)
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "state.json"))
        orch.spawn("echo", "enforcing")

        with caplog.at_level(logging.WARNING, logger="orchestrator"):
            orch.talk("echo", "plain")
            orch.talk("echo", "structured", schema=SCHEMA, retries=0)

        assert _messages(caplog) == []

    def test_spawn_talk_persists_and_resumes(self, tmp_path: Path, monkeypatch) -> None:
        state_file = tmp_path / "state.json"
        seen_session_ids: list[str | None] = []
        # A literal, not a module-level path: comparing a module constant
        # against itself would pass whatever it happened to be set to.
        workdir = tmp_path / "workdir"

        def fake_backend_run(args, **kwargs):
            _assert_subprocess_kwargs(kwargs, workdir)
            return subprocess.CompletedProcess(
                args,
                0,
                stdout=json.dumps(
                    {"is_error": False, "session_id": "persist-me", "result": "reply"}
                ),
                stderr="",
            )

        class RecordingBackend(AgentBackend):
            @property
            def name(self) -> str:
                return "recording"

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
                seen_session_ids.append(session_id)
                return TurnResult(session_id="persist-me", reply="reply", raw="")

        register_backend("recording", RecordingBackend)
        monkeypatch.setattr(subprocess, "run", fake_backend_run)
        path_snapshot = runtime_paths(tmp_path, state_file=state_file, workdir=workdir)

        orch = Orchestrator(path_snapshot)
        agent = orch.spawn("a1", "claude")
        assert agent.session_id is None

        orch2 = Orchestrator(path_snapshot)
        agent2 = orch2.spawn("a2", "recording")
        assert agent2.name == "a2"
        orch2.talk("a2", "first")
        orch2.talk("a2", "second")
        assert seen_session_ids == [None, "persist-me"]

        result = orch.talk("a1", "first")
        assert result.reply == "reply"
        assert orch.agents["a1"].session_id == "persist-me"

        orch3 = Orchestrator(path_snapshot)
        assert "a1" in orch3.agents
        assert orch3.agents["a1"].name == "a1"
        assert orch3.agents["a1"].session_id == "persist-me"
        assert orch3.talk("a1", "second").reply == "reply"

    def test_persist_writes_sorted_indented_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state_file = tmp_path / "state.json"
        monkeypatch.setattr(core, "_utcnow", lambda: "2026-08-25T00:00:00Z")
        orch = Orchestrator(runtime_paths(tmp_path, state_file=state_file))
        orch.spawn("b", "codex")
        orch.spawn("a", "claude")
        assert state_file.read_text(encoding="utf-8") == (
            "{\n"
            '  "a": {\n'
            '    "backend": "claude",\n'
            '    "created_at": "2026-08-25T00:00:00Z",\n'
            '    "session_id": null,\n'
            '    "turns": 0\n'
            "  },\n"
            '  "b": {\n'
            '    "backend": "codex",\n'
            '    "created_at": "2026-08-25T00:00:00Z",\n'
            '    "session_id": null,\n'
            '    "turns": 0\n'
            "  }\n"
            "}\n"
        )

    def test_spawn_defaults_to_claude_backend(self, tmp_path: Path) -> None:
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "s.json"))
        agent = orch.spawn("a1")
        assert agent.backend.name == "claude"

    def test_spawn_persists_a_grok_agent(self, tmp_path: Path) -> None:
        state_file = tmp_path / "s.json"
        orch = Orchestrator(runtime_paths(tmp_path, state_file=state_file))
        agent = orch.spawn("g", "grok")
        assert agent.backend.name == "grok"
        loaded = Orchestrator(runtime_paths(tmp_path, state_file=state_file))
        assert loaded.agents["g"].backend.name == "grok"
        assert loaded.agents["g"].session_id is None

    def test_spawn_persists_model_and_reasoning_effort(self, tmp_path: Path) -> None:
        state_file = tmp_path / "s.json"
        orch = Orchestrator(runtime_paths(tmp_path, state_file=state_file))
        agent = orch.spawn(
            "configured",
            "codex",
            model="gpt-test",
            reasoning_effort="high",
        )
        assert agent.backend.model == "gpt-test"
        assert agent.backend.reasoning_effort == "high"

        loaded = (
            Orchestrator(runtime_paths(tmp_path, state_file=state_file))
            .agents["configured"]
            .backend
        )
        assert loaded.name == "codex"
        assert loaded.model == "gpt-test"
        assert loaded.reasoning_effort == "high"

    def test_spawn_duplicate_raises(self, tmp_path: Path) -> None:
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "s.json"))
        orch.spawn("a1", "claude")
        with pytest.raises(ValueError, match="already exists"):
            orch.spawn("a1", "claude")

    def test_ensure_creates_a_missing_agent(self, tmp_path: Path) -> None:
        state_file = tmp_path / "s.json"
        orch = Orchestrator(runtime_paths(tmp_path, state_file=state_file))
        agent, created = orch.ensure("fresh", "echo")
        assert created is True
        assert agent.name == "fresh"
        assert agent.backend.name == "echo"
        assert Orchestrator(
            runtime_paths(tmp_path, state_file=state_file)
        ).list_agents() == ["fresh"]

    def test_ensure_defaults_to_the_default_backend(self, tmp_path: Path) -> None:
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "s.json"))
        agent, _created = orch.ensure("fresh")
        assert agent.backend.name == core.DEFAULT_BACKEND

    def test_ensure_returns_an_existing_agent_untouched(self, tmp_path: Path) -> None:
        """An existing agent keeps its backend and its session, not a new one."""
        state_file = tmp_path / "s.json"
        orch = Orchestrator(runtime_paths(tmp_path, state_file=state_file))
        orch.spawn("a", "echo")
        orch.talk("a", "hi")
        agent, created = orch.ensure("a", "codex")
        assert created is False
        assert agent.backend.name == "echo"
        assert agent.session_id == "echo-sid"

    def test_ensure_takes_an_exclusive_lock(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The lookup and the create are one critical section, not two."""
        seen: list[int] = []

        def fake_flock(_fd: int, op: int) -> None:
            seen.append(op)

        monkeypatch.setattr(fcntl, "flock", fake_flock)
        Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "s.json")).ensure(
            "a", "echo"
        )
        assert fcntl.LOCK_EX in seen
        assert fcntl.LOCK_UN in seen

    def test_talk_unknown_raises(self, tmp_path: Path) -> None:
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "s.json"))
        with pytest.raises(KeyError, match="no agent named"):
            orch.talk("nope", "hi")

    def test_chat_runs_the_interactive_cli_without_state_mutation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state_file = tmp_path / "state.json"
        seen_argv: list[tuple[str, Path]] = []

        class ChatBackend(AgentBackend):
            name = "chat-backend"
            supports_chat = True

            def chat_argv(self, session_id: str, cwd: Path) -> list[str]:
                seen_argv.append((session_id, cwd))
                return ["chat-cli", session_id]

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
                raise AssertionError("chat test must not run a headless turn")

        register_backend("chat-backend", ChatBackend)
        orch = Orchestrator(runtime_paths(tmp_path, state_file=state_file))
        agent = orch.spawn("a", "chat-backend")
        agent.session_id = "session-1"
        calls: list[tuple[list[str], dict]] = []

        def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess:
            calls.append((args, kwargs))
            return subprocess.CompletedProcess(args, 17)

        monkeypatch.setattr(core.subprocess, "run", fake_run)
        monkeypatch.setattr(
            orch,
            "_exclusive",
            lambda: (_ for _ in ()).throw(
                AssertionError("chat must not take the state lock")
            ),
        )

        state_file.write_text(
            json.dumps({"a": {"backend": "chat-backend", "session_id": "new-session"}}),
            encoding="utf-8",
        )
        before = state_file.read_bytes()

        assert orch.chat("a") == 17
        assert seen_argv == [("new-session", agent.workdir)]
        assert calls == [
            (
                ["chat-cli", "new-session"],
                {"cwd": str(agent.workdir), "check": False},
            )
        ]
        assert state_file.read_bytes() == before

    def test_chat_unknown_agent_refuses_before_launching_cli(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "state.json"))
        monkeypatch.setattr(
            core.subprocess,
            "run",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("unknown agent launched a CLI")
            ),
        )

        with (
            caplog.at_level(logging.DEBUG, logger="orchestrator"),
            pytest.raises(KeyError, match="no agent named 'missing'"),
        ):
            orch.chat("missing")
        assert not orch._agent_lock_path("missing").exists()
        assert [record.getMessage() for record in caplog.records] == [
            "agent 'missing': reclaimed lock file, no such agent"
        ]

    def test_chat_refuses_an_agent_without_a_materialized_session(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "state.json"))
        orch.spawn("a", "echo")
        monkeypatch.setattr(
            core.subprocess,
            "run",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("session-less agent launched a CLI")
            ),
        )

        with pytest.raises(
            core.OrchestratorError,
            match="agent 'a' has no session to fork yet; talk to it first",
        ):
            orch.chat("a")

    def test_chat_refuses_a_pending_fork_before_launching_cli(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "state.json"))
        agent = orch.spawn("a", "echo")
        agent.pending_fork_from = "source-session"
        orch.state_file.write_text(
            json.dumps(
                {
                    "a": {
                        "backend": "echo",
                        "pending_fork_from": "source-session",
                    }
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            core.subprocess,
            "run",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("pending fork launched a CLI")
            ),
        )

        with pytest.raises(
            core.OrchestratorError,
            match="agent 'a' has no session to fork yet; talk to it first",
        ):
            orch.chat("a")

    def test_chat_refuses_a_backend_without_interactive_support(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "state.json"))
        agent = orch.spawn("a", "echo")
        agent.session_id = "session-1"
        orch.state_file.write_text(
            json.dumps({"a": {"backend": "echo", "session_id": "session-1"}}),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            core.subprocess,
            "run",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("unsupported backend launched a CLI")
            ),
        )

        with pytest.raises(
            core.OrchestratorError,
            match="agent 'a' runs on backend 'echo', which cannot chat",
        ):
            orch.chat("a")

    def test_chat_refuses_a_busy_agent_without_waiting_or_launching_cli(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "state.json"))
        agent = orch.spawn("a", "echo")
        agent.session_id = "session-1"
        monkeypatch.setattr(
            core.subprocess,
            "run",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("busy agent launched a CLI")
            ),
        )

        with (
            orch._agent_lock("a"),
            pytest.raises(
                AgentBusyError,
                match="agent 'a' is in use by another command; try again once it finishes",
            ),
        ):
            orch.chat("a")

    def test_list_agents(self, tmp_path: Path) -> None:
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "s.json"))
        orch.spawn("b", "codex")
        orch.spawn("a", "claude")
        assert orch.list_agents() == ["a", "b"]

    def test_delete_agent(self, tmp_path: Path, monkeypatch) -> None:
        state_file = tmp_path / "state.json"
        # A literal, not a module-level path: comparing a module constant
        # against itself would pass whatever it happened to be set to.
        workdir = tmp_path / "workdir"

        def fake_backend_run(args, **kwargs):
            _assert_subprocess_kwargs(kwargs, workdir)
            return subprocess.CompletedProcess(
                args,
                0,
                stdout=json.dumps(
                    {"is_error": False, "session_id": "s1", "result": "reply"}
                ),
                stderr="",
            )

        monkeypatch.setattr(subprocess, "run", fake_backend_run)

        path_snapshot = runtime_paths(tmp_path, state_file=state_file, workdir=workdir)
        orch = Orchestrator(path_snapshot)
        orch.spawn("a", "claude")
        orch.spawn("b", "codex")
        orch.delete("a")
        assert orch.list_agents() == ["b"]

        orch2 = Orchestrator(path_snapshot)
        assert orch2.list_agents() == ["b"]

    def test_delete_unknown_raises(self, tmp_path: Path) -> None:
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "s.json"))
        with pytest.raises(KeyError, match="no agent named"):
            orch.delete("nope")

    def test_reloaded_agent_keeps_the_name_from_state(self, tmp_path: Path) -> None:
        state_file = tmp_path / "s.json"
        Orchestrator(runtime_paths(tmp_path, state_file=state_file)).spawn(
            "named", "echo"
        )
        loaded = Orchestrator(runtime_paths(tmp_path, state_file=state_file))
        assert loaded.agents["named"].name == "named"

    def test_agent_talk_resumes_on_the_same_instance(self, tmp_path: Path) -> None:
        seen: list[str | None] = []

        class Rec(AgentBackend):
            @property
            def name(self) -> str:
                return "rec"

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
                seen.append(session_id)
                return TurnResult(session_id="sid", reply="r", raw="")

        register_backend("rec", Rec)
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "s.json"))
        orch.spawn("a", "rec")
        agent = orch.agents["a"]
        agent.talk("one")
        agent.talk("two")
        assert seen == [None, "sid"]

    def test_spawn_takes_an_exclusive_lock(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[int] = []

        def fake_flock(_fd: int, op: int) -> None:
            seen.append(op)

        monkeypatch.setattr(fcntl, "flock", fake_flock)
        Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "s.json")).spawn(
            "a", "echo"
        )
        assert fcntl.LOCK_EX in seen
        assert fcntl.LOCK_UN in seen
        assert (tmp_path / "s.json.lock").is_file()

    def test_talk_keeps_agents_spawned_during_the_turn(self, tmp_path: Path) -> None:
        """A long talk must not clobber a concurrent spawn (issue #2)."""
        state_file = tmp_path / "state.json"

        class MidTurnSpawn(AgentBackend):
            @property
            def name(self) -> str:
                return "midturn"

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
                Orchestrator(runtime_paths(tmp_path, state_file=state_file)).spawn(
                    "b", "echo"
                )
                return TurnResult(session_id="s1", reply="ok", raw="")

        register_backend("midturn", MidTurnSpawn)
        orch = Orchestrator(runtime_paths(tmp_path, state_file=state_file))
        orch.spawn("a", "midturn")
        orch.talk("a", "hi")
        reloaded = Orchestrator(runtime_paths(tmp_path, state_file=state_file))
        assert reloaded.list_agents() == ["a", "b"]
        assert reloaded.agents["a"].session_id == "s1"

    def test_talk_holds_a_per_agent_lock_for_the_whole_turn(
        self, tmp_path: Path
    ) -> None:
        """Two turns on one agent must not both resume its session."""
        state_file = tmp_path / "state.json"
        held: list[tuple[bool, bool]] = []

        class ProbeLock(AgentBackend):
            @property
            def name(self) -> str:
                return "probelock"

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
                probe = Orchestrator(runtime_paths(tmp_path, state_file=state_file))
                held.append(
                    (
                        _flock_is_held(probe._agent_lock_path("a")),
                        _flock_is_held(probe._agent_lock_path("other")),
                    )
                )
                return TurnResult(session_id="s1", reply="ok", raw="")

        register_backend("probelock", ProbeLock)
        orch = Orchestrator(runtime_paths(tmp_path, state_file=state_file))
        orch.spawn("a", "probelock")
        orch.talk("a", "hi")
        # Locked for this agent only, and released once the turn is over.
        assert held == [(True, False)]
        assert _flock_is_held(orch._agent_lock_path("a")) is False

    def test_agent_lock_paths_sit_in_the_locks_dir(self, tmp_path: Path) -> None:
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "state.json"))
        first = orch._agent_lock_path("a")
        assert first.parent == tmp_path / "state.json.locks"
        assert first.name == hashlib.sha256(b"a").hexdigest()
        assert first != orch._agent_lock_path("b")
        assert first != orch._lock_path()

    def test_a_backend_reporting_no_session_keeps_the_stored_one(
        self, tmp_path: Path
    ) -> None:
        """Overwriting a live session id with None restarts the conversation."""
        state_file = tmp_path / "state.json"
        replies = iter([TurnResult("s1", "first", ""), TurnResult(None, "second", "")])

        class Forgetful(AgentBackend):
            @property
            def name(self) -> str:
                return "forgetful"

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
                return next(replies)

        register_backend("forgetful", Forgetful)
        orch = Orchestrator(runtime_paths(tmp_path, state_file=state_file))
        orch.spawn("a", "forgetful")
        orch.talk("a", "one")
        assert orch.talk("a", "two").reply == "second"
        assert orch.agents["a"].session_id == "s1"
        assert (
            Orchestrator(runtime_paths(tmp_path, state_file=state_file))
            .agents["a"]
            .session_id
            == "s1"
        )

    def test_corrupt_state_names_the_file(self, tmp_path: Path) -> None:
        state_file = tmp_path / "state.json"
        state_file.write_text("{", encoding="utf-8")
        with pytest.raises(core.StateError) as excinfo:
            Orchestrator(runtime_paths(tmp_path, state_file=state_file))
        assert str(excinfo.value).startswith(f"{state_file} is not valid JSON: ")

    def test_state_entry_without_a_backend_is_reported(self, tmp_path: Path) -> None:
        state_file = tmp_path / "state.json"
        state_file.write_text('{"a": {"session_id": "s1"}}', encoding="utf-8")
        with pytest.raises(core.StateError) as excinfo:
            Orchestrator(runtime_paths(tmp_path, state_file=state_file))
        assert excinfo.value.args[0] == f"{state_file}: agent 'a' has no backend"

    def test_state_naming_an_unknown_backend_is_reported(self, tmp_path: Path) -> None:
        state_file = tmp_path / "state.json"
        state_file.write_text('{"a": {"backend": "ghost"}}', encoding="utf-8")
        with pytest.raises(UnknownBackendError, match="Unknown backend 'ghost'"):
            Orchestrator(runtime_paths(tmp_path, state_file=state_file))

    def test_talk_fails_if_the_agent_is_deleted_during_the_turn(
        self, tmp_path: Path
    ) -> None:
        state_file = tmp_path / "state.json"

        class DeleteDuring(AgentBackend):
            @property
            def name(self) -> str:
                return "delduring"

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
                Orchestrator(runtime_paths(tmp_path, state_file=state_file)).delete("a")
                return TurnResult(session_id="s1", reply="ok", raw="")

        register_backend("delduring", DeleteDuring)
        orch = Orchestrator(runtime_paths(tmp_path, state_file=state_file))
        orch.spawn("a", "delduring")
        with pytest.raises(KeyError, match="no agent named 'a'"):
            orch.talk("a", "hi")
        assert (
            Orchestrator(runtime_paths(tmp_path, state_file=state_file)).list_agents()
            == []
        )

    def test_delete_on_an_idle_agent_unlinks_its_lock_file(
        self, tmp_path: Path
    ) -> None:
        state_file = tmp_path / "state.json"
        orch = Orchestrator(runtime_paths(tmp_path, state_file=state_file))
        orch.spawn("a", "echo")
        orch.talk("a", "hi")
        lock_path = orch._agent_lock_path("a")
        assert lock_path.is_file()

        orch.delete("a")
        assert not lock_path.exists()

    def test_delete_during_its_own_turn_keeps_the_lock_file(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        state_file = tmp_path / "state.json"
        entered = threading.Event()
        release = threading.Event()

        register_backend("gate-self", _gate_backend("gate-self", entered, release))
        orch = Orchestrator(runtime_paths(tmp_path, state_file=state_file))
        orch.spawn("a", "gate-self")

        def run_turn() -> None:
            # Deleting the agent mid-turn (below) makes the registry entry
            # this turn resumes into vanish before it persists, which is the
            # pre-existing AgentNotFoundError path this test isn't about —
            # the reclaim-on-that-path is covered separately, below.
            with pytest.raises(core.AgentNotFoundError):
                orch.talk("a", "hi")

        # daemon=True on both threads below: several asserts run before
        # their matching join(), and one firing must not leave a thread
        # running past this test.
        turn_thread = threading.Thread(target=run_turn, daemon=True)
        turn_thread.start()
        assert entered.wait(timeout=5)
        lock_path = orch._agent_lock_path("a")
        assert lock_path.is_file()

        # The probe loses to the in-flight turn and backs off rather than
        # refusing: `delete` never fails and never changes its exit code.
        # Run on its own thread and asserted to complete quickly, not
        # called synchronously here: a probe that regressed into blocking
        # (dropping delete's LOCK_NB) would deadlock this test against
        # `release` below instead of failing it — the whole point of a
        # non-blocking probe is that `delete` never waits for someone
        # else's turn, so that has to be checked, not assumed.
        results: list[Agent] = []
        delete_thread = threading.Thread(
            target=lambda: results.append(orch.delete("a")), daemon=True
        )
        with caplog.at_level("DEBUG", logger="orchestrator"):
            delete_thread.start()
            delete_thread.join(timeout=5)
        assert not delete_thread.is_alive(), "delete blocked instead of backing off"
        agent = results[0]
        assert agent.name == "a"
        assert lock_path.is_file()
        assert "agent 'a': lock file in use, not reclaiming" in [
            r.getMessage() for r in caplog.records
        ]

        release.set()
        turn_thread.join(timeout=5)
        # Once the turn ends, its own AgentNotFoundError handler reclaims
        # what delete's probe could not — see the next test.
        assert not lock_path.exists()

    def test_a_turn_reclaims_its_lock_when_deleted_mid_turn(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        state_file = tmp_path / "state.json"

        class DeleteDuring(AgentBackend):
            @property
            def name(self) -> str:
                return "delduring-reclaim"

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
                # Runs with the outer talk() holding the agent lock, so this
                # delete's own reclaim probe backs off (BlockingIOError,
                # caught inside _reclaim_agent_lock) rather than blocking on
                # a lock this very thread already holds through a different
                # fd — flock() isn't reentrant, so a probe that regressed
                # into blocking would deadlock this thread against itself,
                # not just wait. Run on its own thread and joined with a
                # timeout so that shows up as a clean failure instead; the
                # file is left for talk()'s AgentNotFoundError handler to
                # clean up.
                delete_thread = threading.Thread(
                    target=lambda: Orchestrator(
                        runtime_paths(tmp_path, state_file=state_file)
                    ).delete("a"),
                    daemon=True,
                )
                delete_thread.start()
                delete_thread.join(timeout=5)
                assert not delete_thread.is_alive(), (
                    "delete blocked instead of backing off"
                )
                return TurnResult(session_id="s1", reply="ok", raw="")

        register_backend("delduring-reclaim", DeleteDuring)
        orch = Orchestrator(runtime_paths(tmp_path, state_file=state_file))
        orch.spawn("a", "delduring-reclaim")
        with (
            caplog.at_level("DEBUG", logger="orchestrator"),
            pytest.raises(core.AgentNotFoundError),
        ):
            orch.talk("a", "hi")
        assert list(orch._locks_dir().iterdir()) == []
        # Asserted through the log line, not just the empty directory: an
        # agent that was simply never talked to also leaves it empty.
        assert "agent 'a': reclaimed lock file, no such agent" in [
            r.getMessage() for r in caplog.records
        ]

    def test_delete_during_a_sibling_agents_turn_still_reclaims(
        self, tmp_path: Path
    ) -> None:
        """Rules out a home-wide lock: only the deleted agent's own turn
        should block a reclaim, not an unrelated agent's."""
        state_file = tmp_path / "state.json"
        entered = threading.Event()
        release = threading.Event()

        register_backend(
            "gate-sibling", _gate_backend("gate-sibling", entered, release)
        )
        orch = Orchestrator(runtime_paths(tmp_path, state_file=state_file))
        orch.spawn("bob", "gate-sibling")
        orch.spawn("alice", "echo")
        orch.talk("alice", "warm up")

        # daemon=True: the asserts below run before this thread's own
        # join(), and one firing must not leave it running past this test.
        turn_thread = threading.Thread(
            target=lambda: orch.talk("bob", "hi"), daemon=True
        )
        turn_thread.start()
        assert entered.wait(timeout=5)

        alice_lock = orch._agent_lock_path("alice")
        assert alice_lock.is_file()
        orch.delete("alice")
        assert not alice_lock.exists()

        release.set()
        turn_thread.join(timeout=5)

    def test_a_reclaim_racing_its_own_toctou_does_not_evict_a_fresh_acquirer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression for a TOCTOU in `_reclaim_agent_lock`.

        `_flock`'s open() and its flock() are two syscalls, not one: a probe
        can end up holding a lock on an inode a *different*, legitimate
        reclaim already orphaned in between, while a live acquirer has since
        put a fresh file at the same path. Unlinking blindly at that point
        destroys the live acquirer's file instead of the dead one the probe
        actually holds.

        Exercises the real `_reclaim_agent_lock`, not a hand-rolled stand-in:
        driving `_flock` directly instead misses this entirely, since
        nothing about a bare `_flock` call reproduces the probe's own
        open()-then-flock() window.
        """
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "state.json"))
        lock_path = orch._agent_lock_path("bob")
        record, state = _overlap_recorder(lock_path)

        # Pauses the reclaimer thread's first flock() call, after its
        # `_flock` has already open()ed the path — the window the bug lives
        # in — so a second, unpaused reclaim can legitimately acquire,
        # unlink, and release that same inode before this one's flock()
        # finally runs and is granted on the now-dead inode.
        reclaimer_opened, let_reclaimer_flock = _pause_first_flock(
            monkeypatch, "reclaimer"
        )
        waiter_entered = threading.Event()

        def reclaimer() -> None:
            orch._reclaim_agent_lock("bob")

        def waiter() -> None:
            with orch._agent_lock("bob"):
                record("waiter", entered=waiter_entered)

        def late_arrival() -> None:
            with orch._agent_lock("bob"):
                record("late")

        # daemon=True: several asserts below run before their matching
        # join(); one firing must not leave a thread running past this test
        # (a paused one, via _pause_first_flock, would otherwise survive it
        # by however long its own wait(timeout=...) takes to give up).
        t_reclaim = threading.Thread(target=reclaimer, name="reclaimer", daemon=True)
        t_reclaim.start()
        assert reclaimer_opened.wait(timeout=5)

        # A second, unpaused reclaim on the same (still-existing) inode the
        # paused thread already opened: it acquires, finds it live, unlinks,
        # and releases — a plain `delete` on an idle agent.
        orch._reclaim_agent_lock("bob")

        t_wait = threading.Thread(target=waiter, daemon=True)
        t_wait.start()
        assert waiter_entered.wait(timeout=5)

        let_reclaimer_flock.set()
        t_reclaim.join(timeout=5)

        t_late = threading.Thread(target=late_arrival, daemon=True)
        t_late.start()
        t_wait.join(timeout=5)
        t_late.join(timeout=5)

        assert not state.overlapped
        assert state.inodes["waiter"] == state.inodes["late"]

    def test_a_waiter_blocked_on_a_reclaimed_inode_reacquires_without_overlap(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression for the original race in the PR description: a
        distinct mechanism from the TOCTOU above, and from a genuinely
        distinct bug. Replacing this test with the TOCTOU one above,
        instead of adding it alongside, would leave `_agent_lock`'s
        `revalidate=True` completely untested.

        `_agent_lock`'s `flock()` can be granted on an inode a reclaimer has
        already unlinked and released out from under it — `flock` binds to
        the inode, not the path. `_flock`'s `revalidate=True` loop is what
        notices via `_is_live` and re-acquires instead of proceeding on a
        dead lock; disabling `revalidate`, or making `_is_live` treat a
        vanished path as still live, both make this test fail.
        """
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "state.json"))
        lock_path = orch._agent_lock_path("bob")
        record, state = _overlap_recorder(lock_path)

        # Pauses the waiter's first flock() call, after its `_agent_lock`
        # has already open()ed the still-existing path — so it shares
        # whatever inode the reclaim below is about to unlink and release.
        waiter_opened, let_waiter_flock = _pause_first_flock(monkeypatch, "waiter")
        waiter_entered = threading.Event()

        def waiter() -> None:
            with orch._agent_lock("bob"):
                record("waiter", entered=waiter_entered)

        def late_arrival() -> None:
            with orch._agent_lock("bob"):
                record("late")

        # daemon=True: see the same note in the TOCTOU test above.
        t_wait = threading.Thread(target=waiter, name="waiter", daemon=True)
        t_wait.start()
        assert waiter_opened.wait(timeout=5)

        # A plain reclaim on the same (still-existing) inode the paused
        # waiter already opened: it acquires, finds it live, unlinks, and
        # releases — the file the waiter is about to be granted a lock on.
        orch._reclaim_agent_lock("bob")
        let_waiter_flock.set()
        assert waiter_entered.wait(timeout=5)

        t_late = threading.Thread(target=late_arrival, daemon=True)
        t_late.start()
        t_wait.join(timeout=5)
        t_late.join(timeout=5)

        assert not state.overlapped
        assert state.inodes["waiter"] == state.inodes["late"]

    def test_talk_on_an_unknown_agent_leaves_no_lock_file(self, tmp_path: Path) -> None:
        """Written against `Orchestrator.talk` directly: `cmd_talk` creates
        the agent first, so `main(["talk", ...])` would assert something
        false here."""
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "state.json"))
        with pytest.raises(core.AgentNotFoundError):
            orch.talk("ghost", "hi")
        assert list(orch._locks_dir().iterdir()) == []

    def test_locks_dir_holds_one_file_per_live_agent(self, tmp_path: Path) -> None:
        state_file = tmp_path / "state.json"
        orch = Orchestrator(runtime_paths(tmp_path, state_file=state_file))
        orch.spawn("a", "echo")
        orch.spawn("b", "echo")
        orch.talk("a", "hi")
        orch.talk("b", "hi")

        locks_dir = orch._locks_dir()
        assert sorted(p.name for p in locks_dir.iterdir()) == sorted(
            path.name
            for path in (orch._agent_lock_path("a"), orch._agent_lock_path("b"))
        )

        orch.delete("a")
        orch.delete("b")
        assert list(locks_dir.iterdir()) == []
