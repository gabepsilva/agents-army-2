"""Unit tests for agent backend interface, implementations, and orchestrator."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import tomllib
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import cast

import pytest

import orchestrator.cli as cli
import orchestrator.core as core
import orchestrator.doctor as doctor
from backends.base import (
    DEFAULT_TURN_TIMEOUT,
    AgentBackend,
    OutputSchema,
    TurnError,
    TurnResult,
)
from backends.claude import (
    ClaudeTurnError,
)
from backends.codex import (
    CodexTurnError,
)
from backends.grok import (
    GrokTurnError,
)
from backends.registry import (
    UnknownBackendError,
    register_backend,
)
from orchestrator.cli import (
    cmd_create as _cmd_create,
)
from orchestrator.cli import (
    cmd_delete as _cmd_delete,
)
from orchestrator.cli import (
    cmd_list as _cmd_list,
)
from orchestrator.cli import (
    cmd_talk as _cmd_talk,
)
from orchestrator.cli import main
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
    _reported_seconds,
)
from tests.path_helpers import runtime_paths


def _talk_options(argv: list[str]) -> argparse.Namespace:
    separator = argv.index("--") if "--" in argv else None
    head = argv if separator is None else argv[:separator]
    tail = [] if separator is None else argv[separator + 1 :]
    options = cli._build_parser().parse_args(head)
    cli._resolve_talk_prompt(options, tail, separator is not None)
    return options


def _options(argv: list[str]) -> argparse.Namespace:
    return cli._build_parser().parse_args(argv)


def run_version(argv: list[str] | None = None) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"] if argv is None else argv)
    assert excinfo.value.code == 0


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

ALL_TOOLS_PRESENT = {
    "uv": "uv 0.4.18",
    "claude": "2.1.234 (Claude Code)",
    "codex": "codex-cli 0.147.0",
    "grok": "grok 1.0.5",
    "opencode": "1.18.21",
    "jq": "jq-1.7",
}


def _running_python() -> str:
    """The interpreter version the report is expected to name."""
    return ".".join(str(part) for part in sys.version_info[:3])


def _fake_dependency_env(
    monkeypatch: pytest.MonkeyPatch, versions: dict[str, str]
) -> list[list[str]]:
    """Stand in for PATH and for every `<tool> --version` process.

    Only the tools named in `versions` are on PATH; the rest are absent. The
    returned list records each command actually run, so a test can assert on
    the invocation and not merely that something was spawned.
    """
    calls: list[list[str]] = []

    def which(tool: str) -> str | None:
        return f"/usr/bin/{tool}" if tool in versions else None

    def run(args, **kwargs):
        calls.append(args)
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["check"] is False
        assert kwargs["timeout"] == 5
        assert kwargs["stdin"] == subprocess.DEVNULL
        return subprocess.CompletedProcess(args, 0, stdout=versions[args[0]], stderr="")

    monkeypatch.setattr(doctor.shutil, "which", which)
    monkeypatch.setattr(doctor.subprocess, "run", run)
    return calls


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


class TestCLI:
    """cmd_* dispatch and printed output, backed by a fake CLI-free backend."""

    def test_cmd_create_prints_confirmation(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "s.json"))
        _cmd_create(
            orch,
            _options(
                [
                    "create",
                    "a",
                    "-b",
                    "echo",
                    "--model",
                    "test-model",
                    "--reasoning-effort",
                    "high",
                ]
            ),
        )
        assert capsys.readouterr().out == "created agent 'a' backend=echo\n"
        assert orch.agents["a"].backend.model == "test-model"
        assert orch.agents["a"].backend.reasoning_effort == "high"

    def test_cmd_create_defaults_to_claude_backend(self, tmp_path: Path) -> None:
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "s.json"))
        _cmd_create(orch, _options(["create", "a"]))
        assert orch.agents["a"].backend.name == "claude"

    def test_cmd_create_follows_default_backend(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DEFAULT_BACKEND documents itself as the one `create` uses.

        An argparse default of its own would leave `create` on claude however
        that constant changed, so the two would silently disagree.
        """
        monkeypatch.setattr(core, "DEFAULT_BACKEND", "codex")
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "s.json"))
        _cmd_create(orch, _options(["create", "a"]))
        assert orch.agents["a"].backend.name == "codex"

    def test_cmd_create_rejects_unknown_backend(self, tmp_path: Path) -> None:
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "s.json"))
        with pytest.raises(SystemExit, match="2"):
            _cmd_create(orch, _options(["create", "a", "-b", "not-a-backend"]))

    def test_cmd_create_missing_name_reports_its_own_prog(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "s.json"))
        with pytest.raises(SystemExit, match="2"):
            _cmd_create(orch, _options(["create"]))
        assert capsys.readouterr().err.startswith("usage: orchestrator create ")

    def test_cmd_talk_prints_reply(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "s.json"))
        orch.spawn("a", "echo")
        _cmd_talk(orch, _talk_options(["talk", "a", "--", "hello", "there"]))
        out = capsys.readouterr().out
        assert "session=echo-sid" in out
        assert "echo:hello there" in out

    def test_main_chat_propagates_child_status_without_writing_registry(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        class ChatBackend(AgentBackend):
            name = "chat-status"
            supports_chat = True

            def chat_argv(self, session_id: str, cwd: Path) -> list[str]:
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

        register_backend("chat-status", ChatBackend)
        state_file = tmp_path / "state.json"
        state_file.write_text(
            json.dumps({"a": {"backend": "chat-status", "session_id": "s1"}}),
            encoding="utf-8",
        )
        before = state_file.read_bytes()
        monkeypatch.setenv("AGENTS_ARMY_STATE_FILE", str(state_file))
        monkeypatch.setenv("AGENTS_ARMY_HOME", str(tmp_path))
        calls: list[tuple[list[str], dict]] = []

        def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess:
            calls.append((args, kwargs))
            return subprocess.CompletedProcess(args, 7)

        monkeypatch.setattr(core.subprocess, "run", fake_run)

        with pytest.raises(SystemExit) as excinfo:
            main(["chat", "a"])

        assert excinfo.value.code == 7
        captured = capsys.readouterr()
        assert calls == [(["chat-cli", "s1"], {"cwd": str(tmp_path), "check": False})]
        assert captured.out == ""
        assert captured.err == ""
        assert state_file.read_bytes() == before

    def test_cmd_talk_creates_a_missing_agent(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Talking to a name that does not exist spawns it and runs the turn."""
        monkeypatch.setattr(core, "DEFAULT_BACKEND", "echo")
        state_file = tmp_path / "s.json"
        orch = Orchestrator(runtime_paths(tmp_path, state_file=state_file))
        _cmd_talk(orch, _talk_options(["talk", "fresh", "--", "hello"]))
        captured = capsys.readouterr()
        assert captured.err == "created agent 'fresh' backend=echo\n"
        assert "echo:hello" in captured.out
        assert Orchestrator(
            runtime_paths(tmp_path, state_file=state_file)
        ).list_agents() == ["fresh"]

    def test_cmd_talk_says_nothing_extra_for_an_existing_agent(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "s.json"))
        orch.spawn("a", "echo")
        _cmd_talk(orch, _talk_options(["talk", "a", "--", "hello"]))
        assert capsys.readouterr().err == ""

    def test_cmd_talk_flags_create_exact_configuration(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "s.json"))
        _cmd_talk(
            orch,
            _talk_options(
                [
                    "talk",
                    "-b",
                    "echo",
                    "--model",
                    "m",
                    "--reasoning-effort",
                    "high",
                    "fresh",
                    "-p",
                    "hello",
                ],
            ),
        )
        assert capsys.readouterr().err == "created agent 'fresh' backend=echo\n"
        assert cli._agent_config(orch.agents["fresh"]) == ("echo", "m", "high")

    def test_cmd_talk_matching_flags_reuse_silently(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "s.json"))
        orch.spawn("a", "echo", model="m", reasoning_effort="high")
        _cmd_talk(
            orch,
            _talk_options(
                [
                    "talk",
                    "-b",
                    "echo",
                    "--model",
                    "m",
                    "--reasoning-effort",
                    "high",
                    "a",
                    "-p",
                    "hi",
                ]
            ),
        )
        assert capsys.readouterr().err == ""

    def test_cmd_talk_mismatch_reports_the_exact_configuration_tuple(
        self, tmp_path: Path
    ) -> None:
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "s.json"))
        orch.spawn("a", "echo", model="first", reasoning_effort="high")
        with pytest.raises(
            core.OrchestratorError,
            match=r"agent 'a' already uses backend/model/effort \('echo', 'first', 'high'\); configured \('codex', None, None\)",
        ):
            _cmd_talk(orch, _talk_options(["talk", "-b", "codex", "a", "-p", "hi"]))

    def test_cmd_talk_omitting_model_still_asserts_the_exact_tuple(
        self, tmp_path: Path
    ) -> None:
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "s.json"))
        orch.spawn("a", "echo", model="stored")
        with pytest.raises(
            core.OrchestratorError,
            match=r"configured \('echo', None, None\)$",
        ):
            _cmd_talk(orch, _talk_options(["talk", "-b", "echo", "a", "-p", "hi"]))

    def test_cmd_talk_without_config_flags_reuses_any_stored_config(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "s.json"))
        orch.spawn("a", "echo", model="stored", reasoning_effort="high")
        _cmd_talk(orch, _talk_options(["talk", "a", "--", "hi"]))
        assert capsys.readouterr().err == ""

    def test_cmd_talk_rejects_unknown_backend(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit, match="2"):
            _talk_options(["talk", "-b", "unknown", "a", "-p", "hi"])

    def test_talk_flags_after_name_are_rejected_before_agent_creation(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        state_file = tmp_path / "s.json"
        monkeypatch.setenv("AGENTS_ARMY_STATE_FILE", str(state_file))
        monkeypatch.setattr(
            core,
            "Orchestrator",
            lambda *_: (_ for _ in ()).throw(AssertionError("constructed")),
        )
        with pytest.raises(SystemExit, match="2"):
            main(["talk", "a", "hi", "--timeout", "5"])
        assert "usage: orchestrator talk " in capsys.readouterr().err
        assert not state_file.exists()

    def test_cmd_talk_empty_prompt_creates_nothing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A rejected invocation must not leave an agent behind."""
        state_file = tmp_path / "s.json"
        with pytest.raises(SystemExit, match="2"):
            _talk_options(["talk", "fresh", "-p", "   "])
        assert (
            Orchestrator(runtime_paths(tmp_path, state_file=state_file)).list_agents()
            == []
        )

    def test_cmd_talk_empty_prompt_exits_nonzero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Exit 0 here reads as a turn that ran to a caller under `set -e`."""
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "s.json"))
        orch.spawn("a", "echo")
        with pytest.raises(SystemExit, match="2"):
            _talk_options(["talk", "a", "-p", "   "])
        captured = capsys.readouterr()
        assert "usage: orchestrator talk " in captured.err
        assert "talk prompt must not be empty" in captured.err
        assert captured.out == ""

    def test_cmd_talk_missing_name_reports_its_own_prog(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit, match="2"):
            _options(["talk"])
        assert capsys.readouterr().err.startswith("usage: orchestrator talk ")

    def test_cmd_talk_prints_backend_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        class BoomBackend(AgentBackend):
            @property
            def name(self) -> str:
                return "boom"

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
                raise ClaudeTurnError("claude output was not JSON")

        register_backend("boom", BoomBackend)
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "s.json"))
        orch.spawn("b", "boom")
        monkeypatch.setattr(core, "Orchestrator", lambda *_: orch)
        with pytest.raises(SystemExit, match="1"):
            main(["talk", "b", "-p", "hi"])
        captured = capsys.readouterr()
        assert captured.err == "claude output was not JSON\n"
        assert captured.out == ""

    def test_cmd_talk_prints_codex_backend_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        class BoomCodex(AgentBackend):
            @property
            def name(self) -> str:
                return "boomcodex"

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
                raise CodexTurnError("codex did not report a thread_id")

        register_backend("boomcodex", BoomCodex)
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "s.json"))
        orch.spawn("c", "boomcodex")
        monkeypatch.setattr(core, "Orchestrator", lambda *_: orch)
        with pytest.raises(SystemExit, match="1"):
            main(["talk", "c", "-p", "hi"])
        assert capsys.readouterr().err == "codex did not report a thread_id\n"

    def test_cmd_talk_prints_any_turn_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The CLI catches TurnError, so a new backend does not need a new except."""

        class BoomAny(AgentBackend):
            @property
            def name(self) -> str:
                return "boomany"

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
                raise TurnError("cli failed")

        register_backend("boomany", BoomAny)
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "s.json"))
        orch.spawn("d", "boomany")
        monkeypatch.setattr(core, "Orchestrator", lambda *_: orch)
        with pytest.raises(SystemExit, match="1"):
            main(["talk", "d", "-p", "hi"])
        captured = capsys.readouterr()
        assert captured.err == "cli failed\n"
        assert captured.out == ""

    @pytest.mark.parametrize(
        ("incidental", "message"),
        [(RuntimeError, "dict changed size"), (ValueError, "bad literal")],
    )
    def test_an_incidental_builtin_from_a_backend_still_gets_a_traceback(
        self,
        incidental: type[Exception],
        message: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The boundary lists leaf types, never the bases they inherit from.

        `TurnError` is a `RuntimeError` and `UnknownBackendError` is a
        `ValueError`; catching either base would reprint a genuine bug inside
        a backend as a one-line user mistake with the traceback thrown away.
        """

        class IncidentalBackend(AgentBackend):
            @property
            def name(self) -> str:
                return "incidental"

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
                raise incidental(message)

        register_backend("incidental", IncidentalBackend)
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "s.json"))
        orch.spawn("i", "incidental")
        monkeypatch.setattr(core, "Orchestrator", lambda *_: orch)
        with pytest.raises(incidental, match=message):
            main(["talk", "i", "-p", "hi"])
        assert capsys.readouterr().err == ""

    def test_cmd_talk_prints_grok_backend_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        class BoomGrok(AgentBackend):
            @property
            def name(self) -> str:
                return "boomgrok"

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
                raise GrokTurnError("grok did not report a sessionId")

        register_backend("boomgrok", BoomGrok)
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "s.json"))
        orch.spawn("g", "boomgrok")
        monkeypatch.setattr(core, "Orchestrator", lambda *_: orch)
        with pytest.raises(SystemExit, match="1"):
            main(["talk", "g", "-p", "hi"])
        assert capsys.readouterr().err == "grok did not report a sessionId\n"

    def test_cmd_create_accepts_grok(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "s.json"))
        _cmd_create(orch, _options(["create", "a", "-b", "grok"]))
        assert capsys.readouterr().out == "created agent 'a' backend=grok\n"
        assert orch.agents["a"].backend.name == "grok"

    def test_cmd_list_empty(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        state_file = tmp_path / "s.json"
        orch = Orchestrator(runtime_paths(tmp_path, state_file=state_file))
        _cmd_list(orch, _options(["list"]))
        assert capsys.readouterr().out == f"registry: {state_file}\nno agents\n"

    def test_cmd_list_prints_each_agent(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "s.json"))
        orch.spawn("a", "echo")
        _cmd_list(orch, _options(["list"]))
        out = capsys.readouterr().out
        assert "a" in out
        assert "backend=echo" in out
        assert "session=-" in out

    def test_cmd_list_rejects_unexpected_arg_reporting_its_own_prog(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit, match="2"):
            _options(["list", "unexpected"])
        assert capsys.readouterr().err.startswith("usage: orchestrator list ")

    def test_cmd_delete_prints_confirmation(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "s.json"))
        orch.spawn("a", "echo")
        _cmd_delete(orch, _options(["delete", "a"]))
        assert "deleted agent 'a' backend=echo" in capsys.readouterr().out

    def test_cmd_delete_missing_name_reports_its_own_prog(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # `name` is nargs="?" now (`--team T` alone tears a team down), so
        # argparse itself no longer rejects a bare `delete`; main() does.
        with pytest.raises(SystemExit, match="2"):
            main(["delete"])
        assert capsys.readouterr().err.startswith("usage: orchestrator delete ")

    def test_main_no_args_prints_usage_and_exits(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit, match="2"):
            main([])
        captured = capsys.readouterr()
        assert captured.err.startswith("usage: orchestrator ")
        assert "the following arguments are required: <verb>" in captured.err
        assert captured.out == ""

    @pytest.mark.parametrize("flags", [[], ["-v"], ["--verbose"], ["-vv"], ["-vvv"]])
    def test_main_version_is_clean_and_early(
        self,
        flags: list[str],
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        def fail_orchestrator() -> None:
            raise AssertionError("version must not construct Orchestrator")

        monkeypatch.setattr(core, "Orchestrator", fail_orchestrator)
        run_version([*flags, "--version", "ignored"])
        captured = capsys.readouterr()
        assert captured.out == "0.1.0\n"
        assert captured.err == ""

    def test_main_version_prefers_checkout_metadata(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(
            doctor.importlib.metadata,
            "version",
            lambda _: "installed-version",
        )
        run_version()
        assert capsys.readouterr().out == "0.1.0\n"

    def test_main_version_uses_installed_metadata_as_fallback(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(doctor, "_project_version", lambda: None)
        monkeypatch.setattr(
            doctor.importlib.metadata,
            "version",
            lambda _: "installed-version",
        )
        run_version()
        assert capsys.readouterr().out == "installed-version\n"

    def test_main_version_rejects_missing_installed_metadata(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(doctor, "_project_version", lambda: None)
        monkeypatch.setattr(doctor.importlib.metadata, "version", lambda _: None)
        with pytest.raises(SystemExit, match="1"):
            main(["--version"])
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == "unable to determine agents-army version\n"

    def test_project_version_ignores_unreadable_metadata(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def unreadable(*args: object, **kwargs: object) -> object:
            raise OSError("metadata unavailable")

        monkeypatch.setattr(Path, "open", unreadable)
        assert doctor._project_version() is None

    def test_main_version_falls_back_for_invalid_utf8_metadata(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def invalid_metadata(*_args: object, **_kwargs: object) -> io.BytesIO:
            return io.BytesIO(b"\xff")

        monkeypatch.setattr(Path, "open", invalid_metadata)
        monkeypatch.setattr(
            doctor.importlib.metadata,
            "version",
            lambda _: "installed-version",
        )
        run_version()
        assert capsys.readouterr().out == "installed-version\n"

    def test_main_version_reports_unavailable_without_traceback(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(doctor, "_project_version", lambda: None)

        def missing(_: str) -> str:
            raise doctor.importlib.metadata.PackageNotFoundError

        monkeypatch.setattr(doctor.importlib.metadata, "version", missing)
        with pytest.raises(SystemExit, match="1"):
            main(["--version"])
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == "unable to determine agents-army version\n"
        assert "Traceback" not in captured.err

    def test_main_version_after_command_is_rejected(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("AGENTS_ARMY_STATE_FILE", str(tmp_path / "s.json"))
        monkeypatch.setattr(core, "DEFAULT_BACKEND", "echo")
        with pytest.raises(SystemExit) as excinfo:
            main(["talk", "agent", "a", "--version"])
        assert excinfo.value.code == 2
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "usage: orchestrator talk" in captured.err
        assert "unrecognized arguments: a --version" in captured.err

    @pytest.mark.parametrize("flags", [[], ["-v"], ["--verbose"], ["-vv"], ["-vvv"]])
    def test_main_dependency_check_is_clean_and_early(
        self,
        flags: list[str],
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        def fail_orchestrator() -> None:
            raise AssertionError("dependency check must not construct Orchestrator")

        monkeypatch.setattr(core, "Orchestrator", fail_orchestrator)
        _fake_dependency_env(monkeypatch, ALL_TOOLS_PRESENT)
        main([*flags, "doctor"])
        captured = capsys.readouterr()
        assert captured.out == (
            f"\u2713 Python {_running_python()}\n"
            "\u2713 uv 0.4.18\n"
            "\u2713 claude 2.1.234 (Claude Code)\n"
            "\u2713 codex-cli 0.147.0\n"
            "\u2713 grok 1.0.5\n"
            "\u2713 opencode 1.18.21\n"
            "\u25cb jq 1.7 (optional)\n"
        )
        assert captured.err == ""

    def test_dependency_check_probes_each_tool_with_its_own_version_flag(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        calls = _fake_dependency_env(monkeypatch, ALL_TOOLS_PRESENT)
        main(["doctor"])
        assert calls == [
            ["uv", "--version"],
            ["claude", "--version"],
            ["codex", "--version"],
            ["grok", "--version"],
            ["opencode", "--version"],
            ["jq", "--version"],
        ]
        assert capsys.readouterr().err == ""

    def test_dependency_check_reports_every_missing_agent_cli_and_still_exits_zero(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Zero usable backends is a report, not a failure."""
        _fake_dependency_env(monkeypatch, {"uv": "uv 0.4.18", "jq": "jq-1.7"})
        main(["doctor"])
        captured = capsys.readouterr()
        assert captured.out == (
            f"\u2713 Python {_running_python()}\n"
            "\u2713 uv 0.4.18\n"
            "\u2717 claude (not found)\n"
            "\u2717 codex (not found)\n"
            "\u2717 grok (not found)\n"
            "\u2717 opencode (not found)\n"
            "\u25cb jq 1.7 (optional)\n"
        )
        assert captured.err == ""

    def test_dependency_check_exits_zero_with_nothing_installed(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _fake_dependency_env(monkeypatch, {})
        main(["doctor"])
        captured = capsys.readouterr()
        assert captured.out == (
            f"\u2713 Python {_running_python()}\n"
            "\u2717 uv (not found)\n"
            "\u2717 claude (not found)\n"
            "\u2717 codex (not found)\n"
            "\u2717 grok (not found)\n"
            "\u2717 opencode (not found)\n"
            "\u2717 jq (not found, optional)\n"
        )
        assert captured.err == ""

    def test_dependency_check_marks_absent_jq_optional_and_leaves_the_rest_alone(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        present = {
            tool: version for tool, version in ALL_TOOLS_PRESENT.items() if tool != "jq"
        }
        _fake_dependency_env(monkeypatch, present)
        main(["doctor"])
        captured = capsys.readouterr()
        assert captured.out == (
            f"\u2713 Python {_running_python()}\n"
            "\u2713 uv 0.4.18\n"
            "\u2713 claude 2.1.234 (Claude Code)\n"
            "\u2713 codex-cli 0.147.0\n"
            "\u2713 grok 1.0.5\n"
            "\u2713 opencode 1.18.21\n"
            "\u2717 jq (not found, optional)\n"
        )
        assert captured.err == ""

    def test_dependency_check_keeps_a_silent_tool_on_its_own_line(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Empty stdout means installed, version unknown — not missing."""
        _fake_dependency_env(monkeypatch, {"uv": "", "jq": "\n"})
        main(["doctor"])
        captured = capsys.readouterr()
        assert captured.out == (
            f"\u2713 Python {_running_python()}\n"
            "\u2713 uv (version unknown)\n"
            "\u2717 claude (not found)\n"
            "\u2717 codex (not found)\n"
            "\u2717 grok (not found)\n"
            "\u2717 opencode (not found)\n"
            "\u25cb jq (version unknown, optional)\n"
        )
        assert captured.err == ""

    def test_dependency_check_reads_only_the_first_line_of_a_version(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _fake_dependency_env(monkeypatch, {"uv": "  uv 0.4.18  \nbuild deadbeef\n"})
        main(["doctor"])
        assert "\u2713 uv 0.4.18\n" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "failure",
        [
            OSError("no such file"),
            subprocess.TimeoutExpired(cmd=["uv", "--version"], timeout=5),
            subprocess.SubprocessError("spawn failed"),
        ],
    )
    def test_dependency_check_survives_a_failing_version_probe(
        self,
        failure: Exception,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(doctor.shutil, "which", lambda tool: f"/usr/bin/{tool}")

        def explode(args, **kwargs):
            raise failure

        monkeypatch.setattr(doctor.subprocess, "run", explode)
        main(["doctor"])
        captured = capsys.readouterr()
        assert captured.out == (
            f"\u2713 Python {_running_python()}\n"
            "\u2713 uv (version unknown)\n"
            "\u2713 claude (version unknown)\n"
            "\u2713 codex (version unknown)\n"
            "\u2713 grok (version unknown)\n"
            "\u2713 opencode (version unknown)\n"
            "\u25cb jq (version unknown, optional)\n"
        )
        assert captured.err == ""

    def test_dependency_check_ignores_a_version_probe_that_exits_nonzero(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(doctor.shutil, "which", lambda _tool: "/usr/bin/uv")

        def refuse(args, **kwargs):
            return subprocess.CompletedProcess(args, 1, stdout="uv 0.4.18", stderr="")

        monkeypatch.setattr(doctor.subprocess, "run", refuse)
        main(["doctor"])
        captured = capsys.readouterr()
        assert "\u2713 uv (version unknown)\n" in captured.out
        assert "0.4.18" not in captured.out
        assert captured.err == ""

    def test_dependency_check_reports_an_interpreter_below_the_floor(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _fake_dependency_env(monkeypatch, {})
        monkeypatch.setattr(doctor.sys, "version_info", (3, 10, 2, "final", 0))
        main(["doctor"])
        assert capsys.readouterr().out.startswith(
            "\u2717 Python 3.10.2 (needs 3.11+)\n"
        )

    def test_dependency_check_accepts_an_interpreter_exactly_at_the_floor(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """3.11.0 is the floor, not the first version above it."""
        _fake_dependency_env(monkeypatch, {})
        monkeypatch.setattr(doctor.sys, "version_info", (3, 11, 0, "final", 0))
        main(["doctor"])
        assert capsys.readouterr().out.startswith("\u2713 Python 3.11.0\n")

    def test_dependency_check_reports_a_later_major_as_satisfying_the_floor(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _fake_dependency_env(monkeypatch, {})
        monkeypatch.setattr(doctor.sys, "version_info", (4, 0, 1, "final", 0))
        main(["doctor"])
        assert capsys.readouterr().out.startswith("\u2713 Python 4.0.1\n")

    def test_only_a_space_or_hyphen_separates_a_name_from_its_version(self) -> None:
        assert doctor.NAME_SEPARATORS == (" ", "-")

    def test_dependency_floor_matches_the_packaged_requires_python(self) -> None:
        required = tomllib.loads(
            Path(__file__)
            .resolve()
            .parents[1]
            .joinpath("pyproject.toml")
            .read_text(encoding="utf-8")
        )["project"]["requires-python"]
        parsed = re.fullmatch(r">=(\d+)\.(\d+)", required)
        assert parsed is not None, required
        assert tuple(int(part) for part in parsed.groups()) == doctor.MIN_PYTHON

    def test_only_jq_is_optional(self) -> None:
        assert doctor.DEPENDENCY_TOOLS == (
            ("uv", False),
            ("claude", False),
            ("codex", False),
            ("grok", False),
            ("opencode", False),
            ("jq", True),
        )

    @pytest.mark.parametrize(
        ("tool", "reported", "expected"),
        [
            ("uv", "uv 0.4.18", "uv 0.4.18"),
            ("jq", "jq-1.7", "jq 1.7"),
            ("claude", "2.1.234 (Claude Code)", "claude 2.1.234 (Claude Code)"),
            ("codex", "codex-cli 0.147.0", "codex-cli 0.147.0"),
            ("jq", "jq", "jq"),
            ("grok", "unreleased build", "grok unreleased build"),
            ("uv", "uv  0.4.18", "uv  0.4.18"),
            ("jq", "jqX1.7", "jqX1.7"),
        ],
    )
    def test_describe_version_names_each_tool_exactly_once(
        self, tool: str, reported: str, expected: str
    ) -> None:
        assert doctor._describe_version(tool, reported) == expected

    def test_main_dependency_check_after_command_remains_prompt_data(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("AGENTS_ARMY_STATE_FILE", str(tmp_path / "s.json"))
        monkeypatch.setattr(core, "DEFAULT_BACKEND", "echo")
        with pytest.raises(SystemExit, match="2"):
            main(["talk", "agent", "a", "--dependency-check"])
        assert capsys.readouterr().out == ""

    def test_main_reads_sys_argv_when_none_given(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr("sys.argv", ["orchestrator", "bogus"])
        with pytest.raises(SystemExit, match="2"):
            main()
        assert "usage: orchestrator " in capsys.readouterr().err

    def test_main_unknown_command_exits(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit, match="2"):
            main(["bogus"])
        assert "usage: orchestrator " in capsys.readouterr().err

    @pytest.mark.parametrize(
        ("argv", "usage"),
        [
            (["talk", "a", "hi", "--timeout", "5"], "usage: orchestrator talk "),
            (["doctor", "ignored"], "usage: orchestrator doctor "),
        ],
    )
    def test_trailing_argument_errors_use_the_selected_verb_usage(
        self,
        argv: list[str],
        usage: str,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            core,
            "Orchestrator",
            lambda *_: (_ for _ in ()).throw(AssertionError("constructed")),
        )
        with pytest.raises(SystemExit, match="2"):
            main(argv)
        assert capsys.readouterr().err.startswith(usage)

    @pytest.mark.parametrize("removed", ["spawn", "ensure"])
    def test_removed_lifecycle_commands_are_not_dispatchable(
        self, removed: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit, match="2"):
            main([removed])
        captured = capsys.readouterr()
        assert "unrecognized" not in captured.err
        assert "invalid choice" in captured.err

    def test_main_dispatches_using_sys_argv_when_none_given(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr("sys.argv", ["orchestrator", "create", "a", "-b", "echo"])
        monkeypatch.setattr(
            core,
            "Orchestrator",
            lambda *_: Orchestrator(
                runtime_paths(tmp_path, state_file=tmp_path / "s.json")
            ),
        )
        main()
        assert "created agent 'a' backend=echo" in capsys.readouterr().out

    def test_main_dispatches_to_command(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("AGENTS_ARMY_STATE_FILE", str(tmp_path / "s.json"))
        monkeypatch.setattr(
            core,
            "Orchestrator",
            lambda *_: Orchestrator(
                runtime_paths(tmp_path, state_file=tmp_path / "s.json")
            ),
        )
        main(["create", "a", "-b", "echo"])
        assert "created agent 'a' backend=echo" in capsys.readouterr().out

    def test_main_talk_creates_a_missing_agent(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """`talk <new-name>` creates it and runs the turn, not an error."""
        monkeypatch.setenv("AGENTS_ARMY_STATE_FILE", str(tmp_path / "s.json"))
        monkeypatch.setattr(core, "DEFAULT_BACKEND", "echo")
        main(["talk", "nope", "-p", "hi"])
        captured = capsys.readouterr()
        assert captured.err == "created agent 'nope' backend=echo\n"
        assert "echo:hi" in captured.out

    def test_main_duplicate_create_is_one_line(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("AGENTS_ARMY_STATE_FILE", str(tmp_path / "s.json"))
        main(["create", "a", "-b", "echo"])
        capsys.readouterr()
        with pytest.raises(SystemExit, match="1"):
            main(["create", "a", "-b", "echo"])
        captured = capsys.readouterr()
        assert captured.err == "agent 'a' already exists\n"
        assert "Traceback" not in captured.err

    def test_main_unknown_delete_is_one_line(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("AGENTS_ARMY_STATE_FILE", str(tmp_path / "s.json"))
        with pytest.raises(SystemExit, match="1"):
            main(["delete", "nope"])
        assert capsys.readouterr().err == "no agent named 'nope'\n"

    @pytest.mark.parametrize("flag", ["-h", "--help"])
    def test_main_help_describes_the_whole_cli(
        self, flag: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Routing every dash token to the skill parser hid the commands."""
        with pytest.raises(SystemExit) as excinfo:
            main([flag])
        assert excinfo.value.code == 0
        captured = capsys.readouterr()
        assert "usage: orchestrator " in captured.out
        with pytest.raises(SystemExit) as talk_help:
            main(["talk", "-h"])
        assert talk_help.value.code == 0
        assert "--skill" in capsys.readouterr().out
        assert captured.err == ""

    def test_main_lets_an_internal_error_surface(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A bug inside a backend must not print as a one-word usage error."""

        class BuggyBackend(AgentBackend):
            @property
            def name(self) -> str:
                return "buggy"

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
                raise KeyError("some internal dict key")

        register_backend("buggy", BuggyBackend)
        monkeypatch.setenv("AGENTS_ARMY_STATE_FILE", str(tmp_path / "s.json"))
        main(["create", "b", "-b", "buggy"])
        capsys.readouterr()
        with pytest.raises(KeyError, match="some internal dict key"):
            main(["talk", "b", "-p", "hi"])

    def test_main_corrupt_state_is_one_line(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        state = tmp_path / "s.json"
        state.write_text("{", encoding="utf-8")
        monkeypatch.setenv("AGENTS_ARMY_STATE_FILE", str(state))
        with pytest.raises(SystemExit, match="1"):
            main(["list"])
        err = capsys.readouterr().err
        assert err.endswith("\n")
        assert "Traceback" not in err
        assert err.count("\n") == 1


class TestVerboseFlag:
    @pytest.fixture(autouse=True)
    def isolated_orchestrator(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            core,
            "Orchestrator",
            lambda *_: Orchestrator(
                runtime_paths(tmp_path, state_file=tmp_path / "s.json")
            ),
        )
        for name in ("orchestrator", "backends", "third_party"):
            logging.getLogger(name).setLevel(logging.NOTSET)

    @pytest.mark.parametrize(
        ("flag", "expected"),
        [
            ("-v", logging.DEBUG),
            ("--verbose", logging.DEBUG),
            ("-vv", core.TRACE),
            ("-vvv", core.TRACE),
        ],
    )
    def test_each_flag_selects_its_level(self, flag: str, expected: int) -> None:
        main([flag, "list"])
        assert logging.getLogger("orchestrator").level == expected
        assert logging.getLogger("backends").level == expected

    @pytest.mark.parametrize("flag", ["-v", "--verbose", "-vv", "-vvv"])
    def test_no_flag_leaks_third_party_debug_output(self, flag: str) -> None:
        """A dependency's debug output would bury the signal being asked for."""
        main([flag, "list"])
        assert logging.getLogger("third_party").getEffectiveLevel() > logging.DEBUG

    @pytest.mark.parametrize("flag", ["-v", "--verbose", "-vv", "-vvv"])
    def test_flag_is_not_treated_as_a_command(self, flag: str, capsys) -> None:
        main([flag, "list"])
        out = capsys.readouterr().out
        assert out.startswith("registry: ")
        assert out.endswith("no agents\n")

    def test_the_loudest_flag_given_wins(self) -> None:
        main(["-v", "-vv", "list"])
        assert logging.getLogger("orchestrator").level == core.TRACE

    def test_verbosity_counts_before_and_after_the_verb(self) -> None:
        main(["-v", "list", "-v"])
        assert logging.getLogger("orchestrator").level == core.TRACE
        assert logging.getLogger("backends").level == core.TRACE

    def test_without_a_flag_our_loggers_stay_quiet(self) -> None:
        main(["list"])
        assert logging.getLogger("orchestrator").getEffectiveLevel() > logging.DEBUG

    def test_no_arguments_at_all_stays_quiet(self, capsys) -> None:
        """The empty argv path still picks a verbosity before printing usage."""
        with pytest.raises(SystemExit, match="2"):
            main([])
        assert logging.getLogger("orchestrator").getEffectiveLevel() > logging.DEBUG

    def test_trace_sits_below_debug(self) -> None:
        """Above DEBUG it would fire at -v, dumping transcripts unasked."""
        assert core.TRACE < logging.DEBUG
        assert logging.getLevelName(core.TRACE) == "TRACE"

    def test_prompt_token_equal_to_a_verbose_flag_is_kept(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "s.json"))
        orch.spawn("a", "echo")
        monkeypatch.setattr(core, "Orchestrator", lambda *_: orch)
        main(["talk", "a", "--", "add", "-v", "please"])
        assert "echo:add -v please" in capsys.readouterr().out
        assert logging.getLogger("orchestrator").getEffectiveLevel() > logging.DEBUG

    def test_leading_flag_does_not_strip_the_same_token_from_the_prompt(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "s.json"))
        orch.spawn("a", "echo")
        monkeypatch.setattr(core, "Orchestrator", lambda *_: orch)
        main(["-v", "talk", "a", "--", "add", "-v", "please"])
        assert "echo:add -v please" in capsys.readouterr().out
        assert logging.getLogger("orchestrator").level == logging.DEBUG


class TestStepLogging:
    def test_state_load_and_write_are_logged(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        state_file = tmp_path / "s.json"
        with caplog.at_level("DEBUG", logger="orchestrator"):
            orch = Orchestrator(runtime_paths(tmp_path, state_file=state_file))
            orch.spawn("a", "echo")
        messages = _messages(caplog)
        assert f"state: loaded 0 agent(s) from {state_file}" in messages
        assert f"state: wrote 1 agent(s) to {state_file}" in messages

    def test_turn_start_and_duration_are_logged(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "s.json"))
        orch.spawn("a", "echo")
        with caplog.at_level("DEBUG", logger="orchestrator"):
            orch.talk("a", "hi")
        messages = _messages(caplog)
        assert "agent 'a' (echo): starting turn, resume=False fork=False" in messages
        durations = [
            _reported_seconds(m, r"agent 'a': turn finished in (\d+\.\d)s")
            for m in messages
            if m.startswith("agent 'a': turn finished")
        ]
        assert len(durations) == 1
        assert durations[0] < 60

    def test_prompt_and_reply_are_logged_in_full_at_trace(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "s.json"))
        orch.spawn("a", "echo")
        prompt = "line one\nline two"
        with caplog.at_level(core.TRACE, logger="orchestrator"):
            orch.talk("a", prompt)
        messages = _messages(caplog)
        # Multi-line prompts must survive intact; this is the level you turn on
        # precisely to read exactly what went in and came back.
        assert f"agent 'a' prompt in:\n{prompt}" in messages
        assert f"agent 'a' reply out:\necho:{prompt}" in messages

    def test_transcripts_stay_out_of_the_step_level_view(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """-v is the level meant to be readable and safe to paste."""
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "s.json"))
        orch.spawn("a", "echo")
        with caplog.at_level(logging.DEBUG, logger="orchestrator"):
            orch.talk("a", "secret prompt")
        assert not any("secret prompt" in message for message in _messages(caplog))

    def test_a_resumed_turn_is_logged_as_such(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """resume= must reflect the session, not be hardcoded by either turn."""
        orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "s.json"))
        orch.spawn("a", "echo")
        orch.talk("a", "first")
        with caplog.at_level("DEBUG", logger="orchestrator"):
            orch.talk("a", "second")
        assert "agent 'a' (echo): starting turn, resume=True fork=False" in _messages(
            caplog
        )

    def test_cli_argument_shape_and_dispatch_are_logged(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(
            core,
            "Orchestrator",
            lambda *_: Orchestrator(
                runtime_paths(tmp_path, state_file=tmp_path / "s.json")
            ),
        )
        with caplog.at_level("DEBUG", logger="orchestrator"):
            main(["-v", "create", "a", "-b", "echo"])
        messages = _messages(caplog)
        # Four arguments remain once -v is stripped.
        assert "cli: 5 argument(s) after flag splitting" in messages
        assert "cli: dispatching 'create'" in messages


class TestLoggingConfiguration:
    """basicConfig() only acts on a root logger that has no handlers yet.

    pytest installs its own capture handler around every test phase, so the
    handlers have to be cleared inside the test body — clearing them in a
    fixture is undone before the body runs, and every assertion here would
    then be measuring pytest's logging setup rather than this project's.
    """

    @pytest.fixture(autouse=True)
    def restore_root(self) -> Iterator[None]:
        root = logging.getLogger()
        handlers, level = root.handlers[:], root.level
        yield
        root.handlers[:] = handlers
        root.setLevel(level)

    def test_root_is_pinned_to_warning(self) -> None:
        root = logging.getLogger()
        root.handlers.clear()
        # Start from the level the call must overwrite: WARNING is also the
        # default, so starting there would pass without the level ever applying.
        root.setLevel(logging.DEBUG)

        cli._configure_logging(False)

        assert root.level == logging.WARNING

    def test_records_carry_time_level_and_logger_name(self) -> None:
        root = logging.getLogger()
        root.handlers.clear()

        cli._configure_logging(False)

        record = logging.LogRecord(
            "backends.claude", logging.WARNING, "p", 1, "msg", None, None
        )
        assert re.fullmatch(
            r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} "
            r"WARNING backends\.claude: msg",
            root.handlers[0].format(record),
        )


def test_parser_exposes_the_complete_new_surface() -> None:
    parser = cli._build_parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )

    assert set(subparsers.choices) == {
        "create",
        "talk",
        "chat",
        "fork",
        "list",
        "delete",
        "doctor",
    }
    child_args = {
        "create": ["a"],
        "talk": ["a", "-p", "x"],
        "chat": ["a"],
        "fork": ["a", "b"],
        "list": [],
        "delete": ["a"],
        "doctor": [],
    }
    for verb in subparsers.choices:
        child = subparsers.choices[verb]
        assert child.prog == f"orchestrator {verb}"
        assert child.get_default("_parser") is child
        assert child.parse_args(child_args[verb]).verbosity_after == 0

    talk = subparsers.choices["talk"]
    assert (
        "prompt source: orchestrator talk NAME "
        "[-p TEXT | --prompt-file PATH | -- PROMPT...]" in talk.epilog
    )
    assert (
        talk.parse_args(
            [
                "a",
                "-s",
                "review",
                "--schema",
                "out.json",
                "--retries",
                "3",
                "--timeout",
                "7",
                "-p",
                "x",
            ]
        ).skill
        == "review"
    )

    for action in (parser._actions[1],):
        assert action.option_strings == ["--version"]
        assert action.nargs == 0
        assert action.default is argparse.SUPPRESS
        assert action.help == "show the installed version"

    verbosity = next(action for action in parser._actions if action.dest == "verbosity")
    assert verbosity.option_strings == ["-v", "--verbose"]
    assert verbosity.default == 0
    assert (
        verbosity.help == "log each step and how long it took; repeat for full prompts"
    )


def test_agent_config_short_options_are_forwarded_by_create_and_talk() -> None:
    parser = cli._build_parser()
    create = parser.parse_args(["create", "a", "-m", "model", "-e", "high"])
    talk = parser.parse_args(["talk", "a", "-m", "model", "-e", "high", "-p", "x"])
    assert (create.model, create.reasoning_effort) == ("model", "high")
    assert (talk.model, talk.reasoning_effort) == ("model", "high")


def test_version_fallback_uses_the_distribution_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(doctor, "_project_version", lambda: None)
    seen: list[str] = []

    def version(package: str) -> str:
        seen.append(package)
        return "9.8.7"

    monkeypatch.setattr(doctor.importlib.metadata, "version", version)
    assert doctor._resolve_version() == "9.8.7"
    assert seen == ["agents-army"]


def test_version_fallback_rejects_an_empty_metadata_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(doctor, "_project_version", lambda: None)
    monkeypatch.setattr(doctor.importlib.metadata, "version", lambda _: "")
    with pytest.raises(ValueError, match=r"^$"):
        doctor._resolve_version()


def test_project_version_rejects_a_non_string_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        doctor.tomllib,
        "load",
        lambda _: {"project": {"version": 123}},
    )
    monkeypatch.setattr(Path, "open", lambda *_: io.BytesIO(b"project = {}"))
    assert doctor._project_version() is None
    monkeypatch.setattr(doctor.tomllib, "load", lambda _: {})
    assert doctor._project_version() is None


def test_prompt_errors_have_exact_messages_and_strip_tail(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = cli._build_parser()
    options = parser.parse_args(["talk", "a"])
    with pytest.raises(SystemExit) as missing:
        cli._resolve_talk_prompt(options, [], False)
    assert missing.value.code == 2
    assert (
        "orchestrator talk: error: talk requires exactly one prompt source"
        in capsys.readouterr().err
    )

    options = parser.parse_args(["talk", "a", "-p", " "])
    with pytest.raises(SystemExit) as empty:
        cli._resolve_talk_prompt(options, [], False)
    assert empty.value.code == 2
    assert (
        "orchestrator talk: error: talk prompt must not be empty"
        in capsys.readouterr().err
    )

    options = parser.parse_args(["talk", "a"])
    cli._resolve_talk_prompt(options, ["  one", "two  "], True)
    assert options.prompt == "one two"

    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("  one\n two  \n", encoding="utf-8")
    options = parser.parse_args(["talk", "a", "--prompt-file", str(prompt_file)])
    cli._resolve_talk_prompt(options, [], False)
    assert options.prompt == "one\n two"


@pytest.mark.parametrize("kind", ["missing", "directory", "binary", "empty"])
def test_prompt_file_errors_are_exact_and_prevent_side_effects(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prompt_file = tmp_path / "prompt.txt"
    if kind == "directory":
        prompt_file.mkdir()
    elif kind == "binary":
        prompt_file.write_bytes(b"\xff\xfe")
    elif kind == "empty":
        prompt_file.write_text(" \t\n", encoding="utf-8")

    monkeypatch.setattr(
        core,
        "Orchestrator",
        lambda *_: (_ for _ in ()).throw(AssertionError("constructed")),
    )
    with pytest.raises(SystemExit) as excinfo:
        main(["talk", "a", "--prompt-file", str(prompt_file)])
    assert excinfo.value.code == 2
    error = capsys.readouterr().err
    if kind == "empty":
        assert "orchestrator talk: error: talk prompt must not be empty" in error
    else:
        try:
            prompt_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            expected = f"cannot read prompt file {prompt_file.resolve()}: {exc}"
        else:
            raise AssertionError("expected prompt file read to fail")
        assert expected in error


def test_prompt_file_errors_have_exact_messages(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    parser = cli._build_parser()

    missing_path = tmp_path / "missing.txt"
    options = parser.parse_args(["talk", "a", "--prompt-file", str(missing_path)])
    with pytest.raises(SystemExit) as missing:
        cli._resolve_talk_prompt(options, [], False)
    assert missing.value.code == 2
    resolved_missing = missing_path.resolve()
    assert (
        f"orchestrator talk: error: cannot read prompt file {resolved_missing}: "
        in capsys.readouterr().err
    )

    directory_path = tmp_path / "adir"
    directory_path.mkdir()
    options = parser.parse_args(["talk", "a", "--prompt-file", str(directory_path)])
    with pytest.raises(SystemExit) as is_dir:
        cli._resolve_talk_prompt(options, [], False)
    assert is_dir.value.code == 2
    resolved_dir = directory_path.resolve()
    assert (
        f"orchestrator talk: error: cannot read prompt file {resolved_dir}: "
        in capsys.readouterr().err
    )

    binary_path = tmp_path / "binary.txt"
    binary_path.write_bytes(b"\xff\xfe\x00not utf-8")
    options = parser.parse_args(["talk", "a", "--prompt-file", str(binary_path)])
    with pytest.raises(SystemExit) as bad_decode:
        cli._resolve_talk_prompt(options, [], False)
    assert bad_decode.value.code == 2
    resolved_binary = binary_path.resolve()
    assert (
        f"orchestrator talk: error: cannot read prompt file {resolved_binary}: "
        in capsys.readouterr().err
    )

    blank_path = tmp_path / "blank.txt"
    blank_path.write_text(" \n\t \n", encoding="utf-8")
    options = parser.parse_args(["talk", "a", "--prompt-file", str(blank_path)])
    with pytest.raises(SystemExit) as blank:
        cli._resolve_talk_prompt(options, [], False)
    assert blank.value.code == 2
    assert (
        "orchestrator talk: error: talk prompt must not be empty"
        in capsys.readouterr().err
    )


def test_prompt_file_strips_outer_whitespace_and_keeps_interior_newlines(
    tmp_path: Path,
) -> None:
    parser = cli._build_parser()
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("  line one\nline two  \n", encoding="utf-8")
    options = parser.parse_args(["talk", "a", "--prompt-file", str(prompt_path)])
    cli._resolve_talk_prompt(options, [], False)
    assert options.prompt == "line one\nline two"


def test_separator_error_and_cli_error_without_arguments_are_exact(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as separator:
        main(["create", "a", "--", "extra"])
    assert separator.value.code == 2
    assert (
        "orchestrator create: error: the -- separator is only valid for talk"
        in capsys.readouterr().err
    )

    monkeypatch.setattr(
        core,
        "Orchestrator",
        lambda *_: (_ for _ in ()).throw(core.OrchestratorError()),
    )
    with pytest.raises(SystemExit) as empty_error:
        main(["list"])
    assert empty_error.value.code == 1
    assert capsys.readouterr().err == "\n"


def test_cli_log_counts_head_and_tail_arguments(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setitem(cli.VERBS, "talk", lambda _orch, _opts: None)
    monkeypatch.setattr(core, "Orchestrator", lambda *_: object())
    caplog.set_level(logging.DEBUG, logger="orchestrator")
    main(["-v", "talk", "a", "--", "one", "two"])
    assert "cli: 5 argument(s) after flag splitting" in caplog.text
