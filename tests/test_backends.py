"""Unit tests for agent backend interface, implementations, and orchestrator."""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import subprocess
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path

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
    Orchestrator,
)
from tests.backend_helpers import (
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
