"""Unit tests for agent backend interface, implementations, and orchestrator."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import orchestrator
from backends.base import AgentBackend, TurnResult
from backends.claude import ClaudeBackend, ClaudeTurnError
from backends.codex import CodexBackend, CodexTurnError
from backends.registry import get_backend, list_backends, register_backend
from orchestrator import (
    Orchestrator,
    cmd_delete,
    cmd_list,
    cmd_spawn,
    cmd_talk,
    main,
)


def _assert_subprocess_kwargs(kwargs: dict, cwd: Path) -> None:
    """Every backend must run its subprocess the same disciplined way."""
    assert kwargs["cwd"] == str(cwd)
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert kwargs["check"] is False
    assert kwargs["timeout"] == 1800


class TestAgentBackendInterface:
    def test_claude_name(self) -> None:
        assert ClaudeBackend().name == "claude"

    def test_codex_name(self) -> None:
        assert CodexBackend().name == "codex"

    def test_custom_backend_registration(self, tmp_path: Path) -> None:
        class CustomBackend(AgentBackend):
            @property
            def name(self) -> str:
                return "custom"

            def run_turn(
                self, prompt: str, session_id: str | None, cwd: Path
            ) -> TurnResult:
                return TurnResult(session_id="custom-sid", reply=prompt, raw="")

        register_backend("custom", CustomBackend)
        assert "custom" in list_backends()

        backend = get_backend("custom")
        assert isinstance(backend, CustomBackend)
        assert backend.run_turn("hi", None, tmp_path).reply == "hi"

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend") as excinfo:
            get_backend("nonexistent")
        assert str(excinfo.value) == (
            f"Unknown backend 'nonexistent'. Available backends: "
            f"{', '.join(list_backends())}"
        )


class TestClaudeRunTurn:
    def test_new_turn_parses_session_and_result(
        self, tmp_path: Path, monkeypatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        backend = ClaudeBackend()
        payload = json.dumps({"is_error": False, "session_id": "s1", "result": "hi"})

        def fake_run(args, **kwargs):
            assert args == [
                "claude",
                "--print",
                "--output-format",
                "json",
                "-p",
                "hello",
            ]
            _assert_subprocess_kwargs(kwargs, tmp_path)
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with caplog.at_level("DEBUG"):
            result = backend.run_turn("hello", None, tmp_path)
        assert result.session_id == "s1"
        assert result.reply == "hi"
        assert result.raw == payload
        assert caplog.records[-1].getMessage() == (
            f"claude turn: cwd={tmp_path} resume=False"
        )

    def test_resume_turn_passes_resume_flag(
        self, tmp_path: Path, monkeypatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        backend = ClaudeBackend()

        def fake_run(args, **kwargs):
            assert args == [
                "claude",
                "--print",
                "--output-format",
                "json",
                "--resume",
                "s1",
                "-p",
                "again",
            ]
            _assert_subprocess_kwargs(kwargs, tmp_path)
            payload = json.dumps(
                {"is_error": False, "session_id": "s1", "result": "still here"}
            )
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with caplog.at_level("DEBUG"):
            result = backend.run_turn("again", "s1", tmp_path)
        assert result.session_id == "s1"
        assert result.reply == "still here"
        assert caplog.records[-1].getMessage() == (
            f"claude turn: cwd={tmp_path} resume=True"
        )

    def test_error_reply_raises(self, tmp_path: Path, monkeypatch) -> None:
        backend = ClaudeBackend()

        def fake_run(args, **kwargs):
            _assert_subprocess_kwargs(kwargs, tmp_path)
            payload = json.dumps({"is_error": True, "result": "boom"})
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(ClaudeTurnError, match="boom"):
            backend.run_turn("x", None, tmp_path)

    def test_result_defaults_to_empty_reply(self, tmp_path: Path, monkeypatch) -> None:
        backend = ClaudeBackend()

        def fake_run(args, **kwargs):
            payload = json.dumps({"is_error": False, "session_id": "s1"})
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = backend.run_turn("x", None, tmp_path)
        assert result.reply == ""

    def test_nonzero_exit_raises(self, tmp_path: Path, monkeypatch) -> None:
        backend = ClaudeBackend()
        stderr = "s" * 500 + "M" + "e" * 1999  # 2500 chars total

        def fake_run(args, **kwargs):
            _assert_subprocess_kwargs(kwargs, tmp_path)
            return subprocess.CompletedProcess(args, 1, stdout="", stderr=stderr)

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(ClaudeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == f"claude exited 1\nstderr: {stderr[-2000:]}"

    def test_malformed_json_raises(self, tmp_path: Path, monkeypatch) -> None:
        backend = ClaudeBackend()
        stdout = "s" * 500 + "M" + "e" * 1999  # 2500 chars total, not valid JSON

        def fake_run(args, **kwargs):
            _assert_subprocess_kwargs(kwargs, tmp_path)
            return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(ClaudeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == (
            f"claude output was not JSON\nstdout: {stdout[-2000:]}"
        )


class TestCodexRunTurn:
    def test_new_turn_parses_thread_and_reply(
        self, tmp_path: Path, monkeypatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        backend = CodexBackend()

        def fake_run(args, **kwargs):
            assert args == ["codex", "exec", "hello", "--json", "--skip-git-repo-check"]
            _assert_subprocess_kwargs(kwargs, tmp_path)
            stdout = (
                '{"type":"thread.started","thread_id":"t1"}\n'
                '{"type":"item.completed","item":{"type":"agent_message","text":"yo"}}\n'
            )
            return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with caplog.at_level("DEBUG"):
            result = backend.run_turn("hello", None, tmp_path)
        assert result.session_id == "t1"
        assert result.reply == "yo"
        assert caplog.records[-1].getMessage() == (
            f"codex turn: cwd={tmp_path} resume=False"
        )

    def test_resume_turn_uses_thread_id(
        self, tmp_path: Path, monkeypatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        backend = CodexBackend()

        def fake_run(args, **kwargs):
            assert args == [
                "codex",
                "exec",
                "resume",
                "t1",
                "again",
                "--json",
                "--skip-git-repo-check",
            ]
            _assert_subprocess_kwargs(kwargs, tmp_path)
            stdout = (
                '{"type":"thread.started","thread_id":"t1"}\n'
                '{"type":"item.completed","item":{"type":"agent_message","text":"back"}}\n'
            )
            return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with caplog.at_level("DEBUG"):
            result = backend.run_turn("again", "t1", tmp_path)
        assert result.session_id == "t1"
        assert result.reply == "back"
        assert caplog.records[-1].getMessage() == (
            f"codex turn: cwd={tmp_path} resume=True"
        )

    def test_no_thread_id_raises(self, tmp_path: Path, monkeypatch) -> None:
        backend = CodexBackend()

        def fake_run(args, **kwargs):
            _assert_subprocess_kwargs(kwargs, tmp_path)
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(CodexTurnError, match="thread_id"):
            backend.run_turn("x", None, tmp_path)

    def test_no_thread_id_error_keeps_the_tail_of_long_output(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        backend = CodexBackend()
        stdout = "o" * 2500  # not JSON, so session_id stays None
        stderr = "e" * 2500

        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr=stderr)

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(CodexTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == (
            f"codex did not report a thread_id\nstdout: {stdout[-2000:]}"
            f"\nstderr: {stderr[-2000:]}"
        )

    def test_nonzero_exit_raises(self, tmp_path: Path, monkeypatch) -> None:
        backend = CodexBackend()
        stderr = "e" * 2500

        def fake_run(args, **kwargs):
            _assert_subprocess_kwargs(kwargs, tmp_path)
            return subprocess.CompletedProcess(args, 1, stdout="", stderr=stderr)

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(CodexTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == f"codex exited 1\nstderr: {stderr[-2000:]}"

    def test_parse_skips_blank_and_malformed_lines_and_other_events(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        backend = CodexBackend()
        stdout = "\n".join(
            [
                "",
                "not json",
                '{"type":"thread.started","thread_id":"t1"}',
                '{"type":"other.event"}',
                '{"type":"item.completed","item":{"type":"reasoning"}}',
                '{"type":"item.completed","item":{"type":"agent_message"}}',
                '{"type":"item.completed","item":{"type":"agent_message","text":"hi"}}',
                '{"type":"item.completed","item":{"type":"agent_message","text":"there"}}',
            ]
        )

        def fake_run(args, **kwargs):
            _assert_subprocess_kwargs(kwargs, tmp_path)
            return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = backend.run_turn("x", None, tmp_path)
        assert result.session_id == "t1"
        assert result.reply == "hi\nthere"
        assert result.raw == stdout


class TestOrchestrator:
    def test_spawn_talk_persists_and_resumes(self, tmp_path: Path, monkeypatch) -> None:
        state_file = tmp_path / "state.json"
        seen_session_ids: list[str | None] = []

        def fake_backend_run(args, **kwargs):
            _assert_subprocess_kwargs(kwargs, orchestrator.WORKDIR)
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

            def run_turn(
                self, prompt: str, session_id: str | None, cwd: Path
            ) -> TurnResult:
                seen_session_ids.append(session_id)
                return TurnResult(session_id="persist-me", reply="reply", raw="")

        register_backend("recording", RecordingBackend)
        monkeypatch.setattr(subprocess, "run", fake_backend_run)

        orch = Orchestrator(state_file=state_file)
        agent = orch.spawn("a1", "claude")
        assert agent.session_id is None

        orch2 = Orchestrator(state_file=state_file)
        agent2 = orch2.spawn("a2", "recording")
        assert agent2.name == "a2"
        orch2.talk("a2", "first")
        orch2.talk("a2", "second")
        assert seen_session_ids == [None, "persist-me"]

        result = orch.talk("a1", "first")
        assert result.reply == "reply"
        assert orch.agents["a1"].session_id == "persist-me"

        orch3 = Orchestrator(state_file=state_file)
        assert "a1" in orch3.agents
        assert orch3.agents["a1"].name == "a1"
        assert orch3.agents["a1"].session_id == "persist-me"
        assert orch3.talk("a1", "second").reply == "reply"

    def test_persist_writes_sorted_indented_json(self, tmp_path: Path) -> None:
        state_file = tmp_path / "state.json"
        orch = Orchestrator(state_file=state_file)
        orch.spawn("b", "codex")
        orch.spawn("a", "claude")
        assert state_file.read_text(encoding="utf-8") == (
            "{\n"
            '  "a": {\n'
            '    "backend": "claude",\n'
            '    "session_id": null\n'
            "  },\n"
            '  "b": {\n'
            '    "backend": "codex",\n'
            '    "session_id": null\n'
            "  }\n"
            "}\n"
        )

    def test_spawn_defaults_to_claude_backend(self, tmp_path: Path) -> None:
        orch = Orchestrator(state_file=tmp_path / "s.json")
        agent = orch.spawn("a1")
        assert agent.backend.name == "claude"

    def test_spawn_duplicate_raises(self, tmp_path: Path) -> None:
        orch = Orchestrator(state_file=tmp_path / "s.json")
        orch.spawn("a1", "claude")
        with pytest.raises(ValueError, match="already exists"):
            orch.spawn("a1", "claude")

    def test_talk_unknown_raises(self, tmp_path: Path) -> None:
        orch = Orchestrator(state_file=tmp_path / "s.json")
        with pytest.raises(KeyError, match="no agent named"):
            orch.talk("nope", "hi")

    def test_list_agents(self, tmp_path: Path) -> None:
        orch = Orchestrator(state_file=tmp_path / "s.json")
        orch.spawn("b", "codex")
        orch.spawn("a", "claude")
        assert orch.list_agents() == ["a", "b"]

    def test_delete_agent(self, tmp_path: Path, monkeypatch) -> None:
        state_file = tmp_path / "state.json"

        def fake_backend_run(args, **kwargs):
            _assert_subprocess_kwargs(kwargs, orchestrator.WORKDIR)
            return subprocess.CompletedProcess(
                args,
                0,
                stdout=json.dumps(
                    {"is_error": False, "session_id": "s1", "result": "reply"}
                ),
                stderr="",
            )

        monkeypatch.setattr(subprocess, "run", fake_backend_run)

        orch = Orchestrator(state_file=state_file)
        orch.spawn("a", "claude")
        orch.spawn("b", "codex")
        orch.delete("a")
        assert orch.list_agents() == ["b"]

        orch2 = Orchestrator(state_file=state_file)
        assert orch2.list_agents() == ["b"]

    def test_delete_unknown_raises(self, tmp_path: Path) -> None:
        orch = Orchestrator(state_file=tmp_path / "s.json")
        with pytest.raises(KeyError, match="no agent named"):
            orch.delete("nope")


class TestCLI:
    """cmd_* dispatch and printed output, backed by a fake CLI-free backend."""

    class _EchoBackend(AgentBackend):
        @property
        def name(self) -> str:
            return "echo"

        def run_turn(
            self, prompt: str, session_id: str | None, cwd: Path
        ) -> TurnResult:
            return TurnResult(session_id="echo-sid", reply=f"echo:{prompt}", raw="")

    @pytest.fixture(autouse=True)
    def _register_echo(self) -> None:
        register_backend("echo", self._EchoBackend)

    def test_cmd_spawn_prints_confirmation(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        orch = Orchestrator(state_file=tmp_path / "s.json")
        cmd_spawn(orch, ["a", "-b", "echo"])
        assert "spawned agent 'a' backend=echo" in capsys.readouterr().out

    def test_cmd_spawn_defaults_to_claude_backend(self, tmp_path: Path) -> None:
        orch = Orchestrator(state_file=tmp_path / "s.json")
        cmd_spawn(orch, ["a"])
        assert orch.agents["a"].backend.name == "claude"

    def test_cmd_spawn_rejects_unknown_backend(self, tmp_path: Path) -> None:
        orch = Orchestrator(state_file=tmp_path / "s.json")
        with pytest.raises(SystemExit, match="2"):
            cmd_spawn(orch, ["a", "-b", "not-a-backend"])

    def test_cmd_spawn_missing_name_reports_its_own_prog(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        orch = Orchestrator(state_file=tmp_path / "s.json")
        with pytest.raises(SystemExit, match="2"):
            cmd_spawn(orch, [])
        assert capsys.readouterr().err.startswith("usage: spawn ")

    def test_cmd_talk_prints_reply(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        orch = Orchestrator(state_file=tmp_path / "s.json")
        orch.spawn("a", "echo")
        cmd_talk(orch, ["a", "hello", "there"])
        out = capsys.readouterr().out
        assert "session=echo-sid" in out
        assert "echo:hello there" in out

    def test_cmd_talk_empty_prompt_warns(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        orch = Orchestrator(state_file=tmp_path / "s.json")
        orch.spawn("a", "echo")
        cmd_talk(orch, ["a", "   "])
        captured = capsys.readouterr()
        assert captured.err == "usage: talk <agent> <prompt>\n"
        assert captured.out == ""

    def test_cmd_talk_missing_name_reports_its_own_prog(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        orch = Orchestrator(state_file=tmp_path / "s.json")
        with pytest.raises(SystemExit, match="2"):
            cmd_talk(orch, [])
        assert capsys.readouterr().err.startswith("usage: talk ")

    def test_cmd_list_empty(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        orch = Orchestrator(state_file=tmp_path / "s.json")
        cmd_list(orch, [])
        assert capsys.readouterr().out == "no agents\n"

    def test_cmd_list_prints_each_agent(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        orch = Orchestrator(state_file=tmp_path / "s.json")
        orch.spawn("a", "echo")
        cmd_list(orch, [])
        out = capsys.readouterr().out
        assert "a" in out
        assert "backend=echo" in out
        assert "session=-" in out

    def test_cmd_list_rejects_unexpected_arg_reporting_its_own_prog(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        orch = Orchestrator(state_file=tmp_path / "s.json")
        with pytest.raises(SystemExit, match="2"):
            cmd_list(orch, ["unexpected"])
        assert capsys.readouterr().err.startswith("usage: list ")

    def test_cmd_delete_prints_confirmation(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        orch = Orchestrator(state_file=tmp_path / "s.json")
        orch.spawn("a", "echo")
        cmd_delete(orch, ["a"])
        assert "deleted agent 'a' backend=echo" in capsys.readouterr().out

    def test_cmd_delete_missing_name_reports_its_own_prog(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        orch = Orchestrator(state_file=tmp_path / "s.json")
        with pytest.raises(SystemExit, match="2"):
            cmd_delete(orch, [])
        assert capsys.readouterr().err.startswith("usage: delete ")

    def test_main_no_args_prints_usage_and_exits(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit, match="2"):
            main([])
        captured = capsys.readouterr()
        assert captured.err == (
            "usage: orchestrator <command> [args...]\n"
            "commands: spawn, talk, list, delete\n"
        )
        assert captured.out == ""

    def test_main_reads_sys_argv_when_none_given(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr("sys.argv", ["orchestrator", "bogus"])
        with pytest.raises(SystemExit, match="2"):
            main()
        assert "usage: orchestrator <command>" in capsys.readouterr().err

    def test_main_unknown_command_exits(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit, match="2"):
            main(["bogus"])
        assert "usage: orchestrator <command>" in capsys.readouterr().err

    def test_main_dispatches_using_sys_argv_when_none_given(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr("sys.argv", ["orchestrator", "spawn", "a", "-b", "echo"])
        monkeypatch.setattr(
            orchestrator,
            "Orchestrator",
            lambda: Orchestrator(state_file=tmp_path / "s.json"),
        )
        main()
        assert "spawned agent 'a' backend=echo" in capsys.readouterr().out

    def test_main_dispatches_to_command(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(orchestrator, "STATE_FILE", tmp_path / "s.json")
        monkeypatch.setattr(
            orchestrator,
            "Orchestrator",
            lambda: Orchestrator(state_file=tmp_path / "s.json"),
        )
        main(["spawn", "a", "-b", "echo"])
        assert "spawned agent 'a' backend=echo" in capsys.readouterr().out
