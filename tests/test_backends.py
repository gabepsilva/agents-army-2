"""Unit tests for agent backend interface, implementations, and orchestrator."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from backends.base import AgentBackend, TurnResult
from backends.claude import ClaudeBackend, ClaudeTurnError
from backends.codex import CodexBackend, CodexTurnError
from backends.registry import get_backend, list_backends, register_backend
from orchestrator import Orchestrator


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

            def run_turn(self, prompt: str, session_id: str | None, cwd: Path) -> TurnResult:
                return TurnResult(session_id="custom-sid", reply=prompt, raw="")

        register_backend("custom", CustomBackend)
        assert "custom" in list_backends()

        backend = get_backend("custom")
        assert isinstance(backend, CustomBackend)
        assert backend.run_turn("hi", None, tmp_path).reply == "hi"

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend 'nonexistent'"):
            get_backend("nonexistent")


class TestClaudeRunTurn:
    def test_new_turn_parses_session_and_result(self, tmp_path: Path, monkeypatch) -> None:
        backend = ClaudeBackend()

        def fake_run(args, **kwargs):
            assert args == ["claude", "--print", "--output-format", "json", "-p", "hello"]
            payload = json.dumps({"is_error": False, "session_id": "s1", "result": "hi"})
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = backend.run_turn("hello", None, tmp_path)
        assert result.session_id == "s1"
        assert result.reply == "hi"

    def test_resume_turn_passes_resume_flag(self, tmp_path: Path, monkeypatch) -> None:
        backend = ClaudeBackend()

        def fake_run(args, **kwargs):
            assert args == ["claude", "--print", "--output-format", "json", "--resume", "s1", "-p", "again"]
            payload = json.dumps({"is_error": False, "session_id": "s1", "result": "still here"})
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = backend.run_turn("again", "s1", tmp_path)
        assert result.session_id == "s1"
        assert result.reply == "still here"

    def test_error_reply_raises(self, tmp_path: Path, monkeypatch) -> None:
        backend = ClaudeBackend()

        def fake_run(args, **kwargs):
            payload = json.dumps({"is_error": True, "result": "boom"})
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(ClaudeTurnError, match="boom"):
            backend.run_turn("x", None, tmp_path)

    def test_nonzero_exit_raises(self, tmp_path: Path, monkeypatch) -> None:
        backend = ClaudeBackend()

        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="bad")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(ClaudeTurnError, match="exited 1"):
            backend.run_turn("x", None, tmp_path)


class TestCodexRunTurn:
    def test_new_turn_parses_thread_and_reply(self, tmp_path: Path, monkeypatch) -> None:
        backend = CodexBackend()

        def fake_run(args, **kwargs):
            assert args == ["codex", "exec", "hello", "--json", "--skip-git-repo-check"]
            stdout = (
                '{"type":"thread.started","thread_id":"t1"}\n'
                '{"type":"item.completed","item":{"type":"agent_message","text":"yo"}}\n'
            )
            return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = backend.run_turn("hello", None, tmp_path)
        assert result.session_id == "t1"
        assert result.reply == "yo"

    def test_resume_turn_uses_thread_id(self, tmp_path: Path, monkeypatch) -> None:
        backend = CodexBackend()

        def fake_run(args, **kwargs):
            assert args == ["codex", "exec", "resume", "t1", "again", "--json", "--skip-git-repo-check"]
            stdout = (
                '{"type":"thread.started","thread_id":"t1"}\n'
                '{"type":"item.completed","item":{"type":"agent_message","text":"back"}}\n'
            )
            return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = backend.run_turn("again", "t1", tmp_path)
        assert result.session_id == "t1"
        assert result.reply == "back"

    def test_no_thread_id_raises(self, tmp_path: Path, monkeypatch) -> None:
        backend = CodexBackend()

        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(CodexTurnError, match="thread_id"):
            backend.run_turn("x", None, tmp_path)


class TestOrchestrator:
    def test_spawn_talk_persists_and_resumes(self, tmp_path: Path, monkeypatch) -> None:
        state_file = tmp_path / "state.json"

        def fake_backend_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout=json.dumps(
                {"is_error": False, "session_id": "persist-me", "result": "reply"}
            ), stderr="")

        monkeypatch.setattr(subprocess, "run", fake_backend_run)

        orch = Orchestrator(state_file=state_file)
        agent = orch.spawn("a1", "claude")
        result = orch.talk("a1", "first")
        assert result.reply == "reply"
        assert orch.agents["a1"].session_id == "persist-me"

        orch2 = Orchestrator(state_file=state_file)
        assert "a1" in orch2.agents
        assert orch2.agents["a1"].session_id == "persist-me"
        assert orch2.talk("a1", "second").reply == "reply"

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
            return subprocess.CompletedProcess(args, 0, stdout=json.dumps(
                {"is_error": False, "session_id": "s1", "result": "reply"}
            ), stderr="")

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
