"""Unit tests for agent backend interface, implementations, and orchestrator."""

from __future__ import annotations

import fcntl
import json
import logging
import re
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

import orchestrator
from backends.base import AgentBackend, TurnResult, describe_command
from backends.claude import (
    OPT_IN_REQUIRED_REASON,
    PERMISSION_MODE,
    ClaudeBackend,
    ClaudeTurnError,
    _stdout_for_error,
    parse_claude_stdout,
)
from backends.codex import CodexBackend, CodexTurnError
from backends.registry import (
    UnknownBackendError,
    get_backend,
    list_backends,
    register_backend,
)
from orchestrator import (
    Orchestrator,
    cmd_delete,
    cmd_list,
    cmd_spawn,
    cmd_talk,
    main,
)


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

    def run_turn(self, prompt: str, session_id: str | None, cwd: Path) -> TurnResult:
        return TurnResult(session_id="echo-sid", reply=f"echo:{prompt}", raw="")


@pytest.fixture(autouse=True)
def register_echo_backend() -> None:
    """Registered for every test, not just the class that introduced it.

    The registry is module-level state, so a class relying on another class
    having registered it first passes or fails on test ordering — which xdist
    is free to change.
    """
    register_backend("echo", EchoBackend)


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
    def test_permission_mode_is_the_noninteractive_opt_in(self) -> None:
        assert PERMISSION_MODE == "bypassPermissions"

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
                "--permission-mode",
                PERMISSION_MODE,
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
        messages = _messages(caplog)
        assert messages[0] == (
            f"claude turn: cwd={tmp_path} resume=False prompt_chars=5 timeout=1800s"
        )
        assert messages[1] == (
            "claude turn: invoking "
            "claude --print --output-format json --permission-mode "
            f"{PERMISSION_MODE} -p <prompt:5chars>"
        )
        assert (
            _reported_seconds(
                messages[2],
                rf"claude turn: exited 0 after (\d+\.\d)s with {len(payload)} "
                rf"chars of stdout",
            )
            < 60
        )
        assert messages[3] == "claude turn: parsed session=s1 reply_chars=2"

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
                "--permission-mode",
                PERMISSION_MODE,
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
        messages = _messages(caplog)
        assert messages[0] == (
            f"claude turn: cwd={tmp_path} resume=True prompt_chars=5 timeout=1800s"
        )
        # The resumed session id stays readable; only the prompt is summarised.
        assert messages[1] == (
            "claude turn: invoking "
            "claude --print --output-format json --permission-mode "
            f"{PERMISSION_MODE} --resume s1 -p <prompt:5chars>"
        )
        assert messages[3] == "claude turn: parsed session=s1 reply_chars=10"

    def test_error_reply_raises(self, tmp_path: Path, monkeypatch) -> None:
        backend = ClaudeBackend()

        def fake_run(args, **kwargs):
            _assert_subprocess_kwargs(kwargs, tmp_path)
            # Session id present, so is_error is the only thing making this a
            # failure: a check that stopped reading the flag would return a
            # perfectly ordinary reply here.
            payload = json.dumps(
                {"is_error": True, "session_id": "s1", "result": "boom"}
            )
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(ClaudeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == "claude reported an error: boom"

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
            f"claude output was not JSON\nstdout: {stdout[:400]}\n…\n{stdout[-1600:]}"
        )
        assert stdout[:400] in str(excinfo.value)
        assert stdout[-1600:] in str(excinfo.value)

    def test_text_prefix_then_json_envelope_is_parsed(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Regression: Claude print mode wrote the reply, then the JSON."""
        backend = ClaudeBackend()
        envelope = {
            "type": "result",
            "subtype": "success",
            "session_id": "s1",
            "result": "I've read both skills",
        }
        stdout = "I've read both skills\n" + json.dumps(envelope)

        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = backend.run_turn("x", None, tmp_path)
        assert result.session_id == "s1"
        assert result.reply == "I've read both skills"

    def test_opt_in_reason_is_a_failure_not_a_reply(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A turn that ran with no tools exits 0 and says so only here."""
        backend = ClaudeBackend()
        payload = json.dumps(
            {
                "type": "result",
                "reason": OPT_IN_REQUIRED_REASON,
                "session_id": "s1",
                "result": "I could not run that command",
            }
        )

        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(ClaudeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == (
            f"claude ran without tools: reason={OPT_IN_REQUIRED_REASON}. "
            f"--permission-mode {PERMISSION_MODE} did not take effect."
        )

    def test_another_reason_is_still_a_normal_reply(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        backend = ClaudeBackend()
        payload = json.dumps(
            {"type": "result", "reason": "stop", "session_id": "s1", "result": "done"}
        )

        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert backend.run_turn("x", None, tmp_path).reply == "done"

    def test_missing_session_id_raises_rather_than_returning_none(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """None here would be persisted over the id the agent already has."""
        backend = ClaudeBackend()
        payload = json.dumps({"type": "system", "result": "hi"})

        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(ClaudeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == (
            f"claude did not report a session_id\nstdout: {payload}"
        )

    def test_blank_session_id_raises(self, tmp_path: Path, monkeypatch) -> None:
        backend = ClaudeBackend()
        payload = json.dumps({"session_id": "", "result": "hi"})

        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(ClaudeTurnError, match="did not report a session_id"):
            backend.run_turn("x", None, tmp_path)

    def test_non_string_session_id_raises(self, tmp_path: Path, monkeypatch) -> None:
        backend = ClaudeBackend()
        payload = json.dumps({"session_id": 17, "result": "hi"})

        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(ClaudeTurnError, match="did not report a session_id"):
            backend.run_turn("x", None, tmp_path)


class TestParseClaudeStdout:
    def test_plain_object(self) -> None:
        payload = {"session_id": "s1", "result": "hi"}
        assert parse_claude_stdout(json.dumps(payload)) == payload

    def test_empty_is_an_error(self) -> None:
        with pytest.raises(ClaudeTurnError) as excinfo:
            parse_claude_stdout("   ")
        assert str(excinfo.value) == "claude output was not JSON\nstdout: "

    def test_picks_the_type_result_object(self) -> None:
        result_obj = json.dumps(
            {"type": "result", "session_id": "keep", "result": "done"}
        )
        later = json.dumps({"type": "system", "session_id": "later"})
        parsed = parse_claude_stdout(f"{result_obj}\n{later}")
        assert parsed["session_id"] == "keep"
        assert parsed["result"] == "done"

    def test_falls_back_to_the_last_object(self) -> None:
        parsed = parse_claude_stdout(
            'noise {"other": 1} then {"result": "ok", "session_id": "s"}'
        )
        assert parsed == {"result": "ok", "session_id": "s"}

    def test_short_garbage_keeps_the_whole_dump(self) -> None:
        with pytest.raises(ClaudeTurnError) as excinfo:
            parse_claude_stdout("not json")
        assert str(excinfo.value) == "claude output was not JSON\nstdout: not json"

    def test_skips_a_broken_brace_and_keeps_the_next_object(self) -> None:
        parsed = parse_claude_stdout('{{"result": "ok"}')
        assert parsed == {"result": "ok"}

    def test_last_object_when_none_look_like_a_result(self) -> None:
        assert parse_claude_stdout('{"a": 1}{"b": 2}{"c": 3}') == {"c": 3}

    def test_error_snippet_keeps_exactly_2000_chars(self) -> None:
        text = "x" * 2000
        assert _stdout_for_error(text) == text

    def test_error_snippet_splits_a_2001_char_dump(self) -> None:
        text = ("H" * 400) + "M" + ("T" * 1600)
        assert _stdout_for_error(text) == f"{'H' * 400}\n…\n{'T' * 1600}"


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
        messages = _messages(caplog)
        assert messages[0] == (
            f"codex turn: cwd={tmp_path} resume=False prompt_chars=5 timeout=1800s"
        )
        # The prompt sits mid-argv for codex, so the summary must find it there.
        assert messages[1] == (
            "codex turn: invoking "
            "codex exec <prompt:5chars> --json --skip-git-repo-check"
        )
        assert (
            _reported_seconds(
                messages[2],
                r"codex turn: exited 0 after (\d+\.\d)s with \d+ chars of stdout",
            )
            < 60
        )
        assert messages[3] == "codex turn: parsed session=t1 messages=1 reply_chars=2"

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
        messages = _messages(caplog)
        assert messages[0] == (
            f"codex turn: cwd={tmp_path} resume=True prompt_chars=5 timeout=1800s"
        )
        assert messages[1] == (
            "codex turn: invoking "
            "codex exec resume t1 <prompt:5chars> --json --skip-git-repo-check"
        )
        assert messages[3] == "codex turn: parsed session=t1 messages=1 reply_chars=4"

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

    def test_reloaded_agent_keeps_the_name_from_state(self, tmp_path: Path) -> None:
        state_file = tmp_path / "s.json"
        Orchestrator(state_file=state_file).spawn("named", "echo")
        loaded = Orchestrator(state_file=state_file)
        assert loaded.agents["named"].name == "named"

    def test_agent_talk_resumes_on_the_same_instance(self, tmp_path: Path) -> None:
        seen: list[str | None] = []

        class Rec(AgentBackend):
            @property
            def name(self) -> str:
                return "rec"

            def run_turn(
                self, prompt: str, session_id: str | None, cwd: Path
            ) -> TurnResult:
                seen.append(session_id)
                return TurnResult(session_id="sid", reply="r", raw="")

        register_backend("rec", Rec)
        orch = Orchestrator(state_file=tmp_path / "s.json")
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
        Orchestrator(state_file=tmp_path / "s.json").spawn("a", "echo")
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

            def run_turn(
                self, prompt: str, session_id: str | None, cwd: Path
            ) -> TurnResult:
                Orchestrator(state_file=state_file).spawn("b", "echo")
                return TurnResult(session_id="s1", reply="ok", raw="")

        register_backend("midturn", MidTurnSpawn)
        orch = Orchestrator(state_file=state_file)
        orch.spawn("a", "midturn")
        orch.talk("a", "hi")
        reloaded = Orchestrator(state_file=state_file)
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

            def run_turn(
                self, prompt: str, session_id: str | None, cwd: Path
            ) -> TurnResult:
                probe = Orchestrator(state_file=state_file)
                held.append(
                    (
                        _flock_is_held(probe._agent_lock_path("a")),
                        _flock_is_held(probe._agent_lock_path("other")),
                    )
                )
                return TurnResult(session_id="s1", reply="ok", raw="")

        register_backend("probelock", ProbeLock)
        orch = Orchestrator(state_file=state_file)
        orch.spawn("a", "probelock")
        orch.talk("a", "hi")
        # Locked for this agent only, and released once the turn is over.
        assert held == [(True, False)]
        assert _flock_is_held(orch._agent_lock_path("a")) is False

    def test_agent_lock_paths_sit_beside_the_state_file(self, tmp_path: Path) -> None:
        orch = Orchestrator(state_file=tmp_path / "state.json")
        first = orch._agent_lock_path("a")
        assert first.parent == tmp_path
        assert first.name.startswith("state.json.")
        assert first.name.endswith(".lock")
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

            def run_turn(
                self, prompt: str, session_id: str | None, cwd: Path
            ) -> TurnResult:
                return next(replies)

        register_backend("forgetful", Forgetful)
        orch = Orchestrator(state_file=state_file)
        orch.spawn("a", "forgetful")
        orch.talk("a", "one")
        assert orch.talk("a", "two").reply == "second"
        assert orch.agents["a"].session_id == "s1"
        assert Orchestrator(state_file=state_file).agents["a"].session_id == "s1"

    def test_corrupt_state_names_the_file(self, tmp_path: Path) -> None:
        state_file = tmp_path / "state.json"
        state_file.write_text("{", encoding="utf-8")
        with pytest.raises(orchestrator.StateError) as excinfo:
            Orchestrator(state_file=state_file)
        assert str(excinfo.value).startswith(f"{state_file} is not valid JSON: ")

    def test_state_entry_without_a_backend_is_reported(self, tmp_path: Path) -> None:
        state_file = tmp_path / "state.json"
        state_file.write_text('{"a": {"session_id": "s1"}}', encoding="utf-8")
        with pytest.raises(orchestrator.StateError) as excinfo:
            Orchestrator(state_file=state_file)
        assert excinfo.value.args[0] == f"{state_file}: agent 'a' has no backend"

    def test_state_naming_an_unknown_backend_is_reported(self, tmp_path: Path) -> None:
        state_file = tmp_path / "state.json"
        state_file.write_text('{"a": {"backend": "ghost"}}', encoding="utf-8")
        with pytest.raises(UnknownBackendError, match="Unknown backend 'ghost'"):
            Orchestrator(state_file=state_file)

    def test_talk_fails_if_the_agent_is_deleted_during_the_turn(
        self, tmp_path: Path
    ) -> None:
        state_file = tmp_path / "state.json"

        class DeleteDuring(AgentBackend):
            @property
            def name(self) -> str:
                return "delduring"

            def run_turn(
                self, prompt: str, session_id: str | None, cwd: Path
            ) -> TurnResult:
                Orchestrator(state_file=state_file).delete("a")
                return TurnResult(session_id="s1", reply="ok", raw="")

        register_backend("delduring", DeleteDuring)
        orch = Orchestrator(state_file=state_file)
        orch.spawn("a", "delduring")
        with pytest.raises(KeyError, match="no agent named 'a'"):
            orch.talk("a", "hi")
        assert Orchestrator(state_file=state_file).list_agents() == []


class TestCLI:
    """cmd_* dispatch and printed output, backed by a fake CLI-free backend."""

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

    def test_cmd_talk_empty_prompt_exits_nonzero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Exit 0 here reads as a turn that ran to a caller under `set -e`."""
        orch = Orchestrator(state_file=tmp_path / "s.json")
        orch.spawn("a", "echo")
        with pytest.raises(SystemExit, match="2"):
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

    def test_cmd_talk_prints_backend_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        class BoomBackend(AgentBackend):
            @property
            def name(self) -> str:
                return "boom"

            def run_turn(
                self, prompt: str, session_id: str | None, cwd: Path
            ) -> TurnResult:
                raise ClaudeTurnError("claude output was not JSON")

        register_backend("boom", BoomBackend)
        orch = Orchestrator(state_file=tmp_path / "s.json")
        orch.spawn("b", "boom")
        with pytest.raises(SystemExit, match="1"):
            cmd_talk(orch, ["b", "hi"])
        captured = capsys.readouterr()
        assert captured.err == "claude output was not JSON\n"
        assert captured.out == ""

    def test_cmd_talk_prints_codex_backend_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        class BoomCodex(AgentBackend):
            @property
            def name(self) -> str:
                return "boomcodex"

            def run_turn(
                self, prompt: str, session_id: str | None, cwd: Path
            ) -> TurnResult:
                raise CodexTurnError("codex did not report a thread_id")

        register_backend("boomcodex", BoomCodex)
        orch = Orchestrator(state_file=tmp_path / "s.json")
        orch.spawn("c", "boomcodex")
        with pytest.raises(SystemExit, match="1"):
            cmd_talk(orch, ["c", "hi"])
        assert capsys.readouterr().err == "codex did not report a thread_id\n"

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
            "usage: orchestrator [-v|-vv] <command> [args...]\n"
            "       orchestrator [-v|-vv] --agent NAME --skill NAME[,NAME...] "
            "--prompt TEXT\n"
            "       orchestrator [-v|-vv] --list {agents,skills}\n"
            "  -h, --help      show this message\n"
            "  -v, --verbose   log each step and how long it took\n"
            "  -vv, --verbose2  also log full prompts and replies\n"
            "commands: spawn, talk, list, delete\n"
        )
        assert captured.out == ""

    def test_main_reads_sys_argv_when_none_given(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr("sys.argv", ["orchestrator", "bogus"])
        with pytest.raises(SystemExit, match="2"):
            main()
        assert "usage: orchestrator [-v|-vv] <command>" in capsys.readouterr().err

    def test_main_unknown_command_exits(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit, match="2"):
            main(["bogus"])
        assert "usage: orchestrator [-v|-vv] <command>" in capsys.readouterr().err

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

    def test_main_unknown_agent_is_one_line(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(orchestrator, "STATE_FILE", tmp_path / "s.json")
        with pytest.raises(SystemExit, match="1"):
            main(["talk", "nope", "hi"])
        captured = capsys.readouterr()
        assert captured.err == "no agent named 'nope'\n"
        assert captured.out == ""
        assert "Traceback" not in captured.err

    def test_main_duplicate_spawn_is_one_line(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(orchestrator, "STATE_FILE", tmp_path / "s.json")
        main(["spawn", "a", "-b", "echo"])
        capsys.readouterr()
        with pytest.raises(SystemExit, match="1"):
            main(["spawn", "a", "-b", "echo"])
        captured = capsys.readouterr()
        assert captured.err == "agent 'a' already exists\n"
        assert "Traceback" not in captured.err

    def test_main_unknown_delete_is_one_line(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(orchestrator, "STATE_FILE", tmp_path / "s.json")
        with pytest.raises(SystemExit, match="1"):
            main(["delete", "nope"])
        assert capsys.readouterr().err == "no agent named 'nope'\n"

    @pytest.mark.parametrize("flag", ["-h", "--help"])
    def test_main_help_describes_the_whole_cli(
        self, flag: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Routing every dash token to the skill parser hid the commands."""
        main([flag])
        captured = capsys.readouterr()
        assert (
            captured.out
            == f"{orchestrator.USAGE}\ncommands: spawn, talk, list, delete\n"
        )
        assert "--skill" in captured.out
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

            def run_turn(
                self, prompt: str, session_id: str | None, cwd: Path
            ) -> TurnResult:
                raise KeyError("some internal dict key")

        register_backend("buggy", BuggyBackend)
        monkeypatch.setattr(orchestrator, "STATE_FILE", tmp_path / "s.json")
        main(["spawn", "b", "-b", "buggy"])
        capsys.readouterr()
        with pytest.raises(KeyError, match="some internal dict key"):
            main(["talk", "b", "hi"])

    def test_main_corrupt_state_is_one_line(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        state = tmp_path / "s.json"
        state.write_text("{", encoding="utf-8")
        monkeypatch.setattr(orchestrator, "STATE_FILE", state)
        with pytest.raises(SystemExit, match="1"):
            main(["list"])
        err = capsys.readouterr().err
        assert err.endswith("\n")
        assert "Traceback" not in err
        assert err.count("\n") == 1


class TestDescribeCommand:
    def test_prompt_is_replaced_by_its_size(self) -> None:
        assert describe_command(["claude", "-p", "hello"], "hello") == (
            "claude -p <prompt:5chars>"
        )

    def test_prompt_in_the_middle_is_found(self) -> None:
        assert describe_command(["codex", "exec", "hi", "--json"], "hi") == (
            "codex exec <prompt:2chars> --json"
        )

    def test_output_does_not_grow_with_the_prompt(self) -> None:
        """The whole point: a 10k prompt must not produce a 10k log line."""
        short = describe_command(["claude", "-p", "x"], "x")
        long = describe_command(["claude", "-p", "x" * 10_000], "x" * 10_000)
        assert len(long) - len(short) == len("10000") - len("1")

    def test_flags_are_left_intact(self) -> None:
        args = ["claude", "--resume", "s1", "-p", "hi"]
        assert describe_command(args, "hi") == "claude --resume s1 -p <prompt:2chars>"

    def test_only_the_prompt_slot_is_replaced_when_it_matches_a_flag(self) -> None:
        args = ["codex", "exec", "resume", "t1", "resume", "--json"]
        assert describe_command(args, "resume") == (
            "codex exec resume t1 <prompt:6chars> --json"
        )

    def test_unmatched_prompt_leaves_args_unchanged(self) -> None:
        assert describe_command(["claude", "-p", "hello"], "other") == "claude -p hello"

    def test_empty_args_render_as_empty(self) -> None:
        assert describe_command([], "hello") == ""

    def test_prompt_in_the_first_two_slots_is_still_replaced(self) -> None:
        assert describe_command(["hello"], "hello") == "<prompt:5chars>"
        assert describe_command(["codex", "hi"], "hi") == "codex <prompt:2chars>"


class TestVerboseFlag:
    @pytest.fixture(autouse=True)
    def isolated_orchestrator(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            orchestrator,
            "Orchestrator",
            lambda: Orchestrator(state_file=tmp_path / "s.json"),
        )
        for name in ("orchestrator", "backends", "third_party"):
            logging.getLogger(name).setLevel(logging.NOTSET)

    @pytest.mark.parametrize(
        ("flag", "expected"),
        [
            ("-v", logging.DEBUG),
            ("--verbose", logging.DEBUG),
            ("-vv", orchestrator.TRACE),
            ("--verbose2", orchestrator.TRACE),
        ],
    )
    def test_each_flag_selects_its_level(self, flag: str, expected: int) -> None:
        main([flag, "list"])
        assert logging.getLogger("orchestrator").level == expected
        assert logging.getLogger("backends").level == expected

    @pytest.mark.parametrize("flag", ["-v", "--verbose", "-vv", "--verbose2"])
    def test_no_flag_leaks_third_party_debug_output(self, flag: str) -> None:
        """A dependency's debug output would bury the signal being asked for."""
        main([flag, "list"])
        assert logging.getLogger("third_party").getEffectiveLevel() > logging.DEBUG

    @pytest.mark.parametrize("flag", ["-v", "--verbose", "-vv", "--verbose2"])
    def test_flag_is_not_treated_as_a_command(self, flag: str, capsys) -> None:
        main([flag, "list"])
        assert capsys.readouterr().out == "no agents\n"

    def test_the_loudest_flag_given_wins(self) -> None:
        main(["-v", "-vv", "list"])
        assert logging.getLogger("orchestrator").level == orchestrator.TRACE

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
        assert orchestrator.TRACE < logging.DEBUG
        assert logging.getLevelName(orchestrator.TRACE) == "TRACE"

    def test_take_verbosity_consumes_only_a_leading_run(self) -> None:
        assert orchestrator._take_verbosity([]) == (0, [])
        assert orchestrator._take_verbosity(["-v"]) == (1, [])
        assert orchestrator._take_verbosity(["-v", "-vv", "talk", "-v"]) == (
            2,
            ["talk", "-v"],
        )
        assert orchestrator._take_verbosity(["talk", "-v"]) == (0, ["talk", "-v"])

    def test_prompt_token_equal_to_a_verbose_flag_is_kept(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        orch = Orchestrator(state_file=tmp_path / "s.json")
        orch.spawn("a", "echo")
        monkeypatch.setattr(orchestrator, "Orchestrator", lambda: orch)
        main(["talk", "a", "add", "-v", "please"])
        assert "echo:add -v please" in capsys.readouterr().out
        assert logging.getLogger("orchestrator").getEffectiveLevel() > logging.DEBUG

    def test_leading_flag_does_not_strip_the_same_token_from_the_prompt(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        orch = Orchestrator(state_file=tmp_path / "s.json")
        orch.spawn("a", "echo")
        monkeypatch.setattr(orchestrator, "Orchestrator", lambda: orch)
        main(["-v", "talk", "a", "add", "-v", "please"])
        assert "echo:add -v please" in capsys.readouterr().out
        assert logging.getLogger("orchestrator").level == logging.DEBUG


class TestStepLogging:
    def test_state_load_and_write_are_logged(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        state_file = tmp_path / "s.json"
        with caplog.at_level("DEBUG", logger="orchestrator"):
            orch = Orchestrator(state_file=state_file)
            orch.spawn("a", "echo")
        messages = _messages(caplog)
        assert f"state: loaded 0 agent(s) from {state_file}" in messages
        assert f"state: wrote 1 agent(s) to {state_file}" in messages

    def test_turn_start_and_duration_are_logged(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        orch = Orchestrator(state_file=tmp_path / "s.json")
        orch.spawn("a", "echo")
        with caplog.at_level("DEBUG", logger="orchestrator"):
            orch.talk("a", "hi")
        messages = _messages(caplog)
        assert "agent 'a' (echo): starting turn, resume=False" in messages
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
        orch = Orchestrator(state_file=tmp_path / "s.json")
        orch.spawn("a", "echo")
        prompt = "line one\nline two"
        with caplog.at_level(orchestrator.TRACE, logger="orchestrator"):
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
        orch = Orchestrator(state_file=tmp_path / "s.json")
        orch.spawn("a", "echo")
        with caplog.at_level(logging.DEBUG, logger="orchestrator"):
            orch.talk("a", "secret prompt")
        assert not any("secret prompt" in message for message in _messages(caplog))

    def test_a_resumed_turn_is_logged_as_such(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """resume= must reflect the session, not be hardcoded by either turn."""
        orch = Orchestrator(state_file=tmp_path / "s.json")
        orch.spawn("a", "echo")
        orch.talk("a", "first")
        with caplog.at_level("DEBUG", logger="orchestrator"):
            orch.talk("a", "second")
        assert "agent 'a' (echo): starting turn, resume=True" in _messages(caplog)

    def test_cli_argument_shape_and_dispatch_are_logged(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(
            orchestrator,
            "Orchestrator",
            lambda: Orchestrator(state_file=tmp_path / "s.json"),
        )
        with caplog.at_level("DEBUG", logger="orchestrator"):
            main(["-v", "spawn", "a", "-b", "echo"])
        messages = _messages(caplog)
        # Four arguments remain once -v is stripped.
        assert "cli: 4 argument(s) after flag removal" in messages
        assert "cli: dispatching 'spawn'" in messages


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

        orchestrator._configure_logging(False)

        assert root.level == logging.WARNING

    def test_records_carry_time_level_and_logger_name(self) -> None:
        root = logging.getLogger()
        root.handlers.clear()

        orchestrator._configure_logging(False)

        record = logging.LogRecord(
            "backends.claude", logging.WARNING, "p", 1, "msg", None, None
        )
        assert re.fullmatch(
            r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} "
            r"WARNING backends\.claude: msg",
            root.handlers[0].format(record),
        )
