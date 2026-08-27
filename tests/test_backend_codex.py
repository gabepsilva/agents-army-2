"""Unit tests for agent backend interface, implementations, and orchestrator."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from backends.base import (
    OutputSchema,
    TurnResult,
)
from backends.codex import FORK_COMMAND as CODEX_FORK_COMMAND
from backends.codex import SCHEMA_FLAG as CODEX_SCHEMA_FLAG
from backends.codex import YOLO_FLAG as CODEX_YOLO_FLAG
from backends.codex import (
    CodexBackend,
    CodexTurnError,
)
from backends.codex import (
    format_event as codex_format_event,
)
from tests.backend_helpers import (
    SCHEMA,
    _assert_subprocess_kwargs,
    _completed,
    _messages,
    _reported_seconds,
    _subprocess_recorder,
)


class TestCodexRunTurn:
    def test_model_and_effort_are_forwarded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = CodexBackend(model="gpt-test", reasoning_effort="xhigh")
        stdout = (
            '{"type":"thread.started","thread_id":"t1"}\n'
            '{"type":"item.completed","item":'
            '{"type":"agent_message","text":"done"}}\n'
        )

        fake_run, calls = _subprocess_recorder(_completed(0, stdout))
        monkeypatch.setattr(subprocess, "run", fake_run)
        result = backend.run_turn("work", None, tmp_path)
        assert calls[0][0] == [
            "codex",
            "exec",
            CODEX_YOLO_FLAG,
            "--model",
            "gpt-test",
            "--config",
            'model_reasoning_effort="xhigh"',
            "work",
            "--json",
            "--skip-git-repo-check",
        ]
        assert result.reply == "done"

    def test_streaming_turn_passes_formatter_and_keeps_codex_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = CodexBackend()
        stdout = "\n".join(
            [
                '{"type":"thread.started","thread_id":"t1"}',
                '{"type":"item.completed","item":{"type":"agent_message","text":"done"}}',
                '{"type":"turn.completed","usage":{"output_tokens":5}}',
            ]
        )

        def fake_run(name, args, **kwargs):
            assert name == "codex"
            assert args == [
                "codex",
                "exec",
                CODEX_YOLO_FLAG,
                "work",
                "--json",
                "--skip-git-repo-check",
            ]
            assert kwargs["stream"] is True
            assert kwargs["format_event"] is codex_format_event
            return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

        monkeypatch.setattr("backends.codex.run_cli_turn", fake_run)
        result = backend.run_turn("work", None, tmp_path, stream=True)

        assert result == TurnResult("t1", "done", stdout)

    def test_new_turn_parses_thread_and_reply(
        self, tmp_path: Path, monkeypatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        backend = CodexBackend()
        stdout = (
            '{"type":"thread.started","thread_id":"t1"}\n'
            '{"type":"item.completed","item":{"type":"agent_message","text":"yo"}}\n'
        )

        fake_run, calls = _subprocess_recorder(_completed(0, stdout))
        monkeypatch.setattr(subprocess, "run", fake_run)
        with caplog.at_level("DEBUG"):
            result = backend.run_turn("hello", None, tmp_path)
        assert calls[0][0] == [
            "codex",
            "exec",
            CODEX_YOLO_FLAG,
            "hello",
            "--json",
            "--skip-git-repo-check",
        ]
        _assert_subprocess_kwargs(calls[0][1], tmp_path)
        assert result.session_id == "t1"
        assert result.reply == "yo"
        messages = _messages(caplog)
        assert messages[0] == (
            f"codex turn: cwd={tmp_path} resume=False prompt_chars=5 timeout=3600s"
        )
        # The prompt sits mid-argv for codex, so the summary must find it there.
        assert messages[1] == (
            "codex turn: invoking "
            "codex exec --yolo <prompt:5chars> --json --skip-git-repo-check"
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
        stdout = (
            '{"type":"thread.started","thread_id":"t1"}\n'
            '{"type":"item.completed","item":{"type":"agent_message","text":"back"}}\n'
        )

        fake_run, calls = _subprocess_recorder(_completed(0, stdout))
        monkeypatch.setattr(subprocess, "run", fake_run)
        with caplog.at_level("DEBUG"):
            result = backend.run_turn("again", "t1", tmp_path)
        assert calls[0][0] == [
            "codex",
            "exec",
            CODEX_YOLO_FLAG,
            "resume",
            "t1",
            "again",
            "--json",
            "--skip-git-repo-check",
        ]
        _assert_subprocess_kwargs(calls[0][1], tmp_path)
        assert result.session_id == "t1"
        assert result.reply == "back"
        messages = _messages(caplog)
        assert messages[0] == (
            f"codex turn: cwd={tmp_path} resume=True prompt_chars=5 timeout=3600s"
        )
        assert messages[1] == (
            "codex turn: invoking "
            "codex exec --yolo resume t1 <prompt:5chars> --json --skip-git-repo-check"
        )
        assert messages[3] == "codex turn: parsed session=t1 messages=1 reply_chars=4"

    def test_forked_resume_swaps_fork_in_for_resume(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`codex exec fork <source> <prompt>` takes `resume`'s whole slot: the
        config flags that precede it and the prompt that follows are where an
        ordinary resume puts them, and no `resume` survives in the argv."""
        backend = CodexBackend(model="gpt-test", reasoning_effort="xhigh")
        schema = OutputSchema(text="{}", path=tmp_path / "schema.json")
        stdout = (
            '{"type":"thread.started","thread_id":"forked-tid"}\n'
            '{"type":"item.completed","item":'
            '{"type":"agent_message","text":"hi"}}\n'
        )

        fake_run, calls = _subprocess_recorder(_completed(0, stdout))
        monkeypatch.setattr(subprocess, "run", fake_run)
        result = backend.run_turn(
            "again", "source-tid", tmp_path, schema=schema, resume_as_fork=True
        )
        assert calls[0][0] == [
            "codex",
            "exec",
            CODEX_YOLO_FLAG,
            "--model",
            "gpt-test",
            "--config",
            'model_reasoning_effort="xhigh"',
            CODEX_FORK_COMMAND,
            "source-tid",
            "again",
            "--json",
            "--skip-git-repo-check",
            CODEX_SCHEMA_FLAG,
            str(schema.path),
        ]
        assert "resume" not in calls[0][0]
        _assert_subprocess_kwargs(calls[0][1], tmp_path)
        assert result.session_id == "forked-tid"

    def test_no_thread_id_raises(self, tmp_path: Path, monkeypatch) -> None:
        backend = CodexBackend()

        fake_run, calls = _subprocess_recorder(_completed(0, ""))
        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(CodexTurnError, match="thread_id"):
            backend.run_turn("x", None, tmp_path)
        _assert_subprocess_kwargs(calls[0][1], tmp_path)

    def test_no_thread_id_error_keeps_the_tail_of_long_output(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        backend = CodexBackend()
        stdout = "o" * 2500  # not JSON, so session_id stays None
        stderr = "e" * 2500

        fake_run, _ = _subprocess_recorder(_completed(0, stdout, stderr))
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

        fake_run, calls = _subprocess_recorder(_completed(1, "", stderr))
        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(CodexTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        _assert_subprocess_kwargs(calls[0][1], tmp_path)
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

        fake_run, calls = _subprocess_recorder(_completed(0, stdout))
        monkeypatch.setattr(subprocess, "run", fake_run)
        result = backend.run_turn("x", None, tmp_path)
        _assert_subprocess_kwargs(calls[0][1], tmp_path)
        assert result.session_id == "t1"
        assert result.reply == "hi\nthere"
        assert result.raw == stdout

    def test_schema_is_passed_as_a_file_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """codex reads the schema itself, so it gets the path, not the text."""
        backend = CodexBackend()
        stdout = (
            '{"type":"thread.started","thread_id":"t1"}\n'
            '{"type":"item.completed","item":{"type":"agent_message",'
            '"text":"{\\"verdict\\":\\"pass\\"}"}}\n'
        )

        fake_run, calls = _subprocess_recorder(_completed(0, stdout))
        monkeypatch.setattr(subprocess, "run", fake_run)
        result = backend.run_turn("hello", None, tmp_path, schema=SCHEMA)
        assert calls[0][0] == [
            "codex",
            "exec",
            CODEX_YOLO_FLAG,
            "hello",
            "--json",
            "--skip-git-repo-check",
            CODEX_SCHEMA_FLAG,
            str(SCHEMA.path),
        ]
        assert SCHEMA.text not in calls[0][0]
        _assert_subprocess_kwargs(calls[0][1], tmp_path)
        # No pre-parsed field anywhere in the stream: the object is the text.
        assert result.structured == {"verdict": "pass"}

    def test_prose_around_the_object_leaves_structured_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = CodexBackend()
        stdout = (
            '{"type":"thread.started","thread_id":"t1"}\n'
            '{"type":"item.completed","item":{"type":"agent_message",'
            '"text":"Sure! {\\"verdict\\":\\"pass\\"}"}}\n'
        )
        monkeypatch.setattr(subprocess, "run", _completed(0, stdout, ""))
        assert backend.run_turn("hi", None, tmp_path, schema=SCHEMA).structured is None

    def test_no_schema_omits_the_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = CodexBackend()
        stdout = '{"type":"thread.started","thread_id":"t1"}\n'

        fake_run, calls = _subprocess_recorder(_completed(0, stdout))
        monkeypatch.setattr(subprocess, "run", fake_run)
        result = backend.run_turn("hi", None, tmp_path)
        assert CODEX_SCHEMA_FLAG not in calls[0][0]
        assert result.structured is None

    def test_nonzero_exit_reports_the_api_message_not_the_stderr_tail(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A rejected schema is the case: the API says which part it refused,
        while stderr holds only the CLI's stdin notice."""
        backend = CodexBackend()
        stdout = (
            '{"type":"thread.started","thread_id":"t1"}\n'
            '{"type":"error","message":"Invalid schema for response_format: '
            "In context=('properties', 'detail'), 'additionalProperties' is "
            'required to be supplied and to be false."}\n'
        )
        monkeypatch.setattr(
            subprocess,
            "run",
            _completed(1, stdout, "Reading additional input from stdin..."),
        )
        with pytest.raises(CodexTurnError) as excinfo:
            backend.run_turn("hi", None, tmp_path, schema=SCHEMA)
        assert "In context=('properties', 'detail')" in str(excinfo.value)
        assert "stdin" not in str(excinfo.value)

    def test_an_error_before_a_later_event_is_still_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The failure is not always the last line: codex keeps narrating."""
        backend = CodexBackend()
        stdout = (
            '{"type":"thread.started","thread_id":"t1"}\n'
            '{"type":"error","message":"the real failure"}\n'
            '{"type":"item.completed","item":{"type":"agent_message","text":""}}\n'
        )
        monkeypatch.setattr(subprocess, "run", _completed(1, stdout, "noise"))
        with pytest.raises(CodexTurnError) as excinfo:
            backend.run_turn("hi", None, tmp_path)
        assert str(excinfo.value) == "codex reported an error: the real failure"

    def test_nonzero_exit_reads_a_failed_turn_event(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = CodexBackend()
        stdout = '{"type":"turn.failed","error":{"message":"usage limit reached"}}\n'
        monkeypatch.setattr(subprocess, "run", _completed(1, stdout, "noise"))
        with pytest.raises(
            CodexTurnError, match="codex reported an error: usage limit"
        ):
            backend.run_turn("hi", None, tmp_path)

    def test_nonzero_exit_without_a_message_keeps_the_exit_and_stderr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every shape that carries no message to report: an error event with
        no message, one with an empty message, and a failed turn whose error is
        not an object. An empty message is not a message: reporting it would
        print `codex reported an error: ` and nothing else."""
        backend = CodexBackend()
        stdout = (
            '{"type":"thread.started","thread_id":"t1"}\n'
            '{"type":"error"}\n'
            '{"type":"error","message":""}\n'
            '{"type":"turn.failed","error":"boom"}\n'
        )
        monkeypatch.setattr(subprocess, "run", _completed(3, stdout, "segfault"))
        with pytest.raises(CodexTurnError) as excinfo:
            backend.run_turn("hi", None, tmp_path)
        assert str(excinfo.value) == "codex exited 3\nstderr: segfault"

    def test_the_last_error_reported_is_the_one_raised(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An earlier event can be one the CLI recovered from; the last one is
        what it gave up on."""
        backend = CodexBackend()
        stdout = (
            '{"type":"error","message":"retrying after a stream hiccup"}\n'
            '{"type":"error","message":"the real failure"}\n'
        )
        monkeypatch.setattr(subprocess, "run", _completed(1, stdout, ""))
        with pytest.raises(CodexTurnError) as excinfo:
            backend.run_turn("hi", None, tmp_path)
        assert str(excinfo.value) == "codex reported an error: the real failure"
