"""Unit tests for agent backend interface, implementations, and orchestrator."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from backends.base import (
    OutputSchema,
    stdout_for_error,
)
from backends.claude import FORK_FLAG as CLAUDE_FORK_FLAG
from backends.claude import (
    OPT_IN_REQUIRED_REASON,
    PERMISSION_MODE,
    ClaudeBackend,
    ClaudeTurnError,
    parse_claude_stdout,
)
from backends.claude import SCHEMA_FLAG as CLAUDE_SCHEMA_FLAG
from backends.claude import (
    format_event as claude_format_event,
)
from tests.backend_helpers import (
    SCHEMA,
    _assert_subprocess_kwargs,
    _completed,
    _messages,
    _reported_seconds,
    _subprocess_recorder,
)

# The same schema as the orchestrator loads it from a file that declares its
# dialect. Its keys are deliberately unsorted, so a re-serialisation that lost
# the canonical ordering would be visible.
DIALECT_SCHEMA = OutputSchema(
    text=(
        '{"type":"object","$schema":"https://json-schema.org/draft/2020-12/schema"'
        ',"additionalProperties":false,"properties":{}}'
    ),
    path=Path("/schemas/reply.json"),
)


class TestClaudeRunTurn:
    def test_permission_mode_is_the_noninteractive_opt_in(self) -> None:
        assert PERMISSION_MODE == "bypassPermissions"

    def test_null_result_is_an_empty_reply_not_a_crash(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """An explicit null must not reach len() in the debug log."""
        backend = ClaudeBackend()
        payload = json.dumps({"type": "result", "session_id": "s1", "result": None})

        fake_run, _ = _subprocess_recorder(_completed(0, payload))
        monkeypatch.setattr(subprocess, "run", fake_run)
        result = backend.run_turn("x", None, tmp_path)
        assert result.reply == ""
        assert result.session_id == "s1"

    def test_model_and_effort_are_forwarded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = ClaudeBackend(model="sonnet", reasoning_effort="high")
        payload = json.dumps({"session_id": "s1", "result": "done"})

        fake_run, calls = _subprocess_recorder(_completed(0, payload))
        monkeypatch.setattr(subprocess, "run", fake_run)
        result = backend.run_turn("work", None, tmp_path)

        assert calls[0][0] == [
            "claude",
            "--print",
            "--output-format",
            "json",
            "--permission-mode",
            PERMISSION_MODE,
            "--model",
            "sonnet",
            "--effort",
            "high",
            "-p",
            "work",
        ]
        assert result.reply == "done"

    def test_streaming_turn_uses_stream_json_and_keeps_result_fields(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = ClaudeBackend(model="sonnet", reasoning_effort="high")
        stdout = "\n".join(
            [
                json.dumps({"type": "system", "subtype": "init"}),
                json.dumps(
                    {
                        "type": "result",
                        "session_id": "s1",
                        "result": '{"verdict":"pass"}',
                        "structured_output": {"verdict": "pass"},
                    }
                ),
                json.dumps({"type": "system", "subtype": "after_result"}),
            ]
        )

        def fake_run(name, args, **kwargs):
            assert name == "claude"
            assert args == [
                "claude",
                "--print",
                "--output-format",
                "stream-json",
                "--verbose",
                "--permission-mode",
                PERMISSION_MODE,
                "--model",
                "sonnet",
                "--effort",
                "high",
                "--json-schema",
                SCHEMA.text,
                "-p",
                "work",
            ]
            assert kwargs["stream"] is True
            assert kwargs["format_event"] is claude_format_event
            return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

        monkeypatch.setattr("backends.claude.run_cli_turn", fake_run)
        result = backend.run_turn("work", None, tmp_path, schema=SCHEMA, stream=True)

        assert result.session_id == "s1"
        assert result.reply == '{"verdict":"pass"}'
        assert result.raw == stdout
        assert result.structured == {"verdict": "pass"}

    def test_new_turn_parses_session_and_result(
        self, tmp_path: Path, monkeypatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        backend = ClaudeBackend()
        payload = json.dumps({"is_error": False, "session_id": "s1", "result": "hi"})

        fake_run, calls = _subprocess_recorder(_completed(0, payload))
        monkeypatch.setattr(subprocess, "run", fake_run)
        with caplog.at_level("DEBUG"):
            result = backend.run_turn("hello", None, tmp_path)
        assert calls[0][0] == [
            "claude",
            "--print",
            "--output-format",
            "json",
            "--permission-mode",
            PERMISSION_MODE,
            "-p",
            "hello",
        ]
        _assert_subprocess_kwargs(calls[0][1], tmp_path)
        assert result.session_id == "s1"
        assert result.reply == "hi"
        assert result.raw == payload
        messages = _messages(caplog)
        assert messages[0] == (
            f"claude turn: cwd={tmp_path} resume=False prompt_chars=5 timeout=3600s"
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
        payload = json.dumps(
            {"is_error": False, "session_id": "s1", "result": "still here"}
        )

        fake_run, calls = _subprocess_recorder(_completed(0, payload))
        monkeypatch.setattr(subprocess, "run", fake_run)
        with caplog.at_level("DEBUG"):
            result = backend.run_turn("again", "s1", tmp_path)
        assert calls[0][0] == [
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
        _assert_subprocess_kwargs(calls[0][1], tmp_path)
        assert result.session_id == "s1"
        assert result.reply == "still here"
        messages = _messages(caplog)
        assert messages[0] == (
            f"claude turn: cwd={tmp_path} resume=True prompt_chars=5 timeout=3600s"
        )
        # The resumed session id stays readable; only the prompt is summarised.
        assert messages[1] == (
            "claude turn: invoking "
            "claude --print --output-format json --permission-mode "
            f"{PERMISSION_MODE} --resume s1 -p <prompt:5chars>"
        )
        assert messages[3] == "claude turn: parsed session=s1 reply_chars=10"

    def test_forked_resume_adds_the_fork_flag_next_to_the_source_session(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A forked resume continues the *source's* session in a copy, so the
        id on the command line is the source's and the flag is what makes the
        turn land in a new one."""
        backend = ClaudeBackend()
        payload = json.dumps(
            {"is_error": False, "session_id": "forked-sid", "result": "hi"}
        )

        fake_run, calls = _subprocess_recorder(_completed(0, payload))
        monkeypatch.setattr(subprocess, "run", fake_run)
        result = backend.run_turn("again", "source-sid", tmp_path, resume_as_fork=True)
        assert calls[0][0] == [
            "claude",
            "--print",
            "--output-format",
            "json",
            "--permission-mode",
            PERMISSION_MODE,
            "--resume",
            "source-sid",
            CLAUDE_FORK_FLAG,
            "-p",
            "again",
        ]
        _assert_subprocess_kwargs(calls[0][1], tmp_path)
        assert result.session_id == "forked-sid"

    def test_error_reply_raises(self, tmp_path: Path, monkeypatch) -> None:
        backend = ClaudeBackend()

        # Session id present, so is_error is the only thing making this a
        # failure: a check that stopped reading the flag would return a
        # perfectly ordinary reply here.
        payload = json.dumps({"is_error": True, "session_id": "s1", "result": "boom"})
        fake_run, calls = _subprocess_recorder(_completed(0, payload))
        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(ClaudeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        _assert_subprocess_kwargs(calls[0][1], tmp_path)
        assert str(excinfo.value) == "claude reported an error: boom"

    def test_result_defaults_to_empty_reply(self, tmp_path: Path, monkeypatch) -> None:
        backend = ClaudeBackend()
        payload = json.dumps({"is_error": False, "session_id": "s1"})

        fake_run, _ = _subprocess_recorder(_completed(0, payload))
        monkeypatch.setattr(subprocess, "run", fake_run)
        result = backend.run_turn("x", None, tmp_path)
        assert result.reply == ""

    def test_nonzero_exit_raises(self, tmp_path: Path, monkeypatch) -> None:
        backend = ClaudeBackend()
        stderr = "s" * 500 + "M" + "e" * 1999  # 2500 chars total

        fake_run, calls = _subprocess_recorder(_completed(1, "", stderr))
        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(ClaudeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        _assert_subprocess_kwargs(calls[0][1], tmp_path)
        assert str(excinfo.value) == (
            f"claude exited 1\nstderr: {stderr[-2000:]}\nstdout: "
        )

    def test_malformed_json_raises(self, tmp_path: Path, monkeypatch) -> None:
        backend = ClaudeBackend()
        stdout = "s" * 500 + "M" + "e" * 1999  # 2500 chars total, not valid JSON

        fake_run, calls = _subprocess_recorder(_completed(0, stdout))
        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(ClaudeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        _assert_subprocess_kwargs(calls[0][1], tmp_path)
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

        fake_run, _ = _subprocess_recorder(_completed(0, stdout))
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

        fake_run, _ = _subprocess_recorder(_completed(0, payload))
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

        fake_run, _ = _subprocess_recorder(_completed(0, payload))
        monkeypatch.setattr(subprocess, "run", fake_run)
        assert backend.run_turn("x", None, tmp_path).reply == "done"

    def test_missing_session_id_raises_rather_than_returning_none(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """None here would be persisted over the id the agent already has."""
        backend = ClaudeBackend()
        payload = json.dumps({"type": "system", "result": "hi"})

        fake_run, _ = _subprocess_recorder(_completed(0, payload))
        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(ClaudeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == (
            f"claude did not report a session_id\nstdout: {payload}"
        )

    def test_blank_session_id_raises(self, tmp_path: Path, monkeypatch) -> None:
        backend = ClaudeBackend()
        payload = json.dumps({"session_id": "", "result": "hi"})

        fake_run, _ = _subprocess_recorder(_completed(0, payload))
        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(ClaudeTurnError, match="did not report a session_id"):
            backend.run_turn("x", None, tmp_path)

    def test_non_string_session_id_raises(self, tmp_path: Path, monkeypatch) -> None:
        backend = ClaudeBackend()
        payload = json.dumps({"session_id": 17, "result": "hi"})

        fake_run, _ = _subprocess_recorder(_completed(0, payload))
        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(ClaudeTurnError, match="did not report a session_id"):
            backend.run_turn("x", None, tmp_path)

    def test_schema_is_passed_inline_and_parsed_from_its_own_field(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = ClaudeBackend()
        # The reply text is deliberately not the object: claude's own parse is
        # preferred, and parsing `result` is only the fallback below.
        payload = json.dumps(
            {
                "session_id": "s1",
                "result": "see structured_output",
                "structured_output": {"verdict": "pass"},
            }
        )

        fake_run, calls = _subprocess_recorder(_completed(0, payload))
        monkeypatch.setattr(subprocess, "run", fake_run)
        result = backend.run_turn("hello", None, tmp_path, schema=SCHEMA)
        assert calls[0][0] == [
            "claude",
            "--print",
            "--output-format",
            "json",
            "--permission-mode",
            PERMISSION_MODE,
            CLAUDE_SCHEMA_FLAG,
            SCHEMA.text,
            "-p",
            "hello",
        ]
        _assert_subprocess_kwargs(calls[0][1], tmp_path)
        assert result.structured == {"verdict": "pass"}
        assert result.reply == "see structured_output"

    def test_the_dialect_keyword_is_dropped_from_the_inline_schema(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`--json-schema` is checked by a draft-07 ajv: a `$schema` naming a
        newer dialect makes the CLI exit 1 before the turn runs."""
        backend = ClaudeBackend()
        payload = json.dumps(
            {"session_id": "s1", "result": "{}", "structured_output": {}}
        )
        fake_run, calls = _subprocess_recorder(_completed(0, payload))
        monkeypatch.setattr(subprocess, "run", fake_run)
        backend.run_turn("hello", None, tmp_path, schema=DIALECT_SCHEMA)
        backend.run_turn("hello", None, tmp_path, schema=SCHEMA)

        assert [
            call_args[0][call_args[0].index(CLAUDE_SCHEMA_FLAG) + 1]
            for call_args in calls
        ] == [
            '{"additionalProperties":false,"properties":{},"type":"object"}',
            SCHEMA.text,
        ]

    def test_structured_falls_back_to_parsing_the_result_text(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A CLI that stops publishing its own parse still has the reply."""
        backend = ClaudeBackend()
        payload = json.dumps({"session_id": "s1", "result": '{"verdict":"pass"}'})
        monkeypatch.setattr(subprocess, "run", _completed(0, payload, ""))
        result = backend.run_turn("hello", None, tmp_path, schema=SCHEMA)
        assert result.structured == {"verdict": "pass"}

    def test_a_reply_that_is_not_an_object_is_not_a_failed_turn(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The validator retries that; raising here would deny it the chance."""
        backend = ClaudeBackend()
        payload = json.dumps({"session_id": "s1", "result": "here you go:"})
        monkeypatch.setattr(subprocess, "run", _completed(0, payload, ""))
        result = backend.run_turn("hello", None, tmp_path, schema=SCHEMA)
        assert result.structured is None
        assert result.reply == "here you go:"

    def test_no_schema_leaves_structured_unset_even_for_a_json_reply(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = ClaudeBackend()
        payload = json.dumps({"session_id": "s1", "result": '{"verdict":"pass"}'})

        fake_run, calls = _subprocess_recorder(_completed(0, payload))
        monkeypatch.setattr(subprocess, "run", fake_run)
        result = backend.run_turn("hello", None, tmp_path)
        assert CLAUDE_SCHEMA_FLAG not in calls[0][0]
        assert result.structured is None

    def test_nonzero_exit_reports_the_error_envelope_over_the_exit_code(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The envelope names the failure; the exit code only counts it."""
        backend = ClaudeBackend()
        stdout = json.dumps(
            {"type": "result", "is_error": True, "result": "credit balance too low"}
        )

        fake_run, calls = _subprocess_recorder(_completed(1, stdout))
        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(ClaudeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        _assert_subprocess_kwargs(calls[0][1], tmp_path)
        assert str(excinfo.value) == "claude reported an error: credit balance too low"

    def test_nonzero_exit_keeps_stdout_when_stderr_is_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The failure that prompted this: exit 1, stderr empty, and the only
        thing the CLI said sitting unread on stdout."""
        backend = ClaudeBackend()
        stdout = "Invalid API key · Please run /login"

        fake_run, calls = _subprocess_recorder(_completed(1, stdout))
        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(ClaudeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        _assert_subprocess_kwargs(calls[0][1], tmp_path)
        assert str(excinfo.value) == (f"claude exited 1\nstderr: \nstdout: {stdout}")

    def test_nonzero_exit_bounds_a_long_stdout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both ends of the dump, the same bound the parse failure uses."""
        backend = ClaudeBackend()
        stdout = "s" * 500 + "M" + "e" * 1999  # 2500 chars, not an envelope

        fake_run, calls = _subprocess_recorder(_completed(2, stdout, "boom"))
        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(ClaudeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        _assert_subprocess_kwargs(calls[0][1], tmp_path)
        assert str(excinfo.value) == (
            f"claude exited 2\nstderr: boom\nstdout: {stdout_for_error(stdout)}"
        )

    def test_nonzero_exit_ignores_a_non_error_envelope(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`is_error` false with a non-zero exit is not a reported error, so the
        exit code stays the headline rather than `result` being read as one."""
        backend = ClaudeBackend()
        stdout = json.dumps({"type": "result", "is_error": False, "result": "hi"})

        fake_run, calls = _subprocess_recorder(_completed(3, stdout))
        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(ClaudeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        _assert_subprocess_kwargs(calls[0][1], tmp_path)
        assert str(excinfo.value) == f"claude exited 3\nstderr: \nstdout: {stdout}"


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
