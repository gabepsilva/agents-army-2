"""Unit tests for agent backend interface, implementations, and orchestrator."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from backends.grok import (
    ALWAYS_APPROVE_FLAG,
    PROMPT_FLAG,
    GrokBackend,
    GrokTurnError,
    parse_grok_stdout,
)
from backends.grok import FORK_FLAG as GROK_FORK_FLAG
from backends.grok import SCHEMA_FLAG as GROK_SCHEMA_FLAG
from tests.backend_helpers import (
    SCHEMA,
    _assert_subprocess_kwargs,
    _completed,
    _messages,
    _reported_seconds,
)


class TestGrokRunTurn:
    def test_model_and_effort_are_forwarded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = GrokBackend(model="grok-test", reasoning_effort="high")
        payload = json.dumps({"sessionId": "s1", "text": "done"})

        def fake_run(args, **_kwargs):
            assert args == [
                "grok",
                "--output-format",
                "json",
                "--always-approve",
                "--model",
                "grok-test",
                "--reasoning-effort",
                "high",
                "--single=work",
            ]
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert backend.run_turn("work", None, tmp_path).reply == "done"

    def test_new_turn_parses_session_and_text(
        self, tmp_path: Path, monkeypatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        backend = GrokBackend()
        payload = json.dumps(
            {"sessionId": "s1", "text": "hi", "stopReason": "end_turn"}
        )

        def fake_run(args, **kwargs):
            assert args == [
                "grok",
                "--output-format",
                "json",
                "--always-approve",
                "--single=hello",
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
            f"grok turn: cwd={tmp_path} resume=False prompt_chars=5 timeout=3600s"
        )
        assert messages[1] == (
            "grok turn: invoking "
            "grok --output-format json --always-approve --single=<prompt:5chars>"
        )
        assert (
            _reported_seconds(
                messages[2],
                rf"grok turn: exited 0 after (\d+\.\d)s with {len(payload)} "
                rf"chars of stdout",
            )
            < 60
        )
        assert messages[3] == "grok turn: parsed session=s1 reply_chars=2"

    def test_resume_uses_resume_flag_not_session_id(
        self, tmp_path: Path, monkeypatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """`--session-id` names a new session and errors if that id exists."""
        backend = GrokBackend()

        def fake_run(args, **kwargs):
            assert args == [
                "grok",
                "--output-format",
                "json",
                "--always-approve",
                "--resume",
                "s1",
                "--single=again",
            ]
            assert "--session-id" not in args
            assert "-s" not in args
            _assert_subprocess_kwargs(kwargs, tmp_path)
            payload = json.dumps({"sessionId": "s1", "text": "still here"})
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with caplog.at_level("DEBUG"):
            result = backend.run_turn("again", "s1", tmp_path)
        assert result.session_id == "s1"
        assert result.reply == "still here"
        messages = _messages(caplog)
        assert messages[0] == (
            f"grok turn: cwd={tmp_path} resume=True prompt_chars=5 timeout=3600s"
        )
        assert messages[1] == (
            "grok turn: invoking "
            "grok --output-format json --always-approve --resume s1 "
            "--single=<prompt:5chars>"
        )
        assert messages[3] == "grok turn: parsed session=s1 reply_chars=10"

    def test_forked_resume_adds_the_fork_flag_next_to_the_source_session(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        backend = GrokBackend()

        def fake_run(args, **kwargs):
            assert args == [
                "grok",
                "--output-format",
                "json",
                ALWAYS_APPROVE_FLAG,
                "--resume",
                "source-sid",
                GROK_FORK_FLAG,
                f"{PROMPT_FLAG}=again",
            ]
            _assert_subprocess_kwargs(kwargs, tmp_path)
            payload = json.dumps({"sessionId": "forked-sid", "text": "hi"})
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = backend.run_turn("again", "source-sid", tmp_path, resume_as_fork=True)
        assert result.session_id == "forked-sid"

    def test_always_approve_is_the_noninteractive_opt_in(self) -> None:
        assert ALWAYS_APPROVE_FLAG == "--always-approve"

    def test_prompt_flag_is_the_attached_form(self) -> None:
        assert PROMPT_FLAG == "--single"

    def test_hyphen_leading_prompt_is_attached_to_its_flag(
        self, tmp_path: Path, monkeypatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A bare `--fix ...` argument is read by grok's parser as a flag.

        `talk` builds the prompt from argparse.REMAINDER, so a prompt opening
        with a dash reaches the CLI verbatim and the run dies at argv parsing
        with a usage dump instead of answering.
        """
        backend = GrokBackend()
        prompt = "--fix the parser"

        def fake_run(args, **kwargs):
            assert prompt not in args
            assert args[-1] == f"--single={prompt}"
            _assert_subprocess_kwargs(kwargs, tmp_path)
            payload = json.dumps({"sessionId": "s1", "text": "ok"})
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with caplog.at_level("DEBUG", logger="backends"):
            result = backend.run_turn(prompt, None, tmp_path)
        assert result.reply == "ok"
        # The flag stays readable; only the prompt is collapsed.
        assert _messages(caplog)[1] == (
            "grok turn: invoking "
            "grok --output-format json --always-approve --single=<prompt:16chars>"
        )

    def test_null_text_is_an_empty_reply_not_a_crash(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """An explicit null must not reach len() in the debug log."""
        backend = GrokBackend()
        payload = json.dumps({"sessionId": "s1", "text": None})

        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = backend.run_turn("x", None, tmp_path)
        assert result.reply == ""
        assert result.session_id == "s1"

    def test_reply_comes_from_text_not_result(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Claude's field names must not be read as Grok's envelope."""
        backend = GrokBackend()
        payload = json.dumps(
            {
                "sessionId": "s1",
                "text": "grok-reply",
                "result": "claude-field",
                "session_id": "wrong",
            }
        )

        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = backend.run_turn("x", None, tmp_path)
        assert result.session_id == "s1"
        assert result.reply == "grok-reply"

    def test_claude_shaped_envelope_is_not_a_session(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        backend = GrokBackend()
        payload = json.dumps({"session_id": "s1", "result": "hi"})

        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(GrokTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == (
            f"grok did not report a sessionId\nstdout: {payload}"
        )

    def test_error_envelope_raises(self, tmp_path: Path, monkeypatch) -> None:
        backend = GrokBackend()

        def fake_run(args, **kwargs):
            _assert_subprocess_kwargs(kwargs, tmp_path)
            payload = json.dumps(
                {"type": "error", "message": "boom", "sessionId": "s1"}
            )
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(GrokTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == "grok reported an error: boom"

    def test_nonzero_exit_prefers_the_json_error_message(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        backend = GrokBackend()
        payload = json.dumps({"type": "error", "message": "auth failed"})

        def fake_run(args, **kwargs):
            _assert_subprocess_kwargs(kwargs, tmp_path)
            return subprocess.CompletedProcess(
                args, 1, stdout=payload, stderr="ignored"
            )

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(GrokTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == "grok reported an error: auth failed"

    def test_nonzero_exit_without_error_object_uses_stderr(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        backend = GrokBackend()
        stderr = "s" * 500 + "M" + "e" * 1999  # 2500 chars total

        def fake_run(args, **kwargs):
            _assert_subprocess_kwargs(kwargs, tmp_path)
            return subprocess.CompletedProcess(args, 1, stdout="", stderr=stderr)

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(GrokTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == f"grok exited 1\nstderr: {stderr[-2000:]}"

    def test_nonzero_exit_with_a_success_envelope_still_fails(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Exit 0 is part of success; a payload is not enough on its own."""
        backend = GrokBackend()
        payload = json.dumps({"sessionId": "s1", "text": "hi"})
        stderr = "e" * 2500

        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 2, stdout=payload, stderr=stderr)

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(GrokTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == f"grok exited 2\nstderr: {stderr[-2000:]}"

    def test_text_defaults_to_empty_reply(self, tmp_path: Path, monkeypatch) -> None:
        backend = GrokBackend()

        def fake_run(args, **kwargs):
            payload = json.dumps({"sessionId": "s1"})
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = backend.run_turn("x", None, tmp_path)
        assert result.reply == ""

    def test_malformed_json_raises(self, tmp_path: Path, monkeypatch) -> None:
        backend = GrokBackend()
        stdout = "s" * 500 + "M" + "e" * 1999  # 2500 chars total, not valid JSON

        def fake_run(args, **kwargs):
            _assert_subprocess_kwargs(kwargs, tmp_path)
            return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(GrokTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == (
            f"grok output was not JSON\nstdout: {stdout[:400]}\n…\n{stdout[-1600:]}"
        )
        assert stdout[:400] in str(excinfo.value)
        assert stdout[-1600:] in str(excinfo.value)

    def test_text_prefix_then_json_envelope_is_parsed(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        backend = GrokBackend()
        envelope = {"sessionId": "s1", "text": "I've read both skills"}
        stdout = "I've read both skills\n" + json.dumps(envelope)

        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = backend.run_turn("x", None, tmp_path)
        assert result.session_id == "s1"
        assert result.reply == "I've read both skills"

    def test_another_type_is_still_a_normal_reply(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        backend = GrokBackend()
        payload = json.dumps(
            {"type": "end", "sessionId": "s1", "text": "done", "stopReason": "end_turn"}
        )

        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert backend.run_turn("x", None, tmp_path).reply == "done"

    def test_missing_session_id_raises_rather_than_returning_none(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        backend = GrokBackend()
        payload = json.dumps({"text": "hi"})

        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(GrokTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == (
            f"grok did not report a sessionId\nstdout: {payload}"
        )

    def test_blank_session_id_raises(self, tmp_path: Path, monkeypatch) -> None:
        backend = GrokBackend()
        payload = json.dumps({"sessionId": "", "text": "hi"})

        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(GrokTurnError, match="did not report a sessionId"):
            backend.run_turn("x", None, tmp_path)

    def test_non_string_session_id_raises(self, tmp_path: Path, monkeypatch) -> None:
        backend = GrokBackend()
        payload = json.dumps({"sessionId": 17, "text": "hi"})

        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(GrokTurnError, match="did not report a sessionId"):
            backend.run_turn("x", None, tmp_path)

    def test_schema_is_passed_inline_as_its_own_argument(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Its own argument, not glued on like --single=: the value starts with
        `{`, which no parser reads as a flag."""
        backend = GrokBackend()
        # As with claude: the text is not the object, so this can only pass by
        # reading grok's own parse.
        payload = json.dumps(
            {
                "sessionId": "s1",
                "text": "see structuredOutput",
                "structuredOutput": {"verdict": "pass"},
            }
        )

        def fake_run(args, **kwargs):
            assert args == [
                "grok",
                "--output-format",
                "json",
                ALWAYS_APPROVE_FLAG,
                GROK_SCHEMA_FLAG,
                SCHEMA.text,
                "--single=hello",
            ]
            _assert_subprocess_kwargs(kwargs, tmp_path)
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = backend.run_turn("hello", None, tmp_path, schema=SCHEMA)
        assert result.structured == {"verdict": "pass"}

    def test_structured_falls_back_to_parsing_the_text(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = GrokBackend()
        payload = json.dumps({"sessionId": "s1", "text": '{"verdict":"pass"}'})
        monkeypatch.setattr(subprocess, "run", _completed(0, payload, ""))
        result = backend.run_turn("hello", None, tmp_path, schema=SCHEMA)
        assert result.structured == {"verdict": "pass"}

    def test_no_schema_omits_the_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = GrokBackend()
        payload = json.dumps({"sessionId": "s1", "text": '{"verdict":"pass"}'})

        def fake_run(args, **kwargs):
            assert GROK_SCHEMA_FLAG not in args
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert backend.run_turn("hello", None, tmp_path).structured is None


class TestParseGrokStdout:
    def test_plain_object(self) -> None:
        payload = {"sessionId": "s1", "text": "hi"}
        assert parse_grok_stdout(json.dumps(payload)) == payload

    def test_empty_is_an_error(self) -> None:
        with pytest.raises(GrokTurnError) as excinfo:
            parse_grok_stdout("   ")
        assert str(excinfo.value) == "grok output was not JSON\nstdout: "

    def test_picks_the_session_envelope(self) -> None:
        noise = json.dumps({"progress": 1})
        envelope = json.dumps({"sessionId": "keep", "text": "done"})
        parsed = parse_grok_stdout(f"{noise}\n{envelope}")
        assert parsed["sessionId"] == "keep"
        assert parsed["text"] == "done"

    def test_picks_the_session_envelope_ahead_of_later_noise(self) -> None:
        """Last-object fallback would take the trailing progress event."""
        envelope = json.dumps({"sessionId": "keep", "text": "done"})
        later = json.dumps({"progress": 1})
        parsed = parse_grok_stdout(f"{envelope}\n{later}")
        assert parsed["sessionId"] == "keep"

    def test_picks_an_error_object(self) -> None:
        noise = json.dumps({"progress": 1})
        error = json.dumps({"type": "error", "message": "nope"})
        parsed = parse_grok_stdout(f"{noise}\n{error}")
        assert parsed == {"type": "error", "message": "nope"}

    def test_picks_an_error_object_ahead_of_later_noise(self) -> None:
        error = json.dumps({"type": "error", "message": "nope"})
        later = json.dumps({"progress": 1})
        parsed = parse_grok_stdout(f"{error}\n{later}")
        assert parsed == {"type": "error", "message": "nope"}

    def test_picks_a_text_only_object_ahead_of_later_noise(self) -> None:
        envelope = json.dumps({"text": "done"})
        later = json.dumps({"progress": 1})
        parsed = parse_grok_stdout(f"{envelope}\n{later}")
        assert parsed == {"text": "done"}

    def test_picks_a_session_id_only_object_ahead_of_later_noise(self) -> None:
        """A success envelope can omit text; sessionId alone must still win."""
        envelope = json.dumps({"sessionId": "keep"})
        later = json.dumps({"progress": 1})
        parsed = parse_grok_stdout(f"{envelope}\n{later}")
        assert parsed == {"sessionId": "keep"}

    def test_a_trailing_tip_does_not_outrank_the_real_envelope(self) -> None:
        """`text` alone is a weak marker: tips and warnings carry one too."""
        envelope = json.dumps({"sessionId": "keep", "text": "done"})
        tip = json.dumps({"type": "tip", "text": "try --help"})
        assert parse_grok_stdout(f"{envelope}\n{tip}") == {
            "sessionId": "keep",
            "text": "done",
        }

    def test_a_trailing_tip_does_not_outrank_an_error_envelope(self) -> None:
        error = json.dumps({"type": "error", "message": "nope"})
        tip = json.dumps({"type": "tip", "text": "try --help"})
        assert parse_grok_stdout(f"{error}\n{tip}") == {
            "type": "error",
            "message": "nope",
        }

    def test_falls_back_to_the_last_object(self) -> None:
        parsed = parse_grok_stdout('noise {"other": 1} then {"later": 2}')
        assert parsed == {"later": 2}

    def test_short_garbage_keeps_the_whole_dump(self) -> None:
        with pytest.raises(GrokTurnError) as excinfo:
            parse_grok_stdout("not json")
        assert str(excinfo.value) == "grok output was not JSON\nstdout: not json"

    def test_objects_inside_a_json_array_are_still_found(self) -> None:
        parsed = parse_grok_stdout('[{"a": 1}, {"sessionId": "s", "text": "hi"}]')
        assert parsed == {"sessionId": "s", "text": "hi"}

    def test_skips_a_broken_brace_and_keeps_the_next_object(self) -> None:
        parsed = parse_grok_stdout('{{"sessionId": "s"}')
        assert parsed == {"sessionId": "s"}
