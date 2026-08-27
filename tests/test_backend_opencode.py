"""Unit tests for agent backend interface, implementations, and orchestrator."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from backends.base import (
    TurnResult,
)
from backends.opencode import FORK_FLAG as OPENCODE_FORK_FLAG
from backends.opencode import OpenCodeBackend, OpenCodeTurnError
from backends.opencode import (
    format_event as opencode_format_event,
)
from tests.backend_helpers import (
    SCHEMA,
    _assert_subprocess_kwargs,
    _completed,
    _messages,
    _reported_seconds,
)


class TestOpenCodeRunTurn:
    def test_streaming_turn_passes_formatter_and_keeps_opencode_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = OpenCodeBackend()
        stdout = "\n".join(
            [
                json.dumps({"type": "step_start", "sessionID": "s1"}),
                json.dumps(
                    {
                        "type": "text",
                        "sessionID": "s1",
                        "part": {"id": "p1", "text": "done"},
                    }
                ),
            ]
        )

        def fake_run(name, args, **kwargs):
            assert name == "opencode"
            assert args == [
                "opencode",
                "run",
                "--format",
                "json",
                "--auto",
                "--dir",
                str(tmp_path),
            ]
            assert kwargs["stream"] is True
            assert kwargs["format_event"] is opencode_format_event
            return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

        monkeypatch.setattr("backends.opencode.run_cli_turn", fake_run)
        result = backend.run_turn("work", None, tmp_path, stream=True)

        assert result == TurnResult("s1", "done", stdout)

    def test_fresh_turn_uses_exact_argv_and_verbatim_input(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        backend = OpenCodeBackend(model="provider/model", reasoning_effort="high")
        prompt = '- leading "quote"\\path\nsecond line — café'
        stdout = "\n".join(
            [
                json.dumps({"type": "step-start", "sessionID": "s1"}),
                json.dumps(
                    {
                        "type": "text",
                        "sessionID": "s1",
                        "part": {"id": "p1", "text": "done"},
                    }
                ),
            ]
        )
        calls: list[tuple[list[str], dict]] = []

        def fake_run(args, **kwargs):
            calls.append((args, kwargs))
            return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with caplog.at_level("DEBUG", logger="backends"):
            result = backend.run_turn(prompt, None, tmp_path)

        assert calls[0][0] == [
            "opencode",
            "run",
            "--format",
            "json",
            "--auto",
            "--dir",
            str(tmp_path),
            "--model",
            "provider/model",
            "--variant",
            "high",
        ]
        assert prompt not in calls[0][0]
        _assert_subprocess_kwargs(
            calls[0][1], tmp_path, expected_stdin=None, expected_input=prompt
        )
        assert result == TurnResult("s1", "done", stdout)
        messages = _messages(caplog)
        assert messages[0] == (
            f"opencode turn: cwd={tmp_path} resume=False prompt_chars={len(prompt)} "
            "timeout=3600s"
        )
        assert messages[1] == (
            "opencode turn: invoking opencode run --format json --auto --dir "
            f"{tmp_path} --model provider/model --variant high"
        )
        assert (
            _reported_seconds(
                messages[2],
                rf"opencode turn: exited 0 after (\d+\.\d)s with {len(stdout)} "
                rf"chars of stdout",
            )
            < 60
        )
        assert messages[3] == "opencode turn: parsed session=s1 parts=1 reply_chars=4"

    def test_resumed_turn_adds_session_flag(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        backend = OpenCodeBackend()
        stdout = json.dumps({"type": "step-finish", "sessionID": "s1"})

        def fake_run(args, **kwargs):
            assert args == [
                "opencode",
                "run",
                "--format",
                "json",
                "--auto",
                "--dir",
                str(tmp_path),
                "--session",
                "s1",
            ]
            _assert_subprocess_kwargs(
                kwargs, tmp_path, expected_stdin=None, expected_input="again"
            )
            return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with caplog.at_level("DEBUG", logger="backends"):
            assert backend.run_turn("again", "s1", tmp_path).session_id == "s1"
        assert _messages(caplog)[0] == (
            f"opencode turn: cwd={tmp_path} resume=True prompt_chars=5 timeout=3600s"
        )

    def test_forked_resume_adds_the_fork_flag_next_to_the_source_session(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`--fork` needs the source session on the same command line: it forks
        what `--session` names, and the new id is the one the events report."""
        backend = OpenCodeBackend()
        stdout = json.dumps({"type": "step-finish", "sessionID": "forked-sid"})

        def fake_run(args, **kwargs):
            assert args == [
                "opencode",
                "run",
                "--format",
                "json",
                "--auto",
                "--dir",
                str(tmp_path),
                "--session",
                "source-sid",
                OPENCODE_FORK_FLAG,
            ]
            _assert_subprocess_kwargs(
                kwargs, tmp_path, expected_stdin=None, expected_input="again"
            )
            return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = backend.run_turn("again", "source-sid", tmp_path, resume_as_fork=True)
        assert result.session_id == "forked-sid"

    def test_noisy_events_are_combined_and_structured_is_parsed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = OpenCodeBackend()
        reply = '{"verdict":\n"pass"}'
        stdout = "\nnot json\n" + "\n".join(
            [
                json.dumps({"type": "step-start", "sessionID": "s1"}),
                json.dumps(
                    {
                        "type": "text",
                        "sessionID": "s1",
                        "part": {"id": "a", "text": '{"verdict":'},
                    }
                ),
                json.dumps(
                    {
                        "type": "text",
                        "sessionID": "s1",
                        "part": {"id": "b", "text": '"pass"}'},
                    }
                ),
            ]
        )
        monkeypatch.setattr(subprocess, "run", _completed(0, stdout))
        result = backend.run_turn("x", None, tmp_path, schema=SCHEMA)
        assert result.session_id == "s1"
        assert result.reply == reply
        assert result.raw == stdout
        assert result.structured == {"verdict": "pass"}

    def test_duplicate_parts_keep_last_text_in_first_seen_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = OpenCodeBackend()
        events = [
            {"type": "text", "sessionID": "s1", "part": {"id": "a", "text": "old"}},
            {"type": "text", "sessionID": "s1", "part": {"id": "b", "text": "two"}},
            {"type": "text", "sessionID": "s1", "part": {"id": "a", "text": "new"}},
        ]
        stdout = "\n".join(json.dumps(event) for event in events)
        monkeypatch.setattr(subprocess, "run", _completed(0, stdout))
        assert backend.run_turn("x", None, tmp_path).reply == "new\ntwo"

    @pytest.mark.parametrize("session_id", [None, "", " \t\n", 17])
    def test_invalid_session_id_raises(
        self,
        session_id: object,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        backend = OpenCodeBackend()
        payload = {"type": "step-start"}
        if session_id is not None:
            payload["sessionID"] = session_id
        stdout = json.dumps(payload)
        monkeypatch.setattr(subprocess, "run", _completed(0, stdout))
        with pytest.raises(OpenCodeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == (
            f"opencode did not report a sessionID\nstdout: {stdout}\nstderr: "
        )

    def test_error_event_on_zero_exit_raises(self, tmp_path: Path, monkeypatch) -> None:
        backend = OpenCodeBackend()
        stdout = json.dumps(
            {
                "type": "error",
                "sessionID": "s1",
                "error": {"data": {"message": "bad request"}},
            }
        )
        monkeypatch.setattr(subprocess, "run", _completed(0, stdout))
        with pytest.raises(OpenCodeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == "opencode reported an error: bad request"

    def test_nonzero_exit_prefers_data_message(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = OpenCodeBackend()
        stdout = json.dumps(
            {
                "type": "error",
                "sessionID": "s1",
                "error": {"name": "ProviderError", "data": {"message": "auth failed"}},
            }
        )
        monkeypatch.setattr(subprocess, "run", _completed(2, stdout, "ignored"))
        with pytest.raises(OpenCodeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == "opencode reported an error: auth failed"

    def test_error_detail_requires_string_message_before_using_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = OpenCodeBackend()
        stdout = json.dumps(
            {
                "type": "error",
                "sessionID": "s1",
                "error": {
                    "name": "ProviderError",
                    "data": {"message": 17},
                },
            }
        )
        monkeypatch.setattr(subprocess, "run", _completed(2, stdout))
        with pytest.raises(OpenCodeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == "opencode reported an error: ProviderError"

    def test_error_detail_requires_string_name_before_using_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = OpenCodeBackend()
        stdout = json.dumps(
            {
                "type": "error",
                "sessionID": "s1",
                "error": {"name": 17},
            }
        )
        stderr = "stderr detail"
        monkeypatch.setattr(subprocess, "run", _completed(2, stdout, stderr))
        with pytest.raises(OpenCodeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == f"opencode exited 2\nstderr: {stderr}"

    def test_nonzero_exit_uses_last_error_event(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = OpenCodeBackend()
        stdout = "\n".join(
            [
                json.dumps(
                    {
                        "type": "error",
                        "sessionID": "s1",
                        "error": {"data": {"message": "old"}},
                    }
                ),
                json.dumps(
                    {
                        "type": "error",
                        "sessionID": "s1",
                        "error": {"data": {"message": "latest"}},
                    }
                ),
                json.dumps({"type": "step-start", "sessionID": "s1"}),
            ]
        )
        monkeypatch.setattr(subprocess, "run", _completed(2, stdout))
        with pytest.raises(OpenCodeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == "opencode reported an error: latest"

    def test_nonzero_exit_uses_error_name_when_message_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = OpenCodeBackend()
        stdout = json.dumps(
            {
                "type": "error",
                "sessionID": "s1",
                "error": {"name": "ProviderError", "data": {}},
            }
        )
        monkeypatch.setattr(subprocess, "run", _completed(2, stdout, "ignored"))
        with pytest.raises(OpenCodeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == "opencode reported an error: ProviderError"

    def test_nonzero_exit_falls_back_to_stderr_tail(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = OpenCodeBackend()
        stderr = "s" * 2500
        monkeypatch.setattr(subprocess, "run", _completed(3, "not json", stderr))
        with pytest.raises(OpenCodeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == f"opencode exited 3\nstderr: {stderr[-2000:]}"

    def test_error_event_without_detail_on_zero_exit_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = OpenCodeBackend()
        stdout = "\n".join(
            [
                json.dumps({"type": "error", "sessionID": "s1", "error": "bad"}),
                json.dumps({"type": "error", "sessionID": "s1", "error": {}}),
                json.dumps({"type": "step-start", "sessionID": "s1"}),
            ]
        )
        monkeypatch.setattr(subprocess, "run", _completed(0, stdout))
        with pytest.raises(OpenCodeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == "opencode reported an error event"

    def test_non_dict_and_empty_stream_events_are_ignored_or_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = OpenCodeBackend()
        stdout = '["progress"]\n{"type":"step-start","sessionID":"s1"}'
        monkeypatch.setattr(subprocess, "run", _completed(0, stdout))
        assert backend.run_turn("x", None, tmp_path).session_id == "s1"

        monkeypatch.setattr(subprocess, "run", _completed(0, "not json"))
        with pytest.raises(OpenCodeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == (
            "opencode did not report a sessionID\nstdout: not json\nstderr: "
        )

    def test_malformed_text_part_is_ignored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = OpenCodeBackend()
        stdout = "\n".join(
            [
                json.dumps({"type": "step-start", "sessionID": "s1"}),
                json.dumps({"type": "text", "sessionID": "s1", "part": None}),
                json.dumps(
                    {
                        "type": "text",
                        "sessionID": "s1",
                        "part": {"id": "a", "text": "one"},
                    }
                ),
                json.dumps(
                    {
                        "type": "text",
                        "sessionID": "s1",
                        "part": {"id": "missing"},
                    }
                ),
                json.dumps(
                    {
                        "type": "text",
                        "sessionID": "s1",
                        "part": {"id": "b", "text": "two"},
                    }
                ),
            ]
        )
        monkeypatch.setattr(subprocess, "run", _completed(0, stdout))
        assert backend.run_turn("x", None, tmp_path).reply == "one\ntwo"

    def test_missing_session_error_preserves_both_output_tails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = OpenCodeBackend()
        stdout = json.dumps({"type": "text"}) + "\n" + "o" * 2500
        stderr = "e" * 2500
        monkeypatch.setattr(subprocess, "run", _completed(0, stdout, stderr))
        with pytest.raises(OpenCodeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == (
            "opencode did not report a sessionID\n"
            f"stdout: {stdout[-2000:]}\nstderr: {stderr[-2000:]}"
        )

    def test_empty_event_stream_preserves_both_output_tails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = OpenCodeBackend()
        stdout = "o" * 2500
        stderr = "e" * 2500
        monkeypatch.setattr(subprocess, "run", _completed(0, stdout, stderr))
        with pytest.raises(OpenCodeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == (
            "opencode did not report a sessionID\n"
            f"stdout: {stdout[-2000:]}\nstderr: {stderr[-2000:]}"
        )

    def test_no_schema_does_not_parse_json_reply_into_structured_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = OpenCodeBackend()
        stdout = "\n".join(
            [
                json.dumps({"type": "step-start", "sessionID": "s1"}),
                json.dumps(
                    {
                        "type": "text",
                        "sessionID": "s1",
                        "part": {"id": "p1", "text": "{}"},
                    }
                ),
            ]
        )
        monkeypatch.setattr(subprocess, "run", _completed(0, stdout))
        assert backend.run_turn("x", None, tmp_path).structured is None

    def test_two_completed_blocks_are_separated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A turn that speaks either side of a tool call emits two completed
        parts. Concatenating them ran the sentences together in a live turn."""
        backend = OpenCodeBackend()
        events = [
            {
                "type": "text",
                "sessionID": "s1",
                "part": {"id": "a", "text": "I'll read marker.txt."},
            },
            {
                "type": "tool_use",
                "sessionID": "s1",
                "part": {"id": "t", "tool": "read"},
            },
            {"type": "text", "sessionID": "s1", "part": {"id": "b", "text": "rhubarb"}},
        ]
        stdout = "\n".join(json.dumps(event) for event in events)
        monkeypatch.setattr(subprocess, "run", _completed(0, stdout))
        assert backend.run_turn("x", None, tmp_path).reply == (
            "I'll read marker.txt.\nrhubarb"
        )

    def test_a_fenced_reply_still_yields_the_structured_object(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The measured 1.18.21 behaviour: asked for JSON, it answers in a
        ```json fence. `json.loads` of that whole reply fails, so without the
        scan every schema turn burns its retries and then fails."""
        backend = OpenCodeBackend()
        reply = '```json\n{"verdict":"pass"}\n```'
        stdout = json.dumps(
            {"type": "text", "sessionID": "s1", "part": {"id": "a", "text": reply}}
        )
        monkeypatch.setattr(subprocess, "run", _completed(0, stdout))
        result = backend.run_turn("x", None, tmp_path, schema=SCHEMA)
        assert result.reply == reply
        assert result.structured == {"verdict": "pass"}

    def test_an_object_introduced_by_prose_is_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = OpenCodeBackend()
        reply = 'Here is the result:\n{"verdict":"pass"}'
        stdout = json.dumps(
            {"type": "text", "sessionID": "s1", "part": {"id": "a", "text": reply}}
        )
        monkeypatch.setattr(subprocess, "run", _completed(0, stdout))
        assert backend.run_turn("x", None, tmp_path, schema=SCHEMA).structured == {
            "verdict": "pass"
        }

    def test_the_last_object_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A model that shows its working puts the answer last, and the schema
        check downstream is what decides whether it is the right one."""
        backend = OpenCodeBackend()
        reply = 'Not this: {"verdict":"draft"}\nFinal: {"verdict":"pass"}'
        stdout = json.dumps(
            {"type": "text", "sessionID": "s1", "part": {"id": "a", "text": reply}}
        )
        monkeypatch.setattr(subprocess, "run", _completed(0, stdout))
        assert backend.run_turn("x", None, tmp_path, schema=SCHEMA).structured == {
            "verdict": "pass"
        }

    def test_a_reply_with_no_object_is_none_not_an_exception(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Not an object is a contract failure the validator retries, not a
        parse error for the adapter to raise on."""
        backend = OpenCodeBackend()
        stdout = json.dumps(
            {
                "type": "text",
                "sessionID": "s1",
                "part": {"id": "a", "text": "Sure! Here you go:"},
            }
        )
        monkeypatch.setattr(subprocess, "run", _completed(0, stdout))
        assert backend.run_turn("x", None, tmp_path, schema=SCHEMA).structured is None

    def test_a_fenced_reply_stays_unparsed_without_a_schema(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The scan must not turn an ordinary reply that happens to quote JSON
        into a structured object nobody asked for."""
        backend = OpenCodeBackend()
        stdout = json.dumps(
            {
                "type": "text",
                "sessionID": "s1",
                "part": {"id": "a", "text": '```json\n{"verdict":"pass"}\n```'},
            }
        )
        monkeypatch.setattr(subprocess, "run", _completed(0, stdout))
        assert backend.run_turn("x", None, tmp_path).structured is None
