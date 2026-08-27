"""Unit tests for agent backend interface, implementations, and orchestrator."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from backends.base import (
    DEFAULT_TURN_TIMEOUT,
    AgentBackend,
    OutputSchema,
    TurnError,
    TurnResult,
    describe_command,
    json_objects,
    reply_text,
    run_cli_turn,
    stdout_for_error,
    structured_reply,
)
from backends.claude import (
    ClaudeBackend,
    ClaudeTurnError,
)
from backends.codex import (
    CodexBackend,
    CodexTurnError,
)
from backends.grok import (
    GrokBackend,
    GrokTurnError,
)
from backends.opencode import OpenCodeBackend, OpenCodeTurnError
from backends.registry import (
    get_backend,
    list_backends,
    register_backend,
)
from tests.backend_helpers import (
    SCHEMA,
    EchoBackend,
    _assert_subprocess_kwargs,
    _completed,
    _messages,
    _reported_seconds,
    _subprocess_recorder,
)


@dataclass(frozen=True)
class BackendRow:
    """One shipped adapter's enrollment in the shared backend contracts.

    Every expectation below is a literal written here, never a value read off
    the class under test: a row that asks the object what it declares would
    pass against a corrupted declaration and assert nothing.
    """

    module: str
    backend_cls: type[AgentBackend]
    expected_name: str
    expected_enforces_schema: bool
    expected_supports_fork: bool
    expected_error: type[TurnError]
    # The smallest stdout that reaches this adapter's normal result path,
    # in that CLI's own envelope dialect. Each was checked against the real
    # parser rather than assumed portable between them.
    stdout: str
    # OpenCode is the one intended divergence: it takes its prompt on stdin
    # because it joins positional arguments before sending them to the model.
    prompt_on_stdin: bool = False


BACKENDS = [
    BackendRow(
        module="claude",
        backend_cls=ClaudeBackend,
        expected_name="claude",
        expected_enforces_schema=True,
        expected_supports_fork=True,
        expected_error=ClaudeTurnError,
        stdout='{"session_id": "s1", "result": "ok"}',
    ),
    BackendRow(
        module="codex",
        backend_cls=CodexBackend,
        expected_name="codex",
        expected_enforces_schema=True,
        expected_supports_fork=True,
        expected_error=CodexTurnError,
        stdout='{"type": "thread.started", "thread_id": "s1"}',
    ),
    BackendRow(
        module="grok",
        backend_cls=GrokBackend,
        expected_name="grok",
        expected_enforces_schema=True,
        expected_supports_fork=True,
        expected_error=GrokTurnError,
        stdout='{"sessionId": "s1", "text": "ok"}',
    ),
    BackendRow(
        module="opencode",
        backend_cls=OpenCodeBackend,
        expected_name="opencode",
        expected_enforces_schema=False,
        expected_supports_fork=True,
        expected_error=OpenCodeTurnError,
        stdout='{"type": "text", "sessionID": "s1", "part": {"id": "p", "text": "ok"}}',
        prompt_on_stdin=True,
    ),
]

# pytest ids come from the row's own module name, so a failure names the
# backend rather than "backends2".
BACKEND_ROWS = pytest.mark.parametrize(
    "row", BACKENDS, ids=[row.module for row in BACKENDS]
)

# Distinctive enough that finding it in argv or a log line is never a
# coincidence.
PROMPT = "sequoia rutabaga"


class TestSharedSubprocessBoundary:
    """The contract every shipped adapter owes the operating system.

    One row per backend, so a fifth adapter inherits all of it by enrolling
    rather than by remembering to call a helper.
    """

    def _run(
        self,
        row: BackendRow,
        monkeypatch: pytest.MonkeyPatch,
        cwd: Path,
        *,
        returncode: int = 0,
        timeout: int | None = None,
    ) -> dict:
        """Drive the row's real ``run_turn`` and return the subprocess kwargs."""
        fake_run, calls = _subprocess_recorder(_completed(returncode, row.stdout))
        monkeypatch.setattr(subprocess, "run", fake_run)
        args = (PROMPT, None, cwd) if timeout is None else (PROMPT, None, cwd, timeout)
        row.backend_cls().run_turn(*args)
        return calls[0][1]

    @BACKEND_ROWS
    def test_runs_its_cli_under_the_shared_subprocess_discipline(
        self, row: BackendRow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        kwargs = self._run(row, monkeypatch, tmp_path)

        _assert_subprocess_kwargs(
            kwargs,
            tmp_path,
            expected_stdin=None if row.prompt_on_stdin else subprocess.DEVNULL,
            expected_input=PROMPT if row.prompt_on_stdin else None,
        )

    @BACKEND_ROWS
    def test_stdin_is_closed_unless_the_prompt_goes_there(
        self, row: BackendRow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A CLI left reading an inherited pipe blocks until it is killed."""
        kwargs = self._run(row, monkeypatch, tmp_path)

        if row.prompt_on_stdin:
            assert "stdin" not in kwargs
            assert kwargs["input"] == PROMPT
        else:
            assert kwargs["stdin"] == subprocess.DEVNULL
            assert "input" not in kwargs

    @BACKEND_ROWS
    def test_forwards_an_explicit_timeout_rather_than_the_default(
        self, row: BackendRow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The caller's budget, not a number each adapter picked for itself."""
        kwargs = self._run(row, monkeypatch, tmp_path, timeout=17)

        assert kwargs["timeout"] == 17

    # The prompt-redaction row is deliberately absent: the design's claim that
    # the `<prompt:Nchars>` form is emitted uniformly by all four adapters is
    # false, and opencode emits no placeholder at all. Blocked pending the
    # answer on PR #156 rather than worked around here.

    @BACKEND_ROWS
    def test_a_non_zero_exit_raises_that_backends_own_error(
        self, row: BackendRow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with pytest.raises(row.expected_error):
            self._run(row, monkeypatch, tmp_path, returncode=1)


class TestRunCliTurn:
    """The one place every backend hands its argv to the operating system."""

    def _record(self, monkeypatch: pytest.MonkeyPatch) -> list[tuple[list[str], dict]]:
        fake_run, calls = _subprocess_recorder(_completed(0, "out"))
        monkeypatch.setattr(subprocess, "run", fake_run)
        return calls

    def test_default_arm_closes_stdin_and_returns_the_completed_process(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = self._record(monkeypatch)

        proc = run_cli_turn(
            "demo",
            ["demo", "-p", "hello"],
            prompt="hello",
            session_id=None,
            cwd=tmp_path,
            timeout=DEFAULT_TURN_TIMEOUT,
        )

        assert calls[0][0] == ["demo", "-p", "hello"]
        _assert_subprocess_kwargs(calls[0][1], tmp_path)
        assert "input" not in calls[0][1]
        assert proc.returncode == 0
        assert proc.stdout == "out"

    def test_stdin_prompt_arm_sends_the_prompt_and_leaves_stdin_unset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = self._record(monkeypatch)

        run_cli_turn(
            "demo",
            ["demo"],
            prompt="hello",
            session_id=None,
            cwd=tmp_path,
            timeout=DEFAULT_TURN_TIMEOUT,
            prompt_on_stdin=True,
        )

        _assert_subprocess_kwargs(
            calls[0][1], tmp_path, expected_stdin=None, expected_input="hello"
        )
        assert "stdin" not in calls[0][1]

    def test_logs_name_cwd_resume_and_the_redacted_command(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        self._record(monkeypatch)

        with caplog.at_level("DEBUG"):
            run_cli_turn(
                "demo",
                ["demo", "-p", "hello"],
                prompt="hello",
                session_id="s1",
                cwd=tmp_path,
                timeout=42,
            )

        messages = _messages(caplog)
        assert messages[0] == (
            f"demo turn: cwd={tmp_path} resume=True prompt_chars=5 timeout=42s"
        )
        assert messages[1] == "demo turn: invoking demo -p <prompt:5chars>"
        assert (
            _reported_seconds(
                messages[2],
                r"demo turn: exited 0 after (\d+\.\d)s with 3 chars of stdout",
            )
            < 60
        )

    def test_a_fresh_session_reports_resume_false_and_the_real_exit_code(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(subprocess, "run", _completed(3, "abcd"))

        with caplog.at_level("DEBUG"):
            proc = run_cli_turn(
                "demo", ["demo"], prompt="", session_id=None, cwd=tmp_path, timeout=7
            )

        assert proc.returncode == 3
        messages = _messages(caplog)
        assert messages[0] == (
            f"demo turn: cwd={tmp_path} resume=False prompt_chars=0 timeout=7s"
        )
        assert messages[2].startswith("demo turn: exited 3 after ")
        assert messages[2].endswith("s with 4 chars of stdout")


class TestAgentBackendInterface:
    def test_claude_name(self) -> None:
        assert ClaudeBackend().name == "claude"

    def test_codex_name(self) -> None:
        assert CodexBackend().name == "codex"

    def test_grok_name(self) -> None:
        assert GrokBackend().name == "grok"

    def test_opencode_name(self) -> None:
        assert OpenCodeBackend().name == "opencode"
        assert "opencode" in list_backends()

    def test_backend_turn_errors_share_the_orchestrator_type(self) -> None:
        """cmd_talk catches TurnError, not a per-CLI tuple that grows."""
        assert issubclass(ClaudeTurnError, TurnError)
        assert issubclass(CodexTurnError, TurnError)
        assert issubclass(GrokTurnError, TurnError)
        assert issubclass(OpenCodeTurnError, TurnError)

    def test_schema_enforcement_defaults_and_opencode_override(self) -> None:
        assert ClaudeBackend.enforces_schema is True
        assert CodexBackend.enforces_schema is True
        assert GrokBackend.enforces_schema is True
        assert OpenCodeBackend.enforces_schema is False

    def test_fork_support_is_declared_per_backend(self) -> None:
        """The capability the CLI checks before it will fork an agent."""
        assert AgentBackend.supports_fork is False
        assert ClaudeBackend.supports_fork is True
        assert GrokBackend.supports_fork is True
        assert CodexBackend.supports_fork is True
        assert OpenCodeBackend.supports_fork is True

    def test_chat_support_is_declared_per_backend(self) -> None:
        """Interactive chat is opt-in, just like session forking."""
        assert AgentBackend.supports_chat is False
        assert ClaudeBackend.supports_chat is True
        assert CodexBackend.supports_chat is True
        assert GrokBackend.supports_chat is True
        assert OpenCodeBackend.supports_chat is True

    def test_backend_without_chat_support_has_no_interactive_command(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(NotImplementedError, match="no interactive chat command"):
            EchoBackend().chat_argv("session-1", tmp_path)

    @pytest.mark.parametrize(
        ("backend_cls", "expected"),
        [
            (ClaudeBackend, ["claude", "--resume", "session-1"]),
            (CodexBackend, ["codex", "resume", "session-1"]),
            (GrokBackend, ["grok", "--resume", "session-1"]),
            (OpenCodeBackend, ["opencode", "--session", "session-1"]),
        ],
    )
    def test_chat_argv_resumes_the_stored_session(
        self,
        backend_cls: type[AgentBackend],
        expected: list[str],
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        backend = backend_cls()
        with caplog.at_level(logging.DEBUG, logger=backend_cls.__module__):
            actual = backend.chat_argv("session-1", tmp_path)

        assert actual == expected
        assert [
            record.getMessage()
            for record in caplog.records
            if record.name == backend_cls.__module__
        ] == [f"{backend.name} chat: cwd={tmp_path} session=session-1"]

    def test_get_backend_resolves_grok(self) -> None:
        backend = get_backend("grok", model="grok-test", reasoning_effort="high")
        assert isinstance(backend, GrokBackend)
        assert backend.name == "grok"
        assert backend.model == "grok-test"
        assert backend.reasoning_effort == "high"

    def test_custom_backend_registration(self, tmp_path: Path) -> None:
        class CustomBackend(AgentBackend):
            @property
            def name(self) -> str:
                return "custom"

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

    def test_prompt_attached_to_its_flag_is_replaced(self) -> None:
        """Grok takes `--single=<prompt>`; the flag must stay readable."""
        args = ["grok", "--always-approve", "--single=hello"]
        assert describe_command(args, "hello") == (
            "grok --always-approve --single=<prompt:5chars>"
        )

    def test_attached_prompt_does_not_grow_the_line(self) -> None:
        short = describe_command(["grok", "--single=x"], "x")
        long = describe_command(["grok", f"--single={'x' * 10_000}"], "x" * 10_000)
        assert len(long) - len(short) == len("10000") - len("1")

    def test_attached_form_keeps_a_hyphen_leading_prompt_out_of_the_log(
        self,
    ) -> None:
        args = ["grok", "--single=--fix the parser"]
        assert describe_command(args, "--fix the parser") == (
            "grok --single=<prompt:16chars>"
        )

    def test_a_standalone_prompt_wins_over_an_earlier_attached_one(self) -> None:
        args = ["grok", "--single=hi", "hi"]
        assert describe_command(args, "hi") == "grok --single=hi <prompt:2chars>"

    def test_an_empty_prompt_matches_no_attached_flag(self) -> None:
        """Every argument ends with "=" + "", so the attached form must not
        fire and swallow the flag it is glued to."""
        assert describe_command(["grok", "--single="], "") == "grok --single="

    def test_unmatched_prompt_leaves_args_unchanged(self) -> None:
        assert describe_command(["claude", "-p", "hello"], "other") == "claude -p hello"

    def test_empty_args_render_as_empty(self) -> None:
        assert describe_command([], "hello") == ""

    def test_prompt_in_the_first_two_slots_is_still_replaced(self) -> None:
        assert describe_command(["hello"], "hello") == "<prompt:5chars>"
        assert describe_command(["codex", "hi"], "hi") == "codex <prompt:2chars>"


class TestReplyText:
    def test_returns_the_string(self) -> None:
        assert reply_text({"text": "hi"}, "text") == "hi"

    def test_missing_key_is_empty(self) -> None:
        assert reply_text({"sessionId": "s1"}, "text") == ""

    def test_explicit_null_is_empty(self) -> None:
        """`.get(key, "")` would hand None straight to len()."""
        assert reply_text({"text": None}, "text") == ""

    def test_non_string_is_empty(self) -> None:
        assert reply_text({"text": ["block"]}, "text") == ""

    def test_empty_string_is_kept(self) -> None:
        assert reply_text({"text": ""}, "text") == ""

    def test_reads_the_key_it_was_given(self) -> None:
        payload = {"text": "grok", "result": "claude"}
        assert reply_text(payload, "result") == "claude"


class TestJsonObjects:
    def test_finds_each_object_in_order(self) -> None:
        assert json_objects('{"a": 1} noise {"b": 2}') == [{"a": 1}, {"b": 2}]

    def test_empty_when_there_is_no_object(self) -> None:
        assert json_objects("not json") == []

    def test_skips_a_broken_brace(self) -> None:
        assert json_objects('{{"ok": true}') == [{"ok": True}]


class TestStructuredReply:
    """The one place backend envelopes turn into one object."""

    def test_no_schema_means_no_object_however_json_the_reply_is(self) -> None:
        assert structured_reply(None, '{"a":1}', {"a": 1}) is None

    def test_the_pre_parsed_field_wins(self) -> None:
        assert structured_reply(SCHEMA, "ignored", {"a": 1}) == {"a": 1}

    def test_falls_back_to_parsing_the_reply(self) -> None:
        assert structured_reply(SCHEMA, '{"a":1}') == {"a": 1}

    def test_a_non_dict_pre_parsed_field_is_not_trusted(self) -> None:
        assert structured_reply(SCHEMA, '{"a":1}', "not an object") == {"a": 1}

    def test_unparseable_reply_is_none_not_an_exception(self) -> None:
        assert structured_reply(SCHEMA, "Sure! Here you go:") is None

    def test_a_json_value_that_is_not_an_object_is_none(self) -> None:
        """A schema's reply is one object; a bare list or number is not it."""
        assert structured_reply(SCHEMA, "[1, 2]") is None


class TestStdoutForError:
    def test_keeps_a_short_dump(self) -> None:
        assert stdout_for_error("short") == "short"

    def test_keeps_exactly_2000_chars(self) -> None:
        text = "x" * 2000
        assert stdout_for_error(text) == text

    def test_splits_a_2001_char_dump(self) -> None:
        text = ("H" * 400) + "M" + ("T" * 1600)
        assert stdout_for_error(text) == f"{'H' * 400}\n…\n{'T' * 1600}"
