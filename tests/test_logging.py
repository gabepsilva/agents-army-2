"""Unit tests for agent backend interface, implementations, and orchestrator."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

import orchestrator.cli as cli
import orchestrator.core as core
from orchestrator.cli import main
from orchestrator.core import (
    Orchestrator,
)
from tests.backend_helpers import (
    _messages,
    _reported_seconds,
)
from tests.path_helpers import runtime_paths


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
