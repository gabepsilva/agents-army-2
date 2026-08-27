from __future__ import annotations

import logging
from pathlib import Path

import pytest

import orchestrator
from backends.base import AgentBackend, TurnResult
from backends.registry import register_backend


class RecordingBackend(AgentBackend):
    @property
    def name(self) -> str:
        return "recording"

    def run_turn(  # noqa: PLR0913 - test double mirrors AgentBackend interface
        self,
        prompt: str,
        session_id: str | None,
        cwd: Path,
        timeout: int = orchestrator.DEFAULT_TURN_TIMEOUT,
        schema=None,
        *,
        resume_as_fork: bool = False,
        stream: bool = False,
    ) -> TurnResult:
        return TurnResult(session_id="sid", reply=f"reply:{prompt}", raw="")


@pytest.fixture(autouse=True)
def recording_backend() -> None:
    register_backend("recording", RecordingBackend)


def test_prompt_flag_and_separator_forward_identical_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(orchestrator, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(orchestrator, "DEFAULT_BACKEND", "recording")

    orchestrator.main(["talk", "a", "-p", "same prompt"])
    flag_output = capsys.readouterr().out
    orchestrator.main(["talk", "a", "--", "same", "prompt"])
    tail_output = capsys.readouterr().out
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("same prompt", encoding="utf-8")
    orchestrator.main(["talk", "a", "--prompt-file", str(prompt_file)])
    file_output = capsys.readouterr().out

    assert flag_output.endswith("reply:same prompt\n")
    assert tail_output.endswith("reply:same prompt\n")
    assert file_output.endswith("reply:same prompt\n")


def test_chat_help_exposes_only_the_interactive_agent_selection(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        orchestrator.main(["chat", "--help"])

    assert excinfo.value.code == 0
    output = capsys.readouterr().out
    assert "usage: orchestrator chat" in output
    assert "--team TEAM" in output
    assert "--schema" not in output
    assert "--skill" not in output
    assert "--timeout" not in output


def test_talk_forwards_schema_retries_timeout_stream_and_short_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeAgent:
        name = "a"
        backend = type(
            "Backend",
            (),
            {"name": "recording", "model": "model", "reasoning_effort": "high"},
        )()

    class FakeOrchestrator:
        def __init__(self) -> None:
            self.calls: list[tuple] = []

        def ensure(self, *args, **kwargs):
            return FakeAgent(), False

        def talk(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return TurnResult(
                session_id="sid",
                reply="raw",
                raw="",
                structured={"ok": True},
            )

    schema_path = tmp_path / "out.json"
    schema_path.write_text(
        '{"type":"object","additionalProperties":false,"properties":{},"required":[]}',
        encoding="utf-8",
    )
    fake = FakeOrchestrator()
    monkeypatch.setattr(orchestrator, "Orchestrator", lambda *_: fake)
    monkeypatch.setattr(orchestrator, "SKILLS_DIR", tmp_path / "skills")

    orchestrator.main(
        [
            "talk",
            "a",
            "-b",
            "recording",
            "-m",
            "model",
            "-e",
            "high",
            "--schema",
            str(schema_path),
            "--retries",
            "2",
            "--timeout",
            "30",
            "--stream",
            "-p",
            "question",
        ]
    )

    assert fake.calls[0][0] == ("a", "question")
    assert fake.calls[0][1]["schema"].path == schema_path.resolve()
    assert fake.calls[0][1]["retries"] == 2
    assert fake.calls[0][1]["timeout"] == 30
    assert fake.calls[0][1]["stream"] is True
    assert capsys.readouterr().out.endswith('{\n  "ok": true\n}\n')


def test_stream_flag_defaults_off_and_describes_its_stderr_contract(capsys) -> None:
    parser = orchestrator._build_parser()
    defaults = parser.parse_args(["talk", "a", "-p", "x"])
    assert defaults.stream is False

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["talk", "--help"])

    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out
    assert "--stream" in help_text
    assert "echo complete CLI output lines to stderr while the" in help_text
    assert "turn runs" in help_text


@pytest.mark.parametrize(
    "argv",
    [
        ["talk", "a", "hi", "--timeout", "5"],
        ["talk", "a"],
        ["talk", "a", "--"],
        ["talk", "a", "-p", " \t"],
        ["talk", "a", "-p", "one", "--", "two"],
        ["talk", "a", "-p", "x", "--prompt-file", "unused.txt"],
        ["talk", "a", "--prompt-file", "unused.txt", "--", "two"],
        ["talk", "a", "-p", "x", "--prompt-file", "unused.txt", "--", "two"],
        ["create", "a", "--", "foo"],
        ["fork", "a", "b", "--", "foo"],
        ["fork", "a"],
        ["list", "--", "foo"],
        ["delete", "a", "--", "foo"],
        ["doctor", "--", "foo"],
        ["doctor", "ignored"],
        ["--agent", "a"],
        ["--list", "agents"],
        ["--validate-schema", "out.json"],
        ["--validation-retries", "2"],
        ["--dependency-check"],
        ["--verbose2", "list"],
    ],
)
def test_invalid_prompt_or_separator_does_not_construct_orchestrator(
    argv: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_: object) -> None:
        raise AssertionError("invalid CLI input constructed Orchestrator")

    monkeypatch.setattr(orchestrator, "Orchestrator", fail)
    with pytest.raises(SystemExit) as excinfo:
        orchestrator.main(argv)
    assert excinfo.value.code == 2


def test_prompt_file_conflicts_are_rejected_before_constructing_orchestrator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("prompt", encoding="utf-8")

    def fail(*_: object) -> None:
        raise AssertionError("invalid CLI input constructed Orchestrator")

    monkeypatch.setattr(orchestrator, "Orchestrator", fail)
    for argv in (
        ["talk", "a", "-p", "one", "--prompt-file", str(prompt_file)],
        ["talk", "a", "--prompt-file", str(prompt_file), "--", "two"],
        [
            "talk",
            "a",
            "-p",
            "one",
            "--prompt-file",
            str(prompt_file),
            "--",
            "two",
        ],
    ):
        with pytest.raises(SystemExit) as excinfo:
            orchestrator.main(argv)
        assert excinfo.value.code == 2
        assert (
            "orchestrator talk: error: talk requires exactly one prompt source"
            in capsys.readouterr().err
        )


# Each verb, with the minimum arguments it needs to get past its own required
# positionals — otherwise argparse reports the missing name instead of the
# leftover flag, and the flag's absence goes unproven.
VERB_INVOCATIONS = (
    ("create", ["create", "a"]),
    ("talk", ["talk", "a", "-p", "hi"]),
    ("chat", ["chat", "a"]),
    ("fork", ["fork", "a", "b"]),
    ("list", ["list"]),
    ("delete", ["delete", "a"]),
    ("doctor", ["doctor"]),
)
VERBS = tuple(verb for verb, _ in VERB_INVOCATIONS)


@pytest.mark.parametrize(
    "argv",
    [
        ["--version"],
        ["--version", "ignored"],
        ["-v", "--version"],
    ],
)
def test_version_exits_before_constructing_orchestrator(
    argv: list[str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        orchestrator,
        "Orchestrator",
        lambda *_: (_ for _ in ()).throw(AssertionError("constructed")),
    )
    with pytest.raises(SystemExit) as excinfo:
        orchestrator.main(argv)
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert captured.out == "0.1.0\n"
    assert captured.err == ""


@pytest.mark.parametrize(("verb", "argv"), VERB_INVOCATIONS)
def test_version_after_a_verb_is_an_unrecognized_argument(
    verb: str, argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        orchestrator.main([*argv, "--version"])
    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"usage: orchestrator {verb}" in captured.err
    assert "unrecognized arguments: --version" in captured.err


@pytest.mark.parametrize("verb", VERBS)
def test_verb_help_does_not_offer_version(
    verb: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        orchestrator.main([verb, "--help"])
    assert excinfo.value.code == 0
    assert "--version" not in capsys.readouterr().out


def test_version_after_the_separator_stays_prompt_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(orchestrator, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(orchestrator, "DEFAULT_BACKEND", "recording")

    orchestrator.main(["talk", "a", "--", "text", "with", "--version", "in", "it"])

    out = capsys.readouterr().out
    assert out.endswith("reply:text with --version in it\n")
    assert "0.1.0" not in out


def test_doctor_ignores_corrupt_state_without_constructing_orchestrator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "orchestrator_state.json").write_text("{", encoding="utf-8")
    called = False

    def report() -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(
        orchestrator, "STATE_FILE", tmp_path / "orchestrator_state.json"
    )
    monkeypatch.setattr(orchestrator, "_print_dependency_check", report)
    monkeypatch.setattr(
        orchestrator,
        "Orchestrator",
        lambda *_: (_ for _ in ()).throw(AssertionError("constructed")),
    )

    orchestrator.main(["-v", "doctor"])
    assert called is True


def test_missing_skills_directory_has_one_stderr_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(orchestrator, "SKILLS_DIR", tmp_path / "missing")
    monkeypatch.setattr(orchestrator, "STATE_FILE", tmp_path / "state.json")
    with pytest.raises(SystemExit) as excinfo:
        orchestrator.main(["list", "skills"])
    captured = capsys.readouterr()
    assert excinfo.value.code == 1
    assert captured.err == f"skills directory not found: {tmp_path / 'missing'}\n"
    assert captured.out == ""


@pytest.mark.parametrize(
    ("argv", "level"),
    [
        (["-v", "list"], logging.DEBUG),
        (["-vv", "list"], orchestrator.TRACE),
        (["-vvv", "list"], orchestrator.TRACE),
        (["-v", "talk", "-v", "a", "-p", "x"], orchestrator.TRACE),
    ],
)
def test_verbosity_counts_before_and_after_verb(
    argv: list[str], level: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeOrchestrator:
        state_file = Path("/dev/null")

        def __init__(self, runtime_paths=None) -> None:
            self.runtime_paths = runtime_paths

        def list_agents(self) -> list[str]:
            return []

        def ensure(self, *args, **kwargs):
            return type(
                "Agent",
                (),
                {
                    "backend": type(
                        "Backend",
                        (),
                        {
                            "name": "recording",
                            "model": None,
                            "reasoning_effort": None,
                        },
                    )()
                },
            )(), False

        def talk(self, *args, **kwargs) -> TurnResult:
            return TurnResult(session_id="sid", reply="reply", raw="")

    monkeypatch.setattr(orchestrator, "Orchestrator", FakeOrchestrator)
    for logger_name in orchestrator.OWN_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.NOTSET)
    orchestrator.main(argv)
    assert logging.getLogger("orchestrator").level == level
    assert logging.getLogger("backends").level == level


@pytest.mark.parametrize(("verb", "argv"), VERB_INVOCATIONS)
def test_every_verb_accepts_verbosity_after_the_verb(
    verb: str, argv: list[str]
) -> None:
    opts = orchestrator._build_parser().parse_args([*argv, "-vv"])

    assert opts.verbosity_after == 2
    assert opts._parser.prog == f"orchestrator {verb}"
