from __future__ import annotations

import ast
import importlib.metadata
import logging
from pathlib import Path
from types import ModuleType

import pytest

import orchestrator
import orchestrator.cli as cli
import orchestrator.core as core
import orchestrator.doctor as doctor
from backends.base import AgentBackend, TurnResult
from backends.registry import register_backend


class RecordingBackend(AgentBackend):
    @property
    def name(self) -> str:
        return "recording"

    def run_turn(  # noqa: PLR0913 - test doubles mirror AgentBackend.run_turn public seam
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
    monkeypatch.setenv("AGENTS_ARMY_STATE_FILE", str(tmp_path / "state.json"))
    monkeypatch.setattr(core, "DEFAULT_BACKEND", "recording")

    cli.main(["talk", "a", "-p", "same prompt"])
    flag_output = capsys.readouterr().out
    cli.main(["talk", "a", "--", "same", "prompt"])
    tail_output = capsys.readouterr().out
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("same prompt", encoding="utf-8")
    cli.main(["talk", "a", "--prompt-file", str(prompt_file)])
    file_output = capsys.readouterr().out

    assert flag_output.endswith("reply:same prompt\n")
    assert tail_output.endswith("reply:same prompt\n")
    assert file_output.endswith("reply:same prompt\n")


def test_chat_help_exposes_only_the_interactive_agent_selection(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["chat", "--help"])

    assert excinfo.value.code == 0
    output = capsys.readouterr().out
    assert "usage: orchestrator chat" in output
    assert "--team TEAM" in output
    assert "--schema" not in output
    assert "--skill" not in output
    assert "--timeout" not in output


def test_talk_help_describes_backend_event_streaming(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["talk", "--help"])

    assert excinfo.value.code == 0
    output = " ".join(capsys.readouterr().out.split())
    assert "render recognized backend events to stderr while the turn runs" in output


def test_package_and_console_entry_point_resolve_to_cli_main() -> None:
    import orchestrator.cli as cli

    assert orchestrator.main is cli.main
    console_scripts = {
        entry_point.name: entry_point
        for entry_point in importlib.metadata.entry_points(group="console_scripts")
    }
    # `aarmy` is an alias, not a rename: all three names stay installed and
    # load the same main.
    for name in ("agents-army", "orchestrator", "aarmy"):
        assert console_scripts[name].load() is cli.main


def test_package_surface_reexports_supported_objects_without_shrinking() -> None:
    import backends
    import backends.base as base
    import backends.registry as registry
    import orchestrator.cli as cli
    import orchestrator.paths as paths
    import orchestrator.schema as schema
    import orchestrator.skills as skills
    import orchestrator.teams as teams

    supported = {
        "Agent",
        "AgentBackend",
        "AgentBusyError",
        "AgentExistsError",
        "AgentNotFoundError",
        "DEFAULT_BACKEND",
        "DEFAULT_TURN_TIMEOUT",
        "DEFAULT_VALIDATION_RETRIES",
        "Orchestrator",
        "OrchestratorError",
        "OutputSchema",
        "SchemaError",
        "SchemaLoadError",
        "SkillError",
        "StateError",
        "TeamBusyError",
        "TRACE",
        "TurnError",
        "TurnResult",
        "UnknownBackendError",
        "cmd_chat",
        "cmd_create",
        "cmd_delete",
        "cmd_fork",
        "cmd_list",
        "cmd_talk",
        "compose_skill_prompt",
        "format_skill_listing",
        "get_backend",
        "index_skills",
        "list_backends",
        "load_schema",
        "main",
        "parse_skill_names",
        "paths",
        "resolve_catalog_dir",
        "resolve_skills",
        "schema",
        "skills",
        "teams",
    }
    assert supported <= set(orchestrator.__all__)
    assert orchestrator.__all__ == sorted(set(orchestrator.__all__))

    owners = (backends, base, cli, core, doctor, paths, registry, schema, skills, teams)
    module_objects = set(owners)
    missing = object()
    for name in orchestrator.__all__:
        value = getattr(orchestrator, name)
        if isinstance(value, ModuleType):
            assert value in module_objects
        else:
            assert any(getattr(owner, name, missing) is value for owner in owners)


def test_package_initializer_contains_only_imports_and_derived_all() -> None:
    package_file = Path(orchestrator.__file__)
    tree = ast.parse(package_file.read_text(encoding="utf-8"))

    assert len(package_file.read_text(encoding="utf-8").splitlines()) <= 80
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            assert isinstance(node.value.value, str)
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        assert isinstance(node, ast.Assign)
        assert len(node.targets) == 1
        assert isinstance(node.targets[0], ast.Name)
        assert node.targets[0].id == "__all__"


def test_talk_forwards_schema_retries_timeout_and_short_options(
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
    monkeypatch.setattr(core, "Orchestrator", lambda *_: fake)
    monkeypatch.setenv("AGENTS_ARMY_SKILLS", str(tmp_path / "skills"))

    cli.main(
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
            "-p",
            "question",
        ]
    )

    assert fake.calls[0][0] == ("a", "question")
    assert fake.calls[0][1]["schema"].path == schema_path.resolve()
    assert fake.calls[0][1]["retries"] == 2
    assert fake.calls[0][1]["timeout"] == 30
    assert capsys.readouterr().out.endswith('{\n  "ok": true\n}\n')


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

    monkeypatch.setattr(core, "Orchestrator", fail)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(argv)
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

    monkeypatch.setattr(core, "Orchestrator", fail)

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
            cli.main(argv)
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
        core,
        "Orchestrator",
        lambda *_: (_ for _ in ()).throw(AssertionError("constructed")),
    )
    with pytest.raises(SystemExit) as excinfo:
        cli.main(argv)
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert captured.out == "0.1.0\n"
    assert captured.err == ""


@pytest.mark.parametrize(("verb", "argv"), VERB_INVOCATIONS)
def test_version_after_a_verb_is_an_unrecognized_argument(
    verb: str, argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main([*argv, "--version"])
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
        cli.main([verb, "--help"])
    assert excinfo.value.code == 0
    assert "--version" not in capsys.readouterr().out


def test_version_after_the_separator_stays_prompt_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("AGENTS_ARMY_STATE_FILE", str(tmp_path / "state.json"))
    monkeypatch.setattr(core, "DEFAULT_BACKEND", "recording")

    cli.main(["talk", "a", "--", "text", "with", "--version", "in", "it"])

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

    monkeypatch.setenv(
        "AGENTS_ARMY_STATE_FILE", str(tmp_path / "orchestrator_state.json")
    )
    monkeypatch.setattr(cli, "_print_dependency_check", report)
    monkeypatch.setattr(
        core,
        "Orchestrator",
        lambda *_: (_ for _ in ()).throw(AssertionError("constructed")),
    )

    cli.main(["-v", "doctor"])
    assert called is True


def test_doctor_reporting_names_keep_their_owning_module_surface() -> None:
    reexported = (
        "DEPENDENCY_TOOLS",
        "FOUND",
        "FOUND_OPTIONAL",
        "MIN_PYTHON",
        "NAME_SEPARATORS",
        "NOT_FOUND",
        "VERSION_PROBE_TIMEOUT",
    )
    private = (
        "_dependency_report",
        "_describe_version",
        "_print_dependency_check",
        "_print_version",
        "_project_version",
        "_python_line",
        "_resolve_version",
        "_status_line",
        "_tool_line",
        "_tool_version",
    )

    assert set(reexported) <= set(orchestrator.__all__)
    assert all(
        getattr(orchestrator, name) is getattr(doctor, name) for name in reexported
    )
    assert set(private) <= vars(doctor).keys()
    assert all(not hasattr(orchestrator, name) for name in private)


def test_core_names_keep_their_owning_module_surface() -> None:
    reexported = (
        "Agent",
        "AgentBusyError",
        "AgentExistsError",
        "AgentNotFoundError",
        "Orchestrator",
        "OrchestratorError",
        "StateError",
        "TeamBusyError",
        "TRACE",
        "DEFAULT_BACKEND",
        "DEFAULT_VALIDATION_RETRIES",
    )
    private = (
        "_AgentRecord",
        "_MAX_REVALIDATE_ATTEMPTS",
        "_flock",
        "_is_live",
        "_load_state_file",
        "_utcnow",
    )

    assert set(reexported) <= set(orchestrator.__all__)
    assert all(
        getattr(orchestrator, name) is getattr(core, name) for name in reexported
    )
    assert set(private) <= vars(core).keys()
    assert all(not hasattr(orchestrator, name) for name in private)


def test_missing_skills_directory_has_one_stderr_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("AGENTS_ARMY_SKILLS", str(tmp_path / "missing"))
    monkeypatch.setenv("AGENTS_ARMY_STATE_FILE", str(tmp_path / "state.json"))
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["list", "skills"])
    captured = capsys.readouterr()
    assert excinfo.value.code == 1
    # An explicit catalog never falls back, so no AGENTS_ARMY_ROOT pin is
    # needed here: a real ~/.agents-army/SKILLS cannot answer instead.
    assert captured.err == f"skills directory not found: {tmp_path / 'missing'}\n"
    assert captured.out == ""


@pytest.mark.parametrize(
    ("argv", "level"),
    [
        (["-v", "list"], logging.DEBUG),
        (["-vv", "list"], core.TRACE),
        (["-vvv", "list"], core.TRACE),
        (["-v", "talk", "-v", "a", "-p", "x"], core.TRACE),
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

    monkeypatch.setattr(core, "Orchestrator", FakeOrchestrator)
    for logger_name in cli.OWN_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.NOTSET)
    cli.main(argv)
    assert logging.getLogger("orchestrator").level == level
    assert logging.getLogger("backends").level == level


@pytest.mark.parametrize(("verb", "argv"), VERB_INVOCATIONS)
def test_every_verb_accepts_verbosity_after_the_verb(
    verb: str, argv: list[str]
) -> None:
    opts = cli._build_parser().parse_args([*argv, "-vv"])

    assert opts.verbosity_after == 2
    assert opts._parser.prog == f"orchestrator {verb}"
