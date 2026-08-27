"""`fork`: a second agent that starts from another's session.

The verb records a fork rather than running one — no model turn is spent at
fork time — so what these tests pin down is the *first* turn the new agent
takes: which session id reaches the backend, whether the fork flag goes with
it, and what is written back afterwards.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import pytest

import orchestrator
from backends.base import (
    DEFAULT_TURN_TIMEOUT,
    AgentBackend,
    OutputSchema,
    TurnResult,
)
from backends.registry import register_backend
from orchestrator import (
    AgentExistsError,
    AgentNotFoundError,
    Orchestrator,
    OrchestratorError,
)
from tests.path_helpers import runtime_paths


class ForkingBackend(AgentBackend):
    """Records every turn's resume target, and answers with a fresh id.

    The reply's session id is derived from the resume target rather than
    fixed, so a forked turn reports an id that is *not* the source's — which
    is what makes "the new id was stored, and the source's was not" a real
    assertion rather than one that holds by construction.
    """

    name = "forking"
    supports_fork = True
    turns: ClassVar[list[tuple[str, str | None, bool]]] = []

    def run_turn(  # noqa: PLR0913 - test double mirrors AgentBackend interface
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
        ForkingBackend.turns.append((prompt, session_id, resume_as_fork))
        if resume_as_fork:
            return TurnResult(
                session_id=f"forked-from-{session_id}", reply="ok", raw=""
            )
        return TurnResult(session_id=session_id or "fresh-sid", reply="ok", raw="")


class UnforkableBackend(ForkingBackend):
    """Same double, minus the capability."""

    name = "unforkable"
    supports_fork = False


@pytest.fixture(autouse=True)
def _backends() -> None:
    ForkingBackend.turns = []
    register_backend("forking", ForkingBackend)
    register_backend("unforkable", UnforkableBackend)


def _primed(state_file: Path, backend: str = "forking") -> Orchestrator:
    """A registry holding one agent that has already had a turn."""
    orch = Orchestrator(runtime_paths(state_file.parent, state_file=state_file))
    orch.spawn("source", backend, model="m", reasoning_effort="high")
    orch.talk("source", "prime")
    return orch


def test_fork_inherits_the_configuration_and_records_the_source_session(
    tmp_path: Path,
) -> None:
    orch = _primed(tmp_path / "state.json")
    ForkingBackend.turns = []

    forked = orch.fork("source", "copy")

    assert (
        forked.backend.name,
        forked.backend.model,
        forked.backend.reasoning_effort,
    ) == (
        "forking",
        "m",
        "high",
    )
    # The whole point of a recorded fork: nothing ran.
    assert ForkingBackend.turns == []
    assert forked.session_id is None
    assert forked.pending_fork_from == "fresh-sid"
    assert orch.agents["source"].session_id == "fresh-sid"


def test_an_agent_that_was_never_forked_takes_an_ordinary_first_turn(
    tmp_path: Path,
) -> None:
    """Nothing pending means nothing to fork: the resume target is this
    agent's own session, and the fork flag stays off."""
    orchestrator.Agent("solo", ForkingBackend(), workdir=tmp_path).talk("hello")

    assert ForkingBackend.turns == [("hello", None, False)]


def test_the_first_turn_forks_the_source_and_later_turns_resume_the_copy(
    tmp_path: Path,
) -> None:
    state_file = tmp_path / "state.json"
    orch = _primed(state_file)
    orch.fork("source", "copy")
    ForkingBackend.turns = []

    orch.talk("copy", "first")
    orch.talk("copy", "second")

    assert ForkingBackend.turns == [
        ("first", "fresh-sid", True),
        ("second", "forked-from-fresh-sid", False),
    ]
    entries = json.loads(state_file.read_text(encoding="utf-8"))
    assert entries["copy"]["session_id"] == "forked-from-fresh-sid"
    assert "pending_fork_from" not in entries["copy"]
    # The source is untouched by its copy's turns and stays talkable.
    assert entries["source"]["session_id"] == "fresh-sid"


def test_a_pending_forks_first_turn_is_logged_as_a_fork(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The turn does resume a session — the source's — so a log line reading
    `resume=False` would misreport what the CLI was asked to do."""
    orch = _primed(tmp_path / "state.json")
    orch.fork("source", "copy")

    with caplog.at_level("INFO", logger="orchestrator"):
        orch.talk("copy", "first")

    assert "agent 'copy' (forking): starting turn, resume=True fork=True" in [
        record.getMessage() for record in caplog.records
    ]


def test_the_recorded_fork_survives_a_reload_before_the_first_turn(
    tmp_path: Path,
) -> None:
    """The marker is state, not a live attribute: another process must pick
    the fork up from the file."""
    state_file = tmp_path / "state.json"
    _primed(state_file).fork("source", "copy")
    ForkingBackend.turns = []

    Orchestrator(runtime_paths(tmp_path, state_file=state_file)).talk("copy", "first")

    assert ForkingBackend.turns == [("first", "fresh-sid", True)]


def test_a_fork_the_cli_did_not_perform_is_refused_instead_of_aliased(
    tmp_path: Path,
) -> None:
    """A forked resume that comes back with the *source's* id did not fork.

    Storing it would leave two agents pointing at one session under different
    names — the exact overlap the per-agent lock exists to prevent, and one
    it cannot prevent here because the names differ.
    """
    state_file = tmp_path / "state.json"
    orch = _primed(state_file)
    orch.fork("source", "copy")

    class Unforking(ForkingBackend):
        name = "forking"

        def run_turn(  # noqa: PLR0913 - test double mirrors AgentBackend interface
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
            super().run_turn(
                prompt, session_id, cwd, timeout, schema, resume_as_fork=resume_as_fork
            )
            # The failure mode under test: the CLI ignored the fork flag and
            # continued the source's session instead of copying it.
            return TurnResult(session_id=session_id, reply="ok", raw="")

    register_backend("forking", Unforking)
    before = state_file.read_text(encoding="utf-8")

    with pytest.raises(OrchestratorError) as excinfo:
        orch.talk("copy", "first")

    assert excinfo.value.args[0] == (
        "agent 'copy': forking reported the source's own session id "
        "('fresh-sid'), so the fork did not happen; refusing to point two "
        "agents at one session"
    )
    # Nothing was written: the fork is still pending and can be retried.
    assert state_file.read_text(encoding="utf-8") == before


def test_a_failed_first_turn_leaves_the_fork_pending(tmp_path: Path) -> None:
    """A fork that never produced a session id has not happened yet, so the
    next turn must fork again rather than start a fresh conversation."""
    state_file = tmp_path / "state.json"
    orch = _primed(state_file)
    orch.fork("source", "copy")

    class Forgetful(ForkingBackend):
        name = "forking"

        def run_turn(self, *args, **kwargs) -> TurnResult:
            super().run_turn(*args, **kwargs)
            return TurnResult(session_id=None, reply="ok", raw="")

    register_backend("forking", Forgetful)
    ForkingBackend.turns = []
    orch.talk("copy", "first")
    register_backend("forking", ForkingBackend)
    orch.talk("copy", "second")

    assert ForkingBackend.turns == [
        ("first", "fresh-sid", True),
        ("second", "fresh-sid", True),
    ]


@pytest.mark.parametrize(
    ("setup", "source", "dest", "error", "message"),
    [
        (
            "primed",
            "ghost",
            "copy",
            AgentNotFoundError,
            "no agent named 'ghost'",
        ),
        (
            "fresh",
            "source",
            "copy",
            OrchestratorError,
            "agent 'source' has no session to fork yet; talk to it first",
        ),
        (
            "taken",
            "source",
            "copy",
            AgentExistsError,
            "agent 'copy' already exists",
        ),
        (
            "unforkable",
            "source",
            "copy",
            OrchestratorError,
            "agent 'source' runs on backend 'unforkable', which cannot fork",
        ),
    ],
    ids=["unknown-source", "no-session-yet", "name-taken", "backend-cannot-fork"],
)
def test_a_rejected_fork_says_why_and_leaves_the_registry_alone(
    tmp_path: Path,
    setup: str,
    source: str,
    dest: str,
    error: type[Exception],
    message: str,
) -> None:
    state_file = tmp_path / "state.json"
    if setup == "fresh":
        orch = Orchestrator(runtime_paths(tmp_path, state_file=state_file))
        orch.spawn("source", "forking")
    elif setup == "unforkable":
        orch = _primed(state_file, "unforkable")
    else:
        orch = _primed(state_file)
        if setup == "taken":
            orch.spawn("copy", "forking")
    before = state_file.read_text(encoding="utf-8")

    with pytest.raises(error) as excinfo:
        orch.fork(source, dest)

    assert excinfo.value.args[0] == message
    assert state_file.read_text(encoding="utf-8") == before


@pytest.mark.parametrize("backend", ["claude", "codex", "grok", "opencode"])
def test_every_shipped_backend_can_be_forked(tmp_path: Path, backend: str) -> None:
    """Not the double: the answer must follow the real backend classes."""
    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps({"source": {"backend": backend, "session_id": "s1"}}),
        encoding="utf-8",
    )
    orch = Orchestrator(runtime_paths(tmp_path, state_file=state_file))

    orch.fork("source", "copy")

    copy = json.loads(state_file.read_text(encoding="utf-8"))["copy"]
    assert copy["backend"] == backend
    assert copy["pending_fork_from"] == "s1"


def test_the_cli_forks_and_reports_the_new_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state_file = tmp_path / "state.json"
    monkeypatch.setattr(orchestrator, "STATE_FILE", state_file)
    _primed(state_file)
    capsys.readouterr()

    orchestrator.main(["fork", "source", "copy"])

    assert capsys.readouterr().out == (
        "forked agent 'source' into 'copy' backend=forking\n"
    )
    assert (
        json.loads(state_file.read_text(encoding="utf-8"))["copy"]["pending_fork_from"]
        == "fresh-sid"
    )


def test_the_cli_reports_a_rejected_fork_in_one_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(orchestrator, "STATE_FILE", tmp_path / "state.json")

    with pytest.raises(SystemExit) as excinfo:
        orchestrator.main(["fork", "ghost", "copy"])

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "no agent named 'ghost'\n"
