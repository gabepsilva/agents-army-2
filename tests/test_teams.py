"""`--team`: an isolated registry + working directory for a group of agents."""

from __future__ import annotations

import errno
import fcntl
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

import orchestrator
from backends.base import AgentBackend, TurnResult
from backends.registry import register_backend


@pytest.fixture(autouse=True)
def _protect_module_globals(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore the path constants after every test.

    `--team` reassigns `orchestrator.STATE_FILE`/`WORKDIR`/`SKILLS_DIR` as
    real module globals (by design — see the PR description), not through
    `monkeypatch`. Registering their current values here means monkeypatch's
    teardown puts them back regardless of what a test or `main()` did to
    them meanwhile, so one test's team never leaks into the next.
    """
    monkeypatch.setattr(orchestrator, "STATE_FILE", orchestrator.STATE_FILE)
    monkeypatch.setattr(orchestrator, "WORKDIR", orchestrator.WORKDIR)
    monkeypatch.setattr(orchestrator, "SKILLS_DIR", orchestrator.SKILLS_DIR)
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", orchestrator.TEAMS_DIR)
    monkeypatch.setattr(orchestrator, "ROOT", orchestrator.ROOT)


def _make_recording_backend(sink: list[Path]) -> type[AgentBackend]:
    """A backend whose turns record the cwd they ran in and echo the prompt.

    The session id embeds the prompt so two teams talking to same-named
    agents can be told apart by what got persisted for each.
    """

    class RecordingBackend(AgentBackend):
        @property
        def name(self) -> str:
            return "recording"

        def run_turn(
            self,
            prompt: str,
            session_id: str | None,
            cwd: Path,
            timeout: int = orchestrator.DEFAULT_TURN_TIMEOUT,
            schema=None,
        ) -> TurnResult:
            sink.append(cwd)
            return TurnResult(
                session_id=f"sid-{prompt}", reply=f"reply:{prompt}", raw=""
            )

    return RecordingBackend


# A GIT_* variable pointing at the *outer* repo (GIT_DIR, GIT_INDEX_FILE, ...)
# leaks into a subprocess from any ambient git invocation — notably a git
# hook, which is exactly how `make verify` itself runs pre-commit. Left
# alone, it redirects these scratch-repo commands at the wrong repository
# instead of the tmp_path one `cwd` points to.
_GIT_ENV = {
    key: value for key, value in os.environ.items() if not key.startswith("GIT_")
}


def _run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
        env=_GIT_ENV,
    )


def _init_repo(root: Path) -> None:
    _run_git(["init", "-q", "-b", "main"], cwd=root)
    _run_git(
        [
            "-c",
            "user.email=t@example.com",
            "-c",
            "user.name=t",
            "commit",
            "--allow-empty",
            "-q",
            "-m",
            "init",
        ],
        cwd=root,
    )


def _read_state(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _make_team(
    root: Path,
    *parts: str,
    agents: dict[str, str] | None = None,
    worktree: bool = True,
) -> Path:
    """Write a minimal team directory (`agents/orchestrator_state.json`,
    optionally `worktree/`) at `root/<parts>` and return its path."""
    team_dir = root.joinpath(*parts)
    agents_dir = team_dir / "agents"
    agents_dir.mkdir(parents=True)
    payload = {name: {"backend": backend} for name, backend in (agents or {}).items()}
    (agents_dir / "orchestrator_state.json").write_text(json.dumps(payload))
    if worktree:
        (team_dir / "worktree").mkdir()
    return team_dir


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def test_workdir_reaches_the_backend_as_the_team_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Path] = []
    register_backend("recording", _make_recording_backend(seen))
    teams_dir = tmp_path / "teams"
    worktree = teams_dir / "t1" / "worktree"
    worktree.mkdir(parents=True)
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", teams_dir)

    orchestrator.main(["talk", "a", "--team", "t1", "-b", "recording", "-p", "hi"])

    assert seen == [worktree]


def test_state_file_and_skills_dir_resolve_under_the_team_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    register_backend("recording", _make_recording_backend([]))
    teams_dir = tmp_path / "teams"
    worktree = teams_dir / "t1" / "worktree"
    worktree.mkdir(parents=True)
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", teams_dir)

    orchestrator.main(["create", "a", "--team", "t1", "-b", "recording"])

    assert (
        teams_dir / "t1" / "agents" / "orchestrator_state.json"
        == orchestrator.STATE_FILE
    )
    assert orchestrator.STATE_FILE.exists()
    assert worktree == orchestrator.WORKDIR
    assert worktree / "SKILLS" == orchestrator.SKILLS_DIR


def test_explicit_agents_army_skills_still_wins_under_team(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    register_backend("recording", _make_recording_backend([]))
    teams_dir = tmp_path / "teams"
    worktree = teams_dir / "t1" / "worktree"
    worktree.mkdir(parents=True)
    custom_skills = tmp_path / "custom-skills"
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", teams_dir)
    monkeypatch.setenv("AGENTS_ARMY_SKILLS", str(custom_skills))

    orchestrator.main(["create", "a", "--team", "t1", "-b", "recording"])

    assert custom_skills == orchestrator.SKILLS_DIR


def test_teamless_behavior_is_unaffected_by_team_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Path] = []
    register_backend("recording", _make_recording_backend(seen))
    state_file = tmp_path / "state.json"
    workdir = tmp_path / "home"
    monkeypatch.setattr(orchestrator, "STATE_FILE", state_file)
    monkeypatch.setattr(orchestrator, "WORKDIR", workdir)

    orchestrator.main(["talk", "a", "-b", "recording", "-p", "hi"])

    assert seen == [workdir]
    assert _read_state(state_file)["a"]["backend"] == "recording"


# ---------------------------------------------------------------------------
# Validation (V1-V5): exit 2, nothing created on disk
# ---------------------------------------------------------------------------


def test_v1_missing_teams_dir_env_exits_2_and_names_the_variable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", None)

    with pytest.raises(SystemExit) as excinfo:
        orchestrator.main(["list", "agents", "--team", "t1"])

    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert (
        "orchestrator list: error: --team requires AGENTS_ARMY_TEAMS_DIR to "
        "be set; export it first, e.g.:\n"
        '  export AGENTS_ARMY_TEAMS_DIR="$HOME/.agents-army/<repo>/<workflow>"\n' in err
    )


@pytest.mark.parametrize("bad_name", ["a/b", "..", ".", "", "a b", "a$b"])
def test_v2_invalid_team_name_exits_2_and_creates_nothing(
    bad_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    teams_dir = tmp_path / "teams"
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", teams_dir)

    with pytest.raises(SystemExit) as excinfo:
        orchestrator.main(["list", "agents", "--team", bad_name])

    assert excinfo.value.code == 2
    assert not teams_dir.exists()
    assert (
        f"orchestrator list: error: invalid team name {bad_name!r}: must "
        "match '[-_.A-Za-z0-9]+' and not be '.' or '..'\n" in capsys.readouterr().err
    )


def test_v3_team_with_explicit_state_file_exits_2_and_creates_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    teams_dir = tmp_path / "teams"
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", teams_dir)
    monkeypatch.setenv("AGENTS_ARMY_STATE_FILE", str(tmp_path / "explicit_state.json"))

    with pytest.raises(SystemExit) as excinfo:
        orchestrator.main(["list", "agents", "--team", "t1"])

    assert excinfo.value.code == 2
    assert not teams_dir.exists()
    assert (
        "orchestrator list: error: --team cannot be combined with an "
        "explicit AGENTS_ARMY_STATE_FILE (unset it, or drop --team)\n"
        in capsys.readouterr().err
    )


def test_v4_team_with_explicit_home_exits_2_and_creates_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    teams_dir = tmp_path / "teams"
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", teams_dir)
    monkeypatch.setenv("AGENTS_ARMY_HOME", str(tmp_path / "explicit_home"))

    with pytest.raises(SystemExit) as excinfo:
        orchestrator.main(["list", "agents", "--team", "t1"])

    assert excinfo.value.code == 2
    assert not teams_dir.exists()
    assert (
        "orchestrator list: error: --team cannot be combined with an "
        "explicit AGENTS_ARMY_HOME (unset it, or drop --team)\n"
        in capsys.readouterr().err
    )


def test_v5_missing_worktree_exits_2_for_create_talk_list_and_delete_by_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    teams_dir = tmp_path / "teams"
    worktree = teams_dir / "t1" / "worktree"
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", teams_dir)

    for argv, prog in (
        (["create", "a", "--team", "t1", "-b", "claude"], "create"),
        (["talk", "a", "--team", "t1", "-b", "claude", "-p", "hi"], "talk"),
        (["list", "agents", "--team", "t1"], "list"),
        (["delete", "a", "--team", "t1"], "delete"),
    ):
        with pytest.raises(SystemExit) as excinfo:
            orchestrator.main(argv)
        assert excinfo.value.code == 2
        assert (
            f"orchestrator {prog}: error: team workspace {worktree} does "
            f"not exist; create it first with 'git worktree add "
            f"{worktree} ...'\n" in capsys.readouterr().err
        )

    assert not teams_dir.exists()


def test_team_help_text_names_the_layout(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        orchestrator.main(["talk", "--help"])
    assert (
        "  --team TEAM           run against\n"
        "                        $AGENTS_ARMY_TEAMS_DIR/<team>/{agents,worktree}\n"
        "                        instead of the teamless layout\n"
        in capsys.readouterr().out
    )


def test_v5_does_not_apply_to_teardown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Teardown must stay possible after `git worktree remove`."""
    teams_dir = tmp_path / "teams"
    register_backend("recording", _make_recording_backend([]))
    worktree = teams_dir / "t1" / "worktree"
    worktree.mkdir(parents=True)
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", teams_dir)
    orchestrator.main(["create", "owen", "--team", "t1", "-b", "recording"])
    # Simulate the worktree already having been removed by the caller.
    shutil.rmtree(worktree)

    orchestrator.main(["delete", "--team", "t1"])

    assert not (teams_dir / "t1" / "agents").exists()


# ---------------------------------------------------------------------------
# Isolation between teams
# ---------------------------------------------------------------------------


def test_two_teams_with_the_same_agent_name_are_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Path] = []
    register_backend("recording", _make_recording_backend(seen))
    teams_dir = tmp_path / "teams"
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", teams_dir)
    worktree1 = teams_dir / "t1" / "worktree"
    worktree2 = teams_dir / "t2" / "worktree"
    worktree1.mkdir(parents=True)
    worktree2.mkdir(parents=True)

    orchestrator.main(["create", "owen", "--team", "t1", "-b", "recording"])
    orchestrator.main(["create", "owen", "--team", "t2", "-b", "recording"])

    state_file1 = teams_dir / "t1" / "agents" / "orchestrator_state.json"
    state_file2 = teams_dir / "t2" / "agents" / "orchestrator_state.json"
    assert state_file1 != state_file2
    assert "owen" in _read_state(state_file1)
    assert "owen" in _read_state(state_file2)

    orchestrator.main(["talk", "owen", "--team", "t1", "-p", "hi1"])
    orchestrator.main(["talk", "owen", "--team", "t2", "-p", "hi2"])

    assert seen == [worktree1, worktree2]
    assert _read_state(state_file1)["owen"]["session_id"] == "sid-hi1"
    assert _read_state(state_file2)["owen"]["session_id"] == "sid-hi2"


# ---------------------------------------------------------------------------
# delete --team: single agent, teardown, locking
# ---------------------------------------------------------------------------


def test_delete_team_name_deletes_one_agent_leaving_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    register_backend("recording", _make_recording_backend([]))
    teams_dir = tmp_path / "teams"
    (teams_dir / "t1" / "worktree").mkdir(parents=True)
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", teams_dir)
    orchestrator.main(["create", "owen", "--team", "t1", "-b", "recording"])
    orchestrator.main(["create", "spectacle", "--team", "t1", "-b", "recording"])

    orchestrator.main(["delete", "owen", "--team", "t1"])

    state = _read_state(teams_dir / "t1" / "agents" / "orchestrator_state.json")
    assert "owen" not in state
    assert "spectacle" in state


def test_bare_delete_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        orchestrator.main(["delete"])
    assert excinfo.value.code == 2
    assert (
        "orchestrator delete: error: delete requires NAME or --team\n"
        in capsys.readouterr().err
    )


def test_teardown_of_a_team_with_no_agents_ever_created_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The team root exists (worktree added) but `create` was never run."""
    teams_dir = tmp_path / "teams"
    (teams_dir / "t1" / "worktree").mkdir(parents=True)
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", teams_dir)

    orchestrator.main(["delete", "--team", "t1"])

    assert capsys.readouterr().out == "deleted team 't1'\n"
    assert not (teams_dir / "t1" / "agents").exists()


@pytest.mark.parametrize(
    ("label", "payload"),
    [
        ("invalid json", "not json"),
        (
            "unknown backend",
            '{"owen": {"backend": "retired-backend", "session_id": null}}',
        ),
    ],
)
def test_teardown_removes_a_registry_it_cannot_parse(
    label: str,
    payload: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Teardown must not need the registry it deletes to be readable.

    Constructing an `Orchestrator` first made both of these unrecoverable:
    the constructor parses the state file, so the one command whose job is
    to remove a broken registry was the one the broken registry stopped,
    leaving `rm -rf` as the only way out.
    """
    teams_dir = tmp_path / "teams"
    agents_dir = teams_dir / "t1" / "agents"
    agents_dir.mkdir(parents=True)
    (teams_dir / "t1" / "worktree").mkdir()
    (agents_dir / "orchestrator_state.json").write_text(payload, encoding="utf-8")
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", teams_dir)

    orchestrator.main(["delete", "--team", "t1"])

    captured = capsys.readouterr()
    assert captured.out == "deleted team 't1'\n"
    assert captured.err == ""
    assert not agents_dir.exists()


def _make_blocking_backend() -> type[AgentBackend]:
    """A backend whose turn fails the way a non-blocking pipe does.

    Not hypothetical: `print`ing a large reply to a stdout someone left
    O_NONBLOCK raises exactly this, and so can a backend's own pipe.
    """

    class BlockingBackend(AgentBackend):
        @property
        def name(self) -> str:
            return "blocking"

        def run_turn(
            self,
            prompt: str,
            session_id: str | None,
            cwd: Path,
            timeout: int = orchestrator.DEFAULT_TURN_TIMEOUT,
            schema=None,
        ) -> TurnResult:
            raise BlockingIOError(
                errno.EAGAIN, "write could not complete without blocking"
            )

    return BlockingBackend


@pytest.mark.parametrize("team_argv", [[], ["--team", "t1"]])
def test_an_incidental_blocking_io_error_is_not_a_busy_team(
    team_argv: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Only the team lock may report that a team is in use.

    Catching `BlockingIOError` around the whole dispatch claimed every other
    one for the lock — a teamless `talk` died with "team 'None' is in use by
    another command", exit 1, and the real error was gone.
    """
    register_backend("blocking", _make_blocking_backend())
    teams_dir = tmp_path / "teams"
    (teams_dir / "t1" / "worktree").mkdir(parents=True)
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", teams_dir)
    monkeypatch.setattr(orchestrator, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(orchestrator, "WORKDIR", tmp_path)

    with pytest.raises(BlockingIOError):
        orchestrator.main(["talk", "a", *team_argv, "-b", "blocking", "-p", "hi"])

    assert "in use by another command" not in capsys.readouterr().err


def test_teardown_of_an_unknown_team_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    teams_dir = tmp_path / "teams"
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", teams_dir)

    with pytest.raises(SystemExit) as excinfo:
        orchestrator.main(["delete", "--team", "nope"])

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert captured.err == f"team 'nope' not found at {teams_dir / 'nope'}\n"
    assert captured.out == ""


def test_delete_team_removes_agents_dir_leaves_worktree_and_git_metadata_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    teams_dir = tmp_path / "teams"
    worktree = teams_dir / "t1" / "worktree"
    _run_git(["worktree", "add", "-q", "-B", "t1-branch", str(worktree)], cwd=repo)
    register_backend("recording", _make_recording_backend([]))
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", teams_dir)
    orchestrator.main(["create", "owen", "--team", "t1", "-b", "recording"])
    agents_dir = teams_dir / "t1" / "agents"
    assert agents_dir.exists()

    orchestrator.main(["delete", "--team", "t1"])

    assert not agents_dir.exists()
    assert worktree.is_dir()
    listing = _run_git(["worktree", "list", "--porcelain"], cwd=repo).stdout
    assert "prunable" not in listing


def test_teardown_overlapping_a_held_shared_lock_exits_nonzero_and_removes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    register_backend("recording", _make_recording_backend([]))
    teams_dir = tmp_path / "teams"
    (teams_dir / "t1" / "worktree").mkdir(parents=True)
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", teams_dir)
    orchestrator.main(["create", "owen", "--team", "t1", "-b", "recording"])
    agents_dir = teams_dir / "t1" / "agents"
    assert agents_dir.exists()
    capsys.readouterr()  # discard the "created agent" noise above

    lock_path = teams_dir / "t1" / ".lock"
    with lock_path.open("a+", encoding="utf-8") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_SH)
        with pytest.raises(SystemExit) as excinfo:
            orchestrator.main(["delete", "--team", "t1"])
        fcntl.flock(held.fileno(), fcntl.LOCK_UN)

    assert excinfo.value.code == 1
    assert agents_dir.exists()
    captured = capsys.readouterr()
    assert captured.err == (
        "team 't1' is in use by another command; try again once it finishes\n"
    )
    assert captured.out == ""


# ---------------------------------------------------------------------------
# `list teams` (discovery under AGENTS_ARMY_ROOT)
# ---------------------------------------------------------------------------


def test_list_teams_empty_root_prints_no_teams_and_exits_0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(orchestrator, "ROOT", tmp_path / "does-not-exist")
    monkeypatch.setattr(
        orchestrator,
        "STATE_FILE",
        tmp_path / "does-not-exist" / "orchestrator_state.json",
    )
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", None)

    orchestrator.main(["list", "teams"])

    assert capsys.readouterr().out == "no teams\n"


def test_list_teams_several_teams_under_one_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "root"
    _make_team(root, "repo-a", "gdw-v3", "issue-1", agents={"dev": "claude"})
    _make_team(
        root, "repo-a", "gdw-v3", "issue-2", agents={"dev": "claude", "rev": "codex"}
    )
    monkeypatch.setattr(orchestrator, "ROOT", root)
    monkeypatch.setattr(orchestrator, "STATE_FILE", root / "orchestrator_state.json")
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", None)

    orchestrator.main(["list", "teams"])

    out = capsys.readouterr().out
    assert "repo-a/gdw-v3/issue-1" in out
    assert "1 agent" in out
    assert "dev/claude" in out
    assert "repo-a/gdw-v3/issue-2" in out
    assert "2 agents" in out
    assert "rev/codex" in out


def test_list_teams_namespaced_root_and_flat_root_in_same_walk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "root"
    _make_team(root, "repo-a", "gdw-v3", "issue-1", agents={"dev": "claude"})
    _make_team(root, "flat", agents={"ops": "codex"})
    monkeypatch.setattr(orchestrator, "ROOT", root)
    monkeypatch.setattr(orchestrator, "STATE_FILE", root / "orchestrator_state.json")
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", None)

    orchestrator.main(["list", "teams"])

    out = capsys.readouterr().out
    assert "repo-a/gdw-v3/issue-1" in out
    assert "flat" in out


def test_list_teams_directory_without_agents_dir_is_not_a_team(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "root"
    (root / "not-a-team" / "worktree").mkdir(parents=True)
    monkeypatch.setattr(orchestrator, "ROOT", root)
    monkeypatch.setattr(orchestrator, "STATE_FILE", root / "orchestrator_state.json")
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", None)

    orchestrator.main(["list", "teams"])

    assert "not-a-team" not in capsys.readouterr().out


def test_list_teams_flags_a_team_missing_its_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "root"
    _make_team(root, "orphan", agents={"dev": "claude"}, worktree=False)
    monkeypatch.setattr(orchestrator, "ROOT", root)
    monkeypatch.setattr(orchestrator, "STATE_FILE", root / "orchestrator_state.json")
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", None)

    orchestrator.main(["list", "teams"])

    out = capsys.readouterr().out
    assert "orphan" in out
    assert "[worktree missing]" in out


def test_list_teams_team_with_intact_worktree_is_not_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "root"
    _make_team(root, "healthy", agents={"dev": "claude"}, worktree=True)
    monkeypatch.setattr(orchestrator, "ROOT", root)
    monkeypatch.setattr(orchestrator, "STATE_FILE", root / "orchestrator_state.json")
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", None)

    orchestrator.main(["list", "teams"])

    assert "[worktree missing]" not in capsys.readouterr().out


def test_list_teams_decoy_worktree_with_its_own_registry_yields_one_team(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A team is never descended into, so a decoy `agents/orchestrator_state.json`
    inside its own `worktree/` must not be discovered as a second team."""
    root = tmp_path / "root"
    team = _make_team(root, "issue-1", agents={"dev": "claude"})
    decoy_agents = team / "worktree" / "agents"
    decoy_agents.mkdir(parents=True)
    (decoy_agents / "orchestrator_state.json").write_text(
        json.dumps({"decoy": {"backend": "codex"}})
    )
    monkeypatch.setattr(orchestrator, "ROOT", root)
    monkeypatch.setattr(orchestrator, "STATE_FILE", root / "orchestrator_state.json")
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", None)

    orchestrator.main(["list", "teams"])

    out = capsys.readouterr().out
    assert out.count("issue-1") == 1
    assert "(1 agent: dev/claude)" in out
    assert "decoy/codex" not in out


def test_list_teams_depth_bound_finds_depth_4_but_not_depth_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "root"
    # Depth 4: $ROOT/a/b/c/d — exactly the team-then-worktree headroom the
    # namespaced layout needs (see teams.SEARCH_DEPTH's comment).
    _make_team(root, "a", "b", "c", "d", agents={"dev": "claude"})
    # Depth 5, nested one level past the bound: never reached, because a
    # depth-4 candidate is only ever stat'd, never scandir'd.
    _make_team(root, "x", "y", "z", "w", "v", agents={"dev": "claude"})
    monkeypatch.setattr(orchestrator, "ROOT", root)
    monkeypatch.setattr(orchestrator, "STATE_FILE", root / "orchestrator_state.json")
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", None)

    orchestrator.main(["list", "teams"])

    out = capsys.readouterr().out
    assert "a/b/c/d" in out
    assert "x/y/z/w/v" not in out
    assert "no teams" not in out


def test_list_teams_symlink_loop_terminates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A symlink is never treated as a directory to scan (`follow_symlinks=False`),
    so one pointing back at an ancestor cannot turn the walk into a loop."""
    root = tmp_path / "root"
    loopy = root / "loopy"
    loopy.mkdir(parents=True)
    (loopy / "self").symlink_to(root, target_is_directory=True)
    monkeypatch.setattr(orchestrator, "ROOT", root)
    monkeypatch.setattr(orchestrator, "STATE_FILE", root / "orchestrator_state.json")
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", None)

    orchestrator.main(["list", "teams"])  # must return, not hang

    assert capsys.readouterr().out == "no teams\n"


def test_list_teams_out_of_root_teams_dir_is_its_own_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    teams_dir = tmp_path / "elsewhere"
    _make_team(teams_dir, "t1", agents={"dev": "claude"})
    monkeypatch.setattr(orchestrator, "ROOT", root)
    monkeypatch.setattr(orchestrator, "STATE_FILE", root / "orchestrator_state.json")
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", teams_dir)

    orchestrator.main(["list", "teams"])

    out = capsys.readouterr().out
    assert str(root) in out
    assert str(teams_dir) in out
    assert "t1" in out


def test_list_teams_teams_dir_under_root_is_not_a_second_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "root"
    teams_dir = root / "nested"
    _make_team(teams_dir, "t1", agents={"dev": "claude"})
    monkeypatch.setattr(orchestrator, "ROOT", root)
    monkeypatch.setattr(orchestrator, "STATE_FILE", root / "orchestrator_state.json")
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", teams_dir)

    orchestrator.main(["list", "teams"])

    out = capsys.readouterr().out
    assert out.count("t1") == 1
    assert out.count(str(root)) == 1


def test_list_teams_finds_a_team_outside_root_when_teams_dir_is_an_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The reverse nesting of the case above: AGENTS_ARMY_TEAMS_DIR is an
    ancestor of AGENTS_ARMY_ROOT rather than the usual descendant.

    A team inside ROOT and a team outside it but still under TEAMS_DIR must
    both show up, exactly once each. An earlier fix skipped the whole
    TEAMS_DIR group whenever it overlapped ROOT at all, which silently
    dropped `outside_team` (still resolvable via `--team`) from a command
    whose whole job is "show me everything".
    """
    teams_dir = tmp_path / "teams_dir"
    root = teams_dir / "root"
    _make_team(root, "inside_team", agents={"dev": "claude"})
    _make_team(teams_dir, "outside_team", agents={"ops": "codex"})
    monkeypatch.setattr(orchestrator, "ROOT", root)
    monkeypatch.setattr(orchestrator, "STATE_FILE", root / "orchestrator_state.json")
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", teams_dir)

    orchestrator.main(["list", "teams"])

    out = capsys.readouterr().out
    assert out.count("inside_team") == 1
    assert out.count("outside_team") == 1
    assert "ops/codex" in out


@pytest.mark.parametrize(
    ("label", "payload"),
    [
        ("invalid json", "not json"),
        ("top-level list, not an object", '["not", "a", "dict"]'),
        ("entry is not an object", '{"dev": "claude"}'),
    ],
)
def test_list_teams_unreadable_team_registry_does_not_abort_the_walk(
    label: str,
    payload: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "root"
    _make_team(root, "broken", worktree=False)
    (root / "broken" / "agents" / "orchestrator_state.json").write_text(payload)
    _make_team(root, "healthy", agents={"dev": "claude"})
    monkeypatch.setattr(orchestrator, "ROOT", root)
    monkeypatch.setattr(orchestrator, "STATE_FILE", root / "orchestrator_state.json")
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", None)

    orchestrator.main(["list", "teams"])

    out = capsys.readouterr().out
    assert "registry unreadable" in out, label
    assert "broken" in out
    assert "healthy" in out
    assert "dev/claude" in out


def test_list_teams_unreadable_teamless_registry_does_not_abort_the_walk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "root"
    _make_team(root, "healthy", agents={"dev": "claude"})
    (root / "orchestrator_state.json").write_text("not json")
    monkeypatch.setattr(orchestrator, "ROOT", root)
    monkeypatch.setattr(orchestrator, "STATE_FILE", root / "orchestrator_state.json")
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", None)

    orchestrator.main(["list", "teams"])

    out = capsys.readouterr().out
    assert "healthy" in out
    assert "dev/claude" in out
    assert "(teamless)" in out
    assert "registry unreadable" in out


def test_list_teams_unreadable_teamless_registry_is_not_hidden_behind_no_teams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A corrupt teamless registry with zero teams elsewhere must still be
    flagged, not collapsed into the same 'no teams' as a genuinely empty
    root — `list agents` on the same file fails loudly, so `list teams`
    silently reporting success here would be misleading."""
    root = tmp_path / "root"
    root.mkdir()
    (root / "orchestrator_state.json").write_text("not json")
    monkeypatch.setattr(orchestrator, "ROOT", root)
    monkeypatch.setattr(orchestrator, "STATE_FILE", root / "orchestrator_state.json")
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", None)

    orchestrator.main(["list", "teams"])

    out = capsys.readouterr().out
    assert out != "no teams\n"
    assert "(teamless)" in out
    assert "registry unreadable" in out


def test_list_teams_teamless_group_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "orchestrator_state.json").write_text(
        json.dumps({"tir": {"backend": "codex"}})
    )
    monkeypatch.setattr(orchestrator, "ROOT", root)
    monkeypatch.setattr(orchestrator, "STATE_FILE", root / "orchestrator_state.json")
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", None)

    orchestrator.main(["list", "teams"])

    out = capsys.readouterr().out
    assert "(teamless)" in out
    assert "tir" in out
    assert "backend=codex" in out
    assert f"(teamless) {root / 'orchestrator_state.json'}" in out


def test_list_teams_teamless_group_reads_state_file_not_bare_root_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The `(teamless)` group must report the registry `list agents`/`talk`
    actually use — STATE_FILE, wherever an explicit AGENTS_ARMY_HOME or
    AGENTS_ARMY_STATE_FILE relocated it — not always `$ROOT/orchestrator_state.json`.
    A stray file sitting at that bare path when STATE_FILE points elsewhere
    must not be reported at all."""
    root = tmp_path / "root"
    root.mkdir()
    (root / "orchestrator_state.json").write_text(
        json.dumps({"decoy": {"backend": "codex"}})
    )
    relocated = tmp_path / "home" / "orchestrator_state.json"
    relocated.parent.mkdir(parents=True)
    relocated.write_text(json.dumps({"realagent": {"backend": "claude"}}))
    monkeypatch.setattr(orchestrator, "ROOT", root)
    monkeypatch.setattr(orchestrator, "STATE_FILE", relocated)
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", None)

    orchestrator.main(["list", "teams"])

    out = capsys.readouterr().out
    assert f"(teamless) {relocated}" in out
    assert "realagent" in out
    assert "decoy" not in out


def test_list_teams_teamless_group_present_but_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "orchestrator_state.json").write_text("{}")
    monkeypatch.setattr(orchestrator, "ROOT", root)
    monkeypatch.setattr(orchestrator, "STATE_FILE", root / "orchestrator_state.json")
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", None)

    orchestrator.main(["list", "teams"])

    out = capsys.readouterr().out
    assert "(teamless)" in out
    assert "0 agents" in out
    assert "registry unreadable" not in out


def test_list_teams_teamless_group_absent_when_no_bare_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "root"
    _make_team(root, "t1", agents={"dev": "claude"})
    monkeypatch.setattr(orchestrator, "ROOT", root)
    monkeypatch.setattr(orchestrator, "STATE_FILE", root / "orchestrator_state.json")
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", None)

    orchestrator.main(["list", "teams"])

    assert "(teamless)" not in capsys.readouterr().out


def test_list_teams_with_team_option_exits_2(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        orchestrator.main(["list", "teams", "--team", "x"])

    assert excinfo.value.code == 2
    assert (
        "orchestrator list: error: list teams cannot be combined with --team\n"
        in capsys.readouterr().err
    )
