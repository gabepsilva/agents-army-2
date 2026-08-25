"""`--team`: an isolated registry + working directory for a group of agents."""

from __future__ import annotations

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
        '  export AGENTS_ARMY_TEAMS_DIR="$(git rev-parse '
        '--path-format=absolute --git-common-dir)/gdw-v3"\n' in err
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
