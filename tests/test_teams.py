"""`--team`: an isolated registry + working directory for a group of agents."""

from __future__ import annotations

import errno
import fcntl
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import orchestrator
from backends.base import AgentBackend, TurnError, TurnResult
from backends.registry import register_backend
from orchestrator import paths, teams


@pytest.fixture(autouse=True)
def _protect_module_globals(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore `ROOT`/`TEAMS_DIR` after every test.

    Nothing rebinds them any more — `--team` resolves its paths into a
    returned `RuntimePaths` instead. These two are here because most tests
    in this file point them at a `tmp_path` with a bare
    `monkeypatch.setattr` of their own and the machine's real
    `$AGENTS_ARMY_ROOT` must not be what a test that forgets one walks;
    registering the current values makes that restoration unconditional.
    `STATE_FILE`/`WORKDIR`/`SKILLS_DIR` no longer need it: a team run leaves
    them alone (see `test_a_team_run_leaves_the_module_path_globals_alone`).
    """
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


def _env_without(*unset: str, **overrides: str) -> dict[str, str]:
    """A subprocess env: the real environment, minus `unset`, plus `overrides`.

    Not `os.environ.copy()` + `.pop(name, None)`: mixing a `str` value type
    with a `None` default makes `dict.pop` return `str | None`, which the
    type checker then attributes back to the dict itself.
    """
    env = {k: v for k, v in os.environ.items() if k not in unset}
    env.update(overrides)
    return env


def _read_state(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolved_paths(argv: list[str]) -> paths.RuntimePaths:
    """The `RuntimePaths` a run of `argv` resolves, without running the verb.

    `_resolve_team` is where a `--team` run's paths are now decided, and the
    object it returns is the only thing downstream reads — so this is the
    seam that says what a team resolves to, in place of the module globals
    the old assertions read.
    """
    opts = orchestrator._build_parser().parse_args(argv)
    runtime_paths, lock = orchestrator._resolve_team(opts, teardown=False)
    with lock:
        return runtime_paths


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
# teams.resolve — unit tests, no CLI coupling
# ---------------------------------------------------------------------------


def test_resolve_single_hit_from_an_unrelated_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    team_dir = _make_team(root, "t1", agents={"dev": "claude"})

    assert teams.resolve(root, "t1") == [team_dir]


def test_resolve_finds_a_worktree_only_team_with_no_registry(tmp_path: Path) -> None:
    """The `go.sh` first-command case: `git worktree add` has run but no
    orchestrator command has created the registry yet, so `agents/` doesn't
    exist at all."""
    root = tmp_path / "root"
    team_dir = root / "repo" / "wf" / "t1"
    (team_dir / "worktree").mkdir(parents=True)

    assert teams.resolve(root, "t1") == [team_dir]


def test_resolve_finds_a_registry_only_orphan(tmp_path: Path) -> None:
    """`delete --team`'s own teardown residue: `worktree/` removed,
    `agents/` (with its marker) still there."""
    root = tmp_path / "root"
    team_dir = _make_team(root, "t1", agents={"dev": "claude"}, worktree=False)

    assert teams.resolve(root, "t1") == [team_dir]


def test_resolve_finds_an_agents_dir_without_the_marker_file(tmp_path: Path) -> None:
    """The `_flock` residue: `path.parent.mkdir(...)` runs before the lock
    file is opened, so a command that takes the registry lock and fails
    before persisting leaves `agents/` behind with no
    `orchestrator_state.json` inside it. `is_team` (the marker) would miss
    this; the looser candidate rule (`agents/`-the-directory) must not."""
    root = tmp_path / "root"
    team_dir = root / "t1"
    (team_dir / "agents").mkdir(parents=True)

    assert teams.resolve(root, "t1") == [team_dir]
    assert not teams.is_team(team_dir)


def test_resolve_shadow_regression_stray_agents_dir_does_not_hide_a_marked_team(
    tmp_path: Path,
) -> None:
    """The candidate predicate (what `resolve` may name) and the
    never-descend predicate (proof you are standing inside a real team
    root) are deliberately different predicates. A stray, unmarked
    `agents/` above a real, marked team must not stop the walk from
    reaching that team — only a directory that actually has
    `agents/orchestrator_state.json` may swallow its own children.

    This test fails if never-descend is keyed on the candidate predicate
    instead of `is_team` — that is the point of writing it: widening
    never-descend to match the looser candidate rule would make this one
    stray directory hide every marked team beneath it, from every name,
    reintroducing the exact defect (`list teams` sees a team, `--team`
    can't reach it) that this PR exists to fix.
    """
    root = tmp_path / "root"
    (root / "repoA" / "agents").mkdir(parents=True)  # stray, unmarked
    team_dir = _make_team(root, "repoA", "wf", "issue-97", agents={"dev": "claude"})

    assert teams.resolve(root, "issue-97") == [team_dir]


def test_resolve_zero_hit_message_scenario_no_candidate_at_all(tmp_path: Path) -> None:
    """A directory with neither `agents/` nor `worktree/` (just leftover
    `logs/`, `.lock`, `spectacle.prompt`) is not a candidate."""
    root = tmp_path / "root"
    team_dir = root / "t1"
    team_dir.mkdir(parents=True)
    (team_dir / "logs").mkdir()
    (team_dir / ".lock").touch()
    (team_dir / "spectacle.prompt").touch()

    assert teams.resolve(root, "t1") == []


def test_resolve_ambiguity_across_two_repo_prefixes(tmp_path: Path) -> None:
    root = tmp_path / "root"
    team_a = _make_team(root, "repo-a", "wf", "issue-97", agents={"dev": "claude"})
    team_b = _make_team(root, "repo-b", "wf", "issue-97", agents={"dev": "codex"})

    assert teams.resolve(root, "issue-97") == [team_a, team_b]


def test_resolve_qualified_name_selects_one_of_two_ambiguous_teams(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    team_a = _make_team(root, "repo-a", "wf", "issue-97", agents={"dev": "claude"})
    _make_team(root, "repo-b", "wf", "issue-97", agents={"dev": "codex"})

    assert teams.resolve(root, "repo-a/wf/issue-97") == [team_a]


def test_resolve_matches_whole_segments_not_a_string_suffix(tmp_path: Path) -> None:
    """'97' must not match 'issue-97' — segment-wise matching, not
    `str.endswith()`."""
    root = tmp_path / "root"
    _make_team(root, "issue-97", agents={"dev": "claude"})

    assert teams.resolve(root, "97") == []


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

    resolved = _resolved_paths(["create", "a", "--team", "t1", "-b", "recording"])

    assert (
        resolved.state_file == teams_dir / "t1" / "agents" / "orchestrator_state.json"
    )
    assert resolved.workdir == worktree
    assert resolved.skills_dir == worktree / "SKILLS"

    orchestrator.main(["create", "a", "--team", "t1", "-b", "recording"])

    assert resolved.state_file.exists()


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

    resolved = _resolved_paths(["create", "a", "--team", "t1", "-b", "recording"])

    assert resolved.skills_dir == custom_skills


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


def test_v1_teams_dir_unset_resolves_under_root_instead_of_erroring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The old hard failure is gone: with AGENTS_ARMY_TEAMS_DIR unset,
    --team now resolves the team under AGENTS_ARMY_ROOT instead of refusing
    to run at all."""
    root = tmp_path / "root"
    _make_team(root, "t1", agents={"dev": "claude"})
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", None)
    monkeypatch.setattr(orchestrator, "ROOT", root)

    orchestrator.main(["list", "agents", "--team", "t1"])

    out = capsys.readouterr().out
    assert "dev" in out
    assert "backend=claude" in out


@pytest.mark.parametrize(
    "bad_name", ["..", ".", "", "a b", "a$b", "a/../b", "a/", "/a", "a//b"]
)
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
        "match '[-_.A-Za-z0-9]+(?:/[-_.A-Za-z0-9]+)*' segment-by-segment, "
        "and no segment may be '.' or '..'\n" in capsys.readouterr().err
    )


def test_v2_qualified_name_rejected_while_teams_dir_is_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A `/`-qualified name is root-relative by construction; TEAMS_DIR
    supplies its own namespace, so joining one under it would double-join
    instead of resolving — reject it with the recovery instruction rather
    than letting it silently miss."""
    teams_dir = tmp_path / "teams"
    root = tmp_path / "root"
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", teams_dir)
    monkeypatch.setattr(orchestrator, "ROOT", root)

    with pytest.raises(SystemExit) as excinfo:
        orchestrator.main(["list", "agents", "--team", "agents-army-2/gdw-v3/issue-97"])

    assert excinfo.value.code == 2
    assert not teams_dir.exists()
    assert (
        "orchestrator list: error: invalid team name "
        "'agents-army-2/gdw-v3/issue-97': a qualified name is relative to "
        "$AGENTS_ARMY_ROOT and cannot be used while AGENTS_ARMY_TEAMS_DIR "
        f"is set. Use the bare name 'issue-97', or unset AGENTS_ARMY_TEAMS_DIR "
        f"to resolve under {root}.\n" in capsys.readouterr().err
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


def test_v5_missing_worktree_exits_2_for_create_and_talk_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Narrowed gate: only create/talk launch a backend and need WORKDIR to
    exist. list agents/delete NAME just read and edit a JSON file."""
    teams_dir = tmp_path / "teams"
    worktree = teams_dir / "t1" / "worktree"
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", teams_dir)

    for argv, prog in (
        (["create", "a", "--team", "t1", "-b", "claude"], "create"),
        (["talk", "a", "--team", "t1", "-b", "claude", "-p", "hi"], "talk"),
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


@pytest.mark.parametrize(
    "argv",
    [
        ["list", "agents", "--team", "nosuch"],
        ["delete", "someagent", "--team", "nosuch"],
    ],
)
def test_bogus_team_under_teams_dir_exits_1_and_creates_nothing(
    argv: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """list agents/delete NAME must not reach `_team_locked` -> `_flock` for
    a team_root that was never created — `_flock`'s own
    `path.parent.mkdir(parents=True, exist_ok=True)` would otherwise
    fabricate `nosuch/` (and `.lock`) on disk for a plain typo, and that
    residue is a real `agents/`-or-`worktree/` candidate on every future
    AGENTS_ARMY_ROOT walk."""
    teams_dir = tmp_path / "teams"
    teams_dir.mkdir()
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", teams_dir)

    with pytest.raises(SystemExit) as excinfo:
        orchestrator.main(argv)

    assert excinfo.value.code == 1
    assert (
        capsys.readouterr().err
        == f"team 'nosuch' not found at {teams_dir / 'nosuch'}\n"
    )
    assert list(teams_dir.iterdir()) == []


def test_list_agents_team_succeeds_with_no_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The narrowed worktree gate (§4): list agents --team is read-only and
    must work on the orphan state `delete --team` deliberately leaves
    behind — registry present, worktree gone."""
    teams_dir = tmp_path / "teams"
    _make_team(teams_dir, "t1", agents={"dev": "claude"}, worktree=False)
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", teams_dir)

    orchestrator.main(["list", "agents", "--team", "t1"])

    out = capsys.readouterr().out
    assert "dev" in out
    assert "backend=claude" in out


def test_delete_by_name_team_succeeds_with_no_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    teams_dir = tmp_path / "teams"
    _make_team(teams_dir, "t1", agents={"dev": "claude"}, worktree=False)
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", teams_dir)

    orchestrator.main(["delete", "dev", "--team", "t1"])

    state = _read_state(teams_dir / "t1" / "agents" / "orchestrator_state.json")
    assert "dev" not in state


def test_team_help_text_names_the_layout(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        orchestrator.main(["talk", "--help"])
    assert (
        "  --team TEAM           run against team <team>'s {agents,worktree} "
        "instead of\n"
        "                        the teamless layout; found under\n"
        "                        $AGENTS_ARMY_TEAMS_DIR if set, otherwise resolved\n"
        "                        under $AGENTS_ARMY_ROOT\n" in capsys.readouterr().out
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
# The --team resolution ladder (AGENTS_ARMY_TEAMS_DIR unset): CLI behaviour
# ---------------------------------------------------------------------------


def test_zero_hits_under_root_exits_2_with_recovery_instructions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setattr(orchestrator, "ROOT", root)
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", None)

    with pytest.raises(SystemExit) as excinfo:
        orchestrator.main(["list", "agents", "--team", "nope"])

    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert (
        f"orchestrator list: error: no team named 'nope' under {root}; a "
        "team is a directory with an agents/ or worktree/ subdirectory, "
        "e.g.:\n"
        f"  git worktree add -B nope {root}/<repo>/<workflow>/nope/worktree ...\n"
        "if the team lives outside $AGENTS_ARMY_ROOT, export "
        "AGENTS_ARMY_TEAMS_DIR to point at its parent\n" in err
    )


def test_zero_hits_exits_2_for_teardown_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Unlike the TEAMS_DIR-set not-found case (exit 1, a missing
    resource), zero hits under the AGENTS_ARMY_ROOT walk is a usage
    problem — bad name, wrong environment, team lives elsewhere — and stays
    exit 2 even for teardown. Points --team at the §6 residue shape (`logs/`
    + `.lock` + `spectacle.prompt`, neither `agents/` nor `worktree/`) so
    this also covers that residue being a zero-hit case for teardown, not
    just for `teams.resolve` at the unit level.
    """
    root = tmp_path / "root"
    residue = root / "nope"
    residue.mkdir(parents=True)
    (residue / "logs").mkdir()
    (residue / ".lock").touch()
    (residue / "spectacle.prompt").touch()
    monkeypatch.setattr(orchestrator, "ROOT", root)
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", None)

    with pytest.raises(SystemExit) as excinfo:
        orchestrator.main(["delete", "--team", "nope"])

    assert excinfo.value.code == 2
    assert f"no team named 'nope' under {root}" in capsys.readouterr().err


def test_ambiguity_under_root_exits_2_and_lists_full_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "root"
    team_a = _make_team(root, "repo-a", "wf", "issue-97", agents={"dev": "claude"})
    team_b = _make_team(root, "repo-b", "wf", "issue-97", agents={"dev": "codex"})
    monkeypatch.setattr(orchestrator, "ROOT", root)
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", None)

    with pytest.raises(SystemExit) as excinfo:
        orchestrator.main(["list", "agents", "--team", "issue-97"])

    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert (
        "orchestrator list: error: team name 'issue-97' is ambiguous under "
        f"{root}:\n"
        f"  {team_a}\n"
        f"  {team_b}\n"
        "re-run with a qualified name, e.g. --team repo-a/wf/issue-97\n" in err
    )


def test_qualified_name_resolves_one_of_two_ambiguous_teams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "root"
    _make_team(root, "repo-a", "wf", "issue-97", agents={"dev": "claude"})
    _make_team(root, "repo-b", "wf", "issue-97", agents={"dev": "codex"})
    monkeypatch.setattr(orchestrator, "ROOT", root)
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", None)

    orchestrator.main(["list", "agents", "--team", "repo-a/wf/issue-97"])

    out = capsys.readouterr().out
    assert "dev" in out
    assert "backend=claude" in out


def test_teams_dir_set_short_circuits_the_root_walk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """AGENTS_ARMY_TEAMS_DIR set must short-circuit straight to
    TEAMS_DIR/NAME, with no walk and no fallback — even when a team of the
    same name is perfectly resolvable under AGENTS_ARMY_ROOT."""
    root = tmp_path / "root"
    _make_team(root, "t1", agents={"dev": "claude"})
    teams_dir = tmp_path / "teams"  # no t1 here
    monkeypatch.setattr(orchestrator, "ROOT", root)
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", teams_dir)

    with pytest.raises(SystemExit) as excinfo:
        orchestrator.main(["create", "a", "--team", "t1", "-b", "claude"])

    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert str(teams_dir / "t1" / "worktree") in err
    assert str(root) not in err


def test_delete_team_tears_down_an_agents_dir_without_the_marker_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The `_flock` residue (an `agents/` directory with no
    `orchestrator_state.json` inside it) must resolve under AGENTS_ARMY_ROOT
    and be torn down, the same as a fully-formed team."""
    root = tmp_path / "root"
    agents_dir = root / "t1" / "agents"
    agents_dir.mkdir(parents=True)
    monkeypatch.setattr(orchestrator, "ROOT", root)
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", None)

    orchestrator.main(["delete", "--team", "t1"])

    assert capsys.readouterr().out == "deleted team 't1'\n"
    assert not agents_dir.exists()


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


def test_a_boundary_caught_error_releases_the_team_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The next command must not find the team busy because the last one failed.

    The lock is a context manager inside the boundary's `try`, so it unwinds
    on the way out; this pins that, since the boundary is now the only place
    a user-facing failure turns into an exit code.
    """

    class BoomBackend(AgentBackend):
        @property
        def name(self) -> str:
            return "boom"

        def run_turn(
            self,
            prompt: str,
            session_id: str | None,
            cwd: Path,
            timeout: int = orchestrator.DEFAULT_TURN_TIMEOUT,
            schema=None,
        ) -> TurnResult:
            raise TurnError("cli failed")

    register_backend("boom", BoomBackend)
    teams_dir = tmp_path / "teams"
    (teams_dir / "t1" / "worktree").mkdir(parents=True)
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", teams_dir)

    with pytest.raises(SystemExit) as excinfo:
        orchestrator.main(["talk", "a", "--team", "t1", "-b", "boom", "-p", "hi"])

    assert excinfo.value.code == 1
    assert capsys.readouterr().err.endswith("cli failed\n")
    with (teams_dir / "t1" / ".lock").open("a+", encoding="utf-8") as lock:
        # Would raise BlockingIOError if the failed command still held it.
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


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


def test_list_teams_reports_a_team_whose_backend_plugin_is_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A registry naming a backend nobody registers any more is well-formed —
    the plugin is the thing that is missing, not the entry. `list teams` reads
    the stored backend *name* and must print it, so this stays off the loading
    path that resolves a name into a backend and raises for an unknown one."""
    root = tmp_path / "root"
    _make_team(root, "legacy", agents={"dev": "retired-plugin"})
    _make_team(root, "healthy", agents={"ops": "claude"})
    monkeypatch.setattr(orchestrator, "ROOT", root)
    monkeypatch.setattr(orchestrator, "STATE_FILE", root / "orchestrator_state.json")
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", None)

    orchestrator.main(["list", "teams"])

    out = capsys.readouterr().out
    assert "dev/retired-plugin" in out
    assert "ops/claude" in out
    assert "registry unreadable" not in out


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


# ---------------------------------------------------------------------------
# Teamless registry: the AGENTS_ARMY_ROOT default ladder
# ---------------------------------------------------------------------------


def test_state_file_ladder_prefers_explicit_state_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    explicit = tmp_path / "explicit" / "state.json"
    monkeypatch.setenv("AGENTS_ARMY_STATE_FILE", str(explicit))
    monkeypatch.setenv("AGENTS_ARMY_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("AGENTS_ARMY_ROOT", str(tmp_path / "root"))

    result = subprocess.run(
        [sys.executable, "-c", "import orchestrator; print(orchestrator.STATE_FILE)"],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).parent.parent,
    )

    assert result.stdout.strip() == str(explicit)


def test_state_file_ladder_prefers_explicit_home_over_root(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    root = tmp_path / "root"
    env = _env_without(
        "AGENTS_ARMY_STATE_FILE",
        AGENTS_ARMY_HOME=str(home),
        AGENTS_ARMY_ROOT=str(root),
    )

    result = subprocess.run(
        [sys.executable, "-c", "import orchestrator; print(orchestrator.STATE_FILE)"],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).parent.parent,
        env=env,
    )

    assert result.stdout.strip() == str(home / "orchestrator_state.json")


def test_state_file_ladder_defaults_under_root_when_nothing_else_is_set(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    env = _env_without(
        "AGENTS_ARMY_HOME", "AGENTS_ARMY_STATE_FILE", AGENTS_ARMY_ROOT=str(root)
    )

    result = subprocess.run(
        [sys.executable, "-c", "import orchestrator; print(orchestrator.STATE_FILE)"],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).parent.parent,
        env=env,
    )

    assert result.stdout.strip() == str(root / "orchestrator_state.json")


def test_root_default_state_file_does_not_consult_cwd(tmp_path: Path) -> None:
    """With AGENTS_ARMY_HOME unset, cwd must never be read for STATE_FILE —
    only AGENTS_ARMY_ROOT (or its default) may supply it."""
    root = tmp_path / "root"
    cwd = tmp_path / "somewhere-else"
    cwd.mkdir()
    env = _env_without(
        "AGENTS_ARMY_HOME", "AGENTS_ARMY_STATE_FILE", AGENTS_ARMY_ROOT=str(root)
    )

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import orchestrator; orchestrator.main(['create', 'dev', '-b', 'claude'])",
        ],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert not (cwd / "orchestrator_state.json").exists()
    assert (root / "orchestrator_state.json").exists()


def test_workdir_and_skills_dir_still_follow_cwd_under_root_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AGENTS_ARMY_HOME", raising=False)
    monkeypatch.delenv("AGENTS_ARMY_STATE_FILE", raising=False)
    monkeypatch.setenv("AGENTS_ARMY_ROOT", str(tmp_path / "root"))
    monkeypatch.chdir(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import orchestrator; print(orchestrator.WORKDIR); "
            "print(orchestrator.SKILLS_DIR)",
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
    )

    workdir, skills_dir = result.stdout.splitlines()
    assert workdir == str(tmp_path)
    assert skills_dir == str(tmp_path / "SKILLS")


def test_agents_army_root_does_not_conflict_with_team(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unlike AGENTS_ARMY_HOME/AGENTS_ARMY_STATE_FILE, AGENTS_ARMY_ROOT is the
    teamless fallback and must stay compatible with --team."""
    register_backend("recording", _make_recording_backend([]))
    teams_dir = tmp_path / "teams"
    (teams_dir / "t1" / "worktree").mkdir(parents=True)
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", teams_dir)
    monkeypatch.setenv("AGENTS_ARMY_ROOT", str(tmp_path / "root"))

    orchestrator.main(["create", "owen", "--team", "t1", "-b", "recording"])

    assert (teams_dir / "t1" / "agents" / "orchestrator_state.json").exists()


def test_two_teams_in_one_process_do_not_share_a_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second `main()` must not inherit the first run's team."""
    register_backend("recording", _make_recording_backend([]))
    teams_dir = tmp_path / "teams"
    for team in ("t1", "t2"):
        (teams_dir / team / "worktree").mkdir(parents=True)
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", teams_dir)

    orchestrator.main(["create", "first", "--team", "t1", "-b", "recording"])
    orchestrator.main(["create", "second", "--team", "t2", "-b", "recording"])

    def state(team: str) -> Path:
        return teams_dir / team / "agents" / "orchestrator_state.json"

    assert sorted(_read_state(state("t1"))) == ["first"]
    assert sorted(_read_state(state("t2"))) == ["second"]


def test_a_second_teams_turn_runs_in_its_own_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The workdir leak a registry assertion alone would miss."""
    seen: list[Path] = []
    register_backend("recording", _make_recording_backend(seen))
    teams_dir = tmp_path / "teams"
    for team in ("t1", "t2"):
        (teams_dir / team / "worktree").mkdir(parents=True)
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", teams_dir)

    orchestrator.main(["talk", "a", "--team", "t1", "-b", "recording", "-p", "hi"])
    orchestrator.main(["talk", "a", "--team", "t2", "-b", "recording", "-p", "hi"])

    assert seen == [teams_dir / "t1" / "worktree", teams_dir / "t2" / "worktree"]


def test_a_team_run_leaves_the_module_path_globals_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--team` resolves paths for the run; it never rebinds the globals."""
    register_backend("recording", _make_recording_backend([]))
    teams_dir = tmp_path / "teams"
    (teams_dir / "t1" / "worktree").mkdir(parents=True)
    monkeypatch.setattr(orchestrator, "TEAMS_DIR", teams_dir)
    before = (orchestrator.STATE_FILE, orchestrator.WORKDIR, orchestrator.SKILLS_DIR)

    orchestrator.main(["create", "a", "--team", "t1", "-b", "recording"])

    assert before == (
        orchestrator.STATE_FILE,
        orchestrator.WORKDIR,
        orchestrator.SKILLS_DIR,
    )
