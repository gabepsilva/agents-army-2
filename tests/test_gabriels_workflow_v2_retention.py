"""Behavior tests for pruning closed V2 issue state directories."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.gabriels_workflow_v2 import prune
from examples.gabriels_workflow_v2.config import RetentionConfig
from examples.gabriels_workflow_v2.contracts import CheckpointStore
from examples.gabriels_workflow_v2.errors import WorkflowError
from examples.gabriels_workflow_v2.retention import prune_issue_state

NOW = datetime(2026, 8, 23, tzinfo=UTC)


class FakeRepository:
    """Spies on `worktree list`/`remove` without touching real git state."""

    def __init__(self, worktrees: list[str]) -> None:
        self.worktrees = worktrees
        self.calls: list[tuple[str, ...]] = []

    def _call(self, *args: str) -> str:
        self.calls.append(args)
        if args == ("worktree", "list", "--porcelain"):
            return "".join(f"worktree {path}\n\n" for path in self.worktrees)
        if args[:2] == ("worktree", "remove"):
            self.worktrees.remove(args[-1])
            return ""
        raise AssertionError(f"unexpected git call: {args}")


def _issue(gdw_root: Path, number: int, *, complete: bool) -> Path:
    issue_root = gdw_root / f"issue-{number}"
    store = CheckpointStore(issue_root)
    store.initialize(number, f"gdwv2/issue-{number}", "base-sha")
    if complete:
        store.update_metadata(complete=True, pr_number=1, pr_url="https://x/1")
    return issue_root


def _age(issue_root: Path, days: float) -> None:
    completed_at = (NOW - timedelta(days=days)).isoformat()
    store = CheckpointStore(issue_root)
    store.update_metadata(completed_at=completed_at)


def test_prune_removes_a_completed_run_past_completed_retention(
    tmp_path: Path,
) -> None:
    gdw_root = tmp_path / "gdw-v2"
    old = _issue(gdw_root, 1, complete=True)
    _age(old, 8)
    repository = FakeRepository([])

    removed = prune_issue_state(
        gdw_root, RetentionConfig(), repository, now=NOW, dry_run=False
    )

    assert removed == [old]
    assert not old.exists()


def test_prune_leaves_an_in_flight_run_and_it_still_resumes(tmp_path: Path) -> None:
    gdw_root = tmp_path / "gdw-v2"
    active = _issue(gdw_root, 2, complete=False)
    (active / "checkpoints").mkdir()
    (active / "checkpoints" / "expansion-1.json").write_text("{}", encoding="utf-8")
    repository = FakeRepository([])

    removed = prune_issue_state(
        gdw_root, RetentionConfig(), repository, now=NOW, dry_run=False
    )

    assert removed == []
    assert active.exists()
    store = CheckpointStore(active)
    assert store.metadata["issue"] == 2
    assert (active / "checkpoints" / "expansion-1.json").exists()


def test_prune_removes_a_legacy_completed_run_via_mtime_fallback(
    tmp_path: Path,
) -> None:
    gdw_root = tmp_path / "gdw-v2"
    legacy = _issue(gdw_root, 3, complete=True)
    old_time = (NOW - timedelta(days=8)).timestamp()
    os.utime(legacy / "workflow.json", (old_time, old_time))
    repository = FakeRepository([])

    removed = prune_issue_state(
        gdw_root, RetentionConfig(), repository, now=NOW, dry_run=False
    )

    assert removed == [legacy]
    assert not legacy.exists()


def test_prune_removes_mode_000_overlay_work_directories_without_prep_chmod(
    tmp_path: Path,
) -> None:
    gdw_root = tmp_path / "gdw-v2"
    old = _issue(gdw_root, 4, complete=True)
    _age(old, 8)
    locked = old / "agents" / "home" / "gdw-4-expander" / ".claude" / "work" / "work"
    locked.mkdir(parents=True)
    (locked / "file").write_text("data", encoding="utf-8")
    locked.chmod(0o000)
    repository = FakeRepository([])

    try:
        removed = prune_issue_state(
            gdw_root, RetentionConfig(), repository, now=NOW, dry_run=False
        )
        assert removed == [old]
        assert not old.exists()
    finally:
        # A mutant that breaks the chmod fail-safe leaves this mode-000
        # directory behind; restore it so pytest's own tmp-dir cleanup
        # doesn't choke on it after such a mutant, independent of whether
        # the assertions above already failed the test.
        if locked.exists():
            locked.chmod(0o700)


def test_prune_dry_run_lists_candidates_without_deleting_or_touching_git(
    tmp_path: Path,
) -> None:
    gdw_root = tmp_path / "gdw-v2"
    old = _issue(gdw_root, 5, complete=True)
    _age(old, 8)
    (old / "worktree").mkdir()
    repository = FakeRepository([str(old / "worktree")])

    removed = prune_issue_state(
        gdw_root, RetentionConfig(), repository, now=NOW, dry_run=True
    )

    assert removed == [old]
    assert old.exists()
    assert repository.calls == []


def test_prune_removes_the_registered_worktree_before_deleting_the_tree(
    tmp_path: Path,
) -> None:
    gdw_root = tmp_path / "gdw-v2"
    old = _issue(gdw_root, 6, complete=True)
    _age(old, 8)
    worktree = old / "worktree"
    worktree.mkdir()
    repository = FakeRepository([str(worktree)])

    removed = prune_issue_state(
        gdw_root, RetentionConfig(), repository, now=NOW, dry_run=False
    )

    assert removed == [old]
    assert not old.exists()
    assert ("worktree", "list", "--porcelain") in repository.calls
    assert ("worktree", "remove", "--force", str(worktree)) in repository.calls
    remove_index = repository.calls.index(
        ("worktree", "remove", "--force", str(worktree))
    )
    list_index = repository.calls.index(("worktree", "list", "--porcelain"))
    assert list_index < remove_index
    assert str(worktree) not in repository.worktrees


def test_prune_hard_ceiling_removes_stale_in_flight_run(tmp_path: Path) -> None:
    gdw_root = tmp_path / "gdw-v2"
    stale = _issue(gdw_root, 7, complete=False)
    old_time = (NOW - timedelta(days=31)).timestamp()
    os.utime(stale / "workflow.json", (old_time, old_time))
    repository = FakeRepository([])

    removed = prune_issue_state(
        gdw_root, RetentionConfig(), repository, now=NOW, dry_run=False
    )

    assert removed == [stale]
    assert not stale.exists()


def test_prune_removes_a_metadata_less_directory_past_the_hard_ceiling(
    tmp_path: Path,
) -> None:
    gdw_root = tmp_path / "gdw-v2"
    orphan = gdw_root / "issue-8"
    (orphan / "worktree").mkdir(parents=True)
    old_time = (NOW - timedelta(days=31)).timestamp()
    os.utime(orphan / "worktree", (old_time, old_time))
    os.utime(orphan, (old_time, old_time))
    repository = FakeRepository([])

    removed = prune_issue_state(
        gdw_root, RetentionConfig(), repository, now=NOW, dry_run=False
    )

    assert removed == [orphan]
    assert not orphan.exists()


def test_prune_falls_back_to_mtime_for_a_naive_completed_at(tmp_path: Path) -> None:
    gdw_root = tmp_path / "gdw-v2"
    old = _issue(gdw_root, 9, complete=True)
    store = CheckpointStore(old)
    store.update_metadata(completed_at="2020-01-01T00:00:00")
    old_time = (NOW - timedelta(days=8)).timestamp()
    os.utime(old / "workflow.json", (old_time, old_time))
    repository = FakeRepository([])

    removed = prune_issue_state(
        gdw_root, RetentionConfig(), repository, now=NOW, dry_run=False
    )

    assert removed == [old]
    assert not old.exists()


def test_prune_cli_prints_removed_paths_and_reports_workflow_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    common_dir = tmp_path / "common"
    candidate = common_dir / "gdw-v2" / "issue-1"
    monkeypatch.setattr(
        prune, "load_config", lambda _path: SimpleNamespace(retention=RetentionConfig())
    )
    monkeypatch.setattr(
        prune,
        "RelayRepository",
        lambda _root: SimpleNamespace(common_git_dir=lambda: common_dir),
    )
    monkeypatch.setattr(
        prune, "prune_issue_state", lambda *_args, **_kwargs: [candidate]
    )

    assert prune.main(["--dry-run", "-v"]) == 0
    assert capsys.readouterr().out.strip() == str(candidate)

    monkeypatch.setattr(
        prune,
        "load_config",
        lambda _path: (_ for _ in ()).throw(WorkflowError("boom")),
    )
    assert prune.main([]) == 1
    assert "V2 prune stopped: boom" in capsys.readouterr().err

    with pytest.raises(SystemExit):
        prune._parser().parse_args(["--unknown-flag"])


def test_prune_never_removes_the_issue_currently_being_prepared(
    tmp_path: Path,
) -> None:
    gdw_root = tmp_path / "gdw-v2"
    current = _issue(gdw_root, 30, complete=True)
    _age(current, 40)
    (current / "worktree").mkdir()
    sibling = _issue(gdw_root, 31, complete=True)
    _age(sibling, 40)
    repository = FakeRepository([])

    removed = prune_issue_state(
        gdw_root,
        RetentionConfig(),
        repository,
        now=NOW,
        dry_run=False,
        skip=current / "worktree" / "..",
    )

    assert removed == [sibling]
    assert current.exists()
    assert CheckpointStore(current).metadata["issue"] == 30
    assert not sibling.exists()


def test_prune_steps_over_a_directory_it_cannot_remove(tmp_path: Path) -> None:
    gdw_root = tmp_path / "gdw-v2"
    stuck = _issue(gdw_root, 40, complete=True)
    unreadable = _issue(gdw_root, 41, complete=True)
    survivor = _issue(gdw_root, 42, complete=True)
    for issue_root in (stuck, unreadable, survivor):
        _age(issue_root, 8)
        (issue_root / "worktree").mkdir()

    class RefusingRepository(FakeRepository):
        def _call(self, *args: str) -> str:
            if args[:2] == ("worktree", "remove"):
                if args[-1] == str(stuck / "worktree"):
                    raise WorkflowError("git worktree remove --force failed")
                if args[-1] == str(unreadable / "worktree"):
                    raise PermissionError("cannot unlink administrative files")
            return super()._call(*args)

    repository = RefusingRepository(
        [str(issue / "worktree") for issue in (stuck, unreadable, survivor)]
    )

    removed = prune_issue_state(
        gdw_root, RetentionConfig(), repository, now=NOW, dry_run=False
    )

    assert removed == [survivor]
    assert not survivor.exists()
    assert stuck.exists()
    assert unreadable.exists()


def test_retention_config_ceiling_must_cover_the_completed_threshold() -> None:
    assert RetentionConfig(completed_retention_days=9, max_retention_days=9)
    with pytest.raises(
        ValueError, match="max_retention_days must be at least completed_retention_days"
    ):
        RetentionConfig(completed_retention_days=10, max_retention_days=9)
