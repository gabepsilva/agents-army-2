"""Reclaim per-issue state left under `$GIT_COMMON_DIR/gdw-v2` after a run ends."""

from __future__ import annotations

import contextlib
import os
import shutil
import stat
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Protocol

from examples.gabriels_workflow_v2.config import RetentionConfig
from examples.gabriels_workflow_v2.contracts import CheckpointStore
from examples.gabriels_workflow_v2.errors import LOGGER, WorkflowError


class WorktreeRepository(Protocol):
    """The one git primitive pruning needs: running a raw git subcommand."""

    def _call(self, *args: str) -> str: ...


def _chmod_and_retry(func: Callable[..., object], path: str, exc_info: object) -> None:
    """`shutil.rmtree` fail-safe for overlayfs's mode-000 work directories.

    A sandboxed turn leaves some overlay work directories at mode 000, so a
    plain `rmtree` raises `PermissionError` on them. Restoring owner
    read/write/execute clears the immediate cause; the failing operation is
    then retried directly for `os.rmdir`/`os.unlink`, or by recursing back
    into `rmtree` for the fd-based `os.open` step Python's newer rmtree
    implementation uses, which needs more than the bare path to retry.
    """

    del exc_info
    os.chmod(path, stat.S_IRWXU)
    if func in (os.rmdir, os.unlink, os.remove):
        func(path)
    else:
        shutil.rmtree(path, onerror=_chmod_and_retry)


def prune_issue_state(
    gdw_root: Path,
    retention: RetentionConfig,
    repository: WorktreeRepository,
    *,
    now: datetime,
    dry_run: bool = False,
    skip: Path | None = None,
) -> list[Path]:
    """Remove completed and stale issue-N directories under `gdw_root`.

    A completed run (`complete=True`) is removed once
    `retention.completed_retention_days` have passed since `completed_at`
    (falling back to the state directory's `workflow.json` mtime for runs
    recorded before that field existed, or the issue directory's own mtime
    when even `workflow.json` is missing). Any issue, complete or not, is
    removed once `retention.max_retention_days` have passed, so an
    abandoned in-flight run does not linger forever. `dry_run` returns the
    same candidate list without deleting anything or touching git's worktree
    registry.

    `skip` names one issue directory to leave alone whatever its age, because
    a run that prunes while preparing its own state would delete the tree it
    is about to resume; it is matched on the resolved path so a caller that
    reaches the same directory by another spelling still protects it. One
    directory that refuses to go — a worktree registration `git worktree
    remove --force` chokes on, an `rmtree` that fails — is logged and stepped
    over rather than abandoning every candidate behind it, and the returned
    list names only what was actually removed.
    """

    protected = skip.resolve() if skip is not None else None
    removed = []
    for issue_root in sorted(gdw_root.glob("issue-*")):
        if not issue_root.is_dir() or issue_root.resolve() == protected:
            continue
        age_days = _age_days(issue_root, now)
        if age_days is None or not _prunable(issue_root, age_days, retention):
            continue
        if dry_run:
            removed.append(issue_root)
            continue
        try:
            _remove(issue_root, repository)
        except (WorkflowError, OSError) as exc:
            LOGGER.warning("retention: leaving %s in place: %s", issue_root, exc)
            continue
        removed.append(issue_root)
    return removed


def _age_days(issue_root: Path, now: datetime) -> float | None:
    completed_at = None
    with contextlib.suppress(WorkflowError):
        completed_at = CheckpointStore(issue_root).metadata.get("completed_at")
    if isinstance(completed_at, str):
        try:
            return (now - datetime.fromisoformat(completed_at)).total_seconds() / 86_400
        except (ValueError, TypeError):
            pass
    metadata_path = issue_root / "workflow.json"
    try:
        mtime = metadata_path.stat().st_mtime
    except OSError:
        try:
            mtime = issue_root.stat().st_mtime
        except OSError:
            return None
    return (now.timestamp() - mtime) / 86_400


def _prunable(issue_root: Path, age_days: float, retention: RetentionConfig) -> bool:
    if age_days >= retention.max_retention_days:
        return True
    try:
        complete = CheckpointStore(issue_root).metadata.get("complete") is True
    except WorkflowError:
        complete = False
    return complete and age_days >= retention.completed_retention_days


def _remove(issue_root: Path, repository: WorktreeRepository) -> None:
    worktree = issue_root / "worktree"
    registered = {
        line.split(" ", 1)[1]
        for line in repository._call("worktree", "list", "--porcelain").splitlines()
        if line.startswith("worktree ")
    }
    if str(worktree) in registered:
        repository._call("worktree", "remove", "--force", str(worktree))
    LOGGER.info("retention: removing %s", issue_root)
    shutil.rmtree(issue_root, onerror=_chmod_and_retry)
