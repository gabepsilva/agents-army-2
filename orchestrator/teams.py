"""Discover team directories under a root, without reading their registries.

A team is a directory containing `agents/orchestrator_state.json` (see
`MARKER`). This module only walks the filesystem to find those directories;
reading a team's registry contents (agent names, backends) is the caller's
job — see `orchestrator._team_agents`.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

MARKER = ("agents", "orchestrator_state.json")

# $ROOT/<repo>/<workflow>/<team>/ puts a team at depth 3 and its worktree/ at
# depth 4, so 4 is exactly one level of headroom, not slack. It is also the
# only thing bounding the walk: a directory missing agents/ is not a team, so
# the never-descend rule in `_find_team_dirs` can't protect it, and the walk
# does enter its worktree/ at depth 4 — costing one scandir plus one stat per
# child. Raising it "for headroom" turns `list teams` into a crawl of
# whatever tree happens to sit under the root.
SEARCH_DEPTH = 4


@dataclass(frozen=True)
class Team:
    """One discovered team: where it lives, its display name, and whether
    its worktree is still there (a team with none is the orphan
    `delete --team` teardown deliberately leaves behind)."""

    path: Path
    name: str
    has_worktree: bool


def marker_path(team_dir: Path) -> Path:
    return team_dir.joinpath(*MARKER)


def is_team(path: Path) -> bool:
    return marker_path(path).is_file()


def _find_team_dirs(directory: Path, child_depth: int) -> Iterator[Path]:
    """Yield team directories under `directory`.

    `directory`'s children are candidates at `child_depth`. Each is checked
    for the marker file with a `stat` (via `is_team`), never a `scandir` of
    it. A candidate that is a team is never descended into — that is the
    only thing making the decoy case (a team whose own `worktree/` contains
    another `agents/orchestrator_state.json`) yield one team, not two.
    Candidates below `SEARCH_DEPTH` are `stat`ed but never `scandir`ed, so
    nothing past that depth is touched.
    """
    try:
        with os.scandir(directory) as it:
            entries = list(it)
    except OSError:
        return
    for entry in entries:
        # follow_symlinks=False: a symlink is never treated as a directory to
        # scan, so a symlinked worktree/ pointing back at an ancestor can't
        # turn into a loop — it is simply never entered.
        if not entry.is_dir(follow_symlinks=False):
            continue
        candidate = Path(entry.path)
        if is_team(candidate):
            yield candidate
        elif child_depth < SEARCH_DEPTH:
            yield from _find_team_dirs(candidate, child_depth + 1)


def discover(group_root: Path) -> list[Team]:
    """Every team under `group_root`, sorted by its path relative to it."""
    teams = [
        Team(
            path=team_dir,
            name=team_dir.relative_to(group_root).as_posix(),
            has_worktree=(team_dir / "worktree").is_dir(),
        )
        for team_dir in _find_team_dirs(group_root, 1)
    ]
    return sorted(teams, key=lambda team: team.name)
