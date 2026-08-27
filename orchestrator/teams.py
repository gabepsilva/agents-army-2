"""Discover or resolve team directories under a root, without reading their
registries.

This module has two walks, built on two deliberately different predicates:

- `discover` (for `list teams`) treats a directory as a team once its
  registry file exists — `is_team`, keyed on `MARKER`. A directory without
  one is not shown, even if it is clearly a team's worktree-only shell.
- `resolve` (for `--team NAME`) treats a directory as *nameable* the moment
  it has an `agents/` or `worktree/` subdirectory, whether or not the
  registry file inside `agents/` has been written yet — see `resolve`'s own
  docstring for why that has to be looser than `is_team`.

Both walks still stop descending at the same place: once a directory *is* a
team (`is_team`, the marker), its children are that team's own worktree
contents, not further teams, and are never scanned. Widening never-descend
to match `resolve`'s looser candidate rule would let one stray, unmarked
`agents/` directory hide every real team beneath it — see `resolve`.

Reading a team's registry contents (agent names, backends) is always the
caller's job — see `orchestrator._team_agents`.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

MARKER = ("agents", "orchestrator_state.json")

# $AGENTS_ARMY_ROOT/<repo>/<workflow>/<team>/ puts a team at depth 3 and its worktree/ at
# depth 4, so 4 is exactly one level of headroom, not slack. It is also the
# only thing bounding the walk: a directory missing agents/ is not a team, so
# the never-descend rule in `_walk` can't protect it, and the walk
# does enter its worktree/ at depth 4 — costing one scandir plus one stat per
# child. Raising it "for headroom" turns `list teams` into a crawl of
# whatever tree happens to sit under the root. `resolve` shares this bound:
# its candidate rule is looser than `is_team`, so it descends into
# marker-less directories `discover` never enters, and this cap is what
# keeps that descent from becoming unbounded too.
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


def _walk(
    directory: Path,
    child_depth: int,
    classify: Callable[[Path], tuple[bool, bool]],
) -> Iterator[Path]:
    """Shared depth-first walk behind both `discover` and `resolve`.

    `classify(candidate)` returns `(is_hit, stop_descending)` together, from
    one call, rather than two separate predicates: what to *yield*
    (`is_hit`) and what counts as proof you are standing *inside* a real
    team root, so its children are that team's own worktree contents rather
    than further teams (`stop_descending`). One call matters because
    `discover`'s `classify` answers both from the same underlying check
    (`is_team`, the marker) — two separate predicate callables would `stat`
    the marker file twice per candidate for no reason. `resolve`'s
    `classify` answers them from two genuinely different checks; see its own
    docstring for why unifying *those* would be wrong.

    `directory`'s children are checked at `child_depth`, each with a `stat`
    (inside `classify`), never a `scandir`. Candidates below `SEARCH_DEPTH`
    are `stat`ed but never `scandir`ed, so nothing past that depth is
    touched. `follow_symlinks=False`: a symlink is never treated as a
    directory to scan, so a symlinked `worktree/` pointing back at an
    ancestor can't turn into a loop — it is simply never entered.
    """
    try:
        with os.scandir(directory) as it:
            entries = list(it)
    except OSError:
        return
    for entry in entries:
        if not entry.is_dir(follow_symlinks=False):
            continue
        candidate = Path(entry.path)
        is_hit, stop_descending = classify(candidate)
        if is_hit:
            yield candidate
        if not stop_descending and child_depth < SEARCH_DEPTH:
            yield from _walk(candidate, child_depth + 1, classify)


def _classify_team(path: Path) -> tuple[bool, bool]:
    hit = is_team(path)
    return hit, hit


def discover(group_root: Path) -> list[Team]:
    """Every team under `group_root`, sorted by its path relative to it."""
    teams = [
        Team(
            path=team_dir,
            name=team_dir.relative_to(group_root).as_posix(),
            has_worktree=(team_dir / "worktree").is_dir(),
        )
        for team_dir in _walk(group_root, 1, _classify_team)
    ]
    return sorted(teams, key=lambda team: team.name)


def is_candidate(path: Path) -> bool:
    """Is `path` something `resolve` is allowed to name a team?

    Looser than `is_team` on purpose: `--team NAME` has to reach a team
    before its registry exists (a freshly `git worktree add`ed team, with
    only `worktree/`) and a team whose registry directory exists but whose
    marker file does not (the residue `_flock` leaves behind when a command
    takes the registry lock and fails before persisting — see
    `orchestrator._flock`). Both are `agents/`-the-directory, not the marker
    file inside it.
    """
    return (path / "agents").is_dir() or (path / "worktree").is_dir()


def _classify_candidate(path: Path) -> tuple[bool, bool]:
    return is_candidate(path), is_team(path)


def resolve(root: Path, name: str) -> list[Path]:
    """Every candidate directory under `root` whose path relative to `root`
    ends in `name`'s segments, sorted by path.

    Walks with `is_candidate` as what to yield and `is_team` (the marker) as
    what stops descent — deliberately different predicates. Keying
    never-descend on `is_candidate` instead would let one stray, unmarked
    `agents/` two levels up hide every marked team beneath it from every
    name — see the module docstring and the PR description's
    shadow-regression repro.

    `name` may be a single segment (`issue-97`) or a `/`-qualified tail
    (`agents-army-2/gdw-v3/issue-97`), matched against the same
    root-relative POSIX path `Team.name` carries — segment-wise, not
    `str.endswith()`, so `--team 97` does not match `issue-97`.

    Always returns a list, 0, 1, or many — the caller decides what each
    count means; this function never guesses. See `SEARCH_DEPTH` for the
    bound on how deep either walk goes.
    """
    name_parts = tuple(name.split("/"))
    matches = [
        candidate
        for candidate in _walk(root, 1, _classify_candidate)
        if candidate.relative_to(root).parts[-len(name_parts) :] == name_parts
    ]
    return sorted(matches, key=lambda path: path.relative_to(root).as_posix())
