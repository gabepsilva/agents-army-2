#!/usr/bin/env python3
"""Fail when the mutation score drops below the recorded floor.

Line coverage proves a line ran. A mutation score proves a test would notice
if that line were wrong: mutmut corrupts the source, reruns the suite, and a
surviving mutant is a change no assertion detected. That is the one signal a
test written only to pass cannot fake, so it is a merge gate rather than a
report.

Raise MUTATION_SCORE_FLOOR as survivors are killed. Never lower it.

Recorded 2026-08-19 at 859/876 (98.1%). The survivors are equivalent
mutants — they cannot change behavior, so no test can detect them:

  * `encoding="utf-8"` -> `encoding="UTF-8"` / `None` (Orchestrator._load_state,
    Orchestrator._persist): codec names are case-insensitive; None falls
    back to UTF-8 wherever this runs.
  * `default="claude"` -> `default="CLAUDE"` (Orchestrator.spawn,
    cmd_spawn's --backend flag): get_backend() lower()s the name.
  * `catalog.get(name, [])` -> `catalog.get(name, None)` (skills.resolve_skills):
    a missing name still hits `if not matches`.
  * `self.agents = {}` -> `None` in Orchestrator.__init__: `_reload()`
    replaces it immediately.
  * `str(exc)` -> `str(None)` in the empty-args branch of main(): every
    KeyError/ValueError/JSONDecodeError we raise has args.

Do not chase this last 1.9%, and do not silence it with
`# pragma: no mutate` either — an equivalent mutant is evidence the code is
precise, not evidence a test is missing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

MUTATION_SCORE_FLOOR = 98.0
STATS_PATH = Path("mutants/mutmut-cicd-stats.json")


def main() -> int:
    if not STATS_PATH.exists():
        print(f"error: {STATS_PATH} is missing; run `make mutation` first.")
        return 1

    stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
    killed = stats["killed"]
    total = stats["total"]
    if total == 0:
        print("error: no mutants were generated; check [tool.mutmut] source_paths.")
        return 1

    # A suspicious or timed-out mutant is not evidence of a detecting test, so
    # only explicit kills count toward the score.
    score = 100.0 * killed / total
    print(
        f"mutation score {score:.1f}% "
        f"({killed} killed, {stats['survived']} survived, {total} mutants) "
        f"floor {MUTATION_SCORE_FLOOR:.1f}%"
    )

    if score < MUTATION_SCORE_FLOOR:
        print(
            "error: mutation score fell below the floor. Add assertions that "
            "distinguish correct output from the surviving mutants "
            "(`uv run mutmut results`, then `uv run mutmut show <mutant>`)."
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
