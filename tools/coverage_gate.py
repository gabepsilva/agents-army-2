#!/usr/bin/env python3
"""Enforce a per-file branch-coverage floor that only ever ratchets upward.

A single project-wide threshold lets a well-covered module pay for an
uncovered one. Each file therefore carries its own floor, recorded at the
value it had when the gate was introduced. Raise an entry when coverage
improves so the gain is locked in. Never lower one — that is the whole point
of a ratchet.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

COVERAGE_PATH = Path("coverage.json")

# Recorded 2026-08-14 when the deterministic gate stack was introduced. Every
# file measured 100% branch coverage at that point (a handful of missing CLI
# and event-parsing branches were closed rather than floored down), so there
# is no legacy slack to grandfather in.
FLOORS: dict[str, float] = {
    "orchestrator/__init__.py": 100.0,
    "orchestrator/schema.py": 100.0,
    "orchestrator/skills.py": 100.0,
    "backends/base.py": 100.0,
    "backends/claude.py": 100.0,
    "backends/codex.py": 100.0,
    "backends/grok.py": 100.0,
    "backends/registry.py": 100.0,
    "tools/coverage_gate.py": 100.0,
    "tools/mutation_cache.py": 100.0,
    "tools/mutation_gate.py": 100.0,
    "tools/ratchet_gate.py": 100.0,
    "tools/test_integrity.py": 100.0,
}

# A module added after this gate existed has no legacy excuse.
NEW_FILE_FLOOR = 60.0


def _check_recorded_floors(
    measured: dict[str, float],
) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    improvements: list[str] = []
    for path, floor in sorted(FLOORS.items()):
        if path not in measured:
            failures.append(
                f"{path}: has a recorded floor but was not measured. Remove its "
                f"FLOORS entry if the file is gone."
            )
            continue
        actual = measured[path]
        if actual < floor:
            failures.append(f"{path}: {actual:.1f}% is below its floor of {floor:.1f}%")
        # Report a full point of headroom so the ratchet gets turned.
        elif math.floor(actual) >= floor + 1:
            improvements.append(
                f"{path}: {actual:.1f}% — raise its floor to {math.floor(actual):.0f}"
            )
    return failures, improvements


def _check_new_files(measured: dict[str, float]) -> list[str]:
    return [
        f"{path}: new file at {actual:.1f}% must reach "
        f"{NEW_FILE_FLOOR:.0f}% or record an explicit floor with a reason."
        for path, actual in sorted(measured.items())
        if path not in FLOORS and actual < NEW_FILE_FLOOR
    ]


def main() -> int:
    if not COVERAGE_PATH.exists():
        print(f"error: {COVERAGE_PATH} is missing; run `make test-coverage` first.")
        return 1

    report = json.loads(COVERAGE_PATH.read_text(encoding="utf-8"))
    measured = {
        path: data["summary"]["percent_covered"]
        for path, data in report["files"].items()
    }

    failures, improvements = _check_recorded_floors(measured)
    failures.extend(_check_new_files(measured))

    for failure in failures:
        print(f"error: {failure}")
    for improvement in improvements:
        print(f"note: {improvement}")

    if failures:
        print(f"\n{len(failures)} per-file coverage failure(s).")
        return 1

    print(f"per-file coverage: {len(FLOORS)} files at or above their floors.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
