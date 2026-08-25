#!/usr/bin/env python3
"""Enforce risk-based per-file branch-coverage floors.

Coverage is evidence that code ran, not that a useful behavior was asserted.
The floors therefore leave deliberate headroom instead of rewarding tests
written only to exercise every defensive branch. Core orchestration and
backend adapters carry the highest floor; quality-gate plumbing carries a
lower one. Changed lines independently need 90% coverage in pull requests.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

COVERAGE_PATH = Path("coverage.json")

# Increment this only for a deliberate, reviewed coverage-policy change. The
# ratchet permits coverage thresholds to move down only across such a version
# change, while continuing to protect every other quality threshold.
COVERAGE_POLICY_VERSION = 2

# Core orchestration and external-process adapters: failures here affect every
# invocation, so these retain the strongest coverage requirement.
FLOORS: dict[str, float] = {
    "orchestrator/__init__.py": 95.0,
    "orchestrator/schema.py": 95.0,
    "orchestrator/skills.py": 95.0,
    "backends/base.py": 95.0,
    "backends/claude.py": 95.0,
    "backends/codex.py": 95.0,
    "backends/grok.py": 95.0,
    "backends/opencode.py": 95.0,
    "backends/registry.py": 95.0,
    # Gate utilities are themselves protected by planted violations. Requiring
    # every plumbing branch to run would add ceremony without more confidence.
    "tools/coverage_gate.py": 80.0,
    "tools/mutation_cache.py": 80.0,
    "tools/mutation_gate.py": 80.0,
    "tools/ratchet_gate.py": 80.0,
    "tools/test_integrity.py": 80.0,
}

# New modules start at the supporting-code floor. Their risk can be classified
# explicitly above when they become part of a critical path.
NEW_FILE_FLOOR = 80.0


def _check_recorded_floors(
    measured: dict[str, float],
) -> list[str]:
    failures: list[str] = []
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
    return failures


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

    failures = _check_recorded_floors(measured)
    failures.extend(_check_new_files(measured))

    for failure in failures:
        print(f"error: {failure}")

    if failures:
        print(f"\n{len(failures)} per-file coverage failure(s).")
        return 1

    print(f"per-file coverage: {len(FLOORS)} files at or above their floors.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
