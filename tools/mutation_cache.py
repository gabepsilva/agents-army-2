#!/usr/bin/env python3
"""Drop mutmut's cache when the tests that judge its mutants have changed.

mutmut re-runs a mutant when the mutated source changes, but not when the
tests do: on 2026-08-21 a run against edited tests returned in 8 seconds
reporting 1751/1751 already complete, while deleting `mutants/` and re-running
took 83 seconds and produced a different score. A stale score is worse than a
slow one here, because a mutation score exists to answer whether the tests
would notice a defect — so the one edit that must invalidate it is an edit to
those tests, and that is exactly the edit mutmut ignores.

The selected tests are hashed and the digest kept beside the cache. A digest
that no longer matches means the cached verdicts were reached by a different
suite, so the cache is removed and mutmut measures again. An unchanged digest
keeps the fast path.

The digest is recorded only by `--record`, which the Makefile runs *after*
mutmut has measured. Writing it up front would mark a suite as measured before
anything measured it: any other invocation — a developer's, or an agent's
mid-edit — would stamp the digest, and the next run would reuse a cache no
mutmut pass had ever validated against those tests.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import tomllib
from pathlib import Path

MUTANTS_DIR = Path("mutants")
DIGEST_PATH = Path("reports") / "mutation-test-inputs.sha256"
PYPROJECT_PATH = Path("pyproject.toml")


def selected_tests(pyproject: Path = PYPROJECT_PATH) -> list[Path]:
    """The test files mutmut runs against each mutant, from [tool.mutmut]."""
    config = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    selection = config["tool"]["mutmut"]["pytest_add_cli_args_test_selection"]
    return sorted(Path(entry) for entry in selection)


def digest(paths: list[Path]) -> str:
    """One digest over every selected test file, path and content alike."""
    running = hashlib.sha256()
    for path in paths:
        running.update(path.as_posix().encode("utf-8"))
        running.update(b"\0")
        running.update(path.read_bytes())
        running.update(b"\0")
    return running.hexdigest()


def main(argv: list[str] | None = None) -> int:
    recording = list(sys.argv[1:] if argv is None else argv) == ["--record"]
    try:
        current = digest(selected_tests(PYPROJECT_PATH))
    except (OSError, KeyError, tomllib.TOMLDecodeError) as exc:
        print(f"error: cannot hash the mutmut test selection: {exc}")
        return 1

    if recording:
        DIGEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        DIGEST_PATH.write_text(current + "\n", encoding="utf-8")
        return 0

    previous = None
    if DIGEST_PATH.exists():
        previous = DIGEST_PATH.read_text(encoding="utf-8").strip()

    if previous == current:
        print("mutation cache: tests unchanged; reusing cached mutant results.")
        return 0

    # The digest is not rewritten here: only a completed run may claim these
    # tests were measured.
    if MUTANTS_DIR.exists():
        shutil.rmtree(MUTANTS_DIR)
        print("mutation cache: tests changed since the cached run; measuring again.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
