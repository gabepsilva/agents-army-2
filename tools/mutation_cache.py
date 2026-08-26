#!/usr/bin/env python3
"""Drop mutmut's cache when anything that decides a verdict has changed.

mutmut hashes what it mutates: every `[tool.mutmut] source_paths` file, per
function, invalidated individually when its AST changes. This file hashes
everything else that can decide a verdict — the selected tests, every other
.py file mutmut copies into `mutants/` without hashing, and the pytest
configuration that governs how those tests run. Analysis done against
mutmut 3.7.0, installed via `uv sync --locked`; reproduce any line a comment
here points at with
`sed -n '<range>p' .venv/lib/python*/site-packages/mutmut/__main__.py`.

Known gap, accepted rather than closed (agents-army-2#86): mutmut's
per-function hash covers only `cst.FunctionDef` bodies
(mutmut/mutation/file_mutation.py's `_compute_mutated_function_hashes`), so a
module-level change in a `source_paths` file — a top-level constant, an
import, a class attribute set outside a method — is invisible to both
mutmut's own hash and this digest, which deliberately excludes
`source_paths` (see digest_inputs below). Widening either would re-hash a
mutated file wholesale on any edit and delete the per-commit reuse this
cache exists for.

The digest is recorded only by `--record`, which the Makefile runs after
mutmut has measured *and* mutation_gate.py has cleared the floor. Writing it
up front, or after a failing gate, would mark a suite as measured before
anything measured it.

Checks A-D run on both invocations (bare and `--record`), before anything
else, and block a result from being trusted or recorded:
  A. closure    every non-source .py file under a watched root must be in the
                digest set or excused in DIGEST_EXCLUSIONS. mutmut's own git
                change detection drops .py files outright (it assumes the
                per-function hashes already track them, which is only true
                inside source_paths), so nothing else in this pipeline would
                have caught a tests/conftest.py edit.
  B. exclusions DIGEST_EXCLUSIONS is closure's off switch and no other gate
                reads it, so it is bounded: no globs, a written reason, and
                the named file must already exist.
  C. copying    every source_paths entry must resolve under a declared
                also_copy root, which is what keeps mutmut's mtime shortcut
                (create_mutants_for_file) unreachable.
  D. selection  every tracked file under a watched root whose basename
                matches pytest's python_files must be in
                pytest_add_cli_args_test_selection or excused in
                DIGEST_EXCLUSIONS with a reason. Closure (Check A) hashes a
                test file the moment it exists, selection is a separate
                fact mutmut decides on its own by filename, and the two can
                disagree: a hashed-but-unselected file correctly moves the
                digest, and mutmut still never collects it, so every mutant
                it alone kills is reported survived. Hashed is not run.

DIGEST_EXCLUSIONS is read by both Check A and Check D, and the two readings
cannot diverge by construction, not by convention. Check A's question about
an excused file is whether it can decide a verdict; Check D's is whether it
is allowed to sit outside the selection. They resolve to one fact: a file
mutmut never collects can never decide a verdict, so excusing it is
correct; a file mutmut does collect can decide a verdict, so Check A's own
closure walk already hashes it and Check D already requires it selected.
Excusing a selected file would claim, in the same dict, both that it
decides nothing and that it is running against every mutant — Check D
rejects that combination rather than letting the contradiction stand.

tests/test_quality_gates.py is the repo's single exclusion because it
shells out to `make` against a path derived from `__file__`
(tests/test_quality_gates.py:25, TestCiGateAnnouncements._make). Inside a
mutant run that path resolves to `mutants/`, which has no Makefile, so
every one of its `make` assertions fails on the baseline collect. It tests
the build, not the code under mutation: it cannot decide a mutant's
verdict, and it cannot survive being collected inside one.

Check E clears the gate's own stats file before measurement, on both the
reuse path and the rmtree path, so a stale mutmut-cicd-stats.json inside a
restored mutants/ can never be read by mutation_gate.py.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import shlex
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

MUTANTS_DIR = Path("mutants")
DIGEST_PATH = Path("reports") / "mutation-test-inputs.sha256"
PYPROJECT_PATH = Path("pyproject.toml")

# Roots whose contents can decide a mutant's verdict without mutmut noticing.
#   - [tool.mutmut] also_copy: read from pyproject, not hardcoded, below.
#   - tests/: mutmut appends it to also_copy implicitly (configuration.py's
#     read_config), so it is copied into mutants/ on every run and never
#     appears in our own config.
#   - tools/nodump/: on PYTHONPATH for the whole `mutation` recipe, so Python
#     imports sitecustomize.py into the gate's own process at startup.
IMPLICIT_WATCHED_ROOTS = ("tests/", "tools/nodump/")

# Paths under a watched root that provably cannot decide a verdict.
# path -> the reason. Bounded by _check_exclusions below.
DIGEST_EXCLUSIONS: dict[str, str] = {
    "tests/test_quality_gates.py": (
        "shells out to `make` against a path derived from __file__, which "
        "resolves to mutants/ (no Makefile there) inside a mutant run, so "
        "every one of its make assertions fails on the baseline collect — "
        "see the module docstring for the full argument"
    ),
}

# Non-.py paths under a watched root that DO decide a verdict. Empty today.
DIGEST_EXTRA: tuple[str, ...] = ()

# pytest's own default python_files, applied when the repo's config is
# silent on the key — as this repo's is. See _python_files_patterns.
DEFAULT_PYTHON_FILES = ("test_*.py", "*_test.py")


def _mutmut_config(pyproject: Path) -> dict:
    config = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return config["tool"]["mutmut"]


def selected_tests(pyproject: Path = PYPROJECT_PATH) -> list[Path]:
    """The test files mutmut runs against each mutant, from [tool.mutmut]."""
    selection = _mutmut_config(pyproject)["pytest_add_cli_args_test_selection"]
    return sorted(Path(entry) for entry in selection)


def _source_paths(pyproject: Path) -> set[str]:
    return set(_mutmut_config(pyproject).get("source_paths", []))


def _also_copy(pyproject: Path) -> list[str]:
    return list(_mutmut_config(pyproject).get("also_copy", []))


def _as_root_prefix(entry: str) -> str:
    """A root used for prefix matching must end in `/`: without it,
    `"backends"` would also match the unrelated sibling `"backends_extra/"`.
    also_copy/source_paths come from pyproject.toml, so this is not
    guaranteed on the way in."""
    return entry if entry.endswith("/") else entry + "/"


def _watched_roots(pyproject: Path) -> tuple[str, ...]:
    also_copy_roots = tuple(_as_root_prefix(entry) for entry in _also_copy(pyproject))
    return also_copy_roots + IMPLICIT_WATCHED_ROOTS


def _pytest_ini_options(pyproject: Path) -> dict:
    config = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return config.get("tool", {}).get("pytest", {}).get("ini_options", {})


def _python_files_patterns(pyproject: Path) -> tuple[str, ...]:
    """The filename patterns pytest collects as test files, read from the
    repo's own config rather than assumed, so this check agrees with pytest
    about what pytest collects instead of hardcoding one convention
    (test_*.py) and missing every file written in the other
    (*_test.py) — pytest's own default accepts both.

    [tool.pytest.ini_options] is pytest's INI-compatibility mode (as
    opposed to native-TOML [tool.pytest]), so a string value here is parsed
    with shlex.split, matching _pytest.config.Config._getini_ini's own
    handling of an "args"-type option — not str.split, which would only
    disagree on a quoted pattern, but disagreeing at all would defeat the
    point of reading pytest's config instead of assuming a convention."""
    patterns = _pytest_ini_options(pyproject).get("python_files", DEFAULT_PYTHON_FILES)
    if isinstance(patterns, str):
        patterns = shlex.split(patterns)
    return tuple(patterns)


def digest(paths: list[Path]) -> str:
    """One digest over every given file, path and content alike."""
    running = hashlib.sha256()
    for path in paths:
        running.update(path.as_posix().encode("utf-8"))
        running.update(b"\0")
        running.update(path.read_bytes())
        running.update(b"\0")
    return running.hexdigest()


def digest_inputs(pyproject: Path) -> list[Path]:
    """Every file this gate hashes: the selected tests, every non-source .py
    file under a watched root, and anything hand-listed in DIGEST_EXTRA.

    source_paths files are deliberately excluded: mutmut already hashes them
    per function, and including them here would drop the whole cache on any
    source edit and destroy the per-commit reuse this cache exists for.
    """
    root = pyproject.parent
    files = repo_files(root)
    if files is None:
        raise RuntimeError("cannot list repository files: git is unavailable")
    sources = _source_paths(pyproject)
    roots = _watched_roots(pyproject)
    watched_py = {
        root / listed
        for listed in files
        if listed.endswith(".py")
        and listed not in sources
        and listed not in DIGEST_EXCLUSIONS
        and any(listed.startswith(prefix) for prefix in roots)
    }
    extra = {root / path for path in DIGEST_EXTRA}
    return sorted(set(selected_tests(pyproject)) | watched_py | extra)


def compute_digest(pyproject: Path) -> str:
    """The digest widened past `selected_tests`: also everything closure's
    walk of the watched roots pulls in, plus the pytest configuration that
    governs how the selected tests run."""
    ini_options = _pytest_ini_options(pyproject)
    fingerprint = digest(digest_inputs(pyproject)) + repr(
        dict(sorted(ini_options.items()))
    )
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Ask git about the working directory, not about whoever invoked this.

    Git exports GIT_DIR and GIT_INDEX_FILE to the hooks it runs, and they
    outrank the current directory — the same reason tools/ratchet_gate.py's
    `_git` scrubs them.
    """
    environment = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    return subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
        cwd=cwd,
    )


def repo_files(root: Path) -> list[str] | None:
    """Every tracked and untracked-not-ignored path under `root`, relative to
    it. `git ls-files` alone reports the index; mutmut's copy_src_dir walks
    the on-disk tree via shutil.copytree, so an untracked file it would still
    copy needs `--others --exclude-standard` too. None means git could not
    answer, in which case the caller must fail rather than skip.

    `--exclude-standard` also honors `core.excludesFile`, a host/user git
    config setting outside this repository's own `.gitignore` — overridden
    to `/dev/null` so this listing is the same on every machine, not just
    ones without a personal excludes file that happens to match a watched
    path.
    """
    result = _git(
        root,
        "-c",
        "core.excludesFile=/dev/null",
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
    )
    if result.returncode != 0:
        return None
    return [line for line in result.stdout.splitlines() if line]


def _check_closure(pyproject: Path) -> list[str]:
    files = repo_files(pyproject.parent)
    if files is None:
        return ["cannot list repository files: git is unavailable"]

    roots = _watched_roots(pyproject)
    extra = set(DIGEST_EXTRA)
    failures = []
    for path in files:
        if not any(path.startswith(prefix) for prefix in roots):
            continue
        if path.endswith(".py"):
            continue  # auto-covered: every .py under a watched root is hashed
        if path in DIGEST_EXCLUSIONS or path in extra:
            continue
        failures.append(
            f"{path} is under a watched root, is not a .py file, and is not "
            "covered. Add it to DIGEST_EXTRA if it can decide a mutant's "
            "verdict, or to DIGEST_EXCLUSIONS with a reason if it cannot."
        )
    return failures


def _check_exclusions(root: Path) -> list[str]:
    failures = []
    for path, reason in DIGEST_EXCLUSIONS.items():
        if any(character in path for character in "*?["):
            failures.append(
                f"DIGEST_EXCLUSIONS[{path!r}] looks like a glob; list the "
                "exact path instead. A glob silences a whole tree for the "
                "cost of silencing one file."
            )
        if not reason.strip():
            failures.append(
                f"DIGEST_EXCLUSIONS[{path!r}] has no reason. The reason is a "
                "claim that this file cannot decide a verdict — write it."
            )
        if not (root / path).is_file():
            failures.append(
                f"DIGEST_EXCLUSIONS[{path!r}] names no file. An exclusion "
                "must land in the same diff as the file it excuses."
            )
    return failures


def _check_copy_invariant(pyproject: Path) -> list[str]:
    also_copy_roots = [_as_root_prefix(entry) for entry in _also_copy(pyproject)]
    uncopied = sorted(
        entry
        for entry in _source_paths(pyproject)
        if not any(entry.startswith(copy_root) for copy_root in also_copy_roots)
    )
    if not uncopied:
        return []
    return [
        "source_paths not covered by any also_copy root, so a mutated file "
        "from a previous run could be left in place uninvalidated: "
        + ", ".join(uncopied)
    ]


def _check_selection(pyproject: Path) -> list[str]:
    files = repo_files(pyproject.parent)
    if files is None:
        return ["cannot list repository files: git is unavailable"]

    root = pyproject.parent
    roots = _watched_roots(pyproject)
    patterns = _python_files_patterns(pyproject)
    excused = set(DIGEST_EXCLUSIONS)

    selected_files = set()
    selected_dirs = []
    for entry in selected_tests(pyproject):
        resolved = (root / entry).resolve()
        if resolved.is_dir():
            selected_dirs.append(_as_root_prefix(resolved.as_posix()))
        else:
            selected_files.add(resolved)

    failures = []
    for path in files:
        if not any(path.startswith(prefix) for prefix in roots):
            continue
        if not any(fnmatch.fnmatch(Path(path).name, pattern) for pattern in patterns):
            continue
        resolved = (root / path).resolve()
        selected = resolved in selected_files or any(
            resolved.as_posix().startswith(prefix) for prefix in selected_dirs
        )
        if path in excused:
            if selected:
                failures.append(
                    f"{path} is both selected and in DIGEST_EXCLUSIONS. mutmut "
                    "collects it, so it can decide a verdict, which is exactly "
                    "what an exclusion claims it cannot do. Remove it from "
                    "whichever one is stale."
                )
            continue
        if selected:
            continue
        failures.append(
            f"{path} matches pytest's python_files but is not in [tool.mutmut] "
            "pytest_add_cli_args_test_selection, so mutmut never collects it "
            "and every mutant it alone kills is reported survived. Add it to "
            "the selection, or — only if it cannot run inside mutants/ at "
            "all — to DIGEST_EXCLUSIONS with that reason."
        )
    return failures


def _run_checks(pyproject: Path) -> list[str]:
    return [
        *_check_closure(pyproject),
        *_check_exclusions(pyproject.parent),
        *_check_copy_invariant(pyproject),
        *_check_selection(pyproject),
    ]


def main(argv: list[str] | None = None) -> int:
    recording = list(sys.argv[1:] if argv is None else argv) == ["--record"]

    try:
        failures = _run_checks(PYPROJECT_PATH)
    except (OSError, KeyError, RuntimeError, tomllib.TOMLDecodeError) as exc:
        print(f"error: cannot read the mutation cache configuration: {exc}")
        return 1
    if failures:
        for failure in failures:
            print(f"error: {failure}")
        return 1

    try:
        current = compute_digest(PYPROJECT_PATH)
    except (OSError, KeyError, RuntimeError, tomllib.TOMLDecodeError) as exc:
        print(f"error: cannot hash the mutation cache inputs: {exc}")
        return 1

    if recording:
        DIGEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        DIGEST_PATH.write_text(current + "\n", encoding="utf-8")
        return 0

    previous = None
    if DIGEST_PATH.exists():
        previous = DIGEST_PATH.read_text(encoding="utf-8").strip()

    # Nothing this run did not measure may survive into the gate's input,
    # whether the cache below is kept or dropped.
    (MUTANTS_DIR / "mutmut-cicd-stats.json").unlink(missing_ok=True)

    if previous == current:
        if MUTANTS_DIR.exists():
            print("mutation cache: tests unchanged; reusing cached mutant results.")
        else:
            print("mutation cache: tests unchanged, but no mutants/ to reuse yet.")
        return 0

    # The digest is not rewritten here: only a completed run may claim these
    # inputs were measured.
    if MUTANTS_DIR.exists():
        shutil.rmtree(MUTANTS_DIR)
        print("mutation cache: tests changed since the cached run; measuring again.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
