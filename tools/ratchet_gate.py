#!/usr/bin/env python3
"""Fail when a quality threshold moves the wrong way.

Every other gate checks the code. This one checks the gates, because none of
them can stop an agent from editing the gate instead of satisfying it: a
lowered floor or a narrowed mutmut scope both produce a green run. That used
to be prose in AGENTS.md asking politely; asking is not a control.

Thresholds are read from the base branch and compared with the working tree.
Raising a floor is always allowed. Lowering one fails. A floor may be dropped
only when its source file is genuinely gone. Coverage is the exception: a
deliberate risk-policy reset can lower its thresholds by advancing
``COVERAGE_POLICY_VERSION`` exactly one version. Gate topology
(``CI_GATES``, Makefile prerequisite chains, hosted workflow jobs,
``quality-gate`` needs, and the ``diff-coverage`` step) is guarded the same
way: it may only grow, and a deliberate reset requires advancing
``GATE_TOPOLOGY_POLICY_VERSION`` exactly one version. Each explicit change is
easy to review and does not weaken the other.

A threshold is only guarded if this file knows where it lives, so every new
one needs an entry here. They currently sit in three places: the gate
scripts, `pyproject.toml`, and the Makefile.

Gate topology is guarded relatively (base branch vs. working tree), not
against a frozen absolute list of protected gates/jobs/needs. A reviewed
``GATE_TOPOLOGY_POLICY_VERSION`` bump that removes a gate therefore drops
that gate from protection for good, the same way a reviewed
``COVERAGE_POLICY_VERSION`` bump permanently lowers a floor. This is a
deliberate choice, not an oversight: an absolute frozen list would need its
own update on every legitimate addition, in a second place, with no way to
tell "author forgot to update the freeze" apart from "author is narrowing
protection on purpose" -- exactly the dual-maintenance failure mode this
file's own docstring already warns against for Makefile chains. Relative
comparison plus a reviewed, one-at-a-time version bump gives the same
protection with one fewer place to go stale.

Suppression counts are deliberately not ratcheted: AGENTS.md permits a
`# noqa` or `# nosec` that carries a finding ID and a justification, so a
count-based gate would fire on legitimate use and be silenced.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

COVERAGE_GATE = "tools/coverage_gate.py"
MUTATION_GATE = "tools/mutation_gate.py"
PYPROJECT = "pyproject.toml"
MAKEFILE = "Makefile"
SEMGREP_RULES = "semgrep.yml"
CI_WORKFLOW = ".github/workflows/ci.yml"
RATCHET_GATE = "tools/ratchet_gate.py"

# Gate topology is a threshold like any other: CI_GATES, the Makefile's
# prerequisite chains, and the workflow's job/needs graph can all be narrowed
# a token at a time while every individual gate still passes. A deliberate
# reset works the same way a coverage policy reset does, and independently of
# it -- the two must not be conflatable.
GATE_TOPOLOGY_POLICY_VERSION = 1

# Which Makefile rules this file walks for a narrowed prerequisite list. Not
# a general graph walker: adding a new chain worth protecting means adding
# its target name here.
TOPOLOGY_TARGETS = (
    "ci",
    "verify",
    "security",
    "verify-quick",
    "verify-coverage",
    "verify-mutation",
    "verify-security",
    "security-static",
    "ci-hosted",
)


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    """Ask git about the working directory, not about whoever invoked this.

    Git exports GIT_DIR and GIT_INDEX_FILE to the hooks it runs, and they
    outrank the current directory. Inherited, this gate would compare the
    thresholds of the repository that launched it rather than the one being
    checked — which is exactly the silent no-op it exists to prevent, and it
    would only happen from a hook, where nobody is watching the output.
    """
    environment = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False, env=environment
    )


def _read_base(base: str, path: str) -> str | None:
    """Return `path` as of `base`, or None when it did not exist there."""
    result = _git("show", f"{base}:{path}")
    return result.stdout if result.returncode == 0 else None


def _binding_count(source: str, name: str) -> int:
    """How many module-level statements bind `name`."""
    count = 0
    for node in ast.parse(source).body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            )
        ) or (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            count += 1
    return count


def _constant(source: str, name: str):
    """Pull a module-level literal without importing the file.

    Fails closed on more than one module-level binding of `name`: Python
    resolves to the last, a first-match reader to the first, and that gap
    lets an added `GATE_TOPOLOGY_POLICY_VERSION = 2` above the real one
    claim a reviewed reset while the constant everyone reads still says 1.
    """
    if _binding_count(source, name) > 1:
        return None
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
            and node.value is not None
        ):
            return ast.literal_eval(node.value)
    return None


def _pyproject_numbers(source: str) -> tuple[float | None, list[str]]:
    config = tomllib.loads(source)
    fail_under = (
        config.get("tool", {}).get("coverage", {}).get("report", {}).get("fail_under")
    )
    source_paths = config.get("tool", {}).get("mutmut", {}).get("source_paths", [])
    return fail_under, source_paths


def _coverage_policy_reset(base: str, failures: list[str]) -> bool:
    """Return whether this change deliberately advances the coverage policy."""
    base_source = _read_base(base, COVERAGE_GATE)
    if base_source is None:
        return False
    now_source = Path(COVERAGE_GATE).read_text(encoding="utf-8")
    was = _constant(base_source, "COVERAGE_POLICY_VERSION") or 1
    now = _constant(now_source, "COVERAGE_POLICY_VERSION") or 1
    if now < was:
        failures.append(f"COVERAGE_POLICY_VERSION lowered {was:g} -> {now:g}.")
        return False
    if now > was + 1:
        failures.append(
            f"COVERAGE_POLICY_VERSION jumped {was:g} -> {now:g}; advance it one "
            "reviewed policy revision at a time."
        )
        return False
    return now == was + 1


def _check_coverage_floors(
    base: str, failures: list[str], *, policy_reset: bool
) -> None:
    base_source = _read_base(base, COVERAGE_GATE)
    if base_source is None or policy_reset:
        return
    base_floors = _constant(base_source, "FLOORS") or {}
    now_floors = _constant(Path(COVERAGE_GATE).read_text(encoding="utf-8"), "FLOORS")
    now_floors = now_floors or {}

    for path, was in base_floors.items():
        if path not in now_floors:
            if Path(path).exists():
                failures.append(
                    f"{path}: coverage floor removed while the file still exists."
                )
            continue
        if now_floors[path] < was:
            failures.append(
                f"{path}: coverage floor lowered {was:g} -> {now_floors[path]:g}."
            )

    base_new = _constant(base_source, "NEW_FILE_FLOOR")
    now_new = _constant(
        Path(COVERAGE_GATE).read_text(encoding="utf-8"), "NEW_FILE_FLOOR"
    )
    if base_new is not None and now_new is not None and now_new < base_new:
        failures.append(f"NEW_FILE_FLOOR lowered {base_new:g} -> {now_new:g}.")


def _check_mutation_floor(base: str, failures: list[str]) -> None:
    base_source = _read_base(base, MUTATION_GATE)
    if base_source is None:
        return
    was = _constant(base_source, "MUTATION_SCORE_FLOOR")
    now = _constant(
        Path(MUTATION_GATE).read_text(encoding="utf-8"), "MUTATION_SCORE_FLOOR"
    )
    if was is not None and now is not None and now < was:
        failures.append(f"MUTATION_SCORE_FLOOR lowered {was:g} -> {now:g}.")


def _check_pyproject(
    base: str, failures: list[str], *, coverage_policy_reset: bool
) -> None:
    base_source = _read_base(base, PYPROJECT)
    if base_source is None:
        return
    was_fail_under, was_paths = _pyproject_numbers(base_source)
    now_fail_under, now_paths = _pyproject_numbers(
        Path(PYPROJECT).read_text(encoding="utf-8")
    )

    if (
        not coverage_policy_reset
        and was_fail_under is not None
        and now_fail_under is not None
        and now_fail_under < was_fail_under
    ):
        failures.append(
            f"coverage fail_under lowered {was_fail_under:g} -> {now_fail_under:g}."
        )

    # Same allowance the coverage floors get: a threshold may be dropped when
    # its source file is genuinely gone. Without this, renaming the package
    # reads as narrowing the scope, because the comparison is stringwise and
    # the base branch still spells the old prefix. Dropping a module that is
    # still there remains a failure, which is the case worth catching.
    dropped = {path for path in set(was_paths) - set(now_paths) if Path(path).exists()}
    if dropped:
        failures.append(
            f"mutmut source_paths narrowed; no longer mutated: {sorted(dropped)}."
        )


def _make_variable(source: str, name: str) -> float | None:
    """Read a `NAME ?= value` assignment without invoking make."""
    if _assignment_count(source, name) != 1:
        return None
    match = re.search(
        rf"^{re.escape(name)}\s*\?*=\s*([0-9]+(?:\.[0-9]+)?)\s*$",
        source,
        flags=re.MULTILINE,
    )
    return float(match.group(1)) if match else None


def _check_diff_coverage_floor(base: str, failures: list[str]) -> None:
    """The diff-coverage floor lives in the Makefile, not in a gate script.

    Every other threshold this file guards sits in Python or TOML, so the one
    written in make syntax was the only one an agent could lower with every
    check still green — and it is the threshold that governs new code, which
    is where a generated change actually lands.
    """
    base_source = _read_base(base, MAKEFILE)
    if base_source is None:
        return
    was = _make_variable(base_source, "DIFF_COVERAGE_MIN")
    if was is None:
        return
    now = _make_variable(
        Path(MAKEFILE).read_text(encoding="utf-8"), "DIFF_COVERAGE_MIN"
    )
    if now is None:
        failures.append(
            "DIFF_COVERAGE_MIN removed from the Makefile; changed lines would "
            "no longer need tests."
        )
    elif now < was:
        failures.append(f"DIFF_COVERAGE_MIN lowered {was:g} -> {now:g}.")


def _semgrep_rule_ids(source: str) -> set[str]:
    # Parsed by regex on purpose: PyYAML is not a dependency, and adding one
    # to read a handful of rule ids is not worth the supply-chain surface.
    return set(re.findall(r"^\s*-\s*id:\s*(\S+)", source, flags=re.MULTILINE))


def _check_semgrep_rules(base: str, failures: list[str]) -> None:
    base_source = _read_base(base, SEMGREP_RULES)
    if base_source is None:
        return
    dropped = _semgrep_rule_ids(base_source) - _semgrep_rule_ids(
        Path(SEMGREP_RULES).read_text(encoding="utf-8")
    )
    if dropped:
        failures.append(f"Semgrep rules deleted: {sorted(dropped)}.")


def _join_continuations(source: str) -> str:
    """Collapse a backslash-continued Makefile line onto one logical line."""
    return re.sub(r"\\\n[ \t]*", " ", source)


# Every way GNU make can bind a variable. `_make_list` reads only the plain
# `NAME :=` form, so any of the others appearing alongside it means the value
# make resolves is not the one this file parsed -- `override NAME := x`,
# `NAME ::= x` and `define NAME` all won a hunt against the first-match-only
# version of this gate.
_ASSIGNMENT_FORMS = (
    r"^(?:override\s+)?{name}\s*(?::::=|::=|:=|\+=|\?=|=)",
    r"^(?:override\s+)?define\s+{name}\s*(?::::=|::=|:=|\+=|\?=|=)?\s*$",
)


def _assignment_count(source: str, name: str) -> int:
    """How many times `name` is bound, across every assignment syntax."""
    return sum(
        len(re.findall(form.format(name=re.escape(name)), source, flags=re.MULTILINE))
        for form in _ASSIGNMENT_FORMS
    )


def _make_list(source: str, name: str) -> list[str] | None:
    """Read a `NAME := val val \\n val` assignment, expanding `$(OTHER)` refs.

    Bounded to this shape on purpose: it is what CI_GATES and its VERIFY_*
    building blocks are written as today. A reformat that this cannot parse
    fails closed via the callers' None handling, not silently.

    Also fails closed on more than one `NAME :=` assignment. GNU make
    resolves a simply-expanded variable to whichever assignment is last
    before the point it is referenced, which is order-dependent and easy to
    get wrong with a regex; a second, narrower assignment inserted just
    before CI_GATES's own definition is exactly how a gate would be quietly
    dropped from what `make` actually runs while still parsing as "defined"
    to a first-match reader. Treating any duplicate as unparsable avoids
    replicating make's resolution order and can't be gamed by insertion
    position.
    """
    joined = _join_continuations(source)
    if _assignment_count(joined, name) != 1:
        return None
    matches = list(
        re.finditer(rf"^{re.escape(name)}\s*:=\s*(.*)$", joined, flags=re.MULTILINE)
    )
    if len(matches) != 1:
        return None
    match = matches[0]
    expanded: list[str] = []
    for token in match.group(1).split():
        ref = re.fullmatch(r"\$\((\w+)\)", token)
        if ref is None:
            expanded.append(token)
            continue
        sub = _make_list(source, ref.group(1))
        if sub is None:
            return None
        expanded.extend(sub)
    return expanded


def _make_prereqs(source: str, target: str) -> list[str] | None:
    """Read a `target: prereq prereq` rule line, expanding `$(VAR)` refs."""
    joined = _join_continuations(source)
    matches = list(
        re.finditer(rf"^{re.escape(target)}::?\s*(.*)$", joined, flags=re.MULTILINE)
    )
    if len(matches) != 1:
        return None
    match = matches[0]
    if match.group(0).startswith(f"{target}::"):
        return None
    expanded: list[str] = []
    for token in match.group(1).split():
        ref = re.fullmatch(r"\$\((\w+)\)", token)
        if ref is None:
            expanded.append(token)
            continue
        expanded.extend(_make_list(source, ref.group(1)) or [])
    return expanded


def _name_defined(source: str, name: str) -> bool:
    """Whether a line starts with `name` at all, regardless of whether the
    rest of that line matches the shape this file knows how to parse.

    Distinguishes "this was never defined here" (nothing to protect, skip
    as always) from "it was defined and got reformatted into something this
    gate can no longer read" (a silent, permanent erosion of protection
    worth failing on). `(?![\\w-])` rather than `\\b` so `name="ci"` does not
    match a `ci-hosted:` line, and a `name` ending in a non-word character
    like `:` still requires a real boundary after it.
    """
    return (
        re.search(rf"^\s*{re.escape(name)}(?![\w-])", source, flags=re.MULTILINE)
        is not None
    )


def _gate_topology_policy_reset(base: str, failures: list[str]) -> bool:
    """Return whether this change deliberately resets gate topology.

    Independent of COVERAGE_POLICY_VERSION: a reviewed threshold reset says
    nothing about whether a gate itself may be dropped.
    """
    base_source = _read_base(base, RATCHET_GATE)
    if base_source is None:
        return False
    now_source = Path(RATCHET_GATE).read_text(encoding="utf-8")
    was = _constant(base_source, "GATE_TOPOLOGY_POLICY_VERSION") or 1
    now = _constant(now_source, "GATE_TOPOLOGY_POLICY_VERSION") or 1
    if now < was:
        failures.append(f"GATE_TOPOLOGY_POLICY_VERSION lowered {was:g} -> {now:g}.")
        return False
    if now > was + 1:
        failures.append(
            f"GATE_TOPOLOGY_POLICY_VERSION jumped {was:g} -> {now:g}; advance it "
            "one reviewed policy revision at a time."
        )
        return False
    return now == was + 1


def _check_gate_topology(base: str, failures: list[str], *, policy_reset: bool) -> None:
    """CI_GATES, and the VERIFY_* lists it expands, may only grow.

    A gate dropped straight from CI_GATES and a gate dropped from a VERIFY_*
    list it references both surface here, because CI_GATES is expanded
    recursively -- editing both places to hide a deletion still leaves the
    expanded set smaller than the base branch's.
    """
    base_source = _read_base(base, MAKEFILE)
    if base_source is None:
        return
    was = _make_list(base_source, "CI_GATES")
    if was is None:
        # A base branch that never defined CI_GATES has nothing to protect
        # here -- skip as always. One that defines it in a shape this file's
        # regex cannot read is the reformat that would let a narrowed set
        # through unnoticed forever, so that case fails instead, unless it
        # is itself the reviewed, version-bumped change.
        if _name_defined(base_source, "CI_GATES") and not policy_reset:
            failures.append(
                "CI_GATES on the base branch could not be resolved to a "
                "single unambiguous value -- either it or a VERIFY_* "
                "variable it expands does not match the `NAME := val val` "
                "shape this gate parses, or is assigned more than once; "
                "update the parser and advance GATE_TOPOLOGY_POLICY_VERSION "
                "if this reformat is deliberate."
            )
        return
    if policy_reset:
        return
    now = _make_list(Path(MAKEFILE).read_text(encoding="utf-8"), "CI_GATES") or []
    dropped = [gate for gate in was if gate not in now]
    if dropped:
        failures.append(f"CI_GATES dropped required gate(s): {dropped}.")


def _check_make_chains(base: str, failures: list[str], *, policy_reset: bool) -> None:
    """The prerequisite chains that wire gates into `ci`/`ci-hosted` may only grow.

    `_make_prereqs` matches any line starting with `target:`, whatever
    follows -- there is no reformat of an existing rule's right-hand side
    that would make it return None, only a rule that never existed on the
    base branch. So a None here has one cause worth handling: skip, the
    same as every other threshold in this file skips a check for code that
    was not there yet on `base`.
    """
    base_source = _read_base(base, MAKEFILE)
    if base_source is None:
        return
    now_source = Path(MAKEFILE).read_text(encoding="utf-8")
    for target in TOPOLOGY_TARGETS:
        was = _make_prereqs(base_source, target)
        if was is None or policy_reset:
            continue
        now = _make_prereqs(now_source, target) or []
        dropped = [item for item in was if item not in now]
        if dropped:
            failures.append(f"{target}: prerequisite(s) dropped: {dropped}.")


def _ci_job_names(source: str) -> set[str]:
    jobs_section = source.partition("\njobs:\n")[2]
    return set(
        re.findall(r"^  ([A-Za-z_][A-Za-z0-9_-]*):", jobs_section, flags=re.MULTILINE)
    )


def _check_hosted_jobs(base: str, failures: list[str], *, policy_reset: bool) -> None:
    base_source = _read_base(base, CI_WORKFLOW)
    if base_source is None or policy_reset:
        return
    now_source = Path(CI_WORKFLOW).read_text(encoding="utf-8")
    dropped = _ci_job_names(base_source) - _ci_job_names(now_source)
    if dropped:
        failures.append(f"CI workflow job(s) deleted: {sorted(dropped)}.")


def _job_block(source: str, job: str) -> str | None:
    """Return a top-level job's body, up to (not including) the next job."""
    marker = f"\n  {job}:\n"
    start = source.find(marker)
    if start == -1:
        return None
    body = source[start + len(marker) :]
    next_job = re.search(r"^  [A-Za-z_][A-Za-z0-9_-]*:", body, flags=re.MULTILINE)
    return body[: next_job.start()] if next_job else body


def _quality_gate_needs(source: str) -> set[str] | None:
    """quality-gate's own `needs:`, not the first flow-style `needs:` in the
    rest of the file. Deleting quality-gate's list while adding an unrelated
    `needs: [...]` to a later job must not be read as quality-gate's own --
    that reformats the exploit into an outright deletion, not a survival.
    """
    block = _job_block(source, "quality-gate")
    if block is None:
        return None
    match = re.search(r"^\s*needs:\s*\[([^\]]*)\]\s*$", block, flags=re.MULTILINE)
    if match is None:
        return None
    return {name.strip() for name in match.group(1).split(",") if name.strip()}


def _check_quality_gate_needs(
    base: str, failures: list[str], *, policy_reset: bool
) -> None:
    """quality-gate's `needs:` list is what makes a lane block the required
    check; a lane that still runs but drops out of `needs:` stops mattering."""
    base_source = _read_base(base, CI_WORKFLOW)
    if base_source is None:
        return
    was = _quality_gate_needs(base_source)
    if was is None:
        # A base branch with no quality-gate job, or one with no needs: line
        # inside it, has nothing here to protect -- skip. One where needs:
        # is present but not in the single-line flow-style shape this gate
        # parses (e.g. reformatted to a multi-line block) fails instead,
        # unless that reformat is itself the reviewed, version-bumped change.
        block = _job_block(base_source, "quality-gate") or ""
        if _name_defined(block, "needs:") and not policy_reset:
            failures.append(
                "quality-gate's needs: on the base branch does not match "
                "the single-line `needs: [a, b]` shape this gate parses; "
                "update the parser and advance GATE_TOPOLOGY_POLICY_VERSION "
                "if this reformat is deliberate."
            )
        return
    if policy_reset:
        return
    now = _quality_gate_needs(Path(CI_WORKFLOW).read_text(encoding="utf-8")) or set()
    dropped = was - now
    if dropped:
        failures.append(
            f"quality-gate needs narrowed; no longer required: {sorted(dropped)}."
        )


def _check_ci_diff_coverage_step(
    base: str, failures: list[str], *, policy_reset: bool
) -> None:
    """The coverage job must keep an actual `run: make diff-coverage` step.

    Scoped to a `run:` line inside the coverage job block, not a whole-file
    substring: a comment mentioning the command, or the step surviving in a
    different job, must not satisfy this.
    """
    base_source = _read_base(base, CI_WORKFLOW)
    if base_source is None:
        return
    base_block = _job_block(base_source, "coverage")
    if base_block is None or not re.search(
        r"^\s*run:\s*make diff-coverage[^|;&]*$", base_block, flags=re.MULTILINE
    ):
        return
    if policy_reset:
        return
    now_source = Path(CI_WORKFLOW).read_text(encoding="utf-8")
    now_block = _job_block(now_source, "coverage")
    if now_block is None or not re.search(
        r"^\s*run:\s*make diff-coverage[^|;&]*$", now_block, flags=re.MULTILINE
    ):
        failures.append(
            "the diff-coverage step was removed from the coverage job in "
            "ci.yml; changed lines would no longer need tests in CI."
        )


def _has_gitleaks_step(block: str) -> bool:
    """A real `uses: gitleaks/gitleaks-action@...` step, not a mention of one.

    A commented-out `# uses: gitleaks/...` line kept the substring form of
    this check green while nothing scanned for secrets.
    """
    return (
        re.search(
            r"^\s*(?:-\s*)?uses:\s*gitleaks/gitleaks-action@",
            block,
            flags=re.MULTILINE,
        )
        is not None
    )


def _check_ci_secret_scan_step(
    base: str, failures: list[str], *, policy_reset: bool
) -> None:
    """The secret-scan job must keep its Gitleaks step.

    Scoped to the secret-scan job block: a gitleaks-action reference
    surviving in an unrelated job, or as a comment, must not satisfy this.
    """
    base_source = _read_base(base, CI_WORKFLOW)
    if base_source is None:
        return
    base_block = _job_block(base_source, "secret-scan")
    if base_block is None or not _has_gitleaks_step(base_block):
        return
    if policy_reset:
        return
    now_source = Path(CI_WORKFLOW).read_text(encoding="utf-8")
    now_block = _job_block(now_source, "secret-scan")
    if now_block is None or not _has_gitleaks_step(now_block):
        failures.append(
            "the Gitleaks step was removed from the secret-scan job in "
            "ci.yml; nothing would scan for leaked secrets."
        )


def _quality_gate_success_count(source: str) -> int | None:
    """How many successful lanes quality-gate's assertion step demands.

    None when the step is gone or is no longer a `test "$LANE_RESULTS" = ...`
    comparison against a run of `success` words, so rewriting it to anything
    else -- `run: true`, a script, a comment -- reads as asserting nothing.
    """
    block = _job_block(source, "quality-gate")
    if block is None:
        return None
    match = re.search(
        r'^\s*run:\s*test\s+"\$LANE_RESULTS"\s*=\s*"([^"]*)"\s*$',
        block,
        flags=re.MULTILINE,
    )
    if match is None:
        return None
    words = match.group(1).split()
    if not words or any(word != "success" for word in words):
        return None
    return len(words)


def _check_ci_quality_gate_assertion(
    base: str, failures: list[str], *, policy_reset: bool
) -> None:
    """quality-gate must keep asserting that every lane it needs succeeded.

    The `needs:` list alone does not make the aggregate check meaningful:
    `if: always()` runs the job whatever the lanes did, and only this step
    turns a failed lane into a failed required check. So the assertion is
    checked against the `needs:` list as well as against the base -- a lane
    the assertion does not count is a lane that can fail in silence.
    """
    base_source = _read_base(base, CI_WORKFLOW)
    if base_source is None or _quality_gate_success_count(base_source) is None:
        return
    if policy_reset:
        return
    now_source = Path(CI_WORKFLOW).read_text(encoding="utf-8")
    now = _quality_gate_success_count(now_source)
    if now is None:
        failures.append(
            "quality-gate no longer asserts that every lane succeeded; keep "
            'its `test "$LANE_RESULTS" = "success ..."` step in ci.yml.'
        )
        return
    required = len(_quality_gate_needs(now_source) or set())
    if now < required:
        failures.append(
            f"quality-gate asserts only {now} successful lane(s) but needs "
            f"{required}; the extra lane(s) could fail without blocking the "
            "required check."
        )


# The paths every other check reads. Repointing one at a file absent from the
# base tree makes `_read_base` return None and that whole family of checks
# skip in silence, so the constants are themselves a guarded threshold.
GUARDED_PATHS = (
    "COVERAGE_GATE",
    "MUTATION_GATE",
    "PYPROJECT",
    "MAKEFILE",
    "SEMGREP_RULES",
    "CI_WORKFLOW",
    "RATCHET_GATE",
)

# Make flags and special targets that turn a failing recipe into a passing
# build. `.IGNORE:` with no prerequisites applies to every target at once.
_MAKE_IGNORE_FLAGS = re.compile(
    r"^\s*MAKEFLAGS\s*\+?=.*(?:\s|^)(?:-i|-k|--ignore-errors|--keep-going)\b",
    re.MULTILINE,
)
_MAKE_IGNORE_TARGET = re.compile(r"^\.IGNORE\s*:\s*$", re.MULTILINE)


def _check_ratchet_self(base: str, failures: list[str], *, policy_reset: bool) -> None:
    """The gate's own guarded paths and target list may not quietly narrow."""
    base_source = _read_base(base, RATCHET_GATE)
    if base_source is None or policy_reset:
        return
    now_source = Path(RATCHET_GATE).read_text(encoding="utf-8")
    for name in GUARDED_PATHS:
        was = _constant(base_source, name)
        now = _constant(now_source, name)
        if was is not None and was != now:
            failures.append(
                f"{name} repointed {was!r} -> {now!r}; the checks reading it "
                "would compare against a path absent from the base tree."
            )
    was_targets = _constant(base_source, "TOPOLOGY_TARGETS") or ()
    now_targets = _constant(now_source, "TOPOLOGY_TARGETS") or ()
    dropped = [target for target in was_targets if target not in now_targets]
    if dropped:
        failures.append(f"TOPOLOGY_TARGETS dropped: {dropped}.")


def _check_make_error_handling(
    base: str, failures: list[str], *, policy_reset: bool
) -> None:
    """`make ci` must keep reporting a failing gate as a failure."""
    base_source = _read_base(base, MAKEFILE)
    if base_source is None or policy_reset:
        return
    now_source = Path(MAKEFILE).read_text(encoding="utf-8")
    if _MAKE_IGNORE_TARGET.search(now_source) and not _MAKE_IGNORE_TARGET.search(
        base_source
    ):
        failures.append(
            "`.IGNORE:` added to the Makefile; every gate's failure would be "
            "ignored while the gate list stays intact."
        )
    if _MAKE_IGNORE_FLAGS.search(now_source) and not _MAKE_IGNORE_FLAGS.search(
        base_source
    ):
        failures.append(
            "MAKEFLAGS gained an error-ignoring flag (-i/-k); a failing gate "
            "would no longer fail the build."
        )


def _recipe(source: str, target: str) -> list[str] | None:
    """A target's recipe lines, in order, without the leading tab."""
    match = re.search(
        rf"^{re.escape(target)}:[^\n]*\n((?:\t[^\n]*\n)*)",
        source,
        flags=re.MULTILINE,
    )
    if match is None:
        return None
    return [line[1:] for line in match.group(1).splitlines()]


def _check_make_recipes(base: str, failures: list[str], *, policy_reset: bool) -> None:
    """A protected gate's commands may only grow, and may not go advisory.

    A gate whose recipe becomes `@true`, or whose command gains make's `-`
    ignore-failure prefix, still announces itself and still appears in
    CI_GATES -- the name survives every topology check while the check it
    names stops running.
    """
    base_source = _read_base(base, MAKEFILE)
    if base_source is None or policy_reset:
        return
    now_source = Path(MAKEFILE).read_text(encoding="utf-8")
    for gate in _make_list(base_source, "CI_GATES") or []:
        was = _recipe(base_source, gate)
        if was is None:
            continue
        now = _recipe(now_source, gate)
        if now is None:
            failures.append(f"{gate}: recipe deleted.")
            continue
        for command in was:
            if command not in now:
                failures.append(
                    f"{gate}: command dropped from its recipe: {command!r}."
                )
        for command in now:
            if command.lstrip().startswith("-") and command not in was:
                failures.append(
                    f"{gate}: command made failure-ignoring with make's `-` "
                    f"prefix: {command!r}."
                )


def _check_make_base_refs(
    base: str, failures: list[str], *, policy_reset: bool
) -> None:
    """The refs every comparison is made against are themselves a threshold.

    Repointing RATCHET_BASE or DIFF_BASE at HEAD compares a branch with
    itself, which cannot fail; hardcoding a ref into the `ratchet:` recipe
    does the same while leaving the variable looking untouched.
    """
    base_source = _read_base(base, MAKEFILE)
    if base_source is None or policy_reset:
        return
    now_source = Path(MAKEFILE).read_text(encoding="utf-8")
    for name in ("RATCHET_BASE", "DIFF_BASE"):
        was = re.search(rf"^{name}\s*\?*=\s*(\S+)\s*$", base_source, re.MULTILINE)
        now = re.search(rf"^{name}\s*\?*=\s*(\S+)\s*$", now_source, re.MULTILINE)
        if was is None:
            continue
        if now is None or now.group(1) != was.group(1):
            got = now.group(1) if now else "<removed>"
            failures.append(
                f"{name} changed {was.group(1)!r} -> {got!r}; comparisons are "
                "made against this ref, so it is a threshold too."
            )
    for command in _recipe(now_source, "ratchet") or []:
        if "ratchet_gate.py" in command and "$(RATCHET_BASE)" not in command:
            failures.append(
                "the ratchet recipe no longer passes $(RATCHET_BASE); a "
                f"hardcoded base cannot fail: {command.strip()!r}."
            )


# What a job's steps promise to do. `run:` and `uses:` are the work itself,
# `if:` decides whether it happens, and a job's `name:` is the identity
# branch protection matches on -- each can be edited to leave a lane that
# exists, reports success, and does nothing.
_JOB_DIRECTIVES = re.compile(
    r"^\s*(?:-\s*)?(run|uses|if|name):\s*(.+?)\s*$", re.MULTILINE
)


def _job_directives(block: str) -> set[str]:
    return {f"{key}: {value}" for key, value in _JOB_DIRECTIVES.findall(block)}


def _check_ci_job_integrity(
    base: str, failures: list[str], *, policy_reset: bool
) -> None:
    """Every lane must keep doing what it did, not merely keep existing."""
    base_source = _read_base(base, CI_WORKFLOW)
    if base_source is None or policy_reset:
        return
    now_source = Path(CI_WORKFLOW).read_text(encoding="utf-8")
    for job in sorted(_ci_job_names(base_source)):
        base_block = _job_block(base_source, job)
        now_block = _job_block(now_source, job)
        if base_block is None:
            continue
        if now_block is None:
            continue
        dropped = sorted(_job_directives(base_block) - _job_directives(now_block))
        if dropped:
            failures.append(f"{job}: job no longer does what it did: {dropped}.")
        if re.search(
            r"^\s*continue-on-error:\s*true\s*$", now_block, flags=re.MULTILINE
        ) and not re.search(
            r"^\s*continue-on-error:\s*true\s*$", base_block, flags=re.MULTILINE
        ):
            failures.append(
                f"{job}: continue-on-error added; a failing lane would report "
                "success to the aggregate check."
            )


def _check_ci_triggers(base: str, failures: list[str], *, policy_reset: bool) -> None:
    """The events that start CI may only grow."""
    base_source = _read_base(base, CI_WORKFLOW)
    if base_source is None or policy_reset:
        return
    now_source = Path(CI_WORKFLOW).read_text(encoding="utf-8")

    def triggers(source: str) -> set[str]:
        section = source.partition("\non:\n")[2].partition("\njobs:")[0]
        return set(re.findall(r"^  ([a-z_]+):", section, flags=re.MULTILINE))

    dropped = triggers(base_source) - triggers(now_source)
    if dropped:
        failures.append(f"CI trigger(s) removed: {sorted(dropped)}.")


def main(argv: list[str]) -> int:
    base = argv[1] if len(argv) > 1 else "origin/master"

    resolved = _git("rev-parse", "--verify", "--quiet", f"{base}^{{commit}}")
    if resolved.returncode != 0:
        # Never pass quietly on a missing base: a ratchet that cannot compare
        # is exactly the silent no-op this gate exists to prevent.
        print(f"error: base ref '{base}' does not resolve.")
        print("Fetch it, or pass an explicit base: make ratchet RATCHET_BASE=<ref>")
        return 1

    failures: list[str] = []
    coverage_policy_reset = _coverage_policy_reset(base, failures)
    _check_coverage_floors(base, failures, policy_reset=coverage_policy_reset)
    _check_mutation_floor(base, failures)
    _check_pyproject(base, failures, coverage_policy_reset=coverage_policy_reset)
    _check_diff_coverage_floor(base, failures)
    _check_semgrep_rules(base, failures)

    topology_policy_reset = _gate_topology_policy_reset(base, failures)
    _check_gate_topology(base, failures, policy_reset=topology_policy_reset)
    _check_make_chains(base, failures, policy_reset=topology_policy_reset)
    _check_hosted_jobs(base, failures, policy_reset=topology_policy_reset)
    _check_quality_gate_needs(base, failures, policy_reset=topology_policy_reset)
    _check_ci_diff_coverage_step(base, failures, policy_reset=topology_policy_reset)
    _check_ci_secret_scan_step(base, failures, policy_reset=topology_policy_reset)
    _check_ci_quality_gate_assertion(base, failures, policy_reset=topology_policy_reset)
    _check_ratchet_self(base, failures, policy_reset=topology_policy_reset)
    _check_make_error_handling(base, failures, policy_reset=topology_policy_reset)
    _check_make_recipes(base, failures, policy_reset=topology_policy_reset)
    _check_make_base_refs(base, failures, policy_reset=topology_policy_reset)
    _check_ci_job_integrity(base, failures, policy_reset=topology_policy_reset)
    _check_ci_triggers(base, failures, policy_reset=topology_policy_reset)

    for failure in failures:
        print(f"error: {failure}")
    if failures:
        print(
            f"\n{len(failures)} threshold(s) moved the wrong way against {base}. "
            "Satisfy the gate instead of relaxing it; if the change is "
            "deliberate, say so explicitly in the PR and get it reviewed."
        )
        return 1

    print(f"ratchet: no threshold weakened against {base}.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv))
