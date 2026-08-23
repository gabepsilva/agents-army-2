"""Git and CI mechanics, and the verdicts read back out of one `make ci` log."""

from __future__ import annotations

import logging
import subprocess
from collections import deque
from pathlib import Path

import pytest

from examples.gabriels_workflow_v2 import errors, git
from examples.gabriels_workflow_v2 import gates as ci_gates


def _completed(
    args: list[str], returncode: int = 0, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args, returncode, stdout=stdout, stderr=stderr)


class ScriptedRun:
    def __init__(self, replies: list[subprocess.CompletedProcess[str]]) -> None:
        self.replies = deque(replies)
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, args: list[str], **kwargs):
        self.calls.append((args, kwargs))
        reply = self.replies.popleft()
        stdout, stderr = reply.stdout, reply.stderr
        if kwargs.get("stderr") is subprocess.STDOUT:
            merged = stdout or ""
            if stderr:
                merged = f"{merged}\n{stderr}" if merged else stderr
            stdout, stderr = merged, ""
        return _completed(args, reply.returncode, stdout, stderr)


def test_git_repository_prepare_ci_commit_push_and_failures(tmp_path: Path) -> None:
    runner = ScriptedRun(
        [
            _completed([], stdout="feature\n"),
            _completed([], stdout=""),
            _completed([], stdout="abc\n"),
            _completed([], stdout="lint\n\n"),
            _completed([], stdout="out", stderr="err"),
            _completed([], stdout="file\0"),
            _completed([], stdout="new.py\0"),
            _completed([]),
            _completed([]),
            _completed([], stdout="1\n"),
            _completed([]),
        ]
    )
    repository = git.GitRepository(tmp_path, runner)
    assert repository.prepare("master", False) == ("feature", "abc")
    assert repository.run_ci().as_json() == {
        "returncode": 0,
        "output": "out\nerr",
        "gates": [{"name": "lint", "status": "not run", "reason": "not run"}],
    }
    repository.commit("message", "abc")
    repository.push("feature")
    commands = [call[0] for call in runner.calls]
    assert ["git", "add", "--", "file", "new.py"] in commands
    assert ["git", "commit", "-m", "message"] in commands
    assert commands[-1] == ["git", "push", "--set-upstream", "origin", "feature"]
    assert ["make", "--no-print-directory", "ci-gates"] in commands
    ci_kwargs = runner.calls[4][1]
    assert ci_kwargs["timeout"] == git.DEFAULT_CI_TIMEOUT
    assert ci_kwargs["stdin"] == subprocess.DEVNULL
    assert ci_kwargs["stdout"] is subprocess.PIPE
    assert ci_kwargs["stderr"] is subprocess.STDOUT
    assert "capture_output" not in ci_kwargs

    detached = git.GitRepository(tmp_path, ScriptedRun([_completed([], stdout="\n")]))
    with pytest.raises(errors.WorkflowError, match="named git branch"):
        detached.prepare("master", False)
    protected = git.GitRepository(
        tmp_path, ScriptedRun([_completed([], stdout="master\n")])
    )
    with pytest.raises(errors.WorkflowError, match="protected base"):
        protected.prepare("master", False)
    dirty = git.GitRepository(
        tmp_path,
        ScriptedRun(
            [_completed([], stdout="feature\n"), _completed([], stdout="dirty")]
        ),
    )
    with pytest.raises(errors.WorkflowError, match="clean worktree"):
        dirty.prepare("master", False)
    failed = git.GitRepository(
        tmp_path, ScriptedRun([_completed([], 1, stderr="boom")])
    )
    with pytest.raises(errors.WorkflowError, match="git status failed"):
        failed._call("status")
    no_commits = git.GitRepository(
        tmp_path,
        ScriptedRun(
            [
                _completed([], stdout=""),
                _completed([], stdout=""),
                _completed([], stdout="0\n"),
            ]
        ),
    )
    with pytest.raises(errors.WorkflowStopped, match="no commits"):
        no_commits.commit("message", "abc")


def test_git_repository_opens_an_empty_commit_only_when_the_branch_matches_base(
    tmp_path: Path,
) -> None:
    ahead = ScriptedRun([_completed([], stdout="2\n")])
    git.GitRepository(tmp_path, ahead).ensure_branch_ahead("Start work", "abc")
    empty = ScriptedRun([_completed([], stdout="0\n"), _completed([])])
    git.GitRepository(tmp_path, empty).ensure_branch_ahead("Start work", "abc")

    assert [call[0] for call in ahead.calls] == [
        ["git", "rev-list", "--count", "abc..HEAD"],
    ]
    assert [call[0] for call in empty.calls] == [
        ["git", "rev-list", "--count", "abc..HEAD"],
        ["git", "commit", "--allow-empty", "-m", "Start work"],
    ]


def test_ensure_issue_worktree_branches_off_the_remote_default(
    tmp_path: Path,
) -> None:
    """`origin/master`, not the local `master` that may have fallen behind.

    Regression: the ratchet and diff-coverage gates measure against
    `origin/master`. Branching off a stale local `master` produced a worktree
    that failed CI on commits it did not contain, and the only repair was a
    merge the agent is instructed not to make — so the run stopped blocked
    with nothing an agent could do about it.
    """

    path = tmp_path / "issue" / "worktree"
    runner = ScriptedRun(
        [
            _completed([], stdout=""),
            _completed(
                [], stdout="worktree /repo\nHEAD abc\nbranch refs/heads/master\n"
            ),
            _completed([], returncode=1),
            _completed([]),
            _completed([]),
        ]
    )
    git.GitRepository(tmp_path, runner).ensure_issue_worktree(
        "gdw/issue-9", "master", path
    )
    assert [call[0] for call in runner.calls] == [
        ["git", "worktree", "prune"],
        ["git", "worktree", "list", "--porcelain"],
        ["git", "rev-parse", "--verify", "--quiet", "refs/heads/gdw/issue-9"],
        ["git", "rev-parse", "--verify", "--quiet", "refs/remotes/origin/master"],
        ["git", "worktree", "add", "-b", "gdw/issue-9", str(path), "origin/master"],
    ]


def test_ensure_issue_worktree_falls_back_to_the_local_branch_without_a_remote(
    tmp_path: Path,
) -> None:
    path = tmp_path / "issue" / "worktree"
    runner = ScriptedRun(
        [
            _completed([], stdout=""),
            _completed(
                [], stdout="worktree /repo\nHEAD abc\nbranch refs/heads/master\n"
            ),
            _completed([], returncode=1),
            _completed([], returncode=1),
            _completed([]),
        ]
    )
    git.GitRepository(tmp_path, runner).ensure_issue_worktree(
        "gdw/issue-9", "master", path
    )
    assert runner.calls[-1][0] == [
        "git",
        "worktree",
        "add",
        "-b",
        "gdw/issue-9",
        str(path),
        "master",
    ]


def test_ensure_issue_worktree_resumes_an_already_registered_worktree(
    tmp_path: Path,
) -> None:
    path = tmp_path / "issue" / "worktree"
    runner = ScriptedRun(
        [
            _completed([], stdout=""),
            _completed(
                [],
                stdout=f"worktree {path}\nHEAD abc\nbranch refs/heads/gdw/issue-9\n",
            ),
        ]
    )
    git.GitRepository(tmp_path, runner).ensure_issue_worktree(
        "gdw/issue-9", "master", path
    )
    assert [call[0] for call in runner.calls] == [
        ["git", "worktree", "prune"],
        ["git", "worktree", "list", "--porcelain"],
    ]


def test_ensure_issue_worktree_resumes_an_existing_branch_without_a_worktree(
    tmp_path: Path,
) -> None:
    path = tmp_path / "issue" / "worktree"
    runner = ScriptedRun(
        [
            _completed([], stdout=""),
            _completed([], stdout=""),
            _completed([], stdout="abc123\n"),
            _completed([]),
        ]
    )
    git.GitRepository(tmp_path, runner).ensure_issue_worktree(
        "gdw/issue-9", "master", path
    )
    assert [call[0] for call in runner.calls] == [
        ["git", "worktree", "prune"],
        ["git", "worktree", "list", "--porcelain"],
        ["git", "rev-parse", "--verify", "--quiet", "refs/heads/gdw/issue-9"],
        ["git", "worktree", "add", str(path), "gdw/issue-9"],
    ]


def test_ensure_issue_worktree_rejects_an_unregistered_existing_path(
    tmp_path: Path,
) -> None:
    runner = ScriptedRun(
        [
            _completed([], stdout=""),
            _completed([], stdout=""),
        ]
    )
    with pytest.raises(errors.WorkflowError, match="not a registered git worktree"):
        git.GitRepository(tmp_path, runner).ensure_issue_worktree(
            "gdw/issue-9", "master", tmp_path
        )


def test_progress_redraws_are_collapsed_out_of_ci_evidence() -> None:
    """mutmut repaints one status line per mutant, and only the tail of the
    output is kept: verbatim frames push the real errors out of the evidence."""
    spinner = "\n".join(f"{glyph} 1719/1719  killed 1659" for glyph in "⠦⠧⠇⠏⠋")
    noisy = f"ruff: would reformat orchestrator/__init__.py\n{spinner}\nError 1"

    assert ci_gates.readable(noisy) == (
        "ruff: would reformat orchestrator/__init__.py\n1719/1719  killed 1659\nError 1"
    )
    assert ci_gates.readable("same\rsame") == "same"
    assert ci_gates.readable("keep\nboth") == "keep\nboth"


def test_a_gate_reason_comes_from_its_own_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Separate stdout/stderr pipes dump every gate's diagnostics after the
    last announcement. Merging the streams at spawn keeps a failing gate's
    stderr inside its own block."""
    monkeypatch.delenv("MAKEFLAGS", raising=False)
    monkeypatch.delenv("MFLAGS", raising=False)
    monkeypatch.delenv("MAKELEVEL", raising=False)
    monkeypatch.setenv("JOBS", "1")
    (tmp_path / "Makefile").write_text(
        "\n".join(
            [
                ".PHONY: ci lint types",
                "MAKEFLAGS += -k",
                "ifneq ($(filter output-sync,$(.FEATURES)),)",
                "MAKEFLAGS += --output-sync=target",
                "endif",
                "gate = @printf '\\n=== gate: %s ===\\n' $@",
                "ci-gates:",
                "\t@printf '%s\\n' lint types",
                "lint:",
                "\t$(gate)",
                "\t@echo uv run ruff check",
                "\t@echo Found 12 errors. >&2",
                "\t@false",
                "types:",
                "\t$(gate)",
                "\t@echo uv run ty check",
                "ci: lint types",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = git.GitRepository(tmp_path).run_ci(timeout=30)

    lint = next(gate for gate in result.gates if gate.name == "lint")
    types = next(gate for gate in result.gates if gate.name == "types")
    assert lint.status == "failed"
    assert "Found 12 errors." in lint.reason
    assert types.status == "passed"
    assert "Found 12 errors." not in types.reason


def test_run_ci_reports_the_readable_evidence_not_the_raw_redraws(
    tmp_path: Path,
) -> None:
    frames = "\n".join(f"{glyph} working" for glyph in "⠦⠧⠇")
    run = ScriptedRun(
        [
            _completed([], stdout="lint\n"),
            _completed([], 2, stdout=f"boom\n{frames}", stderr=""),
        ]
    )
    repository = git.GitRepository(tmp_path, run)

    result = repository.run_ci()

    assert result.returncode == 2
    assert result.output == "boom\nworking"


CI_LOG = """
=== gate: lint ===
uv run ruff check .
All checks passed!

=== gate: types ===
uv run ty check
error[invalid-assignment] orchestrator/state.py:41: not assignable
Found 3 diagnostics.
make: *** [Makefile:88: types] Error 1
make: *** Waiting for unfinished jobs....

=== gate: mutation ===
mutation score 95.4% (1671 killed, 78 survived, 1751 mutants) floor 98.0%
make: *** [Makefile:110: mutation] Error 1
"""


def test_each_gate_is_reported_as_passed_failed_or_never_started() -> None:
    """A gate that never announced itself never ran, which is not passing."""
    gates = ci_gates.gate_results(("lint", "types", "mutation", "secrets"), CI_LOG)

    assert [gate.as_json() for gate in gates] == [
        {"name": "lint", "status": "passed", "reason": ""},
        {
            "name": "types",
            "status": "failed",
            "reason": (
                "uv run ty check error[invalid-assignment] "
                "orchestrator/state.py:41: not assignable Found 3 diagnostics."
            ),
        },
        {
            "name": "mutation",
            "status": "failed",
            "reason": (
                "mutation score 95.4% (1671 killed, 78 survived, 1751 mutants) "
                "floor 98.0%"
            ),
        },
        {"name": "secrets", "status": "not run", "reason": "not run"},
    ]
    assert ci_gates._gate_checklist(gates) == [
        "✅ lint",
        "❌ types — uv run ty check error[invalid-assignment] "
        "orchestrator/state.py:41: not assignable Found 3 diagnostics.",
        "❌ mutation — mutation score 95.4% (1671 killed, 78 survived, "
        "1751 mutants) floor 98.0%",
        "⚪ secrets — not run",
    ]


def test_a_gate_the_makefile_never_advertised_is_still_reported() -> None:
    """The advertised list is the Makefile's; a run that contradicts it is
    evidence about the run, not a reason to drop a gate from the report."""
    gates = ci_gates.gate_results((), CI_LOG)

    assert [(gate.name, gate.status) for gate in gates] == [
        ("lint", "passed"),
        ("mutation", "failed"),
        ("types", "failed"),
    ]


def test_a_failure_reason_is_a_headline_not_a_log() -> None:
    wordy = "=== gate: lint ===\n" + " ".join(f"word{index}" for index in range(40))
    wordy += "\nmake: *** [Makefile:1: lint] Error 1\n"

    reason = ci_gates.gate_results(("lint",), wordy)[0].reason

    assert reason.split() == [f"word{index}" for index in range(15)] + ["…"]


def test_a_gate_that_failed_silently_says_so() -> None:
    silent = "=== gate: secrets ===\nmake: *** [Makefile:1: secrets] Error 1\n"

    assert ci_gates.gate_results(("secrets",), silent) == (
        ci_gates.GateResult("secrets", "failed", "failed without saying anything"),
    )


def test_a_run_whose_gates_are_unknown_still_reports_a_verdict() -> None:
    """An older make cannot list its gates; the run still owes the issue an
    answer, so it reports as the one command it was."""
    assert ci_gates.CommandResult(0, "green").checklist() == ["✅ make ci"]
    assert ci_gates.CommandResult(2, "boom").checklist() == [
        "❌ make ci — exit 2, no gate named itself"
    ]


def test_gates_are_read_from_the_whole_log_not_the_bounded_tail(
    tmp_path: Path,
) -> None:
    """Only the tail of a long run is kept as evidence. Reading the gates from
    that tail would report every early gate as never started."""
    # Distinct lines: identical consecutive ones collapse as progress redraws.
    filler = "\n".join(f"noise {index}" for index in range(ci_gates.CI_EVIDENCE_CHARS))
    log = f"=== gate: lint ===\nok\n{filler}\n=== gate: secrets ===\nclean\n"
    run = ScriptedRun(
        [
            _completed([], stdout="lint\nsecrets\n"),
            _completed([], stdout=log, stderr=""),
        ]
    )

    result = git.GitRepository(tmp_path, run).run_ci()

    assert "=== gate: lint ===" not in result.output
    assert result.checklist() == ["✅ lint", "✅ secrets"]


def test_a_make_that_cannot_list_its_gates_does_not_stop_ci(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    run = ScriptedRun(
        [
            _completed([], 2, stdout="", stderr="No rule to make target 'ci-gates'"),
            _completed([], 1, stdout="boom", stderr=""),
        ]
    )

    with caplog.at_level(logging.WARNING, logger="gdw"):
        result = git.GitRepository(tmp_path, run).run_ci()

    assert result.gates == ()
    assert result.checklist() == ["❌ make ci — exit 1, no gate named itself"]
    assert "unstarted gates will go unreported" in caplog.text


def test_bounded_evidence_keeps_the_tail_and_command_results_report_gates() -> None:
    assert ci_gates.bounded("short", 5) == "short"
    assert ci_gates.bounded("123456", 3) == "… output truncated …\n456"
    assert ci_gates.CommandResult(0, "ok").succeeded is True
    assert ci_gates.CommandResult(1, "bad").succeeded is False
    assert ci_gates.CommandResult(1, "bad").as_json() == {
        "returncode": 1,
        "output": "bad",
        "gates": [],
    }
