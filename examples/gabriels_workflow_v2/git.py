"""Git and CI mechanics: the steps of a run that need no model judgment."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from examples.gabriels_workflow_v2.errors import (
    LOGGER,
    WorkflowError,
    WorkflowStopped,
)
from examples.gabriels_workflow_v2.gates import (
    CommandResult,
    bounded,
    gate_results,
    readable,
)

DEFAULT_CI_TIMEOUT = 7_200

DEFAULT_GATE_LIST_TIMEOUT = 60


class GitRepository:
    """Deterministic git and CI operations that do not require model judgment."""

    def __init__(
        self,
        root: Path,
        run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self.root = root
        self._run_process = run

    def prepare(self, base_branch: str, resuming: bool) -> tuple[str, str]:
        branch = self._call("branch", "--show-current").strip()
        if not branch:
            raise WorkflowError("the workflow requires a named git branch")
        if branch == base_branch:
            raise WorkflowError(
                f"refusing to develop directly on protected base branch '{branch}'"
            )
        if not resuming and self._call("status", "--porcelain").strip():
            raise WorkflowError("start the workflow from a clean worktree")
        head = self._call("rev-parse", "HEAD").strip()
        LOGGER.info(
            "git: branch=%s base=%s head=%s resuming=%s",
            branch,
            base_branch,
            head,
            resuming,
        )
        return branch, head

    def ensure_issue_worktree(self, branch: str, base_branch: str, path: Path) -> None:
        """Create or resume the linked worktree an issue develops in.

        Prunes registrations for worktrees whose directory disappeared, then
        resumes an already-registered worktree or an existing branch without
        one, and only branches off the base when neither exists yet.

        That base is `origin/<base_branch>` when the remote-tracking ref
        exists, not the local branch of the same name. The ratchet and
        diff-coverage gates measure against `origin/master`, so a run started
        from a checkout whose local `master` had fallen behind produced a
        worktree that failed CI on commits it did not contain — a failure no
        agent can repair, because the fix is a merge the agent is told not to
        make.
        """

        self._call("worktree", "prune")
        registered = {
            line.split(" ", 1)[1]
            for line in self._call("worktree", "list", "--porcelain").splitlines()
            if line.startswith("worktree ")
        }
        if str(path) in registered:
            return
        if path.exists():
            raise WorkflowError(f"{path} exists but is not a registered git worktree")
        verify = self._run_process(
            ["git", "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"],
            cwd=str(self.root),
            capture_output=True,
            text=True,
            check=False,
            stdin=subprocess.DEVNULL,
        )
        if verify.returncode == 0:
            self._call("worktree", "add", str(path), branch)
        else:
            start = self._start_point(base_branch)
            LOGGER.info("git: branching %s off %s", branch, start)
            self._call("worktree", "add", "-b", branch, str(path), start)

    def _start_point(self, base_branch: str) -> str:
        """`origin/<base>` when it exists, else `<base>` for a remoteless repo."""

        remote = f"origin/{base_branch}"
        verify = self._run_process(
            ["git", "rev-parse", "--verify", "--quiet", f"refs/remotes/{remote}"],
            cwd=str(self.root),
            capture_output=True,
            text=True,
            check=False,
            stdin=subprocess.DEVNULL,
        )
        return remote if verify.returncode == 0 else base_branch

    def ci_gates(self) -> tuple[str, ...]:
        """The gates `make ci` will attempt, named by the Makefile itself.

        Asked before the run, so a gate that never started can be told apart
        from one that passed. A make too old to answer costs the checklist its
        unstarted gates, never its verdicts, so the run goes ahead regardless.
        """

        proc = self._run_process(
            ["make", "--no-print-directory", "ci-gates"],
            cwd=str(self.root),
            capture_output=True,
            text=True,
            check=False,
            stdin=subprocess.DEVNULL,
            timeout=DEFAULT_GATE_LIST_TIMEOUT,
        )
        if proc.returncode != 0:
            LOGGER.warning(
                "ci: 'make ci-gates' exited %s; unstarted gates will go unreported",
                proc.returncode,
            )
            return ()
        gates = tuple(line.strip() for line in proc.stdout.splitlines() if line.strip())
        LOGGER.info("ci: %s gate(s) expected: %s", len(gates), ", ".join(gates))
        return gates

    def run_ci(self, timeout: int = DEFAULT_CI_TIMEOUT) -> CommandResult:
        LOGGER.info("ci: running 'make ci' in %s (timeout %ss)", self.root, timeout)
        expected = self.ci_gates()
        started = time.monotonic()
        proc = self._run_process(
            ["make", "ci"],
            cwd=str(self.root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
        )
        combined = (proc.stdout or "").strip()
        evidence = readable(combined)
        LOGGER.info(
            "ci: 'make ci' exited %s after %.1fs with %s chars of output "
            "(%s after collapsing progress redraws)",
            proc.returncode,
            time.monotonic() - started,
            len(combined),
            len(evidence),
        )
        gates = gate_results(expected, evidence)
        LOGGER.info(
            "ci: %s",
            ", ".join(f"{gate.name}={gate.status}" for gate in gates)
            or "no gates seen",
        )
        return CommandResult(proc.returncode, bounded(evidence), gates)

    def ensure_branch_ahead(self, message: str, base_sha: str) -> None:
        """Make the branch pushable as a pull request, without implementation.

        GitHub refuses a PR whose head matches the base. An empty commit is
        enough to open the draft that later bot comments belong on; the
        implementation commit still happens in publish().
        """

        commits = self._call("rev-list", "--count", f"{base_sha}..HEAD").strip()
        if commits != "0":
            LOGGER.info("git: already %s commit(s) ahead of %s", commits, base_sha)
            return
        LOGGER.info("git: creating empty start-work commit ahead of %s", base_sha)
        self._call("commit", "--allow-empty", "-m", message)

    def commit(self, message: str, base_sha: str) -> None:
        tracked = self._call("diff", "--no-renames", "--name-only", "-z", "HEAD").split(
            "\0"
        )
        untracked = self._call(
            "ls-files", "--others", "--exclude-standard", "-z"
        ).split("\0")
        changed_paths = [path for path in (*tracked, *untracked) if path]
        LOGGER.info("git: %s path(s) changed since HEAD", len(changed_paths))
        LOGGER.debug("git: changed paths %s", changed_paths)
        if changed_paths:
            self._call("add", "--", *changed_paths)
            self._call("commit", "-m", message)
            LOGGER.info("git: committed %r", message)
        commits = self._call("rev-list", "--count", f"{base_sha}..HEAD").strip()
        LOGGER.info("git: %s commit(s) ahead of %s", commits, base_sha)
        if commits == "0":
            raise WorkflowStopped("implementation produced no commits")

    def push(self, branch: str) -> None:
        LOGGER.info("git: pushing %s to origin", branch)
        self._call("push", "--set-upstream", "origin", branch)
        LOGGER.info("git: pushed %s", branch)

    def _call(self, *args: str) -> str:
        proc = self._run_process(
            ["git", *args],
            cwd=str(self.root),
            capture_output=True,
            text=True,
            check=False,
            stdin=subprocess.DEVNULL,
        )
        if proc.returncode != 0:
            command = " ".join(("git", *args))
            LOGGER.error("git: %s failed: %s", command, proc.stderr.strip())
            raise WorkflowError(f"{command} failed: {proc.stderr.strip()}")
        return proc.stdout
