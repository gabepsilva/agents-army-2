"""Real-`bwrap` behavior of the agent-turn sandbox.

These exercise the actual isolation `AgentGateway._run_cli` builds, not a
faked subprocess boundary — per AGENTS.md, "fake the subprocess boundary to
claude/codex/grok/opencode, never the unit under test" means bubblewrap
itself, the thing that provides the isolation, must run for real here or the
tests couldn't prove anything about the isolation at all.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from examples.gabriels_workflow import development_workflow as gdw

pytestmark = [
    pytest.mark.skipif(
        shutil.which("bwrap") is None, reason="bwrap not installed (#47)"
    ),
    pytest.mark.skipif(sys.platform != "linux", reason="bwrap is Linux-only (#47)"),
]


def _run_sandboxed(tmp_path: Path, *, role: str, shell_command: str):
    """Build one real `bwrap` argv the way `AgentGateway._run_cli` does, and run it."""

    worktree = tmp_path / "worktree"
    worktree.mkdir(parents=True, exist_ok=True)
    state_dir = tmp_path / "agents"
    state_dir.mkdir(parents=True, exist_ok=True)
    schema = tmp_path / "schema.json"
    schema.write_text("{}", encoding="utf-8")

    gateway = gdw.AgentGateway(
        roles={},
        issue=1,
        state_file=state_dir / "agents.json",
        example_root=Path(gdw.__file__).parent,
        workdir=worktree,
    )
    with gateway._sandbox_workspace() as (environment, ephemeral_home, isolation):
        context = gdw.SandboxContext(
            role=role,
            backend="codex",
            worktree=worktree,
            state_dir=state_dir,
            schema_path=schema,
            environment=environment,
            ephemeral_home=ephemeral_home,
            isolation_dir=isolation,
            real_home=Path.home(),
        )
        argv = gdw._build_bwrap_argv(
            gateway._bwrap_path, context, ["sh", "-c", shell_command]
        )
        return (
            subprocess.run(
                argv, cwd=str(worktree), capture_output=True, text=True, timeout=15
            ),
            worktree,
        )


def test_implementer_can_write_the_worktree_and_host_sees_it(tmp_path: Path) -> None:
    result, worktree = _run_sandboxed(
        tmp_path, role="implementer", shell_command="echo written > probe.txt"
    )
    assert result.returncode == 0, result.stderr
    assert (worktree / "probe.txt").read_text(encoding="utf-8") == "written\n"


def test_reviewer_quality_cannot_write_the_worktree(tmp_path: Path) -> None:
    result, worktree = _run_sandboxed(
        tmp_path, role="reviewer-quality", shell_command="echo written > probe.txt"
    )
    assert result.returncode != 0
    assert not (worktree / "probe.txt").exists()


@pytest.fixture
def outside_tmp_dir():
    """A directory outside `/tmp`/`/var/tmp`/`/dev/shm`, unlike pytest's `tmp_path`.

    `_private_tmpfs_flags` mounts a fresh `tmpfs` over `/tmp` *after*
    `_sensitive_shadow_flags` runs, so a sentinel planted under pytest's
    `tmp_path` (itself under `/tmp`) would be masked by that private `/tmp`
    regardless of whether the shadow bind actually did anything -- the probe
    would "pass" even with the shadowing logic deleted. Rooting the sentinel
    under the real `$HOME` instead keeps the probe honest: only the shadow
    mount, not the private `/tmp`, can account for the read failing.
    """

    directory = Path(tempfile.mkdtemp(dir=str(Path.home()), prefix=".gdw-test-home-"))
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_sensitive_home_paths_are_unreadable_inside_the_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outside_tmp_dir: Path
) -> None:
    fake_home = outside_tmp_dir
    (fake_home / ".aws").mkdir(parents=True)
    (fake_home / ".aws" / "credentials").write_text("secret", encoding="utf-8")
    monkeypatch.setenv("HOME", str(fake_home))
    result, _worktree = _run_sandboxed(
        tmp_path,
        role="implementer",
        shell_command=f"cat {fake_home}/.aws/credentials",
    )
    assert result.returncode != 0
    assert "secret" not in result.stdout


def test_ssh_auth_sock_is_unreadable_inside_the_sandbox_when_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outside_tmp_dir: Path
) -> None:
    sock = outside_tmp_dir / "fake.sock"
    sock.write_text("socket-sentinel", encoding="utf-8")
    monkeypatch.setenv("SSH_AUTH_SOCK", str(sock))
    result, _worktree = _run_sandboxed(
        tmp_path, role="implementer", shell_command=f"cat {sock}"
    )
    assert result.returncode != 0
    assert "socket-sentinel" not in result.stdout


def test_proc_and_dev_are_sandbox_local_not_the_hosts(tmp_path: Path) -> None:
    host_proc_count = len(
        [entry for entry in Path("/proc").iterdir() if entry.name.isdigit()]
    )
    result, _worktree = _run_sandboxed(
        tmp_path,
        role="implementer",
        shell_command="ls /proc | grep -Ec '^[0-9]+$'; ls /dev",
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    sandbox_proc_count = int(lines[0])
    assert sandbox_proc_count < host_proc_count
    assert sandbox_proc_count <= 9
    dev_entries = set(lines[1:])
    assert dev_entries <= {
        "null",
        "zero",
        "full",
        "random",
        "urandom",
        "tty",
        "shm",
        "pts",
        "ptmx",
        "stdin",
        "stdout",
        "stderr",
        "fd",
        "core",
        "kcore",
    }


def test_missing_bwrap_fails_closed_before_any_orchestrator_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs):
        calls.append(args)
        raise AssertionError("orchestrator must not run when bwrap is unavailable")

    monkeypatch.setattr(gdw.shutil, "which", lambda _name: None)
    with pytest.raises(gdw.WorkflowError, match=r"(?i)bubblewrap|bwrap"):
        gdw.AgentGateway(
            roles={},
            issue=1,
            state_file=tmp_path / "state" / "agents.json",
            example_root=Path(gdw.__file__).parent,
            workdir=tmp_path / "worktree",
            run=fake_run,
        )
    assert calls == []
