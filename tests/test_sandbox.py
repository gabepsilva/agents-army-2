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

from examples.gabriels_workflow_v2 import gateway as gdw

pytestmark = [
    pytest.mark.skipif(
        shutil.which("bwrap") is None, reason="bwrap not installed (#47)"
    ),
    pytest.mark.skipif(sys.platform != "linux", reason="bwrap is Linux-only (#47)"),
]


@pytest.fixture(autouse=True)
def remove_overlay_workdirs(tmp_path: Path):
    """Take back the `overlayfs` work directories before pytest tries to.

    The kernel leaves `work/work` mode 000 and owned by us. pytest's own
    `rm_rf` of an old `tmp_path` trips over it, and `filterwarnings = error`
    turns that into a failure in whichever run happens to do the cleanup.
    """

    yield
    # A fixed-depth glob, not a recursive one: `work/work` is mode 000, and
    # walking into it is itself what raises during interpreter shutdown.
    for workdir in tmp_path.glob("agents/home/*/*/work"):
        inner = workdir / "work"
        if inner.is_dir():
            inner.chmod(0o700)
        shutil.rmtree(workdir, ignore_errors=True)


def _run_sandboxed(
    tmp_path: Path, *, role: str, shell_command: str, agent: str = "gdw-1-agent"
):
    """Build one real `bwrap` argv the way `AgentGateway._run_cli` does, and run it."""

    worktree = tmp_path / "worktree"
    worktree.mkdir(parents=True, exist_ok=True)
    state_dir = tmp_path / "agents"
    agent_home = state_dir / "home" / agent
    (agent_home / "files").mkdir(parents=True, exist_ok=True)
    for relative in gdw.BACKEND_HOME_DIRS["codex"]:
        layer = agent_home / gdw._layer_name(relative)
        for part in ("upper", "work"):
            (layer / part).mkdir(parents=True, exist_ok=True)
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
            agent_home=agent_home,
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


def test_backend_config_writes_persist_across_turns_but_not_to_the_real_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outside_tmp_dir: Path
) -> None:
    """A backend CLI records its conversation under its own config directory.

    Regression: that directory used to be `--ro-bind`ed under a per-turn
    `tmpfs` `$HOME`, so every write vanished at turn exit and the next turn's
    `--resume <session>` found no conversation. Every agent asked twice died.
    """

    fake_home = outside_tmp_dir
    config = fake_home / ".codex"
    config.mkdir(parents=True)
    (config / "auth.json").write_text("login", encoding="utf-8")
    monkeypatch.setenv("HOME", str(fake_home))

    first, _worktree = _run_sandboxed(
        tmp_path,
        role="implementer",
        shell_command="cat $HOME/.codex/auth.json && echo session > $HOME/.codex/s.json",
    )
    assert first.returncode == 0, first.stderr
    assert "login" in first.stdout

    second, _worktree = _run_sandboxed(
        tmp_path, role="implementer", shell_command="cat $HOME/.codex/s.json"
    )
    assert second.returncode == 0, second.stderr
    assert "session" in second.stdout
    assert not (config / "s.json").exists()


def test_reviewer_cannot_write_the_backend_config_through_the_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outside_tmp_dir: Path
) -> None:
    """The overlay is per-issue scratch, never a path back to the real config.

    A settings file under a backend's real config directory can carry hooks,
    so a turn that could write it would be executing on the host outside the
    sandbox.
    """

    fake_home = outside_tmp_dir
    config = fake_home / ".codex"
    config.mkdir(parents=True)
    (config / "settings.json").write_text("original", encoding="utf-8")
    monkeypatch.setenv("HOME", str(fake_home))

    result, _worktree = _run_sandboxed(
        tmp_path,
        role="reviewer-quality",
        shell_command="echo tampered > $HOME/.codex/settings.json; cat $HOME/.codex/settings.json",
    )

    assert result.returncode == 0, result.stderr
    # The write lands in the issue's upper layer, so the turn sees its own
    # edit — and the real file it was overlaid from is untouched.
    assert result.stdout.strip() == "tampered"
    assert (config / "settings.json").read_text(encoding="utf-8") == "original"


def test_every_backend_state_directory_survives_to_the_next_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outside_tmp_dir: Path
) -> None:
    """Not just the dotfile — the XDG directories a CLI actually writes to.

    Regression: only the first directory per backend was overlaid, so opencode
    kept its config across turns but wrote conversations to `~/.local/share`,
    which stayed a per-turn `tmpfs`. Its second turn died with "Session not
    found" — the bug the dotfile-only overlay was supposed to have fixed.
    """

    fake_home = outside_tmp_dir
    monkeypatch.setenv("HOME", str(fake_home))
    for relative in gdw.BACKEND_HOME_DIRS["codex"]:
        (fake_home / relative).mkdir(parents=True, exist_ok=True)

    writes = " && ".join(
        f"echo session > $HOME/{relative}/probe"
        for relative in gdw.BACKEND_HOME_DIRS["codex"]
    )
    first, _worktree = _run_sandboxed(
        tmp_path, role="implementer", shell_command=writes
    )
    assert first.returncode == 0, first.stderr

    reads = " && ".join(
        f"cat $HOME/{relative}/probe" for relative in gdw.BACKEND_HOME_DIRS["codex"]
    )
    second, _worktree = _run_sandboxed(
        tmp_path, role="implementer", shell_command=reads
    )

    assert second.returncode == 0, second.stderr
    assert second.stdout.count("session") == len(gdw.BACKEND_HOME_DIRS["codex"])
    for relative in gdw.BACKEND_HOME_DIRS["codex"]:
        assert not (fake_home / relative / "probe").exists()


def test_two_agents_on_one_backend_do_not_collide_on_the_config_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outside_tmp_dir: Path
) -> None:
    """Each agent gets its own layer, so a lingering server cannot block one.

    Regression: the layer was keyed by backend. `overlayfs` refuses a second
    mount sharing a live upperdir with EBUSY, and some backend CLIs leave a
    server running past their turn — so the next agent on the same backend
    died with "Can't make overlay mount ... Device or resource busy" before
    its turn began.
    """

    fake_home = outside_tmp_dir
    (fake_home / ".codex").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(fake_home))

    # Stand in for a backend CLI whose server outlives its turn, still
    # holding that agent's overlay when the next agent starts.
    first = tmp_path / "agents" / "home" / "gdw-1-first" / ".codex"
    for part in ("upper", "work"):
        (first / part).mkdir(parents=True, exist_ok=True)
    mount = tmp_path / "held"
    mount.mkdir()
    holder = subprocess.Popen(
        [
            "bwrap",
            "--ro-bind",
            "/",
            "/",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--overlay-src",
            str(fake_home / ".codex"),
            "--overlay",
            str(first / "upper"),
            str(first / "work"),
            str(mount),
            "--",
            "sleep",
            "10",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        second, _worktree = _run_sandboxed(
            tmp_path,
            role="implementer",
            shell_command="echo second-turn-ran",
            agent="gdw-1-second",
        )
    finally:
        holder.kill()
        holder.wait(timeout=10)

    assert second.returncode == 0, second.stderr
    assert "second-turn-ran" in second.stdout


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
