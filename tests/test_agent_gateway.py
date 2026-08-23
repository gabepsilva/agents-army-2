"""One sandboxed agent turn, from filled prompt to structured reply.

The subprocess boundary to the orchestrator CLI is faked here so the argv and
the environment `AgentGateway` builds can be asserted exactly; `test_sandbox.py`
runs the same argv through a real `bwrap` to prove the isolation itself.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.gabriels_workflow_v2 import errors
from examples.gabriels_workflow_v2 import gateway as gdw
from examples.gabriels_workflow_v2.config import AGENT_ROLES, RoleConfig


@pytest.fixture(autouse=True)
def _isolate_workflow_logging():
    """Keep configure_logging() calls from leaking handlers into later tests."""

    logger = errors.LOGGER
    handlers, level = list(logger.handlers), logger.level
    yield
    logger.handlers = handlers
    logger.setLevel(level)


def _completed(
    args: list[str], returncode: int = 0, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args, returncode, stdout=stdout, stderr=stderr)


@pytest.fixture
def available_bwrap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate the external sandbox capability for gateway unit tests."""

    monkeypatch.setattr(gdw.shutil, "which", lambda _name: "/usr/bin/bwrap")
    monkeypatch.setattr(
        gdw.subprocess,
        "run",
        lambda args, **_kwargs: _completed(args),
    )


def _bare_turn_sandbox(tmp_path: Path) -> gdw.TurnSandbox:
    """The `TurnSandbox` `_run_cli` needs to build a `bwrap` argv.

    For tests that call `_run_cli` directly to exercise its own error
    handling rather than a real turn's sandbox shape.
    """

    isolation = tmp_path / "isolation"
    isolation.mkdir(exist_ok=True)
    ephemeral_home = isolation / "home"
    ephemeral_home.mkdir(exist_ok=True)
    return gdw.TurnSandbox(
        role="implementer",
        backend="codex",
        schema_path=tmp_path / "schema.json",
        agent_name="gdw-1-implementer",
        ephemeral_home=ephemeral_home,
        isolation_dir=isolation,
    )


def _implementation(status: str = "complete") -> dict:
    return {
        "status": status,
        "summary": "implemented",
        "files_changed": ["feature.py"],
        "tests_run": ["pytest"],
        "blockers": [] if status == "complete" else ["missing decision"],
    }


def _expansion(decision: str = "proceed", needs_another_round: bool = False) -> dict:
    return {
        "decision": decision,
        "needs_another_round": needs_another_round,
        "reason": "proposal converged" if not needs_another_round else "needs review",
        "summary": "proposal",
        "current_state": ["current"],
        "proposed_changes": ["change"],
        "out_of_scope": [],
        "risks": [],
        "open_questions": [],
    }


def test_agent_gateway_validates_prompt_and_removes_github_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, available_bwrap: None
) -> None:
    calls: list[tuple[list[str], dict]] = []

    def fake_run(args: list[str], **kwargs):
        calls.append((args, kwargs))
        environment = kwargs["env"]
        assert [environment.get(key) for key in gdw.GITHUB_TOKEN_NAMES] == [
            None,
            None,
            None,
            None,
            None,
            None,
        ]
        assert environment["GH_CONFIG_DIR"].endswith("empty-gh-config")
        assert Path(environment["GH_CONFIG_DIR"]).is_dir()
        blocker = Path(environment["PATH"].split(os.pathsep)[0]) / "gh"
        assert "owned by the GDW driver" in blocker.read_text(encoding="utf-8")
        return _completed(
            args,
            stdout=f"[gdw-3-expander session=s1]\n{json.dumps(_expansion())}\n",
        )

    monkeypatch.setenv("GH_TOKEN", "secret")
    original_path = os.environ["PATH"]
    root = Path(gdw.__file__).parent
    gateway = gdw.AgentGateway(
        roles={
            "expander": SimpleNamespace(
                backend="codex", model=None, reasoning_effort=None
            )
        },
        issue=3,
        state_file=tmp_path / "state" / "agents.json",
        example_root=root,
        workdir=tmp_path / "worktree",
        run=fake_run,
    )
    result = gateway.ask(
        role="expander",
        prompt_name="expand",
        schema_name="proposal",
        values={"CONTEXT_JSON": "{}"},
        timeout=17,
    )
    assert result == _expansion()
    argv = calls[0][0]
    assert Path(argv[0]).name == "bwrap"
    assert argv[1:9] == [
        "--unshare-pid",
        "--unshare-uts",
        "--unshare-ipc",
        "--unshare-cgroup-try",
        "--unshare-user",
        "--die-with-parent",
        "--new-session",
        "--clearenv",
    ]
    assert argv[9:11] == ["--setenv", "PATH"]
    assert argv[12:14] == ["--setenv", "HOME"]
    assert Path(argv[14]) != Path.home()
    # PATH, HOME, AGENTS_ARMY_HOME, AGENTS_ARMY_STATE_FILE, then GH_CONFIG_DIR:
    # a fixed --setenv order (SandboxContext._setenv_flags's dict insertion
    # order), so GH_CONFIG_DIR lands at a known offset rather than needing a
    # search.
    assert argv[21:24] == [
        "--setenv",
        "GH_CONFIG_DIR",
        calls[0][1]["env"]["GH_CONFIG_DIR"],
    ]
    ro_bind_index = argv.index("--ro-bind")
    assert argv[ro_bind_index : ro_bind_index + 3] == ["--ro-bind", "/", "/"]
    proc_index = argv.index("--proc")
    assert argv[proc_index : proc_index + 4] == ["--proc", "/proc", "--dev", "/dev"]
    # expander is not a writable role: the worktree is read-only. Search for
    # the *bind* of the worktree specifically, since its path also appears
    # earlier as the --setenv AGENTS_ARMY_HOME value.
    worktree = str(tmp_path / "worktree")
    worktree_bind_indices = [
        index
        for index in range(len(argv) - 2)
        if argv[index] in ("--bind", "--ro-bind")
        and argv[index + 1] == worktree
        and argv[index + 2] == worktree
    ]
    assert worktree_bind_indices == [
        index for index in worktree_bind_indices if argv[index] == "--ro-bind"
    ]
    assert len(worktree_bind_indices) == 1
    state_dir = str(tmp_path / "state")
    assert (state_dir, state_dir) in _bind_pairs(argv, "--bind")
    assert (state_dir, state_dir) not in _bind_pairs(argv, "--ro-bind")
    terminator = argv.index("--")
    assert argv[terminator + 1 :][:9] == [
        "orchestrator",
        "talk",
        "gdw-3-expander",
        "--backend",
        "codex",
        "--schema",
        str(root / "validations" / "proposal.json"),
        "--timeout",
        "17",
    ]
    assert argv[terminator + 1 :][9] == "--prompt"
    assert len(calls) == 1
    assert calls[0][1]["env"]["AGENTS_ARMY_STATE_FILE"] == str(
        tmp_path / "state" / "agents.json"
    )
    assert calls[0][1]["timeout"] == 22
    assert calls[0][1]["cwd"] == worktree
    assert calls[0][1]["env"]["AGENTS_ARMY_HOME"] == worktree
    assert str(gateway.workdir) == worktree
    assert os.environ["GH_TOKEN"] == "secret"
    assert os.environ["PATH"] == original_path


def test_agent_turns_run_in_the_worktree_not_the_directory_the_driver_started_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, available_bwrap: None
) -> None:
    """Agents develop in the tree git and CI use, whatever the process's cwd.

    Regression: the gateway used to hand the orchestrator `Path.cwd()` and set
    no AGENTS_ARMY_HOME, so every agent edited the checkout the driver was
    started from while `make ci`, `git add` and `commit` ran in the issue
    worktree. Nothing failed until `commit` found an unchanged tree.
    """
    driver_cwd = tmp_path / "checkout"
    worktree = tmp_path / "checkout" / ".git" / "gdw" / "issue-3" / "worktree"
    worktree.mkdir(parents=True)
    monkeypatch.chdir(driver_cwd)
    calls: list[dict] = []

    def fake_run(args: list[str], **kwargs):
        calls.append(kwargs)
        return _completed(
            args,
            stdout=f"[gdw-3-expander session=s1]\n{json.dumps(_expansion())}\n",
        )

    gateway = gdw.AgentGateway(
        roles={
            "expander": SimpleNamespace(
                backend="codex", model=None, reasoning_effort=None
            )
        },
        issue=3,
        state_file=worktree.parent / "state" / "agents.json",
        example_root=Path(gdw.__file__).parent,
        workdir=worktree,
        run=fake_run,
    )
    gateway.ask(
        role="expander",
        prompt_name="expand",
        schema_name="proposal",
        values={"CONTEXT_JSON": "{}"},
    )

    assert calls[0]["cwd"] == str(worktree)
    assert calls[0]["env"]["AGENTS_ARMY_HOME"] == str(worktree)
    assert calls[0]["cwd"] != str(driver_cwd)
    assert calls[0]["env"]["AGENTS_ARMY_HOME"] != str(driver_cwd)


def test_agent_gateway_rejects_bad_prompts_backend_and_reply(
    tmp_path: Path, available_bwrap: None
) -> None:
    example_root = tmp_path / "example"
    (example_root / "prompts").mkdir(parents=True)
    (example_root / "validations").mkdir()
    schema = Path(gdw.__file__).parent / "validations" / "proposal.json"
    (example_root / "validations" / "proposal.json").write_text(
        schema.read_text(encoding="utf-8"), encoding="utf-8"
    )

    replies = deque(
        [
            _completed([], returncode=1, stderr="already uses backend/model/effort"),
            _completed([], stdout="[gdw-1-role session=s1]\nnull\n"),
        ]
    )

    def fake_run(args: list[str], **_kwargs):
        reply = replies.popleft()
        return _completed(args, reply.returncode, reply.stdout, reply.stderr)

    gateway = gdw.AgentGateway(
        roles={
            "role": SimpleNamespace(backend="codex", model=None, reasoning_effort=None)
        },
        issue=1,
        state_file=tmp_path / "state" / "agents.json",
        example_root=example_root,
        workdir=tmp_path / "worktree",
        run=fake_run,
    )
    with pytest.raises(gdw.WorkflowError, match="cannot read prompt"):
        gateway.ask(
            role="role",
            prompt_name="missing",
            schema_name="proposal",
            values={},
        )
    (example_root / "prompts" / "bad.md").write_text("{{MISSING}}", encoding="utf-8")
    with pytest.raises(gdw.WorkflowError, match="unresolved placeholders"):
        gateway.ask(role="role", prompt_name="bad", schema_name="proposal", values={})
    (example_root / "prompts" / "ok.md").write_text("ok", encoding="utf-8")
    with pytest.raises(gdw.WorkflowError, match="already uses backend/model/effort"):
        gateway.ask(role="role", prompt_name="ok", schema_name="proposal", values={})
    gateway.roles["role"] = SimpleNamespace(
        backend="claude", model=None, reasoning_effort=None
    )
    with pytest.raises(gdw.WorkflowError, match="no structured response"):
        gateway.ask(role="role", prompt_name="ok", schema_name="proposal", values={})


def test_agent_gateway_fills_prompts_with_brace_bearing_values(
    tmp_path: Path, available_bwrap: None
) -> None:
    """Braces in a substituted value are text, not an unfilled placeholder."""

    example_root = tmp_path / "example"
    (example_root / "prompts").mkdir(parents=True)
    (example_root / "validations").mkdir()
    schema = Path(gdw.__file__).parent / "validations" / "proposal.json"
    (example_root / "validations" / "proposal.json").write_text(
        schema.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (example_root / "prompts" / "grill.md").write_text(
        "review:\n{{EXPANSION_JSON}}\ncomments:\n{{LATEST_COMMENTS_JSON}}\n",
        encoding="utf-8",
    )
    # What an agent actually wrote back on issue #28: a dict comprehension
    # closing on `}}`, and prose naming another placeholder.
    expansion = "{role: f'{role}-x' for role in ROLES}} and {{LATEST_COMMENTS_JSON}}"

    sent: list[str] = []

    def fake_run(args: list[str], **_kwargs):
        sent.append(args[-1])
        return _completed(args, stdout='[gdw-28-role session=s1]\n{"ok": true}\n')

    gateway = gdw.AgentGateway(
        roles={
            "role": SimpleNamespace(backend="codex", model=None, reasoning_effort=None)
        },
        issue=28,
        state_file=tmp_path / "state" / "agents.json",
        example_root=example_root,
        workdir=tmp_path / "worktree",
        run=fake_run,
    )

    reply = gateway.ask(
        role="role",
        prompt_name="grill",
        schema_name="proposal",
        values={"EXPANSION_JSON": expansion, "LATEST_COMMENTS_JSON": "[]"},
    )

    assert reply == {"ok": True}
    assert sent == [f"review:\n{expansion}\ncomments:\n[]\n"]


def test_agent_gateway_names_the_placeholder_no_value_was_given_for(
    tmp_path: Path, available_bwrap: None
) -> None:
    example_root = tmp_path / "example"
    (example_root / "prompts").mkdir(parents=True)
    (example_root / "validations").mkdir()
    (example_root / "prompts" / "grill.md").write_text(
        "{{EXPANSION_JSON}} {{GRILL_JSON}} {{EXPANSION_JSON}}", encoding="utf-8"
    )

    gateway = gdw.AgentGateway(
        roles={
            "role": SimpleNamespace(backend="codex", model=None, reasoning_effort=None)
        },
        issue=28,
        state_file=tmp_path / "state" / "agents.json",
        example_root=example_root,
        workdir=tmp_path / "worktree",
        run=lambda args, **_kwargs: _completed(args),
    )

    with pytest.raises(
        gdw.WorkflowError, match=r"unresolved placeholders: EXPANSION_JSON, GRILL_JSON$"
    ):
        gateway.ask(
            role="role",
            prompt_name="grill",
            schema_name="proposal",
            values={},
        )


@pytest.mark.parametrize(
    ("turn_stdout", "message"),
    [
        ("header only", "no structured response"),
        ("[agent session=s1]\nnot-json\n", "invalid structured JSON"),
    ],
)
def test_agent_gateway_rejects_malformed_cli_output(
    tmp_path: Path, turn_stdout: str, message: str, available_bwrap: None
) -> None:
    replies = deque(
        [
            _completed([], stdout=turn_stdout),
        ]
    )

    def fake_run(args: list[str], **_kwargs):
        reply = replies.popleft()
        return _completed(args, reply.returncode, reply.stdout, reply.stderr)

    gateway = gdw.AgentGateway(
        roles={
            "expander": SimpleNamespace(
                backend="codex", model=None, reasoning_effort=None
            )
        },
        issue=1,
        state_file=tmp_path / "state" / "agents.json",
        example_root=Path(gdw.__file__).parent,
        workdir=tmp_path / "worktree",
        run=fake_run,
    )
    with pytest.raises(gdw.WorkflowError, match=message):
        gateway.ask(
            role="expander",
            prompt_name="expand",
            schema_name="proposal",
            values={"CONTEXT_JSON": "{}"},
        )


@pytest.mark.parametrize(
    "failure",
    [OSError("missing executable"), subprocess.TimeoutExpired("orchestrator", 3)],
)
def test_agent_gateway_reports_cli_launch_failures(
    tmp_path: Path, failure: BaseException, available_bwrap: None
) -> None:
    def failing_run(_args: list[str], **_kwargs):
        raise failure

    gateway = gdw.AgentGateway(
        roles={},
        issue=1,
        state_file=tmp_path / "state" / "agents.json",
        example_root=Path(gdw.__file__).parent,
        workdir=tmp_path / "worktree",
        run=failing_run,
    )
    with pytest.raises(gdw.WorkflowError, match="orchestrator CLI failed"):
        gateway._run_cli(
            ["list"], {"PATH": "/usr/bin"}, _bare_turn_sandbox(tmp_path), timeout=3
        )


def test_agent_gateway_reports_cli_exit_without_stderr(
    tmp_path: Path, available_bwrap: None
) -> None:
    gateway = gdw.AgentGateway(
        roles={},
        issue=1,
        state_file=tmp_path / "state" / "agents.json",
        example_root=Path(gdw.__file__).parent,
        workdir=tmp_path / "worktree",
        run=lambda args, **_kwargs: _completed(args, returncode=9),
    )
    with pytest.raises(gdw.WorkflowError, match="exited 9"):
        gateway._run_cli(
            ["list"], {"PATH": "/usr/bin"}, _bare_turn_sandbox(tmp_path), timeout=3
        )


def test_agent_gateway_uses_each_roles_backend_model_and_effort(
    tmp_path: Path, available_bwrap: None
) -> None:
    calls: list[list[str]] = []
    configured = RoleConfig(
        backend="grok",
        model="grok-code-test",
        reasoning_effort="xhigh",
    )

    def fake_run(args: list[str], **_kwargs):
        calls.append(args)
        return _completed(
            args,
            stdout=f"[gdw-5-expander session=s1]\n{json.dumps(_expansion())}\n",
        )

    gateway = gdw.AgentGateway(
        roles={"expander": configured},
        issue=5,
        state_file=tmp_path / "state" / "agents.json",
        example_root=Path(gdw.__file__).parent,
        workdir=tmp_path / "worktree",
        run=fake_run,
    )
    assert gateway.options("expander") is configured
    with pytest.raises(gdw.WorkflowError, match="role 'missing' is not configured"):
        gateway.options("missing")
    gateway.ask(
        role="expander",
        prompt_name="expand",
        schema_name="proposal",
        values={"CONTEXT_JSON": "{}"},
    )
    expected_prompt = gateway._prompt("expand", {"CONTEXT_JSON": "{}"})
    assert Path(calls[0][0]).name == "bwrap"
    terminator = calls[0].index("--")
    assert calls[0][terminator + 1 :] == [
        "orchestrator",
        "talk",
        "gdw-5-expander",
        "--backend",
        "grok",
        "--model",
        "grok-code-test",
        "--reasoning-effort",
        "xhigh",
        "--schema",
        str(Path(gdw.__file__).parent / "validations" / "proposal.json"),
        "--timeout",
        str(gdw.DEFAULT_AGENT_TIMEOUT),
        "--prompt",
        expected_prompt,
    ]
    with pytest.raises(gdw.WorkflowError, match="role 'missing' is not configured"):
        gateway.ask(
            role="missing",
            prompt_name="expand",
            schema_name="proposal",
            values={"CONTEXT_JSON": "{}"},
        )


def test_agent_gateway_attaches_skills_only_when_given(
    tmp_path: Path, available_bwrap: None
) -> None:
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs):
        calls.append(args)
        return _completed(
            args,
            stdout=f"[gdw-7-implementer session=s1]\n{json.dumps(_implementation())}\n",
        )

    gateway = gdw.AgentGateway(
        roles={
            "implementer": SimpleNamespace(
                backend="codex", model=None, reasoning_effort=None
            )
        },
        issue=7,
        state_file=tmp_path / "state" / "agents.json",
        example_root=Path(gdw.__file__).parent,
        workdir=tmp_path / "worktree",
        run=fake_run,
    )
    gateway.ask(
        role="implementer",
        prompt_name="implement",
        schema_name="work-report",
        values={"CONTEXT_JSON": "{}"},
        skills=("code-simplification",),
    )
    assert "--skill" in calls[0]
    assert calls[0][calls[0].index("--skill") + 1] == "code-simplification"

    gateway.ask(
        role="implementer",
        prompt_name="implement",
        schema_name="work-report",
        values={"CONTEXT_JSON": "{}"},
    )
    assert "--skill" not in calls[1]


def _sandboxed_argv(
    tmp_path: Path, *, role: str, backend: str = "codex"
) -> tuple[list[str], list[dict]]:
    """One recorded `bwrap` argv from a full `AgentGateway.ask` turn."""

    calls: list[tuple[list[str], dict]] = []

    def fake_run(args: list[str], **kwargs):
        calls.append((args, kwargs))
        return _completed(
            args,
            stdout=f"[gdw-9-{role} session=s1]\n{json.dumps(_implementation())}\n",
        )

    gateway = gdw.AgentGateway(
        roles={
            role: SimpleNamespace(backend=backend, model=None, reasoning_effort=None)
        },
        issue=9,
        state_file=tmp_path / "agents" / "agents.json",
        example_root=Path(gdw.__file__).parent,
        workdir=tmp_path / "worktree",
        run=fake_run,
    )
    (tmp_path / "worktree").mkdir()
    gateway.ask(
        role=role,
        prompt_name="implement",
        schema_name="work-report",
        values={"CONTEXT_JSON": "{}"},
    )
    return calls[0][0], [kwargs for _args, kwargs in calls]


def _bind_pairs(argv: list[str], flag: str) -> list[tuple[str, str]]:
    return [
        (argv[index + 1], argv[index + 2])
        for index in range(len(argv) - 2)
        if argv[index] == flag
    ]


@pytest.mark.parametrize("role", sorted(gdw.WRITABLE_ROLES))
def test_agent_gateway_binds_worktree_writable_for_writable_roles(
    tmp_path: Path, role: str, available_bwrap: None
) -> None:
    argv, _kwargs = _sandboxed_argv(tmp_path, role=role)
    worktree = str(tmp_path / "worktree")
    assert (worktree, worktree) in _bind_pairs(argv, "--bind")
    assert (worktree, worktree) not in _bind_pairs(argv, "--ro-bind")


@pytest.mark.parametrize("role", sorted(AGENT_ROLES - gdw.WRITABLE_ROLES))
def test_agent_gateway_binds_worktree_read_only_for_other_roles(
    tmp_path: Path, role: str, available_bwrap: None
) -> None:
    argv, _kwargs = _sandboxed_argv(tmp_path, role=role)
    worktree = str(tmp_path / "worktree")
    assert (worktree, worktree) in _bind_pairs(argv, "--ro-bind")
    assert (worktree, worktree) not in _bind_pairs(argv, "--bind")


def test_agent_gateway_bwrap_argv_binds_the_agent_state_directory_rw(
    tmp_path: Path, available_bwrap: None
) -> None:
    """The directory, not the state file: the orchestrator renames into it.

    `Orchestrator._persist` writes a sibling `.tmp` and renames it over the
    state file, and takes sibling lock files. Binding the file alone leaves
    its parent read-only, and every turn dies on the first `_persist`.
    """

    argv, _kwargs = _sandboxed_argv(tmp_path, role="implementer")
    state_dir = tmp_path / "agents"
    rw_binds = _bind_pairs(argv, "--bind")
    assert (str(state_dir), str(state_dir)) in rw_binds
    assert state_dir.is_dir()
    assert not (tmp_path / "agents.json").exists()


def test_agent_gateway_refuses_a_state_directory_inside_the_worktree(
    tmp_path: Path, available_bwrap: None
) -> None:
    """Overlap would hand every read-only role a writable tree.

    The state directory is bound read-write for all roles and comes after the
    worktree bind, so a state directory under the worktree — or a worktree
    under the state directory — silently re-mounts the tree read-write for a
    reviewer.
    """

    worktree = tmp_path / "worktree"
    for state_file in (
        worktree / "agents" / "agents.json",
        worktree / "agents.json",
        tmp_path / "agents.json",
    ):
        with pytest.raises(gdw.WorkflowError, match="overlaps the worktree"):
            gdw.AgentGateway(
                roles={},
                issue=1,
                state_file=state_file,
                example_root=Path(gdw.__file__).parent,
                workdir=worktree,
            )

    gateway = gdw.AgentGateway(
        roles={},
        issue=1,
        state_file=tmp_path / "state" / "agents.json",
        example_root=Path(gdw.__file__).parent,
        workdir=worktree,
    )
    assert gateway.workdir == worktree


def test_agent_gateway_bwrap_argv_ends_with_terminator_before_payload(
    tmp_path: Path, available_bwrap: None
) -> None:
    argv, _kwargs = _sandboxed_argv(tmp_path, role="implementer")
    terminator = argv.index("--")
    assert argv[terminator + 1] == "orchestrator"
    assert argv[terminator + 2] == "talk"


def test_agent_gateway_bwrap_shadows_sensitive_paths_that_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, available_bwrap: None
) -> None:
    fake_home = tmp_path / "fake-home"
    (fake_home / ".ssh").mkdir(parents=True)
    (fake_home / ".aws").mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("SSH_AUTH_SOCK", raising=False)
    argv, _kwargs = _sandboxed_argv(tmp_path, role="implementer")
    assert argv[argv.index(str(fake_home / ".ssh")) - 1] == "--tmpfs"
    assert argv[argv.index(str(fake_home / ".aws")) - 1] == "--tmpfs"
    # .netrc was never created, so it gets no shadow entry at all.
    assert str(fake_home / ".netrc") not in argv


def test_agent_gateway_bwrap_shadows_ssh_auth_sock_only_when_it_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, available_bwrap: None
) -> None:
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    sock = tmp_path / "agent.sock"
    sock.write_text("socket-sentinel")

    monkeypatch.delenv("SSH_AUTH_SOCK", raising=False)
    (tmp_path / "without-sock").mkdir()
    argv_without, _kwargs = _sandboxed_argv(
        tmp_path / "without-sock", role="implementer"
    )
    assert str(sock) not in argv_without

    monkeypatch.setenv("SSH_AUTH_SOCK", str(sock))
    (tmp_path / "with-sock").mkdir()
    argv_with, _kwargs = _sandboxed_argv(tmp_path / "with-sock", role="implementer")
    assert ("/dev/null", str(sock)) in _bind_pairs(argv_with, "--ro-bind")


def test_agent_gateway_bwrap_argv_skips_backend_home_rebind_for_unknown_backend(
    tmp_path: Path, available_bwrap: None
) -> None:
    argv, _kwargs = _sandboxed_argv(
        tmp_path, role="implementer", backend="carrier-pigeon"
    )
    tmpfs_indices = [index for index, value in enumerate(argv) if value == "--tmpfs"]
    ephemeral_home = argv[tmpfs_indices[-1] + 1]
    assert not any(
        argv[index] == "--ro-bind" and argv[index + 2].startswith(ephemeral_home + "/")
        for index in range(len(argv) - 2)
    )


def test_require_bwrap_self_test_launch_failure_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gdw.shutil, "which", lambda _name: "/usr/bin/bwrap")

    def raising_run(*_args, **_kwargs):
        raise OSError("bwrap vanished")

    monkeypatch.setattr(gdw.subprocess, "run", raising_run)
    with pytest.raises(gdw.WorkflowError, match="self-test failed to run"):
        gdw._require_bwrap()


def test_agent_gateway_requires_bwrap_on_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs):
        calls.append(args)
        return _completed(args, stdout="[s]\n{}\n")

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


def test_agent_gateway_requires_bwrap_on_path_names_user_namespace_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gdw.shutil, "which", lambda _name: None)
    with pytest.raises(gdw.WorkflowError, match=r"(?i)user namespace"):
        gdw.AgentGateway(
            roles={},
            issue=1,
            state_file=tmp_path / "state" / "agents.json",
            example_root=Path(gdw.__file__).parent,
            workdir=tmp_path / "worktree",
        )


def test_agent_gateway_requires_bwrap_self_test_to_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs):
        calls.append(args)
        return _completed(args, stdout="[s]\n{}\n")

    monkeypatch.setattr(gdw.shutil, "which", lambda _name: "/usr/bin/bwrap")
    monkeypatch.setattr(
        gdw.subprocess,
        "run",
        lambda *_a, **_k: _completed([], returncode=1, stderr="no userns"),
    )
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


def test_configure_logging_writes_one_stderr_handler_at_the_requested_level(
    capsys: pytest.CaptureFixture,
) -> None:
    errors.configure_logging()
    assert errors.LOGGER.level == logging.INFO
    assert len(errors.LOGGER.handlers) == 1

    errors.configure_logging(verbose=True)
    assert errors.LOGGER.level == logging.DEBUG
    assert len(errors.LOGGER.handlers) == 1

    errors.LOGGER.debug("hello from the workflow")
    captured = capsys.readouterr()
    assert "hello from the workflow" in captured.err
    assert captured.out == ""
