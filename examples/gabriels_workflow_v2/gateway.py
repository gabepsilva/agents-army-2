"""Structured agent turns, each one wrapped in its own `bwrap` sandbox.

`AgentGateway` owns the single `orchestrator talk` call a stage makes, and is
the only place in this driver that launches a subprocess into which an agent
can see. `docs/security.md` documents the isolation the argv below builds.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from examples.gabriels_workflow_v2.errors import LOGGER, WorkflowError

GITHUB_TOKEN_NAMES = (
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "GH_ENTERPRISE_TOKEN",
    "GITHUB_ENTERPRISE_TOKEN",
    "GITHUB_APP_ID",
    "GITHUB_APP_PRIVATE_KEY",
)

# `{{EXPANSION_JSON}}` in a prompt template. Deliberately narrow: a prompt
# names its placeholders in upper case, so braces around anything else are
# text — the kind of text an agent writes when it quotes code back.
PLACEHOLDER_PATTERN = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")

DEFAULT_AGENT_TIMEOUT = 3_600

# Only these roles change files; every other role reviews or gates and gets a
# read-only worktree inside the sandbox so it cannot corrupt the tree.
WRITABLE_ROLES = frozenset({"implementer", "documenter"})

# Best-effort only: each backend keeps its own login/session state under a
# dotfile in the real $HOME. Re-binding it into the ephemeral sandbox HOME
# lets a backend behave the way it does outside the sandbox. A wrong or
# missing entry just loses that convenience for one backend — the base
# `--ro-bind / /` already leaves the real path readable, so turn correctness
# never depends on this mapping being exact.
# Every directory a backend CLI reads its login from or writes its session
# to. More than one each, because these tools follow XDG: opencode keeps
# config in `~/.config`, conversations in `~/.local/share`, and locks in
# `~/.local/state`. Overlaying only the first left "Session not found" on the
# second turn of every opencode agent. Paths are resolved under the real
# `$HOME`; a host that relocates them with `XDG_*` is not supported, because
# `--clearenv` means the sandboxed CLI looks under `$HOME` regardless.
BACKEND_HOME_DIRS = {
    "claude": (".claude",),
    "codex": (".codex", ".local/state/codex"),
    "grok": (".grok",),
    "opencode": (
        ".config/opencode",
        ".local/share/opencode",
        ".local/state/opencode",
        ".cache/opencode",
    ),
}

# Single-file configs that sit beside, not inside, the directories above.
# `bwrap` overlays a directory, never a file, so these are copied once into
# the issue's own layer and bound from there.
BACKEND_HOME_FILES = {
    "claude": (".claude.json",),
}

# An agent name becomes a directory name under the state directory, and it
# reaches here from configuration, so it is checked rather than trusted.
AGENT_NAME_CHARACTERS = frozenset(
    "-_.abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)

# Shadowed with `--ro-bind /dev/null <path>` when present on the host, so an
# agent turn cannot read them regardless of which env var points at them.
SENSITIVE_HOME_RELATIVE_PATHS = (
    ".ssh",
    ".aws",
    ".config/gcloud",
    ".azure",
    ".netrc",
    ".docker",
    ".config/gh",
)

# `--proc`/`--dev` replace the base `--ro-bind / /`'s view of those two paths
# with the sandbox's own namespace, so this alone proves user namespaces and
# bwrap both work before any real turn runs.
BWRAP_SELF_TEST_ARGS = ("--ro-bind", "/", "/", "--proc", "/proc", "--dev", "/dev")

BWRAP_SELF_TEST_TIMEOUT = 10


class RoleOptions(Protocol):
    @property
    def backend(self) -> str: ...

    @property
    def model(self) -> str | None: ...

    @property
    def reasoning_effort(self) -> str | None: ...


def _layer_name(relative: str) -> str:
    """A flat, filesystem-safe directory name for one overlaid config path."""

    return relative.replace("/", "-")


def _within(path: Path, other: Path) -> bool:
    """Whether `path` is `other` or lives under it. Both must be resolved."""

    return path == other or other in path.parents


@dataclass(frozen=True)
class SandboxContext:
    """Everything `_build_bwrap_argv` needs beyond the payload itself."""

    role: str
    backend: str
    worktree: Path
    state_dir: Path
    agent_home: Path
    schema_path: Path
    environment: Mapping[str, str]
    ephemeral_home: Path
    isolation_dir: Path
    real_home: Path


_BWRAP_UNSHARE_FLAGS = (
    "--unshare-pid",
    "--unshare-uts",
    "--unshare-ipc",
    "--unshare-cgroup-try",
    "--unshare-user",
    "--die-with-parent",
    "--new-session",
)


def _setenv_flags(context: SandboxContext) -> list[str]:
    setenv = {
        "PATH": context.environment.get("PATH", ""),
        "HOME": str(context.ephemeral_home),
        "AGENTS_ARMY_HOME": str(context.worktree),
        "AGENTS_ARMY_STATE_FILE": context.environment.get("AGENTS_ARMY_STATE_FILE", ""),
        "GH_CONFIG_DIR": context.environment.get("GH_CONFIG_DIR", ""),
    }
    for name in ("LANG", "LC_ALL", "TZ", "TERM"):
        value = context.environment.get(name)
        if value is not None:
            setenv[name] = value
    flags: list[str] = []
    for name, value in setenv.items():
        flags += ["--setenv", name, value]
    return flags


def _sensitive_shadow_flags(context: SandboxContext) -> list[str]:
    shadow_paths = [
        context.real_home / relative for relative in SENSITIVE_HOME_RELATIVE_PATHS
    ]
    auth_sock = context.environment.get("SSH_AUTH_SOCK", "")
    if auth_sock:
        shadow_paths.append(Path(auth_sock))
    flags: list[str] = []
    for path in shadow_paths:
        if path.is_dir():
            flags += ["--tmpfs", str(path)]
        elif path.exists():
            flags += ["--ro-bind", "/dev/null", str(path)]
    return flags


def _private_tmpfs_flags() -> list[str]:
    flags: list[str] = []
    # bandit(B108) flags these as hardcoded tmp paths, but they are `bwrap`
    # mount-point arguments naming paths inside the sandbox's own new mount
    # namespace, not paths this process opens or writes to itself: `bwrap`
    # requires the real, well-known absolute paths here, and this process
    # never touches the file at that path, so there is no predictable-tmp-
    # file race for it to fall into.
    for private in ("/tmp", "/var/tmp", "/dev/shm"):  # nosec B108
        flags += ["--tmpfs", private]
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    flags += ["--tmpfs", runtime_dir]
    return flags


def _ephemeral_home_flags(context: SandboxContext) -> list[str]:
    """The per-turn `$HOME`, with the backend's config overlaid into it.

    The backend's real config directory is the overlay's lower layer, so a
    turn reads the login and settings it would outside the sandbox but cannot
    write them. Writes land in the issue's own upper layer instead, which is
    what makes a session resumable: the CLIs record a conversation under their
    config directory, and a turn that could only write a `tmpfs` destroyed at
    exit would leave the next turn's `--resume` with no conversation to find.
    """

    # Re-bound after `_private_tmpfs_flags`'s `--tmpfs /tmp`, since `bwrap`
    # resolves a bind's source against the host filesystem at the time each
    # flag runs, not against the sandbox's own already-`tmpfs`'d view.
    flags = ["--ro-bind", str(context.isolation_dir), str(context.isolation_dir)]
    flags += ["--tmpfs", str(context.ephemeral_home)]
    for relative in BACKEND_HOME_DIRS.get(context.backend, ()):
        source = context.real_home / relative
        if not source.exists():
            continue
        layer = context.agent_home / _layer_name(relative)
        flags += [
            "--overlay-src",
            str(source),
            "--overlay",
            str(layer / "upper"),
            str(layer / "work"),
            str(context.ephemeral_home / relative),
        ]
    for name in BACKEND_HOME_FILES.get(context.backend, ()):
        carried = context.agent_home / "files" / name
        if carried.exists():
            flags += ["--bind", str(carried), str(context.ephemeral_home / name)]
    return flags


def _bind_flags(context: SandboxContext) -> list[str]:
    bind_flag = "--bind" if context.role in WRITABLE_ROLES else "--ro-bind"
    flags = [bind_flag, str(context.worktree), str(context.worktree)]
    # The whole directory, not the state file alone: the orchestrator persists
    # through a sibling `.tmp` and a rename, and takes sibling lock files, none
    # of which a per-file bind can host. It holds nothing but agent session
    # state, so nothing else becomes writable by giving it up.
    flags += ["--bind", str(context.state_dir), str(context.state_dir)]
    flags += ["--ro-bind", str(context.schema_path), str(context.schema_path)]
    return flags


def _build_bwrap_argv(
    bwrap_path: str, context: SandboxContext, orchestrator_args: Sequence[str]
) -> list[str]:
    """The exact, ordered `bwrap` argv wrapping one `orchestrator talk` call.

    Order is the isolation contract: a later mount wins over an earlier one
    at the same path, so anything meant to survive has to come after
    whatever it must not be shadowed by.
    """

    argv = [bwrap_path, *_BWRAP_UNSHARE_FLAGS, "--clearenv"]
    argv += _setenv_flags(context)
    argv += ["--ro-bind", "/", "/"]
    argv += ["--proc", "/proc", "--dev", "/dev"]
    argv += _sensitive_shadow_flags(context)
    argv += _private_tmpfs_flags()
    argv += _ephemeral_home_flags(context)
    argv += _bind_flags(context)
    argv += ["--", *orchestrator_args]
    return argv


@dataclass(frozen=True)
class TurnSandbox:
    """The per-turn identity `_run_cli` needs to build a `SandboxContext`."""

    role: str
    backend: str
    schema_path: Path
    agent_name: str
    ephemeral_home: Path
    isolation_dir: Path


def _require_bwrap() -> str:
    """Fail closed: an agent turn never runs unsandboxed.

    Checked once, not swallowed into `_run_cli`'s existing failure handling,
    so a missing sandbox is reported as what it is instead of surfacing as a
    generic "orchestrator CLI failed" from the first real turn.
    """

    bwrap_path = shutil.which("bwrap")
    if bwrap_path is None:
        raise WorkflowError(
            "bubblewrap ('bwrap') is not installed or not on PATH. Agent "
            "turns run inside a bwrap sandbox and refuse to start without "
            "it — install the 'bubblewrap' package and enable unprivileged "
            "user namespaces (e.g. `sysctl kernel.unprivileged_userns_clone=1`)."
        )
    try:
        probe = subprocess.run(
            [bwrap_path, *BWRAP_SELF_TEST_ARGS, "--", "true"],
            capture_output=True,
            text=True,
            timeout=BWRAP_SELF_TEST_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise WorkflowError(
            f"bubblewrap ('bwrap') self-test failed to run: {exc}"
        ) from exc
    if probe.returncode != 0:
        raise WorkflowError(
            "bubblewrap ('bwrap') self-test failed "
            f"(exit {probe.returncode}): {probe.stderr.strip()}. Agent turns "
            "run inside a bwrap sandbox, which needs unprivileged user "
            "namespaces enabled (e.g. `sysctl kernel.unprivileged_userns_clone=1`)."
        )
    return bwrap_path


class AgentGateway:
    """Structured turns through the public orchestrator CLI."""

    def __init__(
        self,
        *,
        roles: Mapping[str, RoleOptions],
        issue: int,
        state_file: Path,
        example_root: Path,
        workdir: Path,
        run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self.roles = dict(roles)
        self.issue = issue
        self.state_file = state_file
        # Required rather than defaulted to the process's directory. That
        # default is what sent every agent into the driver's own checkout
        # while git and CI worked in the issue's worktree: the implementer
        # edited one tree, `make ci` passed against the other because nothing
        # in it had changed, and the run died at `commit` with "produced no
        # commits". A caller that forgets this argument now fails to build.
        self.workdir = workdir
        # The state directory is bound read-write for every role, including
        # the ones whose worktree bind is read-only. If the two overlapped,
        # that later rw bind would silently hand a reviewer a writable tree.
        state_dir = state_file.parent.resolve()
        tree = workdir.resolve()
        if _within(state_dir, tree) or _within(tree, state_dir):
            raise WorkflowError(
                f"agent state directory {state_dir} overlaps the worktree "
                f"{tree}; they must be separate"
            )
        self.prompts = example_root / "prompts"
        self.validations = example_root / "validations"
        self._run_process = run
        # Once per gateway, not per turn: a fresh `bwrap` process for every
        # turn already pays this cost, so a broken sandbox is caught before
        # the first agent talks rather than after the driver has already
        # logged progress against a turn that never ran isolated.
        self._bwrap_path = _require_bwrap()

    def options(self, role: str) -> RoleOptions:
        role_options = self.roles.get(role)
        if role_options is None:
            raise WorkflowError(f"agent role '{role}' is not configured")
        return role_options

    def ask(
        self,
        *,
        role: str,
        prompt_name: str,
        schema_name: str,
        values: Mapping[str, str],
        skills: Sequence[str] = (),
        timeout: int = DEFAULT_AGENT_TIMEOUT,
    ) -> dict:
        prompt = self._prompt(prompt_name, values)
        schema = self.validations / f"{schema_name}.json"
        role_options = self.options(role)
        backend = role_options.backend
        model = role_options.model
        reasoning_effort = role_options.reasoning_effort
        agent_name = f"gdw-{self.issue}-{role}"
        LOGGER.info(
            "agent %s: using '%s' (backend=%s model=%s effort=%s skills=%s)",
            role,
            agent_name,
            backend,
            model,
            reasoning_effort,
            ",".join(skills) or "-",
        )
        turn_args = ["talk", agent_name, "--backend", backend]
        if model is not None:
            turn_args += ["--model", model]
        if reasoning_effort is not None:
            turn_args += ["--reasoning-effort", reasoning_effort]
        if skills:
            turn_args += ["--skill", ",".join(skills)]
        LOGGER.info(
            "agent %s: turn starting (prompt=%s '%s' of %s chars, schema=%s, "
            "timeout=%ss)",
            role,
            prompt_name,
            schema_name,
            len(prompt),
            schema.name,
            timeout,
        )
        LOGGER.debug("agent %s: prompt sent\n%s", role, prompt)
        started = time.monotonic()
        with self._sandbox_workspace() as (environment, ephemeral_home, isolation):
            turn_args += [
                "--schema",
                str(schema),
                "--timeout",
                str(timeout),
                "--prompt",
                prompt,
            ]
            sandbox_turn = TurnSandbox(
                role=role,
                backend=backend,
                schema_path=schema,
                agent_name=agent_name,
                ephemeral_home=ephemeral_home,
                isolation_dir=isolation,
            )
            turn = self._run_cli(
                turn_args, environment, sandbox_turn, timeout=timeout + 5
            )
        LOGGER.info(
            "agent %s: turn finished in %.1fs with %s chars of stdout",
            role,
            time.monotonic() - started,
            len(turn.stdout),
        )
        LOGGER.debug("agent %s: stdout\n%s", role, turn.stdout)
        _header, separator, payload = turn.stdout.partition("\n")
        if not separator:
            raise WorkflowError(f"agent '{role}' returned no structured response")
        try:
            structured = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise WorkflowError(
                f"agent '{role}' returned invalid structured JSON: {exc}"
            ) from exc
        if not isinstance(structured, dict):
            raise WorkflowError(f"agent '{role}' returned no structured response")
        return structured

    def _run_cli(
        self,
        args: list[str],
        environment: Mapping[str, str],
        turn: TurnSandbox,
        *,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        state_dir = self.state_file.parent
        state_dir.mkdir(parents=True, exist_ok=True)
        agent_home = self._agent_home(turn.agent_name, turn.backend)
        context = SandboxContext(
            role=turn.role,
            backend=turn.backend,
            worktree=self.workdir,
            state_dir=state_dir,
            agent_home=agent_home,
            schema_path=turn.schema_path,
            environment=environment,
            ephemeral_home=turn.ephemeral_home,
            isolation_dir=turn.isolation_dir,
            real_home=Path.home(),
        )
        argv = _build_bwrap_argv(self._bwrap_path, context, ["orchestrator", *args])
        LOGGER.debug("orchestrator: invoking '%s' (timeout %ss)", args[0], timeout)
        try:
            result = self._run_process(
                argv,
                cwd=str(self.workdir),
                env=environment,
                capture_output=True,
                text=True,
                check=False,
                stdin=subprocess.DEVNULL,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            LOGGER.error("orchestrator: '%s' failed to run: %s", args[0], exc)
            raise WorkflowError(f"orchestrator CLI failed: {exc}") from exc
        if result.returncode != 0:
            message = result.stderr.strip() or f"exited {result.returncode}"
            LOGGER.error("orchestrator: '%s' failed: %s", args[0], message)
            raise WorkflowError(f"orchestrator CLI failed: {message}")
        return result

    def _agent_home(self, agent_name: str, backend: str) -> Path:
        """This agent's writable layer over its backend's config directory.

        Per agent, not per backend, for two reasons. A session belongs to one
        agent, so that is the granularity the layer that carries it should
        have. And `overlayfs` refuses a second mount sharing a live upperdir
        (EBUSY) — some backend CLIs leave a server running past the turn that
        started it, so two agents on one backend sharing a layer means the
        second one's turn fails to start.

        `upper` and `work` must be on one filesystem for `overlayfs`; `files`
        carries the single-file configs, copied once so the real ones are
        never written.
        """

        if not agent_name or set(agent_name) - AGENT_NAME_CHARACTERS:
            raise WorkflowError(f"unusable agent name {agent_name!r}")
        home = self.state_file.parent / "home" / agent_name
        (home / "files").mkdir(parents=True, exist_ok=True)
        for relative in BACKEND_HOME_DIRS.get(backend, ()):
            layer = home / _layer_name(relative)
            for name in ("upper", "work"):
                (layer / name).mkdir(parents=True, exist_ok=True)
        for name in BACKEND_HOME_FILES.get(backend, ()):
            carried = home / "files" / name
            source = Path.home() / name
            if not carried.exists() and source.is_file():
                carried.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, carried)
        return home

    def _prompt(self, name: str, values: Mapping[str, str]) -> str:
        """Fill one prompt template, judging completeness by the template alone.

        Agent replies quote code, and code contains braces — a dict
        comprehension closing with `}}` is not an unfilled placeholder. So the
        placeholders are found in the template before anything is substituted,
        and each one is replaced in a single pass, which also stops a value
        that happens to spell `{{OTHER}}` from being expanded in turn.
        """

        path = self.prompts / f"{name}.md"
        try:
            template = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise WorkflowError(f"cannot read prompt {path}: {exc}") from exc
        unresolved = sorted(
            {
                match.group(1)
                for match in PLACEHOLDER_PATTERN.finditer(template)
                if match.group(1) not in values
            }
        )
        if unresolved:
            raise WorkflowError(
                f"prompt {path} has unresolved placeholders: {', '.join(unresolved)}"
            )
        return PLACEHOLDER_PATTERN.sub(lambda match: values[match.group(1)], template)

    @contextmanager
    def _sandbox_workspace(self) -> Iterator[tuple[dict[str, str], Path, Path]]:
        """The env, ephemeral `$HOME`, and isolation dir for one sandboxed turn.

        `isolation` holds the `gh` blocker and empty `GH_CONFIG_DIR` and, like
        `ephemeral_home`, lives under the system temp directory rather than
        under the real `$HOME` — `_build_bwrap_argv` re-binds both by their
        real (host) path after `--tmpfs /tmp`, which only works because
        `bwrap` resolves a bind's source against the host filesystem at the
        time that flag runs, not against the sandbox's own, already-`tmpfs`'d
        view of `/tmp`.
        """

        original = dict(os.environ)
        with tempfile.TemporaryDirectory(prefix="gdw-agent-") as directory:
            isolation = Path(directory)
            blocker = isolation / "gh"
            blocker.write_text(
                "#!/bin/sh\necho 'gh is owned by the GDW driver' >&2\nexit 126\n",
                encoding="utf-8",
            )
            blocker.chmod(0o700)
            ephemeral_home = isolation / "home"
            ephemeral_home.mkdir()
            empty_gh_config = isolation / "empty-gh-config"
            empty_gh_config.mkdir()
            sanitized = dict(original)
            sanitized["PATH"] = f"{isolation}{os.pathsep}{original.get('PATH', '')}"
            sanitized["GH_CONFIG_DIR"] = str(empty_gh_config)
            # cwd already points here, but the orchestrator only derives its
            # working directory from cwd as a fallback. Saying it outright
            # keeps the agents in the worktree even if that call changes.
            sanitized["AGENTS_ARMY_HOME"] = str(self.workdir)
            sanitized["AGENTS_ARMY_STATE_FILE"] = str(self.state_file)
            for name in GITHUB_TOKEN_NAMES:
                sanitized.pop(name, None)
            yield sanitized, ephemeral_home, isolation
