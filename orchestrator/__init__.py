#!/usr/bin/env python3
"""Long-lived orchestrator holding an array of agents.

Each agent owns a persistent Claude Code, Codex, Grok, or OpenCode CLI session. Every
time you talk to an agent it resumes that session with your prompt and returns
the reply, so each agent keeps its own conversation history across messages.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
from collections.abc import Callable, Iterable, Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext
from pathlib import Path
from typing import Any, NoReturn, TextIO, cast

from backends import AgentBackend, TurnError, TurnResult, get_backend, list_backends
from backends.base import DEFAULT_TURN_TIMEOUT, OutputSchema
from backends.registry import UnknownBackendError
from orchestrator import teams
from orchestrator.schema import (
    ReplyValidationError,
    SchemaLoadError,
    compose_schema_prompt,
    load_schema,
    repair_prompt,
    validate_reply,
)
from orchestrator.skills import (
    SkillError,
    compose_skill_prompt,
    format_skill_listing,
    index_skills,
    parse_skill_names,
    resolve_skills,
)

# The one folder agents-army owns in $HOME: default home of the teamless
# registry (see STATE_FILE below) and the root `list teams` walks to find
# every team. Unlike AGENTS_ARMY_HOME, this never becomes an agent's cwd.
ROOT = Path(os.environ.get("AGENTS_ARMY_ROOT", Path.home() / ".agents-army"))
# The backend working directory and skills root default to the caller's cwd
# (override with AGENTS_ARMY_HOME) rather than next to the installed package,
# so they don't leak into the venv. STATE_FILE's own default no longer
# follows HOME — see the ladder below.
HOME = Path(os.environ.get("AGENTS_ARMY_HOME", Path.cwd()))
# STATE_FILE's default ladder: an explicit path wins outright; failing that,
# an explicitly-set AGENTS_ARMY_HOME (not "HOME happens to equal cwd", which
# is the unset case) relocates it alongside WORKDIR/SKILLS_DIR; otherwise it
# defaults under ROOT rather than cwd, so a plain `orchestrator create` run
# from any checkout writes one registry instead of scattering one per repo.
if "AGENTS_ARMY_STATE_FILE" in os.environ:
    STATE_FILE = Path(os.environ["AGENTS_ARMY_STATE_FILE"])
elif "AGENTS_ARMY_HOME" in os.environ:
    STATE_FILE = HOME / "orchestrator_state.json"
else:
    STATE_FILE = ROOT / "orchestrator_state.json"
# Agents run their CLI sessions from a single shared working directory.
WORKDIR = HOME
# Skill markdown catalog. Override with AGENTS_ARMY_SKILLS; default is $HOME/SKILLS.
SKILLS_DIR = Path(os.environ.get("AGENTS_ARMY_SKILLS", HOME / "SKILLS"))
# Team roots: $TEAMS_DIR/<team>/{agents/,worktree/} — state and workspace as
# siblings, never nested. No default: see README.
TEAMS_DIR = (
    Path(os.environ["AGENTS_ARMY_TEAMS_DIR"])
    if "AGENTS_ARMY_TEAMS_DIR" in os.environ
    else None
)
# The backend an agent gets when none is named: by `create`, and by the agent a
# talk creates for a name that does not exist yet.
DEFAULT_BACKEND = "claude"

# How many extra turns a reply that misses the schema is worth. Two, because
# the measured conformance rate makes even the first retry nearly dead code:
# this is the fallback, not the mechanism that gets a conforming reply.
DEFAULT_VALIDATION_RETRIES = 2

# Named explicitly rather than via __name__: this module runs both as a script
# (__main__) and as the `orchestrator` console script, and _configure_logging
# raises the level by logger name.
log = logging.getLogger("orchestrator")

# Full prompts and replies are unbounded and are the only logs that can carry
# the content of a conversation, so they sit below DEBUG: -v stays readable and
# safe to paste, and -vv is the deliberate opt-in to the whole transcript.
TRACE = logging.DEBUG - 5
logging.addLevelName(TRACE, "TRACE")


class OrchestratorError(Exception):
    """A failure the user can act on: one line on stderr, exit 1, no traceback.

    Named types rather than bare KeyError/ValueError so the CLI can catch
    exactly these. Catching the builtins around the whole dispatch swallowed
    any incidental one raised inside a backend too, and printed it as a bare
    one-word line with no traceback — turning a real bug into a mystery.
    """


class AgentNotFoundError(OrchestratorError, KeyError):
    """No agent by that name. Still a KeyError: that is what callers catch."""


class AgentExistsError(OrchestratorError, ValueError):
    """Spawn was asked for a name that is already taken."""


class StateError(OrchestratorError):
    """The state file exists but does not hold the structure this code needs."""


class TeamBusyError(OrchestratorError):
    """Teardown asked for a team another command is still holding."""


# User-facing failures that must print one line and exit 1, never a traceback.
# UnknownBackendError comes from the registry, which cannot import this module.
_CLI_ERRORS = (OrchestratorError, UnknownBackendError)


class Agent:
    """A single named agent backed by one persistent CLI session."""

    def __init__(self, name: str, backend: AgentBackend) -> None:
        self.name = name
        self.backend = backend
        self.session_id: str | None = None

    def talk(
        self,
        prompt: str,
        schema: OutputSchema | None = None,
        timeout: int = DEFAULT_TURN_TIMEOUT,
    ) -> TurnResult:
        log.info(
            "agent '%s' (%s): starting turn, resume=%s",
            self.name,
            self.backend.name,
            bool(self.session_id),
        )
        # Logged here rather than per backend: the turn is the same exchange
        # whichever CLI runs it, so every backend gets this for free.
        log.log(TRACE, "agent '%s' prompt in:\n%s", self.name, prompt)
        started = time.monotonic()
        result = self.backend.run_turn(
            prompt, self.session_id, WORKDIR, timeout, schema
        )
        elapsed = time.monotonic() - started
        log.info("agent '%s': turn finished in %.1fs", self.name, elapsed)
        log.log(TRACE, "agent '%s' reply out:\n%s", self.name, result.reply)
        if result.structured is not None:
            log.log(
                TRACE,
                "agent '%s' structured out:\n%s",
                self.name,
                json.dumps(result.structured, indent=2, sort_keys=True),
            )
        # A backend that reports no session id has not ended the conversation,
        # it has failed to name it. Keeping the previous id lets the next turn
        # resume the session instead of silently starting a fresh one.
        if result.session_id is not None:
            self.session_id = result.session_id
        return result


def _is_live(path: Path, lock: TextIO) -> bool:
    """Is the inode we locked still the one `path` names?

    Checked on `st_ino`, not `st_nlink`: `st_nlink == 0` is true only if the
    reclaimer unlinked, but unlinking is not the only way a path stops naming
    an inode — over NFS, unlinking a file still open elsewhere is emulated by
    renaming it aside, so the link count stays 1 while the path no longer
    names it. `st_ino` catches rename and unlink alike.
    """
    try:
        return os.stat(path).st_ino == os.fstat(lock.fileno()).st_ino
    except FileNotFoundError:
        return False


# A revalidation loop only spins when something else is unlinking this exact
# path faster than this call can observe it as live — real contention settles
# in one or two iterations. A bound turns a bug that breaks that (a corrupted
# _is_live, an unlink with no matching hold) into a clear error instead of a
# silent, unkillable hot loop inside a lock acquisition.
_MAX_REVALIDATE_ATTEMPTS = 100


@contextmanager
def _flock(
    path: Path, mode: int = fcntl.LOCK_EX, *, revalidate: bool = False
) -> Iterator[TextIO]:
    """Hold an flock on `path`, creating its parent directory first.

    Module-level rather than a method on `Orchestrator` so `main()` can take
    the team lock before an `Orchestrator` exists: `Orchestrator._locked`
    delegates here too, so there is exactly one flock implementation.

    `revalidate` guards against a lock file that gets unlinked (or renamed)
    out from under a waiter: `flock` binds to the inode, not the path, so a
    process that was blocked in `flock()` on a now-reclaimed inode wakes up
    holding a lock that guards nothing. When set, every acquisition checks
    that the inode it just locked is still the one `path` names, and loops —
    unlocking, closing, and reopening the path — until that holds.

    Yields the open lock file so a caller that wants to unlink `path` while
    holding this flock can check `_is_live` first, itself: opening and then
    flocking are two syscalls, not one, so a caller that unlinks blindly
    after `_flock` merely returns can still be acting on an inode a
    different, legitimate reclaimer already orphaned in between. See
    `Orchestrator._reclaim_agent_lock`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    attempts = 0
    while True:
        with path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), mode)
            try:
                if revalidate and not _is_live(path, lock):
                    attempts += 1
                    if attempts >= _MAX_REVALIDATE_ATTEMPTS:
                        raise RuntimeError(
                            f"lock {path}: still not live after "
                            f"{_MAX_REVALIDATE_ATTEMPTS} reacquire attempts"
                        )
                    log.debug("lock %s: reclaimed underneath us, re-acquiring", path)
                    continue
                yield lock
                return
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _load_state_file(state_file: Path) -> dict[str, dict]:
    """Raw agent entries from a registry file, `{}` if it doesn't exist.

    Module-level, not just `Orchestrator._load_state`, so `list teams` can
    read a discovered team's registry the same way without going through
    `get_backend`: enumeration only needs the backend *name* already stored
    in each entry, and a renamed or removed backend plugin in one team's
    registry must not stop every other team from listing.
    """
    if not state_file.exists():
        return {}
    try:
        return json.loads(state_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StateError(f"{state_file} is not valid JSON: {exc}") from exc


class Orchestrator:
    """Registry of named agents, each with an independent CLI session.

    Agents persist in `orchestrator_state.json` so any process can spawn an
    agent once and talk to it later, resuming the same CLI session.
    """

    def __init__(self, state_file: Path | None = None) -> None:
        self.state_file = STATE_FILE if state_file is None else state_file
        self.agents: dict[str, Agent] = {}
        self._reload()
        log.debug(
            "state: loaded %d agent(s) from %s", len(self.agents), self.state_file
        )

    def spawn(
        self,
        name: str,
        backend: str | None = None,
        *,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> Agent:
        with self._exclusive():
            self._reload()
            if name in self.agents:
                raise AgentExistsError(f"agent '{name}' already exists")
            return self._create(name, backend, model, reasoning_effort)

    def ensure(
        self,
        name: str,
        backend: str | None = None,
        *,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> tuple[Agent, bool]:
        """Return the named agent, creating it first if it does not exist.

        Reports whether it had to create one, so a caller can say so. The
        lookup and the create share one lock rather than being a `spawn` after
        a failed `talk`: two processes naming the same new agent at once then
        get one agent between them, not a spawn that loses to a duplicate.
        """
        with self._exclusive():
            self._reload()
            existing = self.agents.get(name)
            if existing is not None:
                return existing, False
            return self._create(name, backend, model, reasoning_effort), True

    def _create(
        self,
        name: str,
        backend: str | None,
        model: str | None,
        reasoning_effort: str | None,
    ) -> Agent:
        """Register and persist a new agent. The caller holds `_exclusive()`.

        `None` means "whatever the default backend is now": resolving it here
        rather than in a default argument keeps DEFAULT_BACKEND a live lookup.
        """
        agent = Agent(
            name,
            get_backend(
                DEFAULT_BACKEND if backend is None else backend,
                model=model,
                reasoning_effort=reasoning_effort,
            ),
        )
        self.agents[name] = agent
        self._persist()
        return agent

    def talk(
        self,
        name: str,
        prompt: str,
        schema: OutputSchema | None = None,
        retries: int = DEFAULT_VALIDATION_RETRIES,
        timeout: int = DEFAULT_TURN_TIMEOUT,
    ) -> TurnResult:
        """Run one turn against `name`, or, with a schema, as many as it takes.

        The whole thing happens under one agent lock: a retry has to land on
        the same session as the attempt it is correcting, and another process
        talking to this agent in between would fork the conversation.
        """
        path = self._agent_lock_path(name)
        with self._agent_lock(name) as lock:
            try:
                with self._exclusive():
                    self._reload()
                    agent = self.agents.get(name)
                    if agent is None:
                        raise AgentNotFoundError(f"no agent named '{name}'")
                if schema is None:
                    return self._turn(agent, prompt, None, timeout)
                return self._validated_turn(agent, prompt, schema, retries, timeout)
            except AgentNotFoundError:
                # We hold this lock and the agent provably does not exist, so
                # no turn can be running behind it — `_is_live` is always
                # true here (this lock has been held continuously since a
                # validated acquire, so nothing could have replaced the
                # path), but checking keeps that a local fact instead of one
                # a reader has to reconstruct. A waiter queued on this inode
                # revalidates and re-acquires — see _flock's revalidate loop.
                if _is_live(path, lock):
                    path.unlink(missing_ok=True)
                    log.debug("agent '%s': reclaimed lock file, no such agent", name)
                raise

    def _turn(
        self,
        agent: Agent,
        prompt: str,
        schema: OutputSchema | None,
        timeout: int,
    ) -> TurnResult:
        """One turn, with its session id persisted before anything else runs.

        Persisting per attempt rather than per call is what lets a run that
        exhausts its retries still leave the session where the agent actually
        is: it moved the conversation forward whether or not the last reply
        was usable, and resuming from a stale id would replay it.
        """
        result = agent.talk(prompt, schema, timeout)
        with self._exclusive():
            self._reload()
            if agent.name not in self.agents:
                raise AgentNotFoundError(f"no agent named '{agent.name}'")
            # agent.session_id, not result.session_id: the agent keeps the
            # id it already had when a backend reports none.
            self.agents[agent.name].session_id = agent.session_id
            self._persist()
        return result

    def _validated_turn(
        self,
        agent: Agent,
        prompt: str,
        schema: OutputSchema,
        retries: int,
        timeout: int,
    ) -> TurnResult:
        """Talk until the reply satisfies `schema`, the retries run out, or the
        clock does.

        `timeout` is the budget for the whole loop, not for each attempt: a
        validated call must not be able to cost three times what a plain turn
        can, holding this agent's lock for an hour and a half to do it. Each
        attempt gets whatever is left.
        """
        # None on a CLI that was handed the schema itself; the document on one
        # that was not, so the prompt can carry what the flag could not.
        unenforced = None if agent.backend.enforces_schema else schema
        if unenforced is not None:
            log.warning(
                "backend %s: schema is enforced via validation/repair, not the CLI",
                agent.backend.name,
            )
        deadline = time.monotonic() + timeout
        attempt_prompt = compose_schema_prompt(prompt, unenforced)
        attempt = 0
        while True:
            attempt += 1
            remaining = deadline - time.monotonic()
            result = self._turn(
                agent, attempt_prompt, schema, max(1, math.ceil(remaining))
            )
            try:
                result.structured = validate_reply(
                    result.reply, result.structured, schema
                )
            except ReplyValidationError as exc:
                log.warning(
                    "agent '%s': attempt %d did not satisfy the schema: %s",
                    agent.name,
                    attempt,
                    exc,
                )
                if attempt > retries:
                    log.warning(
                        "agent '%s': %d validation retries exhausted",
                        agent.name,
                        retries,
                    )
                    raise
                if deadline - time.monotonic() <= 0:
                    log.warning(
                        "agent '%s': the %ds budget is spent; not retrying",
                        agent.name,
                        timeout,
                    )
                    raise
                attempt_prompt = repair_prompt(exc, unenforced)
            else:
                # `else`, not a fall-through after the except: a `return` at
                # this indentation would hand back the attempt that just
                # failed validation.
                return result

    def list_agents(self) -> list[str]:
        return sorted(self.agents)

    def delete(self, name: str) -> Agent:
        with self._exclusive():
            self._reload()
            agent = self.agents.pop(name, None)
            if agent is None:
                raise AgentNotFoundError(f"no agent named '{name}'")
            self._persist()
        # After _exclusive() releases the state lock, so agent → state lock
        # ordering never inverts: _reclaim_agent_lock takes the agent lock,
        # and talk() takes agent then state.
        self._reclaim_agent_lock(name)
        return agent

    def _reclaim_agent_lock(self, name: str) -> None:
        """Unlink `name`'s lock file if no turn is in flight for it.

        Create-then-probe, not exists-then-unlink: an `exists()` check first
        would race a turn starting between the check and the unlink. Probing
        by opening the path first means deleting an agent that was never
        talked to creates the lock file and immediately removes it again —
        net zero on disk.

        Guarded by `_is_live`: `_flock`'s own open-then-flock is two
        syscalls, not one, so the inode this probe locks can already be
        orphaned by the time the lock is granted — a different, legitimate
        reclaim unlinked and released it in between. Unlinking the *path*
        unconditionally at that point would remove whatever a live acquirer
        has since put there, not the dead inode this probe actually holds;
        `_is_live` is exactly the check that tells the two apart.

        A `BlockingIOError` means a turn on *this* agent is in flight; that is
        not a failure of `delete`, which never refuses and never changes its
        exit code — the file is left for that turn's own `AgentNotFoundError`
        handler in `talk`, or a later `delete`, to reclaim.
        """
        path = self._agent_lock_path(name)
        try:
            with _flock(path, fcntl.LOCK_EX | fcntl.LOCK_NB) as lock:
                if _is_live(path, lock):
                    path.unlink(missing_ok=True)
        except BlockingIOError:
            log.debug("agent '%s': lock file in use, not reclaiming", name)

    def _lock_path(self) -> Path:
        return self.state_file.with_name(self.state_file.name + ".lock")

    def _locks_dir(self) -> Path:
        return self.state_file.with_name(self.state_file.name + ".locks")

    def _agent_lock_path(self, name: str) -> Path:
        # An agent name is free text and would not survive being used as a
        # filename; the digest only has to be stable, not readable.
        digest = hashlib.sha256(name.encode()).hexdigest()
        return self._locks_dir() / digest

    @contextmanager
    def _locked(self, path: Path, *, revalidate: bool = False) -> Iterator[TextIO]:
        with _flock(path, revalidate=revalidate) as lock:
            yield lock

    def _exclusive(self) -> AbstractContextManager[TextIO]:
        """Serialize reads and writes of the state file."""
        return self._locked(self._lock_path())

    def _agent_lock(self, name: str) -> AbstractContextManager[TextIO]:
        """Serialize whole turns for one agent, leaving other agents free.

        The state lock cannot do this: it covers a file write measured in
        milliseconds, while the thing that must not overlap is the turn, which
        runs for minutes. Two processes resuming the same session fork the
        conversation, and whichever persists last drops the other's reply.

        The only caller that revalidates: this lock file is unlinked by
        `_reclaim_agent_lock` and by `talk`'s own `AgentNotFoundError`
        handler, so a waiter here can wake up holding a dead inode. Neither
        the state lock (never unlinked while held) nor the team lock (left
        alone by `_teardown_team`) can lose their file out from under a
        holder, so they stay non-revalidating.
        """
        return self._locked(self._agent_lock_path(name), revalidate=True)

    def _reload(self) -> None:
        self.agents = {}
        for name, entry in self._load_state().items():
            backend = entry.get("backend")
            if backend is None:
                raise StateError(f"{self.state_file}: agent '{name}' has no backend")
            agent = Agent(
                name,
                get_backend(
                    backend,
                    model=entry.get("model"),
                    reasoning_effort=entry.get("reasoning_effort"),
                ),
            )
            agent.session_id = entry.get("session_id")
            self.agents[name] = agent

    def _load_state(self) -> dict[str, dict]:
        return _load_state_file(self.state_file)

    def _persist(self) -> None:
        state = {
            name: {
                "backend": a.backend.name,
                "session_id": a.session_id,
                **({"model": a.backend.model} if a.backend.model is not None else {}),
                **(
                    {"reasoning_effort": a.backend.reasoning_effort}
                    if a.backend.reasoning_effort is not None
                    else {}
                ),
            }
            for name, a in self.agents.items()
        }
        payload = json.dumps(state, indent=2, sort_keys=True) + "\n"
        tmp = self.state_file.with_name(self.state_file.name + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(self.state_file)
        log.debug("state: wrote %d agent(s) to %s", len(state), self.state_file)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_team_option(parser: argparse.ArgumentParser) -> None:
    # Its own helper, deliberately not folded into _add_agent_config_options:
    # that trio feeds _agent_config/_ensure_agent, which *assert* a stored
    # agent's configuration matches the flags. A team is a namespace to
    # select, not configuration to assert. Long flag only — no -t: it would
    # sit next to talk's --timeout, a footgun for a flag scripts pass once.
    parser.add_argument(
        "--team",
        help=(
            "run against team <team>'s {agents,worktree} instead of the "
            "teamless layout; found under $AGENTS_ARMY_TEAMS_DIR if set, "
            "otherwise resolved under $AGENTS_ARMY_ROOT"
        ),
    )


def _add_agent_config_options(parser: argparse.ArgumentParser) -> None:
    # No argparse default: leaving this None lets create() and ensure() resolve
    # DEFAULT_BACKEND, which is what that constant documents itself as. A
    # literal here would pin them to claude however DEFAULT_BACKEND changed.
    parser.add_argument("--backend", "-b", choices=list_backends())
    parser.add_argument("--model", "-m")
    parser.add_argument("--reasoning-effort", "-e")


def _agent_config(agent: Agent) -> tuple[str, str | None, str | None]:
    return agent.backend.name, agent.backend.model, agent.backend.reasoning_effort


def cmd_create(orchestrator: Orchestrator, opts: argparse.Namespace) -> None:
    agent = orchestrator.spawn(
        opts.name,
        opts.backend,
        model=opts.model,
        reasoning_effort=opts.reasoning_effort,
    )
    print(f"created agent '{agent.name}' backend={agent.backend.name}")


def _ensure_agent(
    orchestrator: Orchestrator,
    name: str,
    backend: str | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> None:
    """Create `name` if talking to it would otherwise fail, and say so.

    The notice goes to stderr: stdout carries the reply and is what a pipe
    reads, and an agent having been created is commentary on the turn, not
    part of it.
    """
    agent, created = orchestrator.ensure(
        name,
        backend,
        model=model,
        reasoning_effort=reasoning_effort,
    )
    if created:
        print(
            f"created agent '{agent.name}' backend={agent.backend.name}",
            file=sys.stderr,
        )
        return
    if backend is None and model is None and reasoning_effort is None:
        return
    expected = (
        DEFAULT_BACKEND if backend is None else backend,
        model,
        reasoning_effort,
    )
    actual = _agent_config(agent)
    if actual != expected:
        raise OrchestratorError(
            f"agent '{agent.name}' already uses backend/model/effort {actual!r}; "
            f"configured {expected!r}"
        )


def cmd_talk(orchestrator: Orchestrator, opts: argparse.Namespace) -> None:
    prompt = opts.prompt
    composed = prompt
    if opts.skill is not None:
        try:
            names = parse_skill_names(opts.skill)
            resolved = resolve_skills(names, SKILLS_DIR)
        except SkillError as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(1) from None
        composed = compose_skill_prompt(resolved, prompt)
        log.info(
            "agent '%s': attaching skill(s) %s",
            opts.name,
            ", ".join(name for name, _path in resolved),
        )
    schema = None
    if opts.schema is not None:
        try:
            schema = load_schema(Path(opts.schema))
        except SchemaLoadError as exc:
            # Exit 2, not 1: the schema file is a bad argument, the same class
            # of mistake argparse exits 2 for. A caller can tell "fix your
            # schema" from "the agent failed" without reading the message.
            print(str(exc), file=sys.stderr)
            raise SystemExit(2) from None
        log.info("agent '%s': validating the reply against %s", opts.name, schema.path)
    # After the skills and the schema resolve, so a bad argument exits without
    # having left a new agent behind for a turn that never ran.
    _ensure_agent(
        orchestrator,
        opts.name,
        opts.backend,
        opts.model,
        opts.reasoning_effort,
    )
    try:
        result = orchestrator.talk(
            opts.name,
            composed,
            schema=schema,
            retries=opts.retries,
            timeout=opts.timeout,
        )
    except (TurnError, ReplyValidationError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None
    print(f"[{opts.name} session={result.session_id}]")
    if schema is None:
        print(result.reply)
        return
    # The validated object rather than the reply text: same content, but
    # parsed once here so a caller piping this gets one canonical spelling.
    print(json.dumps(result.structured, indent=2, sort_keys=True))


def _print_agents(orchestrator: Orchestrator) -> None:
    agents = orchestrator.list_agents()
    if not agents:
        print("no agents")
        return
    width = max(20, max(len(n) for n in agents))
    for name in agents:
        agent = orchestrator.agents[name]
        sid = agent.session_id or "-"
        print(f"{name:{width}} backend={agent.backend.name:6} session={sid}")


def cmd_list(orchestrator: Orchestrator, opts: argparse.Namespace) -> None:
    if opts.target == "agents":
        _print_agents(orchestrator)
        return
    try:
        catalog = index_skills(SKILLS_DIR)
    except SkillError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None
    print(format_skill_listing(catalog))


def _agents_from_registry(state_file: Path) -> dict[str, str] | None:
    """Agent name -> backend, read from `state_file`.

    `None` means the registry couldn't be turned into that mapping: bad
    JSON, removed by a concurrent `delete`/teardown between discovery and
    this read, or valid JSON that isn't the `{name: {"backend": ...}}` shape
    a registry is supposed to have (e.g. a top-level list, or an entry that
    isn't itself an object). `list teams` enumerates every team's registry
    in one pass, so one team's bad file must show up as a flag on that team,
    not abort the report for every other one. Distinct from `{}`, a
    registry that read fine and is simply empty.
    """
    try:
        raw = _load_state_file(state_file)
    except (StateError, OSError):
        return None
    # The shape a registry is supposed to have, checked explicitly rather
    # than caught as an AttributeError off `raw.items()`/`entry.get(...)`:
    # a blanket except there would just as happily swallow a genuine future
    # bug in this function as the JSON's actual shape.
    if not isinstance(raw, dict) or not all(
        isinstance(entry, dict) for entry in raw.values()
    ):
        return None
    return {name: entry.get("backend", "?") for name, entry in raw.items()}


def _team_agents(team: teams.Team) -> dict[str, str] | None:
    return _agents_from_registry(teams.marker_path(team.path))


def _format_agents(agents: dict[str, str] | None) -> str:
    """`(N agents: name/backend, ...)` for a read registry, `(registry
    unreadable)` for one that exists but couldn't be read (see
    `_agents_from_registry`) — the caller decides whether `agents=None`
    means unreadable or "there is nothing here to print" (a registry that
    doesn't exist at all is never formatted, only read ones are)."""
    if agents is None:
        return "registry unreadable"
    count = len(agents)
    plural = "agent" if count == 1 else "agents"
    members = ", ".join(f"{n}/{b}" for n, b in sorted(agents.items()))
    return f"{count} {plural}" + (f": {members}" if members else "")


def _format_team_line(team: teams.Team, agents: dict[str, str] | None) -> str:
    line = f"  {team.name}  ({_format_agents(agents)})"
    if not team.has_worktree:
        line += "  [worktree missing]"
    return line


def _print_teams() -> None:
    root_teams = teams.discover(ROOT)
    groups = [(ROOT, root_teams)]
    if TEAMS_DIR is not None:
        # Walked unconditionally, then deduped by path — not skipped
        # whenever TEAMS_DIR overlaps ROOT. Dropping the whole group
        # whenever TEAMS_DIR is an *ancestor* of ROOT hid every team outside
        # ROOT, exactly what this command exists to show. Deduping instead
        # handles same-dir, descendant, and ancestor with one rule: a team
        # already shown under ROOT is simply never repeated under TEAMS_DIR.
        seen = {team.path for team in root_teams}
        extra_teams = [
            team for team in teams.discover(TEAMS_DIR) if team.path not in seen
        ]
        if extra_teams:
            groups.append((TEAMS_DIR, extra_teams))
    # STATE_FILE, not "$ROOT/orchestrator_state.json": the registry `list
    # teams` reports as (teamless) must be the one `list agents`/`talk`
    # actually use, which an explicit AGENTS_ARMY_STATE_FILE or
    # AGENTS_ARMY_HOME relocates away from ROOT (see the STATE_FILE ladder
    # above) — main() never lets --team reach this function, so STATE_FILE
    # here is always the teamless resolution, never a team's. Not
    # `agents/orchestrator_state.json` inside a directory, so `teams.discover`
    # never finds this bare file on its own — checked explicitly. Its
    # existence and its readability are tracked separately: a corrupt
    # registry here must still show up as a flagged line, the same as a
    # corrupt team registry does, rather than being indistinguishable from
    # "there was never a teamless registry at all".
    has_teamless = STATE_FILE.is_file()

    if not any(group_teams for _, group_teams in groups) and not has_teamless:
        print("no teams")
        return

    for group_root, group_teams in groups:
        print(f"{group_root}")
        if not group_teams:
            print("  no teams")
        for team in group_teams:
            print(_format_team_line(team, _team_agents(team)))
        print()

    if has_teamless:
        print(f"(teamless) {STATE_FILE}")
        teamless = _agents_from_registry(STATE_FILE)
        if not teamless:
            print(f"  {_format_agents(teamless)}")
        else:
            for name, backend in sorted(teamless.items()):
                print(f"  {name} backend={backend}")


def _teardown_team(team: str, team_root: Path) -> None:
    """Remove a team's registry, leaving its worktree and git metadata alone.

    Takes the already-resolved `team_root` rather than rebuilding it from
    `TEAMS_DIR / team`: `_resolve_team` is the one place a team name is
    joined to a root, whether that root is `TEAMS_DIR` or a `teams.resolve`
    hit under `ROOT`.

    Scoped to `agents/`: that directory holds the state file, its lock, and
    the directory of per-agent turn locks. `worktree/` is a git working tree —
    removing it is `git worktree remove`, and it is the caller's call, not
    teardown's. The team lock's own file, a sibling of `agents/`, survives.
    """
    agents_dir = team_root / "agents"
    if agents_dir.exists():
        shutil.rmtree(agents_dir)
    print(f"deleted team '{team}'")


def cmd_delete(orchestrator: Orchestrator, opts: argparse.Namespace) -> None:
    # Always a named agent: `delete --team T` with no name is teardown, and
    # main() runs that itself, before an Orchestrator exists.
    agent = orchestrator.delete(opts.name)
    print(f"deleted agent '{agent.name}' backend={agent.backend.name}")


def _retry_count(raw: str) -> int:
    """--retries as a count, rejecting a negative one.

    argparse turns the raised error into its own exit 2. Without this, -1
    would mean "no attempts at all", which is not a thing this command can do.
    """
    count = int(raw)
    if count < 0:
        raise argparse.ArgumentTypeError(f"expected 0 or more, got {count}")
    return count


def _positive_seconds(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected 1 or more, got {value}")
    return value


# The level each verbosity selects, indexed by the summed argparse counts.
VERBOSITY_LEVELS = (logging.WARNING, logging.DEBUG, TRACE)

# Raised by the verbose flags. Only this project's loggers are turned up:
# setting the root logger to DEBUG would also enable every dependency's debug
# output, so the one signal being asked for would arrive buried in third-party
# noise.
OWN_LOGGERS = ("orchestrator", "backends")


class _VersionAction(argparse.Action):
    """Print the project version and stop before argparse validates the rest."""

    def __init__(self, option_strings: list[str], dest: str, **kwargs: Any) -> None:
        super().__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        del namespace, values, option_string
        _print_version()
        parser.exit(0)


class _CLIArgumentParser(argparse.ArgumentParser):
    """Use the selected verb's usage line for leftover-argument errors."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._verb_parsers: dict[str, _CLIArgumentParser] = {}
        self._error_parser: argparse.ArgumentParser | None = None

    def error(self, message: str) -> NoReturn:
        if self._error_parser is not None:
            self._error_parser.error(message)
        super().error(message)

    def parse_args(
        self,
        args: Iterable[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        arguments = list(args) if args is not None else sys.argv[1:]
        self._error_parser = None
        for token in arguments:
            if token in self._verb_parsers:
                self._error_parser = self._verb_parsers[token]
                break
        return cast(argparse.Namespace, super().parse_args(arguments, namespace))


def _add_version_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--version",
        action=_VersionAction,
        default=argparse.SUPPRESS,
        help="show the installed version",
    )


def _add_verbosity_argument(parser: argparse.ArgumentParser, dest: str) -> None:
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        dest=dest,
        help="log each step and how long it took; repeat for full prompts",
    )


def _add_verb_parser(
    subparsers: argparse._SubParsersAction,
    verb: str,
    **kwargs: Any,
) -> argparse.ArgumentParser:
    kwargs["prog"] = f"orchestrator {verb}"
    return subparsers.add_parser(verb, **kwargs)


def _build_parser() -> argparse.ArgumentParser:
    parser = _CLIArgumentParser(prog="orchestrator")
    _add_version_argument(parser)
    _add_verbosity_argument(parser, "verbosity")
    subparsers = parser.add_subparsers(dest="verb", required=True, metavar="<verb>")

    create = _add_verb_parser(subparsers, "create")
    _add_version_argument(create)
    _add_verbosity_argument(create, "verbosity_after")
    create.add_argument("name")
    _add_agent_config_options(create)
    _add_team_option(create)
    create.set_defaults(_parser=create)

    talk = _add_verb_parser(
        subparsers,
        "talk",
        epilog=(
            "prompt source: orchestrator talk NAME "
            "[-p TEXT | --prompt-file PATH | -- PROMPT...]"
        ),
    )
    _add_version_argument(talk)
    _add_verbosity_argument(talk, "verbosity_after")
    _add_agent_config_options(talk)
    talk.add_argument("name")
    talk.add_argument("-s", "--skill")
    talk.add_argument("--schema")
    talk.add_argument(
        "--retries", type=_retry_count, default=DEFAULT_VALIDATION_RETRIES
    )
    talk.add_argument("--timeout", type=_positive_seconds, default=DEFAULT_TURN_TIMEOUT)
    talk.add_argument("-p", "--prompt")
    talk.add_argument("--prompt-file")
    _add_team_option(talk)
    talk.set_defaults(_parser=talk)

    list_parser = _add_verb_parser(subparsers, "list")
    _add_version_argument(list_parser)
    _add_verbosity_argument(list_parser, "verbosity_after")
    list_parser.add_argument(
        "target", nargs="?", choices=("agents", "skills", "teams"), default="agents"
    )
    _add_team_option(list_parser)
    list_parser.set_defaults(_parser=list_parser)

    delete = _add_verb_parser(subparsers, "delete")
    _add_version_argument(delete)
    _add_verbosity_argument(delete, "verbosity_after")
    # nargs="?": `--team T` alone tears the whole team down (see cmd_delete);
    # a name deletes one agent. Neither is an error — bare `delete` is.
    delete.add_argument("name", nargs="?")
    _add_team_option(delete)
    delete.set_defaults(_parser=delete)

    doctor = _add_verb_parser(subparsers, "doctor")
    _add_version_argument(doctor)
    _add_verbosity_argument(doctor, "verbosity_after")
    doctor.set_defaults(_parser=doctor)

    parser._verb_parsers = subparsers.choices
    return parser


VERBS: dict[str, Callable[[Orchestrator, argparse.Namespace], None]] = {
    "create": cmd_create,
    "talk": cmd_talk,
    "list": cmd_list,
    "delete": cmd_delete,
}


def _configure_logging(verbosity: int) -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if verbosity:
        for name in OWN_LOGGERS:
            logging.getLogger(name).setLevel(VERBOSITY_LEVELS[verbosity])


def _project_version() -> str | None:
    """Read the version from the checkout containing this package, if valid."""
    project_file = Path(__file__).resolve().parent.parent / "pyproject.toml"
    try:
        with project_file.open("rb") as stream:
            document = tomllib.load(stream)
    except (OSError, ValueError, AttributeError, TypeError):
        return None
    project = document.get("project")
    if not isinstance(project, dict):
        return None
    version = project.get("version")
    return version if isinstance(version, str) and version else None


def _resolve_version() -> str:
    """Resolve the distribution version without touching CLI runtime state."""
    version = _project_version()
    if version is not None:
        return version
    try:
        installed_version = importlib.metadata.version("agents-army")
    except (importlib.metadata.PackageNotFoundError, ValueError, TypeError):
        raise ValueError from None
    if not isinstance(installed_version, str) or not installed_version:
        raise ValueError
    return installed_version


def _print_version() -> None:
    try:
        version = _resolve_version()
    except (ValueError, TypeError):
        print("unable to determine agents-army version", file=sys.stderr)
        raise SystemExit(1) from None
    print(version)


# The interpreter floor from pyproject's requires-python. Duplicated as a
# tuple because sys.version_info is what the running process can be compared
# against, and parsing the specifier back out of the metadata would report on
# the checkout rather than on the interpreter actually executing this.
MIN_PYTHON = (3, 11)

# Every tool `doctor` reports, in the order it prints them, paired
# with whether its absence is fine. Only jq is optional: agent CLIs
# are listed separately rather than collapsed into one "at least one" line, so
# the report says which backends this machine can actually run.
DEPENDENCY_TOOLS: tuple[tuple[str, bool], ...] = (
    ("uv", False),
    ("claude", False),
    ("codex", False),
    ("grok", False),
    ("opencode", False),
    ("jq", True),
)

# Present and required, present and optional, absent.
FOUND = "\u2713"
FOUND_OPTIONAL = "\u25cb"
NOT_FOUND = "\u2717"

# What a CLI may put between its own name and its version number, when it
# prints the name at all: `uv 0.4.18` against `jq-1.7`.
NAME_SEPARATORS = (" ", "-")

# A version probe is a courtesy, not the check: a CLI that hangs on --version
# must not hang the report, so it gets seconds rather than the turn timeout.
VERSION_PROBE_TIMEOUT = 5


def _tool_version(tool: str) -> str | None:
    """The first line of `<tool> --version`, or None if it cannot be had.

    Every failure mode is the same answer — the tool is installed and its
    version is unknown — so a CLI that is missing its runtime, hangs, exits
    non-zero, or prints nothing degrades the line instead of the command.
    """
    try:
        proc = subprocess.run(
            [tool, "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=VERSION_PROBE_TIMEOUT,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    lines = proc.stdout.splitlines()
    first_line = lines[0].strip() if lines else ""
    return first_line or None


def _describe_version(tool: str, reported: str) -> str:
    """`<tool> <version>`, without repeating a name the tool printed itself.

    The CLIs disagree about their own version line: `uv --version` prints
    "uv 0.4.18", `jq --version` prints "jq-1.7", `claude --version` prints a
    bare number, and `codex --version` prints "codex-cli 0.147.0". A leading
    copy of the tool's name is dropped only when a version number is what
    follows it, so codex keeps the product name it actually reports instead
    of being rewritten into "codex cli".
    """
    remainder = reported.removeprefix(tool)
    version = remainder[1:] if remainder[:1] in NAME_SEPARATORS else remainder
    if version[:1].isdigit():
        return f"{tool} {version}"
    if reported.startswith(tool):
        return reported
    return f"{tool} {reported}"


def _status_line(symbol: str, subject: str, note: str | None, optional: bool) -> str:
    """One report line, with its parenthesised notes rendered at most once."""
    notes = [note] if note is not None else []
    if optional:
        notes.append("optional")
    if not notes:
        return f"{symbol} {subject}"
    return f"{symbol} {subject} ({', '.join(notes)})"


def _python_line() -> str:
    """The running interpreter, checked against the floor this project needs.

    Not routed through `_status_line`: the interpreter is not a PATH lookup
    and can never be the optional half of that signature.
    """
    running = ".".join(str(part) for part in sys.version_info[:3])
    if (sys.version_info[0], sys.version_info[1]) >= MIN_PYTHON:
        return f"{FOUND} Python {running}"
    required = ".".join(str(part) for part in MIN_PYTHON)
    return f"{NOT_FOUND} Python {running} (needs {required}+)"


def _tool_line(tool: str, optional: bool) -> str:
    """One tool's line: found via PATH, with a version where one is available."""
    if shutil.which(tool) is None:
        return _status_line(NOT_FOUND, tool, "not found", optional)
    symbol = FOUND_OPTIONAL if optional else FOUND
    reported = _tool_version(tool)
    if reported is None:
        return _status_line(symbol, tool, "version unknown", optional)
    return _status_line(symbol, _describe_version(tool, reported), None, optional)


def _dependency_report() -> list[str]:
    """Every line of the setup report, in the fixed order it is printed."""
    return [
        _python_line(),
        *(_tool_line(tool, optional) for tool, optional in DEPENDENCY_TOOLS),
    ]


def _print_dependency_check() -> None:
    """Report the setup and stop.

    A status report, not a gate: it exits 0 whether every tool is present or
    none of them are, because which backends are usable is the user's call and
    a missing optional jq is not a failure at all.
    """
    for line in _dependency_report():
        print(line)


def _resolve_talk_prompt(
    opts: argparse.Namespace, tail: list[str], separator_present: bool
) -> None:
    sources = (
        opts.prompt is not None,
        opts.prompt_file is not None,
        separator_present,
    )
    if sum(sources) != 1:
        opts._parser.error("talk requires exactly one prompt source")
    if opts.prompt_file is not None:
        path = Path(opts.prompt_file).resolve()
        try:
            prompt = Path(opts.prompt_file).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            opts._parser.error(f"cannot read prompt file {path}: {exc}")
    elif separator_present:
        prompt = " ".join(tail)
    else:
        prompt = opts.prompt
    prompt = prompt.strip()
    if not prompt:
        opts._parser.error("talk prompt must not be empty")
    opts.prompt = prompt


# A team name becomes a directory name; an agent name never does (see
# Orchestrator._agent_lock_path, which digests it for exactly that reason).
# One or more '/'-joined segments, so a name can be a bare team ('issue-97')
# or a root-relative qualified tail ('agents-army-2/gdw-v3/issue-97') — the
# same string `teams.discover`/`teams.resolve` print and match against. '.'
# and '..' match the charset per segment but must still be rejected there:
# they escape the root (Path('/teams') / '..' / 'agents' ==
# Path('/teams/../agents')), and the escape works from any segment, not just
# the whole name.
_TEAM_NAME_RE = re.compile(r"[-_.A-Za-z0-9]+(?:/[-_.A-Za-z0-9]+)*")


def _team_lock_path(team_root: Path) -> Path:
    return team_root / ".lock"


@contextmanager
def _team_locked(path: Path, team: str, mode: int) -> Iterator[None]:
    """Hold the team lock, reporting a lost race as `TeamBusyError`.

    The conversion happens around the acquisition and nothing else. Catching
    `BlockingIOError` around the whole dispatch instead would claim it for any
    incidental one raised behind it — a backend's pipe, a write to a
    non-blocking stdout — and answer "the team is in use" to a caller that
    never asked for a team at all. That is the mystery `OrchestratorError`'s
    docstring describes.

    Teardown asks with `LOCK_NB` rather than a blocking `LOCK_EX` because
    Linux flock has no writer fairness: a queued exclusive waiter is
    overtaken by every later shared request, so a blocking teardown on a busy
    team would wait indefinitely. `TeamBusyError` exits 1, not argparse's 2 —
    a busy resource is not a usage mistake.
    """
    with ExitStack() as stack:
        try:
            stack.enter_context(_flock(path, mode))
        except BlockingIOError:
            raise TeamBusyError(
                f"team '{team}' is in use by another command; try again "
                "once it finishes"
            ) from None
        yield


def _usage_error(opts: argparse.Namespace, message: str) -> NoReturn:
    """`opts._parser.error(message)`, typed `NoReturn`.

    `opts._parser` is a dynamically-set `argparse.Namespace` attribute, so
    neither `ty` nor a reader can see that `_CLIArgumentParser.error` never
    returns; a caller that must produce a value (`_resolve_team_root`) needs
    that fact spelled out. The fallback raise carries no message: it can
    never execute (`.error()` always raises `SystemExit` first), so a
    message here would be untestable text with nothing to check it against.
    """
    opts._parser.error(message)
    raise AssertionError  # pragma: no cover


def _validate_team_name(team: str, opts: argparse.Namespace) -> None:
    if not _TEAM_NAME_RE.fullmatch(team) or any(
        segment in (".", "..") for segment in team.split("/")
    ):
        _usage_error(
            opts,
            f"invalid team name {team!r}: must match "
            f"{_TEAM_NAME_RE.pattern!r} segment-by-segment, and no segment "
            "may be '.' or '..'",
        )
    if "/" in team and TEAMS_DIR is not None:
        # A qualified name is ROOT-relative by construction — it is the
        # string `list teams` prints under the ROOT header. TEAMS_DIR
        # supplies its own namespace, so joining a ROOT-relative name under
        # it double-joins instead of resolving. See the PR description's
        # worked example (go.sh's export + the printed qualified name).
        _usage_error(
            opts,
            f"invalid team name {team!r}: a qualified name is relative to "
            "$AGENTS_ARMY_ROOT and cannot be used while AGENTS_ARMY_TEAMS_DIR "
            f"is set. Use the bare name {team.split('/')[-1]!r}, or "
            f"unset AGENTS_ARMY_TEAMS_DIR to resolve under {ROOT}.",
        )


def _resolve_team_root(team: str, opts: argparse.Namespace) -> Path:
    """The one place a team name is joined to a root.

    `AGENTS_ARMY_TEAMS_DIR` set short-circuits: `team_root` is just
    `TEAMS_DIR / team`, exactly as before this function existed, with no
    walk and no ambiguity — the one script that matters (`go.sh`) exports it
    and never reaches the branch below.

    `AGENTS_ARMY_TEAMS_DIR` unset walks `$AGENTS_ARMY_ROOT` with
    `teams.resolve` and never guesses: one hit is used, zero or two-or-more
    are reported through `_usage_error` (exit 2) — a usage problem (bad
    name, wrong environment, team lives elsewhere) regardless of which verb
    asked, teardown included. That is distinct from the `TEAMS_DIR`-set
    branch's own not-found case, handled by the caller once `team_root`
    comes back here: a `TEAMS_DIR/name` that simply does not exist on disk
    is "this resource is not there", not a usage mistake.
    """
    if TEAMS_DIR is not None:
        return TEAMS_DIR / team
    hits = teams.resolve(ROOT, team)
    if len(hits) == 1:
        return hits[0]
    if not hits:
        _usage_error(
            opts,
            f"no team named {team!r} under {ROOT}; a team is a directory "
            "with an agents/ or worktree/ subdirectory, e.g.:\n"
            f"  git worktree add -B {team} "
            f"{ROOT}/<repo>/<workflow>/{team}/worktree ...\n"
            "if the team lives outside $AGENTS_ARMY_ROOT, export "
            "AGENTS_ARMY_TEAMS_DIR to point at its parent",
        )
    _usage_error(
        opts,
        f"team name {team!r} is ambiguous under {ROOT}:\n"
        + "\n".join(f"  {hit}" for hit in hits)
        + "\nre-run with a qualified name, e.g. --team "
        + hits[0].relative_to(ROOT).as_posix(),
    )


def _resolve_team(
    opts: argparse.Namespace, teardown: bool
) -> AbstractContextManager[None]:
    """Point STATE_FILE/WORKDIR/SKILLS_DIR at a team's paths and lock it.

    Teamless commands (`opts.team is None`) are untouched: this returns
    `nullcontext()` and leaves the three globals exactly as they were.

    Every check here runs before `Orchestrator()` is constructed and reports
    through `opts._parser.error(...)` (exit 2), the way `_resolve_talk_prompt`
    already does — except for every verb but `create`/`talk` (`list`,
    `delete NAME`, and teardown) finding a `team_root` that doesn't exist,
    which is not a usage error and is left to raise `OrchestratorError`
    (exit 1), the same as any other `delete` of something that isn't there.

    Reassigning the globals here, once, rather than threading a `Paths`
    object through `Orchestrator`/`Agent.talk`/every `cmd_*`, is deliberate:
    each constant is read at call time at exactly four sites, and `WORKDIR =
    HOME` at import time means reassigning `orchestrator.HOME` alone would
    move nothing — all three must be set individually.
    """
    global STATE_FILE, WORKDIR, SKILLS_DIR
    team = opts.team
    if team is None:
        return nullcontext()
    _validate_team_name(team, opts)
    if "AGENTS_ARMY_STATE_FILE" in os.environ:
        opts._parser.error(
            "--team cannot be combined with an explicit AGENTS_ARMY_STATE_FILE "
            "(unset it, or drop --team)"
        )
    if "AGENTS_ARMY_HOME" in os.environ:
        opts._parser.error(
            "--team cannot be combined with an explicit AGENTS_ARMY_HOME "
            "(unset it, or drop --team)"
        )
    team_root = _resolve_team_root(team, opts)
    opts._team_root = team_root
    worktree = team_root / "worktree"
    if opts.verb in ("create", "talk"):
        # Gated on the verb, not on `teardown`: `list agents --team` and
        # `delete NAME --team` never launch a backend, they read and edit a
        # JSON file, so they must work on a team whose worktree is gone (the
        # state teardown deliberately leaves behind) or not there yet.
        # `create` keeps the gate because it stores WORKDIR at resolution —
        # letting it through would only defer this same failure to `talk`
        # with a registry already written.
        if not worktree.is_dir():
            opts._parser.error(
                f"team workspace {worktree} does not exist; create it first "
                f"with 'git worktree add {worktree} ...'"
            )
    elif not team_root.is_dir():
        # Every other verb (list, delete NAME, and teardown) still needs
        # `team_root` itself to exist, even though none of them touch
        # `worktree/`. Skipping this check let a bogus `--team NAME` reach
        # `_team_locked` -> `_flock`, whose first statement is
        # `path.parent.mkdir(parents=True, exist_ok=True)` — silently
        # fabricating `team_root` (and, for `delete NAME`'s later
        # `_agent_lock_path`, `agents/` alongside it) on disk for a typo.
        # That residue is self-perpetuating: once it exists, `_walk`'s
        # candidate rule sees `agents/` and treats it as a real team on
        # every future AGENTS_ARMY_ROOT walk. Teardown must stay possible
        # after `git worktree remove`, or a team's state is orphaned
        # forever — that removes only `worktree/`, not `team_root`, so this
        # check still passes for it. A no-op under the AGENTS_ARMY_ROOT walk:
        # `teams.resolve` only ever returns directories that already exist.
        raise OrchestratorError(f"team '{team}' not found at {team_root}")
    STATE_FILE = team_root / "agents" / "orchestrator_state.json"
    WORKDIR = worktree
    SKILLS_DIR = Path(os.environ.get("AGENTS_ARMY_SKILLS", worktree / "SKILLS"))
    # LOCK_SH for every team verb but teardown, so concurrent turns in one
    # team don't serialize on each other; LOCK_EX|LOCK_NB for teardown, so a
    # team must not be torn down while a command in it is running, and a
    # busy team fails fast (flock has no writer fairness — see _team_locked).
    mode = fcntl.LOCK_EX | fcntl.LOCK_NB if teardown else fcntl.LOCK_SH
    return _team_locked(_team_lock_path(team_root), team, mode)


def main(argv: list[str] | None = None) -> None:
    raw_argv = sys.argv[1:] if argv is None else argv
    separator_index = raw_argv.index("--") if "--" in raw_argv else len(raw_argv)
    separator_present = separator_index < len(raw_argv)
    if separator_present:
        head = raw_argv[:separator_index]
        tail = raw_argv[separator_index + 1 :]
    else:
        head = raw_argv
        tail = []

    parser = _build_parser()
    opts = parser.parse_args(head)
    if separator_present and opts.verb != "talk":
        opts._parser.error("the -- separator is only valid for talk")
    if opts.verb == "doctor":
        _print_dependency_check()
        return

    verbosity = min(opts.verbosity + opts.verbosity_after, len(VERBOSITY_LEVELS) - 1)
    _configure_logging(verbosity)
    # The prompt is one of these arguments, so log the shape and not the values.
    log.debug("cli: %d argument(s) after flag splitting", len(head) + len(tail))
    if opts.verb == "talk":
        _resolve_talk_prompt(opts, tail, separator_present)
    if opts.verb == "delete" and opts.team is None and opts.name is None:
        opts._parser.error("delete requires NAME or --team")
    # `list teams` reads every team's registry, not one; --team names a
    # single team to resolve, which is a contradiction with "list them all".
    list_teams = opts.verb == "list" and opts.target == "teams"
    if list_teams and opts.team is not None:
        opts._parser.error("list teams cannot be combined with --team")

    # Only `delete` with no NAME tears a team down; create/talk always
    # require NAME, so this is False for them without inspecting opts.team.
    teardown = opts.verb == "delete" and opts.name is None
    try:
        team_lock = _resolve_team(opts, teardown)
        with team_lock:
            log.debug("cli: dispatching '%s'", opts.verb)
            if teardown:
                # Ahead of Orchestrator(), and not through VERBS: the
                # constructor parses the registry, and a registry that will
                # not parse — invalid JSON, a backend this build no longer
                # has — is exactly what teardown exists to remove. Building
                # one first left `rm -rf` as the only way to retire a team
                # whose state file had gone bad.
                _teardown_team(opts.team, opts._team_root)
            elif list_teams:
                # Also ahead of Orchestrator(): that constructor binds one
                # STATE_FILE, and this reads N of them.
                _print_teams()
            else:
                VERBS[opts.verb](Orchestrator(), opts)
    except _CLI_ERRORS as exc:
        # KeyError(str) renders as '"message"' — print the payload, not repr.
        message = exc.args[0] if exc.args else str(exc)
        print(message, file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":  # pragma: no cover
    main()
