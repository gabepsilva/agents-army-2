#!/usr/bin/env python3
"""Agent state, locking, and orchestration primitives.

Each agent owns a persistent Claude Code, Codex, Grok, or OpenCode CLI session. Every
time you talk to an agent it resumes that session with your prompt and returns
the reply, so each agent keeps its own conversation history across messages.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import math
import os
import subprocess
import time
from collections.abc import Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple, TextIO

from backends import AgentBackend, TurnResult, get_backend
from backends.base import DEFAULT_TURN_TIMEOUT, OutputSchema

from . import (
    paths,
    skills,  # noqa: F401 - required by the core module boundary; skills.py is import-safe.
)
from .schema import (
    ReplyValidationError,
    compose_schema_prompt,
    repair_prompt,
    validate_reply,
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


class AgentBusyError(OrchestratorError):
    """Chat tried to open an agent whose turn is already in flight."""


def _utcnow() -> str:
    """The current instant as ISO-8601 UTC, seconds precision, `Z` suffix.

    A string, not an epoch float: `orchestrator_state.json` is a file
    humans open with `cat`, and a string means the display path has no
    formatting logic that can be wrong, and no locale/timezone question.
    """
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class Agent:
    """A single named agent backed by one persistent CLI session."""

    def __init__(self, name: str, backend: AgentBackend, *, workdir: Path) -> None:
        self.name = name
        self.backend = backend
        self.workdir = workdir
        self.session_id: str | None = None
        # The session this agent was forked from, until its first turn has
        # actually forked it. Set by `Orchestrator.fork` and cleared the
        # moment a turn reports a session id of this agent's own, so it is
        # both the instruction for that first turn and the record that it
        # has not happened yet.
        self.pending_fork_from: str | None = None
        self.created_at: str | None = None
        self.last_turn_at: str | None = None
        # Counts CLI turns, not logical `talk()` calls: `_turn` persists once
        # per attempt by design (see its own docstring), so a schema-repair
        # retry inside `_validated_turn` increments this once per attempt.
        self.turns: int | None = None

    def talk(
        self,
        prompt: str,
        schema: OutputSchema | None = None,
        timeout: int = DEFAULT_TURN_TIMEOUT,
        *,
        stream: bool = False,
    ) -> TurnResult:
        # A pending fork resumes the *source's* session, in a copy: this
        # agent has no session of its own until this turn reports one.
        forking = self.pending_fork_from is not None
        resume_from = self.pending_fork_from if forking else self.session_id
        log.info(
            "agent '%s' (%s): starting turn, resume=%s fork=%s",
            self.name,
            self.backend.name,
            bool(resume_from),
            forking,
        )
        # Logged here rather than per backend: the turn is the same exchange
        # whichever CLI runs it, so every backend gets this for free.
        log.log(TRACE, "agent '%s' prompt in:\n%s", self.name, prompt)
        started = time.monotonic()
        result = self.backend.run_turn(
            prompt,
            resume_from,
            self.workdir,
            timeout,
            schema,
            resume_as_fork=forking,
            stream=stream,
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
            # A forked resume that reports the id it was handed did not fork:
            # the CLI continued the source's session instead of copying it.
            # Storing that id would leave two agents resuming one session
            # under different names — the overlap `_agent_lock` exists to
            # prevent, and the one case it cannot, since it keys on the name.
            # Raising here keeps the marker pending and writes nothing.
            if result.session_id == self.pending_fork_from:
                raise OrchestratorError(
                    f"agent '{self.name}': {self.backend.name} reported the "
                    f"source's own session id ('{result.session_id}'), so the "
                    f"fork did not happen; refusing to point two agents at "
                    f"one session"
                )
            self.session_id = result.session_id
            # Only now has the fork happened. A turn that reported no id
            # leaves the marker in place, so the next turn forks the source
            # again instead of starting a conversation from nothing.
            self.pending_fork_from = None
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


class _AgentRecord(NamedTuple):
    """One agent as `orchestrator_state.json` stores it.

    The persisted shape lives here and nowhere else: `_reload` and `_persist`
    both go through this type, so the eight fields are named once per
    direction instead of once per call site drifting apart. A live `Agent`
    spreads them across itself and its `AgentBackend`, which is why this is a
    projection over both rather than a mirror of `Agent.__init__`.

    Immutable because a record is what a file said (or is about to say),
    never the object a turn mutates — that stays `Agent`. A `NamedTuple`
    rather than a frozen dataclass for that immutability: mutmut skips a
    decorated class body wholesale (`mutation/file_mutation.py:292`), so
    `@dataclass` would take all four methods below — the omission branches
    and the backend guard among them — out of the mutation gate's reach
    without anything reporting that it had.
    """

    backend: str
    session_id: str | None
    pending_fork_from: str | None
    model: str | None
    reasoning_effort: str | None
    created_at: str | None
    last_turn_at: str | None
    turns: int | None

    @classmethod
    def from_entry(cls, name: str, entry: dict, state_file: Path) -> _AgentRecord:
        """Read one raw registry entry.

        Only a missing `backend` is rejected here: it is the one field with
        nothing sensible to be `None`, and every other key is legitimately
        absent in a registry written before it existed. An unregistered
        backend *name* is not this function's error — it stays `get_backend`'s,
        raised in `into_agent`, so a read-only reader that never builds an
        Agent is unaffected.
        """
        backend = entry.get("backend")
        if backend is None:
            raise StateError(f"{state_file}: agent '{name}' has no backend")
        return cls(
            backend=backend,
            session_id=entry.get("session_id"),
            pending_fork_from=entry.get("pending_fork_from"),
            model=entry.get("model"),
            reasoning_effort=entry.get("reasoning_effort"),
            created_at=entry.get("created_at"),
            last_turn_at=entry.get("last_turn_at"),
            turns=entry.get("turns"),
        )

    @classmethod
    def of(cls, agent: Agent) -> _AgentRecord:
        """Project a live `Agent` and its backend into a record."""
        return cls(
            backend=agent.backend.name,
            session_id=agent.session_id,
            pending_fork_from=agent.pending_fork_from,
            model=agent.backend.model,
            reasoning_effort=agent.backend.reasoning_effort,
            created_at=agent.created_at,
            last_turn_at=agent.last_turn_at,
            turns=agent.turns,
        )

    def into_agent(self, name: str, workdir: Path) -> Agent:
        """Build the live `Agent` this record describes.

        The one place a stored backend name is resolved: backend construction
        belongs at the registry boundary, so `UnknownBackendError` surfaces
        here, when someone actually asks for the agent.
        """
        agent = Agent(
            name,
            get_backend(
                self.backend,
                model=self.model,
                reasoning_effort=self.reasoning_effort,
            ),
            workdir=workdir,
        )
        agent.session_id = self.session_id
        agent.pending_fork_from = self.pending_fork_from
        agent.created_at = self.created_at
        agent.last_turn_at = self.last_turn_at
        agent.turns = self.turns
        return agent

    def to_entry(self) -> dict:
        """The JSON entry for this record.

        Written field by field rather than from `asdict`, because the mapping
        is not mechanical: `session_id` is written even when `None` (a
        freshly spawned agent's file has said `"session_id": null` since
        before the other fields existed, and dropping it would rewrite every
        such file), while every other optional field is omitted when unset so
        an older registry round-trips unchanged.
        """
        return {
            "backend": self.backend,
            "session_id": self.session_id,
            **(
                {"pending_fork_from": self.pending_fork_from}
                if self.pending_fork_from is not None
                else {}
            ),
            **({"model": self.model} if self.model is not None else {}),
            **(
                {"reasoning_effort": self.reasoning_effort}
                if self.reasoning_effort is not None
                else {}
            ),
            **({"created_at": self.created_at} if self.created_at is not None else {}),
            **(
                {"last_turn_at": self.last_turn_at}
                if self.last_turn_at is not None
                else {}
            ),
            **({"turns": self.turns} if self.turns is not None else {}),
        }


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

    def __init__(
        self,
        runtime_paths: paths.RuntimePaths,
        *,
        state_file: Path | None = None,
    ) -> None:
        self.runtime_paths = runtime_paths
        # Resolved once here, and the authority from now on: an explicit
        # `state_file` deliberately points somewhere `runtime_paths` does
        # not, so nothing downstream reads `runtime_paths.state_file`.
        self.state_file = runtime_paths.state_file if state_file is None else state_file
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

    def fork(self, source: str, dest: str) -> Agent:
        """Register `dest` as a copy of `source`, to be forked on its first turn.

        Nothing runs here: the new agent inherits `source`'s backend, model
        and reasoning effort, and remembers the session id to fork, so the
        cost of a fork is one registry write rather than a model turn. That
        makes the fork *lazy* — `dest` inherits `source`'s context as of
        `dest`'s first turn, not as of this call.

        Every reason to refuse is checked before anything is created, so a
        rejected fork leaves no half-made agent behind.
        """
        with self._exclusive():
            self._reload()
            origin = self.agents.get(source)
            if origin is None:
                raise AgentNotFoundError(f"no agent named '{source}'")
            if origin.session_id is None:
                raise OrchestratorError(
                    f"agent '{source}' has no session to fork yet; talk to it first"
                )
            if not origin.backend.supports_fork:
                raise OrchestratorError(
                    f"agent '{source}' runs on backend "
                    f"'{origin.backend.name}', which cannot fork"
                )
            if dest in self.agents:
                raise AgentExistsError(f"agent '{dest}' already exists")
            return self._create(
                dest,
                origin.backend.name,
                origin.backend.model,
                origin.backend.reasoning_effort,
                pending_fork_from=origin.session_id,
            )

    def _create(
        self,
        name: str,
        backend: str | None,
        model: str | None,
        reasoning_effort: str | None,
        *,
        pending_fork_from: str | None = None,
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
            workdir=self.runtime_paths.workdir,
        )
        agent.created_at = _utcnow()
        agent.turns = 0
        agent.pending_fork_from = pending_fork_from
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
        *,
        stream: bool = False,
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
                    return self._turn(agent, prompt, None, timeout, stream)
                return self._validated_turn(
                    agent, prompt, schema, retries, timeout, stream
                )
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

    def chat(self, name: str) -> int:
        """Hand the terminal to an agent's interactive session.

        Chat is intentionally read-only with respect to the registry. The
        agent lock protects the session from a concurrent headless turn, and
        the registry is re-read after that lock so a waiter uses the current
        stored session. The read needs no state lock because `_persist` swaps
        the completed file into place atomically; the child inherits this
        process's stdio so the human can drive it.
        """
        path = self._agent_lock_path(name)
        with ExitStack() as stack:
            try:
                lock = stack.enter_context(
                    self._agent_lock(name, mode=fcntl.LOCK_EX | fcntl.LOCK_NB)
                )
            except BlockingIOError:
                raise AgentBusyError(
                    f"agent '{name}' is in use by another command; try again "
                    "once it finishes"
                ) from None

            # A chat process may have loaded its registry before a preceding
            # talk acquired this lock and persisted a newer session id. Read
            # after the agent lock, but deliberately do not take the state
            # lock: chat never writes the registry and must not introduce an
            # agent-lock -> state-lock ordering edge.
            self._reload()
            agent = self.agents.get(name)
            if agent is None:
                if _is_live(path, lock):
                    path.unlink(missing_ok=True)
                    log.debug("agent '%s': reclaimed lock file, no such agent", name)
                raise AgentNotFoundError(f"no agent named '{name}'")
            if agent.session_id is None or agent.pending_fork_from is not None:
                raise OrchestratorError(
                    f"agent '{name}' has no session to fork yet; talk to it first"
                )
            if not agent.backend.supports_chat:
                raise OrchestratorError(
                    f"agent '{name}' runs on backend '{agent.backend.name}', "
                    "which cannot chat"
                )

            args = agent.backend.chat_argv(agent.session_id, agent.workdir)
            proc = subprocess.run(args, cwd=str(agent.workdir), check=False)
            return proc.returncode

    def _turn(
        self,
        agent: Agent,
        prompt: str,
        schema: OutputSchema | None,
        timeout: int,
        stream: bool,
    ) -> TurnResult:
        """One turn, with its session id persisted before anything else runs.

        Persisting per attempt rather than per call is what lets a run that
        exhausts its retries still leave the session where the agent actually
        is: it moved the conversation forward whether or not the last reply
        was usable, and resuming from a stale id would replay it.
        """
        result = agent.talk(prompt, schema, timeout, stream=stream)
        with self._exclusive():
            self._reload()
            if agent.name not in self.agents:
                raise AgentNotFoundError(f"no agent named '{agent.name}'")
            # self.agents[agent.name], not agent: _reload just replaced
            # self.agents with brand-new Agent objects, so `agent` is now
            # detached from it — mutating `agent` here would be silently
            # dropped by the _persist() below, which serializes self.agents.
            entry = self.agents[agent.name]
            # agent.session_id, not result.session_id: the agent keeps the
            # id it already had when a backend reports none.
            entry.session_id = agent.session_id
            entry.pending_fork_from = agent.pending_fork_from
            entry.last_turn_at = _utcnow()
            entry.turns = (entry.turns or 0) + 1
            self._persist()
        return result

    def _validated_turn(
        self,
        agent: Agent,
        prompt: str,
        schema: OutputSchema,
        retries: int,
        timeout: int,
        stream: bool,
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
                agent,
                attempt_prompt,
                schema,
                max(1, math.ceil(remaining)),
                stream,
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
    def _locked(
        self,
        path: Path,
        mode: int = fcntl.LOCK_EX,
        *,
        revalidate: bool = False,
    ) -> Iterator[TextIO]:
        with _flock(path, mode, revalidate=revalidate) as lock:
            yield lock

    def _exclusive(self) -> AbstractContextManager[TextIO]:
        """Serialize reads and writes of the state file."""
        return self._locked(self._lock_path())

    def _agent_lock(
        self, name: str, *, mode: int = fcntl.LOCK_EX
    ) -> AbstractContextManager[TextIO]:
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
        return self._locked(self._agent_lock_path(name), mode, revalidate=True)

    def _agent_is_busy(self, name: str) -> bool:
        """Is a turn in flight for `name` right now?

        A turn holds `_agent_lock` (`fcntl.LOCK_EX`) for its whole duration,
        so trying to take it is the signal, with no new state required.

        Guarded on `path.exists()` first: `_flock` opens the path with
        `"a+"`, which *creates* it, and an unguarded probe would scatter
        lock files as a side effect of listing.

        Probed with `LOCK_SH`, not `LOCK_EX`: a shared lock is blocked by
        the exclusive holder a real turn takes — the entire signal — and by
        nothing else. An exclusive probe would also collide with another
        concurrent `list agents` and with `_reclaim_agent_lock`, reporting
        `busy` for an agent that is merely being listed or reclaimed.

        Not `revalidate=True`: the probe does not care whether the inode
        got reclaimed underneath it, and the revalidation loop would spin
        on that instead of just answering the question.
        """
        path = self._agent_lock_path(name)
        if not path.exists():
            return False
        try:
            with _flock(path, fcntl.LOCK_SH | fcntl.LOCK_NB):
                return False
        except BlockingIOError:
            return True

    def _reload(self) -> None:
        workdir = self.runtime_paths.workdir
        self.agents = {
            name: _AgentRecord.from_entry(name, entry, self.state_file).into_agent(
                name, workdir
            )
            for name, entry in self._load_state().items()
        }

    def _load_state(self) -> dict[str, dict]:
        return _load_state_file(self.state_file)

    def _persist(self) -> None:
        state = {
            name: _AgentRecord.of(agent).to_entry()
            for name, agent in self.agents.items()
        }
        payload = json.dumps(state, indent=2, sort_keys=True) + "\n"
        tmp = self.state_file.with_name(self.state_file.name + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(self.state_file)
        log.debug("state: wrote %d agent(s) to %s", len(state), self.state_file)
