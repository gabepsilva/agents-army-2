#!/usr/bin/env python3
"""Long-lived orchestrator holding an array of agents.

Each agent owns a persistent Claude Code, Codex, or Grok CLI session. Every
time you talk to an agent it resumes that session with your prompt and returns
the reply, so each agent keeps its own conversation history across messages.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import os
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path

from backends import AgentBackend, TurnError, TurnResult, get_backend, list_backends
from backends.registry import UnknownBackendError
from orchestrator.skills import (
    SkillError,
    compose_skill_prompt,
    format_skill_listing,
    index_skills,
    parse_skill_names,
    resolve_skills,
)

# State lives in the caller's working directory (override with AGENTS_ARMY_HOME)
# rather than next to the installed package, so it doesn't leak into the venv.
HOME = Path(os.environ.get("AGENTS_ARMY_HOME", Path.cwd()))
STATE_FILE = HOME / "orchestrator_state.json"
# Agents run their CLI sessions from a single shared working directory.
WORKDIR = HOME
# Skill markdown catalog. Override with AGENTS_ARMY_SKILLS; default is $HOME/SKILLS.
SKILLS_DIR = Path(os.environ.get("AGENTS_ARMY_SKILLS", HOME / "SKILLS"))
# The backend an agent gets when none is named: by `spawn`, and by the agent a
# talk creates for a name that does not exist yet.
DEFAULT_BACKEND = "claude"

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


# User-facing failures that must print one line and exit 1, never a traceback.
# UnknownBackendError comes from the registry, which cannot import this module.
_CLI_ERRORS = (OrchestratorError, UnknownBackendError)


class Agent:
    """A single named agent backed by one persistent CLI session."""

    def __init__(self, name: str, backend: AgentBackend) -> None:
        self.name = name
        self.backend = backend
        self.session_id: str | None = None

    def talk(self, prompt: str) -> TurnResult:
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
        result = self.backend.run_turn(prompt, self.session_id, WORKDIR)
        elapsed = time.monotonic() - started
        log.info("agent '%s': turn finished in %.1fs", self.name, elapsed)
        log.log(TRACE, "agent '%s' reply out:\n%s", self.name, result.reply)
        # A backend that reports no session id has not ended the conversation,
        # it has failed to name it. Keeping the previous id lets the next turn
        # resume the session instead of silently starting a fresh one.
        if result.session_id is not None:
            self.session_id = result.session_id
        return result


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

    def spawn(self, name: str, backend: str | None = None) -> Agent:
        with self._exclusive():
            self._reload()
            if name in self.agents:
                raise AgentExistsError(f"agent '{name}' already exists")
            return self._create(name, backend)

    def ensure(self, name: str, backend: str | None = None) -> tuple[Agent, bool]:
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
            return self._create(name, backend), True

    def _create(self, name: str, backend: str | None) -> Agent:
        """Register and persist a new agent. The caller holds `_exclusive()`.

        `None` means "whatever the default backend is now": resolving it here
        rather than in a default argument keeps DEFAULT_BACKEND a live lookup.
        """
        agent = Agent(
            name, get_backend(DEFAULT_BACKEND if backend is None else backend)
        )
        self.agents[name] = agent
        self._persist()
        return agent

    def talk(self, name: str, prompt: str) -> TurnResult:
        with self._agent_lock(name):
            with self._exclusive():
                self._reload()
                agent = self.agents.get(name)
                if agent is None:
                    raise AgentNotFoundError(f"no agent named '{name}'")
            result = agent.talk(prompt)
            with self._exclusive():
                self._reload()
                if name not in self.agents:
                    raise AgentNotFoundError(f"no agent named '{name}'")
                # agent.session_id, not result.session_id: the agent keeps the
                # id it already had when a backend reports none.
                self.agents[name].session_id = agent.session_id
                self._persist()
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
            return agent

    def _lock_path(self) -> Path:
        return self.state_file.with_name(self.state_file.name + ".lock")

    def _agent_lock_path(self, name: str) -> Path:
        # An agent name is free text and would not survive being used as a
        # filename; the digest only has to be stable, not readable.
        digest = hashlib.sha256(name.encode()).hexdigest()
        return self.state_file.with_name(f"{self.state_file.name}.{digest}.lock")

    @contextmanager
    def _locked(self, path: Path) -> Iterator[None]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _exclusive(self) -> AbstractContextManager[None]:
        """Serialize reads and writes of the state file."""
        return self._locked(self._lock_path())

    def _agent_lock(self, name: str) -> AbstractContextManager[None]:
        """Serialize whole turns for one agent, leaving other agents free.

        The state lock cannot do this: it covers a file write measured in
        milliseconds, while the thing that must not overlap is the turn, which
        runs for minutes. Two processes resuming the same session fork the
        conversation, and whichever persists last drops the other's reply.
        """
        return self._locked(self._agent_lock_path(name))

    def _reload(self) -> None:
        self.agents = {}
        for name, entry in self._load_state().items():
            backend = entry.get("backend")
            if backend is None:
                raise StateError(f"{self.state_file}: agent '{name}' has no backend")
            agent = Agent(name, get_backend(backend))
            agent.session_id = entry.get("session_id")
            self.agents[name] = agent

    def _load_state(self) -> dict[str, dict]:
        if not self.state_file.exists():
            return {}
        try:
            return json.loads(self.state_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise StateError(f"{self.state_file} is not valid JSON: {exc}") from exc

    def _persist(self) -> None:
        state = {
            name: {
                "backend": a.backend.name,
                "session_id": a.session_id,
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


def cmd_spawn(orchestrator: Orchestrator, args: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="spawn")
    parser.add_argument("name")
    # No argparse default: leaving this None lets spawn() resolve
    # DEFAULT_BACKEND, which is what that constant documents itself as. A
    # literal here would pin spawn to claude however DEFAULT_BACKEND changed.
    parser.add_argument("--backend", "-b", choices=list_backends())
    opts = parser.parse_args(args)
    agent = orchestrator.spawn(opts.name, opts.backend)
    print(f"spawned agent '{agent.name}' backend={agent.backend.name}")


def _ensure_agent(orchestrator: Orchestrator, name: str) -> None:
    """Create `name` if talking to it would otherwise fail, and say so.

    The notice goes to stderr: stdout carries the reply and is what a pipe
    reads, and an agent having been created is commentary on the turn, not
    part of it.
    """
    agent, created = orchestrator.ensure(name)
    if created:
        print(
            f"spawned agent '{agent.name}' backend={agent.backend.name}",
            file=sys.stderr,
        )


def cmd_talk(orchestrator: Orchestrator, args: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="talk")
    parser.add_argument("name")
    parser.add_argument("prompt", nargs=argparse.REMAINDER)
    opts = parser.parse_args(args)
    prompt = " ".join(opts.prompt).strip()
    if not prompt:
        # Exit 2 like argparse does for a bad invocation: a caller under
        # `set -e` must not read "nothing ran" as a turn that succeeded.
        print("usage: talk <agent> <prompt>", file=sys.stderr)
        raise SystemExit(2)
    _ensure_agent(orchestrator, opts.name)
    try:
        result = orchestrator.talk(opts.name, prompt)
    except TurnError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None
    print(f"[{opts.name} session={result.session_id}]")
    print(result.reply)


def _print_agents(orchestrator: Orchestrator) -> None:
    agents = orchestrator.list_agents()
    if not agents:
        print("no agents")
        return
    for name in agents:
        agent = orchestrator.agents[name]
        sid = agent.session_id or "-"
        print(f"{name:20} backend={agent.backend.name:6} session={sid}")


def cmd_list(orchestrator: Orchestrator, args: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="list")
    parser.parse_args(args)
    _print_agents(orchestrator)


def cmd_delete(orchestrator: Orchestrator, args: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="delete")
    parser.add_argument("name")
    opts = parser.parse_args(args)
    agent = orchestrator.delete(opts.name)
    print(f"deleted agent '{agent.name}' backend={agent.backend.name}")


def cmd_invoke_skills(orchestrator: Orchestrator, args: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="orchestrator")
    parser.add_argument("--agent", required=True)
    parser.add_argument("--skill", required=True)
    parser.add_argument("--prompt", required=True)
    opts = parser.parse_args(args)
    prompt = opts.prompt.strip()
    if not prompt:
        print(
            "usage: orchestrator --agent NAME --skill NAME[,NAME...] --prompt TEXT",
            file=sys.stderr,
        )
        raise SystemExit(2)
    try:
        names = parse_skill_names(opts.skill)
        resolved = resolve_skills(names, SKILLS_DIR)
    except SkillError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None
    composed = compose_skill_prompt(resolved, prompt)
    log.info(
        "agent '%s': attaching skill(s) %s",
        opts.agent,
        ", ".join(name for name, _path in resolved),
    )
    # After the skills resolve, so a bad --skill exits without having left a
    # new agent behind for a turn that never ran.
    _ensure_agent(orchestrator, opts.agent)
    try:
        result = orchestrator.talk(opts.agent, composed)
    except TurnError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None
    print(f"[{opts.agent} session={result.session_id}]")
    print(result.reply)


def cmd_flag_list(orchestrator: Orchestrator, args: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="orchestrator")
    parser.add_argument("--list", required=True, choices=("agents", "skills"))
    opts = parser.parse_args(args)
    if opts.list == "agents":
        _print_agents(orchestrator)
        return
    try:
        catalog = index_skills(SKILLS_DIR)
    except SkillError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None
    print(format_skill_listing(catalog))


def _is_list_invocation(argv: list[str]) -> bool:
    token = argv[0]
    return token == "--list" or token.startswith("--list=")


COMMANDS: dict[str, Callable[[Orchestrator, list[str]], None]] = {
    "spawn": cmd_spawn,
    "talk": cmd_talk,
    "list": cmd_list,
    "delete": cmd_delete,
}


# How loud each flag asks for. The highest one given wins, so `-v -vv` is -vv.
VERBOSE_FLAGS = {"-v": 1, "--verbose": 1, "-vv": 2, "--verbose2": 2}

# The level each verbosity selects, indexed by the count above.
VERBOSITY_LEVELS = (logging.WARNING, logging.DEBUG, TRACE)

# Raised by the verbose flags. Only this project's loggers are turned up:
# setting the root logger to DEBUG would also enable every dependency's debug
# output, so the one signal being asked for would arrive buried in third-party
# noise.
OWN_LOGGERS = ("orchestrator", "backends")

USAGE = (
    "usage: orchestrator [-v|-vv] <command> [args...]\n"
    "       orchestrator [-v|-vv] --agent NAME --skill NAME[,NAME...] --prompt TEXT\n"
    "       orchestrator [-v|-vv] --list {agents,skills}\n"
    "  -h, --help      show this message\n"
    "  -v, --verbose   log each step and how long it took\n"
    "  -vv, --verbose2  also log full prompts and replies"
)

# Handled here rather than by a parser: every dash-led token that is not
# --list belongs to the skill invocation, whose own parser knows nothing about
# the commands, so -h there would advertise a third of the CLI.
HELP_FLAGS = frozenset({"-h", "--help"})


def _configure_logging(verbosity: int) -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if verbosity:
        for name in OWN_LOGGERS:
            logging.getLogger(name).setLevel(VERBOSITY_LEVELS[verbosity])


def _take_verbosity(argv: list[str]) -> tuple[int, list[str]]:
    verbosity = 0
    consumed = 0
    for token in argv:
        if token not in VERBOSE_FLAGS:
            break
        verbosity = max(verbosity, VERBOSE_FLAGS[token])
        consumed += 1
    return verbosity, argv[consumed:]


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    verbosity, argv = _take_verbosity(argv)
    _configure_logging(verbosity)
    # The prompt is one of these arguments, so log the shape and not the values.
    log.debug("cli: %d argument(s) after flag removal", len(argv))

    if argv and argv[0] in HELP_FLAGS:
        print(USAGE)
        print(f"commands: {', '.join(COMMANDS)}")
        return

    if not argv or (argv[0] not in COMMANDS and not argv[0].startswith("-")):
        print(USAGE, file=sys.stderr)
        print(f"commands: {', '.join(COMMANDS)}", file=sys.stderr)
        raise SystemExit(2)

    try:
        orch = Orchestrator()
        if argv[0] in COMMANDS:
            log.debug("cli: dispatching '%s'", argv[0])
            COMMANDS[argv[0]](orch, argv[1:])
            return
        if _is_list_invocation(argv):
            log.debug("cli: dispatching --list")
            cmd_flag_list(orch, argv)
            return
        log.debug("cli: dispatching skill invocation")
        cmd_invoke_skills(orch, argv)
    except _CLI_ERRORS as exc:
        # KeyError(str) renders as '"message"' — print the payload, not repr.
        message = exc.args[0] if exc.args else str(exc)
        print(message, file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":  # pragma: no cover
    main()
