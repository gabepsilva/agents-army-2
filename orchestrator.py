#!/usr/bin/env python3
"""Long-lived orchestrator holding an array of agents.

Each agent owns a persistent Claude Code or Codex CLI session. Every time you
talk to an agent it resumes that session with your prompt and returns the reply,
so each agent keeps its own conversation history across messages.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

from backends import AgentBackend, TurnResult, get_backend, list_backends

# State lives in the caller's working directory (override with AGENTS_ARMY_HOME)
# rather than next to the installed package, so it doesn't leak into the venv.
HOME = Path(os.environ.get("AGENTS_ARMY_HOME", Path.cwd()))
STATE_FILE = HOME / "orchestrator_state.json"
# Agents run their CLI sessions from a single shared working directory.
WORKDIR = HOME

# Named explicitly rather than via __name__: this module runs both as a script
# (__main__) and as the `orchestrator` console script, and _configure_logging
# raises the level by logger name.
log = logging.getLogger("orchestrator")

# Full prompts and replies are unbounded and are the only logs that can carry
# the content of a conversation, so they sit below DEBUG: -v stays readable and
# safe to paste, and -vv is the deliberate opt-in to the whole transcript.
TRACE = logging.DEBUG - 5
logging.addLevelName(TRACE, "TRACE")


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
        self.session_id = result.session_id
        return result


class Orchestrator:
    """Registry of named agents, each with an independent CLI session.

    Agents persist in `orchestrator_state.json` so any process can spawn an
    agent once and talk to it later, resuming the same CLI session.
    """

    def __init__(self, state_file: Path = STATE_FILE) -> None:
        self.state_file = state_file
        self.agents: dict[str, Agent] = {}
        state = self._load_state()
        for name, entry in state.items():
            agent = Agent(name, get_backend(entry["backend"]))
            agent.session_id = entry.get("session_id")
            self.agents[name] = agent
        log.debug(
            "state: loaded %d agent(s) from %s", len(self.agents), self.state_file
        )

    def spawn(self, name: str, backend: str = "claude") -> Agent:
        if name in self.agents:
            raise ValueError(f"agent '{name}' already exists")
        agent = Agent(name, get_backend(backend))
        self.agents[name] = agent
        self._persist()
        return agent

    def talk(self, name: str, prompt: str) -> TurnResult:
        agent = self.agents.get(name)
        if agent is None:
            raise KeyError(f"no agent named '{name}'")
        result = agent.talk(prompt)
        self._persist()
        return result

    def list_agents(self) -> list[str]:
        return sorted(self.agents)

    def delete(self, name: str) -> Agent:
        agent = self.agents.pop(name, None)
        if agent is None:
            raise KeyError(f"no agent named '{name}'")
        self._persist()
        return agent

    def _load_state(self) -> dict[str, dict]:
        if self.state_file.exists():
            return json.loads(self.state_file.read_text(encoding="utf-8"))
        return {}

    def _persist(self) -> None:
        state = {
            name: {
                "backend": a.backend.name,
                "session_id": a.session_id,
            }
            for name, a in self.agents.items()
        }
        self.state_file.write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        log.debug("state: wrote %d agent(s) to %s", len(state), self.state_file)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_spawn(orchestrator: Orchestrator, args: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="spawn")
    parser.add_argument("name")
    parser.add_argument("--backend", "-b", default="claude", choices=list_backends())
    opts = parser.parse_args(args)
    agent = orchestrator.spawn(opts.name, opts.backend)
    print(f"spawned agent '{agent.name}' backend={agent.backend.name}")


def cmd_talk(orchestrator: Orchestrator, args: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="talk")
    parser.add_argument("name")
    parser.add_argument("prompt", nargs=argparse.REMAINDER)
    opts = parser.parse_args(args)
    prompt = " ".join(opts.prompt).strip()
    if not prompt:
        print("usage: talk <agent> <prompt>", file=sys.stderr)
        return
    result = orchestrator.talk(opts.name, prompt)
    print(f"[{opts.name} session={result.session_id}]")
    print(result.reply)


def cmd_list(orchestrator: Orchestrator, args: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="list")
    parser.parse_args(args)
    agents = orchestrator.list_agents()
    if not agents:
        print("no agents")
        return
    for name in agents:
        agent = orchestrator.agents[name]
        sid = agent.session_id or "-"
        print(f"{name:20} backend={agent.backend.name:6} session={sid}")


def cmd_delete(orchestrator: Orchestrator, args: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="delete")
    parser.add_argument("name")
    opts = parser.parse_args(args)
    agent = orchestrator.delete(opts.name)
    print(f"deleted agent '{agent.name}' backend={agent.backend.name}")


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
    "  -v, --verbose   log each step and how long it took\n"
    "  -vv, --verbose2  also log full prompts and replies"
)


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

    if not argv or argv[0] not in COMMANDS:
        print(USAGE, file=sys.stderr)
        print(f"commands: {', '.join(COMMANDS)}", file=sys.stderr)
        raise SystemExit(2)
    orchestrator = Orchestrator()
    log.debug("cli: dispatching '%s'", argv[0])
    COMMANDS[argv[0]](orchestrator, argv[1:])


if __name__ == "__main__":  # pragma: no cover
    main()
