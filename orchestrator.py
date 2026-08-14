#!/usr/bin/env python3
"""Long-lived orchestrator holding an array of agents.

Each agent owns a persistent Claude Code or Codex CLI session. Every time you
talk to an agent it resumes that session with your prompt and returns the reply,
so each agent keeps its own conversation history across messages.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

from backends import AgentBackend, TurnResult, get_backend, list_backends

# State lives in the caller's working directory (override with AGENTS_ARMY_HOME)
# rather than next to the installed package, so it doesn't leak into the venv.
HOME = Path(os.environ.get("AGENTS_ARMY_HOME", Path.cwd()))
STATE_FILE = HOME / "orchestrator_state.json"
# Agents run their CLI sessions from a single shared working directory.
WORKDIR = HOME


class Agent:
    """A single named agent backed by one persistent CLI session."""

    def __init__(self, name: str, backend: AgentBackend) -> None:
        self.name = name
        self.backend = backend
        self.session_id: str | None = None

    def talk(self, prompt: str) -> TurnResult:
        result = self.backend.run_turn(prompt, self.session_id, WORKDIR)
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


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if not argv or argv[0] not in COMMANDS:
        print("usage: orchestrator <command> [args...]", file=sys.stderr)
        print(f"commands: {', '.join(COMMANDS)}", file=sys.stderr)
        raise SystemExit(2)
    orchestrator = Orchestrator()
    COMMANDS[argv[0]](orchestrator, argv[1:])


if __name__ == "__main__":  # pragma: no cover
    main()
