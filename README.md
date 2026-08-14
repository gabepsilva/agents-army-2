# Agents Army

A CLI that manages a fleet of coding-agent CLI sessions (Claude Code, Codex, ...).
Each agent is a named, persistent conversation backed by a real CLI session, so
you can spawn many agents and talk to each one independently, resuming its
session every time.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- At least one agent CLI installed and authenticated:
  - Claude Code: `claude`
  - Codex: `codex`

## Setup

```sh
uv sync --all-groups   # install project + dev tools
uv run pytest          # run the test suite
```

## The orchestrator CLI

`orchestrator` manages a registry of named agents. Every agent keeps its own
session: `talk` resumes that agent's CLI session with your prompt and returns
the reply, so each agent remembers its own conversation history.

### Commands

```sh
# Create a new agent backed by a Claude Code session (default backend)
uv run orchestrator spawn reviewer -b claude

# Create an agent backed by a Codex session
uv run orchestrator spawn dev -b codex

# Talk to an agent (resumes its session, prints the reply)
uv run orchestrator talk reviewer "what did we decide about issue #23?"

# List all agents and their session ids
uv run orchestrator list

# Delete an agent
uv run orchestrator delete reviewer
```

### Example

```sh
uv run orchestrator spawn reviewer -b claude
# spawned agent 'reviewer' backend=claude

uv run orchestrator talk reviewer "reply with only: first"
# [reviewer session=22f8bfee-...]
# first

uv run orchestrator talk reviewer "what did you just reply?"
# [reviewer session=22f8bfee-...]   <-- same session, conversation continued
# you replied "first"
```

### Backends

Each agent is bound to one backend, chosen at `spawn` time with `-b`/`--backend`.
Currently available: `claude`, `codex`.

New CLIs plug in by subclassing `AgentBackend` in `backends/` and registering
the class in the `_BACKENDS` table in `backends/registry.py`.

```python
# backends/grok.py
from pathlib import Path
from backends.base import AgentBackend, TurnResult


class GrokBackend(AgentBackend):
    name = "grok"

    def run_turn(self, prompt, session_id, cwd: Path) -> TurnResult:
        # start a new session when session_id is None, otherwise resume it
        ...
```

```python
# backends/registry.py
from backends.grok import GrokBackend

_BACKENDS = {
    "claude": ClaudeBackend,
    "codex": CodexBackend,
    "grok": GrokBackend,
}
```

A backend only has to implement `name` and `run_turn(prompt, session_id, cwd)`.
`run_turn` starts a fresh CLI session when `session_id` is `None` and resumes it
otherwise, returning a `TurnResult` with the reply and the session id for the
next turn.

### State

The entire registry lives in one JSON file, `orchestrator_state.json`:

```json
{
  "reviewer": { "backend": "claude", "session_id": "22f8bfee-..." },
  "dev":      { "backend": "codex",  "session_id": "01a00087-..." }
}
```

It persists every agent's name, backend, and session id, so any process can
`talk` to an agent spawned earlier and resume its CLI session. No per-agent
folders or session files are written.

The file lives next to where you run the CLI (the working directory), so the
state never leaks into the venv. To relocate it, set `AGENTS_ARMY_HOME`:

```sh
AGENTS_ARMY_HOME=~/.agents-army uv run orchestrator spawn dev -b claude
```

## Project layout

```
backends/          # AgentBackend interface + implementations (claude, codex)
  base.py          # abstract AgentBackend + TurnResult
  claude.py        # ClaudeBackend (resumes via --resume)
  codex.py         # CodexBackend (resumes via codex exec resume)
  registry.py      # _BACKENDS table + register_backend/list_backends/get_backend
orchestrator.py    # the orchestrator CLI (spawn / talk / list / delete)
tests/             # pytest suite
tools/             # gate scripts run by `make` (coverage/mutation/ratchet/test-integrity)
```

## Quality gates

This project is developed mainly by AI agents, so its checks are
deterministic and self-enforcing rather than left to review. `make ci` is
the full gate: lint, format, types, coverage floor, mutation testing,
static security scanning (Bandit, Semgrep, pip-audit), and secret scanning
(Gitleaks). `make hooks` wires it into git so `make verify` runs on every
commit and `make ci` runs on every push. See [AGENTS.md](AGENTS.md) for the
rules behind the gates.

```sh
make hooks   # install the pre-commit/pre-push gate (run once per clone)
make verify  # lint, types, tests, coverage, mutation — the local gate
make ci      # verify + security scanning — the full gate
```
