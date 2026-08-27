# Agents Army

Agents Army is a CLI that manages a fleet of coding-agent CLI sessions —
Claude Code, Codex, Grok, or OpenCode. Each **agent** is a named, persistent
conversation backed by a real CLI session: create as many as you want, and
`talk` to each one independently. Every turn resumes that agent's underlying
session, so the agent remembers the whole conversation.

```sh
uv run orchestrator create reviewer -b claude
uv run orchestrator talk reviewer -p "reply with only: first"
# [reviewer session=22f8bfee-...]
# first

uv run orchestrator talk reviewer -p "what did you just reply?"
# [reviewer session=22f8bfee-...]   <-- same session, conversation continued
# you replied "first"
```

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- At least one agent CLI installed and authenticated: `claude`, `codex`, `grok`, or `opencode`

Run `uv run orchestrator doctor` to check which of these are on `PATH`.
OpenCode 1.18.21 is the tested minimum version.

## Setup

```sh
uv sync --all-groups   # install project + dev tools
uv run pytest          # run the test suite
```

## Where to go next

- [CLI Reference](cli-reference.md) — every verb and flag, with examples
- [Configuration](configuration.md) — environment variables, state file, skills
- [Security](security.md) — what agent turns can touch, the untrusted-data rule, and which security gates run

## Project layout

```
backends/          # AgentBackend interface + implementations
  base.py          # abstract AgentBackend + TurnResult + TurnError
  claude.py        # ClaudeBackend (resumes via --resume)
  codex.py         # CodexBackend (resumes via codex exec resume)
  grok.py          # GrokBackend (resumes via --resume; JSON is sessionId/text)
  opencode.py      # OpenCodeBackend (resumes via --session; NDJSON events)
  registry.py      # _BACKENDS table + register_backend/list_backends/get_backend
orchestrator/      # the orchestrator CLI (create / talk / fork / list / delete / doctor)
  schema.py        # --schema loading, strict-subset checks, reply validation
  skills.py        # --skill name lookup under SKILLS/ + prompt composition
tests/             # pytest suite
tools/             # gate scripts run by `make` (coverage/mutation/ratchet/test-integrity)
```

Each agent is bound to one backend, model, and reasoning effort at first use
(`create`, or the first `talk`). Later turns that pass `-b`/`--model`/`-e`
assert that configuration and fail on a mismatch, rather than silently
switching backends mid-conversation. New backends plug in by subclassing
`AgentBackend` in `backends/` and registering the class in
`backends/registry.py`.

## Quality gates

This project is developed mainly by AI agents, so its checks are
deterministic and self-enforcing rather than left to review.

Branch coverage is risk-based rather than a 100% quota: core orchestration and
backend adapters require 95%, supporting gate utilities require 80%, and
changed lines require 90%. Mutation testing holds core behavior to a fixed
98% floor.

```sh
make hooks   # install the pre-commit/pre-push gate (run once per clone)
make verify  # lint, types, tests, coverage, mutation — the local gate
make ci      # verify + security scanning — the full gate
```

See [AGENTS.md](https://github.com/gabepsilva/agents-army-2/blob/master/AGENTS.md)
for the rules behind the gates.
