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
  - Grok: `grok`

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

# Create an agent backed by a Grok session
uv run orchestrator spawn builder -b grok

# Talk to an agent (resumes its session, prints the reply)
uv run orchestrator talk reviewer "what did we decide about issue #23?"

# Talk with one or more skills: each name is resolved to a markdown file
# under SKILLS/ (any subfolder) and that path is prepended to the prompt
uv run orchestrator --agent reviewer --skill tdd,code-review --prompt "add a test for X"

# Require the reply to be JSON matching a schema, and print the object
uv run orchestrator --agent reviewer --validate-schema verdict.json --prompt "is it ready?"

# List all agents and their session ids
uv run orchestrator list

# Same agent listing, or the SKILLS/ catalog, via flags (`list` stays)
uv run orchestrator --list agents
uv run orchestrator --list skills

# Delete an agent
uv run orchestrator delete reviewer

# Every form above, listed in one place
uv run orchestrator --help
```

### Structured replies: `--validate-schema`

`--validate-schema PATH` constrains a turn's reply to a JSON Schema and prints
the validated object instead of the raw text. It composes with `--skill`, and
it works the same way on all three backends:

```sh
cat > verdict.json <<'JSON'
{
  "type": "object",
  "additionalProperties": false,
  "required": ["stage", "verdict", "reason"],
  "properties": {
    "stage":   { "type": "string" },
    "verdict": { "type": "string", "enum": ["pass", "fail"] },
    "reason":  { "type": "string" }
  }
}
JSON

uv run orchestrator --agent reviewer --validate-schema verdict.json \
  --prompt "did the build pass? stage is 'build'"
# [reviewer session=22f8bfee-...]
# {
#   "reason": "all gates green",
#   "stage": "build",
#   "verdict": "pass"
# }
```

Underneath, each CLI gets the flag it understands — `--json-schema` inline for
`claude` and `grok`, `--output-schema <file>` for `codex` — and the same one
extra prompt line on every backend. The three CLIs constrain the reply
themselves; the line and the retries below are the fallback, not the mechanism.

**Schemas must be strict.** `codex` rejects a lax schema with an HTTP 400
before the turn runs, while `claude` and `grok` accept one, so the orchestrator
enforces the strict subset itself and reports the offending path in the same
wording whatever the backend. Every object — nested ones, and array `items`
too — needs `"additionalProperties": false` and a `"required"` listing every
one of its properties, and `oneOf`, `allOf` and `not` are refused. `anyOf`,
`$ref` and `$defs` are fine: those were measured working on all three.

```sh
uv run orchestrator --agent reviewer --validate-schema lax.json --prompt "..."
# /abs/lax.json: $.properties.detail must set "additionalProperties": false
# (codex rejects it; one schema has to mean the same thing on every backend)
# exit 2 — nothing ran, and no agent was created
```

**A reply that misses the schema is retried**, on the same session, with the
validation error appended so the agent can correct itself.
`--validation-retries N` sets how many corrections are allowed (default 2); a
reply that is not JSON at all counts as a miss and is retried too. The whole
loop is bounded by one turn's wall-clock budget, so a validated call can never
cost more time than a plain turn.

The two failures exit differently, so a script can tell them apart without
parsing the message:

| exit | meaning |
|---|---|
| **2** | the schema file is missing, malformed, or not strict — nothing ran |
| **1** | the agent ran and never produced a conforming reply, or the turn failed |

### Verbosity

A turn blocks until the CLI it drives returns, which can take minutes with no
output. Two flags say what is happening meanwhile:

```sh
# -v / --verbose: each step and how long it took
uv run orchestrator -v talk reviewer "summarise the auth module"

# -vv / --verbose2: the above, plus the full prompt sent and reply received
uv run orchestrator -vv talk reviewer "summarise the auth module"
```

```
DEBUG orchestrator:      cli: dispatching 'talk'
INFO  orchestrator:      agent 'reviewer' (claude): starting turn, resume=True
DEBUG backends.claude:   claude turn: cwd=/w resume=True prompt_chars=25 timeout=1800s
DEBUG backends.claude:   claude turn: invoking claude --print --resume s1 -p <prompt:25chars>
DEBUG backends.claude:   claude turn: exited 0 after 12.4s with 812 chars of stdout
INFO  orchestrator:      agent 'reviewer': turn finished in 12.4s
```

Logging goes to stderr, so it stays out of the reply on stdout and a pipe is
unaffected. Two things worth knowing:

- **The flags must come before the command.** Only a leading run is consumed,
  so `talk dev "compare -v and -vv"` keeps `-v` in the prompt where it belongs.
- **`-vv` writes whole conversations to stderr.** `-v` deliberately prints the
  prompt's size rather than its text, and stays a fixed size however long the
  prompt is; `-vv` is the opt-in to the full transcript, so mind where stderr
  is going before turning it on.

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
Currently available: `claude`, `codex`, `grok`.

New CLIs plug in by subclassing `AgentBackend` in `backends/` and registering
the class in the `_BACKENDS` table in `backends/registry.py`. A backend only
has to implement `name` and
`run_turn(prompt, session_id, cwd, timeout, schema)`. `run_turn` starts a
fresh CLI session when `session_id` is `None` and resumes it otherwise,
returning a `TurnResult` with the reply and the session id for the next turn.
`schema` is `None` unless `--validate-schema` was used; when it is set the
backend passes it to its CLI in whichever form that CLI wants and fills
`TurnResult.structured`. Failures raise a `TurnError` subclass so `talk` can
print the message without knowing which CLI ran.

Every backend runs its CLI with `stdin=DEVNULL`. A CLI whose stdin is an
inherited pipe rather than a terminal blocks until it is killed, so without it
a turn from cron, CI, or any host script burns its whole timeout and returns
nothing.

The Claude backend runs `claude --print --output-format json --permission-mode
bypassPermissions`. Print mode otherwise denies tools (`gh`, Bash, WebFetch)
with `sdk_opt_in_required` and can still exit 0 — the orchestrator would get a
half-written JSON dump instead of a reply.

The Grok backend runs `grok --output-format json --always-approve
--single=<prompt>`. `--always-approve` is Grok's non-interactive opt-in (the
same effect as `--permission-mode bypassPermissions`). The prompt is attached
to `--single` rather than passed as its own argument, because Grok's parser
reads a bare argument starting with `-` as a flag and rejects the run. Resume
uses `--resume`; Grok's `--session-id` only names a **new** session and errors
if that id already exists. The JSON envelope is camelCase (`sessionId`,
`text`), not Claude's `session_id` / `result`.

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

Skill files are read from `SKILLS/` next to that same home. `--skill tdd`
walks the whole tree and attaches the matching markdown path to the prompt.
A skill name must be unique across every subfolder; a collision is an error.
To point at a different catalog, set `AGENTS_ARMY_SKILLS`.

## Project layout

```
backends/          # AgentBackend interface + implementations (claude, codex, grok)
  base.py          # abstract AgentBackend + TurnResult + TurnError
  claude.py        # ClaudeBackend (resumes via --resume)
  codex.py         # CodexBackend (resumes via codex exec resume)
  grok.py          # GrokBackend (resumes via --resume; JSON is sessionId/text)
  registry.py      # _BACKENDS table + register_backend/list_backends/get_backend
orchestrator/      # the orchestrator CLI (spawn / talk / list / delete)
  schema.py        # --validate-schema loading, strict-subset checks, reply validation
  skills.py        # --skill name lookup under SKILLS/ + prompt composition
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
