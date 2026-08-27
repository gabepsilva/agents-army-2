# Agents Army

A CLI that manages a fleet of coding-agent CLI sessions (Claude Code, Codex,
Grok, OpenCode, ...).
Each agent is a named, persistent conversation backed by a real CLI session, so
you can create many agents and talk to each one independently, resuming its
session every time.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- At least one agent CLI installed and authenticated:
  - Claude Code: `claude`
  - Codex: `codex`
  - Grok: `grok`
  - OpenCode 1.18.21+: `opencode`

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
uv run orchestrator create reviewer -b claude

# Create an agent backed by a Codex session
uv run orchestrator create dev -b codex

# Create an agent backed by a Grok session
uv run orchestrator create builder -b grok

# Create or verify an agent's configuration as part of a turn
uv run orchestrator talk -b codex --model gpt-5 --reasoning-effort high \
  reviewer -p "what did we decide about issue #23?"

# Talk to an agent (resumes its session, prints the reply)
uv run orchestrator talk reviewer -p "what did we decide about issue #23?"

# Open an agent's existing session in its interactive terminal UI
uv run orchestrator chat reviewer

# Read the prompt from a file instead of a flag argument or -- tail
uv run orchestrator talk reviewer --prompt-file ./prompts/issue-23-summary.txt

# Talk with one or more skills: each name is resolved to a markdown file
# under SKILLS/ (any subfolder) and that path is prepended to the prompt
uv run orchestrator talk reviewer --skill tdd,code-review --prompt "add a test for X"

# Require the reply to be JSON matching a schema, and print the object
uv run orchestrator talk reviewer --schema verdict.json --prompt "is it ready?"

# Set the wall-clock turn limit for a flag-style invocation
uv run orchestrator talk reviewer --timeout 900 --prompt "review the change"

# Fork a primed agent: 'copy' inherits reviewer's backend/model/effort and
# starts from its session on its own first turn (no turn is spent forking)
uv run orchestrator fork reviewer copy

# List every agent: registry path, backend, model/effort, turn count,
# timestamps, busy status, and session id
uv run orchestrator list

# Same agent listing, or the SKILLS/ catalog
uv run orchestrator list agents
uv run orchestrator list skills

# Delete an agent
uv run orchestrator delete reviewer

# Every form above, listed in one place
uv run orchestrator --help

# Show the project or installed package version
uv run orchestrator --version
```

### Structured replies: `--schema`

`--schema PATH` constrains a turn's reply to a JSON Schema and prints
the validated object instead of the raw text. It composes with `--skill`, and
it works across all backends:

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

uv run orchestrator talk reviewer --schema verdict.json \
  --prompt "did the build pass? stage is 'build'"
# [reviewer session=22f8bfee-...]
# {
#   "reason": "all gates green",
#   "stage": "build",
#   "verdict": "pass"
# }
```

Underneath, each CLI gets the flag it understands — `--json-schema` inline for
`claude` and `grok`, `--output-schema <file>` for `codex`. OpenCode has no
schema flag, so the shared validation and repair loop enforces its reply after
the turn, and the schema document travels in the prompt because nothing else
carries it there. Every backend gets the same instruction line; only the
document itself is added, and only for a backend whose CLI cannot take it.
OpenCode 1.18.21 is the tested minimum because its NDJSON event envelope is
not a stable public contract.

**Schemas must be strict.** `codex` rejects a lax schema with an HTTP 400
before the turn runs, while `claude` and `grok` accept one, so the orchestrator
enforces the strict subset itself and reports the offending path in the same
wording whatever the backend. Every object — nested ones, and array `items`
too — needs `"additionalProperties": false` and a `"required"` listing every
one of its properties, and `oneOf`, `allOf` and `not` are refused. `anyOf`,
`$ref` and `$defs` are fine: those were measured working on the schema-capable
backends.

```sh
uv run orchestrator talk reviewer --schema lax.json --prompt "..."
# /abs/lax.json: $.properties.detail must set "additionalProperties": false
# (codex rejects it; one schema has to mean the same thing on every backend)
# exit 2 — nothing ran, and no agent was created
```

**A reply that misses the schema is retried**, on the same session, with the
validation error appended so the agent can correct itself.
`--retries N` sets how many corrections are allowed (default 2); a
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
uv run orchestrator -v talk reviewer -p "summarise the auth module"

# -vv: the above, plus the full prompt sent and reply received
uv run orchestrator -vv talk reviewer -p "summarise the auth module"
```

```
DEBUG orchestrator:      cli: dispatching 'talk'
INFO  orchestrator:      agent 'reviewer' (claude): starting turn, resume=True
DEBUG backends.base:     claude turn: cwd=/w resume=True prompt_chars=25 timeout=1800s
DEBUG backends.base:     claude turn: invoking claude --print --resume s1 -p <prompt:25chars>
DEBUG backends.base:     claude turn: exited 0 after 12.4s with 812 chars of stdout
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
uv run orchestrator create reviewer -b claude
# created agent 'reviewer' backend=claude

uv run orchestrator talk reviewer -p "reply with only: first"
# [reviewer session=22f8bfee-...]
# first

uv run orchestrator talk reviewer -p "what did you just reply?"
# [reviewer session=22f8bfee-...]   <-- same session, conversation continued
# you replied "first"
```

### Backends

Each agent is bound to one backend, model, and reasoning effort at first use,
whether that is `create` or the first turn that names those values. Later turns
that pass any of `-b`/`--backend`, `--model`, or `--reasoning-effort` assert the
agent's exact configuration and fail on a mismatch. A turn with no config flags
silently reuses the stored configuration.
Currently available: `claude`, `codex`, `grok`, `opencode`.

New CLIs plug in by subclassing `AgentBackend` in `backends/` and registering
the class in the `_BACKENDS` table in `backends/registry.py`. A backend only
has to implement `name` and
`run_turn(prompt, session_id, cwd, timeout, schema)`. `run_turn` starts a
fresh CLI session when `session_id` is `None` and resumes it otherwise,
returning a `TurnResult` with the reply and the session id for the next turn.
`schema` is `None` unless `--schema` was used. Schema-capable backends declare
`enforces_schema = True`, pass the schema to their CLIs in whichever form those
CLIs want, and fill `TurnResult.structured`; OpenCode declares it `False`, is
sent the document inline in the prompt, and has its reply enforced by
orchestrator validation and repair instead. Failures raise a `TurnError` subclass so `talk` can print the
message without knowing which CLI ran.

Backends that support interactive `chat` opt in with `supports_chat = True` and
implement `chat_argv(session_id, cwd)` after verifying that their interactive
resume leaves the session id unchanged. The default is `False`, so adding a
backend without that contract remains safe.

Every headless turn hands its finished argv to one shared boundary,
`run_cli_turn` in `backends/base.py`, which runs the process and logs the
turn around it. Claude, Codex, and Grok take its default `stdin=DEVNULL` to
avoid blocking on an inherited pipe. OpenCode passes `prompt_on_stdin=True`
and receives the prompt through `input=` instead; its no-positional-message
mode reads stdin verbatim. Interactive `chat` is the deliberate exception:
it runs the backend's `chat_argv` with inherited terminal stdio so a person
can drive the resumed session.

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

The OpenCode backend runs `opencode run --format json --auto --dir <cwd>`
with the prompt on stdin and resumes with `--session`. Its `--model` and
`--variant` options map to the configured model and reasoning effort. Version
1.18.21 is the tested minimum. Because OpenCode has no schema-enforcing flag,
the orchestrator's validation and repair loop enforces schemas for it, and the
schema document is appended to the prompt so the reply it asks for is one the
model has actually been shown. A reply that arrives wrapped in a ```json fence
— the measured 1.18.21 behaviour — still yields its object: the adapter scans
the reply for it rather than parsing the whole text.

### State

Without `--team`, the entire registry lives in one JSON file,
`orchestrator_state.json`:

```json
{
  "reviewer": { "backend": "claude", "session_id": "22f8bfee-..." },
  "dev":      { "backend": "codex",  "session_id": "01a00087-..." }
}
```

It persists every agent's name, backend, and session id, so any process can
`talk` to an agent created earlier and resume its CLI session. No per-agent
folders or session files are written.

By default the file lives under `$AGENTS_ARMY_ROOT` (`~/.agents-army` unless
overridden), the one folder agents-army owns in `$HOME` — so a plain
`orchestrator create` run from any checkout writes to the same registry
instead of scattering one per repo. Set `AGENTS_ARMY_HOME` to relocate the
registry, the backend working directory, and the skill catalog together:

```sh
AGENTS_ARMY_HOME=~/.agents-army uv run orchestrator create dev -b claude
```

To relocate only the registry file while leaving the backend working directory
and skill catalog unchanged, set `AGENTS_ARMY_STATE_FILE` to an explicit path.
The registry file's path is resolved in this order: an explicit
`AGENTS_ARMY_STATE_FILE` wins outright; otherwise an explicitly-set
`AGENTS_ARMY_HOME` places it at `$AGENTS_ARMY_HOME/orchestrator_state.json`;
otherwise it defaults to `$AGENTS_ARMY_ROOT/orchestrator_state.json`. The
working directory always defaults to the current directory regardless — only
the registry's *default* location moved to `$AGENTS_ARMY_ROOT`.

`orchestrator list teams` walks `$AGENTS_ARMY_ROOT` (and, if it points
somewhere else, `$AGENTS_ARMY_TEAMS_DIR`) to show every team on the machine
in one shot — see [Teams](#teams) below.

Skill files are read from `SKILLS/` next to that same home. `--skill tdd`
walks the whole tree and attaches the matching markdown path to the prompt.
A skill name must be unique across every subfolder; a collision is an error.
To point at a different catalog, set `AGENTS_ARMY_SKILLS`.

### Teams

`--team NAME`, accepted by `create`, `talk`, `chat`, `fork`, `list`, and `delete`, runs
against a named team instead of the teamless layout above: its own registry,
its own working directory, isolated from every other team. This is what
lets two fleets work two different GitHub issues at once without both
running `git checkout`/`commit`/`gh pr create` against the same tree.

A team lives under `<team_root>/<team>/`, where `<team_root>` is
`$AGENTS_ARMY_TEAMS_DIR` if set, otherwise wherever `--team` resolves the
team under `$AGENTS_ARMY_ROOT` — see below:

```
$AGENTS_ARMY_TEAMS_DIR/<team>/
    agents/          # orchestrator_state.json + its lock + the per-agent locks/ dir
    worktree/        # resolved working directory for every agent — a git worktree you create
    .lock            # the team lock
```

State and workspace are siblings, never nested — a state file and its locks
sitting inside the worktree would show up as untracked litter in every `git
status` an agent runs, and would follow the branch around.

`AGENTS_ARMY_TEAMS_DIR` has **no default** — an in-checkout default gets
deleted by `git clean -xdff` if left ignored, or pollutes every `git status`
if not; a machine-global default collides across repos sharing one checkout,
since a bare team name like `issue-94` from two different clones would then
be one team root for two projects. Namespace it by repo yourself, typically
once per clone:

```sh
repo=$(basename "$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")")
export AGENTS_ARMY_TEAMS_DIR="$HOME/.agents-army/$repo/gdw-v3"
```

Do not export `$(git rev-parse --path-format=absolute
--git-common-dir)/gdw-v3` (no `$HOME/.agents-army` component) — that resolves
*inside* `.git/`, and commit `bc95578` moved
`examples/gabriels_workflow_v3/go.sh` off exactly that value: nested there,
the driver's own checkout is a parent directory of every agent's working
directory, and an agent that wanders up into it edits the code the running
script came from — the #80 run had an agent editing seven files there. This
doc kept recommending the broken value after `go.sh` itself moved on; that is
fixed here too.

`AGENTS_ARMY_TEAMS_DIR` is optional, not required: when it is set, `--team
NAME` resolves straight to `$AGENTS_ARMY_TEAMS_DIR/NAME` — no walk, no
ambiguity, and every example on this page keeps working exactly as shown.
When it is unset, `--team NAME` resolves by walking `$AGENTS_ARMY_ROOT` the
same way `orchestrator list teams` does (see [Listing every
team](#listing-every-team)): any directory with an `agents/` or `worktree/`
subdirectory is a candidate, and `NAME` may be a bare team name (`issue-97`)
or a `/`-qualified tail of the path `list teams` prints
(`agents-army-2/gdw-v3/issue-97`). One matching candidate resolves; zero
matches is an error naming `$AGENTS_ARMY_ROOT` with a recovery instruction;
two or more matches is an error listing every candidate's full path and
asking for a qualified name — `--team` never guesses which team you meant. A
`/`-qualified name is rejected outright while `AGENTS_ARMY_TEAMS_DIR` is set:
it names a path relative to `$AGENTS_ARMY_ROOT`, and joining it under
`AGENTS_ARMY_TEAMS_DIR` would double the path instead of resolving it.

The orchestrator never runs `git` itself — you create the worktree, and
`--team` refuses to run `create`/`talk`/`chat`/`fork` against a team whose
`worktree/` doesn't exist yet, since those bind an agent to it as its
working directory. `list agents --team`
and `delete NAME --team` only read and edit the registry, so they work even
with `worktree/` missing or not yet created:

```sh
git worktree add -B issue-73 "$AGENTS_ARMY_TEAMS_DIR/issue-73/worktree" origin/master

uv run orchestrator create owen --team issue-73 -b claude
uv run orchestrator talk owen --team issue-73 -p "start"
uv run orchestrator list agents --team issue-73
```

`delete --team NAME agent` removes one agent from that team's registry.
`delete --team NAME` with no agent name tears the whole team down: it
removes `agents/` (the state file, its lock, and the directory of per-agent
turn locks) and nothing else. It never touches `worktree/` — that is a git
working tree, and removing it is `git worktree remove`, the caller's call:

```sh
uv run orchestrator delete --team issue-73
git worktree remove "$AGENTS_ARMY_TEAMS_DIR/issue-73/worktree"
rm -rf "$AGENTS_ARMY_TEAMS_DIR/issue-73"
```

Teardown takes an exclusive, non-blocking lock on the team, so it refuses
(exit 1) rather than run underneath another command still using that team.
`--team` cannot be combined with an explicit `AGENTS_ARMY_HOME` or
`AGENTS_ARMY_STATE_FILE` — under `--team`, both are derived from the team
root, so an explicit value alongside `--team` could only be a stale export
worth surfacing rather than silently overriding. `AGENTS_ARMY_ROOT` is not on
that list: it is the teamless registry's own fallback (see [State](#state)),
orthogonal to teams, so it stays compatible with `--team`.

### Listing every team

`orchestrator list teams` finds every team on the machine without needing to
know `AGENTS_ARMY_TEAMS_DIR` up front. It walks `$AGENTS_ARMY_ROOT` up to 4
levels deep — enough for the namespaced `$AGENTS_ARMY_ROOT/<repo>/<workflow>/
<team>/` layout above — treating any directory containing
`agents/orchestrator_state.json` as a team and never looking inside a team it
has already found (so a team's own `worktree/` is never mistaken for a second
team). If `$AGENTS_ARMY_TEAMS_DIR` is set, it is walked too and any team not
already found under `$AGENTS_ARMY_ROOT` is reported as its own group —
whether `$AGENTS_ARMY_TEAMS_DIR` sits outside `$AGENTS_ARMY_ROOT` or is an
ancestor of it, so a team only the wider one reaches is never dropped just
because the two overlap. The resolved registry path (the registry `list agents`/`talk`
actually use — see [State](#state)) is reported as a `(teamless)` group
headed by its own path, when it exists. Each team is printed with its agent
count, its agents' names and backends, and a flag when its `worktree/` is
missing — the state `delete --team NAME` (with no agent name) deliberately
leaves behind:

```sh
uv run orchestrator list teams
```

`--team` is rejected on `list teams` (exit 2): it names one team to resolve,
which contradicts listing all of them.

A team is an isolation boundary **between** teams, not inside one: a team's
agents share one worktree, so concurrent agents in the same team can still
write the same files. See [`docs/configuration.md`](docs/configuration.md)
and [`docs/cli-reference.md`](docs/cli-reference.md) for the full flag and
environment-variable reference.

## Project layout

```
backends/          # AgentBackend interface + implementations
  base.py          # abstract AgentBackend + TurnResult + TurnError
  claude.py        # ClaudeBackend (resumes via --resume)
  codex.py         # CodexBackend (resumes via codex exec resume)
  grok.py          # GrokBackend (resumes via --resume; JSON is sessionId/text)
  opencode.py      # OpenCodeBackend (resumes via --session; NDJSON events)
  registry.py      # _BACKENDS table + register_backend/list_backends/get_backend
orchestrator/      # the orchestrator CLI (create / talk / chat / fork / list / delete / doctor)
  schema.py        # --schema loading, strict-subset checks, reply validation
  skills.py        # --skill name lookup under SKILLS/ + prompt composition
tests/             # pytest suite
tools/             # gate scripts run by `make` (coverage/mutation/ratchet/test-integrity)
```

## End-to-end workflow example

[`examples/gabriels_workflow_v3/go.sh`](examples/gabriels_workflow_v3/go.sh)
is a shell script, not a Python package. It drives five agents — owen,
spectacle, devin, code-reviewer, and doku — from an issue URL to a reviewed PR
using `uv run orchestrator talk --team`. It creates a per-issue git worktree
and tears it down at the end, and the agents converge by adding labels to the
issue and the PR. See [`docs/security.md`](docs/security.md) for the security
posture of a run.

## Quality gates

This project is developed mainly by AI agents, so its checks are
deterministic and self-enforcing rather than left to review. `make ci` is
the full gate: lint, format, types, risk-based branch coverage, mutation testing,
static security scanning (Bandit, Semgrep, pip-audit), and secret scanning
(Gitleaks). `make hooks` wires it into git so `make verify` runs on every
commit and `make ci` runs on every push. See [AGENTS.md](AGENTS.md) for the
rules behind the gates.

Coverage is deliberately not a 100% quota: core orchestration and backend
adapters require 95%, supporting gate utilities require 80%, and changed
lines require 90%. The mutation gate stays at a fixed 98% floor so
assertions must detect corrupted core behavior without rewarding tests
coupled only to implementation details.

Semgrep rules include `no-shell-true-subprocess`, `no-bare-except`, and
`no-inherited-env-agent-subprocess` — currently dormant, held for a future
Python driver that spawns the agent CLI on the project's behalf; today's
agent turns inherit the host environment by design, since each backend CLI
needs its login under the real `$HOME`.

```sh
make hooks   # install the pre-commit/pre-push gate (run once per clone)
make verify  # lint, types, tests, coverage, mutation — the local gate
make ci      # verify + security scanning — the full gate
```
