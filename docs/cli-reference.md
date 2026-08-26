# CLI Reference

```
orchestrator [--version] [-v] <verb> ...
```

Five verbs: `create`, `talk`, `list`, `delete`, `doctor`. Every verb also
accepts `-v`/`--verbose` after its own name, so `orchestrator -v talk ...` and
`orchestrator talk -v ...` both work. `--version` is top-level only —
`orchestrator talk --version` exits 2 with the `talk` usage line.

## Global options

| flag | meaning |
|---|---|
| `--version` | print the installed/project version and exit |
| `-v`, `--verbose` | log each step and how long it took; repeat (`-vv`) to also log full prompts and replies |

Logging goes to **stderr**, so it never pollutes a piped reply on stdout.
The flags are only picked up where they actually appear — `talk dev "compare -v and -vv"`
keeps `-v` in the prompt text rather than treating it as a flag, because it
comes after the positional prompt separator (see `talk` below).

`-vv` writes whole prompts and replies to stderr — mind where stderr is
going before turning it on.

## `create` — make a new agent

```
orchestrator create NAME [-b/--backend {claude,codex,grok,opencode}] [-m/--model MODEL]
                          [-e/--reasoning-effort EFFORT] [--team NAME]
```

| flag | meaning |
|---|---|
| `name` | (positional) agent name, unique in the registry |
| `-b`, `--backend` | backend to bind the agent to; defaults to `claude` |
| `-m`, `--model` | model name to pass to the backend CLI |
| `-e`, `--reasoning-effort` | reasoning-effort value to pass to the backend CLI |
| `--team` | run against team `NAME`'s registry and worktree instead of the teamless layout; found under `$AGENTS_ARMY_TEAMS_DIR` if set, otherwise resolved by walking `$AGENTS_ARMY_ROOT` — see [Teams](#teams) |

```sh
uv run orchestrator create reviewer -b claude
# created agent 'reviewer' backend=claude

uv run orchestrator create dev -b codex
uv run orchestrator create builder -b grok
```

Backend, model, and reasoning effort are fixed at creation (or at the first
`talk`, whichever happens first) — see [Backends](configuration.md#backends).

## `talk` — send a prompt to an agent, resuming its session

```
orchestrator talk NAME [-b/--backend ...] [-m/--model ...] [-e/--reasoning-effort ...]
                       [-s/--skill NAMES] [--schema PATH] [--retries N] [--timeout SECONDS]
                       [--team NAME]
                       (-p/--prompt TEXT | --prompt-file PATH | -- PROMPT...)
```

If `NAME` doesn't exist yet, `talk` creates it first (same as `create`) and
reports that to stderr, then runs the turn — so a script can `talk` a
possibly-new agent without a separate `create` step.

| flag | meaning |
|---|---|
| `name` | (positional) agent to talk to |
| `-p`, `--prompt TEXT` | the prompt, as one flag argument |
| `--prompt-file PATH` | the prompt, read as UTF-8 text from a file |
| `-- PROMPT...` | the prompt, as everything after `--`, joined with spaces |
| `-b`, `--backend` | must match the agent's existing backend if already created |
| `-m`, `--model` | must match the agent's existing model if already created |
| `-e`, `--reasoning-effort` | must match the agent's existing reasoning effort if already created |
| `-s`, `--skill NAMES` | comma-separated skill name(s), resolved under `SKILLS/` and prepended to the prompt |
| `--schema PATH` | JSON Schema file; the reply is validated against it and printed as JSON instead of raw text |
| `--retries N` | correction attempts allowed when a reply misses `--schema` (default `2`) |
| `--timeout SECONDS` | wall-clock budget for the whole turn, corrections included (default `3600`) |
| `--team` | run against team `NAME`'s registry and worktree instead of the teamless layout; found under `$AGENTS_ARMY_TEAMS_DIR` if set, otherwise resolved by walking `$AGENTS_ARMY_ROOT` — see [Teams](#teams) |

Exactly one of `-p/--prompt`, `--prompt-file`, or `-- PROMPT...` is required —
passing more than one, or none, is an error.

```sh
# -p form
uv run orchestrator talk reviewer -p "what did we decide about issue #23?"

# --prompt-file form: reads the file's UTF-8 text as the prompt
uv run orchestrator talk reviewer --prompt-file ./prompts/issue-23-summary.txt

# -- form: everything after -- is the prompt, so it can contain flag-like text
uv run orchestrator talk reviewer -- summarise issue #23 --without spoilers

# --prompt-file form: read and strip the prompt from a UTF-8 file
uv run orchestrator talk reviewer --prompt-file prompt.txt

# create-or-verify config as part of a turn
uv run orchestrator talk -b codex --model gpt-5 --reasoning-effort high \
  reviewer -p "what did we decide about issue #23?"

# attach one or more skills (each resolved to a markdown file under SKILLS/)
uv run orchestrator talk reviewer --skill tdd,code-review --prompt "add a test for X"

# require the reply to be JSON matching a schema
uv run orchestrator talk reviewer --schema verdict.json --prompt "is it ready?"

# cap the turn's wall-clock budget
uv run orchestrator talk reviewer --timeout 900 --prompt "review the change"
```

### `--schema`: structured replies

`--schema PATH` constrains the reply to a JSON Schema and prints the
validated object instead of raw text. Each backend CLI is given the flag it
understands for this (`--json-schema` inline for `claude`/`grok`,
`--output-schema <file>` for `codex`); OpenCode 1.18.21 has no CLI schema
flag, so the schema is inlined in the prompt and validation/repair enforces
its reply.

**Schemas must be strict**: every object (including nested ones and array
`items`) needs `"additionalProperties": false` and a `"required"` list
covering every property; `oneOf`, `allOf`, and `not` are rejected. `anyOf`,
`$ref`, and `$defs` are fine.

A reply that misses the schema is retried on the same session with the
validation error appended, up to `--retries` times (default `2`); a
non-JSON reply counts as a miss too. The whole retry loop still fits inside
one turn's `--timeout`.

| exit code | meaning |
|---|---|
| `0` | success |
| `1` | the agent ran but never produced a conforming reply, or the turn otherwise failed |
| `2` | the schema file is missing, malformed, or not strict — nothing ran, no agent was created |

## `list` — show agents, the skill catalog, or every team

```
orchestrator list [agents|skills|teams] [--team NAME]
```

Defaults to `agents` when no target is given. `--team NAME` reads that
team's registry (`list agents`) or indexes its worktree's `SKILLS/`
(`list skills`) instead of the teamless layout; it is rejected (exit 2) on
`list teams`, which reads every team's registry, not one.

```sh
uv run orchestrator list           # same as: list agents
uv run orchestrator list agents    # registry path, then every agent's full state
uv run orchestrator list skills    # the SKILLS/ catalog
uv run orchestrator list agents --team issue-73   # only that team's agents
uv run orchestrator list teams     # every team under AGENTS_ARMY_ROOT
```

`list agents` prints the registry path first — which file was read is not
always obvious, since `--team` and the `AGENTS_ARMY_STATE_FILE`/
`AGENTS_ARMY_HOME` ladder (see [Configuration](configuration.md#environment-variables))
can each point it somewhere else. Each agent line then shows everything the
registry knows about it: `model=-`/`effort=-` mean no `--model`/
`--reasoning-effort` was ever passed, `turns=-`/`created=-`/`last=-` mean the
agent predates this field and there is nothing truthful to show, and `busy`
appears (in a fixed-width column, so `session=` still lines up) only while a
turn is actually in flight.

```sh
uv run orchestrator list agents
# registry: /home/user/.agents-army/orchestrator_state.json
# devin                 backend=claude  model=sonnet  effort=high  turns=1  created=2026-08-25T23:27:32Z  last=2026-08-25T23:41:09Z  busy  session=8b3d8c7d-...
# owen                  backend=codex   model=-       effort=low   turns=1  created=2026-08-25T23:27:32Z  last=2026-08-25T23:39:02Z        session=1f2e3d4c-...
# scratch               backend=claude  model=-       effort=-     turns=0  created=2026-08-25T23:44:10Z  last=-                           session=-
```

`list teams` walks `$AGENTS_ARMY_ROOT` (up to 4 levels deep — enough for
`$AGENTS_ARMY_ROOT/<repo>/<workflow>/<team>/`) for every directory containing
`agents/orchestrator_state.json`, and never looks inside a team it has
already found — so a team's own `worktree/` is never mistaken for a second
team. If `$AGENTS_ARMY_TEAMS_DIR` is set, it is walked too and any team not
already found under `$AGENTS_ARMY_ROOT` is printed as its own group — this
covers a `AGENTS_ARMY_TEAMS_DIR` that sits outside `$AGENTS_ARMY_ROOT`
*and* one that is an ancestor of it, so a team reachable only through the
wider `$AGENTS_ARMY_TEAMS_DIR` is never dropped just because the two roots
overlap. Whatever registry `list agents`/`talk` currently resolve to
(`STATE_FILE` — see [Configuration](configuration.md#environment-variables)
for its default ladder) is printed as a `(teamless)` group, headed by its
own path. Each team is printed with its agent count, its agents' names and
backends, and a flag when `worktree/` is missing (the state `delete --team
NAME` leaves behind — see [Teams](#teams) below). An empty walk prints
`no teams` and still exits `0`.

```sh
uv run orchestrator list teams
# /home/user/.agents-army
#   my-repo/gdw-v3/issue-73  (2 agents: owen/claude, spectacle/claude)
#   my-repo/gdw-v3/issue-80  (1 agent: devin/claude)  [worktree missing]
#
# (teamless) /home/user/.agents-army/orchestrator_state.json
#   dev backend=claude

uv run orchestrator list teams --team issue-73
# orchestrator list: error: list teams cannot be combined with --team
```

## `delete` — remove an agent, or tear a team down

```
orchestrator delete [NAME] [--team NAME]
```

```sh
uv run orchestrator delete reviewer
# deleted agent 'reviewer' backend=claude

uv run orchestrator delete reviewer --team issue-73
# deleted agent 'reviewer' backend=claude   (from that team's registry)

uv run orchestrator delete --team issue-73
# deleted team 'issue-73'
```

Deleting an agent drops it from the registry; it does not touch the
underlying CLI's own session storage.

`NAME` is optional only so `--team NAME` alone can mean **teardown**: it
removes that team's `agents/` directory (the state file, its lock, and the
directory of per-agent turn locks) and nothing else — `worktree/` and its git
metadata are left for the caller to remove with `git worktree remove`.
Neither `--team` alone nor a bare `NAME` is an error; `delete` with
**neither** is (exit 2). Teardown takes an exclusive, non-blocking lock on
the team and refuses (exit 1) if another command is using it; it also exits
1 if the named team doesn't exist under `$AGENTS_ARMY_TEAMS_DIR` (with
`AGENTS_ARMY_TEAMS_DIR` unset, an unresolvable name is exit 2 instead — see
the table below). It never reads the registry it removes, so it still works
on a team whose state file has gone bad. See [Teams](#teams) below.

Deleting a single agent never fails on account of its lock file: if a turn
for that agent is in flight, the file is left for that turn to clean up as
it ends instead. Builds before this change left per-agent lock files as
`orchestrator_state.json.<digest>.lock`, siblings of the state file rather
than inside `orchestrator_state.json.locks/`; those are unreachable by any
current code path and can be removed with `rm -f
orchestrator_state.json.*.lock` when no orchestrator command is running.

## Teams

`--team NAME`, on `create`/`talk`/`list`/`delete`, points the command at a
team root's `agents/orchestrator_state.json` for state and its `worktree`
for the backend's working directory (and, unless `AGENTS_ARMY_SKILLS` is
set, its skill catalog) instead of the teamless layout — a named group of
agents gets its own registry and its own working directory, isolated from
every other team.

The team root itself resolves one of two ways. With `AGENTS_ARMY_TEAMS_DIR`
set, it is `$AGENTS_ARMY_TEAMS_DIR/NAME` — no walk, no ambiguity:

```sh
repo=$(basename "$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")")
export AGENTS_ARMY_TEAMS_DIR="$HOME/.agents-army/$repo/gdw-v3"
git worktree add -B issue-73 "$AGENTS_ARMY_TEAMS_DIR/issue-73/worktree" origin/master

uv run orchestrator create owen --team issue-73 -b claude
uv run orchestrator talk owen --team issue-73 -p "start"
uv run orchestrator list agents --team issue-73
uv run orchestrator delete --team issue-73          # teardown: agents/ only
git worktree remove "$AGENTS_ARMY_TEAMS_DIR/issue-73/worktree"
```

With `AGENTS_ARMY_TEAMS_DIR` unset, `--team NAME` instead resolves by
walking `$AGENTS_ARMY_ROOT` — the same walk `list teams` uses (see
[`list teams`](#list--show-agents-the-skill-catalog-or-every-team) above):
any directory with an `agents/` or `worktree/` subdirectory is a candidate,
and `NAME` may be a bare team name (`issue-73`) or a `/`-qualified tail of
the path `list teams` prints (`my-repo/gdw-v3/issue-73`). One matching
candidate resolves; zero is an error naming `$AGENTS_ARMY_ROOT` with a
recovery instruction; two or more is an error listing every candidate's
full path and asking for a qualified name instead of guessing. A
`/`-qualified name is rejected outright while `AGENTS_ARMY_TEAMS_DIR` is
set — it names a path relative to `$AGENTS_ARMY_ROOT`, and joining it under
`AGENTS_ARMY_TEAMS_DIR` would double the path instead of resolving it.

`create`/`talk --team` still require the team's `worktree` to exist, since
they launch a backend into it; `list agents`/`delete NAME --team` only read
and edit the registry, so they work whether or not `worktree` is there.

| exit code | meaning |
|---|---|
| `2` | the team name is invalid, ambiguous, or (with `AGENTS_ARMY_TEAMS_DIR` unset) not found under `$AGENTS_ARMY_ROOT`; `--team` was combined with an explicit `AGENTS_ARMY_HOME`/`AGENTS_ARMY_STATE_FILE` or with a `/`-qualified name while `AGENTS_ARMY_TEAMS_DIR` is set; or (for `create`/`talk`) the team's `worktree/` doesn't exist yet |
| `1` | a `--team NAME` with `AGENTS_ARMY_TEAMS_DIR` set names a team root that doesn't exist yet (`list`, `delete NAME`, or teardown — `create`/`talk` exit 2 instead, via the `worktree/` check above), or another command currently holds the team's lock during teardown |

See the [README's Teams section](../README.md#teams) for the full
`agents/`/`worktree/`/`.lock` layout and the locking model, and
[Configuration](configuration.md#teams) for the environment-variable
reference.

## `doctor` — check local setup

```
orchestrator doctor
```

Reports the running Python version against this project's floor, and
whether `uv`, `claude`, `codex`, `grok`, `opencode`, and `jq` (optional) are on `PATH`,
with their versions where available. Always exits `0` — it's a status
report, not a gate; which backends you can actually use is your call.

```sh
uv run orchestrator doctor
# ✓ Python 3.12.3
# ✓ uv 0.4.18
# ✓ claude 1.2.3
# ✗ codex (not found)
# ✗ grok (not found)
# ✓ opencode 1.18.21
# ○ jq-1.7 (optional)
```
