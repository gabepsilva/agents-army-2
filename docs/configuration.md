# Configuration

## Environment variables

| variable | default | controls |
|---|---|---|
| `AGENTS_ARMY_ROOT` | `~/.agents-army` | the one folder agents-army owns in `$HOME`: default home of the teamless registry, and the root `list teams` walks — see [Teams](#teams) |
| `AGENTS_ARMY_HOME` | current working directory | base folder for state (when explicitly set) and, unless overridden, skills |
| `AGENTS_ARMY_STATE_FILE` | see the ladder below | the registry file's exact path |
| `AGENTS_ARMY_SKILLS` | `$AGENTS_ARMY_HOME/SKILLS`, falling back to `$AGENTS_ARMY_ROOT/SKILLS` | the skill catalog root that `--skill` and `list skills` search — see the ladder below |
| `AGENTS_ARMY_TEAMS_DIR` | **no default** | root under which `--team NAME` resolves `$AGENTS_ARMY_TEAMS_DIR/NAME/{agents,worktree}`; when unset, `--team NAME` instead resolves by walking `$AGENTS_ARMY_ROOT` — see [Teams](#teams) |

`AGENTS_ARMY_STATE_FILE` resolves in this order: an explicit
`AGENTS_ARMY_STATE_FILE` wins outright; otherwise an explicitly-set
`AGENTS_ARMY_HOME` places it at `$AGENTS_ARMY_HOME/orchestrator_state.json`;
otherwise it defaults to `$AGENTS_ARMY_ROOT/orchestrator_state.json`. The
current working directory is never consulted for this default — only
`AGENTS_ARMY_ROOT` (or its own default) is.

The skills catalog resolves in this order:

1. `AGENTS_ARMY_SKILLS`, if set, wins outright — teamless and under `--team`.
   Like any configured catalog it is used when it exists on disk; a path that
   does not exist falls through to rung 3 rather than failing on its own.
2. Otherwise the configured catalog — `$AGENTS_ARMY_HOME/SKILLS` (the current
   working directory's `SKILLS/` unless `AGENTS_ARMY_HOME` is set), or
   `$AGENTS_ARMY_TEAMS_DIR/NAME/worktree/SKILLS` under `--team` — **if it
   exists on disk**.
3. Otherwise `$AGENTS_ARMY_ROOT/SKILLS`.

Exactly one catalog wins; the two are never merged, so a checkout carrying
its own `SKILLS/` shadows the root catalog entirely. This is what lets the
driver be run from any repository, or from cron or CI, and still find the
skills you installed once — no exported variable required. If neither
directory exists, `--skill` and `list skills` fail naming both; if the
catalog that won has no skill by that name, the error names the directory
that was searched. `list skills` prints that directory as its header.

```sh
# relocate state, working directory, and skill catalog together
AGENTS_ARMY_HOME=~/.agents-army uv run orchestrator create dev -b claude

# relocate only the registry file, leaving the working directory and skill
# catalog where they are
AGENTS_ARMY_STATE_FILE=/tmp/state.json uv run orchestrator list
```

## Teams

`--team NAME` (on `create`, `talk`, `chat`, `fork`, `list`, `delete`) points
the resolved registry path, working directory, and (unless
`AGENTS_ARMY_SKILLS` is set) skill catalog at
`$AGENTS_ARMY_TEAMS_DIR/NAME/agents/orchestrator_state.json`,
`$AGENTS_ARMY_TEAMS_DIR/NAME/worktree`, and
`$AGENTS_ARMY_TEAMS_DIR/NAME/worktree/SKILLS` respectively — a named group of
agents gets its own registry and its own working directory, isolated from
every other team. A team whose worktree has no `SKILLS/` falls back to
`$AGENTS_ARMY_ROOT/SKILLS` like any other run (see the ladder above).

`AGENTS_ARMY_TEAMS_DIR` has no default; export it once per clone, namespaced
by repo so two clones sharing one `$AGENTS_ARMY_ROOT` don't collide on the
same team name:

```sh
repo=$(basename "$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")")
export AGENTS_ARMY_TEAMS_DIR="$HOME/.agents-army/$repo/gdw-v3"
```

Exporting it is not required, though. When it is unset, `--team NAME`
resolves by walking `$AGENTS_ARMY_ROOT` instead — the same walk `list teams`
uses (see [`list teams`](#list-teams) below): a directory with an `agents/`
or `worktree/` subdirectory is a candidate, `NAME` may be bare (`issue-97`)
or a `/`-qualified tail of the path `list teams` prints
(`agents-army-2/gdw-v3/issue-97`), one match resolves, zero matches is an
error naming `$AGENTS_ARMY_ROOT`, and two or more is an error listing every
candidate's full path rather than guessing. A `/`-qualified name is rejected
outright while `AGENTS_ARMY_TEAMS_DIR` is set, since it is relative to
`$AGENTS_ARMY_ROOT` and joining it under `AGENTS_ARMY_TEAMS_DIR` would double
the path instead of resolving it.

`--team` cannot be combined with an explicit `AGENTS_ARMY_HOME` or
`AGENTS_ARMY_STATE_FILE` (both are derived from the team root instead) —
`AGENTS_ARMY_ROOT` is not on that list, since it is the teamless registry's
own fallback and stays compatible with `--team`. `create`/`talk`/`chat`/`fork --team`
require the team's `worktree` to already exist, since they launch a backend
into it — the orchestrator never runs `git` itself, so the caller creates it
with `git worktree add`. `list agents`/`delete NAME --team` only read and
edit the registry, so they work with `worktree` missing or not yet created.
`delete --team NAME` with no agent name tears the whole team down: it
removes the team's `agents/` directory and nothing else, leaving `worktree/`
(and its git metadata) untouched. See the [README's Teams
section](../README.md#teams) for the full layout, the locking model, and
worked examples.

### `list teams`

```
orchestrator list teams
```

Walks `$AGENTS_ARMY_ROOT` (up to 4 levels deep — enough for
`$AGENTS_ARMY_ROOT/<repo>/<workflow>/<team>/`) for every directory containing
`agents/orchestrator_state.json`, never looking inside a team it has already
found. If `$AGENTS_ARMY_TEAMS_DIR` is set, it is walked too and any team not
already found under `$AGENTS_ARMY_ROOT` is printed as its own group — whether
`$AGENTS_ARMY_TEAMS_DIR` sits outside `$AGENTS_ARMY_ROOT` or is an ancestor
of it, so an overlap between the two never drops a team that only the wider
one reaches. The resolved registry path (the registry `list agents`/`talk` actually use —
see the ladder above) is reported as a `(teamless)` group headed by its own
path, when it exists. Each team is printed with its agent count, its agents'
names and backends, and a flag when `worktree/` is missing (the state
`delete --team NAME` leaves behind). `--team` is rejected on `list teams`
(exit 2) — it names one team to resolve, which contradicts listing all of
them.

## State

Without `--team`, the entire registry lives in one JSON file:

```json
{
  "reviewer": { "backend": "claude", "session_id": "22f8bfee-..." },
  "dev":      { "backend": "codex",  "session_id": "01a00087-..." }
}
```

It records every agent's name, backend, session id, and (when set) model,
reasoning effort, and `pending_fork_from` — enough for any process to `talk`
to an agent created earlier and resume its CLI session. No per-agent folders or session files
are written; each backend CLI owns its own session storage, addressed by the
session id kept here. See the `AGENTS_ARMY_STATE_FILE` ladder above for where
this file defaults to.

`pending_fork_from` appears only between `orchestrator fork` and the new
agent's first turn: it holds the session id that turn resumes with the
backend's fork flag, and is dropped again once the agent has a session of its
own. Every other key is unchanged by this, so a registry written before
`fork` existed still reads and rewrites identically.

## Skills

A skill is a markdown file under the skills catalog (`SKILLS/` by default,
any subfolder; see the resolution ladder above for which `SKILLS/`). `--skill NAME` resolves `NAME` to its path and prepends it to
the prompt; `--skill a,b` attaches more than one. A skill name must be
unique across the whole catalog — a collision between two subfolders is an
error, not a silent pick.

```sh
uv run orchestrator list skills
uv run orchestrator talk reviewer --skill tdd,code-review --prompt "add a test for X"
```

## Backends

Each agent is bound to one backend, model, and reasoning effort at first
use — `create`, or the first `talk` that names those values. A later turn
that passes `-b`/`--model`/`-e` asserts the agent's exact stored
configuration and fails on a mismatch, rather than switching a
conversation's backend mid-stream; a turn with no config flags silently
reuses what's stored.

Currently available: `claude`, `codex`, `grok`, `opencode` (tested minimum 1.18.21).

| backend | CLI invocation | resume | chat | fork | notes |
|---|---|---|---|---|---|
| `claude` | `claude --print --output-format json --permission-mode bypassPermissions` | `--resume <session_id>` | `claude --resume <session_id>` | `--fork-session` | print mode otherwise denies tools (`gh`, Bash, WebFetch) |
| `codex` | `codex exec --yolo` | `codex exec resume` | `codex resume <session_id>` | `codex exec fork` (in `resume`'s place) | `--yolo` is codex's alias for `--dangerously-bypass-approvals-and-sandbox`; without it a turn cannot commit in a linked worktree or reach the network |
| `grok` | `grok --output-format json --always-approve --single=<prompt>` | `--resume` | `grok --resume <session_id>` | `--fork-session` | JSON envelope is camelCase (`sessionId`, `text`); `--session-id` only names a *new* session |
| `opencode` | `opencode run --format json --auto --dir <cwd>` | `--session <session_id>` | `opencode --session <session_id>` | `--fork` | prompt via stdin; schema inlined in the prompt and enforced by validation/repair; tested minimum 1.18.21 |

A backend declares whether it can fork with the class attribute
`supports_fork`, which [`fork`](cli-reference.md) checks before it creates
anything. All four shipped backends answer `True`. The attribute defaults to
`False` on `AgentBackend`, so a backend added outside this repo has to opt in
once it can emit a fork of its own.

A backend declares whether it can be opened by [`chat`](cli-reference.md) with
`supports_chat` and provides `chat_argv(session_id, cwd)` for its interactive
resume command. That flag also defaults to `False`; it is safe for a backend
to remain headless until it has verified that interactive resume preserves the
stored session id.

Every headless turn runs its CLI through one shared boundary, `run_cli_turn`
in `backends/base.py`. Claude, Codex, and Grok take its default
`stdin=DEVNULL` — a CLI whose stdin is an inherited pipe rather than a
terminal blocks until killed. OpenCode passes `prompt_on_stdin=True` and gets
`input=prompt` instead, so its prompt is read verbatim. Interactive `chat`
deliberately uses the backend's `chat_argv` with inherited terminal stdio.

New CLIs plug in by subclassing `AgentBackend` in `backends/` and
registering the class in the `_BACKENDS` table in `backends/registry.py`. A
backend implements `name` and
`run_turn(prompt, session_id, cwd, timeout, schema, *, resume_as_fork=False)`:
it starts a fresh CLI session when `session_id` is `None` and resumes it
otherwise — into a *copy* of that session when `resume_as_fork` is set —
returning a
`TurnResult` with the reply and the session id for the next turn. Failures
raise a `TurnError` subclass so `talk` can print the message without caring
which CLI ran.
