# Configuration

## Environment variables

| variable | default | controls |
|---|---|---|
| `AGENTS_ARMY_HOME` | current working directory | base folder for state and, unless overridden, skills |
| `AGENTS_ARMY_STATE_FILE` | `$AGENTS_ARMY_HOME/orchestrator_state.json` | the registry file's exact path |
| `AGENTS_ARMY_SKILLS` | `$AGENTS_ARMY_HOME/SKILLS` | the skill catalog root that `--skill` and `list skills` search |

```sh
# relocate state, working directory, and skill catalog together
AGENTS_ARMY_HOME=~/.agents-army uv run orchestrator create dev -b claude

# relocate only the registry file, leaving the working directory and skill
# catalog where they are
AGENTS_ARMY_STATE_FILE=/tmp/state.json uv run orchestrator list
```

## State

The entire registry lives in one JSON file:

```json
{
  "reviewer": { "backend": "claude", "session_id": "22f8bfee-..." },
  "dev":      { "backend": "codex",  "session_id": "01a00087-..." }
}
```

It records every agent's name, backend, session id, and (when set) model and
reasoning effort — enough for any process to `talk` to an agent created
earlier and resume its CLI session. No per-agent folders or session files
are written; each backend CLI owns its own session storage, addressed by the
session id kept here.

## Skills

A skill is a markdown file under the skills catalog (`SKILLS/` by default,
any subfolder). `--skill NAME` resolves `NAME` to its path and prepends it to
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

| backend | CLI invocation | resume | notes |
|---|---|---|---|
| `claude` | `claude --print --output-format json --permission-mode bypassPermissions` | `--resume <session_id>` | print mode otherwise denies tools (`gh`, Bash, WebFetch) |
| `codex` | `codex exec` | `codex exec resume` | |
| `grok` | `grok --output-format json --always-approve --single=<prompt>` | `--resume` | JSON envelope is camelCase (`sessionId`, `text`); `--session-id` only names a *new* session |
| `opencode` | `opencode run --format json --auto --dir <cwd>` | `--session <session_id>` | prompt via stdin; schema inlined in the prompt and enforced by validation/repair; tested minimum 1.18.21 |

Claude, Codex, and Grok run their CLIs with `stdin=DEVNULL` — a CLI whose
stdin is an inherited pipe rather than a terminal blocks until killed. OpenCode
uses `input=prompt` instead so its prompt is read verbatim.

New CLIs plug in by subclassing `AgentBackend` in `backends/` and
registering the class in the `_BACKENDS` table in `backends/registry.py`. A
backend implements `name` and
`run_turn(prompt, session_id, cwd, timeout, schema)`: it starts a fresh CLI
session when `session_id` is `None` and resumes it otherwise, returning a
`TurnResult` with the reply and the session id for the next turn. Failures
raise a `TurnError` subclass so `talk` can print the message without caring
which CLI ran.
