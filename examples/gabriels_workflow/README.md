# Gabriel's development workflow

`simple_development_workflow.py` is the readable entrypoint. It shows only the
main flow:

1. Load the issue.
2. Clarify the request with an expander and an independent griller.
3. Produce a specification — the last bot comment on the issue.
4. Open a draft pull request. Implementation, CI, and review talk there.
5. Implement the specification.
6. Run CI and repair failures.
7. Obtain specification and quality approvals.
8. Commit, push, and update the pull request.
9. Ask the PR-only finalizer and post the final Post-mortem comment.

The supporting modules own the details:

- `workflow.local` selects the repository and each role's backend, model,
  reasoning effort, and GitHub App identity.
- `setup.py` validates the configured tools and assembles the workflow services.
- `development_workflow.py` owns checkpoints, agent turns, CI and review loops,
  git operations, and publication.
- `github_app_client.py` reads the issue and creates comments and pull requests
  as a GitHub App installation.
- `prompts/` and `validations/` contain the agent prompts and JSON Schemas.

This example deliberately exercises the public CLI boundary. For every role it
runs one schema-validated `orchestrator talk ... --prompt ...` call per turn,
passing the configured backend/model/effort so the call creates or asserts the
agent before running the turn. The example does not import or construct the
Python `Orchestrator` API. The orchestrator CLI then invokes the selected
`claude`, `codex`, `grok`, or `opencode` backend CLI. OpenCode 1.18.21 is the
tested minimum; its schema is inlined in the prompt and enforced by the
otherwise-shared validation/repair loop.

The issue body is sent only for the initial expansion. Later clarification and
specification turns retain the five latest non-workflow comments. The finalizer
receives bounded issue and pull-request context plus every JSON artifact
checkpoint.

Clarification and review have no configured round limit. Every participant's
structured reply includes `needs_another_round`; the workflow advances only
after every participant in that process reports that no further exchange is
needed. CI repair likewise continues until `make ci` succeeds. Explicit
`stop`, `reject`, and `blocked` outcomes still stop the workflow, and an exact
repeat of the same unresolved state is reported as stalled to prevent a
deadlocked interaction from looping forever.

## Run

The workflow manages its own git worktree per issue, so it can be run from
the main checkout on any branch, including the default branch. For issue
`<n>` it creates a linked worktree at `.git/gdw/issue-<n>/worktree` on branch
`gdw/issue-<n>` if one doesn't exist yet, or resumes into the existing
worktree/branch if it does. Populate the ignored `workflow.local`, then run:

```sh
uv run python examples/gabriels_workflow/simple_development_workflow.py \
  42
```

Add `-v`/`--verbose` to log every prompt sent, every reply received, and every
subprocess invoked.

To use another configuration file, pass `--config path/to/workflow.yaml`.
Without that option, the script loads
`examples/gabriels_workflow/workflow.local`. Model names and reasoning-effort
values are passed through to the configured CLI. Each of `expander`, `griller`,
`specifier`, `implementer`, `documenter`, `reviewer-specification`,
`reviewer-quality`, and `finalizer` must be configured explicitly.

Each stage result is commented through that stage role's GitHub App. Expansion,
grilling, and the specification stay on the issue — the specification is the
last bot comment there. The implementer app then opens a draft pull request and
posts implementation, CI, and review there. The same app commits, pushes, and
updates that pull request when the work is done. The finalizer is PR-only,
read-only, uses no skills, and posts the last workflow-authored comment after
publication.

Each newly generated agent-stage comment ends with a six-field attribution
footer: `backend`, `model`, `reasoning_effort`, `task_duration`, `skills`, and
`worktree`. `backend`, `model`, and `reasoning_effort` render as `` `value` ``
or `_unset_` when the backend CLI chose its default. `task_duration` is the
elapsed `self.agents.ask()` turn, formatted in seconds as `X.Ys`. `skills`
renders as a single backtick-wrapped, comma-joined list (e.g.
`` `code-simplification, caveman` ``) or `_none_` when the stage explicitly
requested no skills. `worktree` renders the resolved basename plus resolved
path, shortening the user's home directory to `~` (for example,
`` `worktree` - `~/.git/gdw/issue-44/worktree` ``). A worktree outside the
home directory uses its resolved absolute path. Cached stages reuse their
checkpoint without a new comment or attribution metadata. The driver-authored
CI checklist comment carries no attribution footer.

CI is reported as a checklist rather than a log: one ✅, ❌ or ⚪ per gate that
`make ci` runs, a failing gate carrying a one-line reason. The gate list comes
from `make ci-gates`, and a gate that never started when an earlier one failed
is marked as such instead of being reported as passing. The full CI output
stays out of GitHub — it is checkpointed under `.git/gdw/` and handed to
the repair agent, which is what actually reads it.

Install all eight apps (one per role) on the configured repository with the
permissions required for issues, pull requests, and repository contents.

Private keys are accepted directly as YAML strings, including `|` multiline
PEM blocks. Never commit a populated configuration; `workflow.local` is
ignored for this reason.

Agent backend/model/effort settings are persisted with their sessions. Changing
one for an issue that has already started produces an explicit mismatch error;
remove that issue's `.git/gdw/issue-<number>/agents.json` to start fresh agent
sessions while retaining completed workflow checkpoints.

Workflow and agent state is stored under `.git/gdw/issue-<number>/`. Running the
same issue again resumes completed stages and avoids duplicate bot comments.

## Progress logs

Timestamped progress goes to stderr, so stdout stays the pull-request URL and
the run stays pipeable. At the default level each line says which stage is
running, which role was asked, how long the turn took, and the reply's
`decision`/`verdict`/`status`; checkpoint reuse on a resumed run is logged as
such, `make ci` reports its exit code, duration, and the tail of a failure, and
a stopped workflow ends with the reminder that rerunning resumes it.

```
2026-08-21 13:04:22,110 INFO    setup: repository gabepsilva/agents-army-2, roles ...
2026-08-21 13:04:24,882 INFO    git: branch=gdw/issue-22 base=master head=9f1c2ab resuming=False
2026-08-21 13:04:26,003 INFO    stage expansion-1: asking expander
2026-08-21 13:07:41,559 INFO    stage expansion-1: expander answered in 195.6s (decision=proceed, needs_another_round=False)
2026-08-21 13:07:42,004 INFO    github-app: commenting 'expansion-1' on issue #22
2026-08-21 13:21:08,771 INFO    ci: 'make ci' exited 0 after 402.1s with 18244 chars of output
```

`-v` adds the full prompt, the full reply, and each subprocess invocation.

The required local tools are `git`, `make`, `uv`, `orchestrator`, and every
authenticated agent CLI referenced by the configuration: `claude`, `codex`,
`grok`, or `opencode`.
