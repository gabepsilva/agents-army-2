# Gabriel's development workflow V2

Eight roles drive a raw GitHub issue to a draft pull request. What
distinguishes this driver is where the run's truth lives and how much of it
each agent is asked to read.

Execution state is a local checkpoint store, not the issue thread. Each agent
gets a **compact handoff** from the stage before it, and GitHub is posted to
only twice — the validated specification, and the final summary. The V1 driver
this replaced published every stage to GitHub and rebuilt each prompt from
issue and pull-request context.

## The relay

The driver, not an agent, decides what runs next. Each stage returns its own
schema-validated payload plus a common envelope:

```json
{
  "handoff": {
    "summary": "...",
    "decisions": ["..."],
    "open_questions": [],
    "next_task": "...",
    "relevant_files": ["..."],
    "required_evidence": ["..."]
  }
}
```

`next_task` is advice. The driver picks the role, prompt, schema, model, skills,
timeout, and budget for the following stage regardless of what an agent asked
for, so a confused or hostile reply cannot reroute the run.

**A handoff supplements canonical evidence; it never replaces it.** Every stage
also receives the artifact it actually depends on, and every agent works inside
the issue's real worktree:

| Stage | Canonical evidence | Handoff |
| --- | --- | --- |
| expander | bounded issue | — |
| griller | bounded issue, proposal | expander's |
| expander (revise) | bounded issue, proposal | griller's |
| specifier | bounded issue, accepted proposal | in the proposal |
| implementer | specification | specifier's |
| documenter | specification | implementer's |
| repair | specification, CI output or review findings | the repaired stage's |
| reviewers | specification, worktree, `diff_against`, CI verdicts | — |
| finalizer | issue, specification, CI, reviews, PR URL | every stage's |

Reviewers are given the specification and the base commit to diff against, never
the implementer's account of its own work. That is deliberate: a reviewer that
reads only a summary reviews the summary.

Every finding carries a `severity`: `critical` and `required` block approval and
send the round to repair; `optional` and `nit` do not — a reviewer can approve
while still listing them, for example a legitimate scope deferral. A verdict
that disagrees with its own findings (an approval carrying a blocking finding,
a request for changes carrying none, or another round asked for without one)
fails the run closed before that reviewer's verdict is ever acted on.

## What GitHub gets

Two comments, both driver-authored from validated schema fields, plus a third
that only fires when it applies:

1. **Validated specification** on the issue, once clarification converges.
2. **Final implementation summary** on the pull request, after publication.
3. **A specification-review deferral**, posted only when the specification
   reviewer approves while still listing a non-blocking finding, so a
   legitimate deferral still reaches a human without costing a repair round
   or an extra agent turn. The quality reviewer's non-blocking findings are
   not posted this way; they stay in the checkpoint, the run ledger, and the
   finalizer's context.

Plus the pull request itself, whose body is assembled from the specification,
the two work summaries, the CI and review verdicts, and any scope the specifier
deferred.

### Scope, and who narrowed it

From `_specify` onward the specification is ground truth, so a bullet the
specifier drops is indistinguishable downstream from work the issue never
asked for. Every `out_of_scope` entry therefore carries its own provenance:

```json
{
  "item": "no scheduling of prune into the Makefile target",
  "source": "specifier_reduction",
  "justification": "wiring is a follow-up the issue did not ask for"
}
```

`source` is `issue_declared` only when the issue's own text excludes the item,
and the justification quotes that text. Anything the specifier decides to drop
or defer is a `specifier_reduction` and must say why. The pull-request body
renders the reductions under **Scope the specifier deferred** — and only the
reductions, since an `issue_declared` entry is already written on the issue the
body links to. The section is omitted when there are none, so its presence is
itself the signal. The specification reviewer audits the same field against the
canonical issue.

### Seeing which agents ran

Two comments hide the fleet, so the process record is published alongside them
rather than reconstructed from eighteen stage comments:

- **A run ledger** appended to the final summary: one row per stage with its
  role, backend, model, reasoning effort, skills, duration, and outcome, plus
  whether the stage ran or was reused from a checkpoint, the agent-turn budget
  consumed,
  and a collapsed list of what each stage reported. CI runs appear as `driver`
  rows; they cost no agent turn.
- **One GitHub check run per stage**, so the Checks tab on the pull request
  lists `gdw-v2 / expansion-1`, `gdw-v2 / grill-1`, … each with its duration
  and conclusion. A reviewer asking for changes is `neutral`, not a failure —
  and so is a CI attempt a later attempt replaced. Checks are published only
  after the pull request is open, so every red row in the ledger is one the run
  went on to repair; publishing those as `failure` would leave a pull request
  that finished green permanently red, and branch protection would read it as a
  blocked merge. The ledger row keeps the real conclusion.

Check runs need `checks:write` on the App and are published after the commit
is pushed, since a check run has to attach to a commit — so they are a record
of the run, not a live progress bar. Publication is best effort: the ledger
carries the same fields, so an App without the permission loses a convenience,
never evidence, and never a run that already opened its pull request.

Markers are prefixed `<!-- gdw-v2:...`, so this driver never adopts or
suppresses comments left by anything else reading the same issue.

Full CI logs, agent replies, and every handoff stay local under
`.git/gdw-v2/issue-<n>/`. They are what the repair agent reads; they are not
what a human scrolls past.

## Budgets

Every loop is bounded and every bound is configurable. Exhausting one raises
`WorkflowStopped` with a resumable state, never a silent retry:

| Budget | Default | Stops |
| --- | --- | --- |
| `max_agent_turns` | 24 | total paid model calls per issue |
| `max_clarification_rounds` | 3 | expander/griller exchanges |
| `max_ci_attempts` | 3 | CI runs before giving up on repair |
| `max_review_rounds` | 3 | review/repair cycles |
| `max_prompt_chars` | 60000 | one prompt, checked before it is sent |
| `max_output_chars` | 30000 | one reply, checked before it is checkpointed |
| `agent_timeout` | 3600 | one agent turn, in seconds |
| `ci_timeout` | 7200 | one `make ci`, in seconds |

A turn is reserved before the agent is asked and is not refunded, so a crash
loop cannot mine the budget. A checkpoint hit costs no turn.

Two more stops are not counters. Clarification that produces the same
unresolved handoffs twice is reported as stalled. A repair that reports
`complete` without changing the worktree stops the run rather than burning the
next attempt on identical input.

## Checkpoints and resume

State lives under the git *common* directory, so a run resumes from any linked
worktree:

```
.git/gdw-v2/issue-<n>/
  workflow.json       identity, turns used, milestones, PR
  issue.json          the bounded issue, read from GitHub once
  agents/agents.json  orchestrator agent sessions
  agents/home/<agent>/ that agent's writable layer over its backend config
  checkpoints/*.json  one per stage, each carrying how its turn ran
  worktree/           the branch gdwv2/issue-<n> develops on
```

Each checkpoint records `input_sha256` — a hash of the stage's role, backend,
model, effort, prompt file, schema file, skills, and full context — and
`output_sha256`. Reusing a checkpoint whose inputs changed, or whose payload was
edited on disk, fails closed rather than relaying stale work forward. A stage
is re-asked only when something it actually depends on moved.

The pre-run tree fingerprint is recorded once, at first setup. Resuming never
re-measures it, so `require_changed` still catches a run that produced nothing.

## Configuration

One GitHub App, not eight — V2 posts milestones rather than per-role stage
comments, so there is no per-role author to distinguish. Populate the ignored
`workflow.local`:

```yaml
repository: owner/project
draft: true

github_app:
  app_id: 123456
  private_key: gdw-v2.pem   # a .pem path beside this file, or an inline PEM
                            # needs issues, pull requests, contents, and
                            # checks:write for the per-stage check runs

budgets:
  max_agent_turns: 24

retention:
  completed_retention_days: 7
  max_retention_days: 30

roles:
  expander:               {backend: codex,  model: gpt-5.1-codex, reasoning_effort: high}
  griller:                {backend: claude}
  specifier:              {backend: codex,  model: gpt-5.1-codex, reasoning_effort: high}
  implementer:            {backend: claude}
  documenter:             {backend: claude}
  reviewer-specification: {backend: codex,  model: gpt-5.1-codex, reasoning_effort: high}
  reviewer-quality:       {backend: claude}
  finalizer:              {backend: codex,  model: gpt-5.1-codex}
```

All eight roles must be named explicitly; an unknown or missing role is a
configuration error, not a default. `retention` is optional and defaults to
`completed_retention_days: 7` (when `complete` is `True`) and a hard ceiling
`max_retention_days: 30` (any run); both are bounded 0–365 and 1–365 and
`max` must be at least `completed`. Never commit a populated configuration.

## Run

```sh
uv run python -m examples.gabriels_workflow_v2.cli 42
```

`-v` logs every prompt, reply, and subprocess. `--config path/to/workflow.yaml`
selects another configuration. stdout is the pull-request URL and nothing else,
so the run stays pipeable; progress goes to stderr.

Every run reclaims disk for itself: setup prunes expired `issue-<n>`
directories under `.git/gdw-v2/` before it creates or resumes its own
worktree, so state cannot accumulate without anyone doing anything. The issue
being prepared is never pruned, however old its state is, and a prune that
fails is logged as a warning and stepped over — housekeeping never costs
someone their run.

`prune.py` remains for on-demand reclamation, for freeing disk between runs or
seeing the candidates first:

```sh
uv run python -m examples.gabriels_workflow_v2.prune [--dry-run] [--config PATH] [-v]
```

Both paths use the same rule: an `issue-<n>` directory goes once a completed
run is older than `retention.completed_retention_days` (default 7), or once
any run, complete or not, is older than `retention.max_retention_days`
(default 30). Age comes from `workflow.json`'s `completed_at` when present,
otherwise the file's mtime for legacy runs. Before deleting, a registered
`issue-<n>/worktree` is deregistered with `git worktree remove --force`; the
directory itself is removed with `shutil.rmtree(..., onerror=_chmod_and_retry)`
so mode-`000` overlay work directories do not need a preparatory `chmod -R`.
One directory that refuses to be removed is logged and skipped rather than
stopping the sweep, so a single wedged worktree cannot pin the rest on disk.
`--dry-run` lists the same candidates without deleting anything or touching
`git worktree`; `--config` defaults to `workflow.local` beside the config
module and still requires a readable `github_app.private_key`.

For issue `<n>` the workflow creates or resumes a linked worktree at
`.git/gdw-v2/issue-<n>/worktree` on branch `gdwv2/issue-<n>`, so it can be run
from the main checkout on any branch. Rerunning the same issue resumes it.

Required on `PATH`: `git`, `make`, `uv`, `orchestrator`, `bwrap`, and every
agent CLI the configuration names. All of them are checked before the first
model is paid for.

## Sandboxing

Every agent turn runs inside a `bubblewrap` sandbox built by
[`gateway.py`](gateway.py): the worktree is writable only for `implementer` and
`documenter`, and GitHub credentials never enter a turn. See
[`../../docs/security.md`](../../docs/security.md) for the full description of
what is isolated and what is deliberately left visible.

`agents/` is its own directory because the sandbox binds it read-write for
every role; nothing else in the state directory is reachable from a turn.

Handoffs do not weaken that boundary. Every prompt wraps its context in
`<untrusted_context_json>` and says so, because a handoff is written by a
previous agent — it is data the next agent evaluates, never instructions it
obeys.

## Module layout

The relay itself is [`workflow.py`](workflow.py); everything it stands on is a
leaf module beside it, so a driver that decides differently can reuse the
mechanics without inheriting the decisions:

- [`errors.py`](errors.py) — `WorkflowError`/`WorkflowStopped` and the logger
- [`gates.py`](gates.py) — what one `make ci` log says each gate did
- [`gateway.py`](gateway.py) — one sandboxed agent turn
- [`git.py`](git.py) — branch, worktree, commit, push, and running CI
- [`github_app.py`](github_app.py) — reading and updating the repository as the
  installed GitHub App, and the markdown one comment is rendered from
- [`contracts.py`](contracts.py) / [`config.py`](config.py) —
  the checkpoint store and handoff schema, and the workflow's own configuration
  including `RetentionConfig` (`completed_retention_days`/`max_retention_days`)
- [`retention.py`](retention.py) — `prune_issue_state` (with the `skip` that spares
  the issue being prepared) and the `onerror=_chmod_and_retry` fail-safe that
  clears mode-`000` overlay work directories during `rmtree`
- [`prune.py`](prune.py) — standalone `prune` entry point (`--dry-run`, `--config`, `-v`)
  that loads `RetentionConfig` and removes stale `issue-<n>` trees via `retention.py`,
  for reclaiming disk between runs rather than at the start of one

These were extracted from the V1 driver this replaced, which has been removed.
