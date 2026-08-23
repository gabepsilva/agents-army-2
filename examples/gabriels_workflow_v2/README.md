# Gabriel's development workflow V2

Same eight roles as V1. What changed is where the run's truth lives and how
much of it each agent is asked to read.

V1 published every stage to GitHub and rebuilt each prompt from issue and
pull-request context. V2 keeps execution state in a local checkpoint store,
hands each agent a **compact handoff** from the stage before it, and posts to
GitHub only twice — the validated specification, and the final summary.

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

## What GitHub gets

Two comments, both driver-authored from validated schema fields:

1. **Validated specification** on the issue, once clarification converges.
2. **Final implementation summary** on the pull request, after publication.

Plus the pull request itself, whose body is assembled from the specification,
the two work summaries, and the CI and review verdicts.

Markers are prefixed `<!-- gdw-v2:...`, so a V2 run and a V1 run on the same
issue never adopt or suppress each other's comments.

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
  agents.json         orchestrator agent sessions
  checkpoints/*.json  one per stage
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

budgets:
  max_agent_turns: 24

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
configuration error, not a default. Never commit a populated configuration.

## Run

```sh
uv run python -m examples.gabriels_workflow_v2.cli 42
```

`-v` logs every prompt, reply, and subprocess. `--config path/to/workflow.yaml`
selects another configuration. stdout is the pull-request URL and nothing else,
so the run stays pipeable; progress goes to stderr.

For issue `<n>` the workflow creates or resumes a linked worktree at
`.git/gdw-v2/issue-<n>/worktree` on branch `gdwv2/issue-<n>`, so it can be run
from the main checkout on any branch. Rerunning the same issue resumes it.

Required on `PATH`: `git`, `make`, `uv`, `orchestrator`, `bwrap`, and every
agent CLI the configuration names. All of them are checked before the first
model is paid for.

## Sandboxing

Unchanged from V1, and shared with it: every agent turn runs inside a
`bubblewrap` sandbox, the worktree is writable only for `implementer` and
`documenter`, and GitHub credentials never enter a turn. See
[`../gabriels_workflow/README.md`](../gabriels_workflow/README.md) for the full
description of what is isolated and what is deliberately left visible.

Handoffs do not weaken that boundary. Every prompt wraps its context in
`<untrusted_context_json>` and says so, because a handoff is written by a
previous agent — it is data the next agent evaluates, never instructions it
obeys.

## Relation to V1

V1 remains at [`../gabriels_workflow/`](../gabriels_workflow/); its behavior is
unchanged. V2 reuses its hardened pieces directly — the sandboxed
`AgentGateway`, git and CI mechanics, the GitHub App client, and the
`WorkflowError`/`WorkflowStopped` vocabulary — and replaces the parts that
decide what an agent is told.

The two write to different state directories, different branches, and different
comment markers, so the same issue can be run under either.
