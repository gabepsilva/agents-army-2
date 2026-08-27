# gdw v5 — implementation brief

You are implementing this workflow. The specification is [flow.md](flow.md):
implement that diagram, exactly, as a single Python driver named `go.py` in
this directory. Every agent box in the diagram maps to one prompt file in
[prompts/](prompts/); the driver renders the prompt and delivers it with one
`uv run orchestrator talk` call. Do not invent phases, checks, or options the
diagram does not show. Where this brief and the diagram disagree, the diagram
wins.

## What the driver does

One command, `python go.py <issue-url>` (or `./go.py`), two behaviors decided
by the issue's labels — the router at the top of the diagram:

- **Planning**: BFS over the issue and any children owen creates (parsed from
  the `CHILDREN: #a #b` line ending his split comment), debating every leaf to
  convergence and posting doku's brief, then a doku tree summary on the root
  if anything split, then exiting. Cap: 12 issues per run, exit 9 beyond it.
- **Build**: one converged leaf per invocation — reuse or open the draft PR,
  devin implements and self-reviews, the driver runs `make ci` itself, at most
  three review rounds, doku's user note, cleanup.

The primer: planning starts by creating one role-neutral agent whose only turn
is to study the repository, then uses `uv run orchestrator fork <src> <dst>`
to give each issue's owen, spectacle, and doku a fresh fork of that session.
Forks are deleted after their issue. The fork verb exists in this repository —
read its docs and `--help` before use. If fork is unavailable at runtime, fall
back to fresh agents per issue and print that the fallback is in effect. A role
whose dict names another backend than the primer's is never forked either — a
session copy cannot cross backends, so it starts fresh on the model it was
given rather than silently coming out as the primer.

## Operational contracts (carried over from v4, all load-bearing)

- One run per issue: a non-blocking flock on `<teams-dir>/<team>.lock`, held
  for the process lifetime; a second run on the same issue exits 4.
- Per issue: a fresh team and a fresh full *clone* at `origin/master`, placed
  at `<team>/worktree` (the path `--team` resolves as the workdir) so
  concurrent runs share no git state. Clone the local repo (hardlinked
  objects), repoint `origin` at the real remote, fetch, detach at
  `origin/master`; delete any stale team and clone from a dead run first.
- The stale-base gate: before the first `make ci` and again after approval,
  fetch and — if `origin/master` is no longer an ancestor of HEAD — merge it
  in. The driver merges and pushes when clean; devin resolves conflicts.
  Unresolved, or master moving again after the one post-approval re-gate, is
  exit 10.
- `make ci` runs under a blocking flock on `<teams-dir>/ci.lock`: one CI at a
  time machine-wide, so parallel runs never fight for cores.
- Run-state vs record: teams and clones die with the run; logs go to
  `~/.agents-army/<repo>/gdw-v5/logs/issue-N/<timestamp>/` — one file per
  agent, `ci-N.log` for the driver's CI runs — and are never deleted.
- Prompts are rendered from `prompts/*.md` substituting ONLY these variables:
  `$issue_url`, `$pr_url`, `$ci_head`, `$ci_log` (use `string.Template`; do
  not substitute anything else — the prose contains other dollar signs).
- The first talk to an agent carries its backend flags. One dict per agent
  holds them (`{"backend": ..., "model": ..., "effort": ...}`), so a single
  line moves an agent to another model: primer/owen `claude opus medium`,
  code-reviewer `claude opus high`, spectacle/devin `codex gpt-5.6-luna max`,
  doku `opencode muse-spark-1.2-contributor-free medium`. The line
  that announces a first talk prints that dict, so the run log says what each
  agent ran on, and every turn is followed by `<agent> worked for N seconds`.
  Later talks resume the session and must not re-send the flags. Devin's first
  build talk also carries
  `-s implement,tdd,code-review-and-quality`; the reviewer's first talk
  carries `-s code-review-and-quality`; skills are never re-sent.
- Before relying on devin's work, verify he committed and pushed: clean
  checkout AND local HEAD == the PR's `headRefOid` (exit 8 otherwise).
- A failed `make ci` gets exactly one devin repair pass, then one re-run.
- A PR still in draft after the self-review is a deliberate stop (exit 6):
  devin found a false assumption and commented on the PR. Do not retry.
- Find the PR with `gh pr list --draft`, matching the issue reference in the
  body or the issue number in the head branch name (a body edit must not
  blind discovery); poll up to 6 × 5 s (read-after-write lag). Never create a
  second PR when one exists.
- Exit codes: the exact table at the bottom of flow.md.

## Coding rules (as binding as the diagram)

The file must be easy to follow with the eyes, top to bottom, and a reader
should be able to predict what the terminal prints. Concretely:

**Allowed, and nothing more:**
- Three tiny helpers, each under ~8 lines: `sh()` (run a command, return
  stdout, raise on failure), `talk()` (one agent turn: render prompt, invoke
  the orchestrator, append to the agent's log), `prompt()` (read a prompt
  file, substitute the explicit variable list).
- Plain data: lists, dicts, sets. At most one frozen dataclass with fields
  only, if bare dicts would need comments to explain their keys.
- Functions named for the diagram's boxes, with the top-to-bottom story at
  the bottom of the file. Reading order = execution order = the diagram.
- Exceptions as the error model: let failures crash with the traceback.
  `try/except` only where the flow genuinely branches on failure (the CI
  repair pass; the fork-unavailable fallback).

**Banned, permanently:**
- Classes with behavior, inheritance, or interfaces of any kind.
- async, threads, or any concurrency — the ordering is the design.
- CLI frameworks, config objects, YAML: one positional argument from
  `sys.argv`, constants at the top of the file.
- Logging frameworks, retry decorators, wrapper layers around `gh`/`git`:
  `print()` narrates, subprocess output goes to the log files.

## Definition of done

- `go.py` implements every node, edge, and exit code in flow.md.
- A dry read of the file against the diagram finds a function per box and no
  code that the diagram cannot explain.
- `python -m py_compile go.py` passes; the driver runs against a test issue
  through planning, and against a converged issue through build.
