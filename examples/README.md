# Gabriel's development workflow

`gabriel_development_workflow.py` is an end-to-end example built on the
orchestrator's Python API. It turns a raw GitHub issue into a draft pull request:

1. The driver fetches the issue with `gh`.
2. An expander checks the request against the repository.
3. An independent griller requests revisions until the proposal is unambiguous.
4. A specifier produces the implementation contract.
5. An implementer changes code and runs focused tests.
6. The driver runs `make ci` and returns failures to the implementer.
7. Independent specification and quality reviewers inspect the result.
8. The driver commits, pushes, and opens the pull request.

Every agent response is constrained by one of the strict JSON Schemas in
`validations/`. Prompt text lives in `prompts/`. JSON remains the internal
validation and checkpoint format, while the driver renders every validated
response as readable Markdown before posting it to GitHub. An idempotency marker
prevents duplicate comments when a run resumes.

## GitHub boundary

Agents do not fetch issues, post comments, push, or create pull requests. During
each agent turn the driver removes GitHub token environment variables, points
`GH_CONFIG_DIR` at an empty temporary directory, and puts a rejecting `gh`
executable first on `PATH`. The real absolute `gh` path and the original
credential environment are retained only by the driver.

This is defense in depth, not a hostile-code sandbox: an agent can inspect and
modify the checkout as required to implement the issue. Run the workflow only in
a repository and environment where the configured coding-agent CLI is trusted.

## Run

Start from a clean feature branch. The script refuses the repository's default
branch and refuses a dirty worktree on the first run.

```sh
uv run python examples/gabriel_development_workflow.py 42 --backend codex
```

The default output is a draft PR. Pass `--ready` to create a non-draft PR.
Use `--repo OWNER/REPO` when the checkout's GitHub repository cannot be inferred
by `gh`.

Workflow and agent session state is stored under `.git/gdw/issue-<number>/`, so
rerunning the same command resumes completed stages without adding duplicate
GitHub comments. Agent names are issue-scoped (`gdw-42-expander`,
`gdw-42-implementer`, and so on).

The required local tools are:

- an authenticated `gh` CLI;
- `git`, `make`, and `uv`;
- at least one authenticated agent CLI supported by this repository.

Useful controls:

```text
--base BRANCH
--clarification-rounds N
--repair-rounds N
--review-rounds N
--ready
```
