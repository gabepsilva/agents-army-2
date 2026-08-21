# Gabriel's development workflow

`simple_development_workflow.py` is the readable entrypoint. It shows only the
main flow:

1. Load the issue.
2. Clarify the request with an expander and an independent griller.
3. Produce a specification.
4. Implement it.
5. Run CI and repair failures.
6. Obtain specification and quality approvals.
7. Commit, push, and create a draft pull request.

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
runs `orchestrator ensure ...` with the configured backend/model/effort, then
runs a schema-validated `orchestrator --agent ... --prompt ...` turn through
`subprocess.run`. The example does not import or construct the Python
`Orchestrator` API. The orchestrator CLI then invokes the selected `claude`,
`codex`, or `grok` backend CLI.

The issue body is sent only for the initial expansion. Later clarification and
specification turns receive the five latest non-workflow comments, preserving
context without repeatedly spending tokens on the entire conversation.

Clarification and review have no configured round limit. Every participant's
structured reply includes `needs_another_round`; the workflow advances only
after every participant in that process reports that no further exchange is
needed. CI repair likewise continues until `make ci` succeeds. Explicit
`stop`, `reject`, and `blocked` outcomes still stop the workflow, and an exact
repeat of the same unresolved state is reported as stalled to prevent a
deadlocked interaction from looping forever.

## Run

Start from a clean feature branch, populate the ignored `workflow.local`, and
then run:

```sh
uv run python examples/gabriels_workflow/simple_development_workflow.py \
  42
```

To use another configuration file, pass `--config path/to/workflow.yaml`.
Without that option, the script loads
`examples/gabriels_workflow/workflow.local`. Model names and reasoning-effort
values are passed through to the configured CLI. Each of `expander`, `griller`,
`specifier`, `implementer`, `reviewer-specification`, and `reviewer-quality`
must be configured explicitly.

Each stage result is commented through that stage role's GitHub App. The
implementer app also loads the issue, posts workflow-level status, and creates
the pull request. Install all six apps on the configured repository with the
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

The required local tools are `git`, `make`, `uv`, `orchestrator`, and every
authenticated agent CLI referenced by the configuration: `claude`, `codex`,
or `grok`.
