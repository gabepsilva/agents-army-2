The issue '$issue_url' has converged. Open a draft PR against the default
branch with an empty commit and no file changes, on a branch named exactly
'codex/issue-<N>-handoff' where <N> is that issue's number - the driver finds
the PR by that name and by nothing else. Its description is the complete
developer handoff. Keep it concise: decided behavior, affected areas,
acceptance criteria, verification, and evidence only for non-obvious
load-bearing decisions.

Describe the work, never the current state of the branch or the tree: what you
observe now may already be false when the developer reads it. Bake no measured
number into the spec or the acceptance criteria - no test counts, line numbers,
or file sizes. Where a baseline matters, tell the developer to measure it
himself first and preserve it.

End the description with an 'Assumptions' section: every claim the debate left
unverified, each phrased so the developer can check it in the code while
implementing - design assumptions only, never a reading of the current tree.
State there that if an assumption turns out false, the developer stops and
comments on the PR instead of improvising around it.

Do not reproduce the debate or rejected alternatives unless omitting one would
cause a likely implementation mistake. Resolve any remaining ambiguity
yourself. Link the issue.
Post as the github app:
app_id: 4287312
private_key: ~/keys/ai-specialist-reviewer.2026-08-04.private-key.pem
