The issue '$issue_url' has converged. Open a draft PR against the default
branch with an empty commit and no file changes. Its description is the
complete developer handoff. Keep it concise: decided behavior, affected areas,
acceptance criteria, verification, and evidence only for non-obvious
load-bearing decisions.

End the description with an 'Assumptions' section: every claim the debate left
unverified, each phrased so the developer can check it in the code while
implementing. State there that if an assumption turns out false, the developer
stops and comments on the PR instead of improvising around it.

Do not reproduce the debate or rejected alternatives unless omitting one would
cause a likely implementation mistake. Resolve any remaining ambiguity
yourself. Link the issue.
Post as the github app:
app_id: 4287312
private_key: ~/keys/ai-specialist-reviewer.2026-08-04.private-key.pem
