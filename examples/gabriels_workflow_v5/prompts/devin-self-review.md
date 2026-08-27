Perform your one final self-review of '$pr_url' across correctness, tests,
maintainability, security, and scope. Inspect the actual diff with fresh
eyes. Fix, commit, and push anything you find. Run required commands in the
foreground. Never background work and poll for it. If you must wait on a
process, wait on its PID - never test for one by matching text in `ps` or
`pgrep` output, because the pattern matches your own polling command and the
wait never ends.

When the code is something you stand behind, mark the PR ready for review. If
you stopped over a false assumption, leave the PR as draft - your PR comment
is what the human reads. This is the only generic self-review turn.
