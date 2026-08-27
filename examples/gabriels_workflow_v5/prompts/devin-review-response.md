Read the latest review feedback on '$pr_url'. Address each blocking finding
exactly once: fix it, or push back with re-checkable code facts. Optional
findings are your judgment. Reply on the PR explaining the decision. If code
changes, commit and push them. Finish foreground commands before returning.
Never background work and poll for it. If you must wait on a process, wait
on its PID - never test for one by matching text in `ps` or `pgrep` output,
because the pattern matches your own polling command and the wait never
ends. Do not perform a generic self-review.
Post as the github app:
app_id: 4579193
private_key: ~/keys/devin-development-specialist.2026-08-13.private-key.pem
