The driver ran 'make ci' on '$pr_url' and it failed. Read the complete log
at '$ci_log'. Fix the actual failure, run focused checks in the foreground,
commit, and push. Never background work and poll for it. If you must wait on
a process, wait on its PID - never test for one by matching text in `ps` or
`pgrep` output, because the pattern matches your own polling command and the
wait never ends. Do not broaden scope. Return only after the branch contains
the fix.
