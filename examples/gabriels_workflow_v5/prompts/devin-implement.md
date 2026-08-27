Implement the complete description of '$pr_url' as a lead software engineer.
Do not read the linked issue or its comments - the description is the whole
spec.

The description ends with an Assumptions section. Verify each assumption as
you touch its area - you are in those files anyway. If one turns out false,
do not improvise around it: comment on the PR saying what is false and what
you recommend, stop, and leave the PR as draft.

Use TDD, keep the design maintainable, and push the implementation to the PR
branch. Commit and push in small increments as you go - one coherent step
per commit, pushed as soon as its tests pass - rather than one big-bang
commit at the end. Run every command in the foreground and let it finish;
never put work in the background and poll for it. If you must wait on a
process, wait on its PID - never test for one by matching text in `ps` or
`pgrep` output, because the pattern matches your own polling command and the
wait never ends. Leave the PR as draft: one separate self-review follows.
Post as the github app:
app_id: 4579193
private_key: ~/keys/devin-development-specialist.2026-08-13.private-key.pem
