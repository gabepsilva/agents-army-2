Review PR '$pr_url'. Its description is the complete spec. Review the current
diff against it and the five quality axes.

The driver has already run 'make ci' successfully on commit '$ci_head'; the
full log is '$ci_log'. Confirm HEAD matches before relying on it, and do not
rerun the full gate without concrete evidence that the logged result is
insufficient.

The description ends with an Assumptions section; the developer was told to
verify each one in-file. Spot-check that the load-bearing ones actually hold
in the diff.

Report only re-checkable findings, clearly separating blockers from optional
observations. Do not change code. If nothing blocks merging, add the label
'reviewer-approves' to the PR with 'gh pr edit'. Writing the label name into
your review comment does not set it, and the driver reads the label rather
than your reply.
Post as the github app:
app_id: 4287312
private_key: ~/keys/ai-specialist-reviewer.2026-08-04.private-key.pem
