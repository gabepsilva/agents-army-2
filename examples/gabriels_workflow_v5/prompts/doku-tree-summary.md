The issue '$issue_url' was split, and every descendant leaf has now been
debated to a decision. Read this issue and each child issue it references
(children may have split further - follow the tree to the leaves), including
the decision brief posted on every converged leaf.

Post one comment on '$issue_url': the plan of record a human reads before
deciding what gets built. Hard budget: 400 words. Cover:

- The tree: every leaf with its issue number and a one-line what-it-delivers,
  in build order, with the dependencies between them stated.
- What the tree as a whole achieves, versus the original ask on this issue.
- Any leaf that was rejected or is blocked, with its one-line reason.
- The open assumptions that span leaves - the ones a later build could
  invalidate - so the reader knows where the plan is still soft.

Decisions and behaviour, not code: no file paths, no function or class names.
Return nothing here, just the comment.
Post as the github app:
app_id: 4577311
private_key: ~/keys/doku-documentation-agent.2026-08-12.private-key.pem
