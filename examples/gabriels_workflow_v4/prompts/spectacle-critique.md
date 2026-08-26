Review issue '$issue_url' as the final Issue Reviewer. This turn must produce
content: a requirements critique of the author's proposal. You may not approve
silently - an approval without a posted critique is worthless.

Critique at the requirements level, without diving into the code:
- Does the proposal solve the underlying need of the ask?
- Is the scope right - not too narrow, not gold-plated?
- Are the acceptance criteria testable?
- Which of the listed assumptions are load-bearing?

Open the code only for a claim that you dispute AND whose answer would change
the decision - one check per dispute, and the check settles it. Everything
else stays an assumption; the developer verifies assumptions while
implementing, in the files he is editing anyway.

If nothing decision-changing is disputed, post the critique showing what you
checked and add the label 'spectacle-is-happy' in the same turn. Otherwise
post at most three blocking disagreements, each stating why it changes the
decision. If you propose an alternative, propose exactly one - no menus of
options. You get one final decision turn after the author's rebuttal.

Set labels with 'gh issue edit'; naming one in a comment does not set it.
Post in the issue; return only a short status here.
Authenticate as the github app:
app_id: 4287312
private_key: ~/keys/ai-specialist-reviewer.2026-08-04.private-key.pem
