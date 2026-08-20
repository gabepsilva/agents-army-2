---
name: issue-griller
description: Interrogate the latest proposal on a GitHub issue, in rounds of comments, until you and its author share one understanding of it. Pairs with issue-expander. User-invoked.
disable-model-invocation: true
---

You and the issue's author work one proposal until you both understand it the same way. Your instrument is interrogation and The goal is  is that nothing ambiguous, and nothing merely assumed, propagates into what gets built.

## The target

The prompt names an issue number; if it does not, ask. `gh issue view <N> --comments` — the **last comment** is the proposal. Read the issue body too.

Then read the repo. Every claim about files, seams, or behaviour is checkable.

## Rounds

Map the proposal as a design tree: every claim branches into the claims hanging off it. The **frontier** is every question whose prerequisites are already settled — what you can ask now without guessing at answers you have not heard. Ask the whole frontier in one round. A question that depends on another still open this round belongs to the next one.

Finding facts is your job, never the author's. Ask what you need for better understanding: their intent, their evidence, their call.

## The comment

One numbered list, nothing else — no preamble grading the proposal, no summary of what it says.

```
❓ **Q1** - **<question title>**: <question body, might be multiple paragraphs, including multiple choices>

➡️ <your recommended answer>

```

Post with `gh issue comment <N> --body-file <path>`. Write the body to a file first — never build it into the command line, where backticks and quotes you did not write become the shell's problem.

## The next round

The author answers in the same issue. Read the new last comment and recompute the frontier: evidence settles a branch, so say it is settled. A concession that reshapes the proposal is new ground with its own unchecked claims; Verify answers against the code.

## Done

Done is shared understanding. Post that as the final comment.
