#!/bin/bash

# A bounded variant of go.sh for lower wall time and cost per resolved issue.
# The original workflow remains in go.sh for comparison.

aarmy() { uv run orchestrator "$@"; }

dirty=$(git status --porcelain)
if [ -n "$dirty" ]; then
    echo "Uncommitted changes in $(pwd) - commit or stash them first:" >&2
    echo "$dirty" >&2
    exit 4
fi

read -r -p "Enter the issue URL: " issue_url
export issue_url

issue_number=${issue_url##*/}
team="issue-$issue_number"
repo=$(basename "$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")")
export AGENTS_ARMY_TEAMS_DIR="$HOME/.agents-army/$repo/gdw-v3"

worktree="$AGENTS_ARMY_TEAMS_DIR/$team/worktree"
log_dir="$AGENTS_ARMY_TEAMS_DIR/$team/logs"
mkdir -p "$log_dir"

issue_labels() {
    gh issue view "$issue_url" --json labels --jq '.labels[].name'
}

issue_is_blocked() {
    grep -qE '^(owens|spectacle)-is-blocked$' <<< "$(issue_labels)"
}

issue_is_converged() {
    local labels
    labels=$(issue_labels)
    grep -qx 'owens-is-happy' <<< "$labels" &&
        grep -qx 'spectacle-is-happy' <<< "$labels"
}

git fetch origin --quiet
git worktree prune
git worktree add --detach "$worktree" origin/master

# Normal path: one compact proposal and one adversarial decision. The author
# gets one rebuttal and the reviewer one final decision only if they disagree.
aarmy talk owen --team "$team" -b claude -m opus -e medium -v \
    -p "Look at issue '$issue_url' as its Issue Author and inspect the relevant code. \
    Write one compact proposal in the GitHub issue: the behavior to change, affected areas, \
    acceptance criteria, and at most three load-bearing risks or decisions. \
    Check claims that could change scope, safety, implementation, or verification. \
    Do not investigate alternatives once they cannot change the decision. \
    Do not maintain a design tree and do not restate the issue. \
    If the proposal is implementable as written, add the label 'owens-is-happy' to the issue. \
    Add the label 'owens-is-blocked' to the issue only when missing external evidence \
    prevents a decision. A label is set with 'gh issue edit', not by naming it in a comment. \
    All communication goes in the GitHub issue; return only a short status here. \
    Post as the github app: \
    app_id: 4578638 \
    private_key: ~/keys/owen-project-owner.2026-08-13.private-key.pem" &> "$log_dir/owen.log"

aarmy talk spectacle --team "$team" -b claude -m opus -e low -v \
    -p "Review issue '$issue_url' as the final Issue Reviewer. Read the issue and relevant code. \
    Challenge only load-bearing claims that could change scope, safety, implementation, or verification. \
    Do not restate the proposal or maintain a design tree. \
    If it is implementable, add the label 'spectacle-is-happy' to the issue with \
    'gh issue edit'; naming it in a comment does not set it. \
    Otherwise post at most three blocking disagreements, each with re-checkable evidence. \
    Do not post optional questions. You get one final decision turn after the author's rebuttal. \
    All communication goes in the GitHub issue; return only a short status here. \
    Post as the github app: \
    app_id: 4287312 \
    private_key: ~/keys/ai-specialist-reviewer.2026-08-04.private-key.pem" &> "$log_dir/spectacle.log"

issue_is_blocked && exit 1

if ! issue_is_converged; then
    aarmy talk owen --team "$team" -v \
        -p "This is your only rebuttal. Read the reviewer's blocking disagreements on '$issue_url'. \
        Answer only those points with re-checkable evidence and state your final position. \
        Add no optional scope or new alternatives. Add the label 'owens-is-happy' to the issue if \
        the reviewer can now make the final implementation decision, or 'owens-is-blocked' only if \
        external evidence is genuinely missing. Set it with 'gh issue edit'; naming a label in a \
        comment does not set it. Post in the issue; return only a short status here." &>> "$log_dir/owen.log"

    issue_is_blocked && exit 1

    aarmy talk spectacle --team "$team" -v \
        -p "Make the final decision on '$issue_url'. Read the author's rebuttal and resolve every \
        remaining point from the available evidence. Ask no further questions and offer no options. \
        Add the label 'spectacle-is-happy' to the issue if there is now one implementable outcome; \
        otherwise add the label 'spectacle-is-blocked'. Set it with 'gh issue edit'; naming a label \
        in a comment does not set it. Post the concise decision in the issue and return only a \
        short status here." \
        &>> "$log_dir/spectacle.log"
fi

issue_is_blocked && exit 1
issue_is_converged || exit 1

echo "Issue converged. Opening draft PR."

git -C "$worktree" reset -q --hard
git -C "$worktree" clean -qfd

aarmy talk spectacle --team "$team" -v \
    -p "The issue '$issue_url' has converged. Open a draft PR against the default branch with an empty \
    commit and no file changes. Its description is the complete developer handoff. Keep it concise: \
    decided behavior, affected areas, acceptance criteria, verification, and evidence only for \
    non-obvious load-bearing decisions. Do not reproduce the debate or rejected alternatives unless \
    omitting one would cause a likely implementation mistake. Resolve any remaining ambiguity yourself. \
    Link the issue. Post as the github app: \
    app_id: 4287312 \
    private_key: ~/keys/ai-specialist-reviewer.2026-08-04.private-key.pem" &>> "$log_dir/spectacle.log"

# The list endpoint is read-after-write. Poll briefly rather than paying a
# fixed 30-second indexing delay.
pr=""
for _ in {1..6}; do
    pr=$(gh pr list --draft --limit 50 --json url,body,headRefName \
        --jq "[.[] | select(.body | test(\"(#|issues/)${issue_number}\\\\b\"))] | .[0] | select(.) | \"\(.url) \(.headRefName)\"")
    [ -n "$pr" ] && break
    sleep 5
done

read -r pr_url pr_branch <<< "$pr"
export pr_url
[ -z "$pr_url" ] && exit 2

git -C "$worktree" reset -q --hard
git -C "$worktree" clean -qfd
git -C "$worktree" fetch origin --quiet
git -C "$worktree" checkout -q -B "$pr_branch" "origin/$pr_branch" || exit 5

aarmy talk devin --team "$team" -v --timeout 10800 -b claude -m opus -e low \
    -s implement,tdd,code-review-and-quality \
    -p "Implement the complete description of '$pr_url' as a lead software engineer. \
    Do not read the linked issue or its comments. Use TDD, keep the design maintainable, and push the \
    implementation to the PR branch. Run commands in the foreground and do not return while tests or \
    gates are still running. Leave the PR as draft: one separate self-review follows. \
    Post as the github app: \
    app_id: 4579193 \
    private_key: ~/keys/devin-development-specialist.2026-08-13.private-key.pem" &> "$log_dir/devin.log"

# Exactly one fresh self-review pass. This preserves the useful independent
# inspection seen in issue #96 without an open-ended generic loop.
aarmy talk devin --team "$team" -v --timeout 10800 \
    -p "Perform your one final self-review of '$pr_url' across correctness, tests, maintainability, \
    security, and scope. Inspect the actual diff with fresh eyes. Fix, commit, and push anything you \
    find. Run required commands in the foreground. When the code is something you stand behind, mark \
    the PR ready for review. This is the only generic self-review turn." &>> "$log_dir/devin.log"

[ "$(gh pr view "$pr_url" --json isDraft --jq '.isDraft')" = false ] || exit 6

# CI is owned by the driver. Reviewers inspect the matching log instead of
# spending an expensive model turn running the same deterministic gates again.
ci_run=0
ci_log=""
ci_head=""
require_committed_and_pushed() {
    local dirty local_head remote_head
    dirty=$(git -C "$worktree" status --porcelain)
    if [ -n "$dirty" ]; then
        echo "Developer left uncommitted work in $worktree:" >&2
        echo "$dirty" >&2
        return 1
    fi

    local_head=$(git -C "$worktree" rev-parse HEAD)
    remote_head=$(gh pr view "$pr_url" --json headRefOid --jq '.headRefOid')
    if [ "$local_head" != "$remote_head" ]; then
        echo "Developer HEAD $local_head was not pushed to PR head $remote_head" >&2
        return 1
    fi
}

run_ci() {
    ci_run=$((ci_run + 1))
    ci_log="$log_dir/ci-$ci_run.log"
    ci_head=$(git -C "$worktree" rev-parse HEAD)
    make -C "$worktree" ci &> "$ci_log"
}

repair_failed_ci_once() {
    aarmy talk devin --team "$team" -v --timeout 10800 \
        -p "The driver ran 'make ci' on '$pr_url' and it failed. Read the complete log at '$ci_log'. \
        Fix the actual failure, run focused checks in the foreground, commit, and push. Do not broaden \
        scope. Return only after the branch contains the fix." &>> "$log_dir/devin.log"
    require_committed_and_pushed || return 1
    run_ci
}

require_committed_and_pushed || exit 8
if ! run_ci; then
    repair_failed_ci_once || exit 7
fi

reviewer_skills=(-s code-review-and-quality)

# At most three concrete review rounds. A developer pushback is valid progress
# even when the branch head does not move, so it is never re-nudged generically.
for review_round in {1..3}; do
    aarmy talk code-reviewer --team "$team" -v --timeout 10800 -b claude -m opus -e high \
        "${reviewer_skills[@]}" \
        -p "Review PR '$pr_url'. Its description is the complete spec. Review the current diff against \
        it and the five quality axes. The driver has already run 'make ci' successfully on commit \
        '$ci_head'; the full log is '$ci_log'. Confirm HEAD matches before relying on it, and do not rerun \
        the full gate without concrete evidence that the logged result is insufficient. \
        Report only re-checkable findings, clearly separating blockers from optional observations. \
        Do not change code. If nothing blocks merging, add the label 'reviewer-approves' to the PR \
        with 'gh pr edit'. Writing the label name into your review comment does not set it, and \
        the driver reads the label rather than your reply. \
        Post as the github app: \
        app_id: 4287312 \
        private_key: ~/keys/ai-specialist-reviewer.2026-08-04.private-key.pem" \
        &>> "$log_dir/code-reviewer.log"

    reviewer_skills=()

    gh pr view "$pr_url" --json labels --jq '.labels[].name' | grep -qx 'reviewer-approves' && break
    [ "$review_round" -eq 3 ] && break

    old_head=$(git -C "$worktree" rev-parse HEAD)
    aarmy talk devin --team "$team" -v --timeout 10800 \
        -p "Read the latest review feedback on '$pr_url'. Address each blocking finding exactly once: \
        fix it, or push back with re-checkable code facts. Optional findings are your judgment. Reply on \
        the PR explaining the decision. If code changes, commit and push them. Finish foreground commands \
        before returning. Do not perform a generic self-review. \
        Post as the github app: \
        app_id: 4579193 \
        private_key: ~/keys/devin-development-specialist.2026-08-13.private-key.pem" &>> "$log_dir/devin.log"

    require_committed_and_pushed || exit 8
    new_head=$(git -C "$worktree" rev-parse HEAD)
    if [ "$new_head" != "$old_head" ]; then
        if ! run_ci; then
            repair_failed_ci_once || exit 7
        fi
    fi
done

gh pr view "$pr_url" --json labels --jq '.labels[].name' | grep -qx 'reviewer-approves' || exit 3

aarmy talk doku --team "$team" -v -b claude -m opus -e low \
    -p "Write one concise user-facing comment for '$pr_url' from its description and diff. \
    Explain what changes, how to use it with a real command when applicable, required migration, and \
    surprises. If it is invisible to users, say so in one line. Every claim must come from the diff. \
    Change no files and perform no review. Post as the github app: \
    app_id: 4577311 \
    private_key: ~/keys/doku-documentation-agent.2026-08-12.private-key.pem" &> "$log_dir/doku.log"

aarmy delete --team "$team"
git worktree remove --force "$worktree"
git branch -q -D "$pr_branch"
rm -rf "$AGENTS_ARMY_TEAMS_DIR/$team"
