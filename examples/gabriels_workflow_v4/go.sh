#!/bin/bash

# gdw v4. Owen triages the ask first - proceed, reshape, reject, or split -
# then the debate stays deep on requirements and lazy on code: the code is
# opened only for a claim that is disputed AND would change the decision.
# Everything unproven rides to the PR as an assumptions ledger and the
# developer verifies it in-file, where he is working anyway. The build half
# is go-lean's: the driver owns CI, reviewers read the log, rounds are bounded.
# The flow diagram is flow.md; every agent turn's prompt is a file in prompts/.

aarmy() { uv run orchestrator "$@"; }

dirty=$(git status --porcelain)
if [ -n "$dirty" ]; then
    echo "Uncommitted changes in $(pwd) - commit or stash them first:" >&2
    echo "$dirty" >&2
    exit 4
fi

read -r -p "Enter the issue URL: " issue_url

issue_number=${issue_url##*/}
team="issue-$issue_number"
repo=$(basename "$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")")
export AGENTS_ARMY_TEAMS_DIR="$HOME/.agents-army/$repo/gdw-v4"

worktree="$AGENTS_ARMY_TEAMS_DIR/$team/worktree"
log_dir="$AGENTS_ARMY_TEAMS_DIR/$team/logs"
mkdir -p "$log_dir"

# Prompts live next to this script. The explicit var list keeps envsubst from
# eating any other dollar sign in the prose.
prompt_dir="$(dirname "$(readlink -f "$0")")/prompts"
export issue_url pr_url ci_head ci_log
prompt() { envsubst '$issue_url $pr_url $ci_head $ci_log' < "$prompt_dir/$1.md"; }

issue_labels() { gh issue view "$issue_url" --json labels --jq '.labels[].name'; }
has_label()    { issue_labels | grep -qx "$1"; }
issue_is_blocked()   { issue_labels | grep -qE '^(owens|spectacle)-is-blocked$'; }
issue_is_converged() { has_label owens-is-happy && has_label spectacle-is-happy; }

# Reject, split, and an approved PR are the successful ends; they clean up.
# Failure exits leave the team and logs behind for the post-mortem.
cleanup() {
    aarmy delete --team "$team"
    git worktree remove --force "$worktree" 2>/dev/null
    rm -rf "$AGENTS_ARMY_TEAMS_DIR/$team"
}

echo "Issue $issue_number in $repo. Agent output goes to $log_dir"

echo "Setting up the worktree at $worktree"
git fetch origin --quiet
# A previous run that died before its cleanup leaves this worktree behind; the
# add then fails and the whole workflow runs against the stale checkout.
git worktree remove --force "$worktree" 2>/dev/null
git worktree prune
git worktree add --detach "$worktree" origin/master || exit 5

# The first talk to an agent carries its model flags; later talks resume the
# same session and must not re-send them.
owen_opts=(-b claude -m opus -e medium)
spectacle_opts=(-b claude -m opus -e low)

# --- Phase 1: owen triages the ask -----------------------------------------

echo "Owen: triaging the ask (proceed / reshape / reject / split)."
aarmy talk owen --team "$team" -v "${owen_opts[@]}" \
    -p "$(prompt owen-triage)" &> "$log_dir/owen.log"
owen_opts=()

if has_label owens-split; then
    echo "Owen split the issue. His breakdown:"
    gh issue view "$issue_url" --json comments --jq '.comments[-1].body'
    echo "Run this script once per child issue."
    cleanup
    exit 0
fi

if has_label owens-rejects; then
    echo "Owen rejected the idea. Spectacle: one concur/veto turn."
    aarmy talk spectacle --team "$team" -v "${spectacle_opts[@]}" \
        -p "$(prompt spectacle-reject-review)" &> "$log_dir/spectacle.log"
    spectacle_opts=()

    if [ "$(gh issue view "$issue_url" --json state --jq .state)" = CLOSED ]; then
        echo "Rejection stands: issue closed as not planned. A decision, not a failure."
        cleanup
        exit 0
    fi

    echo "Spectacle vetoed the rejection. Owen: write the proposal."
    aarmy talk owen --team "$team" -v \
        -p "$(prompt owen-proposal-after-veto)" &>> "$log_dir/owen.log"
fi

issue_is_blocked && { echo "Blocked on external evidence. Stopping." >&2; exit 1; }

# --- Phase 2: the debate - deep on requirements, lazy on code ---------------

echo "Spectacle: requirements critique."
aarmy talk spectacle --team "$team" -v "${spectacle_opts[@]}" \
    -p "$(prompt spectacle-critique)" &>> "$log_dir/spectacle.log"
spectacle_opts=()

issue_is_blocked && { echo "Blocked on external evidence. Stopping." >&2; exit 1; }

if ! issue_is_converged; then
    echo "Disputed. Owen: one rebuttal."
    aarmy talk owen --team "$team" -v \
        -p "$(prompt owen-rebuttal)" &>> "$log_dir/owen.log"

    issue_is_blocked && { echo "Blocked on external evidence. Stopping." >&2; exit 1; }

    echo "Spectacle: final decision."
    aarmy talk spectacle --team "$team" -v \
        -p "$(prompt spectacle-final-decision)" &>> "$log_dir/spectacle.log"
fi

issue_is_blocked && { echo "Blocked on the issue. Stopping." >&2; exit 1; }
issue_is_converged || { echo "Issue did not converge. Stopping." >&2; exit 1; }

# --- Handoff: decision brief and the draft PR --------------------------------

echo "Issue converged. Doku: writing the decision brief."
aarmy talk doku --team "$team" -v -b claude -m opus -e medium \
    -p "$(prompt doku-decision-brief)" &> "$log_dir/doku.log"

echo "Spectacle: opening the draft PR."
git -C "$worktree" reset -q --hard
git -C "$worktree" clean -qfd
aarmy talk spectacle --team "$team" -v \
    -p "$(prompt spectacle-draft-pr)" &>> "$log_dir/spectacle.log"

# The list endpoint is read-after-write. Poll briefly rather than paying a
# fixed indexing delay.
pr=""
for _ in {1..6}; do
    pr=$(gh pr list --draft --limit 50 --json url,body,headRefName \
        --jq "[.[] | select(.body | test(\"(#|issues/)${issue_number}\\\\b\"))] | .[0] | select(.) | \"\(.url) \(.headRefName)\"")
    [ -n "$pr" ] && break
    sleep 5
done

read -r pr_url pr_branch <<< "$pr"
[ -z "$pr_url" ] && { echo "No draft PR found for issue $issue_number." >&2; exit 2; }

echo "Draft PR $pr_url on branch $pr_branch"

git -C "$worktree" reset -q --hard
git -C "$worktree" clean -qfd
git -C "$worktree" fetch origin --quiet
git -C "$worktree" checkout -q -B "$pr_branch" "origin/$pr_branch" || exit 5

# --- Phase 3: build - go-lean's back half ------------------------------------

echo "Devin: implementing the PR description."
aarmy talk devin --team "$team" -v --timeout 10800 -b claude -m opus -e low \
    -s implement,tdd,code-review-and-quality \
    -p "$(prompt devin-implement)" &> "$log_dir/devin.log"

echo "Devin: one self-review pass."
aarmy talk devin --team "$team" -v --timeout 10800 \
    -p "$(prompt devin-self-review)" &>> "$log_dir/devin.log"

[ "$(gh pr view "$pr_url" --json isDraft --jq '.isDraft')" = false ] ||
    { echo "PR is still a draft after the self-review - a false assumption stops here; read devin's PR comment." >&2; exit 6; }

# CI is owned by the driver. Reviewers inspect the matching log instead of
# spending an expensive model turn running the same deterministic gates again.
ci_run=0
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
    echo "Running 'make ci' (run $ci_run) on $ci_head - log: $ci_log"
    make -C "$worktree" ci &> "$ci_log"
}

repair_failed_ci_once() {
    echo "CI failed. Devin: one repair pass."
    aarmy talk devin --team "$team" -v --timeout 10800 \
        -p "$(prompt devin-ci-repair)" &>> "$log_dir/devin.log"
    require_committed_and_pushed || return 1
    run_ci
}

require_committed_and_pushed || exit 8
if ! run_ci; then
    repair_failed_ci_once || { echo "CI still failing after the repair pass." >&2; exit 7; }
fi

reviewer_skills=(-s code-review-and-quality)

# At most three concrete review rounds. A developer pushback is valid progress
# even when the branch head does not move, so it is never re-nudged generically.
for review_round in {1..3}; do
    echo "Review round $review_round of 3. Code reviewer: reviewing the diff."
    aarmy talk code-reviewer --team "$team" -v --timeout 10800 -b claude -m opus -e high \
        "${reviewer_skills[@]}" \
        -p "$(prompt code-reviewer-review)" &>> "$log_dir/code-reviewer.log"
    reviewer_skills=()

    gh pr view "$pr_url" --json labels --jq '.labels[].name' | grep -qx 'reviewer-approves' &&
        { echo "Reviewer approved."; break; }
    [ "$review_round" -eq 3 ] && break

    echo "Devin: addressing the review feedback."
    old_head=$(git -C "$worktree" rev-parse HEAD)
    aarmy talk devin --team "$team" -v --timeout 10800 \
        -p "$(prompt devin-review-response)" &>> "$log_dir/devin.log"

    require_committed_and_pushed || exit 8
    new_head=$(git -C "$worktree" rev-parse HEAD)
    if [ "$new_head" != "$old_head" ]; then
        if ! run_ci; then
            repair_failed_ci_once || { echo "CI still failing after the repair pass." >&2; exit 7; }
        fi
    fi
done

gh pr view "$pr_url" --json labels --jq '.labels[].name' | grep -qx 'reviewer-approves' ||
    { echo "PR not approved after 3 review rounds. Stopping." >&2; exit 3; }

echo "Doku: writing the user-facing note on the PR."
aarmy talk doku --team "$team" -v \
    -p "$(prompt doku-user-note)" &>> "$log_dir/doku.log"

echo "Cleaning up the team, worktree, and local branch."
cleanup
git branch -q -D "$pr_branch" 2>/dev/null

echo "Done. $pr_url is approved and ready for merge."
