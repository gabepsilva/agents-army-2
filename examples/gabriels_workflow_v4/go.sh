#!/bin/bash

# gdw v4, two-phase and state-driven. One command, two invocations:
#
#   Planning - an issue with no verdict gets the full treatment: owen triages
#   (proceed / reshape / reject / split), splits recurse into their children,
#   and every leaf is debated to convergence and briefed by doku. The script
#   then dies. The human reads the briefs and decides what gets built.
#
#   Build - called again with a converged leaf (owens-is-happy +
#   spectacle-is-happy), the script jumps straight to development: draft PR
#   (reused if one exists), devin, driver-owned CI, bounded review, doku note.
#
# The issue's labels are the state; calling the script is always safe and
# always does the right next thing. The debate stays deep on requirements and
# lazy on code; unproven claims ride to the PR as an assumptions ledger.
# The flow diagram is flow.md; every agent turn's prompt is a file in prompts/.

aarmy() { uv run orchestrator "$@"; }

dirty=$(git status --porcelain)
if [ -n "$dirty" ]; then
    echo "Uncommitted changes in $(pwd) - commit or stash them first:" >&2
    echo "$dirty" >&2
    exit 4
fi

read -r -p "Enter the issue URL: " issue_url

repo=$(basename "$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")")
repo_slug=$(gh repo view --json nameWithOwner --jq .nameWithOwner)

# Teams and worktrees are run-state and die with the run. Logs are the record
# of what the agents did and cost - they live outside the team dir, by issue,
# one timestamped dir per run, and no cleanup ever touches them.
base="$HOME/.agents-army/$repo/gdw-v4"
export AGENTS_ARMY_TEAMS_DIR="$base/teams"
run_stamp=$(date +%Y%m%dT%H%M%S)

# Prompts live next to this script. The explicit var list keeps envsubst from
# eating any other dollar sign in the prose.
prompt_dir="$(dirname "$(readlink -f "$0")")/prompts"
export issue_url pr_url ci_head ci_log
prompt() { envsubst '$issue_url $pr_url $ci_head $ci_log' < "$prompt_dir/$1.md"; }

issue_labels() { gh issue view "$issue_url" --json labels --jq '.labels[].name'; }
has_label()    { issue_labels | grep -qx "$1"; }
issue_is_blocked()   { issue_labels | grep -qE '^(owens|spectacle)-is-blocked$'; }
issue_is_converged() { has_label owens-is-happy && has_label spectacle-is-happy; }

# Point every per-issue variable at $issue_url and give the issue a fresh
# team and worktree. A team or worktree left by a dead run would silently
# resume stale sessions or a stale checkout - fresh is the only safe start.
enter_issue() {
    issue_number=${issue_url##*/}
    team="issue-$issue_number"
    worktree="$AGENTS_ARMY_TEAMS_DIR/$team/worktree"
    log_dir="$base/logs/$team/$run_stamp"
    mkdir -p "$log_dir"
    echo "Issue $issue_number - agent output goes to $log_dir"

    git worktree remove --force "$worktree" 2>/dev/null
    git worktree prune
    git worktree add --detach "$worktree" origin/master || exit 5
    aarmy delete --team "$team" 2>/dev/null
}

# Run-state cleanup for one issue. Logs survive; failure paths skip this so
# the team is still there for the post-mortem.
leave_issue() {
    aarmy delete --team "$team"
    git worktree remove --force "$worktree" 2>/dev/null
    rm -rf "${AGENTS_ARMY_TEAMS_DIR:?}/$team"
}

git fetch origin --quiet

# --- The router: labels decide the phase -------------------------------------

if issue_is_blocked; then
    echo "Issue is labelled blocked. Resolve the missing evidence first." >&2
    exit 1
fi

if issue_is_converged; then
    phase=build
else
    phase=plan
fi

# --- Planning: recurse the tree, debate every leaf, brief every leaf ---------

if [ "$phase" = plan ]; then

root_url="$issue_url"
queue=("$issue_url")
converged=() blocked=() rejected=() splits=()
max_issues=12
processed=0

while [ ${#queue[@]} -gt 0 ]; do
    issue_url="${queue[0]}"; queue=("${queue[@]:1}")

    processed=$((processed + 1))
    if [ $processed -gt $max_issues ]; then
        echo "Planned $max_issues issues and the tree is still growing - stopping for a human look." >&2
        echo "Still queued: ${queue[*]}" >&2
        exit 9
    fi

    # A child that already converged in an earlier run needs no new debate.
    issue_is_converged && { converged+=("$issue_url"); continue; }

    enter_issue

    echo "Owen: triaging $issue_url (proceed / reshape / reject / split)."
    aarmy talk owen --team "$team" -v -b claude -m opus -e medium \
        -p "$(prompt owen-triage)" &> "$log_dir/owen.log"

    if has_label owens-split; then
        # Owen ends his breakdown comment with "CHILDREN: #a #b" in build
        # order; that line is the machine-readable half of the split.
        children=$(gh issue view "$issue_url" --json comments \
            --jq '.comments[-1].body' | grep -oP '^CHILDREN:\K.*' | grep -oE '[0-9]+')
        if [ -z "$children" ]; then
            echo "Split with no CHILDREN line on $issue_url - human look needed." >&2
            exit 9
        fi
        echo "Split into: $(echo $children | sed 's/[0-9]\+/#&/g'). Queueing them."
        for c in $children; do queue+=("https://github.com/$repo_slug/issues/$c"); done
        splits+=("$issue_url")
        leave_issue
        continue
    fi

    spectacle_opts=(-b claude -m opus -e low)

    if has_label owens-rejects; then
        echo "Owen rejected the idea. Spectacle: one concur/veto turn."
        aarmy talk spectacle --team "$team" -v "${spectacle_opts[@]}" \
            -p "$(prompt spectacle-reject-review)" &> "$log_dir/spectacle.log"
        spectacle_opts=()

        if [ "$(gh issue view "$issue_url" --json state --jq .state)" = CLOSED ]; then
            echo "Rejection stands: issue closed as not planned."
            rejected+=("$issue_url")
            leave_issue
            continue
        fi

        echo "Spectacle vetoed the rejection. Owen: write the proposal."
        aarmy talk owen --team "$team" -v \
            -p "$(prompt owen-proposal-after-veto)" &>> "$log_dir/owen.log"
    fi

    if ! issue_is_blocked; then
        echo "Spectacle: requirements critique."
        aarmy talk spectacle --team "$team" -v "${spectacle_opts[@]}" \
            -p "$(prompt spectacle-critique)" &>> "$log_dir/spectacle.log"

        if ! issue_is_converged && ! issue_is_blocked; then
            echo "Disputed. Owen: one rebuttal."
            aarmy talk owen --team "$team" -v \
                -p "$(prompt owen-rebuttal)" &>> "$log_dir/owen.log"

            if ! issue_is_blocked; then
                echo "Spectacle: final decision."
                aarmy talk spectacle --team "$team" -v \
                    -p "$(prompt spectacle-final-decision)" &>> "$log_dir/spectacle.log"
            fi
        fi
    fi

    if ! issue_is_converged; then
        echo "$issue_url did not converge - leaving its team for the post-mortem."
        blocked+=("$issue_url")
        continue
    fi

    echo "Converged. Doku: writing the decision brief."
    aarmy talk doku --team "$team" -v -b claude -m opus -e medium \
        -p "$(prompt doku-decision-brief)" &> "$log_dir/doku.log"

    converged+=("$issue_url")
    leave_issue
done

# One comment on the root ties the whole tree together: the plan the human
# reads before deciding what gets built, and in which order.
if [ ${#splits[@]} -gt 0 ]; then
    issue_url="$root_url"
    enter_issue
    echo "Doku: writing the tree summary on the root issue."
    aarmy talk doku --team "$team" -v -b claude -m opus -e medium \
        -p "$(prompt doku-tree-summary)" &> "$log_dir/doku.log"
    leave_issue
fi

echo
echo "Planning done."
[ ${#converged[@]} -gt 0 ] && echo "  Ready to build (call again per leaf, in CHILDREN order): ${converged[*]}"
[ ${#rejected[@]}  -gt 0 ] && echo "  Rejected and closed: ${rejected[*]}"
[ ${#blocked[@]}   -gt 0 ] && { echo "  Blocked or unconverged: ${blocked[*]}"; exit 1; }
exit 0

fi

# --- Build: this leaf is converged; develop it --------------------------------

enter_issue

# Reuse a draft PR left by an earlier build attempt; open one only if none
# exists. The list endpoint is read-after-write - poll briefly on creation.
find_pr() {
    gh pr list --draft --limit 50 --json url,body,headRefName \
        --jq "[.[] | select(.body | test(\"(#|issues/)${issue_number}\\\\b\"))] | .[0] | select(.) | \"\(.url) \(.headRefName)\""
}

pr=$(find_pr)
if [ -z "$pr" ]; then
    echo "Spectacle: opening the draft PR."
    aarmy talk spectacle --team "$team" -v -b claude -m opus -e low \
        -p "$(prompt spectacle-draft-pr)" &> "$log_dir/spectacle.log"
    for _ in {1..6}; do
        pr=$(find_pr)
        [ -n "$pr" ] && break
        sleep 5
    done
fi

read -r pr_url pr_branch <<< "$pr"
[ -z "$pr_url" ] && { echo "No draft PR found for issue $issue_number." >&2; exit 2; }

echo "Draft PR $pr_url on branch $pr_branch"

git -C "$worktree" fetch origin --quiet
git -C "$worktree" checkout -q -B "$pr_branch" "origin/$pr_branch" || exit 5

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
aarmy talk doku --team "$team" -v -b claude -m opus -e low \
    -p "$(prompt doku-user-note)" &> "$log_dir/doku.log"

echo "Cleaning up the team, worktree, and local branch."
leave_issue
git branch -q -D "$pr_branch" 2>/dev/null

echo "Done. $pr_url is approved and ready for merge. Logs kept in $log_dir"
