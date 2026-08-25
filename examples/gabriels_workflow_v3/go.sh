#!/bin/bash

aarmy() { uv run orchestrator "$@"; }


# Input Issue URL, Github

read -p "Enter the issue URL: " issue_url

export issue_url

issue_number=${issue_url##*/}
team="issue-$issue_number"
# Outside the repository, not under .git/. Nested, the driver's own checkout is
# a parent directory of the agents' cwd, and an agent that wanders up into it
# is editing the code this script is running from, mid-run: the #80 run had
# devin editing seven files there and finding the live run's logs and state.
export AGENTS_ARMY_TEAMS_DIR="$HOME/.agents-army/gdw-v3"

# The logs go next to the team, not into the repo. Redirects are relative to
# whatever directory the run was launched from, so a run started at the
# repository root used to drop every agent's log there, outside the path-scoped
# ignore rule that was meant to catch them. .gitignore now ignores *.log
# repo-wide as a backstop; this keeps them out of the tree in the first place.
log_dir="$AGENTS_ARMY_TEAMS_DIR/$team/logs"
mkdir -p "$log_dir"

git fetch origin --quiet
git worktree prune
git worktree add -B "gdwv3/$team" "$AGENTS_ARMY_TEAMS_DIR/$team/worktree" origin/master

aarmy talk owen --team "$team" -b claude -m opus -e medium -v \
    -p "Look issue '$issue_url' \
    You are now taking the role of the Issue Author. State that in you first message.\
    You are also responsible for the decissions from now on, there will be no human in the loop. \
    You'll explore the human's idea. It can actually be a bad idea, so you can propose a better idea or reject it entirely. \
    Develop the idea into something concrete - what would actually change, in which files, in what order. \
    Go explore relevant files before proposing anything; \
    A reviewer will read your proposeal and also read the code. \
    Be ready to be challenged. When the evidence doesn't support you, concede and say so plainly. \
    Your job is to find a good outcome for the project, not a surviving proposal. \
    Goal: Converge on an implementable issue or drop the idea. Must be fact-based recommendations. \
    Every decision argued should be actually checked. \
    All communication should be in the GitHub issue. Dont send any output to this prompt \
    When you think you have nothing to add just add one of these 3 labels to the issue:
    - "owens-is-blocked"
    - "owens-is-happy"
    Post as the github app: \
    app_id: 4578638 \
    private_key: ~/keys/owen-project-owner.2026-08-13.private-key.pem" &> "$log_dir/owen.log"


# Same relative-path trap as the logs, but this one breaks the run rather
# than littering: launched from anywhere but this directory, envsubst reads
# no template and spectacle starts with an empty prompt file. Resolve the
# template next to the script, and render it next to the team.
envsubst < "$(dirname "$(readlink -f "$0")")/spectacle.prompt.tmpl" \
    > "$AGENTS_ARMY_TEAMS_DIR/$team/spectacle.prompt"
aarmy talk spectacle --team "$team" -b claude -m opus -e low -v \
    --prompt-file "$AGENTS_ARMY_TEAMS_DIR/$team/spectacle.prompt" &> "$log_dir/spectacle.log"


# Both agents stop by labelling the issue. Wake whoever has not labelled yet,
# owen first so spectacle always reads the newest comments. talk blocks on the
# agent's lock, so a nudge sent mid-turn just waits its turn.
for i in {1..10}; do
    labels=$(gh issue view $issue_url --json labels --jq '.labels[].name')

    grep -qE '^owens-is-(blocked|happy)$' <<< "$labels"; owen_done=$?
    grep -qE '^spectacle-is-(blocked|happy)$' <<< "$labels"; spectacle_done=$?

    [ $owen_done = 0 ] && [ $spectacle_done = 0 ] && break

    [ $owen_done = 0 ] || aarmy talk owen --team "$team" -v -p 'There are new comments. Read the issue and continue. Label the issue when you have nothing to add.' &>> "$log_dir/owen.log"
    [ $spectacle_done = 0 ] || aarmy talk spectacle --team "$team" -v -p 'There are new comments. Read the issue and continue. Label the issue when you have nothing to add.' &>> "$log_dir/spectacle.log"
done

# Only two happy labels mean the issue converged. A blocked label, or a missing
# one because the loop ran out of rounds, is not something devin can implement.
labels=$(gh issue view $issue_url --json labels --jq '.labels[].name')
grep -qE '^(owens|spectacle)-is-blocked$' <<< "$labels" && exit 1
grep -qx 'owens-is-happy' <<< "$labels" || exit 1
grep -qx 'spectacle-is-happy' <<< "$labels" || exit 1

echo "Issue converged. Opening draft PR."

# owen and spectacle share this worktree with devin and use it as scratch to
# check their claims. `git commit --allow-empty` commits a staged index, so
# whatever they left behind would land in the PR as somebody else's work.
git -C "$AGENTS_ARMY_TEAMS_DIR/$team/worktree" reset -q --hard
git -C "$AGENTS_ARMY_TEAMS_DIR/$team/worktree" clean -qfd


# The draft PR description is the whole handoff: the developer implements from
# it and never opens the issue, so nothing may be left implicit or unresolved.
aarmy talk spectacle --team "$team" -v -p "\
    The issue '$issue_url' has converged. Open a draft PR against the default branch and write its description. \
    Change no files - an empty commit on a new branch is all the PR needs to exist. The description is the deliverable. \
    Link the PR to the issue, but write the description so it stands completely on its own: \
    the developer will be told to implement the PR and never to read the issue or its comments. \
    Write the full converged spec: what changes, and how to verify it. \
    Carry over the evidence behind each decision - the paths, commands and outputs the tree cites - not just the verdicts. \
    If any ambiguity is still left, make the executive decision that is best for the project and state it as decided, \
    with your reasoning. Do not hand the developer an open question or a list of options. \
    Assume a mid-level developer: no context on this discussion, capable of the work once the target is unmistakable. \
    Post as the github app: \
    app_id: 4287312 \
    private_key: ~/keys/ai-specialist-reviewer.2026-08-04.private-key.pem" &>> "$log_dir/spectacle.log"



# The draft PR spectacle just wrote, found by the issue number it links back to.
# --search hits GitHub's *indexed* search, which lags PR creation by up to
# minutes and would report no PR for one spectacle just opened. The list API
# is read-your-writes, so filter its bodies here instead.
export pr_url=$(gh pr list --draft --limit 50 --json url,body \
    --jq "[.[] | select(.body | test(\"(#|issues/)${issue_number}\\\\b\"))] | .[0].url // empty")
[ -z "$pr_url" ] && exit 2

# Same again: devin implements from the PR description, on a clean tree.
git -C "$AGENTS_ARMY_TEAMS_DIR/$team/worktree" reset -q --hard
git -C "$AGENTS_ARMY_TEAMS_DIR/$team/worktree" clean -qfd

aarmy talk devin --team "$team" -v --timeout 10800 -b claude -m sonnet -e high \
    -s implement,tdd,code-review-and-quality \
    -p "\
    You are a lead software engineer. \
    You will produce code that you are proud of and you stand behind it. \
    Never sacrifice or compromise on maintainability and readability. \
    Good quality and good design practices are paramount. \
    The PR URL is: '$pr_url' use gh cli and git to implement its description. \
    The description is the whole spec. Do not read the issue it links to, or any issue comments. \
    It is a draft PR with an empty commit - push your implementation to its branch, \
    but leave as draft for now. A reviewr will give feedback, \
    you have the choice to accept or pushback, but always do justify your decision with \
    code facts and using the skills provided.
    You goals is to converge to a solution that satisfies the PR description and that is Okay \
    it it take a few iterations. \
    Post as the github app: \
    app_id: 4579193 \
    private_key: ~/keys/devin-development-specialist.2026-08-13.private-key.pem" &> "$log_dir/devin.log"

# Leaving draft is how devin says he stands behind the code, so it is the finish
# line. talk blocks on his lock, so a nudge sent mid-turn just waits its turn.
#
# The prompt below admits two outcomes and each has an observable: "I stand
# behind it" flips isDraft, "not there yet" moves headRefOid. A turn with
# neither answered nothing - devin starts `make ci` in the background and
# returns "Still running - I'll wait for completion", but a `claude --print`
# turn ends there and the notification he waits for can never arrive. That
# happened in the #55 run: 16:54:19Z to 16:55:20Z, 61.7s, head unchanged. Such
# a turn is not a round, so re-nudge instead of spending one of the five.
# Three in a row is a devin that is stuck and a fourth copy of the same prompt
# will not move him - the same bound the review loop's nudge below uses.
#
# No -s on the nudges below: this is devin's own resumed session and the
# skills were attached on his first talk above. Re-attaching them makes the
# skill prompt tell him to read the same files again - code-review-and-quality
# alone is 21KB - for a session that already has them.
#
# Check isDraft at the top, before talking: devin often marks the PR ready
# during the implement turn above, and then this loop's first turn is a whole
# talk that only reports back "already ready".
#
# The check also has to come before the head comparison below: marking the PR
# ready needs no commit, so devin's success outcome leaves the head unmoved and
# must not read as a stall.
head=$(gh pr view $pr_url --json headRefOid --jq .headRefOid)
rounds=0
stalls=0
while [ $rounds -lt 5 ] && [ $stalls -lt 3 ]; do

    [ "$(gh pr view $pr_url --json isDraft --jq '.isDraft')" = false ] && break

    aarmy talk devin --team "$team" -v --timeout 10800 \
    -p "Review your own diff across the five axes first. \
    Then: is the code something you are proud of, a solution you stand behind? \
    Judge that against your own standard, not the skill's 'improves code health' bar. \
    If so, mark the PR ready for review - \
    that is how you say you are done. If it is not there yet, \
    leave it as draft, fix, commit and push." &>> "$log_dir/devin.log"

    new_head=$(gh pr view $pr_url --json headRefOid --jq .headRefOid)
    if [ "$new_head" = "$head" ]; then
        stalls=$((stalls + 1))
    else
        head=$new_head
        rounds=$((rounds + 1))
        stalls=0
    fi
done

# A fresh reviewer session - it never saw the issue debate, so the PR
# description is all the spec it gets, same as devin. It posts as the reviewer
# app, which is the app that opened the PR, and GitHub will not let an account
# approve its own PR - so the label is the approval, not a review verdict.
# Review first, then check, so devin is not nudged after an approval.
# --timeout matches devin's: a review that runs `make ci` costs what building
# it did, and the default hour cuts the turn off mid-gate.
head=$(gh pr view $pr_url --json headRefOid --jq .headRefOid)

# Same reason as devin's nudges: attach the skill on the reviewer's first turn
# only, then empty it - every later turn is the same resumed session.
reviewer_skills=(-s code-review-and-quality)

for i in {1..10}; do

    aarmy talk code-reviewer --team "$team" -v --timeout 10800 -b claude -m opus -e high "${reviewer_skills[@]}" \
    -p "You are the code reviewer for the PR '$pr_url'. \
    Its description is the spec - review the diff against it, and against the five axes. \
    Every finding cites something re-checkable: a path and line, a command and its output, an API response. \
    Say which findings block merging and which are optional. \
    Do not change any code yourself - the developer fixes, you review. \
    Verify with 'make ci' here - the local run is far quicker than the same gates on GitHub, \
    so check the code yourself rather than waiting on the remote checks. \
    Where the developer pushed back, weigh the argument on its code facts and concede plainly when it is right. \
    otherwise push back with code facts and continue the review. \
    When nothing left blocks merging, and nite are solved, add the label 'reviewer-approves' to the PR, \
    creating the label first if the repo does not have it. \
    Post as the github app: \
    app_id: 4287312 \
    private_key: ~/keys/ai-specialist-reviewer.2026-08-04.private-key.pem" &>> "$log_dir/code-reviewer.log"

    reviewer_skills=()

    gh pr view $pr_url --json labels --jq '.labels[].name' | grep -qx 'reviewer-approves' && break

    # Nudge until the branch actually moves. A turn can end with the work
    # uncommitted - devin started `make ci` in the background and returned
    # waiting for a notification a one-shot turn can never deliver, twice in
    # the #77 run - and the reviewer would then spend a full opus turn
    # re-reading a commit it has already reviewed.
    for j in {1..3}; do
        aarmy talk devin --team "$team" -v --timeout 10800 \
        -p "There is new review feedback on '$pr_url'. \
        Read the review comment and answer it: fix it, or push back with code facts. \
        Reply saying which you did and why. \
        Commit and push your fixes to the PR branch. \
        Post as the github app: \
        app_id: 4579193 \
        private_key: ~/keys/devin-development-specialist.2026-08-13.private-key.pem" &>> "$log_dir/devin.log"

        new_head=$(gh pr view $pr_url --json headRefOid --jq .headRefOid)
        [ "$new_head" != "$head" ] && head=$new_head && break
    done
done

# The approval is the gate: code the reviewer never signed off gets no release
# notes, and the loop can also run out of rounds without one.
gh pr view $pr_url --json labels --jq '.labels[].name' | grep -qx 'reviewer-approves' || exit 3

# Doku writes for whoever will use this, not for whoever reviewed it: what the
# new version does for them, how to run it, and what will bite them.
aarmy talk doku --team "$team" -v -b claude -m opus -e low \
    -p "You are the documentation writer for the PR '$pr_url'. \
    Read its description and its diff, and read README.md and docs/ to see how the project describes itself today. \
    Post one comment on the PR telling a user of this project what this version changes for them. \
    Write plain, non-technical English: short sentences, no jargon, no file paths, no internal design talk. \
    Cover what is new or different, how to use it with a copy-pasteable command and a real example, \
    what they must change on their side to keep working - renamed flags, new defaults, removed behaviour - \
    and anything else that would surprise them. \
    Every claim comes from the diff. Do not describe behaviour you did not see in the code, \
    and if the change is invisible to users say so in one line instead of inventing detail. \
    Change no files, review nothing, approve nothing - the comment is the whole deliverable. \
    Post as the github app: \
    app_id: 4577311 \
    private_key: ~/keys/doku-documentation-agent.2026-08-12.private-key.pem" &> "$log_dir/doku.log"


# clean up

aarmy delete --team "$team"
git worktree remove --force "$AGENTS_ARMY_TEAMS_DIR/$team/worktree"
rm -rf "$AGENTS_ARMY_TEAMS_DIR/$team"