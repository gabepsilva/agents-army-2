#!/usr/bin/env python3
"""gdw v5. flow.md is the specification, README.md the brief. The issue's
labels route every call: planning (invocation A) primes one role-neutral
session, forks it per issue and role, and debates every leaf of the issue
tree to a decision; build (invocation B) turns one converged leaf into an
approved PR - devin implements and self-reviews, the driver runs make ci,
the reviewer reads the log, doku posts the user note."""

import json
import os
import re
import shutil
import string
import subprocess
import sys
import time
from pathlib import Path


def sh(*cmd):
    """Run a command, return its stripped stdout, raise on failure."""
    return subprocess.run(
        cmd, capture_output=True, text=True, check=True
    ).stdout.strip()


def talk(agent, text, *flags):
    """One agent turn; the transcript appends to the agent's log file."""
    started = time.monotonic()
    with open(LOG_DIR / f"{agent}.log", "ab") as log:
        subprocess.run(
            [
                "uv",
                "run",
                "orchestrator",
                "talk",
                agent,
                "--team",
                TEAM,
                "-v",
                "--timeout",
                TIMEOUT,
                *flags,
                "-p",
                text,
            ],
            stdout=log,
            stderr=log,
            check=True,
        )
    print(f"  {agent} worked for {time.monotonic() - started:.0f} seconds")


def prompt(name, **subs):
    """Render prompts/<name>.md, substituting only the explicit variable list."""
    text = (PROMPT_DIR / f"{name}.md").read_text()
    return string.Template(text).safe_substitute(subs)


if len(sys.argv) != 2:
    sys.exit("usage: python go.py <issue-url>")

ISSUE_URL = sys.argv[1].rstrip("/")
ISSUE_NUMBER = ISSUE_URL.rsplit("/", 1)[1]
TEAM = f"issue-{ISSUE_NUMBER}"
REPO = Path(
    sh("git", "rev-parse", "--path-format=absolute", "--git-common-dir")
).parent.name
TEAMS_DIR = Path.home() / ".agents-army" / REPO / "gdw-v5"
WORKTREE = TEAMS_DIR / TEAM / "worktree"
LOG_DIR = TEAMS_DIR / "logs" / f"issue-{ISSUE_NUMBER}" / time.strftime("%Y%m%d-%H%M%S")
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
os.environ["AGENTS_ARMY_TEAMS_DIR"] = str(TEAMS_DIR)

# One dict per agent - the flags its FIRST talk carries, never again. Edit a
# line here to move that agent to another backend, model, or effort.
PRIMER = {"backend": "claude", "model": "opus", "effort": "medium"}
OWEN = {"backend": "claude", "model": "opus", "effort": "medium"}
# SPECTACLE = {"backend": "claude", "model": "luna", "effort": "max"}
SPECTACLE = {"backend": "opencode", "model": "opencode/gpt-5.6-luna", "effort": "max"}
# DEVIN = {"backend": "claude", "model": "luna", "effort": "max"}
DEVIN = {"backend": "opencode", "model": "opencode/gpt-5.6-luna", "effort": "max"}
CODE_REVIEWER = {"backend": "claude", "model": "opus", "effort": "high"}
# DOKU = {"backend": "claude", "model": "opus", "effort": "medium"}
DOKU = {
    "backend": "opencode",
    "model": "opencode/muse-spark-1.2-contributor-free",
    "effort": "medium",
}
ROLES = {"owen": OWEN, "spectacle": SPECTACLE, "doku": DOKU}  # the forkable roles
TIMEOUT = "10800"
PLAN_CAP = 12
OWEN_VERDICTS = {"owens-is-happy", "owens-rejects", "owens-is-blocked", "owens-split"}


def flags_for(agent):
    """The backend flags an agent's first talk carries."""
    return ["-b", agent["backend"], "-m", agent["model"], "-e", agent["effort"]]


def conf(agent):
    """The agent's dict, printed on the line that announces its first talk."""
    return json.dumps(agent)


PRIMER_PROMPT = (
    "You are a primer session: study this repository so that agents forked from "
    "you start already oriented. Read the README, the docs, and enough of the "
    "code to know the layout, the main components, the conventions, and how the "
    "project is tested. Change nothing and write nothing. Reply with a short "
    "summary of what the project is and how it is put together."
)

FORKS = set()  # the forks alive for the issue being planned
FORK_WORKS = True  # flips off if the fork verb refuses at runtime
CI_RUNS = []  # the driver's own make ci runs: {"head", "log"} each


# --- Labels are the state machine -------------------------------------------


def issue_labels(url):
    """Label names, plus the issue's OPEN/CLOSED state riding along as a
    pseudo-label so one call answers both questions."""
    info = json.loads(sh("gh", "issue", "view", url, "--json", "labels,state"))
    return {label["name"] for label in info["labels"]} | {info["state"]}


def blocked(labels):
    return bool(labels & {"owens-is-blocked", "spectacle-is-blocked"})


def converged(labels):
    return labels >= {"owens-is-happy", "spectacle-is-happy"}


def num(url):
    return url.rsplit("/", 1)[1]


def fresh_team_and_worktree():
    """Per issue: a fresh team and a fresh worktree from origin/master; stale
    leftovers from a dead run are deleted before the first talk."""
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(WORKTREE)], capture_output=True
    )
    sh("git", "worktree", "prune")
    shutil.rmtree(TEAMS_DIR / TEAM, ignore_errors=True)
    sh("git", "fetch", "origin")
    if subprocess.run(
        ["git", "worktree", "add", "--detach", str(WORKTREE), "origin/master"]
    ).returncode:
        print(f"Could not create the worktree at {WORKTREE}.", file=sys.stderr)
        sys.exit(5)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Issue {ISSUE_NUMBER} in {REPO}. Agent output goes to {LOG_DIR}")


def cleanup():  # Y - the team and worktree die with the run; logs are kept
    print("Cleaning up the team and worktree - logs are kept.")
    sh("uv", "run", "orchestrator", "delete", "--team", TEAM)
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(WORKTREE)], capture_output=True
    )
    shutil.rmtree(TEAMS_DIR / TEAM, ignore_errors=True)


# --- Invocation A - planning (recursive, then dies) --------------------------


def prime():  # PRIME - one role-neutral session reads the repo, once per run
    print(f"Priming: one role-neutral session reads the repo. {conf(PRIMER)}")
    talk("primer", PRIMER_PROMPT, *flags_for(PRIMER))


def fork_primer(role, url):
    """TRI, DEB, and BRIEF all begin "fork primer -> <role>". Returns the agent
    name plus the backend flags its next talk must carry: none for a real fork
    (it inherits the primer's config - the fork verb has no override), the full
    role flags when fork is unavailable and fresh agents stand in."""
    global FORK_WORKS
    name = f"{role}-{num(url)}"
    if name in FORKS:
        return name, []
    FORKS.add(name)
    if FORK_WORKS:
        try:
            sh("uv", "run", "orchestrator", "fork", "primer", name, "--team", TEAM)
            print(f"  {name} forks the primer, inheriting {conf(PRIMER)}")
            return name, []
        except subprocess.CalledProcessError:
            FORK_WORKS = False
            print(
                "The fork verb is unavailable - falling back to fresh agents per issue."
            )
    print(f"  {name} is a fresh agent. {conf(ROLES[role])}")
    return name, flags_for(ROLES[role])


def triage(url):  # TRI - owen: proceed / reshape / reject / split
    print(f"Owen: triaging {url} (proceed / reshape / reject / split).")
    name, flags = fork_primer("owen", url)
    talk(name, prompt("owen-triage", issue_url=url), *flags)
    return issue_labels(url)


def split_children(url):  # SPLITN - children from the CHILDREN: line, build order
    bodies = sh(
        "gh", "issue", "view", url, "--json", "comments", "--jq", ".comments[].body"
    )
    lines = [line for line in bodies.splitlines() if line.startswith("CHILDREN:")]
    numbers = re.findall(r"#(\d+)", lines[-1]) if lines else []
    if not numbers:
        print(
            f"{url} is labeled owens-split but has no parseable CHILDREN: line.",
            file=sys.stderr,
        )
        sys.exit(9)
    return [url.rsplit("/", 1)[0] + "/" + n for n in numbers]


def debate(url, labels):  # DEB - critique / concur-veto -> (rebuttal -> final decision)
    if "owens-rejects" in labels:
        print("Owen rejected the idea. Spectacle: one concur/veto turn.")
        name, flags = fork_primer("spectacle", url)
        talk(name, prompt("spectacle-reject-review", issue_url=url), *flags)
        if "CLOSED" in issue_labels(url):
            return "closed as not planned"  # CLOSED - a decision, not a failure
        print("Spectacle vetoed the rejection. Owen: write the proposal.")
        name, flags = fork_primer("owen", url)
        talk(name, prompt("owen-proposal-after-veto", issue_url=url), *flags)
    if blocked(issue_labels(url)):
        return "blocked"
    print("Spectacle: requirements critique.")
    name, flags = fork_primer("spectacle", url)
    talk(name, prompt("spectacle-critique", issue_url=url), *flags)
    labels = issue_labels(url)
    if blocked(labels):
        return "blocked"
    if not converged(labels):
        print("Disputed. Owen: one rebuttal.")
        name, flags = fork_primer("owen", url)
        talk(name, prompt("owen-rebuttal", issue_url=url), *flags)
        if blocked(issue_labels(url)):
            return "blocked"
        print("Spectacle: final decision.")
        name, flags = fork_primer("spectacle", url)
        talk(name, prompt("spectacle-final-decision", issue_url=url), *flags)
    labels = issue_labels(url)
    if blocked(labels):
        return "blocked"
    return "converged" if converged(labels) else "unconverged"


def decision_brief(url):  # BRIEF - doku's 350-word decision brief on the leaf
    print("Doku: writing the decision brief.")
    name, flags = fork_primer("doku", url)
    talk(name, prompt("doku-decision-brief", issue_url=url), *flags)


def discard_forks(url):  # DISC - nothing accumulates, nothing bleeds
    for role in ("owen", "spectacle", "doku"):
        name = f"{role}-{num(url)}"
        if name in FORKS:
            sh("uv", "run", "orchestrator", "delete", name, "--team", TEAM)
            FORKS.discard(name)


def tree_summary():  # SUMM - doku's plan of record on the root, only if anything split
    print("Doku: tree summary on the root.")
    name, flags = fork_primer("doku", ISSUE_URL)
    talk(name, prompt("doku-tree-summary", issue_url=ISSUE_URL), *flags)


def plan():  # Invocation A - BFS the tree, debate every leaf, then die
    fresh_team_and_worktree()
    prime()
    queue, taken, split_any = [ISSUE_URL], 0, False  # PQ
    while queue:  # NEXT
        url, taken = queue.pop(0), taken + 1
        if taken > PLAN_CAP:
            print(
                f"More than {PLAN_CAP} issues in one run - tree too big.",
                file=sys.stderr,
            )
            sys.exit(9)
        labels = issue_labels(url)
        if converged(labels) or "CLOSED" in labels:
            print(f"{url} needs no planning turn - skipping.")
            continue
        if not labels & OWEN_VERDICTS:
            labels = triage(url)
        if "owens-split" in labels:
            children = split_children(url)
            print(f"{url} split into {len(children)} children, queued in build order.")
            queue += children
            split_any = True
        else:
            verdict = debate(url, labels)
            print(f"{url}: {verdict}.")
            if verdict == "converged":
                decision_brief(url)
        discard_forks(url)
    if split_any:
        tree_summary()
    cleanup()
    print(
        "Planning done. Read the briefs, pick the leaves to build, and call "
        "go.py once per leaf, merging between calls."
    )
    sys.exit(0)  # REPORT


# --- Invocation B - build one converged leaf ---------------------------------


def find_draft_pr():
    """The list endpoint is read-after-write; the caller polls."""
    prs = json.loads(
        sh(
            "gh",
            "pr",
            "list",
            "--draft",
            "--limit",
            "50",
            "--json",
            "url,body,headRefName",
        )
    )
    hits = [pr for pr in prs if re.search(rf"(#|issues/){ISSUE_NUMBER}\b", pr["body"])]
    return (hits[0]["url"], hits[0]["headRefName"]) if hits else None


def reuse_or_open_draft_pr():  # BUILD0 - never create a second PR when one exists
    pr = find_draft_pr()
    if pr:
        print(f"Reusing the existing draft PR {pr[0]}.")
    else:
        print(f"Spectacle: opening the draft PR. {conf(SPECTACLE)}")
        talk(
            "spectacle",
            prompt("spectacle-draft-pr", issue_url=ISSUE_URL),
            *flags_for(SPECTACLE),
        )
        for _ in range(6):
            if pr := find_draft_pr():
                break
            time.sleep(5)
    if not pr:
        print(f"No draft PR found for issue {ISSUE_NUMBER}.", file=sys.stderr)
        sys.exit(2)
    pr_url, branch = pr
    print(f"Draft PR {pr_url} on branch {branch}")
    sh("git", "-C", str(WORKTREE), "fetch", "origin")
    if subprocess.run(
        ["git", "-C", str(WORKTREE), "checkout", "-q", "-B", branch, f"origin/{branch}"]
    ).returncode:
        print(f"Could not check out {branch} in the worktree.", file=sys.stderr)
        sys.exit(5)
    return pr_url, branch


def implement(pr_url):  # Q - the PR description is the whole spec
    print(f"Devin: implementing the PR description. {conf(DEVIN)}")
    talk(
        "devin",
        prompt("devin-implement", pr_url=pr_url),
        *flags_for(DEVIN),
        "-s",
        "implement,tdd,code-review-and-quality",
    )


def self_review(pr_url):  # R - one self-review, then mark ready
    print("Devin: one self-review pass.")
    talk("devin", prompt("devin-self-review", pr_url=pr_url))


def require_committed_and_pushed(pr_url):  # exit 8 - never rely on unpushed work
    dirty = sh("git", "-C", str(WORKTREE), "status", "--porcelain")
    if dirty:
        print(f"Devin left uncommitted work in {WORKTREE}:\n{dirty}", file=sys.stderr)
        sys.exit(8)
    local = sh("git", "-C", str(WORKTREE), "rev-parse", "HEAD")
    remote = sh(
        "gh", "pr", "view", pr_url, "--json", "headRefOid", "--jq", ".headRefOid"
    )
    if local != remote:
        print(
            f"Devin's HEAD {local} was not pushed to the PR head {remote}.",
            file=sys.stderr,
        )
        sys.exit(8)


def run_make_ci():  # S - the driver owns make ci; reviewers read the log
    ci_log = LOG_DIR / f"ci-{len(CI_RUNS) + 1}.log"
    ci_head = sh("git", "-C", str(WORKTREE), "rev-parse", "HEAD")
    CI_RUNS.append({"head": ci_head, "log": ci_log})
    print(f"Running 'make ci' (run {len(CI_RUNS)}) on {ci_head} - log: {ci_log}")
    with open(ci_log, "wb") as out:
        return (
            subprocess.run(
                ["make", "-C", str(WORKTREE), "ci"], stdout=out, stderr=out
            ).returncode
            == 0
        )


def ci_gate(pr_url):  # S -> fail -> T (one repair) -> S; a second failure is exit 7
    if run_make_ci():
        return
    print("CI failed. Devin: one repair pass.")
    talk(
        "devin",
        prompt("devin-ci-repair", pr_url=pr_url, ci_log=str(CI_RUNS[-1]["log"])),
    )
    require_committed_and_pushed(pr_url)
    if not run_make_ci():
        print("CI still failing after the repair pass.", file=sys.stderr)
        sys.exit(7)


def reviewer_approves(pr_url):  # W - the label is the verdict, not the review text
    labels = sh(
        "gh", "pr", "view", pr_url, "--json", "labels", "--jq", ".labels[].name"
    )
    return "reviewer-approves" in labels.splitlines()


def review_rounds(pr_url):  # U -> blockers -> V -> (new commit -> S | pushback -> U)
    flags = [*flags_for(CODE_REVIEWER), "-s", "code-review-and-quality"]
    for review_round in (1, 2, 3):
        print(
            f"Review round {review_round} of 3. Code reviewer: reviewing the diff."
            + (f" {conf(CODE_REVIEWER)}" if flags else "")
        )
        talk(
            "code-reviewer",
            prompt(
                "code-reviewer-review",
                pr_url=pr_url,
                ci_head=CI_RUNS[-1]["head"],
                ci_log=str(CI_RUNS[-1]["log"]),
            ),
            *flags,
        )
        flags = []
        if reviewer_approves(pr_url):
            print("Reviewer approved.")
            return
        if review_round == 3:
            break
        print("Devin: addressing the review feedback.")
        old_head = sh("git", "-C", str(WORKTREE), "rev-parse", "HEAD")
        talk("devin", prompt("devin-review-response", pr_url=pr_url))
        require_committed_and_pushed(pr_url)
        if sh("git", "-C", str(WORKTREE), "rev-parse", "HEAD") != old_head:
            ci_gate(pr_url)  # a pushback without a commit goes straight back to U
    print("PR not approved after 3 review rounds.", file=sys.stderr)
    sys.exit(3)  # FAIL3


def user_note(pr_url):  # X - doku's user-facing note on the PR
    print(f"Doku: writing the user-facing note on the PR. {conf(DOKU)}")
    talk("doku", prompt("doku-user-note", pr_url=pr_url), *flags_for(DOKU))


def build():  # Invocation B - one converged leaf to an approved PR
    fresh_team_and_worktree()
    pr_url, branch = reuse_or_open_draft_pr()  # BUILD0
    implement(pr_url)  # Q
    self_review(pr_url)  # R
    if (
        sh("gh", "pr", "view", pr_url, "--json", "isDraft", "--jq", ".isDraft")
        != "false"
    ):
        print(
            "PR is still a draft after the self-review - devin found a false "
            "assumption. Read his PR comment, amend the spec, call again.",
            file=sys.stderr,
        )
        sys.exit(6)  # DR - still draft
    require_committed_and_pushed(pr_url)
    ci_gate(pr_url)  # S and T
    review_rounds(pr_url)  # U and V, approval sets W's label
    user_note(pr_url)  # X
    cleanup()  # Y
    subprocess.run(["git", "branch", "-q", "-D", branch], capture_output=True)
    print(f"Done. {pr_url} is approved and ready for merge.")
    sys.exit(0)  # DONE


# --- The story, top to bottom ------------------------------------------------

dirty = sh("git", "status", "--porcelain")
if dirty:
    print(
        f"Uncommitted changes in {os.getcwd()} - commit or stash them first:\n{dirty}",
        file=sys.stderr,
    )
    sys.exit(4)

labels = issue_labels(ISSUE_URL)  # ROUTE - the labels decide the invocation
if blocked(labels):
    print("The issue is blocked - resolve the missing evidence first.", file=sys.stderr)
    sys.exit(1)  # RB
if converged(labels):
    build()  # Invocation B - build one converged leaf
plan()  # Invocation A - planning (recursive, then dies)
