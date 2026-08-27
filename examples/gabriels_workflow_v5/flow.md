# gdw v5 flow — plan the whole tree with primer forks, then build one leaf per call

This diagram is the specification: [README.md](README.md) tells the
implementing agent how to turn it into `go.py`. Every agent box's prompt is
the matching file in [prompts/](prompts/). The issue's labels are the state:
calling the driver is always safe and always does the right next thing.

```mermaid
flowchart TD
    START(["go.py + issue URL"]) --> ROUTE{"Labels?"}
    ROUTE -- "blocked" --> RB(["exit 1 — resolve evidence first"])
    ROUTE -- "owens-is-happy +<br>spectacle-is-happy" --> BUILD0
    ROUTE -- "anything else" --> PRIME

    subgraph PLAN["Invocation A — planning (recursive, then dies)"]
        PRIME["Prime once per run:<br>one role-neutral session reads the repo"]
        PQ["Queue = [issue]<br>cap: 12 issues per run"]
        NEXT{"Next issue<br>from queue"}
        TRI["fork primer → owen: triage<br>(proceed / reshape / reject / split)"]
        SPLITN["Children from the CHILDREN: line<br>queued in build order<br>(their forks share the same primer)"]
        DEB["fork primer → spectacle:<br>critique / concur-veto →<br>(rebuttal → final decision) if disputed<br>checks routed to wherever correctness is cheapest"]
        BRIEF["fork primer → doku: decision brief<br>(350-word budget)"]
        DISC["Discard the forks —<br>nothing accumulates, nothing bleeds"]
        SUMM["Doku: tree summary on the root<br>(only if anything split)"]
        REPORT(["exit 0 — script dies<br>human reads briefs, picks leaves to build"])
    end

    PRIME --> PQ --> NEXT
    NEXT -- "already converged" --> NEXT
    NEXT --> TRI
    TRI -- "split" --> SPLITN --> NEXT
    TRI -- "reject + concur" --> CLOSED["Issue closed as not planned"] --> NEXT
    TRI -- "reject + veto" --> DEB
    TRI -- "proceed / reshape" --> DEB
    DEB -- "converged" --> BRIEF --> DISC --> NEXT
    DEB -- "blocked" --> DISC
    NEXT -- "queue empty" --> SUMM --> REPORT

    subgraph BUILDP["Invocation B — build one converged leaf"]
        BUILD0["Reuse existing draft PR,<br>or Spectacle opens one<br>(spec + assumptions ledger,<br>written against current master)"]
        Q["Devin: implement<br>verify assumptions in-file<br>false assumption: stop, comment, stay draft"]
        R["Devin: one self-review, mark ready"]
        DR{"PR ready?"}
        SY["Stale-base gate: fetch; if master<br>moved, merge origin/master in<br>(clean: driver merges and pushes;<br>conflicts: devin resolves, else exit 10)"]
        S{"Driver runs make ci<br>(one at a time machine-wide)"}
        T["Devin: one repair pass"]
        U["Reviewer: reads ci log + diff<br>max 3 rounds"]
        V["Devin: fix or push back"]
        SY2{"master moved<br>during review?"}
        REGATE["One re-gate: merge<br>origin/master, make ci again"]
        W["Label: reviewer-approves"]
        X["Doku: user-facing note"]
        Y["Cleanup — logs are kept"]
    end

    BUILD0 --> Q --> R --> DR
    DR -- "still draft" --> EXIT6(["exit 6 — read devin's PR comment,<br>amend the spec, call again"])
    DR -- "ready" --> SY --> S
    S -- "fail" --> T --> S
    S -- "pass" --> U
    U -- "blockers" --> V
    V -- "new commit" --> S
    V -- "pushback only" --> U
    U -- "nothing blocks" --> SY2
    SY2 -- "no" --> W
    SY2 -- "yes" --> REGATE
    REGATE -- "green, master still" --> W
    REGATE -- "master moved again" --> EXIT10(["exit 10 — merge the<br>open queue in order first"])
    W --> X --> Y --> DONE(["exit 0 — PR ready for merge<br>human merges, then calls the next leaf"])
    U -- "3 rounds spent" --> FAIL3(["exit 3 — not approved"])
```

**Reading keys**

- **Two invocations by design.** Planning always ends with the script dying after every leaf is briefed — the human reads doku's briefs (and the tree summary on the root) and decides what gets built, while disagreeing is still cheap. Build runs one leaf per call, in the `CHILDREN:` order, with the human merging between calls so each leaf builds on the previous one's merged code.
- **Labels are the state machine.** No verdict → plan. `owens-split` → its children get planned. Converged → build. Blocked → refused until resolved. Re-calling after a crash lands in the right phase automatically; an existing draft PR is reused, never duplicated.
- **The primer.** Planning primes one role-neutral session on the repo, then `orchestrator fork`s it per issue and per role; forks are discarded after their leaf. Repo knowledge is shared through the primer; debate content never is.
- Code is opened during the debate only when a claim is **disputed and decision-changing**, checked by whoever can check cheapest; everything else rides the assumptions ledger to devin, who verifies in the files he is editing anyway. A false assumption stops the build (exit 6) with devin's finding as a PR comment — amend the spec and call again.
- The driver owns `make ci`; the reviewer reads the log instead of re-running gates. Remote (GitHub-hosted) CI is never consulted — local `make ci` is authoritative *because the stale-base gate guarantees it ran on a branch containing current `origin/master`*. Parallel runs take turns on a machine-wide CI lock, so gates never fight for cores.
- **Parallel runs.** Each issue works in a private full clone (at `<team>/worktree` — the path `--team` resolves as the workdir), so concurrent runs share no git state; a per-issue flock refuses a second run on the same issue (exit 4). Parallelize across *independent* issues only: the leaves of one split tree still build serially in `CHILDREN:` order — siblings edit the same files, and every merge forces every other in-flight branch through a re-gate.
- Run-state vs record: `teams/` content dies with each issue's processing (fresh team + clone per issue, stale ones reset); `logs/issue-N/<timestamp>/` is permanent.
- Exit codes: 0 phase completed (planning done / rejected / PR approved) · 1 blocked or unconverged · 2 no PR · 3 not approved in 3 rounds · 4 another run holds this issue · 5 clone/branch failure · 6 draft after self-review (false assumption) · 7 CI unfixable · 8 devin didn't commit/push · 9 tree too big or unparseable split · 10 stale-base gate failed (merge unresolved, or master kept moving).
