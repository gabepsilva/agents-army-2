# gdw v4 flow — owen triage + cheap debate + driver-owned build

```mermaid
flowchart TD
    subgraph SETUP["Driver setup"]
        A["Clean-tree check<br>worktree from origin/master"]
    end

    subgraph TRIAGE["Phase 1 — Owen triages (product owner)"]
        B["Owen: read the ask + bounded recon<br>(name affected areas, don't prove claims)"]
        C{"Triage decision<br>sizing bar: one readable PR"}
        REJ["Spectacle: concur / veto<br>requirements level, 1 cheap turn"]
        CLOSE["Close issue as not-planned<br>rationale in closing comment"]
        SPLIT["Owen: open self-contained child issues<br>label parent owens-split"]
        KIDS["Driver prints child URLs<br>each child = its own run"]
    end

    subgraph DEBATE["Phase 2 — Debate: deep on requirements, lazy on code"]
        F["Owen: compact proposal<br>behavior, areas, acceptance criteria<br>unproven claims marked as ASSUMPTIONS"]
        I["Spectacle: mandatory requirements critique<br>(no code dive, no free first-turn approval)"]
        D1{"Anything disputed<br>AND decision-changing?"}
        K["One code check per dispute,<br>by whoever asserted the claim"]
        L["Owen: one rebuttal, final position"]
        M{"Spectacle: final decision"}
        HAPPY["Labels: owens-is-happy +<br>spectacle-is-happy"]
    end

    subgraph HANDOFF["Handoff"]
        O["Doku: decision brief on the issue"]
        P["Spectacle: draft PR<br>spec + assumptions ledger"]
    end

    subgraph BUILD["Phase 3 — Build (go-lean back half, unchanged)"]
        Q["Devin: implement from PR description<br>verify assumptions in-file as he goes<br>false assumption → stop and comment"]
        R["Devin: one self-review, mark ready"]
        S{"Driver runs make ci"}
        T["Devin: one repair pass"]
        U["Reviewer: reads ci log + diff<br>max 3 rounds"]
        V["Devin: fix or push back<br>with re-checkable facts"]
        W["Label: reviewer-approves"]
    end

    subgraph DONE["Wrap-up"]
        X["Doku: user-facing note on the PR"]
        Y["Cleanup: team, worktree, branch"]
    end

    A --> B --> C
    C -- "bad idea" --> REJ
    REJ -- "concur" --> CLOSE
    CLOSE --> EXIT0a(["exit 0 — decided: don't build"])
    REJ -- "veto" --> F
    C -- "too big" --> SPLIT --> KIDS --> EXIT0b(["exit 0 — re-run per child"])
    C -- "proceed / reshape<br>(scope may grow)" --> F

    F --> I --> D1
    D1 -- "no" --> HAPPY
    D1 -- "yes" --> K --> L --> M
    M -- "resolved" --> HAPPY
    M -- "external evidence missing" --> BLOCKED(["blocked label — exit 1"])

    HAPPY --> O --> P --> Q --> R --> S
    S -- "fail" --> T --> S
    S -- "pass" --> U
    U -- "blockers" --> V
    V -- "new commit" --> S
    V -- "pushback only" --> U
    U -- "nothing blocks" --> W --> X --> Y --> DONE2(["exit 0 — PR approved"])
    U -- "3 rounds spent" --> FAIL3(["exit 3 — not approved"])
```

**Reading keys**

- Reject and split are *successful* exits (0), not failures — a converged decision whose outcome is "don't build this" or "build it in slices."
- Code is opened during the debate only when a claim is **disputed and decision-changing**; everything else rides to the PR as an assumptions ledger and gets verified by devin, who is in those files anyway.
- The driver owns `make ci`; the reviewer reads the log instead of re-running gates.
