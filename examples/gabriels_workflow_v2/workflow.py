"""Driver-mediated compact relay for Gabriel's development workflow V2."""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from examples.gabriels_workflow_v2.config import BudgetConfig
from examples.gabriels_workflow_v2.contracts import (
    CheckpointStore,
    Stage,
    canonical_json,
    digest,
    validate_handoff,
)
from examples.gabriels_workflow_v2.errors import WorkflowError, WorkflowStopped
from examples.gabriels_workflow_v2.gates import CommandResult
from examples.gabriels_workflow_v2.gateway import AgentGateway, RoleOptions
from examples.gabriels_workflow_v2.git import GitRepository
from examples.gabriels_workflow_v2.github_app import pull_request_number

LEDGER_SUMMARY_CHARS = 400
ISSUE_BODY_CHARS = 20_000
ISSUE_COMMENT_CHARS = 3_000
ISSUE_COMMENTS = 10


def _outcome(result: Mapping[str, object]) -> str:
    """A one-line digest of a structured reply, for the progress log."""

    fields = [
        f"{key}={result[key]}"
        for key in ("decision", "verdict", "status", "needs_another_round")
        if key in result
    ]
    return ", ".join(fields) if fields else "no outcome fields"


def _shorten_home(path: Path) -> str:
    resolved_path = path.resolve()
    home = Path.home().resolve()
    if resolved_path == home:
        return "~"
    try:
        relative = resolved_path.relative_to(home)
    except ValueError:
        return str(resolved_path)
    return f"~/{relative.as_posix()}"


def _conclusion(output: Mapping[str, Any]) -> str:
    """A stage's reply as a GitHub check-run conclusion.

    `neutral` is the one worth explaining: a reviewer asking for changes or a
    griller asking for another round is the process working, not a failure,
    and a run that reached publication had all of those resolved.
    """

    if output.get("status") == "blocked" or output.get("verdict") == "reject":
        return "failure"
    if output.get("decision") == "stop":
        return "failure"
    if output.get("verdict") in {"revise", "changes_requested"}:
        return "neutral"
    return "success"


def _gate_digest(result: Mapping[str, Any]) -> str:
    gates = result.get("gates")
    if not isinstance(gates, list) or not gates:
        return f"make ci exited {result.get('returncode')}"
    passed = sum(1 for gate in gates if gate.get("status") == "passed")
    failed = [str(gate.get("name")) for gate in gates if gate.get("status") == "failed"]
    digested = f"{passed}/{len(gates)} gates passed"
    return f"{digested}; failed: {', '.join(failed)}" if failed else digested


class Publisher(Protocol):
    def issue(self, number: int) -> dict[str, Any]: ...

    def collect_markers(self, number: int) -> None: ...

    def comment_once(
        self,
        number: int,
        key: str,
        title: str,
        payload: object,
        *,
        attribution: str = "",
    ) -> None: ...

    def create_or_find_pr(
        self,
        *,
        base: str,
        branch: str,
        title: str,
        body: str,
        draft: bool,
    ) -> str: ...

    def update_pr(self, number: int, *, body: str) -> None: ...

    def publish_checks(
        self, head_sha: str, ledger: Sequence[Mapping[str, Any]]
    ) -> None: ...


class Repository(Protocol):
    def head(self) -> str: ...

    def snapshot(self) -> str: ...

    def run_ci(self, timeout: int) -> CommandResult: ...

    def require_changed(self, initial_snapshot: str) -> None: ...

    def commit(self, message: str, base_sha: str) -> None: ...

    def push(self, branch: str) -> None: ...


class Agents(Protocol):
    @property
    def workdir(self) -> Path: ...

    def options(self, role: str) -> RoleOptions: ...

    def ask(
        self,
        *,
        role: str,
        prompt_name: str,
        schema_name: str,
        values: Mapping[str, str],
        skills: Sequence[str] = (),
        timeout: int,
    ) -> dict: ...


@dataclass(frozen=True)
class WorkflowOptions:
    issue_number: int
    base_branch: str
    branch: str
    draft: bool
    initial_snapshot: str
    budgets: BudgetConfig


@dataclass(frozen=True)
class WorkflowServices:
    store: CheckpointStore
    publisher: Publisher
    repository: Repository
    agents: Agents


class RelayAgentGateway(AgentGateway):
    """Reuse the hardened sandbox while enforcing a bounded prompt."""

    def __init__(self, *, max_prompt_chars: int, **kwargs: Any) -> None:
        self.max_prompt_chars = max_prompt_chars
        super().__init__(**kwargs)

    def _prompt(self, name: str, values: Mapping[str, str]) -> str:
        prompt = super()._prompt(name, values)
        if len(prompt) > self.max_prompt_chars:
            raise WorkflowError(
                f"prompt {name} has {len(prompt)} characters; budget is "
                f"{self.max_prompt_chars}"
            )
        return prompt


class RelayRepository(GitRepository):
    """Git and CI mechanics plus a commit-independent content fingerprint."""

    def head(self) -> str:
        return self._call("rev-parse", "HEAD").strip()

    def common_git_dir(self) -> Path:
        """The repository's shared git directory, as an absolute path.

        Linked worktrees each have their own `.git` file but share this one,
        so state keyed to it is found again from any of them.
        """

        common = Path(self._call("rev-parse", "--git-common-dir").strip())
        if not common.is_absolute():
            common = self.root / common
        return common.resolve()

    def snapshot(self) -> str:
        names = self._call(
            "ls-files", "--cached", "--others", "--exclude-standard", "-z"
        ).split("\0")
        hashed = hashlib.sha256()
        for name in sorted(path for path in names if path):
            path = self.root / name
            hashed.update(name.encode())
            hashed.update(b"\0")
            try:
                mode = path.lstat().st_mode
            except OSError:
                hashed.update(b"missing\0")
                continue
            hashed.update(str(mode).encode())
            hashed.update(b"\0")
            if path.is_symlink():
                hashed.update(os.readlink(path).encode())
            elif path.is_file():
                with path.open("rb") as source:
                    while chunk := source.read(1024 * 1024):
                        hashed.update(chunk)
            else:
                hashed.update(b"non-file")
            hashed.update(b"\0")
        return hashed.hexdigest()

    def require_changed(self, initial_snapshot: str) -> None:
        if self.snapshot() == initial_snapshot:
            raise WorkflowStopped("implementation did not change the working tree")


class DevelopmentWorkflowV2:
    """Eight-role workflow with compact handoffs and milestone-only publication."""

    def __init__(self, options: WorkflowOptions, services: WorkflowServices) -> None:
        self.issue_number = options.issue_number
        self.base_branch = options.base_branch
        self.branch = options.branch
        self.draft = options.draft
        self.initial_snapshot = options.initial_snapshot
        self.budgets = options.budgets
        self.store = services.store
        self.base_sha = str(services.store.metadata["base_sha"])
        self.publisher = services.publisher
        self.repository = services.repository
        self.agents = services.agents
        self.root = Path(__file__).resolve().parent
        # Rebuilt from checkpoints on every run, so a resumed run's ledger
        # still reports the turns an earlier process paid for.
        self.ledger: list[dict[str, Any]] = []

    def run(self) -> str:
        completed = self.completed_url()
        if completed is not None:
            return completed
        issue = self._issue_snapshot()
        proposal = self._clarify(issue)
        specification = self._specify(issue, proposal)
        self._publish_milestone(
            "specification",
            self.issue_number,
            "Validated specification",
            self._without_handoff(specification),
        )
        implementation = self._work(
            "implementation",
            "implementer",
            "implement",
            {"specification": specification},
            ("code-simplification", "caveman"),
        )
        documentation = self._work(
            "documentation",
            "documenter",
            "document",
            {
                "specification": specification,
                "previous_handoff": implementation["handoff"],
            },
            ("caveman",),
        )
        ci = self._ci_until_green(
            "implementation",
            specification,
            {"implementation": implementation, "documentation": documentation},
        )
        reviews, ci = self._review_until_approved(specification, ci, issue)
        self.repository.require_changed(self.initial_snapshot)
        title = str(specification["title"]).replace("\n", " ")[:72]
        self.repository.commit(
            f"Implement #{self.issue_number}: {title}", self.base_sha
        )
        self.repository.push(self.branch)
        url = self._open_or_update_pull_request(
            title, specification, implementation, documentation, ci, reviews
        )
        finalization = self._stage(
            Stage(
                "finalization",
                "finalizer",
                "finalize",
                "finalization",
                {
                    "issue": issue,
                    "specification": self._without_handoff(specification),
                    "implementation_handoff": implementation["handoff"],
                    "documentation_handoff": documentation["handoff"],
                    "ci": self._ci_summary(ci),
                    "reviews": reviews,
                    "pull_request_url": url,
                },
            )
        )
        self._require_complete(finalization, "finalization")
        number = pull_request_number(url)
        self._publish_checks(self.repository.head())
        self._publish_milestone(
            "final-summary",
            number,
            "Final implementation summary",
            self._without_handoff(finalization),
            attribution=self._ledger_markdown(),
        )
        self.store.update_metadata(
            complete=True,
            pr_number=number,
            pr_url=url,
            completed_at=datetime.now(UTC).isoformat(),
        )
        return url

    def completed_url(self) -> str | None:
        metadata = self.store.metadata
        url = metadata.get("pr_url")
        if metadata.get("complete") is True and isinstance(url, str) and url:
            return url
        return None

    def _issue_snapshot(self) -> dict[str, Any]:
        return self.store.load_or_save_issue(
            lambda: self._bounded_issue(self.publisher.issue(self.issue_number))
        )

    @staticmethod
    def _bounded_issue(issue: dict[str, Any]) -> dict[str, Any]:
        comments = issue.get("comments", [])
        bounded_comments = []
        if isinstance(comments, list):
            for comment in comments[-ISSUE_COMMENTS:]:
                if not isinstance(comment, Mapping):
                    continue
                bounded_comments.append(
                    {
                        "author": comment.get("author"),
                        "body": str(comment.get("body", ""))[:ISSUE_COMMENT_CHARS],
                        "createdAt": comment.get("createdAt"),
                    }
                )
        return {
            "number": issue.get("number"),
            "title": str(issue.get("title", ""))[:ISSUE_BODY_CHARS],
            "body": str(issue.get("body", ""))[:ISSUE_BODY_CHARS],
            "labels": issue.get("labels", []),
            "comments": bounded_comments,
        }

    def _clarify(self, issue: dict[str, Any]) -> dict[str, Any]:
        previous: str | None = None
        outstanding: list[str] = []
        expansion: dict[str, Any] | None = None
        grill: dict[str, Any] | None = None
        for round_number in range(1, self.budgets.max_clarification_rounds + 1):
            context: dict[str, Any] = {"canonical_issue": issue}
            prompt = "expand"
            if expansion is not None and grill is not None:
                prompt = "revise"
                context.update(
                    {
                        "current_proposal": self._without_handoff(expansion),
                        "review_handoff": grill["handoff"],
                    }
                )
            expansion = self._stage(
                Stage(
                    f"expansion-{round_number}",
                    "expander",
                    prompt,
                    "proposal",
                    context,
                )
            )
            if expansion["decision"] == "stop":
                raise WorkflowStopped(str(expansion["summary"]))
            grill = self._stage(
                Stage(
                    f"grill-{round_number}",
                    "griller",
                    "grill",
                    "grill",
                    {
                        "canonical_issue": issue,
                        "proposal": self._without_handoff(expansion),
                        "previous_handoff": expansion["handoff"],
                    },
                )
            )
            if grill["verdict"] == "reject":
                raise WorkflowStopped(str(grill["summary"]))
            if grill["verdict"] == "escalate":
                if not grill["questions"]:
                    raise WorkflowError("griller escalated without questions")
                raise self._escalate(
                    f"clarification escalated: {grill['summary']}",
                    grill["questions"],
                )
            if (
                grill["verdict"] == "ready"
                and not grill["needs_another_round"]
                and not expansion["needs_another_round"]
            ):
                return {"expansion": expansion, "grill": grill}
            unresolved = digest(
                {
                    "expansion": expansion["handoff"],
                    "grill": grill["handoff"],
                }
            )
            if unresolved == previous:
                raise WorkflowStopped("clarification stalled")
            previous = unresolved
            outstanding = grill["questions"] or grill["handoff"]["open_questions"]
        raise self._escalate(
            f"clarification exceeded {self.budgets.max_clarification_rounds} rounds",
            outstanding,
        )

    def _escalate(self, reason: str, questions: Sequence[str]) -> WorkflowStopped:
        """Stop by asking the human the outstanding questions, not by deciding them.

        The griller used to have no exit for a question of authority. `reject`
        abandons the proposal and `ready` decides, so a round budget closing in
        on "may this issue's stated scope be narrowed?" put all the pressure on
        `ready`: a real run cleared its own open question and shipped work the
        issue had not asked for. The questions are published as a milestone
        comment because the issue is the one channel a human already reads and
        the only place an answer can be written that the next run will see.
        Exhausting the round budget surfaces the last round's questions the
        same way, since that is the other exit where they would otherwise die
        in a stop message.

        The comment is keyed by the questions themselves, so re-running without
        answering re-posts nothing while a genuinely different question is
        still asked. Discarding the issue snapshot and the clarification
        checkpoints is what makes the next run a resume rather than a
        stale-checkpoint failure: an answer only reaches an agent by being read
        back off the issue, and every checkpoint recorded against the text that
        predates it would fail its input hash. Nothing after clarification has
        run, so nothing after clarification is disturbed.

        Returned rather than raised so both stop paths keep their `raise` where
        a reader of `_clarify` looks for it.
        """

        asked = list(questions)
        self._publish_milestone(
            f"clarification-escalation-{digest(asked)[:12]}",
            self.issue_number,
            "Clarification needs a human decision",
            {
                "reason": reason,
                "questions": asked,
                "how_to_answer": (
                    "Answer on this issue, then re-run the workflow for it. "
                    "Clarification restarts from the re-read issue; no later "
                    "stage has run."
                ),
            },
        )
        self.store.discard_for_new_issue(
            key
            for round_number in range(1, self.budgets.max_clarification_rounds + 1)
            for key in (f"expansion-{round_number}", f"grill-{round_number}")
        )
        return WorkflowStopped(reason)

    def _specify(
        self, issue: dict[str, Any], proposal: dict[str, Any]
    ) -> dict[str, Any]:
        return self._stage(
            Stage(
                "specification",
                "specifier",
                "specify",
                "specification",
                {"canonical_issue": issue, "accepted_proposal": proposal},
            )
        )

    def _work(
        self,
        key: str,
        role: str,
        prompt: str,
        context: dict[str, Any],
        skills: tuple[str, ...],
    ) -> dict[str, Any]:
        output = self._stage(Stage(key, role, prompt, "work-report", context, skills))
        self._require_complete(output, key)
        return output

    def _ci_until_green(
        self,
        prefix: str,
        specification: dict[str, Any],
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        for attempt in range(1, self.budgets.max_ci_attempts + 1):
            before = self.repository.snapshot()
            key = f"ci-{prefix}-{before[:12]}"
            input_sha = digest({"snapshot": before, "timeout": self.budgets.ci_timeout})
            cached = self.store.load_checkpoint(key, input_sha)
            if cached is None:
                started = datetime.now(UTC)
                clock = time.monotonic()
                result = self.repository.run_ci(self.budgets.ci_timeout).as_json()
                turn = {
                    "role": "driver",
                    "backend": None,
                    "model": None,
                    "reasoning_effort": None,
                    "skills": [],
                    "started_at": started.isoformat(),
                    "duration_seconds": round(time.monotonic() - clock, 1),
                    "context_chars": 0,
                    "output_chars": len(canonical_json(result)),
                    "outcome": f"returncode={result.get('returncode')}",
                    "conclusion": (
                        "success" if result.get("returncode") == 0 else "failure"
                    ),
                    "summary": _gate_digest(result),
                }
                self.store.save_checkpoint(
                    key,
                    role="driver",
                    input_sha256=input_sha,
                    output=result,
                    turn=turn,
                )
                self._record(turn, key, reused=False)
            else:
                result = cached.output
                self._record(cached.turn, key, reused=True)
            if result.get("returncode") == 0:
                return result
            repair = self._work(
                f"repair-{prefix}-{attempt}-{before[:12]}",
                "implementer",
                "repair",
                {
                    "specification": self._without_handoff(specification),
                    "previous_handoffs": {
                        name: value.get("handoff")
                        for name, value in evidence.items()
                        if isinstance(value, dict)
                    },
                    "failure_evidence": result,
                },
                ("code-simplification", "caveman"),
            )
            after = self.repository.snapshot()
            if after == before:
                raise WorkflowStopped("CI repair reported complete without changes")
            evidence = {"repair": repair}
        raise WorkflowStopped(f"CI exceeded {self.budgets.max_ci_attempts} attempts")

    def _review_until_approved(
        self,
        specification: dict[str, Any],
        ci: dict[str, Any],
        issue: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Review, repair, and re-run CI until both reviewers approve.

        Returns the approving reviews together with the CI evidence they were
        approved against, which is not the run this started with once a
        repair round has happened.

        The specification reviewer also gets the canonical issue, because from
        `_specify` onward the specification was the only ground truth: a
        criterion the specifier quietly narrowed, or an `out_of_scope` entry it
        invented, read as satisfied to everyone downstream. This is the one
        stage that sees both documents, so it is where that drift can still be
        caught. The quality reviewer does not get it — its axes are code, not
        scope, a second opinion on the same drift would only double-report it,
        and its context already carries the diff plus two review skills.

        The issue defaults to the run's stored snapshot rather than being
        required from the caller, since it is run state that `_issue_snapshot`
        already serves from local disk; a caller holding it passes it to save
        the read.
        """

        issue = self._issue_snapshot() if issue is None else issue
        for round_number in range(1, self.budgets.max_review_rounds + 1):
            snapshot = self.repository.snapshot()
            reviews = {
                kind: self._stage(
                    Stage(
                        f"review-{round_number}-{kind}-{snapshot[:12]}",
                        f"reviewer-{kind}",
                        f"review-{kind}",
                        "review",
                        {
                            "canonical_specification": self._without_handoff(
                                specification
                            ),
                            "diff_against": self.base_sha,
                            "ci": self._ci_summary(ci),
                            **(
                                {"canonical_issue": issue}
                                if kind == "specification"
                                else {}
                            ),
                        },
                        ()
                        if kind == "specification"
                        else ("code-review-and-quality", "code-simplification"),
                    )
                )
                for kind in ("specification", "quality")
            }
            for kind, review in reviews.items():
                if review["verdict"] == "approve" and review["findings"]:
                    raise WorkflowError(f"{kind} review approved with findings")
            if all(
                review["verdict"] == "approve" and not review["needs_another_round"]
                for review in reviews.values()
            ):
                return reviews, ci
            repair = self._work(
                f"review-repair-{round_number}-{snapshot[:12]}",
                "implementer",
                "repair",
                {
                    "specification": self._without_handoff(specification),
                    "failure_evidence": reviews,
                },
                ("code-simplification", "caveman"),
            )
            if self.repository.snapshot() == snapshot:
                raise WorkflowStopped("review repair reported complete without changes")
            ci = self._ci_until_green(
                f"review-{round_number}", specification, {"repair": repair}
            )
        raise WorkflowStopped(
            f"review exceeded {self.budgets.max_review_rounds} rounds"
        )

    def _stage(self, stage: Stage) -> dict[str, Any]:
        options = self.agents.options(stage.role)
        contract = {
            "stage": stage.key,
            "role": stage.role,
            "backend": options.backend,
            "model": options.model,
            "reasoning_effort": options.reasoning_effort,
            "prompt_sha256": self._file_digest("prompts", f"{stage.prompt}.md"),
            "schema_sha256": self._file_digest("validations", f"{stage.schema}.json"),
            "context": stage.context,
            "skills": stage.skills,
        }
        input_sha = digest(contract)
        cached = self.store.load_checkpoint(stage.key, input_sha)
        if cached is not None:
            self._record(cached.turn, stage.key, reused=True)
            return validate_handoff(cached.output)
        self.store.reserve_turn(self.budgets.max_agent_turns)
        context_json = canonical_json(stage.context)
        started = datetime.now(UTC)
        clock = time.monotonic()
        output = self.agents.ask(
            role=stage.role,
            prompt_name=stage.prompt,
            schema_name=stage.schema,
            values={"CONTEXT_JSON": context_json},
            skills=stage.skills,
            timeout=self.budgets.agent_timeout,
        )
        elapsed = time.monotonic() - clock
        validate_handoff(output)
        rendered = canonical_json(output)
        if len(rendered) > self.budgets.max_output_chars:
            raise WorkflowError(
                f"stage {stage.key} output has {len(rendered)} characters; budget is "
                f"{self.budgets.max_output_chars}"
            )
        turn = {
            "role": stage.role,
            "backend": options.backend,
            "model": options.model,
            "reasoning_effort": options.reasoning_effort,
            "skills": list(stage.skills),
            "started_at": started.isoformat(),
            "duration_seconds": round(elapsed, 1),
            "context_chars": len(context_json),
            "output_chars": len(rendered),
            "outcome": _outcome(output),
            "conclusion": _conclusion(output),
            "summary": self._handoff_summary(output),
        }
        self.store.save_checkpoint(
            stage.key,
            role=stage.role,
            input_sha256=input_sha,
            output=output,
            turn=turn,
        )
        self._record(turn, stage.key, reused=False)
        return output

    def _record(self, turn: dict[str, Any] | None, stage: str, *, reused: bool) -> None:
        entry = dict(turn) if turn else {"role": "unknown"}
        entry["stage"] = stage
        entry["source"] = "reused" if reused else "ran"
        self.ledger.append(entry)

    @staticmethod
    def _handoff_summary(output: Mapping[str, Any]) -> str:
        handoff = output.get("handoff")
        summary = handoff.get("summary") if isinstance(handoff, Mapping) else None
        if not isinstance(summary, str):
            summary = str(output.get("summary", ""))
        return " ".join(summary.split())[:LEDGER_SUMMARY_CHARS]

    def _file_digest(self, directory: str, name: str) -> str:
        path = self.root / directory / name
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise WorkflowError(f"cannot read V2 contract {path}: {exc}") from exc
        return hashlib.sha256(content).hexdigest()

    def _publish_milestone(
        self,
        name: str,
        number: int,
        title: str,
        payload: object,
        attribution: str = "",
    ) -> None:
        if self.store.milestone_complete(name):
            return
        self.store.mark_milestone(name, "pending")
        self.publisher.collect_markers(number)
        self.publisher.comment_once(
            number, name, title, payload, attribution=attribution
        )
        self.store.mark_milestone(name, "complete")

    def _open_or_update_pull_request(
        self,
        title: str,
        specification: dict[str, Any],
        implementation: dict[str, Any],
        documentation: dict[str, Any],
        ci: dict[str, Any],
        reviews: dict[str, Any],
    ) -> str:
        body = self._pull_request_body(
            specification, implementation, documentation, ci, reviews
        )
        existing = self.store.metadata.get("pr_url")
        if isinstance(existing, str) and existing:
            url = existing
        else:
            self.store.mark_milestone("pull-request", "pending")
            url = self.publisher.create_or_find_pr(
                base=self.base_branch,
                branch=self.branch,
                title=title,
                body=body,
                draft=self.draft,
            )
            number = pull_request_number(url)
            self.store.update_metadata(pr_url=url, pr_number=number)
        number = pull_request_number(url)
        self.publisher.update_pr(number, body=body)
        self.store.mark_milestone("pull-request", "complete")
        return url

    def _pull_request_body(
        self,
        specification: dict[str, Any],
        implementation: dict[str, Any],
        documentation: dict[str, Any],
        ci: dict[str, Any],
        reviews: dict[str, Any],
    ) -> str:
        """Assemble the pull-request body, including what the specifier cut.

        Only `specifier_reduction` entries are rendered. An `issue_declared`
        exclusion is already written on the issue this body links to, so
        repeating it here would bury the entries that exist nowhere a human
        reads — the scope the specifier decided on its own to drop. Issue #66
        shipped two such invented deferrals, one of them contradicting a stated
        acceptance criterion, past two approving reviewers precisely because
        the body rendered the criteria and nothing about what had been dropped.

        The section is omitted when there was no reduction rather than
        rendered empty, so a reader learns nothing from its absence and
        everything from its presence.
        """

        criteria = "\n".join(
            f"- {item}" for item in specification["acceptance_criteria"]
        )
        reductions = "\n".join(
            f"- {entry['item']} - {entry['justification']}"
            for entry in specification["out_of_scope"]
            if entry["source"] == "specifier_reduction"
        )
        cut = (
            f"## Scope the specifier deferred\n\n{reductions}\n\n" if reductions else ""
        )
        return (
            f"Closes #{self.issue_number}\n\n"
            f"## Solution\n\n{specification['solution']}\n\n"
            f"## Acceptance criteria\n\n{criteria}\n\n"
            f"{cut}"
            f"## Implementation\n\n{implementation['handoff']['summary']}\n\n"
            f"## Documentation\n\n{documentation['handoff']['summary']}\n\n"
            f"## Validation\n\nFull CI passed with {len(ci.get('gates', []))} reported "
            f"gates. Specification review: {reviews['specification']['verdict']}. "
            f"Quality review: {reviews['quality']['verdict']}.\n\n"
            "Generated by Gabriel's development workflow V2. Detailed handoffs "
            "and CI evidence remain in the local checkpoint store.\n"
        )

    def _ledger_markdown(self) -> str:
        """The run's process record: who ran, on what, for how long, and why.

        A driver that comments per stage can carry these fields in a footer
        under each one. This one posts two comments, so the same information
        is collected into a single table instead of being spread across
        eighteen.
        """

        def cell(value: object) -> str:
            text = str(value).strip() if value is not None else ""
            return f"`{text}`" if text else "_unset_"

        rows = []
        for index, entry in enumerate(self.ledger, 1):
            skills = ", ".join(entry.get("skills") or [])
            duration = entry.get("duration_seconds")
            rows.append(
                f"| {index} | `{entry.get('stage')}` | `{entry.get('role')}` | "
                f"{cell(entry.get('backend'))} | {cell(entry.get('model'))} | "
                f"{cell(entry.get('reasoning_effort'))} | "
                f"{cell(skills) if skills else '_none_'} | "
                f"{f'`{duration}s`' if duration is not None else '_unset_'} | "
                f"{cell(entry.get('source'))} | {entry.get('outcome', '')} |"
            )
        handoffs = "\n".join(
            f"- **`{entry.get('stage')}`** ({entry.get('role')}): "
            f"{entry.get('summary') or '_no summary recorded_'}"
            for entry in self.ledger
        )
        worktree = self.agents.workdir.resolve()
        ran = sum(1 for entry in self.ledger if entry.get("source") == "ran")
        return (
            "\n---\n\n## Run ledger\n\n"
            f"worktree: `{worktree.name}` - `{_shorten_home(worktree)}`  \n"
            f"stages: `{ran}` executed this run, "
            f"`{len(self.ledger) - ran}` reused from checkpoints  \n"
            f"agent-turn budget: `{self.store.metadata.get('turns_used')}` / "
            f"`{self.budgets.max_agent_turns}` "
            "(CI runs are driver work and cost no turn)\n\n"
            "| # | Stage | Role | Backend | Model | Effort | Skills | Duration "
            "| Source | Outcome |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
            + "\n".join(rows)
            + "\n\n<details>\n<summary>What each stage reported</summary>\n\n"
            + handoffs
            + "\n\n</details>\n"
        )

    def _publish_checks(self, head_sha: str) -> None:
        """One check run per stage, so the fleet is visible in the Checks tab.

        Best effort by design: the ledger comment already carries the same
        record, so an App without `checks:write` loses a convenience, not
        evidence, and must not cost a run that otherwise succeeded.
        """

        if self.store.milestone_complete("checks"):
            return
        self.store.mark_milestone("checks", "pending")
        self.publisher.publish_checks(head_sha, self._published_ledger())
        self.store.mark_milestone("checks", "complete")

    def _published_ledger(self) -> list[dict[str, Any]]:
        """The ledger as check runs: a superseded stage is `neutral`.

        Checks are published only after the run committed, pushed, and opened
        its pull request, so every failure in the ledger is one the run went on
        to repair — a red CI attempt that a later attempt replaced, or a
        reviewer that a later round satisfied. Publishing those as `failure`
        leaves the pull request permanently red for work that finished green,
        which branch protection reads as a blocked merge. The stage's own row
        keeps its real conclusion, and the check's title and summary still say
        what happened; only the colour is corrected to match the run.
        """

        return [
            entry
            if entry.get("conclusion") != "failure"
            else {**entry, "conclusion": "neutral"}
            for entry in self.ledger
        ]

    @staticmethod
    def _without_handoff(output: Mapping[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in output.items() if key != "handoff"}

    @staticmethod
    def _ci_summary(ci: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "returncode": ci.get("returncode"),
            "gates": ci.get("gates", []),
        }

    @staticmethod
    def _require_complete(output: Mapping[str, Any], stage: str) -> None:
        if output.get("status") != "complete":
            blockers = output.get("blockers", [])
            if not isinstance(blockers, list):
                blockers = [blockers]
            raise WorkflowStopped(f"{stage} blocked: {'; '.join(map(str, blockers))}")
