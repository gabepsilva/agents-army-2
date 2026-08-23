"""Driver-mediated compact relay for Gabriel's development workflow V2."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from examples.gabriels_workflow.development_workflow import (
    AgentGateway as V1AgentGateway,
)
from examples.gabriels_workflow.development_workflow import (
    CommandResult,
    RoleOptions,
    WorkflowError,
    WorkflowStopped,
    _pull_request_number,
)
from examples.gabriels_workflow.development_workflow import (
    GitRepository as V1GitRepository,
)
from examples.gabriels_workflow_v2.config import BudgetConfig
from examples.gabriels_workflow_v2.contracts import (
    CheckpointStore,
    Stage,
    canonical_json,
    digest,
    validate_handoff,
)

ISSUE_BODY_CHARS = 20_000
ISSUE_COMMENT_CHARS = 3_000
ISSUE_COMMENTS = 10


class Publisher(Protocol):
    markers: set[str]

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


class Repository(Protocol):
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


class RelayAgentGateway(V1AgentGateway):
    """Reuse the hardened V1 sandbox while enforcing a bounded V2 prompt."""

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


class RelayRepository(V1GitRepository):
    """V1 git/CI mechanics plus a commit-independent content fingerprint."""

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
        reviews, ci = self._review_until_approved(specification, ci)
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
        number = _pull_request_number(url)
        self._publish_milestone(
            "final-summary",
            number,
            "Final implementation summary",
            self._without_handoff(finalization),
        )
        self.store.update_metadata(complete=True, pr_number=number, pr_url=url)
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
        raise WorkflowStopped(
            f"clarification exceeded {self.budgets.max_clarification_rounds} rounds"
        )

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
            result = self.store.load_checkpoint(key, input_sha)
            if result is None:
                result = self.repository.run_ci(self.budgets.ci_timeout).as_json()
                self.store.save_checkpoint(
                    key, role="driver", input_sha256=input_sha, output=result
                )
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
        self, specification: dict[str, Any], ci: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Review, repair, and re-run CI until both reviewers approve.

        Returns the approving reviews together with the CI evidence they were
        approved against, which is not the run this started with once a
        repair round has happened.
        """

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
            return validate_handoff(cached)
        self.store.reserve_turn(self.budgets.max_agent_turns)
        output = self.agents.ask(
            role=stage.role,
            prompt_name=stage.prompt,
            schema_name=stage.schema,
            values={"CONTEXT_JSON": canonical_json(stage.context)},
            skills=stage.skills,
            timeout=self.budgets.agent_timeout,
        )
        validate_handoff(output)
        size = len(canonical_json(output))
        if size > self.budgets.max_output_chars:
            raise WorkflowError(
                f"stage {stage.key} output has {size} characters; budget is "
                f"{self.budgets.max_output_chars}"
            )
        self.store.save_checkpoint(
            stage.key, role=stage.role, input_sha256=input_sha, output=output
        )
        return output

    def _file_digest(self, directory: str, name: str) -> str:
        path = self.root / directory / name
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise WorkflowError(f"cannot read V2 contract {path}: {exc}") from exc
        return hashlib.sha256(content).hexdigest()

    def _publish_milestone(
        self, name: str, number: int, title: str, payload: object
    ) -> None:
        if self.store.milestone_complete(name):
            return
        self.store.mark_milestone(name, "pending")
        self.publisher.collect_markers(number)
        self.publisher.comment_once(number, name, title, payload)
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
            number = _pull_request_number(url)
            self.store.update_metadata(pr_url=url, pr_number=number)
        number = _pull_request_number(url)
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
        criteria = "\n".join(
            f"- {item}" for item in specification["acceptance_criteria"]
        )
        return (
            f"Closes #{self.issue_number}\n\n"
            f"## Solution\n\n{specification['solution']}\n\n"
            f"## Acceptance criteria\n\n{criteria}\n\n"
            f"## Implementation\n\n{implementation['handoff']['summary']}\n\n"
            f"## Documentation\n\n{documentation['handoff']['summary']}\n\n"
            f"## Validation\n\nFull CI passed with {len(ci.get('gates', []))} reported "
            f"gates. Specification review: {reviews['specification']['verdict']}. "
            f"Quality review: {reviews['quality']['verdict']}.\n\n"
            "Generated by Gabriel's development workflow V2. Detailed handoffs "
            "and CI evidence remain in the local checkpoint store.\n"
        )

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
