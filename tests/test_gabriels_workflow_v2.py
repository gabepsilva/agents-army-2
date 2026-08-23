"""Behavior tests for Gabriel's compact, local-first workflow V2."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from collections import deque
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import yaml

from examples.gabriels_workflow_v2 import cli, setup
from examples.gabriels_workflow_v2.config import (
    AGENT_ROLES,
    BudgetConfig,
    RoleConfig,
    WorkflowConfig,
    load_config,
)
from examples.gabriels_workflow_v2.contracts import (
    CheckpointStore,
    Stage,
    canonical_json,
    digest,
    validate_handoff,
)
from examples.gabriels_workflow_v2.errors import WorkflowError, WorkflowStopped
from examples.gabriels_workflow_v2.gates import CommandResult, GateResult
from examples.gabriels_workflow_v2.publisher import GitHubPublisher
from examples.gabriels_workflow_v2.workflow import (
    DevelopmentWorkflowV2,
    RelayAgentGateway,
    RelayRepository,
    WorkflowOptions,
    WorkflowServices,
    _shorten_home,
)
from orchestrator.schema import load_schema, validate_reply


def _handoff(summary: str = "ready") -> dict[str, Any]:
    return {
        "summary": summary,
        "decisions": ["use existing behavior"],
        "open_questions": [],
        "next_task": "continue with the driver-selected stage",
        "relevant_files": ["feature.py"],
        "required_evidence": ["targeted tests"],
    }


def _proposal(needs_round: bool = False) -> dict[str, Any]:
    return {
        "decision": "proceed",
        "needs_another_round": needs_round,
        "summary": "proposal",
        "current_state": ["old"],
        "proposed_changes": ["new"],
        "risks": [],
        "open_questions": ["question"] if needs_round else [],
        "handoff": _handoff("proposal handoff"),
    }


def _grill(verdict: str = "ready") -> dict[str, Any]:
    return {
        "verdict": verdict,
        "needs_another_round": verdict == "revise",
        "summary": "ambiguity review",
        "questions": ["resolve this"] if verdict == "revise" else [],
        "required_changes": ["decide"] if verdict == "revise" else [],
        "handoff": _handoff("grill handoff"),
    }


def _specification() -> dict[str, Any]:
    return {
        "title": "Implement compact relay",
        "problem_statement": "prompts repeat context",
        "solution": "relay compact evidence",
        "user_stories": ["As a maintainer, I spend fewer tokens"],
        "implementation_decisions": ["use local checkpoints"],
        "testing_decisions": ["exercise resume behavior"],
        "acceptance_criteria": ["all eight roles run once"],
        "out_of_scope": [
            {
                "item": "remove roles",
                "source": "issue_declared",
                "justification": "the issue excludes role changes",
            }
        ],
        "handoff": _handoff("specification handoff"),
    }


def _work(summary: str = "implemented", status: str = "complete") -> dict[str, Any]:
    return {
        "status": status,
        "summary": summary,
        "files_changed": ["feature.py"],
        "tests_run": ["pytest tests/test_feature.py"],
        "blockers": [] if status == "complete" else ["missing decision"],
        "handoff": _handoff(summary),
    }


def _review(verdict: str = "approve", findings: list[dict] | None = None) -> dict:
    return {
        "verdict": verdict,
        "needs_another_round": verdict != "approve",
        "summary": "reviewed",
        "findings": findings or [],
        "handoff": _handoff("review handoff"),
    }


def _finalization(status: str = "complete") -> dict[str, Any]:
    return {
        "status": status,
        "blockers": [] if status == "complete" else ["missing evidence"],
        "summary": "finished with compact handoffs",
        "agreements": ["implementation matches specification"],
        "disagreements": [],
        "feedback_considered": [],
        "implementation_errors": [],
        "fixes_applied": [],
        "opportunities_to_improve": [],
        "handoff": _handoff("human follow-up"),
    }


class FakePublisher:
    def __init__(self) -> None:
        self.issue_calls = 0
        self.checks: list[tuple[str, list[dict[str, Any]]]] = []
        self.collected: list[int] = []
        self.comments: list[tuple[int, str, str, object, str]] = []
        self.pr_calls: list[dict[str, Any]] = []
        self.updates: list[tuple[int, str]] = []
        self.issue_payload: dict[str, Any] | None = None
        self.fail_on_issue = False

    def issue(self, number: int) -> dict[str, Any]:
        if self.fail_on_issue:
            pytest.fail("resume read GitHub issue")
        self.issue_calls += 1
        return self.issue_payload or {
            "number": number,
            "title": "Small issue",
            "body": "Implement the smallest useful feature",
            "labels": ["enhancement"],
            "comments": [],
        }

    def collect_markers(self, number: int) -> None:
        self.collected.append(number)

    def comment_once(
        self,
        number: int,
        key: str,
        title: str,
        payload: object,
        *,
        attribution: str = "",
    ) -> None:
        self.comments.append((number, key, title, payload, attribution))

    def create_or_find_pr(self, **kwargs: Any) -> str:
        self.pr_calls.append(kwargs)
        return "https://example.test/pull/9"

    def update_pr(self, number: int, *, body: str) -> None:
        self.updates.append((number, body))

    def publish_checks(self, head_sha: str, ledger) -> None:
        self.checks.append((head_sha, [dict(entry) for entry in ledger]))


class FakeRepository:
    def __init__(
        self,
        ci: Sequence[CommandResult] = (),
        snapshots: Sequence[str] = (),
    ) -> None:
        self.content = "changed"
        self.ci = deque(ci or [CommandResult(0, "green")])
        self.ci_timeouts: list[int] = []
        self.commits: list[tuple[str, str]] = []
        self.pushes: list[str] = []
        self.required: list[str] = []
        self.snapshots = deque(snapshots)

    def head(self) -> str:
        return "head-sha"

    def snapshot(self) -> str:
        if self.snapshots:
            return self.snapshots.popleft()
        return digest(self.content)

    def run_ci(self, timeout: int) -> CommandResult:
        self.ci_timeouts.append(timeout)
        return self.ci.popleft()

    def require_changed(self, initial_snapshot: str) -> None:
        self.required.append(initial_snapshot)
        if self.snapshot() == initial_snapshot:
            raise WorkflowStopped("implementation did not change the working tree")

    def commit(self, message: str, base_sha: str) -> None:
        self.commits.append((message, base_sha))

    def push(self, branch: str) -> None:
        self.pushes.append(branch)


class FakeAgents:
    def __init__(self, replies: Sequence[dict[str, Any]]) -> None:
        self.replies = deque(replies)
        self.calls: list[dict[str, Any]] = []
        self.workdir = Path("/tmp/gdw-v2-worktree")
        self.role_options = SimpleNamespace(
            backend="codex", model="test-model", reasoning_effort="low"
        )

    def options(self, role: str):
        assert role in AGENT_ROLES
        return self.role_options

    def ask(
        self,
        *,
        role: str,
        prompt_name: str,
        schema_name: str,
        values: Mapping[str, str],
        skills: Sequence[str] = (),
        timeout: int,
    ) -> dict:
        self.calls.append(
            {
                "role": role,
                "prompt": prompt_name,
                "schema": schema_name,
                "values": values,
                "skills": tuple(skills),
                "timeout": timeout,
            }
        )
        return self.replies.popleft()


def _workflow(
    tmp_path: Path,
    replies: Sequence[dict[str, Any]],
    *,
    repository: FakeRepository | None = None,
    publisher: FakePublisher | None = None,
    budgets: BudgetConfig | None = None,
) -> tuple[DevelopmentWorkflowV2, FakePublisher, FakeRepository, FakeAgents]:
    store = CheckpointStore(tmp_path / "state")
    store.initialize(42, "gdwv2/issue-42", "base-sha")
    publisher = publisher or FakePublisher()
    repository = repository or FakeRepository()
    agents = FakeAgents(replies)
    workflow = DevelopmentWorkflowV2(
        WorkflowOptions(
            42,
            "master",
            "gdwv2/issue-42",
            True,
            digest("initial"),
            budgets or BudgetConfig(),
        ),
        WorkflowServices(store, publisher, repository, agents),
    )
    return workflow, publisher, repository, agents


def test_happy_path_is_eight_compact_turns_and_two_github_summaries(
    tmp_path: Path,
) -> None:
    replies = [
        _proposal(),
        _grill(),
        _specification(),
        _work(),
        _work("documented"),
        _review(),
        _review(),
        _finalization(),
    ]
    workflow, publisher, repository, agents = _workflow(tmp_path, replies)

    assert workflow.run() == "https://example.test/pull/9"
    assert [call["role"] for call in agents.calls] == [
        "expander",
        "griller",
        "specifier",
        "implementer",
        "documenter",
        "reviewer-specification",
        "reviewer-quality",
        "finalizer",
    ]
    assert all(set(call["values"]) == {"CONTEXT_JSON"} for call in agents.calls)
    assert [comment[1] for comment in publisher.comments] == [
        "specification",
        "final-summary",
    ]
    assert publisher.issue_calls == 1
    assert repository.ci_timeouts == [7_200]
    assert repository.pushes == ["gdwv2/issue-42"]
    assert "Detailed handoffs" in publisher.updates[0][1]
    assert workflow.store.metadata["turns_used"] == 8

    assert workflow.run() == "https://example.test/pull/9"
    assert len(agents.calls) == 8
    assert publisher.issue_calls == 1
    assert len(publisher.comments) == 2


def test_a_finished_run_records_when_it_completed_for_retention(
    tmp_path: Path,
) -> None:
    replies = [
        _proposal(),
        _grill(),
        _specification(),
        _work(),
        _work("documented"),
        _review(),
        _review(),
        _finalization(),
    ]
    workflow, _publisher, _repository, _agents = _workflow(tmp_path, replies)
    before = datetime.now(UTC)

    assert workflow.run() == "https://example.test/pull/9"

    payload = json.loads(
        (tmp_path / "state" / "workflow.json").read_text(encoding="utf-8")
    )
    assert payload["complete"] is True
    completed_at = datetime.fromisoformat(payload["completed_at"])
    assert completed_at.tzinfo is not None
    assert before <= completed_at <= datetime.now(UTC)


def test_resume_uses_local_issue_and_stage_checkpoints(tmp_path: Path) -> None:
    workflow, publisher, _repository, agents = _workflow(
        tmp_path,
        [_proposal(), _grill(), _specification()],
    )
    issue = workflow._issue_snapshot()
    proposal = workflow._clarify(issue)
    specification = workflow._specify(issue, proposal)

    publisher.fail_on_issue = True
    assert workflow._issue_snapshot() == issue
    assert workflow._clarify(issue) == proposal
    assert workflow._specify(issue, proposal) == specification
    assert len(agents.calls) == 3


def test_checkpoint_hashes_reject_stale_or_tampered_state(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "state")
    store.initialize(1, "branch", "base")
    output = {"handoff": _handoff()}
    store.save_checkpoint("stage", role="expander", input_sha256="input", output=output)
    loaded = store.load_checkpoint("stage", "input")
    assert loaded is not None
    assert loaded.output == output

    with pytest.raises(WorkflowError, match="stale"):
        store.load_checkpoint("stage", "different")
    path = store.checkpoint_path("stage")
    envelope = json.loads(path.read_text())
    envelope["output"]["handoff"]["summary"] = "tampered"
    path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(WorkflowError, match="output hash"):
        store.load_checkpoint("stage", "input")


def test_checkpoint_store_validates_identity_budget_and_keys(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "state")
    store.initialize(1, "branch", "base")
    assert store.reserve_turn(2) == 1
    assert store.reserve_turn(2) == 2
    with pytest.raises(WorkflowError, match="budget exhausted"):
        store.reserve_turn(2)
    with pytest.raises(WorkflowError, match="belongs to"):
        store.initialize(2, "branch", "base")
    with pytest.raises(WorkflowError, match="invalid checkpoint key"):
        store.checkpoint_path("../escape")
    with pytest.raises(WorkflowError, match="invalid milestone"):
        store.mark_milestone("x", "unknown")


def test_handoff_validation_and_canonical_hashing() -> None:
    first = {"b": 2, "a": 1}
    second = {"a": 1, "b": 2}
    assert canonical_json(first) == canonical_json(second)
    assert digest(first) == digest(second)
    assert validate_handoff({"handoff": _handoff()})["handoff"]["summary"] == "ready"
    with pytest.raises(WorkflowError, match="invalid handoff"):
        validate_handoff({"handoff": {}})
    invalid = _handoff()
    invalid["decisions"] = [""]
    with pytest.raises(WorkflowError, match="decisions"):
        validate_handoff({"handoff": invalid})


def test_prompt_and_output_budgets_fail_before_checkpointing(tmp_path: Path) -> None:
    gateway = RelayAgentGateway.__new__(RelayAgentGateway)
    gateway.prompts = (
        Path(__file__).parents[1] / "examples/gabriels_workflow_v2/prompts"
    )
    gateway.max_prompt_chars = 10
    with pytest.raises(WorkflowError, match="budget is 10"):
        gateway._prompt("expand", {"CONTEXT_JSON": "{}"})

    budgets = BudgetConfig(max_output_chars=1_000)
    oversized = _proposal()
    oversized["summary"] = "x" * 1_000
    workflow, _publisher, _repository, _agents = _workflow(
        tmp_path, [oversized], budgets=budgets
    )
    with pytest.raises(WorkflowError, match="output has"):
        workflow._stage(
            Stage("oversized", "expander", "expand", "proposal", {"issue": {}})
        )
    assert not workflow.store.checkpoint_path("oversized").exists()


def test_clarification_and_review_round_budgets_stop_cleanly(tmp_path: Path) -> None:
    clarification, *_ = _workflow(
        tmp_path / "clarification",
        [_proposal(True), _grill("revise")],
        budgets=BudgetConfig(max_clarification_rounds=1),
    )
    with pytest.raises(WorkflowStopped, match="clarification exceeded 1"):
        clarification._clarify({"number": 42})

    repository = FakeRepository(snapshots=["before", "after"])
    review, *_rest, agents = _workflow(
        tmp_path / "review",
        [_review("changes_requested"), _review(), _work("repaired")],
        repository=repository,
        budgets=BudgetConfig(max_review_rounds=1),
    )
    with pytest.raises(WorkflowStopped, match="review exceeded 1"):
        review._review_until_approved(_specification(), {"returncode": 0, "gates": []})
    assert [call["role"] for call in agents.calls] == [
        "reviewer-specification",
        "reviewer-quality",
        "implementer",
    ]


def test_ci_failure_repairs_then_rechecks_changed_snapshot(tmp_path: Path) -> None:
    repository = FakeRepository(
        [CommandResult(1, "failed"), CommandResult(0, "green")],
        ["before", "after", "after"],
    )
    workflow, _publisher, _repository, agents = _workflow(
        tmp_path, [_work("fixed CI")], repository=repository
    )
    result = workflow._ci_until_green(
        "implementation", _specification(), {"implementation": _work()}
    )
    assert result["returncode"] == 0
    assert [call["prompt"] for call in agents.calls] == ["repair"]
    assert repository.ci_timeouts == [7_200, 7_200]


def test_semantically_contradictory_review_is_rejected(tmp_path: Path) -> None:
    finding = {
        "severity": "critical",
        "axis": "correctness",
        "title": "broken",
        "evidence": "feature.py:1",
        "required_change": "fix it",
    }
    workflow, *_ = _workflow(tmp_path, [_review(findings=[finding]), _review()])
    with pytest.raises(WorkflowError, match="approved with findings"):
        workflow._review_until_approved(
            _specification(), {"returncode": 0, "gates": []}
        )


def test_blocked_work_and_unchanged_implementation_stop(tmp_path: Path) -> None:
    workflow, *_ = _workflow(tmp_path / "blocked", [_work(status="blocked")])
    with pytest.raises(WorkflowStopped, match="implementation blocked"):
        workflow._work(
            "implementation",
            "implementer",
            "implement",
            {"specification": _specification()},
            (),
        )

    repository = FakeRepository()
    initial = repository.snapshot()
    with pytest.raises(WorkflowStopped, match="did not change"):
        repository.require_changed(initial)


def test_issue_context_is_bounded_before_becoming_model_input(tmp_path: Path) -> None:
    publisher = FakePublisher()
    publisher.issue_payload = {
        "number": 42,
        "title": "t" * 30_000,
        "body": "b" * 30_000,
        "labels": [],
        "comments": [
            {"author": str(index), "body": "c" * 4_000, "createdAt": str(index)}
            for index in range(20)
        ],
    }
    workflow, *_ = _workflow(tmp_path, [], publisher=publisher)
    issue = workflow._issue_snapshot()
    assert len(issue["body"]) == 20_000
    assert len(issue["comments"]) == 10
    assert all(len(comment["body"]) == 3_000 for comment in issue["comments"])


def test_repository_snapshot_hashes_content_not_commit_metadata(tmp_path: Path) -> None:
    class SnapshotRepository(RelayRepository):
        def _call(self, *_args: str) -> str:
            return "tracked.txt\0"

    (tmp_path / "tracked.txt").write_text("one", encoding="utf-8")
    repository = SnapshotRepository(tmp_path)
    first = repository.snapshot()
    (tmp_path / "tracked.txt").write_text("two", encoding="utf-8")
    assert repository.snapshot() != first
    (tmp_path / "tracked.txt").unlink()
    assert repository.snapshot() not in {first, digest("")}


def test_publisher_reuses_the_branch_existing_pull_request() -> None:
    pull = SimpleNamespace(html_url="https://example.test/pull/7")

    url = _publisher([pull]).create_or_find_pr(
        base="master", branch="feature", title="title", body="body", draft=True
    )

    assert url == pull.html_url


def _publisher(pulls: list[Any], owner: object = SimpleNamespace(login="owner")) -> Any:
    repository = SimpleNamespace(owner=owner, get_pulls=lambda **_kwargs: pulls)
    return GitHubPublisher(cast(Any, repository))


def test_publisher_refuses_ambiguous_or_unusable_pull_request_state() -> None:
    pull = SimpleNamespace(html_url="https://example.test/pull/7")
    arguments = {
        "base": "master",
        "branch": "feature",
        "title": "t",
        "body": "b",
        "draft": True,
    }

    with pytest.raises(WorkflowError, match="repository owner"):
        _publisher([pull], owner=None).create_or_find_pr(**arguments)
    with pytest.raises(WorkflowError, match="multiple open pull requests"):
        _publisher([pull, pull]).create_or_find_pr(**arguments)
    with pytest.raises(WorkflowError, match="invalid existing pull request"):
        _publisher([SimpleNamespace(html_url=None)]).create_or_find_pr(**arguments)


def test_publisher_opens_a_pull_request_only_when_the_branch_has_none() -> None:
    created: list[dict[str, Any]] = []
    publisher = _publisher([])
    publisher.repository.create_pull = lambda **kwargs: (
        created.append(kwargs)
        or SimpleNamespace(html_url="https://example.test/pull/9")
    )

    url = publisher.create_or_find_pr(
        base="master", branch="feature", title="t", body="b", draft=True
    )

    assert url == "https://example.test/pull/9"
    assert created == [
        {
            "base": "master",
            "head": "feature",
            "title": "t",
            "body": "b",
            "draft": True,
        }
    ]


def test_publisher_markers_do_not_collide_with_older_gdw_comments() -> None:
    comments = [
        SimpleNamespace(body="<!-- gdw:7:summary -->\nolder stage comment"),
        SimpleNamespace(body="<!-- gdw-v2:7:specification -->\nV2 milestone"),
    ]
    posted: list[str] = []
    repository = SimpleNamespace(
        get_issue=lambda _number: SimpleNamespace(
            get_comments=lambda: comments,
            create_comment=lambda body: posted.append(body),
        )
    )
    publisher = GitHubPublisher(cast(Any, repository))

    publisher.collect_markers(7)

    assert publisher.markers == {"<!-- gdw-v2:7:specification -->"}
    publisher.comment_once(7, "specification", "Validated specification", {"a": 1})
    publisher.comment_once(7, "final-summary", "Final implementation summary", {"a": 1})
    assert len(posted) == 1
    assert posted[0].startswith("<!-- gdw-v2:7:final-summary -->\n## GDW V2 — ")


def test_publisher_builds_one_check_run_per_stage_with_attribution_fields() -> None:
    created: list[dict[str, Any]] = []
    repository = SimpleNamespace(
        create_check_run=lambda **kwargs: created.append(kwargs)
    )
    publisher = GitHubPublisher(cast(Any, repository))
    ledger = [
        {
            "stage": "implementation",
            "role": "implementer",
            "backend": "claude",
            "model": "sonnet",
            "reasoning_effort": "medium",
            "skills": ["code-simplification", "caveman"],
            "started_at": "2026-08-23T09:00:00+00:00",
            "duration_seconds": 39.2,
            "outcome": "status=complete",
            "conclusion": "success",
            "summary": "implemented the width fix",
            "source": "ran",
        }
    ]

    publisher.publish_checks("abc123", ledger)

    (call,) = created
    assert call["name"] == "gdw-v2 / implementation"
    assert call["head_sha"] == "abc123"
    assert call["status"] == "completed"
    assert call["conclusion"] == "success"
    assert (call["completed_at"] - call["started_at"]).total_seconds() == pytest.approx(
        39.2
    )
    assert call["output"]["title"] == "implementer - status=complete"
    summary = call["output"]["summary"]
    for line in (
        "backend: `claude`",
        "model: `sonnet`",
        "reasoning_effort: `medium`",
        "task_duration: `39.2s`",
        "skills: `code-simplification, caveman`",
        "source: `ran`",
    ):
        assert line in summary
    assert "implemented the width fix" in summary


def test_publisher_marks_unknown_fields_unset_and_defaults_the_conclusion() -> None:
    created: list[dict[str, Any]] = []
    repository = SimpleNamespace(
        create_check_run=lambda **kwargs: created.append(kwargs)
    )

    GitHubPublisher(cast(Any, repository)).publish_checks(
        "abc123",
        [{"stage": "ci-implementation", "role": "driver", "started_at": "not-a-date"}],
    )

    (call,) = created
    assert call["conclusion"] == "neutral"
    assert call["completed_at"] == call["started_at"]
    summary = call["output"]["summary"]
    assert "backend: _unset_" in summary
    assert "skills: _none_" in summary
    assert "task_duration: _unset_" in summary


def test_check_publication_failure_never_costs_a_finished_run(caplog) -> None:
    def refuse(**_kwargs):
        raise RuntimeError("Resource not accessible by integration")

    repository = SimpleNamespace(create_check_run=refuse)

    with caplog.at_level("WARNING", logger="gdw-v2"):
        GitHubPublisher(cast(Any, repository)).publish_checks(
            "abc123", [{"stage": "implementation"}, {"stage": "documentation"}]
        )

    assert "not accessible by integration" in caplog.text
    # It gives up after the first refusal rather than retrying every stage.
    assert caplog.text.count("not published") == 1


def _role_payload() -> dict[str, dict[str, str]]:
    return {
        role: {"backend": " Codex ", "model": " test-model "} for role in AGENT_ROLES
    }


def test_config_reads_one_app_key_from_a_pem_beside_the_config(
    tmp_path: Path,
) -> None:
    (tmp_path / "app.pem").write_text("PEM BODY", encoding="utf-8")
    path = tmp_path / "workflow.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "repository": " owner/project ",
                "github_app": {"app_id": 7, "private_key": "app.pem"},
                "roles": _role_payload(),
            }
        ),
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.repository == "owner/project"
    assert config.github_app.private_key.get_secret_value() == "PEM BODY"
    assert config.roles["implementer"].backend == "codex"
    assert config.roles["implementer"].model == "test-model"
    assert config.budgets.max_agent_turns == 24


def test_config_rejects_unreadable_key_bad_yaml_and_missing_file(
    tmp_path: Path,
) -> None:
    absent = tmp_path / "workflow.yaml"
    absent.write_text(
        yaml.safe_dump(
            {
                "repository": "owner/project",
                "github_app": {"app_id": 7, "private_key": "missing.pem"},
                "roles": _role_payload(),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(WorkflowError, match="cannot read private key"):
        load_config(absent)

    broken = tmp_path / "broken.yaml"
    broken.write_text("repository: [unclosed", encoding="utf-8")
    with pytest.raises(WorkflowError, match="invalid YAML"):
        load_config(broken)

    with pytest.raises(WorkflowError, match="cannot read V2 workflow config"):
        load_config(tmp_path / "nowhere.yaml")

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        yaml.safe_dump({"repository": "owner/project"}), encoding="utf-8"
    )
    with pytest.raises(WorkflowError, match="invalid V2 workflow config"):
        load_config(invalid)


def test_config_requires_exactly_the_known_roles_and_owner_repo() -> None:
    roles = _role_payload()
    app = {"app_id": 1, "private_key": "key"}
    with pytest.raises(ValueError, match="missing roles"):
        WorkflowConfig.model_validate(
            {
                "repository": "owner/project",
                "github_app": app,
                "roles": {"expander": roles["expander"]},
            }
        )
    with pytest.raises(ValueError, match="unknown roles: auditor"):
        WorkflowConfig.model_validate(
            {
                "repository": "owner/project",
                "github_app": app,
                "roles": {**roles, "auditor": roles["expander"]},
            }
        )
    with pytest.raises(ValueError, match="OWNER/REPO"):
        WorkflowConfig.model_validate(
            {"repository": "project", "github_app": app, "roles": roles}
        )
    with pytest.raises(ValueError, match="must not be empty"):
        WorkflowConfig.model_validate(
            {
                "repository": "owner/project",
                "github_app": {"app_id": 1, "private_key": " "},
                "roles": roles,
            }
        )


def test_all_v2_schemas_and_prompts_are_strict_and_resolvable() -> None:
    root = Path(__file__).parents[1] / "examples/gabriels_workflow_v2"
    samples = {
        "proposal": _proposal(),
        "grill": _grill(),
        "specification": _specification(),
        "work-report": _work(),
        "review": _review(),
        "finalization": _finalization(),
    }
    for name, sample in samples.items():
        schema = load_schema(root / "validations" / f"{name}.json")
        assert validate_reply(json.dumps(sample), sample, schema) == sample
    gateway = RelayAgentGateway.__new__(RelayAgentGateway)
    gateway.prompts = root / "prompts"
    gateway.max_prompt_chars = 60_000
    for path in (root / "prompts").glob("*.md"):
        prompt = gateway._prompt(path.stem, {"CONTEXT_JSON": "{}"})
        assert "{{" not in prompt
        assert "untrusted" in prompt
        assert "invoke another agent" in " ".join(prompt.split())


def test_cli_reports_success_and_workflow_errors(monkeypatch, capsys) -> None:
    fake = SimpleNamespace(run=lambda: "https://example.test/pull/1")
    monkeypatch.setattr(cli, "load_config", lambda _path: object())
    monkeypatch.setattr(cli, "prepare_workflow", lambda _issue, _config: fake)
    assert cli.main(["1"]) == 0
    assert capsys.readouterr().out.strip() == "https://example.test/pull/1"

    monkeypatch.setattr(
        cli,
        "prepare_workflow",
        lambda _issue, _config: (_ for _ in ()).throw(WorkflowError("boom")),
    )
    assert cli.main(["1"]) == 1
    assert "V2 workflow stopped: boom" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli._parser().parse_args(["0"])


def test_role_config_rejects_unsupported_or_empty_values() -> None:
    assert RoleConfig(backend=" OpenCode ").backend == "opencode"
    assert RoleConfig(backend="codex").model is None
    with pytest.raises(ValueError, match="claude, codex, grok, opencode"):
        RoleConfig(backend="other")
    with pytest.raises(ValueError, match="must not be empty"):
        RoleConfig(backend="codex", model=" ")


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)


@pytest.fixture
def origin_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway checkout, insulated from any git environment around pytest.

    `GIT_DIR`/`GIT_INDEX_FILE` and friends are exported by git hooks, so a
    suite run from a pre-commit hook would otherwise point every `git` call
    below at the real repository being committed.
    """

    for name in [name for name in os.environ if name.startswith("GIT_")]:
        monkeypatch.delenv(name)
    root = tmp_path / "checkout"
    root.mkdir()
    _git(root, "init", "--initial-branch", "master")
    _git(root, "config", "user.email", "test@example.test")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "commit.gpgsign", "false")
    (root / "README.md").write_text("start\n", encoding="utf-8")
    _git(root, "add", "README.md")
    _git(root, "commit", "-m", "initial")
    return root


def _prepared(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    config: WorkflowConfig,
    built: list[dict[str, Any]] | None = None,
) -> DevelopmentWorkflowV2:
    """Run setup against a real git checkout, faking only GitHub and the CLI."""

    recorded = built if built is not None else []

    def gateway(**kwargs: Any) -> Any:
        recorded.append(kwargs)
        return SimpleNamespace(workdir=kwargs["workdir"])

    monkeypatch.chdir(root)
    monkeypatch.setattr(setup.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        setup.GitHubPublisher,
        "connect",
        classmethod(
            lambda _cls, *_args: SimpleNamespace(default_branch="master", markers=set())
        ),
    )
    monkeypatch.setattr(setup, "RelayAgentGateway", gateway)
    return setup.prepare_workflow(42, config)


def _setup_config() -> WorkflowConfig:
    return WorkflowConfig.model_validate(
        {
            "repository": "owner/project",
            "github_app": {"app_id": 1, "private_key": "key"},
            "roles": _role_payload(),
        }
    )


def test_setup_builds_the_issue_worktree_and_points_agents_at_it(
    origin_checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[dict[str, Any]] = []

    workflow = _prepared(monkeypatch, origin_checkout, _setup_config(), built)

    issue_root = origin_checkout / ".git" / "gdw-v2" / "issue-42"
    worktree = issue_root / "worktree"
    assert worktree.is_dir()
    assert workflow.branch == "gdwv2/issue-42"
    assert workflow.base_branch == "master"
    assert workflow.store.root == issue_root
    assert workflow.store.metadata["issue"] == 42
    (arguments,) = built
    assert arguments["workdir"] == worktree
    assert arguments["state_file"] == issue_root / "agents" / "agents.json"
    assert arguments["max_prompt_chars"] == 60_000
    assert set(arguments["roles"]) == AGENT_ROLES


def test_setup_keeps_the_pre_run_snapshot_when_resuming(
    origin_checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _setup_config()
    first = _prepared(monkeypatch, origin_checkout, config)
    worktree = origin_checkout / ".git" / "gdw-v2" / "issue-42" / "worktree"
    (worktree / "feature.py").write_text("implemented\n", encoding="utf-8")

    second = _prepared(monkeypatch, origin_checkout, config)

    assert second.initial_snapshot == first.initial_snapshot
    assert second.repository.snapshot() != second.initial_snapshot
    second.repository.require_changed(second.initial_snapshot)


def test_setup_names_the_missing_tool_or_backend_before_paying_a_model(
    origin_checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(origin_checkout)
    monkeypatch.setattr(
        setup.GitHubPublisher,
        "connect",
        classmethod(lambda *_args: pytest.fail("connected before checking PATH")),
    )
    config = _setup_config()

    monkeypatch.setattr(
        setup.shutil, "which", lambda name: None if name == "bwrap" else "/usr/bin/x"
    )
    with pytest.raises(WorkflowError, match="bwrap is not installed"):
        setup.prepare_workflow(42, config)

    monkeypatch.setattr(
        setup.shutil, "which", lambda name: None if name == "codex" else "/usr/bin/x"
    )
    with pytest.raises(WorkflowError, match="codex is not installed"):
        setup.prepare_workflow(42, config)


def test_setup_prunes_a_stale_sibling_issue_before_preparing_this_one(
    origin_checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gdw_root = origin_checkout / ".git" / "gdw-v2"
    stale = CheckpointStore(gdw_root / "issue-7")
    stale.initialize(7, "gdwv2/issue-7", "base-sha")
    stale.update_metadata(
        complete=True,
        pr_number=1,
        pr_url="https://example.test/pull/1",
        completed_at=(datetime.now(UTC) - timedelta(days=40)).isoformat(),
    )

    workflow = _prepared(monkeypatch, origin_checkout, _setup_config())

    assert not (gdw_root / "issue-7").exists()
    assert workflow.store.root == gdw_root / "issue-42"
    assert (gdw_root / "issue-42" / "worktree").is_dir()


def test_setup_never_prunes_the_issue_it_is_preparing(
    origin_checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _setup_config()
    first = _prepared(monkeypatch, origin_checkout, config)
    issue_root = origin_checkout / ".git" / "gdw-v2" / "issue-42"
    checkpoint = first.store.checkpoint_path("expansion-1")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("{}", encoding="utf-8")
    (issue_root / "worktree" / "half-done.py").write_text("wip\n", encoding="utf-8")
    CheckpointStore(issue_root).update_metadata(
        complete=True,
        pr_number=1,
        pr_url="https://example.test/pull/1",
        completed_at=(datetime.now(UTC) - timedelta(days=90)).isoformat(),
    )

    second = _prepared(monkeypatch, origin_checkout, config)

    assert (issue_root / "worktree" / "half-done.py").is_file()
    assert checkpoint.is_file()
    assert second.store.metadata["issue"] == 42
    assert second.initial_snapshot == first.initial_snapshot


def test_setup_prepares_the_run_even_when_retention_pruning_fails(
    origin_checkout: Path, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    def refuse(*_args: Any, **_kwargs: Any) -> list[Path]:
        raise WorkflowError("git worktree list --porcelain failed")

    monkeypatch.setattr(setup, "prune_issue_state", refuse)

    with caplog.at_level(logging.WARNING, logger="gdw"):
        workflow = _prepared(monkeypatch, origin_checkout, _setup_config())

    assert (origin_checkout / ".git" / "gdw-v2" / "issue-42" / "worktree").is_dir()
    assert workflow.store.metadata["issue"] == 42
    assert "git worktree list --porcelain failed" in caplog.text


def test_clarification_revises_with_the_review_handoff_then_settles(
    tmp_path: Path,
) -> None:
    workflow, _publisher, _repository, agents = _workflow(
        tmp_path,
        [_proposal(True), _grill("revise"), _proposal(), _grill()],
    )

    proposal = workflow._clarify({"number": 42})

    assert proposal["grill"]["verdict"] == "ready"
    assert [call["prompt"] for call in agents.calls] == [
        "expand",
        "grill",
        "revise",
        "grill",
    ]
    revision = json.loads(agents.calls[2]["values"]["CONTEXT_JSON"])
    assert revision["review_handoff"] == _grill("revise")["handoff"]
    assert "handoff" not in revision["current_proposal"]


def test_clarification_stops_on_stop_reject_and_a_stalled_repeat(
    tmp_path: Path,
) -> None:
    stopping = _proposal()
    stopping["decision"] = "stop"
    stopping["summary"] = "issue is already fixed"
    workflow, *_ = _workflow(tmp_path / "stop", [stopping])
    with pytest.raises(WorkflowStopped, match="issue is already fixed"):
        workflow._clarify({"number": 42})

    rejecting = _grill("reject")
    rejecting["summary"] = "the issue asks for something unsafe"
    workflow, *_ = _workflow(tmp_path / "reject", [_proposal(), rejecting])
    with pytest.raises(WorkflowStopped, match="something unsafe"):
        workflow._clarify({"number": 42})

    workflow, *_ = _workflow(
        tmp_path / "stall",
        [_proposal(True), _grill("revise"), _proposal(True), _grill("revise")],
    )
    with pytest.raises(WorkflowStopped, match="clarification stalled"):
        workflow._clarify({"number": 42})


def test_ci_stops_when_repairs_run_out_or_change_nothing(tmp_path: Path) -> None:
    exhausted = FakeRepository(
        [CommandResult(1, "red"), CommandResult(1, "still red")],
        ["a", "b", "c", "d"],
    )
    workflow, *_ = _workflow(
        tmp_path / "exhausted",
        [_work("first repair"), _work("second repair")],
        repository=exhausted,
        budgets=BudgetConfig(max_ci_attempts=2),
    )
    with pytest.raises(WorkflowStopped, match="CI exceeded 2 attempts"):
        workflow._ci_until_green("implementation", _specification(), {})

    idle = FakeRepository([CommandResult(1, "red")], ["same", "same"])
    workflow, *_ = _workflow(
        tmp_path / "idle", [_work("claimed a repair")], repository=idle
    )
    with pytest.raises(WorkflowStopped, match="CI repair reported complete without"):
        workflow._ci_until_green("implementation", _specification(), {})


def test_review_repair_that_changes_nothing_stops_the_run(tmp_path: Path) -> None:
    repository = FakeRepository(snapshots=["same", "same"])
    workflow, *_ = _workflow(
        tmp_path,
        [_review("changes_requested"), _review(), _work("claimed a repair")],
        repository=repository,
    )

    with pytest.raises(WorkflowStopped, match="review repair reported complete"):
        workflow._review_until_approved(
            _specification(), {"returncode": 0, "gates": []}
        )


def test_blocked_stage_reports_a_non_list_blocker(tmp_path: Path) -> None:
    blocked = _work(status="blocked")
    blocked["blockers"] = "the specification contradicts AGENTS.md"
    workflow, *_ = _workflow(tmp_path, [blocked])

    with pytest.raises(WorkflowStopped, match=r"contradicts AGENTS\.md"):
        workflow._work("implementation", "implementer", "implement", {}, ())


def test_milestones_and_pull_requests_are_published_once(tmp_path: Path) -> None:
    workflow, publisher, _repository, _agents = _workflow(tmp_path, [])

    workflow._publish_milestone("specification", 42, "Specification", {"a": 1})
    workflow._publish_milestone("specification", 42, "Specification", {"a": 1})

    assert len(publisher.comments) == 1
    assert publisher.collected == [42]

    reviews = {"specification": _review(), "quality": _review()}
    arguments = (
        "title",
        _specification(),
        _work(),
        _work(),
        {"returncode": 0, "gates": []},
        reviews,
    )
    first = workflow._open_or_update_pull_request(*arguments)
    second = workflow._open_or_update_pull_request(*arguments)

    assert first == second == "https://example.test/pull/9"
    assert len(publisher.pr_calls) == 1
    assert len(publisher.updates) == 2


def test_checkpoint_store_rejects_corrupt_state_on_disk(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "state")
    store.initialize(1, "branch", "base")

    store.metadata_path.write_text("[]", encoding="utf-8")
    with pytest.raises(WorkflowError, match="is not an object"):
        _ = store.metadata
    store.metadata_path.write_text("{oops", encoding="utf-8")
    with pytest.raises(WorkflowError, match="cannot read V2 workflow state"):
        _ = store.metadata

    store.metadata_path.unlink()
    store.initialize(1, "branch", "base")
    store.update_metadata(turns_used="many")
    with pytest.raises(WorkflowError, match="invalid turn count"):
        store.reserve_turn(4)
    store.update_metadata(turns_used=0, milestones=[])
    with pytest.raises(WorkflowError, match="invalid milestones"):
        store.mark_milestone("specification", "complete")

    store.save_checkpoint("stage", role="expander", input_sha256="in", output={"a": 1})
    envelope = json.loads(store.checkpoint_path("stage").read_text())
    envelope["format_version"] = 99
    store.checkpoint_path("stage").write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(WorkflowError, match="unsupported format"):
        store.load_checkpoint("stage", "in")

    assert store.load_checkpoint("never-written", "in") is None


def test_stored_issue_must_be_an_object_and_handoff_fields_typed() -> None:
    with pytest.raises(WorkflowError, match="not an object"):
        validate_handoff(["not", "an", "object"])
    for field, value in (("summary", " "), ("next_task", "")):
        broken = _handoff()
        broken[field] = value
        with pytest.raises(WorkflowError, match=field):
            validate_handoff({"handoff": broken})


def test_issue_loader_rejects_a_non_object_payload(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "state")
    store.initialize(1, "branch", "base")

    with pytest.raises(WorkflowError, match="issue that was not an object"):
        store.load_or_save_issue(lambda: cast(Any, ["issue"]))


def test_snapshot_covers_symlinks_and_directories(tmp_path: Path) -> None:
    class ListedRepository(RelayRepository):
        def _call(self, *_args: str) -> str:
            return "link\0directory\0gone\0"

    (tmp_path / "target").write_text("target", encoding="utf-8")
    (tmp_path / "link").symlink_to(tmp_path / "target")
    (tmp_path / "directory").mkdir()
    repository = ListedRepository(tmp_path)
    first = repository.snapshot()

    (tmp_path / "link").unlink()
    (tmp_path / "link").symlink_to(tmp_path / "directory")

    assert repository.snapshot() != first


def test_relay_gateway_keeps_the_prompt_budget_it_was_built_with(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "examples.gabriels_workflow_v2.gateway._require_bwrap",
        lambda: "/usr/bin/bwrap",
    )

    gateway = RelayAgentGateway(
        roles={},
        issue=42,
        state_file=tmp_path / "state" / "agents.json",
        example_root=Path(__file__).parents[1] / "examples/gabriels_workflow_v2",
        workdir=tmp_path / "worktree",
        max_prompt_chars=5_000,
    )

    assert gateway.max_prompt_chars == 5_000
    assert gateway.workdir == tmp_path / "worktree"
    assert len(gateway._prompt("expand", {"CONTEXT_JSON": "{}"})) < 5_000


def test_missing_prompt_or_schema_file_is_named(tmp_path: Path) -> None:
    workflow, *_ = _workflow(tmp_path, [])

    with pytest.raises(WorkflowError, match="cannot read V2 contract"):
        workflow._file_digest("prompts", "absent.md")


def test_review_reports_the_ci_run_it_actually_approved(tmp_path: Path) -> None:
    stale = CommandResult(0, "green before review", (GateResult("lint", "passed"),))
    fresh = CommandResult(
        0,
        "green after repair",
        (GateResult("lint", "passed"), GateResult("types", "passed")),
    )
    repository = FakeRepository([fresh], ["r1", "after-repair", "ci-before", "r2"])
    workflow, *_rest, agents = _workflow(
        tmp_path,
        [
            _review("changes_requested"),
            _review(),
            _work("repaired the finding"),
            _review(),
            _review(),
        ],
        repository=repository,
    )

    reviews, ci = workflow._review_until_approved(_specification(), stale.as_json())

    assert all(review["verdict"] == "approve" for review in reviews.values())
    assert ci == fresh.as_json()
    body = workflow._pull_request_body(_specification(), _work(), _work(), ci, reviews)
    assert "2 reported gates" in body
    repair_context = json.loads(agents.calls[2]["values"]["CONTEXT_JSON"])
    assert "handoff" not in repair_context["specification"]


def test_reviewers_are_told_which_commit_to_diff_against(tmp_path: Path) -> None:
    workflow, *_rest, agents = _workflow(tmp_path, [_review(), _review()])

    workflow._review_until_approved(_specification(), {"returncode": 0, "gates": []})

    for call in agents.calls:
        context = json.loads(call["values"]["CONTEXT_JSON"])
        assert context["diff_against"] == "base-sha"
        assert "handoff" not in context["canonical_specification"]


def test_ledger_records_how_every_turn_ran_and_survives_a_resume(
    tmp_path: Path,
) -> None:
    replies = [
        _proposal(),
        _grill(),
        _specification(),
        _work(),
        _work("documented"),
        _review(),
        _review(),
        _finalization(),
    ]
    workflow, publisher, _repository, _agents = _workflow(tmp_path, replies)

    workflow.run()

    paid = [entry for entry in workflow.ledger if entry["source"] == "ran"]
    assert len(paid) == 9  # eight agent turns plus the driver's CI run
    expander = paid[0]
    assert expander["role"] == "expander"
    assert expander["backend"] == "codex"
    assert expander["model"] == "test-model"
    assert expander["reasoning_effort"] == "low"
    assert expander["duration_seconds"] >= 0
    assert expander["outcome"].startswith("decision=proceed")
    assert expander["summary"] == "proposal handoff"
    assert [entry["role"] for entry in workflow.ledger if entry["role"] == "driver"]

    implementer = next(e for e in paid if e["stage"] == "implementation")
    assert implementer["skills"] == ["code-simplification", "caveman"]

    # A second process re-reads the same checkpoints and reports the turns the
    # first one paid for, rather than an empty ledger.
    resumed = DevelopmentWorkflowV2(
        WorkflowOptions(
            42, "master", "gdwv2/issue-42", True, digest("initial"), BudgetConfig()
        ),
        WorkflowServices(workflow.store, publisher, FakeRepository(), FakeAgents([])),
    )
    resumed.store.update_metadata(complete=False)
    resumed.run()
    assert [entry["source"] for entry in resumed.ledger] == ["reused"] * len(
        resumed.ledger
    )
    assert (
        next(e for e in resumed.ledger if e["stage"] == "expansion-1")["model"]
        == "test-model"
    )


def test_final_summary_carries_the_ledger_table_and_stage_summaries(
    tmp_path: Path,
) -> None:
    workflow, publisher, _repository, _agents = _workflow(
        tmp_path,
        [
            _proposal(),
            _grill(),
            _specification(),
            _work(),
            _work("documented"),
            _review(),
            _review(),
            _finalization(),
        ],
    )

    workflow.run()

    final = next(c for c in publisher.comments if c[1] == "final-summary")
    ledger = final[4]
    assert "## Run ledger" in ledger
    assert "| # | Stage | Role | Backend | Model | Effort |" in ledger
    assert "`expander`" in ledger
    assert "`finalizer`" in ledger
    assert "agent-turn budget: `8`" in ledger
    assert "worktree: `gdw-v2-worktree`" in ledger
    assert "<details>" in ledger
    assert "proposal handoff" in ledger
    # The specification milestone stays a clean decision record.
    specification = next(c for c in publisher.comments if c[1] == "specification")
    assert specification[4] == ""


def test_check_runs_are_published_once_per_stage_against_the_pushed_commit(
    tmp_path: Path,
) -> None:
    workflow, publisher, _repository, _agents = _workflow(
        tmp_path,
        [
            _proposal(),
            _grill(),
            _specification(),
            _work(),
            _work("documented"),
            _review(),
            _review(),
            _finalization(),
        ],
    )

    workflow.run()

    assert len(publisher.checks) == 1
    head_sha, ledger = publisher.checks[0]
    assert head_sha == "head-sha"
    assert [entry["stage"] for entry in ledger][:2] == ["expansion-1", "grill-1"]
    assert all(
        entry["conclusion"] in {"success", "failure", "neutral"} for entry in ledger
    )

    workflow.store.update_metadata(complete=False)
    workflow.run()
    assert len(publisher.checks) == 1


def test_repaired_ci_attempt_publishes_neutral_not_failure(tmp_path: Path) -> None:
    """A red attempt the run went on to fix must not leave the PR red.

    Checks are published only after the pull request is open, so a `failure`
    row is always one a later stage resolved. Publishing it as a failure blocks
    merge on work that finished green; the ledger keeps the real conclusion.
    """

    repository = FakeRepository(
        [CommandResult(1, "failed"), CommandResult(0, "green")],
        ["before", "after", "after", "after", "after"],
    )
    workflow, publisher, _repository, _agents = _workflow(
        tmp_path,
        [
            _proposal(),
            _grill(),
            _specification(),
            _work(),
            _work("documented"),
            _work("repaired CI"),
            _review(),
            _review(),
            _finalization(),
        ],
        repository=repository,
    )

    workflow.run()

    ledger = {entry["stage"]: entry["conclusion"] for entry in workflow.ledger}
    published = {
        entry["stage"]: entry["conclusion"] for entry in publisher.checks[0][1]
    }
    failed = [stage for stage, verdict in ledger.items() if verdict == "failure"]
    assert failed, "the run under test is meant to fail CI once"
    assert all(published[stage] == "neutral" for stage in failed)
    assert "failure" not in published.values()


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ({"status": "complete"}, "success"),
        ({"verdict": "approve"}, "success"),
        ({"decision": "proceed"}, "success"),
        ({"status": "blocked"}, "failure"),
        ({"verdict": "reject"}, "failure"),
        ({"decision": "stop"}, "failure"),
        ({"verdict": "revise"}, "neutral"),
        ({"verdict": "changes_requested"}, "neutral"),
    ],
)
def test_stage_conclusion_maps_replies_to_check_run_verdicts(
    output: dict[str, Any], expected: str
) -> None:
    from examples.gabriels_workflow_v2.workflow import _conclusion

    assert _conclusion(output) == expected


def test_gate_digest_names_the_failing_gates() -> None:
    from examples.gabriels_workflow_v2.workflow import _gate_digest

    green = {
        "returncode": 0,
        "gates": [{"name": "lint", "status": "passed"}],
    }
    red = {
        "returncode": 2,
        "gates": [
            {"name": "lint", "status": "passed"},
            {"name": "types", "status": "failed"},
        ],
    }
    assert _gate_digest(green) == "1/1 gates passed"
    assert _gate_digest(red) == "1/2 gates passed; failed: types"
    assert _gate_digest({"returncode": 2, "gates": []}) == "make ci exited 2"


def test_shorten_home_resolves_home_boundaries_and_symlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))

    target = home / "resolved" / "worktree"
    target.mkdir(parents=True)
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    outside = tmp_path / "outside" / "worktree"

    assert _shorten_home(home) == "~"
    assert _shorten_home(home / "project" / "worktree") == "~/project/worktree"
    assert _shorten_home(link) == "~/resolved/worktree"
    assert _shorten_home(outside) == str(outside.resolve())
