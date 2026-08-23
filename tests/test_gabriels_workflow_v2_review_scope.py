"""Behavior tests for the V2 specification reviewer's scope-drift audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from examples.gabriels_workflow_v2.config import BudgetConfig
from examples.gabriels_workflow_v2.errors import WorkflowError, WorkflowStopped
from examples.gabriels_workflow_v2.gates import CommandResult, GateResult
from examples.gabriels_workflow_v2.workflow import RelayAgentGateway
from orchestrator.schema import load_schema, validate_reply
from tests.test_gabriels_workflow_v2 import (
    FakePublisher,
    FakeRepository,
    _finalization,
    _grill,
    _proposal,
    _review,
    _specification,
    _work,
    _workflow,
)

GREEN = {"returncode": 0, "gates": []}


def _issue_payload(
    body: str = "", comments: list[dict[str, Any]] | None = None
) -> dict:
    return {
        "number": 42,
        "title": "State directories are never pruned",
        "body": body
        or (
            "Acceptance criteria:\n"
            "- A completed run's state directory is removed without manual "
            "intervention.\n"
            "Out of scope:\n- Changing the checkpoint format.\n"
        ),
        "labels": ["bug"],
        "comments": comments or [],
    }


def _contexts(agents: Any) -> dict[str, dict[str, Any]]:
    """The context each role was actually handed, keyed by role."""

    return {
        call["role"]: json.loads(call["values"]["CONTEXT_JSON"])
        for call in agents.calls
    }


def _review_workflow(
    tmp_path: Path,
    replies: list[dict[str, Any]],
    *,
    repository: FakeRepository | None = None,
    budgets: BudgetConfig | None = None,
) -> tuple[Any, FakePublisher, FakeRepository, Any]:
    publisher = FakePublisher()
    publisher.issue_payload = _issue_payload()
    return _workflow(
        tmp_path,
        replies,
        repository=repository,
        publisher=publisher,
        budgets=budgets,
    )


def test_specification_reviewer_is_given_the_issue_it_must_audit_against(
    tmp_path: Path,
) -> None:
    workflow, _publisher, _repository, agents = _review_workflow(
        tmp_path, [_review(), _review()]
    )
    issue = workflow._issue_snapshot()

    workflow._review_until_approved(_specification(), GREEN, issue)

    context = _contexts(agents)["reviewer-specification"]
    assert context["canonical_issue"] == issue
    assert "removed without manual intervention" in context["canonical_issue"]["body"]
    assert "Changing the checkpoint format" in context["canonical_issue"]["body"]
    assert context["canonical_specification"]["out_of_scope"] == ["remove roles"]


def test_specification_review_checkpoint_is_bound_to_the_issue_it_saw(
    tmp_path: Path,
) -> None:
    workflow, _publisher, _repository, agents = _review_workflow(
        tmp_path, [_review(), _review()]
    )
    issue = workflow._issue_snapshot()
    workflow._review_until_approved(_specification(), GREEN, issue)
    assert len(agents.calls) == 2

    workflow._review_until_approved(_specification(), GREEN, issue)
    assert len(agents.calls) == 2

    edited = {**issue, "body": "Out of scope:\n- Removing state directories.\n"}
    with pytest.raises(WorkflowError, match="stale"):
        workflow._review_until_approved(_specification(), GREEN, edited)


def test_quality_reviewer_is_left_out_of_the_scope_audit(tmp_path: Path) -> None:
    workflow, _publisher, _repository, agents = _review_workflow(
        tmp_path, [_review(), _review()]
    )

    workflow._review_until_approved(_specification(), GREEN, workflow._issue_snapshot())

    quality = _contexts(agents)["reviewer-quality"]
    assert set(quality) == {"canonical_specification", "diff_against", "ci"}
    assert [call["skills"] for call in agents.calls] == [
        (),
        ("code-review-and-quality", "code-simplification"),
    ]


def test_review_without_an_issue_argument_reads_the_stored_snapshot(
    tmp_path: Path,
) -> None:
    workflow, publisher, _repository, agents = _review_workflow(
        tmp_path, [_review(), _review()]
    )

    workflow._review_until_approved(_specification(), GREEN)

    assert _contexts(agents)["reviewer-specification"]["canonical_issue"] == (
        workflow._issue_snapshot()
    )
    assert publisher.issue_calls == 1


def test_run_hands_the_reviewer_the_same_issue_the_specifier_saw(
    tmp_path: Path,
) -> None:
    workflow, publisher, _repository, agents = _review_workflow(
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

    assert workflow.run() == "https://example.test/pull/9"

    contexts = _contexts(agents)
    assert (
        contexts["reviewer-specification"]["canonical_issue"]
        == contexts["specifier"]["canonical_issue"]
    )
    assert publisher.issue_calls == 1


def test_approving_reviews_still_end_the_loop_with_their_ci_evidence(
    tmp_path: Path,
) -> None:
    ci = CommandResult(0, "green", (GateResult("lint", "passed"),)).as_json()
    workflow, _publisher, _repository, agents = _review_workflow(
        tmp_path, [_review(), _review()]
    )

    reviews, approved_ci = workflow._review_until_approved(_specification(), ci)

    assert set(reviews) == {"specification", "quality"}
    assert all(review["verdict"] == "approve" for review in reviews.values())
    assert approved_ci == ci
    assert [call["role"] for call in agents.calls] == [
        "reviewer-specification",
        "reviewer-quality",
    ]


def test_changes_requested_still_repairs_reruns_ci_and_rereviews(
    tmp_path: Path,
) -> None:
    stale = CommandResult(0, "green before review", (GateResult("lint", "passed"),))
    fresh = CommandResult(
        0,
        "green after repair",
        (GateResult("lint", "passed"), GateResult("types", "passed")),
    )
    repository = FakeRepository([fresh], ["r1", "after-repair", "ci-before", "r2"])
    workflow, _publisher, _repository, agents = _review_workflow(
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

    reviews, ci = workflow._review_until_approved(
        _specification(), stale.as_json(), workflow._issue_snapshot()
    )

    assert all(review["verdict"] == "approve" for review in reviews.values())
    assert ci == fresh.as_json()
    assert [call["role"] for call in agents.calls] == [
        "reviewer-specification",
        "reviewer-quality",
        "implementer",
        "reviewer-specification",
        "reviewer-quality",
    ]
    repair_context = json.loads(agents.calls[2]["values"]["CONTEXT_JSON"])
    assert set(repair_context) == {"specification", "failure_evidence"}


def test_specification_review_approved_with_findings_is_still_rejected(
    tmp_path: Path,
) -> None:
    drift = {
        "severity": "required",
        "axis": "specification",
        "title": "specification narrowed an issue acceptance criterion",
        "evidence": "issue: removed without manual intervention",
        "required_change": "restore the criterion or record the deferral",
    }
    workflow, *_rest = _review_workflow(
        tmp_path, [_review(findings=[drift]), _review()]
    )

    with pytest.raises(WorkflowError, match="specification review approved with"):
        workflow._review_until_approved(
            _specification(), GREEN, workflow._issue_snapshot()
        )


def test_scope_drift_findings_fit_the_review_schema_unchanged(tmp_path: Path) -> None:
    drift = _review(
        "changes_requested",
        findings=[
            {
                "severity": "required",
                "axis": "specification",
                "title": "out_of_scope entry the issue never excluded",
                "evidence": "specification out_of_scope[1] vs issue body",
                "required_change": "drop the entry or surface it for a human",
            }
        ],
    )
    schema = load_schema(
        Path(__file__).parents[1]
        / "examples/gabriels_workflow_v2/validations/review.json"
    )

    assert validate_reply(json.dumps(drift), drift, schema) == drift

    workflow, _publisher, _repository, agents = _review_workflow(
        tmp_path,
        [drift, _review(), _work("repaired")],
        repository=FakeRepository(snapshots=["before", "after"]),
        budgets=BudgetConfig(max_review_rounds=1),
    )
    with pytest.raises(WorkflowStopped, match="review exceeded 1"):
        workflow._review_until_approved(
            _specification(), GREEN, workflow._issue_snapshot()
        )
    assert [call["role"] for call in agents.calls] == [
        "reviewer-specification",
        "reviewer-quality",
        "implementer",
    ]


def test_specification_review_prompt_stays_inside_the_default_budget(
    tmp_path: Path,
) -> None:
    body = "Acceptance criteria:\n" + ("- keep the state directory bounded.\n" * 200)
    comments = [
        {"author": "gabe", "body": "context " * 200, "createdAt": "2026-08-01"}
        for _ in range(10)
    ]
    publisher = FakePublisher()
    publisher.issue_payload = _issue_payload(body, comments)
    workflow, _publisher, _repository, agents = _workflow(
        tmp_path, [_review(), _review()], publisher=publisher
    )
    specification = _specification()
    specification["acceptance_criteria"] = [
        f"criterion {index}: observable behavior stated in full" for index in range(30)
    ]
    specification["implementation_decisions"] = [
        f"decision {index}: recorded with its reasoning" for index in range(30)
    ]
    ci = CommandResult(
        0, "green", tuple(GateResult(f"gate-{index}", "passed") for index in range(20))
    ).as_json()

    workflow._review_until_approved(specification, ci, workflow._issue_snapshot())

    gateway = RelayAgentGateway.__new__(RelayAgentGateway)
    gateway.prompts = (
        Path(__file__).parents[1] / "examples/gabriels_workflow_v2/prompts"
    )
    gateway.max_prompt_chars = BudgetConfig().max_prompt_chars
    prompt = gateway._prompt("review-specification", agents.calls[0]["values"])

    assert "keep the state directory bounded" in prompt
    assert len(prompt) < BudgetConfig().max_prompt_chars
