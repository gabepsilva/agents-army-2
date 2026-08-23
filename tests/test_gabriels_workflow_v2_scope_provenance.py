"""Behavior tests for the provenance carried by every `out_of_scope` entry.

A specification narrowed at the specifier stage used to be indistinguishable
from an issue that was narrow to begin with: both arrived as bare strings, and
every stage after `_specify` read the specification as ground truth. These
tests pin the two halves of the fix — the schema refuses an untagged entry, and
the pull-request body names what the specifier chose to drop.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from orchestrator.schema import ReplyValidationError, load_schema, validate_reply
from tests.test_gabriels_workflow_v2 import _review, _specification, _work, _workflow

SPECIFICATION_SCHEMA = (
    Path(__file__).parents[1]
    / "examples/gabriels_workflow_v2/validations/specification.json"
)


def _validate(specification: dict[str, Any]) -> dict[str, Any]:
    schema = load_schema(SPECIFICATION_SCHEMA)
    return validate_reply(json.dumps(specification), specification, schema)


def _with_scope(*entries: dict[str, Any]) -> dict[str, Any]:
    return {**_specification(), "out_of_scope": list(entries)}


def _reduction(
    item: str = "no scheduling of prune into the Makefile target",
    justification: str = "wiring is a follow-up the issue did not ask for",
) -> dict[str, Any]:
    return {
        "item": item,
        "source": "specifier_reduction",
        "justification": justification,
    }


def _declared(
    item: str = "no change to the sandbox isolation model",
    justification: str = "the issue's out-of-scope section says exactly this",
) -> dict[str, Any]:
    return {"item": item, "source": "issue_declared", "justification": justification}


def test_an_out_of_scope_entry_without_provenance_is_rejected() -> None:
    with pytest.raises(ReplyValidationError, match="out_of_scope"):
        _validate({**_specification(), "out_of_scope": ["remove roles"]})

    untagged = {"item": "remove roles", "justification": "not worth the churn"}
    with pytest.raises(ReplyValidationError, match="source"):
        _validate(_with_scope(untagged))

    invented_source = {**_reduction(), "source": "proposal"}
    with pytest.raises(ReplyValidationError, match="specifier_reduction"):
        _validate(_with_scope(invented_source))

    unjustified = {"item": "remove roles", "source": "specifier_reduction"}
    with pytest.raises(ReplyValidationError, match="justification"):
        _validate(_with_scope(unjustified))


def test_both_scope_sources_validate_when_well_formed() -> None:
    both = _with_scope(_declared(), _reduction())

    assert _validate(both) == both
    assert _validate(_with_scope()) == _with_scope()


def _body(workflow: Any, specification: dict[str, Any]) -> str:
    return workflow._pull_request_body(
        specification,
        _work(),
        _work("documented"),
        {"gates": []},
        {"specification": _review(), "quality": _review()},
    )


def test_pull_request_body_names_scope_the_specifier_chose_to_cut(
    tmp_path: Path,
) -> None:
    workflow, _publisher, _repository, _agents = _workflow(tmp_path, [])

    body = _body(workflow, _with_scope(_declared(), _reduction()))

    assert "## Scope the specifier deferred" in body
    assert "- no scheduling of prune into the Makefile target - " in body
    assert "wiring is a follow-up the issue did not ask for" in body
    assert "no change to the sandbox isolation model" not in body
    assert "## Implementation" in body


def test_pull_request_body_stays_quiet_when_nothing_was_cut(tmp_path: Path) -> None:
    workflow, _publisher, _repository, _agents = _workflow(tmp_path, [])

    declared_only = _body(workflow, _with_scope(_declared()))
    empty = _body(workflow, _with_scope())

    for body in (declared_only, empty):
        assert "Scope the specifier deferred" not in body
        assert "deferred" not in body
        assert "## Acceptance criteria" in body
        assert "## Implementation" in body
