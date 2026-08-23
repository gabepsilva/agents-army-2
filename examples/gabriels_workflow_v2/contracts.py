"""Local checkpoint and handoff contracts for Gabriel's workflow V2."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from examples.gabriels_workflow_v2.errors import WorkflowError

FORMAT_VERSION = 1
BLOCKING_SEVERITIES = frozenset({"critical", "required"})
HANDOFF_FIELDS = frozenset(
    {
        "summary",
        "decisions",
        "open_questions",
        "next_task",
        "relevant_files",
        "required_evidence",
    }
)


def canonical_json(value: object) -> str:
    """Return stable compact JSON suitable for hashing and prompt handoffs."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def validate_handoff(output: object) -> dict[str, Any]:
    """Validate the common relay envelope even when loading a checkpoint."""

    if not isinstance(output, dict):
        raise WorkflowError("agent output is not an object")
    handoff = output.get("handoff")
    if not isinstance(handoff, dict) or set(handoff) != HANDOFF_FIELDS:
        raise WorkflowError("agent output has an invalid handoff")
    if not isinstance(handoff["summary"], str) or not handoff["summary"].strip():
        raise WorkflowError("agent handoff summary must be non-empty")
    if not isinstance(handoff["next_task"], str) or not handoff["next_task"].strip():
        raise WorkflowError("agent handoff next_task must be non-empty")
    for field in HANDOFF_FIELDS - {"summary", "next_task"}:
        values = handoff[field]
        if not isinstance(values, list) or any(
            not isinstance(item, str) or not item.strip() for item in values
        ):
            raise WorkflowError(f"agent handoff {field} must be an array of strings")
    return output


def blocking_findings(review: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The subset of a review's findings severe enough to require repair."""

    return [
        finding
        for finding in review.get("findings", [])
        if finding.get("severity") in BLOCKING_SEVERITIES
    ]


def check_review_consistency(kind: str, review: Mapping[str, Any]) -> None:
    """Fail closed when a review's verdict disagrees with its own findings.

    Only three combinations are rejected: an `approve` carrying a blocking
    finding, a `changes_requested` carrying none, and a round that asks for
    another pass without one either. Every other schema-valid combination —
    including `changes_requested` with a blocking finding regardless of
    `needs_another_round`, or `approve` with only non-blocking findings — is
    accepted by construction and resolved by the verdict-based control flow
    that follows this check.
    """

    blocking = bool(blocking_findings(review))
    if review["verdict"] == "approve" and blocking:
        raise WorkflowError(f"{kind} review approved with a blocking finding")
    if review["verdict"] == "changes_requested" and not blocking:
        raise WorkflowError(
            f"{kind} review requested changes without a blocking finding"
        )
    if review["needs_another_round"] and not blocking:
        raise WorkflowError(
            f"{kind} review needs another round without a blocking finding"
        )


@dataclass(frozen=True)
class Checkpoint:
    """A stage's validated output plus how the turn that produced it ran."""

    output: dict[str, Any]
    turn: dict[str, Any] | None


@dataclass(frozen=True)
class Stage:
    key: str
    role: str
    prompt: str
    schema: str
    context: dict[str, Any]
    skills: tuple[str, ...] = ()


class CheckpointStore:
    """Atomic local state whose hashes make stale handoffs fail closed."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.checkpoints = root / "checkpoints"
        self.metadata_path = root / "workflow.json"
        self.issue_path = root / "issue.json"

    @property
    def initialized(self) -> bool:
        return self.metadata_path.exists()

    def initialize(self, issue: int, branch: str, base_sha: str) -> None:
        expected = {
            "format_version": FORMAT_VERSION,
            "issue": issue,
            "branch": branch,
            "base_sha": base_sha,
        }
        if self.initialized:
            metadata = self.metadata
            actual = {key: metadata.get(key) for key in expected}
            if actual != expected:
                raise WorkflowError(
                    f"V2 workflow state belongs to {actual}, not {expected}"
                )
            return
        self._write(
            self.metadata_path,
            {
                **expected,
                "turns_used": 0,
                "milestones": {},
                "pr_number": None,
                "pr_url": None,
                "complete": False,
            },
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return self._read(self.metadata_path)

    def update_metadata(self, **changes: object) -> None:
        metadata = self.metadata
        metadata.update(changes)
        self._write(self.metadata_path, metadata)

    def reserve_turn(self, limit: int) -> int:
        metadata = self.metadata
        used = metadata.get("turns_used")
        if not isinstance(used, int) or isinstance(used, bool) or used < 0:
            raise WorkflowError("V2 workflow state has an invalid turn count")
        if used >= limit:
            raise WorkflowError(f"agent-turn budget exhausted ({used}/{limit})")
        used += 1
        metadata["turns_used"] = used
        self._write(self.metadata_path, metadata)
        return used

    def load_or_save_issue(
        self, loader: Callable[[], dict[str, Any]]
    ) -> dict[str, Any]:
        if self.issue_path.exists():
            return self._read(self.issue_path)
        issue = loader()
        if not isinstance(issue, dict):
            raise WorkflowError("GitHub returned an issue that was not an object")
        self._write(self.issue_path, issue)
        return issue

    def discard_for_new_issue(self, keys: Iterable[str]) -> None:
        """Forget the cached issue and the checkpoints derived from it.

        Escalation hands a run back to a human, who answers on the issue
        itself. Keeping the snapshot would replay the next run against text
        that predates the answer, and keeping checkpoints whose recorded
        inputs still hash that text would fail `load_checkpoint`'s stale-input
        rule rather than resume past it. Checkpoints go first: interrupted
        after them, the next run merely re-does clarification, while the
        reverse order would leave exactly the stale pair this avoids.

        Metadata is deliberately untouched. The turns already paid for still
        count against the budget, so a run that keeps escalating runs out
        rather than buying itself unlimited rounds.
        """

        for key in keys:
            self.checkpoint_path(key).unlink(missing_ok=True)
        self.issue_path.unlink(missing_ok=True)

    def load_checkpoint(self, key: str, input_sha256: str) -> Checkpoint | None:
        path = self.checkpoint_path(key)
        if not path.exists():
            return None
        envelope = self._read(path)
        if envelope.get("format_version") != FORMAT_VERSION:
            raise WorkflowError(f"checkpoint {key} has an unsupported format")
        if envelope.get("input_sha256") != input_sha256:
            raise WorkflowError(
                f"checkpoint {key} is stale; its canonical inputs changed"
            )
        output = envelope.get("output")
        if envelope.get("output_sha256") != digest(output):
            raise WorkflowError(f"checkpoint {key} failed its output hash")
        if not isinstance(output, dict):
            raise WorkflowError(f"checkpoint {key} output is not an object")
        # The turn is how the stage ran, not what it decided, so it is kept
        # outside the hashed output and is optional: a checkpoint written
        # before the ledger existed still loads, and reports what it knows.
        turn = envelope.get("turn")
        return Checkpoint(output, turn if isinstance(turn, dict) else None)

    def save_checkpoint(
        self,
        key: str,
        *,
        role: str,
        input_sha256: str,
        output: dict[str, Any],
        turn: dict[str, Any] | None = None,
    ) -> None:
        envelope = {
            "format_version": FORMAT_VERSION,
            "stage": key,
            "role": role,
            "input_sha256": input_sha256,
            "output_sha256": digest(output),
            "output": output,
        }
        if turn is not None:
            envelope["turn"] = turn
        self._write(self.checkpoint_path(key), envelope)

    def milestone_complete(self, name: str) -> bool:
        milestones = self.metadata.get("milestones", {})
        return isinstance(milestones, dict) and milestones.get(name) == "complete"

    def mark_milestone(self, name: str, status: str) -> None:
        if status not in {"pending", "complete"}:
            raise WorkflowError(f"invalid milestone status {status!r}")
        metadata = self.metadata
        milestones = metadata.get("milestones")
        if not isinstance(milestones, dict):
            raise WorkflowError("V2 workflow state has invalid milestones")
        milestones[name] = status
        self._write(self.metadata_path, metadata)

    def checkpoint_path(self, key: str) -> Path:
        if not key or any(
            character not in "-_.abcdefghijklmnopqrstuvwxyz0123456789"
            for character in key
        ):
            raise WorkflowError(f"invalid checkpoint key {key!r}")
        return self.checkpoints / f"{key}.json"

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise WorkflowError(f"cannot read V2 workflow state {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise WorkflowError(f"V2 workflow state {path} is not an object")
        return value

    @staticmethod
    def _write(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(canonical_json(payload) + "\n", encoding="utf-8")
        temporary.replace(path)
