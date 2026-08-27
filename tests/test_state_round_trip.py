"""The registry file's round trip: what `Orchestrator` reads it must write
back byte for byte. See PR #118 (issue #110).

Nothing else in `tests/` asserts the state file's raw bytes, so this is the
only guard against a load/store rewrite silently changing the on-disk format
— an omitted optional key, a dropped `"session_id": null`, different
indentation, a lost trailing newline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backends.base import (
    DEFAULT_TURN_TIMEOUT,
    AgentBackend,
    OutputSchema,
    TurnResult,
)
from backends.registry import UnknownBackendError, register_backend
from orchestrator import Orchestrator, StateError
from tests.path_helpers import runtime_paths


class RecordingBackend(AgentBackend):
    """A backend that never runs a CLI: these tests only persist it."""

    @property
    def name(self) -> str:
        return "recording"

    def run_turn(
        self,
        prompt: str,
        session_id: str | None,
        cwd: Path,
        timeout: int = DEFAULT_TURN_TIMEOUT,
        schema: OutputSchema | None = None,
        *,
        resume_as_fork: bool = False,
    ) -> TurnResult:
        return TurnResult(session_id="recording-sid", reply=prompt, raw="")


@pytest.fixture(autouse=True)
def register_recording_backend() -> None:
    register_backend("recording", RecordingBackend)


# Both registries are written the way the orchestrator writes them, so a
# faithful round trip reproduces them exactly. The two differ only in the
# optional keys: `to_entry` gives each one its own omit-when-None branch, and
# a fixture that exercised only one side would leave the other unasserted.
_EVERY_OPTIONAL_KEY_PRESENT = """\
{
  "full": {
    "backend": "recording",
    "created_at": "2026-01-02T03:04:05Z",
    "last_turn_at": "2026-01-02T03:09:05Z",
    "model": "some-model",
    "reasoning_effort": "high",
    "session_id": "abc-123",
    "turns": 7
  }
}
"""

# A fork that has been recorded but not yet taken: `pending_fork_from` is the
# one key that appears between `create` and the agent's first turn, and it
# must be omitted again once that turn has stored a session id of its own.
_A_RECORDED_FORK = """\
{
  "copy": {
    "backend": "recording",
    "created_at": "2026-01-02T03:04:05Z",
    "pending_fork_from": "source-sid",
    "session_id": null,
    "turns": 0
  }
}
"""

_EVERY_OPTIONAL_KEY_ABSENT = """\
{
  "bare": {
    "backend": "recording",
    "session_id": null
  }
}
"""


@pytest.mark.parametrize(
    "registry",
    [_EVERY_OPTIONAL_KEY_PRESENT, _EVERY_OPTIONAL_KEY_ABSENT, _A_RECORDED_FORK],
    ids=[
        "every-optional-key-present",
        "every-optional-key-absent",
        "a-recorded-fork",
    ],
)
def test_a_registry_is_rewritten_byte_for_byte(tmp_path: Path, registry: str) -> None:
    state_file = tmp_path / "state.json"
    state_file.write_text(registry, encoding="utf-8")

    orch = Orchestrator(runtime_paths(tmp_path, state_file=state_file))
    orch._persist()

    assert state_file.read_text(encoding="utf-8") == registry


def test_a_freshly_spawned_agent_is_written_with_every_field_it_has(
    tmp_path: Path,
) -> None:
    """`spawn` persists on its own; the file it leaves must name the backend,
    a `created_at` stamp and `turns: 0`, and carry the null session id the
    agent has before its first turn."""
    state_file = tmp_path / "state.json"
    orch = Orchestrator(runtime_paths(tmp_path, state_file=state_file))
    orch.spawn("a", "recording", model="m", reasoning_effort="low")

    entry = json.loads(state_file.read_text(encoding="utf-8"))["a"]
    assert entry["backend"] == "recording"
    assert entry["model"] == "m"
    assert entry["reasoning_effort"] == "low"
    assert entry["turns"] == 0
    assert entry["created_at"] == orch.agents["a"].created_at
    assert entry["session_id"] is None
    assert "last_turn_at" not in entry
    # An agent that was created rather than forked carries no marker, so a
    # registry written before `fork` existed round-trips unchanged.
    assert "pending_fork_from" not in entry


def test_an_entry_without_a_backend_is_rejected(tmp_path: Path) -> None:
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({"a": {"session_id": None}}), encoding="utf-8")

    with pytest.raises(StateError, match="agent 'a' has no backend"):
        Orchestrator(runtime_paths(tmp_path, state_file=state_file))


def test_an_unknown_backend_name_surfaces_from_the_registry(tmp_path: Path) -> None:
    """A removed backend plugin is `get_backend`'s error, not a `StateError`:
    the entry is well-formed, the plugin is the thing that is gone."""
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({"a": {"backend": "gone"}}), encoding="utf-8")

    with pytest.raises(UnknownBackendError):
        Orchestrator(runtime_paths(tmp_path, state_file=state_file))
