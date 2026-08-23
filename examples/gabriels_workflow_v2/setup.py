"""Assemble the local-first V2 workflow from validated configuration."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import cast

from examples.gabriels_workflow.development_workflow import LOGGER, WorkflowError
from examples.gabriels_workflow_v2.config import WorkflowConfig
from examples.gabriels_workflow_v2.contracts import CheckpointStore
from examples.gabriels_workflow_v2.publisher import GitHubPublisher
from examples.gabriels_workflow_v2.workflow import (
    DevelopmentWorkflowV2,
    RelayAgentGateway,
    RelayRepository,
    WorkflowOptions,
    WorkflowServices,
)

REQUIRED_COMMANDS = ("git", "make", "uv", "orchestrator", "bwrap")


def prepare_workflow(
    issue_number: int, config: WorkflowConfig
) -> DevelopmentWorkflowV2:
    """Build the workflow for one issue, failing before any model is paid for."""

    root = Path.cwd().resolve()
    _require_commands(config)

    publisher = cast(
        GitHubPublisher,
        GitHubPublisher.connect(
            config.github_app.app_id,
            config.github_app.private_key.get_secret_value(),
            config.repository,
        ),
    )
    base_branch = publisher.default_branch

    # Under the git common dir so a run started from a linked worktree still
    # finds the state of a previous run on the same issue.
    checkout = RelayRepository(root)
    issue_root = checkout.common_git_dir() / "gdw-v2" / f"issue-{issue_number}"
    worktree = issue_root / "worktree"
    branch = f"gdwv2/issue-{issue_number}"
    checkout.ensure_issue_worktree(branch, base_branch, worktree)

    store = CheckpointStore(issue_root)
    repository = RelayRepository(worktree)
    branch, base_sha = repository.prepare(base_branch, store.initialized)
    store.initialize(issue_number, branch, base_sha)
    initial_snapshot = _recorded_initial_snapshot(store, repository)
    LOGGER.info(
        "setup: issue #%s on %s (base %s), state in %s",
        issue_number,
        branch,
        base_branch,
        issue_root,
    )
    agents = RelayAgentGateway(
        roles=config.roles,
        issue=issue_number,
        state_file=store.root / "agents.json",
        example_root=Path(__file__).resolve().parent,
        workdir=worktree,
        max_prompt_chars=config.budgets.max_prompt_chars,
    )
    return DevelopmentWorkflowV2(
        WorkflowOptions(
            issue_number,
            base_branch,
            branch,
            config.draft,
            initial_snapshot,
            config.budgets,
        ),
        WorkflowServices(store, publisher, repository, agents),
    )


def _require_commands(config: WorkflowConfig) -> None:
    backends = sorted({role.backend for role in config.roles.values()})
    for command in (*REQUIRED_COMMANDS, *backends):
        if shutil.which(command) is None:
            raise WorkflowError(f"{command} is not installed or is not on PATH")
    LOGGER.debug("setup: located %s", ", ".join((*REQUIRED_COMMANDS, *backends)))


def _recorded_initial_snapshot(
    store: CheckpointStore, repository: RelayRepository
) -> str:
    """The tree fingerprint from before the first agent turn, taken once.

    Re-measuring it on resume would fingerprint a tree the implementer has
    already edited, and `require_changed` would then accept a run that
    changed nothing.
    """

    recorded = store.metadata.get("initial_snapshot")
    if isinstance(recorded, str) and recorded:
        return recorded
    snapshot = repository.snapshot()
    store.update_metadata(initial_snapshot=snapshot)
    return snapshot
