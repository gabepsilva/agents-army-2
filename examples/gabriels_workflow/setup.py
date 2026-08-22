"""Bootstrap services for the readable development-workflow entrypoint."""

from __future__ import annotations

import shutil
from pathlib import Path

from examples.gabriels_workflow.config import WorkflowConfig
from examples.gabriels_workflow.development_workflow import (
    LOGGER,
    AgentGateway,
    ArtifactStore,
    DevelopmentWorkflow,
    GitRepository,
    WorkflowError,
    WorkflowOptions,
    WorkflowServices,
)
from examples.gabriels_workflow.github_app_client import GitHubAppClient


def prepare_workflow(issue_number: int, config: WorkflowConfig) -> DevelopmentWorkflow:
    repository_root = Path.cwd().resolve()

    for command in ("git", "make", "uv", "orchestrator"):
        located = shutil.which(command)
        if located is None:
            raise WorkflowError(f"{command} is not installed or is not on PATH")
        LOGGER.debug("setup: %s found at %s", command, located)

    for backend in sorted({role.backend for role in config.roles.values()}):
        located = shutil.which(backend)
        if located is None:
            raise WorkflowError(
                f"configured agent backend '{backend}' is not installed or is not on PATH"
            )
        LOGGER.debug("setup: backend %s found at %s", backend, located)

    LOGGER.info(
        "setup: repository %s, roles %s",
        config.repository,
        ", ".join(
            f"{role}={options.backend}/{options.model}/{options.reasoning_effort}"
            for role, options in sorted(config.roles.items())
        ),
    )
    role_github = {
        role: GitHubAppClient.connect(
            role_config.github_app.app_id,
            role_config.github_app.private_key.get_secret_value(),
            config.repository,
        )
        for role, role_config in config.roles.items()
    }
    github = role_github["implementer"]

    issue_root = repository_root / ".git" / "gdw" / f"issue-{issue_number}"
    worktree_path = issue_root / "worktree"
    GitRepository(repository_root).ensure_issue_worktree(
        f"gdw/issue-{issue_number}", github.default_branch, worktree_path
    )

    store = ArtifactStore(issue_root)
    repository = GitRepository(worktree_path)
    LOGGER.info("setup: connected %s GitHub App installation(s)", len(role_github))
    branch, base_sha = repository.prepare(github.default_branch, store.initialized)
    store.initialize(issue_number, branch, base_sha)
    agents = AgentGateway(
        roles=config.roles,
        issue=issue_number,
        state_file=store.root / "agents.json",
        example_root=Path(__file__).resolve().parent,
    )
    return DevelopmentWorkflow(
        WorkflowOptions(
            issue_number,
            github.default_branch,
            branch,
            config.draft,
        ),
        WorkflowServices(store, github, repository, agents, role_github),
    )
