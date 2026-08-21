"""Bootstrap services for the readable development-workflow entrypoint."""

from __future__ import annotations

import shutil
from pathlib import Path

from examples.gabriels_workflow.config import WorkflowConfig
from examples.gabriels_workflow.development_workflow import (
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
        if shutil.which(command) is None:
            raise WorkflowError(f"{command} is not installed or is not on PATH")

    for backend in sorted({role.backend for role in config.roles.values()}):
        if shutil.which(backend) is None:
            raise WorkflowError(
                f"configured agent backend '{backend}' is not installed or is not on PATH"
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

    store = ArtifactStore(repository_root / ".git" / "gdw" / f"issue-{issue_number}")
    repository = GitRepository(repository_root)
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
