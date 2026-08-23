"""Validated YAML configuration for Gabriel's development workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)

from examples.gabriels_workflow.development_workflow import AGENT_ROLES, WorkflowError

REQUIRED_ROLES = AGENT_ROLES
DEFAULT_CONFIG_PATH = Path(__file__).with_name("workflow.local")


class GitHubAppConfig(BaseModel):
    """GitHub App identity used for one role's comments and operations."""

    model_config = ConfigDict(extra="forbid")

    app_id: int = Field(gt=0)
    private_key: SecretStr

    @field_validator("private_key", mode="before")
    @classmethod
    def nonempty_private_key(cls, value: object) -> object:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("must not be empty")
        return value


class RoleConfig(BaseModel):
    """CLI backend settings for one persistent workflow role."""

    model_config = ConfigDict(extra="forbid")

    backend: str
    model: str | None = None
    reasoning_effort: str | None = None
    github_app: GitHubAppConfig

    @field_validator("backend")
    @classmethod
    def supported_backend(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"claude", "codex", "grok", "opencode"}:
            raise ValueError("must be one of: claude, codex, grok, opencode")
        return normalized

    @field_validator("model", "reasoning_effort")
    @classmethod
    def nonempty_optional_value(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be empty")
        return normalized


class WorkflowConfig(BaseModel):
    """All non-secret settings needed to assemble the workflow."""

    model_config = ConfigDict(extra="forbid")

    repository: str
    draft: bool = True
    roles: dict[str, RoleConfig]

    @field_validator("repository")
    @classmethod
    def owner_and_repository(cls, value: str) -> str:
        normalized = value.strip()
        parts = normalized.split("/")
        if len(parts) != 2 or not all(part.strip() for part in parts):
            raise ValueError("must use OWNER/REPO format")
        return normalized

    @model_validator(mode="after")
    def every_role_is_configured(self) -> Self:
        configured = set(self.roles)
        missing = sorted(REQUIRED_ROLES - configured)
        unknown = sorted(configured - REQUIRED_ROLES)
        if missing or unknown:
            details = []
            if missing:
                details.append(f"missing roles: {', '.join(missing)}")
            if unknown:
                details.append(f"unknown roles: {', '.join(unknown)}")
            raise ValueError("; ".join(details))
        return self


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> WorkflowConfig:
    """Read and validate one workflow YAML document."""

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise WorkflowError(f"cannot read workflow config {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise WorkflowError(f"invalid YAML in workflow config {path}: {exc}") from exc
    try:
        _load_private_key_files(payload, path)
    except OSError as exc:
        raise WorkflowError(
            f"cannot read private key referenced by {path}: {exc}"
        ) from exc
    try:
        return WorkflowConfig.model_validate(payload)
    except ValidationError as exc:
        raise WorkflowError(f"invalid workflow config {path}: {exc}") from exc


def _load_private_key_files(payload: object, config_path: Path) -> None:
    """Replace YAML private-key paths with their PEM contents in place."""

    if not isinstance(payload, dict):
        return
    roles = payload.get("roles")
    if not isinstance(roles, dict):
        return
    for role in roles.values():
        if not isinstance(role, dict):
            continue
        github_app = role.get("github_app")
        if not isinstance(github_app, dict):
            continue
        private_key = github_app.get("private_key")
        if not isinstance(private_key, str):
            continue
        candidate = Path(private_key).expanduser()
        if not candidate.is_absolute():
            candidate = config_path.parent / candidate
        if candidate.suffix != ".pem":
            continue
        github_app["private_key"] = candidate.read_text(encoding="utf-8")
