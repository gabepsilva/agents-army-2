"""Validated configuration for Gabriel's development workflow V2."""

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

from examples.gabriels_workflow.development_workflow import WorkflowError

AGENT_ROLES = frozenset(
    {
        "expander",
        "griller",
        "specifier",
        "implementer",
        "documenter",
        "reviewer-specification",
        "reviewer-quality",
        "finalizer",
    }
)
BACKENDS = frozenset({"claude", "codex", "grok", "opencode"})
DEFAULT_CONFIG_PATH = Path(__file__).with_name("workflow.local")


class GitHubAppConfig(BaseModel):
    """The single App identity V2 publishes its milestones as.

    V1 gave every role its own App so each stage comment carried an author.
    V2 posts milestones, not stage comments, so one identity is enough and
    seven fewer Apps have to be installed before a run.
    """

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
    """Which agent CLI, model, and effort one workflow role runs with."""

    model_config = ConfigDict(extra="forbid")

    backend: str
    model: str | None = None
    reasoning_effort: str | None = None

    @field_validator("backend")
    @classmethod
    def supported_backend(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in BACKENDS:
            raise ValueError(f"must be one of: {', '.join(sorted(BACKENDS))}")
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


class BudgetConfig(BaseModel):
    """Hard limits; cached turns do not consume the model-call budget again."""

    model_config = ConfigDict(extra="forbid")

    max_agent_turns: int = Field(default=24, ge=8, le=100)
    max_clarification_rounds: int = Field(default=3, ge=1, le=10)
    max_ci_attempts: int = Field(default=3, ge=1, le=10)
    max_review_rounds: int = Field(default=3, ge=1, le=10)
    max_prompt_chars: int = Field(default=60_000, ge=1_000, le=120_000)
    max_output_chars: int = Field(default=30_000, ge=1_000, le=100_000)
    agent_timeout: int = Field(default=3_600, ge=60, le=7_200)
    ci_timeout: int = Field(default=7_200, ge=60, le=14_400)


class WorkflowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repository: str
    github_app: GitHubAppConfig
    draft: bool = True
    budgets: BudgetConfig = Field(default_factory=BudgetConfig)
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
    def exactly_the_known_roles(self) -> Self:
        configured = set(self.roles)
        missing = sorted(AGENT_ROLES - configured)
        unknown = sorted(configured - AGENT_ROLES)
        details = []
        if missing:
            details.append(f"missing roles: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown roles: {', '.join(unknown)}")
        if details:
            raise ValueError("; ".join(details))
        return self


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> WorkflowConfig:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise WorkflowError(f"cannot read V2 workflow config {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise WorkflowError(
            f"invalid YAML in V2 workflow config {path}: {exc}"
        ) from exc
    try:
        _resolve_private_key_file(payload, path)
        return WorkflowConfig.model_validate(payload)
    except OSError as exc:
        raise WorkflowError(
            f"cannot read private key referenced by {path}: {exc}"
        ) from exc
    except ValidationError as exc:
        raise WorkflowError(f"invalid V2 workflow config {path}: {exc}") from exc


def _resolve_private_key_file(payload: object, config_path: Path) -> None:
    """Read `private_key` in place when it names a `.pem` beside the config.

    Keeps the key itself out of the config file, which is the only reason a
    path is accepted where a PEM body is otherwise expected.
    """

    if not isinstance(payload, dict):
        return
    app = payload.get("github_app")
    if not isinstance(app, dict):
        return
    private_key = app.get("private_key")
    if not isinstance(private_key, str):
        return
    candidate = Path(private_key).expanduser()
    if not candidate.is_absolute():
        candidate = config_path.parent / candidate
    if candidate.suffix == ".pem":
        app["private_key"] = candidate.read_text(encoding="utf-8")
