"""Public package surface for the agent orchestrator."""

# These imports are deliberate reexports consumed by the derived ``__all__``;
# the module has no other code that could reference them directly.
# ruff: noqa: F401

from backends import AgentBackend, TurnError, TurnResult, get_backend, list_backends
from backends.base import DEFAULT_TURN_TIMEOUT, OutputSchema
from backends.registry import UnknownBackendError

from . import cli, core, doctor, paths, schema, skills, teams
from .cli import cmd_chat, cmd_create, cmd_delete, cmd_fork, cmd_list, cmd_talk, main
from .core import (
    DEFAULT_BACKEND,
    DEFAULT_VALIDATION_RETRIES,
    TRACE,
    Agent,
    AgentBusyError,
    AgentExistsError,
    AgentNotFoundError,
    Orchestrator,
    OrchestratorError,
    StateError,
    TeamBusyError,
    log,
)
from .doctor import (
    DEPENDENCY_TOOLS,
    FOUND,
    FOUND_OPTIONAL,
    MIN_PYTHON,
    NAME_SEPARATORS,
    NOT_FOUND,
    VERSION_PROBE_TIMEOUT,
)
from .schema import SchemaError, SchemaLoadError, load_schema
from .skills import (
    SkillError,
    compose_skill_prompt,
    format_skill_listing,
    index_skills,
    parse_skill_names,
    resolve_catalog_dir,
    resolve_skills,
)

__all__ = sorted(name for name in globals() if not name.startswith("_"))
