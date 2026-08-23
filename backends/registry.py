"""Registry to manage and resolve agent CLI backends."""

from __future__ import annotations

from backends.base import AgentBackend
from backends.claude import ClaudeBackend
from backends.codex import CodexBackend
from backends.grok import GrokBackend
from backends.opencode import OpenCodeBackend


class UnknownBackendError(ValueError):
    """Asked for a backend name that is not registered."""


_BACKENDS: dict[str, type[AgentBackend]] = {
    "claude": ClaudeBackend,
    "codex": CodexBackend,
    "grok": GrokBackend,
    "opencode": OpenCodeBackend,
}


def register_backend(name: str, backend_cls: type[AgentBackend]) -> None:
    """Register a new backend class by name."""
    _BACKENDS[name.lower()] = backend_cls


def list_backends() -> list[str]:
    """List all registered backend names."""
    return sorted(_BACKENDS.keys())


def get_backend(
    name: str,
    *,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> AgentBackend:
    """Instantiate and return the backend registered under `name`."""
    normalized = name.lower().strip()
    backend_cls = _BACKENDS.get(normalized)
    if backend_cls is None:
        valid = ", ".join(list_backends())
        raise UnknownBackendError(
            f"Unknown backend '{name}'. Available backends: {valid}"
        )
    return backend_cls(model=model, reasoning_effort=reasoning_effort)
