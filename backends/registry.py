"""Registry to manage and resolve agent CLI backends."""

from __future__ import annotations

from typing import Type

from backends.base import AgentBackend
from backends.claude import ClaudeBackend
from backends.codex import CodexBackend

_BACKENDS: dict[str, Type[AgentBackend]] = {
    "claude": ClaudeBackend,
    "codex": CodexBackend,
}


def register_backend(name: str, backend_cls: Type[AgentBackend]) -> None:
    """Register a new backend class by name."""
    _BACKENDS[name.lower()] = backend_cls


def list_backends() -> list[str]:
    """List all registered backend names."""
    return sorted(_BACKENDS.keys())


def get_backend(name: str) -> AgentBackend:
    """Instantiate and return the backend registered under `name`."""
    normalized = name.lower().strip()
    backend_cls = _BACKENDS.get(normalized)
    if backend_cls is None:
        valid = ", ".join(list_backends())
        raise ValueError(f"Unknown backend '{name}'. Available backends: {valid}")
    return backend_cls()
