"""Backend abstractions and implementations for coding-agent CLIs."""

from backends.base import AgentBackend, TurnResult
from backends.claude import ClaudeBackend
from backends.codex import CodexBackend
from backends.registry import get_backend, list_backends, register_backend

__all__ = [
    "AgentBackend",
    "ClaudeBackend",
    "CodexBackend",
    "TurnResult",
    "get_backend",
    "list_backends",
    "register_backend",
]
