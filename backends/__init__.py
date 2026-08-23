"""Backend abstractions and implementations for coding-agent CLIs."""

from backends.base import AgentBackend, TurnError, TurnResult
from backends.claude import ClaudeBackend
from backends.codex import CodexBackend
from backends.grok import GrokBackend
from backends.opencode import OpenCodeBackend
from backends.registry import get_backend, list_backends, register_backend

__all__ = [
    "AgentBackend",
    "ClaudeBackend",
    "CodexBackend",
    "GrokBackend",
    "OpenCodeBackend",
    "TurnError",
    "TurnResult",
    "get_backend",
    "list_backends",
    "register_backend",
]
