"""Abstract base interface for coding-agent CLI backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


def describe_command(args: list[str], prompt: str) -> str:
    """Render a CLI invocation for logs with the prompt replaced by its size.

    The prompt is the one unbounded argument, and a verbose run that echoes it
    buries the flags — the part worth reading — under the whole request.
    """
    rendered = list(args)
    for i in range(len(rendered) - 1, -1, -1):
        if rendered[i] == prompt:
            rendered[i] = f"<prompt:{len(prompt)}chars>"
            break
    return " ".join(rendered)


@dataclass
class TurnResult:
    """Outcome of one non-interactive turn against a CLI session."""

    session_id: str | None
    reply: str
    raw: str


class AgentBackend(ABC):
    """Abstract interface defining interaction with coding-agent CLIs.

    Different CLIs (Claude, Codex, etc.) have different flag conventions,
    session lifecycle mechanisms, and execution requirements. Concrete
    subclasses encapsulate these differences.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier name for this backend (e.g., 'claude', 'codex')."""
        ...

    @abstractmethod
    def run_turn(self, prompt: str, session_id: str | None, cwd: Path) -> TurnResult:
        """Start (session_id=None) or resume (session_id set) a CLI session with
        `prompt` in directory `cwd` and return the model's text reply along with
        the session id to use for the next turn."""
        ...
