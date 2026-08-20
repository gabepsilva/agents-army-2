"""Abstract base interface for coding-agent CLI backends."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


class TurnError(RuntimeError):
    """A CLI turn returned something the orchestrator cannot use.

    Backend-specific subclasses keep tests and logs precise. The CLI
    catches this base type so a new backend is not a new except-clause.
    """


def describe_command(args: list[str], prompt: str) -> str:
    """Render a CLI invocation for logs with the prompt replaced by its size.

    The prompt is the one unbounded argument, and a verbose run that echoes it
    buries the flags — the part worth reading — under the whole request.

    A backend that has to glue the prompt onto its flag (`--single=<prompt>`,
    so an argument parser cannot read a leading `-` as a flag) is handled too:
    the flag stays readable and only the value is replaced.
    """
    rendered = list(args)
    placeholder = f"<prompt:{len(prompt)}chars>"
    attached = f"={prompt}"
    for i in range(len(rendered) - 1, -1, -1):
        arg = rendered[i]
        if arg == prompt:
            rendered[i] = placeholder
            break
        # `prompt and` guards the empty prompt: every argument ends with
        # "=" + "", and slicing one off by length would eat the flag too.
        if prompt and arg.endswith(attached):
            rendered[i] = f"{arg.removesuffix(prompt)}{placeholder}"
            break
    return " ".join(rendered)


def stdout_for_error(stdout: str) -> str:
    """Keep both ends of a long dump: the parse error is at char 0, the
    envelope is usually at the tail."""
    if len(stdout) <= 2000:
        return stdout
    return f"{stdout[:400]}\n…\n{stdout[-1600:]}"


def json_objects(text: str) -> list[dict]:
    """Scan `text` for every top-level JSON object, in order of appearance.

    CLIs that promise a single envelope still sometimes write a text prefix
    first. json.loads of the whole buffer then fails even though a valid
    object is sitting in the stream.
    """
    decoder = json.JSONDecoder()
    found: list[dict] = []
    idx = 0
    while idx < len(text):
        start = text.find("{", idx)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            idx = start + 1
            continue
        found.append(obj)
        idx = end
    return found


def reply_text(payload: dict, key: str) -> str:
    """The turn's reply as text, whatever the CLI actually put under `key`.

    A missing key and an explicit null both mean the same thing — the turn
    produced no assistant text — and neither is worth failing the turn over:
    the session id it did report is what the next turn needs, and raising
    here would throw that away before the orchestrator could persist it.
    """
    value = payload.get(key)
    return value if isinstance(value, str) else ""


@dataclass
class TurnResult:
    """Outcome of one non-interactive turn against a CLI session."""

    session_id: str | None
    reply: str
    raw: str


class AgentBackend(ABC):
    """Abstract interface defining interaction with coding-agent CLIs.

    Different CLIs (Claude, Codex, Grok, etc.) have different flag conventions,
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
