"""Abstract base interface for coding-agent CLI backends."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

# One turn's wall-clock ceiling, shared by every backend and by the
# orchestrator that budgets a retry loop against it. A per-backend literal
# would let one CLI drift away from the number the orchestrator plans with.
#
# TODO(pipeline): one number for every stage is the wrong shape. A grill round
# answers in a minute; an `implement` turn against a real spec can run for
# most of an hour, and when it overruns the reply is lost even though the
# session survives. 3600 is the interim ceiling picked for the longest stage,
# which means every short stage now waits an hour before it gives up. Replace
# it with a per-turn budget the caller sets -- a `--timeout` flag on the CLI,
# and a per-stage value in the workflow driver -- rather than raising this
# again.
DEFAULT_TURN_TIMEOUT = 3600


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


@dataclass(frozen=True)
class OutputSchema:
    """A JSON Schema for a turn's reply, in both forms the CLIs want.

    claude and grok take the document inline as a `--json-schema` argument;
    codex takes a `--output-schema` path and reads the file itself. Carrying
    both spellings of one schema keeps the choice inside each adapter, where
    the rest of that CLI's dialect already lives.
    """

    text: str
    path: Path


@dataclass
class TurnResult:
    """Outcome of one non-interactive turn against a CLI session."""

    session_id: str | None
    reply: str
    raw: str
    # None whenever no schema was asked for, and also when a schema was asked
    # for and the reply was not a JSON object: that is a reply that failed the
    # contract, which the validator retries rather than a parse the adapter
    # should raise on.
    structured: dict | None = None


def structured_reply(
    schema: OutputSchema | None, reply: str, pre_parsed: object = None
) -> dict | None:
    """The reply as the JSON object a schema asked for, or None.

    `pre_parsed` is the CLI's own parse of that object where it publishes one
    (claude's `structured_output`, grok's `structuredOutput`); codex has no
    such field and passes nothing, so its reply text is parsed here instead.

    Everything that is not an object — a reply the model wrapped in prose, a
    bare string, a turn that asked for no schema at all — comes back as None
    rather than raising. Whether that breaks the contract is the validator's
    call, and a raise here would deny it the retry it is entitled to.
    """
    if schema is None:
        return None
    if isinstance(pre_parsed, dict):
        return pre_parsed
    try:
        value = json.loads(reply)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


class AgentBackend(ABC):
    """Abstract interface defining interaction with coding-agent CLIs.

    Different CLIs (Claude, Codex, Grok, etc.) have different flag conventions,
    session lifecycle mechanisms, and execution requirements. Concrete
    subclasses encapsulate these differences.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier name for this backend (e.g., 'claude', 'codex')."""
        ...

    @abstractmethod
    def run_turn(
        self,
        prompt: str,
        session_id: str | None,
        cwd: Path,
        timeout: int = DEFAULT_TURN_TIMEOUT,
        schema: OutputSchema | None = None,
    ) -> TurnResult:
        """Start (session_id=None) or resume (session_id set) a CLI session with
        `prompt` in directory `cwd` and return the model's text reply along with
        the session id to use for the next turn.

        `timeout` and `schema` are declared here rather than left to each
        subclass: a parameter that exists only as a subclass default is one a
        fourth backend can silently drop, and a type checker cannot see the
        omission because Liskov is only checked against a declared base
        signature. When `schema` is set the reply must be a JSON object
        satisfying it, and `TurnResult.structured` carries that object.
        """
        ...
