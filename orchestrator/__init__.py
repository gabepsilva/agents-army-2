#!/usr/bin/env python3
"""Long-lived orchestrator holding an array of agents.

Each agent owns a persistent Claude Code, Codex, Grok, or OpenCode CLI session. Every
time you talk to an agent it resumes that session with your prompt and returns
the reply, so each agent keeps its own conversation history across messages.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
import tomllib
from collections.abc import Callable, Iterable, Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Any, NoReturn, cast

from backends import AgentBackend, TurnError, TurnResult, get_backend, list_backends
from backends.base import DEFAULT_TURN_TIMEOUT, OutputSchema
from backends.registry import UnknownBackendError
from orchestrator.schema import (
    ReplyValidationError,
    SchemaLoadError,
    compose_schema_prompt,
    load_schema,
    repair_prompt,
    validate_reply,
)
from orchestrator.skills import (
    SkillError,
    compose_skill_prompt,
    format_skill_listing,
    index_skills,
    parse_skill_names,
    resolve_skills,
)

# State lives in the caller's working directory (override with AGENTS_ARMY_HOME)
# rather than next to the installed package, so it doesn't leak into the venv.
HOME = Path(os.environ.get("AGENTS_ARMY_HOME", Path.cwd()))
STATE_FILE = Path(
    os.environ.get("AGENTS_ARMY_STATE_FILE", HOME / "orchestrator_state.json")
)
# Agents run their CLI sessions from a single shared working directory.
WORKDIR = HOME
# Skill markdown catalog. Override with AGENTS_ARMY_SKILLS; default is $HOME/SKILLS.
SKILLS_DIR = Path(os.environ.get("AGENTS_ARMY_SKILLS", HOME / "SKILLS"))
# The backend an agent gets when none is named: by `create`, and by the agent a
# talk creates for a name that does not exist yet.
DEFAULT_BACKEND = "claude"

# How many extra turns a reply that misses the schema is worth. Two, because
# the measured conformance rate makes even the first retry nearly dead code:
# this is the fallback, not the mechanism that gets a conforming reply.
DEFAULT_VALIDATION_RETRIES = 2

# Named explicitly rather than via __name__: this module runs both as a script
# (__main__) and as the `orchestrator` console script, and _configure_logging
# raises the level by logger name.
log = logging.getLogger("orchestrator")

# Full prompts and replies are unbounded and are the only logs that can carry
# the content of a conversation, so they sit below DEBUG: -v stays readable and
# safe to paste, and -vv is the deliberate opt-in to the whole transcript.
TRACE = logging.DEBUG - 5
logging.addLevelName(TRACE, "TRACE")


class OrchestratorError(Exception):
    """A failure the user can act on: one line on stderr, exit 1, no traceback.

    Named types rather than bare KeyError/ValueError so the CLI can catch
    exactly these. Catching the builtins around the whole dispatch swallowed
    any incidental one raised inside a backend too, and printed it as a bare
    one-word line with no traceback — turning a real bug into a mystery.
    """


class AgentNotFoundError(OrchestratorError, KeyError):
    """No agent by that name. Still a KeyError: that is what callers catch."""


class AgentExistsError(OrchestratorError, ValueError):
    """Spawn was asked for a name that is already taken."""


class StateError(OrchestratorError):
    """The state file exists but does not hold the structure this code needs."""


# User-facing failures that must print one line and exit 1, never a traceback.
# UnknownBackendError comes from the registry, which cannot import this module.
_CLI_ERRORS = (OrchestratorError, UnknownBackendError)


class Agent:
    """A single named agent backed by one persistent CLI session."""

    def __init__(self, name: str, backend: AgentBackend) -> None:
        self.name = name
        self.backend = backend
        self.session_id: str | None = None

    def talk(
        self,
        prompt: str,
        schema: OutputSchema | None = None,
        timeout: int = DEFAULT_TURN_TIMEOUT,
    ) -> TurnResult:
        log.info(
            "agent '%s' (%s): starting turn, resume=%s",
            self.name,
            self.backend.name,
            bool(self.session_id),
        )
        # Logged here rather than per backend: the turn is the same exchange
        # whichever CLI runs it, so every backend gets this for free.
        log.log(TRACE, "agent '%s' prompt in:\n%s", self.name, prompt)
        started = time.monotonic()
        result = self.backend.run_turn(
            prompt, self.session_id, WORKDIR, timeout, schema
        )
        elapsed = time.monotonic() - started
        log.info("agent '%s': turn finished in %.1fs", self.name, elapsed)
        log.log(TRACE, "agent '%s' reply out:\n%s", self.name, result.reply)
        if result.structured is not None:
            log.log(
                TRACE,
                "agent '%s' structured out:\n%s",
                self.name,
                json.dumps(result.structured, indent=2, sort_keys=True),
            )
        # A backend that reports no session id has not ended the conversation,
        # it has failed to name it. Keeping the previous id lets the next turn
        # resume the session instead of silently starting a fresh one.
        if result.session_id is not None:
            self.session_id = result.session_id
        return result


class Orchestrator:
    """Registry of named agents, each with an independent CLI session.

    Agents persist in `orchestrator_state.json` so any process can spawn an
    agent once and talk to it later, resuming the same CLI session.
    """

    def __init__(self, state_file: Path | None = None) -> None:
        self.state_file = STATE_FILE if state_file is None else state_file
        self.agents: dict[str, Agent] = {}
        self._reload()
        log.debug(
            "state: loaded %d agent(s) from %s", len(self.agents), self.state_file
        )

    def spawn(
        self,
        name: str,
        backend: str | None = None,
        *,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> Agent:
        with self._exclusive():
            self._reload()
            if name in self.agents:
                raise AgentExistsError(f"agent '{name}' already exists")
            return self._create(name, backend, model, reasoning_effort)

    def ensure(
        self,
        name: str,
        backend: str | None = None,
        *,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> tuple[Agent, bool]:
        """Return the named agent, creating it first if it does not exist.

        Reports whether it had to create one, so a caller can say so. The
        lookup and the create share one lock rather than being a `spawn` after
        a failed `talk`: two processes naming the same new agent at once then
        get one agent between them, not a spawn that loses to a duplicate.
        """
        with self._exclusive():
            self._reload()
            existing = self.agents.get(name)
            if existing is not None:
                return existing, False
            return self._create(name, backend, model, reasoning_effort), True

    def _create(
        self,
        name: str,
        backend: str | None,
        model: str | None,
        reasoning_effort: str | None,
    ) -> Agent:
        """Register and persist a new agent. The caller holds `_exclusive()`.

        `None` means "whatever the default backend is now": resolving it here
        rather than in a default argument keeps DEFAULT_BACKEND a live lookup.
        """
        agent = Agent(
            name,
            get_backend(
                DEFAULT_BACKEND if backend is None else backend,
                model=model,
                reasoning_effort=reasoning_effort,
            ),
        )
        self.agents[name] = agent
        self._persist()
        return agent

    def talk(
        self,
        name: str,
        prompt: str,
        schema: OutputSchema | None = None,
        retries: int = DEFAULT_VALIDATION_RETRIES,
        timeout: int = DEFAULT_TURN_TIMEOUT,
    ) -> TurnResult:
        """Run one turn against `name`, or, with a schema, as many as it takes.

        The whole thing happens under one agent lock: a retry has to land on
        the same session as the attempt it is correcting, and another process
        talking to this agent in between would fork the conversation.
        """
        with self._agent_lock(name):
            with self._exclusive():
                self._reload()
                agent = self.agents.get(name)
                if agent is None:
                    raise AgentNotFoundError(f"no agent named '{name}'")
            if schema is None:
                return self._turn(agent, prompt, None, timeout)
            return self._validated_turn(agent, prompt, schema, retries, timeout)

    def _turn(
        self,
        agent: Agent,
        prompt: str,
        schema: OutputSchema | None,
        timeout: int,
    ) -> TurnResult:
        """One turn, with its session id persisted before anything else runs.

        Persisting per attempt rather than per call is what lets a run that
        exhausts its retries still leave the session where the agent actually
        is: it moved the conversation forward whether or not the last reply
        was usable, and resuming from a stale id would replay it.
        """
        result = agent.talk(prompt, schema, timeout)
        with self._exclusive():
            self._reload()
            if agent.name not in self.agents:
                raise AgentNotFoundError(f"no agent named '{agent.name}'")
            # agent.session_id, not result.session_id: the agent keeps the
            # id it already had when a backend reports none.
            self.agents[agent.name].session_id = agent.session_id
            self._persist()
        return result

    def _validated_turn(
        self,
        agent: Agent,
        prompt: str,
        schema: OutputSchema,
        retries: int,
        timeout: int,
    ) -> TurnResult:
        """Talk until the reply satisfies `schema`, the retries run out, or the
        clock does.

        `timeout` is the budget for the whole loop, not for each attempt: a
        validated call must not be able to cost three times what a plain turn
        can, holding this agent's lock for an hour and a half to do it. Each
        attempt gets whatever is left.
        """
        # None on a CLI that was handed the schema itself; the document on one
        # that was not, so the prompt can carry what the flag could not.
        unenforced = None if agent.backend.enforces_schema else schema
        if unenforced is not None:
            log.warning(
                "backend %s: schema is enforced via validation/repair, not the CLI",
                agent.backend.name,
            )
        deadline = time.monotonic() + timeout
        attempt_prompt = compose_schema_prompt(prompt, unenforced)
        attempt = 0
        while True:
            attempt += 1
            remaining = deadline - time.monotonic()
            result = self._turn(
                agent, attempt_prompt, schema, max(1, math.ceil(remaining))
            )
            try:
                result.structured = validate_reply(
                    result.reply, result.structured, schema
                )
            except ReplyValidationError as exc:
                log.warning(
                    "agent '%s': attempt %d did not satisfy the schema: %s",
                    agent.name,
                    attempt,
                    exc,
                )
                if attempt > retries:
                    log.warning(
                        "agent '%s': %d validation retries exhausted",
                        agent.name,
                        retries,
                    )
                    raise
                if deadline - time.monotonic() <= 0:
                    log.warning(
                        "agent '%s': the %ds budget is spent; not retrying",
                        agent.name,
                        timeout,
                    )
                    raise
                attempt_prompt = repair_prompt(exc, unenforced)
            else:
                # `else`, not a fall-through after the except: a `return` at
                # this indentation would hand back the attempt that just
                # failed validation.
                return result

    def list_agents(self) -> list[str]:
        return sorted(self.agents)

    def delete(self, name: str) -> Agent:
        with self._exclusive():
            self._reload()
            agent = self.agents.pop(name, None)
            if agent is None:
                raise AgentNotFoundError(f"no agent named '{name}'")
            self._persist()
            return agent

    def _lock_path(self) -> Path:
        return self.state_file.with_name(self.state_file.name + ".lock")

    def _agent_lock_path(self, name: str) -> Path:
        # An agent name is free text and would not survive being used as a
        # filename; the digest only has to be stable, not readable.
        digest = hashlib.sha256(name.encode()).hexdigest()
        return self.state_file.with_name(f"{self.state_file.name}.{digest}.lock")

    @contextmanager
    def _locked(self, path: Path) -> Iterator[None]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _exclusive(self) -> AbstractContextManager[None]:
        """Serialize reads and writes of the state file."""
        return self._locked(self._lock_path())

    def _agent_lock(self, name: str) -> AbstractContextManager[None]:
        """Serialize whole turns for one agent, leaving other agents free.

        The state lock cannot do this: it covers a file write measured in
        milliseconds, while the thing that must not overlap is the turn, which
        runs for minutes. Two processes resuming the same session fork the
        conversation, and whichever persists last drops the other's reply.
        """
        return self._locked(self._agent_lock_path(name))

    def _reload(self) -> None:
        self.agents = {}
        for name, entry in self._load_state().items():
            backend = entry.get("backend")
            if backend is None:
                raise StateError(f"{self.state_file}: agent '{name}' has no backend")
            agent = Agent(
                name,
                get_backend(
                    backend,
                    model=entry.get("model"),
                    reasoning_effort=entry.get("reasoning_effort"),
                ),
            )
            agent.session_id = entry.get("session_id")
            self.agents[name] = agent

    def _load_state(self) -> dict[str, dict]:
        if not self.state_file.exists():
            return {}
        try:
            return json.loads(self.state_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise StateError(f"{self.state_file} is not valid JSON: {exc}") from exc

    def _persist(self) -> None:
        state = {
            name: {
                "backend": a.backend.name,
                "session_id": a.session_id,
                **({"model": a.backend.model} if a.backend.model is not None else {}),
                **(
                    {"reasoning_effort": a.backend.reasoning_effort}
                    if a.backend.reasoning_effort is not None
                    else {}
                ),
            }
            for name, a in self.agents.items()
        }
        payload = json.dumps(state, indent=2, sort_keys=True) + "\n"
        tmp = self.state_file.with_name(self.state_file.name + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(self.state_file)
        log.debug("state: wrote %d agent(s) to %s", len(state), self.state_file)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_agent_config_options(parser: argparse.ArgumentParser) -> None:
    # No argparse default: leaving this None lets create() and ensure() resolve
    # DEFAULT_BACKEND, which is what that constant documents itself as. A
    # literal here would pin them to claude however DEFAULT_BACKEND changed.
    parser.add_argument("--backend", "-b", choices=list_backends())
    parser.add_argument("--model", "-m")
    parser.add_argument("--reasoning-effort", "-e")


def _agent_config(agent: Agent) -> tuple[str, str | None, str | None]:
    return agent.backend.name, agent.backend.model, agent.backend.reasoning_effort


def cmd_create(orchestrator: Orchestrator, opts: argparse.Namespace) -> None:
    agent = orchestrator.spawn(
        opts.name,
        opts.backend,
        model=opts.model,
        reasoning_effort=opts.reasoning_effort,
    )
    print(f"created agent '{agent.name}' backend={agent.backend.name}")


def _ensure_agent(
    orchestrator: Orchestrator,
    name: str,
    backend: str | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> None:
    """Create `name` if talking to it would otherwise fail, and say so.

    The notice goes to stderr: stdout carries the reply and is what a pipe
    reads, and an agent having been created is commentary on the turn, not
    part of it.
    """
    agent, created = orchestrator.ensure(
        name,
        backend,
        model=model,
        reasoning_effort=reasoning_effort,
    )
    if created:
        print(
            f"created agent '{agent.name}' backend={agent.backend.name}",
            file=sys.stderr,
        )
        return
    if backend is None and model is None and reasoning_effort is None:
        return
    expected = (
        DEFAULT_BACKEND if backend is None else backend,
        model,
        reasoning_effort,
    )
    actual = _agent_config(agent)
    if actual != expected:
        raise OrchestratorError(
            f"agent '{agent.name}' already uses backend/model/effort {actual!r}; "
            f"configured {expected!r}"
        )


def cmd_talk(orchestrator: Orchestrator, opts: argparse.Namespace) -> None:
    prompt = opts.prompt
    composed = prompt
    if opts.skill is not None:
        try:
            names = parse_skill_names(opts.skill)
            resolved = resolve_skills(names, SKILLS_DIR)
        except SkillError as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(1) from None
        composed = compose_skill_prompt(resolved, prompt)
        log.info(
            "agent '%s': attaching skill(s) %s",
            opts.name,
            ", ".join(name for name, _path in resolved),
        )
    schema = None
    if opts.schema is not None:
        try:
            schema = load_schema(Path(opts.schema))
        except SchemaLoadError as exc:
            # Exit 2, not 1: the schema file is a bad argument, the same class
            # of mistake argparse exits 2 for. A caller can tell "fix your
            # schema" from "the agent failed" without reading the message.
            print(str(exc), file=sys.stderr)
            raise SystemExit(2) from None
        log.info("agent '%s': validating the reply against %s", opts.name, schema.path)
    # After the skills and the schema resolve, so a bad argument exits without
    # having left a new agent behind for a turn that never ran.
    _ensure_agent(
        orchestrator,
        opts.name,
        opts.backend,
        opts.model,
        opts.reasoning_effort,
    )
    try:
        result = orchestrator.talk(
            opts.name,
            composed,
            schema=schema,
            retries=opts.retries,
            timeout=opts.timeout,
        )
    except (TurnError, ReplyValidationError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None
    print(f"[{opts.name} session={result.session_id}]")
    if schema is None:
        print(result.reply)
        return
    # The validated object rather than the reply text: same content, but
    # parsed once here so a caller piping this gets one canonical spelling.
    print(json.dumps(result.structured, indent=2, sort_keys=True))


def _print_agents(orchestrator: Orchestrator) -> None:
    agents = orchestrator.list_agents()
    if not agents:
        print("no agents")
        return
    for name in agents:
        agent = orchestrator.agents[name]
        sid = agent.session_id or "-"
        print(f"{name:20} backend={agent.backend.name:6} session={sid}")


def cmd_list(orchestrator: Orchestrator, opts: argparse.Namespace) -> None:
    if opts.target == "agents":
        _print_agents(orchestrator)
        return
    try:
        catalog = index_skills(SKILLS_DIR)
    except SkillError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None
    print(format_skill_listing(catalog))


def cmd_delete(orchestrator: Orchestrator, opts: argparse.Namespace) -> None:
    agent = orchestrator.delete(opts.name)
    print(f"deleted agent '{agent.name}' backend={agent.backend.name}")


def _retry_count(raw: str) -> int:
    """--retries as a count, rejecting a negative one.

    argparse turns the raised error into its own exit 2. Without this, -1
    would mean "no attempts at all", which is not a thing this command can do.
    """
    count = int(raw)
    if count < 0:
        raise argparse.ArgumentTypeError(f"expected 0 or more, got {count}")
    return count


def _positive_seconds(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected 1 or more, got {value}")
    return value


# The level each verbosity selects, indexed by the summed argparse counts.
VERBOSITY_LEVELS = (logging.WARNING, logging.DEBUG, TRACE)

# Raised by the verbose flags. Only this project's loggers are turned up:
# setting the root logger to DEBUG would also enable every dependency's debug
# output, so the one signal being asked for would arrive buried in third-party
# noise.
OWN_LOGGERS = ("orchestrator", "backends")


class _VersionAction(argparse.Action):
    """Print the project version and stop before argparse validates the rest."""

    def __init__(self, option_strings: list[str], dest: str, **kwargs: Any) -> None:
        super().__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        del namespace, values, option_string
        _print_version()
        parser.exit(0)


class _CLIArgumentParser(argparse.ArgumentParser):
    """Use the selected verb's usage line for leftover-argument errors."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._verb_parsers: dict[str, _CLIArgumentParser] = {}
        self._error_parser: argparse.ArgumentParser | None = None

    def error(self, message: str) -> NoReturn:
        if self._error_parser is not None:
            self._error_parser.error(message)
        super().error(message)

    def parse_args(
        self,
        args: Iterable[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        arguments = list(args) if args is not None else sys.argv[1:]
        self._error_parser = None
        for token in arguments:
            if token in self._verb_parsers:
                self._error_parser = self._verb_parsers[token]
                break
        return cast(argparse.Namespace, super().parse_args(arguments, namespace))


def _add_version_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--version",
        action=_VersionAction,
        default=argparse.SUPPRESS,
        help="show the installed version",
    )


def _add_verbosity_argument(parser: argparse.ArgumentParser, dest: str) -> None:
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        dest=dest,
        help="log each step and how long it took; repeat for full prompts",
    )


def _add_verb_parser(
    subparsers: argparse._SubParsersAction,
    verb: str,
    **kwargs: Any,
) -> argparse.ArgumentParser:
    kwargs["prog"] = f"orchestrator {verb}"
    return subparsers.add_parser(verb, **kwargs)


def _build_parser() -> argparse.ArgumentParser:
    parser = _CLIArgumentParser(prog="orchestrator")
    _add_version_argument(parser)
    _add_verbosity_argument(parser, "verbosity")
    subparsers = parser.add_subparsers(dest="verb", required=True, metavar="<verb>")

    create = _add_verb_parser(subparsers, "create")
    _add_version_argument(create)
    _add_verbosity_argument(create, "verbosity_after")
    create.add_argument("name")
    _add_agent_config_options(create)
    create.set_defaults(_parser=create)

    talk = _add_verb_parser(
        subparsers,
        "talk",
        epilog=(
            "prompt source: orchestrator talk NAME "
            "[-p TEXT | --prompt-file PATH | -- PROMPT...]"
        ),
    )
    _add_version_argument(talk)
    _add_verbosity_argument(talk, "verbosity_after")
    _add_agent_config_options(talk)
    talk.add_argument("name")
    talk.add_argument("-s", "--skill")
    talk.add_argument("--schema")
    talk.add_argument(
        "--retries", type=_retry_count, default=DEFAULT_VALIDATION_RETRIES
    )
    talk.add_argument("--timeout", type=_positive_seconds, default=DEFAULT_TURN_TIMEOUT)
    talk.add_argument("-p", "--prompt")
    talk.add_argument("--prompt-file")
    talk.set_defaults(_parser=talk)

    list_parser = _add_verb_parser(subparsers, "list")
    _add_version_argument(list_parser)
    _add_verbosity_argument(list_parser, "verbosity_after")
    list_parser.add_argument(
        "target", nargs="?", choices=("agents", "skills"), default="agents"
    )
    list_parser.set_defaults(_parser=list_parser)

    delete = _add_verb_parser(subparsers, "delete")
    _add_version_argument(delete)
    _add_verbosity_argument(delete, "verbosity_after")
    delete.add_argument("name")
    delete.set_defaults(_parser=delete)

    doctor = _add_verb_parser(subparsers, "doctor")
    _add_version_argument(doctor)
    _add_verbosity_argument(doctor, "verbosity_after")
    doctor.set_defaults(_parser=doctor)

    parser._verb_parsers = subparsers.choices
    return parser


VERBS: dict[str, Callable[[Orchestrator, argparse.Namespace], None]] = {
    "create": cmd_create,
    "talk": cmd_talk,
    "list": cmd_list,
    "delete": cmd_delete,
}


def _configure_logging(verbosity: int) -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if verbosity:
        for name in OWN_LOGGERS:
            logging.getLogger(name).setLevel(VERBOSITY_LEVELS[verbosity])


def _project_version() -> str | None:
    """Read the version from the checkout containing this package, if valid."""
    project_file = Path(__file__).resolve().parent.parent / "pyproject.toml"
    try:
        with project_file.open("rb") as stream:
            document = tomllib.load(stream)
    except (OSError, ValueError, AttributeError, TypeError):
        return None
    project = document.get("project")
    if not isinstance(project, dict):
        return None
    version = project.get("version")
    return version if isinstance(version, str) and version else None


def _resolve_version() -> str:
    """Resolve the distribution version without touching CLI runtime state."""
    version = _project_version()
    if version is not None:
        return version
    try:
        installed_version = importlib.metadata.version("agents-army")
    except (importlib.metadata.PackageNotFoundError, ValueError, TypeError):
        raise ValueError from None
    if not isinstance(installed_version, str) or not installed_version:
        raise ValueError
    return installed_version


def _print_version() -> None:
    try:
        version = _resolve_version()
    except (ValueError, TypeError):
        print("unable to determine agents-army version", file=sys.stderr)
        raise SystemExit(1) from None
    print(version)


# The interpreter floor from pyproject's requires-python. Duplicated as a
# tuple because sys.version_info is what the running process can be compared
# against, and parsing the specifier back out of the metadata would report on
# the checkout rather than on the interpreter actually executing this.
MIN_PYTHON = (3, 11)

# Every tool `doctor` reports, in the order it prints them, paired
# with whether its absence is fine. Only jq is optional: agent CLIs
# are listed separately rather than collapsed into one "at least one" line, so
# the report says which backends this machine can actually run.
DEPENDENCY_TOOLS: tuple[tuple[str, bool], ...] = (
    ("uv", False),
    ("claude", False),
    ("codex", False),
    ("grok", False),
    ("opencode", False),
    ("jq", True),
)

# Present and required, present and optional, absent.
FOUND = "\u2713"
FOUND_OPTIONAL = "\u25cb"
NOT_FOUND = "\u2717"

# What a CLI may put between its own name and its version number, when it
# prints the name at all: `uv 0.4.18` against `jq-1.7`.
NAME_SEPARATORS = (" ", "-")

# A version probe is a courtesy, not the check: a CLI that hangs on --version
# must not hang the report, so it gets seconds rather than the turn timeout.
VERSION_PROBE_TIMEOUT = 5


def _tool_version(tool: str) -> str | None:
    """The first line of `<tool> --version`, or None if it cannot be had.

    Every failure mode is the same answer — the tool is installed and its
    version is unknown — so a CLI that is missing its runtime, hangs, exits
    non-zero, or prints nothing degrades the line instead of the command.
    """
    try:
        proc = subprocess.run(
            [tool, "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=VERSION_PROBE_TIMEOUT,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    lines = proc.stdout.splitlines()
    first_line = lines[0].strip() if lines else ""
    return first_line or None


def _describe_version(tool: str, reported: str) -> str:
    """`<tool> <version>`, without repeating a name the tool printed itself.

    The CLIs disagree about their own version line: `uv --version` prints
    "uv 0.4.18", `jq --version` prints "jq-1.7", `claude --version` prints a
    bare number, and `codex --version` prints "codex-cli 0.147.0". A leading
    copy of the tool's name is dropped only when a version number is what
    follows it, so codex keeps the product name it actually reports instead
    of being rewritten into "codex cli".
    """
    remainder = reported.removeprefix(tool)
    version = remainder[1:] if remainder[:1] in NAME_SEPARATORS else remainder
    if version[:1].isdigit():
        return f"{tool} {version}"
    if reported.startswith(tool):
        return reported
    return f"{tool} {reported}"


def _status_line(symbol: str, subject: str, note: str | None, optional: bool) -> str:
    """One report line, with its parenthesised notes rendered at most once."""
    notes = [note] if note is not None else []
    if optional:
        notes.append("optional")
    if not notes:
        return f"{symbol} {subject}"
    return f"{symbol} {subject} ({', '.join(notes)})"


def _python_line() -> str:
    """The running interpreter, checked against the floor this project needs.

    Not routed through `_status_line`: the interpreter is not a PATH lookup
    and can never be the optional half of that signature.
    """
    running = ".".join(str(part) for part in sys.version_info[:3])
    if (sys.version_info[0], sys.version_info[1]) >= MIN_PYTHON:
        return f"{FOUND} Python {running}"
    required = ".".join(str(part) for part in MIN_PYTHON)
    return f"{NOT_FOUND} Python {running} (needs {required}+)"


def _tool_line(tool: str, optional: bool) -> str:
    """One tool's line: found via PATH, with a version where one is available."""
    if shutil.which(tool) is None:
        return _status_line(NOT_FOUND, tool, "not found", optional)
    symbol = FOUND_OPTIONAL if optional else FOUND
    reported = _tool_version(tool)
    if reported is None:
        return _status_line(symbol, tool, "version unknown", optional)
    return _status_line(symbol, _describe_version(tool, reported), None, optional)


def _dependency_report() -> list[str]:
    """Every line of the setup report, in the fixed order it is printed."""
    return [
        _python_line(),
        *(_tool_line(tool, optional) for tool, optional in DEPENDENCY_TOOLS),
    ]


def _print_dependency_check() -> None:
    """Report the setup and stop.

    A status report, not a gate: it exits 0 whether every tool is present or
    none of them are, because which backends are usable is the user's call and
    a missing optional jq is not a failure at all.
    """
    for line in _dependency_report():
        print(line)


def _resolve_talk_prompt(
    opts: argparse.Namespace, tail: list[str], separator_present: bool
) -> None:
    sources = (
        opts.prompt is not None,
        opts.prompt_file is not None,
        separator_present,
    )
    if sum(sources) != 1:
        opts._parser.error("talk requires exactly one prompt source")
    if opts.prompt_file is not None:
        path = Path(opts.prompt_file).resolve()
        try:
            prompt = Path(opts.prompt_file).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            opts._parser.error(f"cannot read prompt file {path}: {exc}")
    elif separator_present:
        prompt = " ".join(tail)
    else:
        prompt = opts.prompt
    prompt = prompt.strip()
    if not prompt:
        opts._parser.error("talk prompt must not be empty")
    opts.prompt = prompt


def main(argv: list[str] | None = None) -> None:
    raw_argv = sys.argv[1:] if argv is None else argv
    separator_index = raw_argv.index("--") if "--" in raw_argv else len(raw_argv)
    separator_present = separator_index < len(raw_argv)
    if separator_present:
        head = raw_argv[:separator_index]
        tail = raw_argv[separator_index + 1 :]
    else:
        head = raw_argv
        tail = []

    parser = _build_parser()
    opts = parser.parse_args(head)
    if separator_present and opts.verb != "talk":
        opts._parser.error("the -- separator is only valid for talk")
    if opts.verb == "doctor":
        _print_dependency_check()
        return

    verbosity = min(opts.verbosity + opts.verbosity_after, len(VERBOSITY_LEVELS) - 1)
    _configure_logging(verbosity)
    # The prompt is one of these arguments, so log the shape and not the values.
    log.debug("cli: %d argument(s) after flag splitting", len(head) + len(tail))
    if opts.verb == "talk":
        _resolve_talk_prompt(opts, tail, separator_present)

    try:
        orch = Orchestrator()
        log.debug("cli: dispatching '%s'", opts.verb)
        VERBS[opts.verb](orch, opts)
    except _CLI_ERRORS as exc:
        # KeyError(str) renders as '"message"' — print the payload, not repr.
        message = exc.args[0] if exc.args else str(exc)
        print(message, file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":  # pragma: no cover
    main()
