#!/usr/bin/env python3
"""Command-line interface for the long-lived agent orchestrator."""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import re
import shutil
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext
from pathlib import Path
from typing import Any, NoReturn, cast

import orchestrator.core as core
import orchestrator.paths as paths
import orchestrator.teams as teams
from backends import TurnError, list_backends
from backends.base import DEFAULT_TURN_TIMEOUT
from backends.registry import UnknownBackendError
from orchestrator.core import (
    TRACE,
    Agent,
    OrchestratorError,
    StateError,
    TeamBusyError,
    log,
)
from orchestrator.doctor import (
    _print_dependency_check,
    _print_version,
)
from orchestrator.schema import SchemaError, SchemaLoadError, load_schema
from orchestrator.skills import (
    SkillError,
    compose_skill_prompt,
    format_skill_listing,
    index_skills,
    parse_skill_names,
    resolve_skills,
)

# User-facing failures that must print one line and exit, never a traceback,
# paired with the exit code each one earns. Scanned in order and the first
# match wins, so the most specific entry comes first: SchemaLoadError is a
# SchemaError but exits 2, and an exact-type dict would miss the subclasses
# backends raise (ClaudeTurnError and friends).
#
# Every entry is a leaf the code raises on purpose. Listing a base broad
# enough to catch an incidental builtin — ValueError under
# UnknownBackendError, RuntimeError under TurnError — would reprint a real
# bug inside a backend as a one-line user mistake with no traceback, which
# is what OrchestratorError above exists to prevent.
# UnknownBackendError comes from the registry, which cannot import this module.
_CLI_EXIT_CODES: tuple[tuple[tuple[type[Exception], ...], int], ...] = (
    # A schema file that will not load is a bad argument, the same class of
    # mistake argparse exits 2 for. A caller can tell "fix your schema" from
    # "the agent failed" without reading the message.
    ((SchemaLoadError,), 2),
    (
        (
            core.OrchestratorError,
            UnknownBackendError,
            SkillError,
            SchemaError,
            TurnError,
        ),
        1,
    ),
)
# The `except` clause's tuple, derived from the table above so the two cannot
# drift: every type the boundary catches has a code, and vice versa.
_CLI_ERRORS = tuple(error for errors, _code in _CLI_EXIT_CODES for error in errors)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_team_option(parser: argparse.ArgumentParser) -> None:
    # Its own helper, deliberately not folded into _add_agent_config_options:
    # that trio feeds _agent_config/_ensure_agent, which *assert* a stored
    # agent's configuration matches the flags. A team is a namespace to
    # select, not configuration to assert. Long flag only — no -t: it would
    # sit next to talk's --timeout, a footgun for a flag scripts pass once.
    parser.add_argument(
        "--team",
        help=(
            "run against team <team>'s {agents,worktree} instead of the "
            "teamless layout; found under $AGENTS_ARMY_TEAMS_DIR if set, "
            "otherwise resolved under $AGENTS_ARMY_ROOT"
        ),
    )


def _add_agent_config_options(parser: argparse.ArgumentParser) -> None:
    # No argparse default: leaving this None lets create() and ensure() resolve
    # DEFAULT_BACKEND, which is what that constant documents itself as. A
    # literal here would pin them to claude however DEFAULT_BACKEND changed.
    parser.add_argument("--backend", "-b", choices=list_backends())
    parser.add_argument("--model", "-m")
    parser.add_argument("--reasoning-effort", "-e")


def _agent_config(agent: Agent) -> tuple[str, str | None, str | None]:
    return agent.backend.name, agent.backend.model, agent.backend.reasoning_effort


def cmd_create(orchestrator: core.Orchestrator, opts: argparse.Namespace) -> None:
    agent = orchestrator.spawn(
        opts.name,
        opts.backend,
        model=opts.model,
        reasoning_effort=opts.reasoning_effort,
    )
    print(f"created agent '{agent.name}' backend={agent.backend.name}")


def cmd_fork(orchestrator: core.Orchestrator, opts: argparse.Namespace) -> None:
    agent = orchestrator.fork(opts.source, opts.dest)
    print(
        f"forked agent '{opts.source}' into '{agent.name}' backend={agent.backend.name}"
    )


def _ensure_agent(
    orchestrator: core.Orchestrator,
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
        core.DEFAULT_BACKEND if backend is None else backend,
        model,
        reasoning_effort,
    )
    actual = _agent_config(agent)
    if actual != expected:
        raise OrchestratorError(
            f"agent '{agent.name}' already uses backend/model/effort {actual!r}; "
            f"configured {expected!r}"
        )


def cmd_talk(orchestrator: core.Orchestrator, opts: argparse.Namespace) -> None:
    prompt = opts.prompt
    composed = prompt
    if opts.skill is not None:
        names = parse_skill_names(opts.skill)
        resolved = resolve_skills(names, orchestrator.runtime_paths.skills_dir)
        composed = compose_skill_prompt(resolved, prompt)
        log.info(
            "agent '%s': attaching skill(s) %s",
            opts.name,
            ", ".join(name for name, _path in resolved),
        )
    schema = None
    if opts.schema is not None:
        schema = load_schema(Path(opts.schema))
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
    result = orchestrator.talk(
        opts.name,
        composed,
        schema=schema,
        retries=opts.retries,
        timeout=opts.timeout,
        stream=opts.stream,
    )
    print(f"[{opts.name} session={result.session_id}]")
    if schema is None:
        print(result.reply)
        return
    # The validated object rather than the reply text: same content, but
    # parsed once here so a caller piping this gets one canonical spelling.
    print(json.dumps(result.structured, indent=2, sort_keys=True))


def cmd_chat(orchestrator: core.Orchestrator, opts: argparse.Namespace) -> None:
    """Run the selected backend's interactive session and preserve its status."""
    returncode = orchestrator.chat(opts.name)
    if returncode:
        raise SystemExit(returncode)


def _print_agents(orchestrator: core.Orchestrator) -> None:
    # Printed unconditionally, including the `no agents` case: which
    # registry a `--team`/AGENTS_ARMY_STATE_FILE/AGENTS_ARMY_HOME ladder
    # resolved to is exactly what's unknowable without this line.
    print(f"registry: {orchestrator.state_file}")
    agents = orchestrator.list_agents()
    if not agents:
        print("no agents")
        return
    name_width = max(20, max(len(n) for n in agents))
    rows = []
    for name in agents:
        agent = orchestrator.agents[name]
        model = agent.backend.model or "-"
        effort = agent.backend.reasoning_effort or "-"
        turns = "-" if agent.turns is None else str(agent.turns)
        created = agent.created_at or "-"
        last = agent.last_turn_at or "-"
        # A fixed-width marker, not a bare "busy" appended only when true, so
        # the session= column starts at the same offset whether or not this
        # agent is mid-turn.
        busy = "busy" if orchestrator._agent_is_busy(name) else "    "
        sid = agent.session_id or "-"
        rows.append(
            (name, agent.backend.name, model, effort, turns, created, last, busy, sid)
        )
    # Every column but session= is measured from the data, the same way the
    # name column already was: a fixed width (a hard-coded 6 for `model`, say)
    # just moves the overflow cliff to the first value wider than the
    # constant — a `gpt-5-codex` model name or an `opencode` backend both
    # overflowed a `:6` field, dragging session= out of alignment with it.
    # session= is left unpadded and last on purpose: it's a 36-character
    # uuid, and padding it would only move the cliff onto the next listing.
    backend_w, model_w, effort_w, turns_w, created_w, last_w = (
        max(len(row[i]) for row in rows) for i in range(1, 7)
    )
    for name, backend, model, effort, turns, created, last, busy, sid in rows:
        print(
            f"{name:{name_width}}  backend={backend:{backend_w}}  "
            f"model={model:{model_w}}  effort={effort:{effort_w}}  "
            f"turns={turns:>{turns_w}}  created={created:{created_w}}  "
            f"last={last:{last_w}}  {busy}  session={sid}"
        )


def cmd_list(orchestrator: core.Orchestrator, opts: argparse.Namespace) -> None:
    if opts.target == "agents":
        _print_agents(orchestrator)
        return
    print(format_skill_listing(index_skills(orchestrator.runtime_paths.skills_dir)))


def _agents_from_registry(state_file: Path) -> dict[str, str] | None:
    """Agent name -> backend, read from `state_file`.

    `None` means the registry couldn't be turned into that mapping: bad
    JSON, removed by a concurrent `delete`/teardown between discovery and
    this read, or valid JSON that isn't the `{name: {"backend": ...}}` shape
    a registry is supposed to have (e.g. a top-level list, or an entry that
    isn't itself an object). `list teams` enumerates every team's registry
    in one pass, so one team's bad file must show up as a flag on that team,
    not abort the report for every other one. Distinct from `{}`, a
    registry that read fine and is simply empty.
    """
    try:
        raw = core._load_state_file(state_file)
    except (StateError, OSError):
        return None
    # The shape a registry is supposed to have, checked explicitly rather
    # than caught as an AttributeError off `raw.items()`/`entry.get(...)`:
    # a blanket except there would just as happily swallow a genuine future
    # bug in this function as the JSON's actual shape.
    if not isinstance(raw, dict) or not all(
        isinstance(entry, dict) for entry in raw.values()
    ):
        return None
    return {name: entry.get("backend", "?") for name, entry in raw.items()}


def _team_agents(team: teams.Team) -> dict[str, str] | None:
    return _agents_from_registry(teams.marker_path(team.path))


def _format_agents(agents: dict[str, str] | None) -> str:
    """`(N agents: name/backend, ...)` for a read registry, `(registry
    unreadable)` for one that exists but couldn't be read (see
    `_agents_from_registry`) — the caller decides whether `agents=None`
    means unreadable or "there is nothing here to print" (a registry that
    doesn't exist at all is never formatted, only read ones are)."""
    if agents is None:
        return "registry unreadable"
    count = len(agents)
    plural = "agent" if count == 1 else "agents"
    members = ", ".join(f"{n}/{b}" for n, b in sorted(agents.items()))
    return f"{count} {plural}" + (f": {members}" if members else "")


def _format_team_line(team: teams.Team, agents: dict[str, str] | None) -> str:
    line = f"  {team.name}  ({_format_agents(agents)})"
    if not team.has_worktree:
        line += "  [worktree missing]"
    return line


def _print_teams(runtime_paths: paths.RuntimePaths) -> None:
    root = runtime_paths.root
    teams_dir = runtime_paths.teams_dir
    state_file = runtime_paths.state_file
    root_teams = teams.discover(root)
    groups = [(root, root_teams)]
    if teams_dir is not None:
        # Walked unconditionally, then deduped by path — not skipped
        # whenever the configured team directory overlaps the configured root.
        # Dropping the whole group whenever the team directory is an *ancestor*
        # of the root hid every team outside the root, exactly what this command
        # exists to show. Deduping instead handles same-dir, descendant, and
        # ancestor with one rule: a team already shown under the root is simply
        # never repeated under the team directory.
        seen = {team.path for team in root_teams}
        extra_teams = [
            team for team in teams.discover(teams_dir) if team.path not in seen
        ]
        if extra_teams:
            groups.append((teams_dir, extra_teams))
    # The resolved state file, not "$root/orchestrator_state.json": the
    # registry `list teams` reports as (teamless) must be the one `list
    # agents`/`talk` actually use, which an explicit AGENTS_ARMY_STATE_FILE
    # or AGENTS_ARMY_HOME relocates away from root (see the state file
    # ladder in orchestrator.paths) — main() never lets --team reach this
    # function, so these paths are always the teamless resolution. Not
    # `agents/orchestrator_state.json` inside a directory, so `teams.discover`
    # never finds this bare file on its own — checked explicitly. Its
    # existence and its readability are tracked separately: a corrupt
    # registry here must still show up as a flagged line, the same as a
    # corrupt team registry does, rather than being indistinguishable from
    # "there was never a teamless registry at all".
    has_teamless = state_file.is_file()

    if not any(group_teams for _, group_teams in groups) and not has_teamless:
        print("no teams")
        return

    for group_root, group_teams in groups:
        print(f"{group_root}")
        if not group_teams:
            print("  no teams")
        for team in group_teams:
            print(_format_team_line(team, _team_agents(team)))
        print()

    if has_teamless:
        print(f"(teamless) {state_file}")
        teamless = _agents_from_registry(state_file)
        if not teamless:
            print(f"  {_format_agents(teamless)}")
        else:
            for name, backend in sorted(teamless.items()):
                print(f"  {name} backend={backend}")


def _teardown_team(team: str, team_root: Path) -> None:
    """Remove a team's registry, leaving its worktree and git metadata alone.

    Takes the already-resolved `team_root` rather than rebuilding it from a
    configured team directory: `_resolve_team` is the one place a team name
    is joined to a root, whether that root is configured directly or found by
    `teams.resolve`.

    Scoped to `agents/`: that directory holds the state file, its lock, and
    the directory of per-agent turn locks. `worktree/` is a git working tree —
    removing it is `git worktree remove`, and it is the caller's call, not
    teardown's. The team lock's own file, a sibling of `agents/`, survives.
    """
    agents_dir = team_root / "agents"
    if agents_dir.exists():
        shutil.rmtree(agents_dir)
    print(f"deleted team '{team}'")


def cmd_delete(orchestrator: core.Orchestrator, opts: argparse.Namespace) -> None:
    # Always a named agent: `delete --team T` with no name is teardown, and
    # main() runs that itself, before an Orchestrator exists.
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
    parser = subparsers.add_parser(verb, **kwargs)
    # Registered here, before the caller's own arguments, so `-v` stays the
    # first option after `-h` in every verb's usage line.
    _add_verbosity_argument(parser, "verbosity_after")
    parser.set_defaults(_parser=parser)
    return parser


def _build_parser() -> argparse.ArgumentParser:
    parser = _CLIArgumentParser(prog="orchestrator")
    _add_version_argument(parser)
    _add_verbosity_argument(parser, "verbosity")
    subparsers = parser.add_subparsers(dest="verb", required=True, metavar="<verb>")

    create = _add_verb_parser(subparsers, "create")
    create.add_argument("name")
    _add_agent_config_options(create)
    _add_team_option(create)

    talk = _add_verb_parser(
        subparsers,
        "talk",
        epilog=(
            "prompt source: orchestrator talk NAME "
            "[-p TEXT | --prompt-file PATH | -- PROMPT...]"
        ),
    )
    _add_agent_config_options(talk)
    talk.add_argument("name")
    talk.add_argument("-s", "--skill")
    talk.add_argument("--schema")
    talk.add_argument(
        "--retries", type=_retry_count, default=core.DEFAULT_VALIDATION_RETRIES
    )
    talk.add_argument("--timeout", type=_positive_seconds, default=DEFAULT_TURN_TIMEOUT)
    talk.add_argument(
        "--stream",
        action="store_true",
        help="render recognized backend events to stderr while the turn runs",
    )
    talk.add_argument("-p", "--prompt")
    talk.add_argument("--prompt-file")
    _add_team_option(talk)

    chat = _add_verb_parser(subparsers, "chat")
    chat.add_argument("name")
    _add_team_option(chat)

    fork = _add_verb_parser(subparsers, "fork")
    fork.add_argument("source")
    fork.add_argument("dest")
    _add_team_option(fork)

    list_parser = _add_verb_parser(subparsers, "list")
    list_parser.add_argument(
        "target", nargs="?", choices=("agents", "skills", "teams"), default="agents"
    )
    _add_team_option(list_parser)

    delete = _add_verb_parser(subparsers, "delete")
    # nargs="?": `--team T` alone tears the whole team down (see cmd_delete);
    # a name deletes one agent. Neither is an error — bare `delete` is.
    delete.add_argument("name", nargs="?")
    _add_team_option(delete)

    _add_verb_parser(subparsers, "doctor")

    parser._verb_parsers = subparsers.choices
    return parser


VERBS: dict[str, Callable[[core.Orchestrator, argparse.Namespace], None]] = {
    "create": cmd_create,
    "talk": cmd_talk,
    "chat": cmd_chat,
    "fork": cmd_fork,
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


# A team name becomes a directory name; an agent name never does (see
# Orchestrator._agent_lock_path, which digests it for exactly that reason).
# One or more '/'-joined segments, so a name can be a bare team ('issue-97')
# or a root-relative qualified tail ('agents-army-2/gdw-v3/issue-97') — the
# same string `teams.discover`/`teams.resolve` print and match against. '.'
# and '..' match the charset per segment but must still be rejected there:
# they escape the root (Path('/teams') / '..' / 'agents' ==
# Path('/teams/../agents')), and the escape works from any segment, not just
# the whole name.
_TEAM_NAME_RE = re.compile(r"[-_.A-Za-z0-9]+(?:/[-_.A-Za-z0-9]+)*")


def _team_lock_path(team_root: Path) -> Path:
    return team_root / ".lock"


@contextmanager
def _team_locked(path: Path, team: str, mode: int) -> Iterator[None]:
    """Hold the team lock, reporting a lost race as `TeamBusyError`.

    The conversion happens around the acquisition and nothing else. Catching
    `BlockingIOError` around the whole dispatch instead would claim it for any
    incidental one raised behind it — a backend's pipe, a write to a
    non-blocking stdout — and answer "the team is in use" to a caller that
    never asked for a team at all. That is the mystery `OrchestratorError`'s
    docstring describes.

    Teardown asks with `LOCK_NB` rather than a blocking `LOCK_EX` because
    Linux flock has no writer fairness: a queued exclusive waiter is
    overtaken by every later shared request, so a blocking teardown on a busy
    team would wait indefinitely. `TeamBusyError` exits 1, not argparse's 2 —
    a busy resource is not a usage mistake.
    """
    with ExitStack() as stack:
        try:
            stack.enter_context(core._flock(path, mode))
        except BlockingIOError:
            raise TeamBusyError(
                f"team '{team}' is in use by another command; try again "
                "once it finishes"
            ) from None
        yield


def _usage_error(opts: argparse.Namespace, message: str) -> NoReturn:
    """`opts._parser.error(message)`, typed `NoReturn`.

    `opts._parser` is a dynamically-set `argparse.Namespace` attribute, so
    neither `ty` nor a reader can see that `_CLIArgumentParser.error` never
    returns; a caller that must produce a value (`_resolve_team_root`) needs
    that fact spelled out. The fallback raise carries no message: it can
    never execute (`.error()` always raises `SystemExit` first), so a
    message here would be untestable text with nothing to check it against.
    """
    opts._parser.error(message)
    raise AssertionError  # pragma: no cover


def _validate_team_name(
    team: str, opts: argparse.Namespace, runtime_paths: paths.RuntimePaths
) -> None:
    if not _TEAM_NAME_RE.fullmatch(team) or any(
        segment in (".", "..") for segment in team.split("/")
    ):
        _usage_error(
            opts,
            f"invalid team name {team!r}: must match "
            f"{_TEAM_NAME_RE.pattern!r} segment-by-segment, and no segment "
            "may be '.' or '..'",
        )
    if "/" in team and runtime_paths.teams_dir is not None:
        # A qualified name is root-relative by construction — it is the
        # string `list teams` prints under the root header. A configured team
        # directory supplies its own namespace, so joining one under it
        # double-joins instead of resolving.
        _usage_error(
            opts,
            f"invalid team name {team!r}: a qualified name is relative to "
            "$AGENTS_ARMY_ROOT and cannot be used while AGENTS_ARMY_TEAMS_DIR "
            f"is set. Use the bare name {team.split('/')[-1]!r}, or "
            f"unset AGENTS_ARMY_TEAMS_DIR to resolve under {runtime_paths.root}.",
        )


def _resolve_team_root(
    team: str, opts: argparse.Namespace, runtime_paths: paths.RuntimePaths
) -> Path:
    """The one place a team name is joined to a root.

    `AGENTS_ARMY_TEAMS_DIR` set short-circuits: `team_root` is just
    the configured team directory joined with `team`, exactly as before this
    function existed, with no
    walk and no ambiguity — the one script that matters (`go.sh`) exports it
    and never reaches the branch below.

    `AGENTS_ARMY_TEAMS_DIR` unset walks `$AGENTS_ARMY_ROOT` with
    `teams.resolve` and never guesses: one hit is used, zero or two-or-more
    are reported through `_usage_error` (exit 2) — a usage problem (bad
    name, wrong environment, team lives elsewhere) regardless of which verb
    asked, teardown included. That is distinct from the configured-team-
    directory branch's own not-found case, handled by the caller once
    `team_root` comes back here: a team directory/name that simply does not exist on disk
    is "this resource is not there", not a usage mistake.
    """
    if runtime_paths.teams_dir is not None:
        return runtime_paths.teams_dir / team
    hits = teams.resolve(runtime_paths.root, team)
    if len(hits) == 1:
        return hits[0]
    if not hits:
        _usage_error(
            opts,
            f"no team named {team!r} under {runtime_paths.root}; a team is a directory "
            "with an agents/ or worktree/ subdirectory, e.g.:\n"
            f"  git worktree add -B {team} "
            f"{runtime_paths.root}/<repo>/<workflow>/{team}/worktree ...\n"
            "if the team lives outside $AGENTS_ARMY_ROOT, export "
            "AGENTS_ARMY_TEAMS_DIR to point at its parent",
        )
    _usage_error(
        opts,
        f"team name {team!r} is ambiguous under {runtime_paths.root}:\n"
        + "\n".join(f"  {hit}" for hit in hits)
        + "\nre-run with a qualified name, e.g. --team "
        + hits[0].relative_to(runtime_paths.root).as_posix(),
    )


def _resolve_team(
    opts: argparse.Namespace,
    runtime_paths: paths.RuntimePaths,
    env: Mapping[str, str],
    teardown: bool,
) -> tuple[paths.RuntimePaths, AbstractContextManager[None]]:
    """Resolve the run's paths and, for a `--team` run, lock the team.

    Returns the `RuntimePaths` every read site downstream works from, so
    nothing here rebinds module state and a second `main()` in one process
    cannot inherit the first one's team.

    Teamless commands (`opts.team is None`) get the supplied runtime paths plus
    `nullcontext()`.

    Every check here runs before `Orchestrator()` is constructed and reports
    through `opts._parser.error(...)` (exit 2), the way `_resolve_talk_prompt`
    already does — except for every verb but `create`/`talk`/`chat`/`fork`
    (`list`, `delete NAME`, and teardown) finding a `team_root` that doesn't exist,
    which is not a usage error and is left to raise `OrchestratorError`
    (exit 1), the same as any other `delete` of something that isn't there.
    """
    team = opts.team
    if team is None:
        return runtime_paths, nullcontext()
    _validate_team_name(team, opts, runtime_paths)
    if "AGENTS_ARMY_STATE_FILE" in env:
        opts._parser.error(
            "--team cannot be combined with an explicit AGENTS_ARMY_STATE_FILE "
            "(unset it, or drop --team)"
        )
    if "AGENTS_ARMY_HOME" in env:
        opts._parser.error(
            "--team cannot be combined with an explicit AGENTS_ARMY_HOME "
            "(unset it, or drop --team)"
        )
    team_root = _resolve_team_root(team, opts, runtime_paths)
    opts._team_root = team_root
    worktree = team_root / "worktree"
    if opts.verb in ("create", "talk", "chat", "fork"):
        # Gated on the verb, not on `teardown`: `list agents --team` and
        # `delete NAME --team` never launch a backend, they read and edit a
        # JSON file, so they must work on a team whose worktree is gone (the
        # state teardown deliberately leaves behind) or not there yet.
        # `create` and `fork` keep the gate because they store the workdir at
        # resolution — letting them through would only defer this same
        # failure to `talk` with a registry already written.
        if not worktree.is_dir():
            opts._parser.error(
                f"team workspace {worktree} does not exist; create it first "
                f"with 'git worktree add {worktree} ...'"
            )
    elif not team_root.is_dir():
        # Every other verb (list, delete NAME, and teardown) still needs
        # `team_root` itself to exist, even though none of them touch
        # `worktree/`. Skipping this check let a bogus `--team NAME` reach
        # `_team_locked` -> `_flock`, whose first statement is
        # `path.parent.mkdir(parents=True, exist_ok=True)` — silently
        # fabricating `team_root` (and, for `delete NAME`'s later
        # `_agent_lock_path`, `agents/` alongside it) on disk for a typo.
        # That residue is self-perpetuating: once it exists, `_walk`'s
        # candidate rule sees `agents/` and treats it as a real team on
        # every future AGENTS_ARMY_ROOT walk. Teardown must stay possible
        # after `git worktree remove`, or a team's state is orphaned
        # forever — that removes only `worktree/`, not `team_root`, so this
        # check still passes for it. A no-op under the AGENTS_ARMY_ROOT walk:
        # `teams.resolve` only ever returns directories that already exist.
        raise OrchestratorError(f"team '{team}' not found at {team_root}")
    # LOCK_SH for every team verb but teardown, so concurrent turns in one
    # team don't serialize on each other; LOCK_EX|LOCK_NB for teardown, so a
    # team must not be torn down while a command in it is running, and a
    # busy team fails fast (flock has no writer fairness — see _team_locked).
    mode = fcntl.LOCK_EX | fcntl.LOCK_NB if teardown else fcntl.LOCK_SH
    return (
        runtime_paths.for_team(team_root, env),
        _team_locked(_team_lock_path(team_root), team, mode),
    )


def main(argv: list[str] | None = None) -> None:
    env = dict(os.environ)
    runtime_paths = paths.RuntimePaths.from_env(
        env, cwd=Path.cwd(), user_home=Path.home()
    )
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
    if opts.verb == "delete" and opts.team is None and opts.name is None:
        opts._parser.error("delete requires NAME or --team")
    # `list teams` reads every team's registry, not one; --team names a
    # single team to resolve, which is a contradiction with "list them all".
    list_teams = opts.verb == "list" and opts.target == "teams"
    if list_teams and opts.team is not None:
        opts._parser.error("list teams cannot be combined with --team")

    # Only `delete` with no NAME tears a team down; create/talk always
    # require NAME, so this is False for them without inspecting opts.team.
    teardown = opts.verb == "delete" and opts.name is None
    try:
        runtime_paths, team_lock = _resolve_team(opts, runtime_paths, env, teardown)
        with team_lock:
            log.debug("cli: dispatching '%s'", opts.verb)
            if teardown:
                # Ahead of Orchestrator(), and not through VERBS: the
                # constructor parses the registry, and a registry that will
                # not parse — invalid JSON, a backend this build no longer
                # has — is exactly what teardown exists to remove. Building
                # one first left `rm -rf` as the only way to retire a team
                # whose state file had gone bad.
                _teardown_team(opts.team, opts._team_root)
            elif list_teams:
                # Also ahead of Orchestrator(): that constructor binds one
                # state file, and this reads N of them.
                _print_teams(runtime_paths)
            else:
                VERBS[opts.verb](core.Orchestrator(runtime_paths), opts)
    except _CLI_ERRORS as exc:
        # KeyError(str) renders as '"message"' — print the payload, not repr.
        message = exc.args[0] if exc.args else str(exc)
        print(message, file=sys.stderr)
        code = next(
            exit_code
            for errors, exit_code in _CLI_EXIT_CODES
            if isinstance(exc, errors)
        )
        raise SystemExit(code) from None


if __name__ == "__main__":  # pragma: no cover
    main()
