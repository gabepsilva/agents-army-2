"""Behavior of the runtime-path ladder, driven directly.

Every case supplies its own env mapping, cwd, and user home, so nothing here
depends on the environment the test process happens to run in.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

from orchestrator.paths import RuntimePaths

REPO = Path(__file__).resolve().parents[1]
CWD = Path("/cwd")
USER_HOME = Path("/user")


def resolve(**env: str) -> RuntimePaths:
    return RuntimePaths.from_env(env, cwd=CWD, user_home=USER_HOME)


class TestFromEnv:
    def test_defaults_when_nothing_is_set(self) -> None:
        paths = resolve()
        assert paths.root == USER_HOME / ".agents-army"
        assert paths.home == CWD
        assert (
            paths.state_file == USER_HOME / ".agents-army" / "orchestrator_state.json"
        )
        assert paths.workdir == CWD
        assert paths.skills_dir == CWD / "SKILLS"
        assert paths.teams_dir is None

    def test_root_override_moves_root_and_the_default_state_file(self) -> None:
        paths = resolve(AGENTS_ARMY_ROOT="/root")
        assert paths.root == Path("/root")
        assert paths.state_file == Path("/root/orchestrator_state.json")
        assert paths.home == CWD

    def test_home_override_moves_workdir_skills_and_state_file(self) -> None:
        paths = resolve(AGENTS_ARMY_HOME="/home")
        assert paths.home == Path("/home")
        assert paths.workdir == Path("/home")
        assert paths.skills_dir == Path("/home/SKILLS")
        assert paths.state_file == Path("/home/orchestrator_state.json")
        assert paths.root == USER_HOME / ".agents-army"

    def test_explicit_state_file_wins_over_home(self) -> None:
        paths = resolve(
            AGENTS_ARMY_HOME="/home", AGENTS_ARMY_STATE_FILE="/s/state.json"
        )
        assert paths.state_file == Path("/s/state.json")
        assert paths.workdir == Path("/home")

    def test_explicit_state_file_wins_over_root(self) -> None:
        paths = resolve(
            AGENTS_ARMY_ROOT="/root", AGENTS_ARMY_STATE_FILE="/s/state.json"
        )
        assert paths.state_file == Path("/s/state.json")

    def test_home_set_to_the_cwd_still_takes_the_home_branch(self) -> None:
        """Presence of the key decides, not whether its value differs from cwd.

        An implementation comparing `home != cwd` would fall through to the
        root fallback branch here and put the registry somewhere else entirely.
        """
        paths = resolve(AGENTS_ARMY_HOME=str(CWD))
        assert paths.state_file == CWD / "orchestrator_state.json"

    def test_skills_override_is_independent_of_home(self) -> None:
        paths = resolve(AGENTS_ARMY_HOME="/home", AGENTS_ARMY_SKILLS="/skills")
        assert paths.skills_dir == Path("/skills")
        assert paths.workdir == Path("/home")

    def test_teams_dir_is_none_unless_set(self) -> None:
        assert resolve().teams_dir is None
        assert resolve(AGENTS_ARMY_TEAMS_DIR="/teams").teams_dir == Path("/teams")

    def test_every_variable_at_once(self) -> None:
        paths = resolve(
            AGENTS_ARMY_ROOT="/root",
            AGENTS_ARMY_HOME="/home",
            AGENTS_ARMY_STATE_FILE="/s/state.json",
            AGENTS_ARMY_SKILLS="/skills",
            AGENTS_ARMY_TEAMS_DIR="/teams",
        )
        assert paths == RuntimePaths(
            root=Path("/root"),
            home=Path("/home"),
            state_file=Path("/s/state.json"),
            workdir=Path("/home"),
            skills_dir=Path("/skills"),
            teams_dir=Path("/teams"),
        )

    def test_fields_cannot_be_rebound(self) -> None:
        paths = resolve()
        with pytest.raises(AttributeError):
            # The suppression below is the point of the test: ty rejects
            # this assignment statically, and the assertion is that the
            # frozen dataclass rejects it at runtime too. Narrowed to this
            # one line, in a test; no production code path involved.
            paths.state_file = Path("/elsewhere")  # ty: ignore[invalid-assignment]


class TestForTeam:
    def test_team_paths_sit_under_the_team_root(self) -> None:
        team = resolve().for_team(Path("/teams/alpha"), {})
        assert team.state_file == Path("/teams/alpha/agents/orchestrator_state.json")
        assert team.workdir == Path("/teams/alpha/worktree")
        assert team.skills_dir == Path("/teams/alpha/worktree/SKILLS")

    def test_explicit_skills_beats_the_worktree_default(self) -> None:
        team = resolve().for_team(
            Path("/teams/alpha"), {"AGENTS_ARMY_SKILLS": "/skills"}
        )
        assert team.skills_dir == Path("/skills")

    def test_root_home_and_teams_dir_are_carried_through(self) -> None:
        base = resolve(AGENTS_ARMY_ROOT="/root", AGENTS_ARMY_TEAMS_DIR="/teams")
        team = base.for_team(Path("/teams/alpha"), {})
        assert (team.root, team.home, team.teams_dir) == (
            base.root,
            base.home,
            base.teams_dir,
        )

    def test_the_base_object_is_left_alone(self) -> None:
        base = resolve()
        base.for_team(Path("/teams/alpha"), {})
        assert base.state_file == USER_HOME / ".agents-army" / "orchestrator_state.json"
        assert base.workdir == CWD


# Dotted forms the module uses today, plus the bare names a
# `from os import environ` would leave behind — a rule that only knew
# `os.environ` would wave that rewrite straight through.
AMBIENT = ("os.environ", "os.getenv", "Path.cwd", "Path.home", "environ", "getenv")


def _dotted(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


NESTED_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _same_scope(node: ast.AST) -> Iterator[ast.AST]:
    """Every node under `node` except what a def, class or lambda body hides.

    A nested scope's own header — decorators, default arguments, base
    classes — still executes at module scope, so it is walked; only the
    body is skipped.
    """
    yield node
    children = (
        _header_nodes(node)
        if isinstance(node, NESTED_SCOPES)
        else ast.iter_child_nodes(node)
    )
    for child in children:
        yield from _same_scope(child)


def _header_nodes(node: ast.AST) -> Iterator[ast.AST]:
    """The parts of a def/class/lambda that run where it is written."""
    for field in ("decorator_list", "bases", "keywords", "args", "returns"):
        value = getattr(node, field, None)
        for child in value if isinstance(value, list) else [value]:
            if isinstance(child, ast.AST):
                yield child


def _reads_ambient_state(node: ast.AST) -> bool:
    return any(_dotted(inner) in AMBIENT for inner in _same_scope(node))


class TestModuleScopeIsAmbientFree:
    """The extraction only holds if the ladder cannot regrow at import time."""

    def test_orchestrator_does_not_read_ambient_state_at_module_scope(self) -> None:
        source = (REPO / "orchestrator/__init__.py").read_text(encoding="utf-8")
        module = ast.parse(source)
        reading = [
            statement for statement in module.body if _reads_ambient_state(statement)
        ]
        rendered = [ast.unparse(statement) for statement in reading]
        assert reading == [], rendered

    def test_the_paths_module_reads_nothing_ambient_at_all(self) -> None:
        module = ast.parse((REPO / "orchestrator/paths.py").read_text(encoding="utf-8"))
        offenders = [
            ast.unparse(node)
            for node in ast.walk(module)
            if isinstance(node, ast.Attribute | ast.Name) and _dotted(node) in AMBIENT
        ]
        assert offenders == []
