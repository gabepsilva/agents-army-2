"""Skill catalog lookup, prompt composition, and flag-based CLI invocation."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

import orchestrator.cli as cli
import orchestrator.core as core
import orchestrator.skills as skills_module
from backends.base import (
    DEFAULT_TURN_TIMEOUT,
    AgentBackend,
    OutputSchema,
    TurnResult,
)
from backends.claude import ClaudeTurnError
from backends.registry import register_backend
from orchestrator.cli import (
    cmd_list as _cmd_list,
)
from orchestrator.cli import (
    cmd_talk as _cmd_talk,
)
from orchestrator.cli import main
from orchestrator.core import Orchestrator
from orchestrator.skills import (
    PROMPT_HEADER,
    SkillError,
    compose_skill_prompt,
    format_skill_listing,
    index_skills,
    parse_skill_names,
    resolve_skills,
)
from tests.path_helpers import runtime_paths


def _talk_options(argv: list[str]) -> argparse.Namespace:
    separator = argv.index("--") if "--" in argv else None
    head = argv if separator is None else argv[:separator]
    tail = [] if separator is None else argv[separator + 1 :]
    options = cli._build_parser().parse_args(head)
    cli._resolve_talk_prompt(options, tail, separator is not None)
    return options


def _options(argv: list[str]) -> argparse.Namespace:
    return cli._build_parser().parse_args(argv)


class EchoBackend(AgentBackend):
    """CLI-free backend so these tests never spawn a real agent."""

    @property
    def name(self) -> str:
        return "echo"

    def run_turn(  # noqa: PLR0913 - test doubles mirror AgentBackend.run_turn public seam
        self,
        prompt: str,
        session_id: str | None,
        cwd: Path,
        timeout: int = DEFAULT_TURN_TIMEOUT,
        schema: OutputSchema | None = None,
        *,
        resume_as_fork: bool = False,
        stream: bool = False,
    ) -> TurnResult:
        return TurnResult(session_id="echo-sid", reply=f"echo:{prompt}", raw="")


@pytest.fixture(autouse=True)
def register_echo_backend() -> None:
    register_backend("echo", EchoBackend)


@pytest.fixture
def skills_tree(tmp_path: Path) -> Path:
    """A catalog with unique names, a three-way clash, and files that are not skills."""
    root = tmp_path / "SKILLS"
    foo = root / "nested" / "foo"
    foo.mkdir(parents=True)
    (foo / "SKILL.md").write_text("# foo\n", encoding="utf-8")
    (foo / "extra.md").write_text("# extra companion\n", encoding="utf-8")
    (foo / "README.md").write_text("# skill-local readme\n", encoding="utf-8")
    bar = root / "nested" / "deep" / "bar"
    bar.mkdir(parents=True)
    (bar / "SKILL.md").write_text("# bar\n", encoding="utf-8")
    one = root / "one" / "clash"
    one.mkdir(parents=True)
    (one / "SKILL.md").write_text("# clash-one\n", encoding="utf-8")
    two = root / "two" / "clash"
    two.mkdir(parents=True)
    (two / "SKILL.md").write_text("# clash-two\n", encoding="utf-8")
    also = root / "also"
    also.mkdir()
    (also / "clash.md").write_text("# clash-loose\n", encoding="utf-8")
    (root / "loose.md").write_text("# loose\n", encoding="utf-8")
    (root / "README.md").write_text("# catalog readme\n", encoding="utf-8")
    # A directory whose name matches the glob must not be indexed as a skill.
    (root / "not-a-file.md").mkdir()
    return root


def _foo_path(root: Path) -> Path:
    return (root / "nested" / "foo" / "SKILL.md").resolve()


def _bar_path(root: Path) -> Path:
    return (root / "nested" / "deep" / "bar" / "SKILL.md").resolve()


def _loose_path(root: Path) -> Path:
    return (root / "loose.md").resolve()


def _clash_paths(root: Path) -> list[Path]:
    return sorted(
        [
            (root / "also" / "clash.md").resolve(),
            (root / "one" / "clash" / "SKILL.md").resolve(),
            (root / "two" / "clash" / "SKILL.md").resolve(),
        ],
        key=str,
    )


def _expected_skill_listing(root: Path) -> str:
    lines = [f"{'bar':20} {_bar_path(root)}"]
    lines.extend(f"{'clash':20} {path}" for path in _clash_paths(root))
    lines.append(f"{'foo':20} {_foo_path(root)}")
    lines.append(f"{'loose':20} {_loose_path(root)}")
    return "\n".join(lines)


class TestParseSkillNames:
    def test_splits_on_comma_and_strips_whitespace(self) -> None:
        assert parse_skill_names("tdd, code-review") == ["tdd", "code-review"]

    def test_preserves_cli_order(self) -> None:
        assert parse_skill_names("code-review,tdd") == ["code-review", "tdd"]

    def test_single_name(self) -> None:
        assert parse_skill_names("tdd") == ["tdd"]

    def test_empty_token_is_an_error(self) -> None:
        for raw in ("tdd,", ",tdd", "", "tdd,,code-review"):
            with pytest.raises(SkillError) as excinfo:
                parse_skill_names(raw)
            assert str(excinfo.value) == "empty skill name in --skill"

    def test_duplicate_name_is_an_error(self) -> None:
        for raw in ("tdd,tdd", "tdd, code-review, tdd"):
            with pytest.raises(SkillError) as excinfo:
                parse_skill_names(raw)
            assert str(excinfo.value) == "duplicate skill name 'tdd' in --skill"


class TestIndexSkills:
    def test_indexes_directory_and_loose_skills(self, skills_tree: Path) -> None:
        catalog = index_skills(skills_tree)
        assert set(catalog) == {"foo", "bar", "loose", "clash"}
        assert catalog["foo"] == [_foo_path(skills_tree)]
        assert catalog["bar"] == [_bar_path(skills_tree)]
        assert catalog["loose"] == [_loose_path(skills_tree)]
        assert catalog["clash"] == _clash_paths(skills_tree)

    def test_does_not_index_companions_or_readmes(self, skills_tree: Path) -> None:
        catalog = index_skills(skills_tree)
        for paths in catalog.values():
            names = {path.name for path in paths}
            assert "extra.md" not in names
            assert "README.md" not in names

    def test_missing_directory_is_an_error(self, tmp_path: Path) -> None:
        missing = tmp_path / "nope"
        with pytest.raises(SkillError, match=f"skills directory not found: {missing}"):
            index_skills(missing)

    def test_file_where_a_directory_is_expected_is_an_error(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "SKILLS"
        target.write_text("not a dir\n", encoding="utf-8")
        with pytest.raises(SkillError, match=f"skills directory not found: {target}"):
            index_skills(target)

    def test_empty_directory_indexes_nothing(self, tmp_path: Path) -> None:
        root = tmp_path / "SKILLS"
        root.mkdir()
        assert index_skills(root) == {}


class TestResolveSkills:
    def test_lookup_uses_an_empty_list_as_the_missing_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        requested_defaults: list[object] = []

        class Catalog:
            def __init__(self) -> None:
                self.entries = {"foo": [skill]}

            def get(
                self, key: str, default: list[Path] | None = None
            ) -> list[Path] | None:
                requested_defaults.append(default)
                return self.entries.get(key, default)

            def __iter__(self) -> Iterator[str]:
                return iter(self.entries)

        skill = tmp_path / "SKILL.md"
        catalog = Catalog()
        monkeypatch.setattr(skills_module, "index_skills", lambda _: catalog)

        assert resolve_skills(["foo"], tmp_path) == [("foo", skill)]
        assert requested_defaults == [[]]

    def test_directory_skill_is_the_skill_md(self, skills_tree: Path) -> None:
        assert resolve_skills(["foo"], skills_tree) == [("foo", _foo_path(skills_tree))]

    def test_nested_directory_skill(self, skills_tree: Path) -> None:
        assert resolve_skills(["bar"], skills_tree) == [("bar", _bar_path(skills_tree))]

    def test_loose_file_skill(self, skills_tree: Path) -> None:
        assert resolve_skills(["loose"], skills_tree) == [
            ("loose", _loose_path(skills_tree))
        ]

    def test_preserves_requested_order(self, skills_tree: Path) -> None:
        assert resolve_skills(["bar", "foo"], skills_tree) == [
            ("bar", _bar_path(skills_tree)),
            ("foo", _foo_path(skills_tree)),
        ]

    def test_unique_skill_resolves_when_another_name_clashes(
        self, skills_tree: Path
    ) -> None:
        assert resolve_skills(["foo"], skills_tree)[0][1] == _foo_path(skills_tree)

    def test_unknown_name_lists_available_skills(self, skills_tree: Path) -> None:
        with pytest.raises(SkillError, match="unknown skill 'nope'") as excinfo:
            resolve_skills(["nope"], skills_tree)
        assert str(excinfo.value) == (
            "unknown skill 'nope'. available skills: bar, clash, foo, loose"
        )

    def test_unknown_name_in_empty_catalog(self, tmp_path: Path) -> None:
        root = tmp_path / "SKILLS"
        root.mkdir()
        with pytest.raises(SkillError) as excinfo:
            resolve_skills(["nope"], root)
        assert str(excinfo.value) == "unknown skill 'nope'. no skills found"

    def test_conflict_lists_every_colliding_path(self, skills_tree: Path) -> None:
        with pytest.raises(SkillError) as excinfo:
            resolve_skills(["clash"], skills_tree)
        listed = "\n".join(f"  {path}" for path in _clash_paths(skills_tree))
        assert str(excinfo.value) == f"skill name 'clash' is not unique:\n{listed}"

    def test_two_way_conflict_is_still_a_conflict(self, tmp_path: Path) -> None:
        root = tmp_path / "SKILLS"
        left = root / "left" / "dup"
        right = root / "right" / "dup"
        left.mkdir(parents=True)
        right.mkdir(parents=True)
        (left / "SKILL.md").write_text("# left\n", encoding="utf-8")
        (right / "SKILL.md").write_text("# right\n", encoding="utf-8")
        with pytest.raises(SkillError) as excinfo:
            resolve_skills(["dup"], root)
        listed = "\n".join(
            f"  {path}"
            for path in sorted(
                [(left / "SKILL.md").resolve(), (right / "SKILL.md").resolve()],
                key=str,
            )
        )
        assert str(excinfo.value) == f"skill name 'dup' is not unique:\n{listed}"

    def test_conflict_paths_are_sorted_as_strings(self, tmp_path: Path) -> None:
        """Path comparison orders these opposite to str; the message uses str."""
        root = tmp_path / "SKILLS"
        (root / "x").mkdir(parents=True)
        (root / "x-y").mkdir(parents=True)
        (root / "x" / "clash.md").write_text("# x\n", encoding="utf-8")
        (root / "x-y" / "clash.md").write_text("# x-y\n", encoding="utf-8")
        hyphen = (root / "x-y" / "clash.md").resolve()
        slash = (root / "x" / "clash.md").resolve()
        with pytest.raises(SkillError) as excinfo:
            resolve_skills(["clash"], root)
        assert str(excinfo.value) == (
            f"skill name 'clash' is not unique:\n  {hyphen}\n  {slash}"
        )


class TestComposeSkillPrompt:
    def test_paths_come_before_the_user_text(self, skills_tree: Path) -> None:
        resolved = resolve_skills(["foo", "bar"], skills_tree)
        prompt = compose_skill_prompt(resolved, "do the work")
        foo = _foo_path(skills_tree)
        bar = _bar_path(skills_tree)
        extra = (skills_tree / "nested" / "foo" / "extra.md").resolve()
        assert extra not in {foo, bar}
        assert str(extra) not in prompt
        assert prompt == (
            f"{PROMPT_HEADER}\n\n- foo: {foo}\n- bar: {bar}\n\n---\n\ndo the work"
        )

    def test_single_skill_still_separates_path_from_prompt(
        self, skills_tree: Path
    ) -> None:
        resolved = resolve_skills(["loose"], skills_tree)
        path = _loose_path(skills_tree)
        assert compose_skill_prompt(resolved, "only this") == (
            f"{PROMPT_HEADER}\n\n- loose: {path}\n\n---\n\nonly this"
        )


class TestFormatSkillListing:
    def test_empty_catalog(self) -> None:
        assert format_skill_listing({}) == "no skills"

    def test_lists_each_file_sorted_by_name(self, skills_tree: Path) -> None:
        catalog = index_skills(skills_tree)
        listing = format_skill_listing(catalog)
        assert listing == _expected_skill_listing(skills_tree)
        extra = str((skills_tree / "nested" / "foo" / "extra.md").resolve())
        assert extra not in listing
        assert "README.md" not in listing

    def test_does_not_use_path_comparison_order(self, tmp_path: Path) -> None:
        """str sort puts x-y before x/...; Path sort is the opposite."""
        root = tmp_path / "SKILLS"
        (root / "x").mkdir(parents=True)
        (root / "x-y").mkdir(parents=True)
        (root / "x" / "clash.md").write_text("# x\n", encoding="utf-8")
        (root / "x-y" / "clash.md").write_text("# x-y\n", encoding="utf-8")
        hyphen = (root / "x-y" / "clash.md").resolve()
        slash = (root / "x" / "clash.md").resolve()
        listing = format_skill_listing(index_skills(root))
        assert listing == f"{'clash':20} {hyphen}\n{'clash':20} {slash}"

    def test_aligns_columns_for_long_names(self, tmp_path: Path) -> None:
        long_name = "a-very-long-skill-name-exceeding-twenty"
        foo_path = tmp_path / "foo.md"
        long_path = tmp_path / "long.md"
        catalog = {"foo": [foo_path], long_name: [long_path]}
        listing = format_skill_listing(catalog)
        path_col = len(long_name) + 1
        assert listing == (
            f"{long_name} {long_path}\n{'foo':{path_col - 1}} {foo_path}"
        )
        for line, path in zip(listing.splitlines(), (long_path, foo_path), strict=True):
            assert line.index(str(path)) == path_col


class TestListCommand:
    @pytest.fixture
    def orch(self, tmp_path: Path, skills_tree: Path) -> Orchestrator:
        return Orchestrator(
            runtime_paths(
                tmp_path,
                state_file=tmp_path / "s.json",
                skills_dir=skills_tree,
            )
        )

    def test_list_agents_matches_the_list_command(
        self,
        orch: Orchestrator,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(core, "_utcnow", lambda: "2026-08-25T00:00:00Z")
        orch.spawn("a", "echo")
        _cmd_list(orch, _options(["list"]))
        via_command = capsys.readouterr().out
        _cmd_list(orch, _options(["list", "agents"]))
        via_flag = capsys.readouterr().out
        assert via_flag == via_command
        assert via_flag == (
            f"registry: {orch.state_file}\n"
            "a                     backend=echo  model=-  effort=-  "
            "turns=0  created=2026-08-25T00:00:00Z  last=-        "
            "session=-\n"
        )

    def test_list_agents_aligns_columns_for_long_names(
        self, orch: Orchestrator, capsys: pytest.CaptureFixture[str]
    ) -> None:
        long_name = "gdw-58-reviewer-specification"
        orch.spawn("a", "echo")
        orch.spawn(long_name, "echo")
        _cmd_list(orch, _options(["list", "agents"]))
        header, *lines = capsys.readouterr().out.splitlines()
        assert header == f"registry: {orch.state_file}"
        assert len(lines) == 2
        backend_offsets = {line.index("backend=") for line in lines}
        session_offsets = {line.index("session=") for line in lines}
        assert len(backend_offsets) == 1
        assert len(session_offsets) == 1
        assert backend_offsets.pop() == len(long_name) + 2

    def test_list_agents_empty(
        self, orch: Orchestrator, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _cmd_list(orch, _options(["list", "agents"]))
        assert capsys.readouterr().out == f"registry: {orch.state_file}\nno agents\n"

    def test_list_skills_prints_catalog(
        self,
        orch: Orchestrator,
        skills_tree: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _cmd_list(orch, _options(["list", "skills"]))
        assert capsys.readouterr().out == _expected_skill_listing(skills_tree) + "\n"

    def test_list_skills_empty_catalog(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        empty = tmp_path / "SKILLS"
        empty.mkdir()
        orch = Orchestrator(
            runtime_paths(tmp_path, state_file=tmp_path / "s.json", skills_dir=empty)
        )
        _cmd_list(orch, _options(["list", "skills"]))
        assert capsys.readouterr().out == "no skills\n"

    def test_list_skills_missing_dir(
        self, tmp_path: Path, monkeypatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        missing = tmp_path / "absent"
        orch = Orchestrator(
            runtime_paths(tmp_path, state_file=tmp_path / "s.json", skills_dir=missing)
        )
        monkeypatch.setattr(core, "Orchestrator", lambda *_: orch)
        with pytest.raises(SystemExit, match="1"):
            main(["list", "skills"])
        captured = capsys.readouterr()
        assert captured.err == f"skills directory not found: {missing}\n"
        assert captured.out == ""

    def test_list_without_target_defaults_to_agents(
        self, orch: Orchestrator, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _cmd_list(orch, _options(["list"]))
        assert capsys.readouterr().out == f"registry: {orch.state_file}\nno agents\n"

    def test_unknown_list_target_is_argparse_exit_2(
        self, orch: Orchestrator, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit, match="2"):
            _options(["list", "bogus"])
        err = capsys.readouterr().err
        assert "usage: orchestrator" in err
        assert "bogus" in err

    def test_list_rejects_extra_args(
        self, orch: Orchestrator, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit, match="2"):
            _options(["list", "agents", "extra"])
        assert "usage: orchestrator" in capsys.readouterr().err


class TestTalkSkills:
    @pytest.fixture
    def orch(self, tmp_path: Path, skills_tree: Path) -> Orchestrator:
        orch = Orchestrator(
            runtime_paths(
                tmp_path,
                state_file=tmp_path / "s.json",
                skills_dir=skills_tree,
            )
        )
        orch.spawn("a", "echo")
        return orch

    def test_prints_talk_style_reply(
        self,
        orch: Orchestrator,
        skills_tree: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _cmd_talk(orch, _talk_options(["talk", "a", "--skill", "foo", "-p", "do it"]))
        out = capsys.readouterr().out
        assert "session=echo-sid" in out
        assert f"- foo: {_foo_path(skills_tree)}" in out
        assert out.strip().endswith("do it")
        extra = str((skills_tree / "nested" / "foo" / "extra.md").resolve())
        assert extra not in out

    def test_attaches_skills_in_cli_order(
        self,
        orch: Orchestrator,
        skills_tree: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _cmd_talk(
            orch,
            _talk_options(["talk", "a", "--skill", "bar, foo", "-p", "both"]),
        )
        out = capsys.readouterr().out
        assert out.index(f"- bar: {_bar_path(skills_tree)}") < out.index(
            f"- foo: {_foo_path(skills_tree)}"
        )

    def test_unknown_skill_exits_without_talking(
        self,
        orch: Orchestrator,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(core, "Orchestrator", lambda *_: orch)
        with pytest.raises(SystemExit, match="1"):
            main(["talk", "a", "--skill", "nope", "-p", "x"])
        captured = capsys.readouterr()
        assert "unknown skill 'nope'" in captured.err
        assert captured.out == ""

    def test_unknown_skill_creates_no_agent(
        self,
        orch: Orchestrator,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(core, "Orchestrator", lambda *_: orch)
        with pytest.raises(SystemExit, match="1"):
            main(["talk", "missing", "--skill", "nope", "-p", "x"])
        assert orch.list_agents() == ["a"]

    def test_conflict_exits_without_talking(
        self,
        orch: Orchestrator,
        skills_tree: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(core, "Orchestrator", lambda *_: orch)
        with pytest.raises(SystemExit, match="1"):
            main(["talk", "a", "--skill", "clash", "-p", "x"])
        captured = capsys.readouterr()
        assert "skill name 'clash' is not unique" in captured.err
        for path in _clash_paths(skills_tree):
            assert str(path) in captured.err
        assert captured.out == ""

    def test_unknown_agent_is_created_then_talked_to(
        self, orch: Orchestrator, monkeypatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(core, "DEFAULT_BACKEND", "echo")
        _cmd_talk(
            orch,
            _talk_options(["talk", "missing", "--skill", "foo", "-p", "x"]),
        )
        captured = capsys.readouterr()
        assert captured.err == "created agent 'missing' backend=echo\n"
        assert "session=echo-sid" in captured.out
        assert orch.list_agents() == ["a", "missing"]

    def test_config_flags_create_exact_agent_before_skill_turn(
        self, orch: Orchestrator, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _cmd_talk(
            orch,
            _talk_options(
                [
                    "talk",
                    "missing",
                    "-b",
                    "echo",
                    "--model",
                    "m",
                    "--reasoning-effort",
                    "high",
                    "-p",
                    "x",
                ]
            ),
        )
        assert capsys.readouterr().err == "created agent 'missing' backend=echo\n"
        agent = orch.agents["missing"]
        assert (
            agent.backend.name,
            agent.backend.model,
            agent.backend.reasoning_effort,
        ) == (
            "echo",
            "m",
            "high",
        )

    def test_matching_config_flags_reuse_silently(
        self, orch: Orchestrator, capsys: pytest.CaptureFixture[str]
    ) -> None:
        orch.spawn("configured", "echo", model="m", reasoning_effort="high")
        _cmd_talk(
            orch,
            _talk_options(
                [
                    "talk",
                    "configured",
                    "-b",
                    "echo",
                    "--model",
                    "m",
                    "--reasoning-effort",
                    "high",
                    "-p",
                    "x",
                ]
            ),
        )
        assert capsys.readouterr().err == ""

    def test_mismatched_config_flags_fail_before_skill_turn(
        self, orch: Orchestrator
    ) -> None:
        orch.spawn("configured", "echo", model="stored", reasoning_effort="high")
        with pytest.raises(
            core.OrchestratorError,
            match=r"agent 'configured' already uses backend/model/effort \('echo', 'stored', 'high'\); configured \('codex', None, None\)",
        ):
            _cmd_talk(
                orch,
                _talk_options(["talk", "configured", "-b", "codex", "-p", "x"]),
            )

    def test_omitting_model_still_asserts_on_skill_invocation(
        self, orch: Orchestrator
    ) -> None:
        orch.spawn("configured", "echo", model="stored")
        with pytest.raises(
            core.OrchestratorError,
            match=r"configured \('echo', None, None\)$",
        ):
            _cmd_talk(
                orch,
                _talk_options(["talk", "configured", "-b", "echo", "-p", "x"]),
            )

    def test_unknown_backend_is_argparse_exit_2(
        self, orch: Orchestrator, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit, match="2"):
            _talk_options(["talk", "missing", "-b", "unknown", "-p", "x"])
        assert "invalid choice" in capsys.readouterr().err

    def test_empty_prompt_exits_nonzero_like_talk(
        self, orch: Orchestrator, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit, match="2"):
            _talk_options(["talk", "a", "--skill", "foo", "-p", "   "])
        captured = capsys.readouterr()
        assert "usage: orchestrator talk " in captured.err
        assert "talk prompt must not be empty" in captured.err
        assert captured.out == ""

    def test_missing_prompt_flag_is_argparse_exit_2(
        self, orch: Orchestrator, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit, match="2"):
            _talk_options(["talk", "a", "--skill", "foo"])
        err = capsys.readouterr().err
        assert "usage: orchestrator" in err
        assert "talk requires exactly one prompt source" in err

    def test_missing_talk_name_is_argparse_exit_2(
        self, orch: Orchestrator, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit, match="2"):
            main(["talk", "--skill", "foo", "--prompt", "x"])
        err = capsys.readouterr().err
        assert "usage: orchestrator talk" in err
        assert "required: name" in err

    def test_missing_skill_flag_talks_without_attaching_any_skill(
        self, orch: Orchestrator, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _cmd_talk(orch, _talk_options(["talk", "a", "-p", "x"]))
        out = capsys.readouterr().out
        assert "session=echo-sid" in out
        assert out.strip().endswith("x")
        assert "- foo:" not in out

    def test_duplicate_skill_flag_value_exits(
        self,
        orch: Orchestrator,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(core, "Orchestrator", lambda *_: orch)
        with pytest.raises(SystemExit, match="1"):
            main(["talk", "a", "--skill", "foo,foo", "-p", "x"])
        assert "duplicate skill name 'foo'" in capsys.readouterr().err

    def test_backend_error_is_printed_not_a_traceback(
        self,
        orch: Orchestrator,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        class BoomBackend(AgentBackend):
            @property
            def name(self) -> str:
                return "boom"

            def run_turn(  # noqa: PLR0913 - test doubles mirror AgentBackend.run_turn public seam
                self,
                prompt: str,
                session_id: str | None,
                cwd: Path,
                timeout: int = DEFAULT_TURN_TIMEOUT,
                schema: OutputSchema | None = None,
                *,
                resume_as_fork: bool = False,
                stream: bool = False,
            ) -> TurnResult:
                raise ClaudeTurnError("claude output was not JSON")

        register_backend("boom", BoomBackend)
        orch.spawn("b", "boom")
        monkeypatch.setattr(core, "Orchestrator", lambda *_: orch)
        with pytest.raises(SystemExit, match="1"):
            main(["talk", "b", "--skill", "foo", "-p", "x"])
        captured = capsys.readouterr()
        assert captured.err == "claude output was not JSON\n"
        assert captured.out == ""

    def test_logs_attached_skill_names(
        self, orch: Orchestrator, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("INFO", logger="orchestrator"):
            _cmd_talk(
                orch,
                _talk_options(["talk", "a", "--skill", "foo,bar", "-p", "x"]),
            )
        assert "agent 'a': attaching skill(s) foo, bar" in [
            r.getMessage() for r in caplog.records
        ]


class TestMainSkillInvocation:
    @pytest.fixture
    def isolated(
        self, tmp_path: Path, skills_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> Orchestrator:
        orch = Orchestrator(
            runtime_paths(
                tmp_path,
                state_file=tmp_path / "s.json",
                skills_dir=skills_tree,
            )
        )
        orch.spawn("a", "echo")
        monkeypatch.setattr(core, "Orchestrator", lambda *_: orch)
        for name in ("orchestrator", "backends"):
            logging.getLogger(name).setLevel(logging.NOTSET)
        return orch

    def test_flag_path_sends_composed_prompt(
        self,
        isolated: Orchestrator,
        skills_tree: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        main(["talk", "a", "--skill", "foo", "--prompt", "do it"])
        out = capsys.readouterr().out
        assert "session=echo-sid" in out
        assert f"- foo: {_foo_path(skills_tree)}" in out
        assert "do it" in out

    def test_verbose_flag_is_peeled_before_skill_flags(
        self,
        isolated: Orchestrator,
        capsys: pytest.CaptureFixture[str],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level("DEBUG", logger="orchestrator"):
            main(["-v", "talk", "a", "--skill", "foo", "--prompt", "do it"])
        assert "session=echo-sid" in capsys.readouterr().out
        assert "cli: dispatching 'talk'" in [r.getMessage() for r in caplog.records]

    def test_prompt_token_equal_to_a_verbose_flag_is_kept(
        self,
        isolated: Orchestrator,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        main(["talk", "a", "--skill", "foo", "--prompt", "add -v please"])
        assert "add -v please" in capsys.readouterr().out
        assert logging.getLogger("orchestrator").getEffectiveLevel() > logging.DEBUG

    def test_talk_separator_form_is_unchanged(
        self,
        isolated: Orchestrator,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _cmd_talk(isolated, _talk_options(["talk", "a", "--", "hello", "there"]))
        assert "echo:hello there" in capsys.readouterr().out

    def test_talk_via_main_is_unchanged(
        self,
        isolated: Orchestrator,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        main(["talk", "a", "-p", "plain"])
        out = capsys.readouterr().out
        assert "echo:plain" in out
        assert PROMPT_HEADER not in out

    def test_list_agents_via_main(
        self, isolated: Orchestrator, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["list", "agents"])
        created = isolated.agents["a"].created_at
        assert capsys.readouterr().out == (
            f"registry: {isolated.state_file}\n"
            "a                     backend=echo  model=-  effort=-  "
            f"turns=0  created={created}  last=-        "
            "session=-\n"
        )

    def test_list_skills_via_main(
        self,
        isolated: Orchestrator,
        skills_tree: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        main(["list", "skills"])
        assert capsys.readouterr().out == _expected_skill_listing(skills_tree) + "\n"

    def test_removed_list_equals_form_is_rejected(
        self,
        isolated: Orchestrator,
        skills_tree: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        with pytest.raises(SystemExit, match="2"):
            main(["--list=skills"])
        assert capsys.readouterr().out == ""

    def test_list_command_is_unchanged(
        self, isolated: Orchestrator, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["list"])
        created = isolated.agents["a"].created_at
        assert capsys.readouterr().out == (
            f"registry: {isolated.state_file}\n"
            "a                     backend=echo  model=-  effort=-  "
            f"turns=0  created={created}  last=-        "
            "session=-\n"
        )

    def test_verbose_flag_is_peeled_before_list(
        self,
        isolated: Orchestrator,
        capsys: pytest.CaptureFixture[str],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level("DEBUG", logger="orchestrator"):
            main(["-v", "list", "agents"])
        created = isolated.agents["a"].created_at
        assert capsys.readouterr().out == (
            f"registry: {isolated.state_file}\n"
            "a                     backend=echo  model=-  effort=-  "
            f"turns=0  created={created}  last=-        "
            "session=-\n"
        )
        assert "cli: dispatching 'list'" in [r.getMessage() for r in caplog.records]

    def test_missing_skills_dir_exits(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        missing = tmp_path / "absent"
        orch = Orchestrator(
            runtime_paths(tmp_path, state_file=tmp_path / "s.json", skills_dir=missing)
        )
        orch.spawn("a", "echo")
        monkeypatch.setattr(core, "Orchestrator", lambda *_: orch)
        with pytest.raises(SystemExit, match="1"):
            main(["talk", "a", "--skill", "foo", "--prompt", "x"])
        captured = capsys.readouterr()
        assert captured.err == f"skills directory not found: {missing}\n"
        assert captured.out == ""
