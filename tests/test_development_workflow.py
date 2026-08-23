"""Behavior tests for the raw-issue-to-PR workflow example."""

from __future__ import annotations

import json
import logging
import os
import runpy
import subprocess
import sys
from collections import deque
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest
import yaml

from examples.gabriels_workflow import config as workflow_config
from examples.gabriels_workflow import development_workflow as gdw
from examples.gabriels_workflow import github_app_client as app_github
from examples.gabriels_workflow import setup as simple_setup
from examples.gabriels_workflow import simple_development_workflow as simple_entrypoint
from examples.gabriels_workflow.config import (
    REQUIRED_ROLES,
    RoleConfig,
    WorkflowConfig,
    load_config,
)
from orchestrator.schema import load_schema


@pytest.fixture(autouse=True)
def _isolate_workflow_logging():
    """Keep configure_logging() calls from leaking handlers into later tests."""

    logger = gdw.LOGGER
    handlers, level = list(logger.handlers), logger.level
    yield
    logger.handlers = handlers
    logger.setLevel(level)


def _completed(
    args: list[str], returncode: int = 0, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args, returncode, stdout=stdout, stderr=stderr)


class ScriptedRun:
    def __init__(self, replies: list[subprocess.CompletedProcess[str]]) -> None:
        self.replies = deque(replies)
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, args: list[str], **kwargs):
        self.calls.append((args, kwargs))
        reply = self.replies.popleft()
        stdout, stderr = reply.stdout, reply.stderr
        if kwargs.get("stderr") is subprocess.STDOUT:
            merged = stdout or ""
            if stderr:
                merged = f"{merged}\n{stderr}" if merged else stderr
            stdout, stderr = merged, ""
        return _completed(args, reply.returncode, stdout, stderr)


def _expansion(decision: str = "proceed", needs_another_round: bool = False) -> dict:
    return {
        "decision": decision,
        "needs_another_round": needs_another_round,
        "reason": "proposal converged" if not needs_another_round else "needs review",
        "summary": "proposal",
        "current_state": ["current"],
        "proposed_changes": ["change"],
        "out_of_scope": [],
        "risks": [],
        "open_questions": [],
    }


def _grill(verdict: str = "ready", needs_another_round: bool | None = None) -> dict:
    if needs_another_round is None:
        needs_another_round = verdict == "revise"
    return {
        "verdict": verdict,
        "needs_another_round": needs_another_round,
        "reason": "review converged" if not needs_another_round else "needs revision",
        "summary": "review",
        "questions": [] if verdict == "ready" else ["question"],
        "required_changes": [],
    }


def _specification() -> dict:
    return {
        "title": "Implement the thing\nwith detail",
        "problem_statement": "problem",
        "solution": "solution",
        "user_stories": ["story"],
        "implementation_decisions": ["decision"],
        "testing_decisions": ["test"],
        "acceptance_criteria": ["criterion"],
        "out_of_scope": [],
    }


def _implementation(status: str = "complete") -> dict:
    return {
        "status": status,
        "summary": "implemented",
        "files_changed": ["feature.py"],
        "tests_run": ["pytest"],
        "blockers": [] if status == "complete" else ["missing decision"],
    }


def _documentation(status: str = "complete") -> dict:
    return {
        "status": status,
        "summary": "documented",
        "files_changed": ["README.md"],
        "blockers": [] if status == "complete" else ["missing decision"],
    }


def _workflow_config(**overrides: object) -> WorkflowConfig:
    values: dict[str, object] = {
        "repository": "owner/project",
        "draft": False,
        "roles": {
            role: RoleConfig(
                backend="codex",
                model="gpt-test",
                reasoning_effort="high",
                github_app={"app_id": index, "private_key": f"key-{role}"},
            )
            for index, role in enumerate(sorted(REQUIRED_ROLES), start=1)
        },
    }
    values.update(overrides)
    return WorkflowConfig.model_validate(values)


def test_workflow_yaml_loads_and_normalizes_role_settings(tmp_path: Path) -> None:
    defaults = RoleConfig(
        backend="claude",
        model=None,
        reasoning_effort=None,
        github_app={"app_id": 9, "private_key": "key"},
    )
    assert defaults.model is None
    assert defaults.reasoning_effort is None

    path = tmp_path / "workflow.yaml"
    roles = "\n".join(
        f"  {role}:\n"
        "    backend: ' Codex '\n"
        "    model: ' test-model '\n"
        "    reasoning_effort: ' high '\n"
        "    github_app:\n"
        "      app_id: 1\n"
        "      private_key: test-private-key"
        for role in sorted(REQUIRED_ROLES)
    )
    path.write_text(
        f"repository: ' owner/project '\ndraft: false\nroles:\n{roles}\n",
        encoding="utf-8",
    )

    config = load_config(path)
    assert config.repository == "owner/project"
    assert config.draft is False
    assert config.roles["implementer"] == RoleConfig(
        backend="codex",
        model="test-model",
        reasoning_effort="high",
        github_app={"app_id": 1, "private_key": "test-private-key"},
    )


def test_workflow_yaml_loads_private_key_from_path(tmp_path: Path) -> None:
    key_path = tmp_path / "bot.pem"
    key_path.write_text("test-file-key\n", encoding="utf-8")
    path = tmp_path / "workflow.yaml"
    roles = "\n".join(
        f"  {role}:\n"
        "    backend: codex\n"
        "    github_app:\n"
        "      app_id: 1\n"
        f"      private_key: {key_path if role == 'implementer' else 'bot.pem'}"
        for role in sorted(REQUIRED_ROLES)
    )
    path.write_text(f"repository: owner/project\nroles:\n{roles}\n", encoding="utf-8")

    config = load_config(path)

    assert (
        config.roles["implementer"].github_app.private_key.get_secret_value()
        == "test-file-key\n"
    )


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"repository": "owner/project", "roles": []},
        {
            "repository": "owner/project",
            "roles": {
                "role": [],
                "app": {"github_app": []},
                "key": {"github_app": {"private_key": 1}},
                "inline": {"github_app": {"private_key": "inline-key"}},
            },
        },
    ],
)
def test_workflow_config_ignores_non_path_private_key_payloads(
    tmp_path: Path, payload: object
) -> None:
    path = tmp_path / "workflow.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(gdw.WorkflowError, match="invalid workflow config"):
        load_config(path)


def test_workflow_config_reports_missing_private_key_file(tmp_path: Path) -> None:
    path = tmp_path / "workflow.yaml"
    roles = "\n".join(
        f"  {role}:\n"
        "    backend: codex\n"
        "    github_app:\n"
        "      app_id: 1\n"
        "      private_key: missing.pem"
        for role in sorted(REQUIRED_ROLES)
    )
    path.write_text(f"repository: owner/project\nroles:\n{roles}\n", encoding="utf-8")

    with pytest.raises(gdw.WorkflowError, match="cannot read private key"):
        load_config(path)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"repository": "project"}, "OWNER/REPO"),
        (
            {
                "roles": {
                    "expander": {
                        "backend": "codex",
                        "github_app": {"app_id": 1, "private_key": "key"},
                    }
                }
            },
            "missing roles",
        ),
        (
            {
                "roles": {
                    **{
                        role: {
                            "backend": "codex",
                            "github_app": {"app_id": 1, "private_key": "key"},
                        }
                        for role in REQUIRED_ROLES
                    },
                    "unknown": {
                        "backend": "codex",
                        "github_app": {"app_id": 1, "private_key": "key"},
                    },
                }
            },
            "unknown roles",
        ),
        (
            {
                "roles": {
                    role: {
                        "backend": "other",
                        "github_app": {"app_id": 1, "private_key": "key"},
                    }
                    for role in REQUIRED_ROLES
                }
            },
            "claude, codex, grok, opencode",
        ),
        (
            {
                "roles": {
                    role: {
                        "backend": "codex",
                        "model": " ",
                        "github_app": {"app_id": 1, "private_key": "key"},
                    }
                    for role in REQUIRED_ROLES
                }
            },
            "must not be empty",
        ),
        (
            {
                "roles": {
                    role: {
                        "backend": "codex",
                        "github_app": {"app_id": 0, "private_key": "key"},
                    }
                    for role in REQUIRED_ROLES
                }
            },
            "greater than 0",
        ),
        (
            {
                "roles": {
                    role: {
                        "backend": "codex",
                        "github_app": {"app_id": 1, "private_key": " "},
                    }
                    for role in REQUIRED_ROLES
                }
            },
            "must not be empty",
        ),
        ({"clarification_rounds": 1}, "Extra inputs are not permitted"),
        ({"unexpected": True}, "Extra inputs are not permitted"),
    ],
)
def test_workflow_config_rejects_invalid_values(
    tmp_path: Path, overrides: dict[str, object], message: str
) -> None:
    values = _workflow_config().model_dump(mode="json")
    values.update(overrides)
    path = tmp_path / "workflow.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    with pytest.raises(gdw.WorkflowError, match=message):
        load_config(path)


def test_workflow_config_reports_missing_and_malformed_yaml(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    with pytest.raises(gdw.WorkflowError, match="cannot read workflow config"):
        load_config(missing)

    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("roles: [", encoding="utf-8")
    with pytest.raises(gdw.WorkflowError, match="invalid YAML"):
        load_config(malformed)


def _review(verdict: str = "approve", needs_another_round: bool | None = None) -> dict:
    if needs_another_round is None:
        needs_another_round = verdict == "changes_requested"
    return {
        "verdict": verdict,
        "needs_another_round": needs_another_round,
        "reason": "review converged" if not needs_another_round else "changes remain",
        "summary": "reviewed",
        "findings": []
        if verdict == "approve"
        else [
            {
                "severity": "medium",
                "title": "finding",
                "evidence": "evidence",
                "required_change": "fix it",
            }
        ],
    }


class FakeGitHub:
    def __init__(self, pr_url: str = "https://example.test/pr/1") -> None:
        self.comments: list[tuple] = []
        self.attributions: list[str] = []
        self.pr_calls: list[dict] = []
        self.pr_updates: list[dict] = []
        self.collected_markers: list[int] = []
        self.pr_url = pr_url
        self.markers: set[str] = set()

    def issue(self, number: int) -> dict:
        return {"number": number, "title": "Raw issue", "body": "Please build it"}

    def adopt_markers(self, markers: set[str]) -> None:
        self.markers |= markers

    def collect_markers(self, number: int) -> None:
        self.collected_markers.append(number)

    def comment_once(
        self,
        number: int,
        key: str,
        title: str,
        payload: object,
        *,
        attribution: str = "",
    ) -> None:
        if key in self.markers:
            return
        self.comments.append((number, key, title, payload))
        self.attributions.append(attribution)
        self.markers.add(key)

    def create_pr(
        self,
        *,
        base: str,
        branch: str,
        title: str,
        body: str,
        draft: bool,
    ) -> str:
        self.pr_calls.append(
            {
                "base": base,
                "branch": branch,
                "title": title,
                "body": body,
                "draft": draft,
            }
        )
        return self.pr_url

    def update_pr(self, number: int, *, body: str) -> None:
        self.pr_updates.append({"number": number, "body": body})


class FakeRepository:
    def __init__(self, ci: list[gdw.CommandResult] | None = None) -> None:
        self.ci = deque(ci or [gdw.CommandResult(0, "green")])
        self.commits: list[tuple[str, str]] = []
        self.start_commits: list[tuple[str, str]] = []
        self.pushes: list[str] = []

    def run_ci(self) -> gdw.CommandResult:
        return self.ci.popleft()

    def ensure_branch_ahead(self, message: str, base_sha: str) -> None:
        self.start_commits.append((message, base_sha))

    def commit(self, message: str, base_sha: str) -> None:
        self.commits.append((message, base_sha))

    def push(self, branch: str) -> None:
        self.pushes.append(branch)


class EntryWorkflow:
    def __init__(self, completed: str | None) -> None:
        self.completed = completed
        self.calls: list[object] = []

    def completed_url(self) -> str | None:
        self.calls.append("completed")
        return self.completed

    def load_issue(self) -> dict:
        self.calls.append("load")
        return {"issue": 1}

    def clarify(self, issue: dict) -> dict:
        self.calls.append(("clarify", issue))
        return {"proposal": 1}

    def specify(self, issue: dict, proposal: dict) -> dict:
        self.calls.append(("specify", issue, proposal))
        return {"specification": 1}

    def open_pull_request(self, specification: dict) -> str:
        self.calls.append(("open_pull_request", specification))
        return "https://example.test/pulls/new"

    def implement(self, specification: dict) -> None:
        self.calls.append(("implement", specification))

    def stabilize(self, specification: dict) -> dict:
        self.calls.append(("stabilize", specification))
        return {"ci": "green"}

    def review(self, specification: dict, ci: dict) -> dict:
        self.calls.append(("review", specification, ci))
        return {"reviews": "approved"}

    def publish(self, specification: dict, reviews: dict) -> str:
        self.calls.append(("publish", specification, reviews))
        return "https://example.test/pulls/new"


class FakeAgents:
    def __init__(
        self,
        replies: list[dict],
        roles: Mapping[str, gdw.RoleOptions] | None = None,
    ) -> None:
        self.replies = deque(replies)
        self.calls: list[dict] = []
        self.roles = dict(roles or {})
        self.default_options = gdw._StaticRoleOptions(
            "fake-backend", "fake-model", "fake-effort"
        )

    def options(self, role: str) -> gdw.RoleOptions:
        return self.roles.get(role, self.default_options)

    def ask(
        self,
        *,
        role: str,
        prompt_name: str,
        schema_name: str,
        values: Mapping[str, str],
        skills: Sequence[str] = (),
        timeout: int = gdw.DEFAULT_AGENT_TIMEOUT,
    ) -> dict:
        self.calls.append(
            {
                "role": role,
                "prompt_name": prompt_name,
                "schema_name": schema_name,
                "values": values,
                "skills": skills,
                "timeout": timeout,
            }
        )
        return self.replies.popleft()


def _workflow(
    tmp_path: Path,
    agent_replies: list[dict],
    overrides: dict | None = None,
) -> tuple[gdw.DevelopmentWorkflow, FakeGitHub, FakeRepository, FakeAgents]:
    settings = {
        "github": None,
        "repository": None,
        "agent_roles": None,
    }
    settings.update(overrides or {})
    store = gdw.ArtifactStore(tmp_path / "state")
    store.initialize(42, "feature", "base-sha")
    github = settings["github"] or FakeGitHub()
    repository = settings["repository"] or FakeRepository()
    agents = FakeAgents(agent_replies, settings["agent_roles"])
    workflow = gdw.DevelopmentWorkflow(
        gdw.WorkflowOptions(
            42,
            "master",
            "feature",
            True,
        ),
        gdw.WorkflowServices(store, github, repository, agents),
    )
    return workflow, github, repository, agents


def test_helpers_render_and_bound() -> None:
    assert gdw._bounded("short", 5) == "short"
    assert gdw._bounded("123456", 3) == "… output truncated …\n456"
    rendered = gdw._render_comment(
        "<!-- marker -->",
        "Title",
        {
            "summary": "Supports **Markdown**.",
            "open_questions": ["First?", "Second?"],
            "findings": [{"severity": "high", "required_change": "Fix the parser."}],
            "approved": True,
            "optional": None,
            "empty": [],
        },
    )
    assert rendered.startswith("<!-- marker -->\n## GDW — Title")
    assert "### Summary\n\nSupports **Markdown**." in rendered
    assert "- First?\n- Second?" in rendered
    assert "#### Item 1" in rendered
    assert "##### Required Change\n\nFix the parser." in rendered
    assert "### Approved\n\nYes" in rendered
    assert rendered.count("_None._") == 2
    assert "```json" not in rendered
    assert '{"' not in rendered
    assert gdw._pull_request_number("https://github.com/owner/repo/pull/12") == 12
    assert gdw._pull_request_number("https://example.test/pr/1/") == 1
    with pytest.raises(gdw.WorkflowError, match="without a number"):
        gdw._pull_request_number("https://example.test/pulls")
    assert gdw.CommandResult(0, "ok").succeeded is True
    assert gdw.CommandResult(1, "bad").succeeded is False
    assert gdw.CommandResult(1, "bad").as_json() == {
        "returncode": 1,
        "output": "bad",
        "gates": [],
    }


@pytest.mark.parametrize(
    ("options", "skills", "expected"),
    [
        (
            gdw._StaticRoleOptions("grok", "grok-4.6", "high"),
            ("code-simplification", "caveman"),
            "\n---\n\nbackend: `grok`  \nmodel: `grok-4.6`  \nreasoning_effort: `high`  \nskills: `code-simplification, caveman`",
        ),
        (
            gdw._StaticRoleOptions("claude", None, "high"),
            ("caveman",),
            "\n---\n\nbackend: `claude`  \nmodel: _unset_  \nreasoning_effort: `high`  \nskills: `caveman`",
        ),
        (
            gdw._StaticRoleOptions("codex"),
            (),
            "\n---\n\nbackend: `codex`  \nmodel: _unset_  \nreasoning_effort: _unset_  \nskills: _none_",
        ),
    ],
)
def test_attribution_renders_each_configured_field_exactly(
    options: gdw.RoleOptions, skills: Sequence[str], expected: str
) -> None:
    attribution = gdw._attribution(options, skills)

    assert attribution == expected
    assert "`_unset_`" not in attribution
    assert "`_none_`" not in attribution
    assert "default" not in attribution


def test_render_comment_only_appends_a_nonempty_attribution() -> None:
    base = "<!-- marker -->\n## GDW — Title\n\n### Answer\n\n1\n"
    footer = (
        "\n---\n\nbackend: `codex`  \nmodel: _unset_  \n"
        "reasoning_effort: _unset_  \nskills: _none_"
    )

    assert gdw._render_comment("<!-- marker -->", "Title", {"answer": 1}) == base
    assert gdw._render_comment("<!-- marker -->", "Title", {"answer": 1}, "") == base
    assert (
        gdw._render_comment("<!-- marker -->", "Title", {"answer": 1}, footer)
        == base + footer
    )


def test_artifact_store_round_trip_resume_and_errors(tmp_path: Path) -> None:
    store = gdw.ArtifactStore(tmp_path / "state")
    assert store.initialized is False
    with pytest.raises(gdw.WorkflowError, match="has not been initialized"):
        _ = store.metadata
    store.initialize(7, "feature", "abc")
    assert store.initialized is True
    assert store.metadata["base_sha"] == "abc"
    assert store.has("stage") is False
    store.save("stage", {"answer": 1})
    assert store.has("stage") is True
    assert store.load("stage") == {"answer": 1}
    store.record_development_pr(12, "https://example.test/pr/12")
    assert store.metadata["pr_number"] == 12
    assert store.metadata["development_pr_url"] == "https://example.test/pr/12"
    assert store.metadata["pr_url"] is None
    store.record_pr("https://example.test/pr")
    assert store.metadata["pr_url"] == "https://example.test/pr"
    store.initialize(7, "feature", "ignored-on-resume")
    assert store.metadata["base_sha"] == "abc"
    with pytest.raises(gdw.WorkflowError, match="belongs to"):
        store.initialize(8, "other", "abc")

    broken = gdw.ArtifactStore(tmp_path / "broken")
    broken.root.mkdir()
    broken.metadata_path.write_text("not-json", encoding="utf-8")
    with pytest.raises(gdw.WorkflowError, match="cannot read workflow state"):
        _ = broken.metadata
    broken.metadata_path.write_text("[]", encoding="utf-8")
    with pytest.raises(gdw.WorkflowError, match="not a JSON object"):
        _ = broken.metadata


def test_git_repository_prepare_ci_commit_push_and_failures(tmp_path: Path) -> None:
    runner = ScriptedRun(
        [
            _completed([], stdout="feature\n"),
            _completed([], stdout=""),
            _completed([], stdout="abc\n"),
            _completed([], stdout="lint\n\n"),
            _completed([], stdout="out", stderr="err"),
            _completed([], stdout="file\0"),
            _completed([], stdout="new.py\0"),
            _completed([]),
            _completed([]),
            _completed([], stdout="1\n"),
            _completed([]),
        ]
    )
    repository = gdw.GitRepository(tmp_path, runner)
    assert repository.prepare("master", False) == ("feature", "abc")
    assert repository.run_ci().as_json() == {
        "returncode": 0,
        "output": "out\nerr",
        "gates": [{"name": "lint", "status": "not run", "reason": "not run"}],
    }
    repository.commit("message", "abc")
    repository.push("feature")
    commands = [call[0] for call in runner.calls]
    assert ["git", "add", "--", "file", "new.py"] in commands
    assert ["git", "commit", "-m", "message"] in commands
    assert commands[-1] == ["git", "push", "--set-upstream", "origin", "feature"]
    assert ["make", "--no-print-directory", "ci-gates"] in commands
    ci_kwargs = runner.calls[4][1]
    assert ci_kwargs["timeout"] == gdw.DEFAULT_CI_TIMEOUT
    assert ci_kwargs["stdin"] == subprocess.DEVNULL
    assert ci_kwargs["stdout"] is subprocess.PIPE
    assert ci_kwargs["stderr"] is subprocess.STDOUT
    assert "capture_output" not in ci_kwargs

    detached = gdw.GitRepository(tmp_path, ScriptedRun([_completed([], stdout="\n")]))
    with pytest.raises(gdw.WorkflowError, match="named git branch"):
        detached.prepare("master", False)
    protected = gdw.GitRepository(
        tmp_path, ScriptedRun([_completed([], stdout="master\n")])
    )
    with pytest.raises(gdw.WorkflowError, match="protected base"):
        protected.prepare("master", False)
    dirty = gdw.GitRepository(
        tmp_path,
        ScriptedRun(
            [_completed([], stdout="feature\n"), _completed([], stdout="dirty")]
        ),
    )
    with pytest.raises(gdw.WorkflowError, match="clean worktree"):
        dirty.prepare("master", False)
    failed = gdw.GitRepository(
        tmp_path, ScriptedRun([_completed([], 1, stderr="boom")])
    )
    with pytest.raises(gdw.WorkflowError, match="git status failed"):
        failed._call("status")
    no_commits = gdw.GitRepository(
        tmp_path,
        ScriptedRun(
            [
                _completed([], stdout=""),
                _completed([], stdout=""),
                _completed([], stdout="0\n"),
            ]
        ),
    )
    with pytest.raises(gdw.WorkflowStopped, match="no commits"):
        no_commits.commit("message", "abc")


def test_git_repository_opens_an_empty_commit_only_when_the_branch_matches_base(
    tmp_path: Path,
) -> None:
    ahead = ScriptedRun([_completed([], stdout="2\n")])
    gdw.GitRepository(tmp_path, ahead).ensure_branch_ahead("Start work", "abc")
    empty = ScriptedRun([_completed([], stdout="0\n"), _completed([])])
    gdw.GitRepository(tmp_path, empty).ensure_branch_ahead("Start work", "abc")

    assert [call[0] for call in ahead.calls] == [
        ["git", "rev-list", "--count", "abc..HEAD"],
    ]
    assert [call[0] for call in empty.calls] == [
        ["git", "rev-list", "--count", "abc..HEAD"],
        ["git", "commit", "--allow-empty", "-m", "Start work"],
    ]


def test_ensure_issue_worktree_creates_branch_when_neither_exists(
    tmp_path: Path,
) -> None:
    path = tmp_path / "issue" / "worktree"
    runner = ScriptedRun(
        [
            _completed([], stdout=""),
            _completed(
                [], stdout="worktree /repo\nHEAD abc\nbranch refs/heads/master\n"
            ),
            _completed([], returncode=1),
            _completed([]),
        ]
    )
    gdw.GitRepository(tmp_path, runner).ensure_issue_worktree(
        "gdw/issue-9", "master", path
    )
    assert [call[0] for call in runner.calls] == [
        ["git", "worktree", "prune"],
        ["git", "worktree", "list", "--porcelain"],
        ["git", "rev-parse", "--verify", "--quiet", "refs/heads/gdw/issue-9"],
        ["git", "worktree", "add", "-b", "gdw/issue-9", str(path), "master"],
    ]


def test_ensure_issue_worktree_resumes_an_already_registered_worktree(
    tmp_path: Path,
) -> None:
    path = tmp_path / "issue" / "worktree"
    runner = ScriptedRun(
        [
            _completed([], stdout=""),
            _completed(
                [],
                stdout=f"worktree {path}\nHEAD abc\nbranch refs/heads/gdw/issue-9\n",
            ),
        ]
    )
    gdw.GitRepository(tmp_path, runner).ensure_issue_worktree(
        "gdw/issue-9", "master", path
    )
    assert [call[0] for call in runner.calls] == [
        ["git", "worktree", "prune"],
        ["git", "worktree", "list", "--porcelain"],
    ]


def test_ensure_issue_worktree_resumes_an_existing_branch_without_a_worktree(
    tmp_path: Path,
) -> None:
    path = tmp_path / "issue" / "worktree"
    runner = ScriptedRun(
        [
            _completed([], stdout=""),
            _completed([], stdout=""),
            _completed([], stdout="abc123\n"),
            _completed([]),
        ]
    )
    gdw.GitRepository(tmp_path, runner).ensure_issue_worktree(
        "gdw/issue-9", "master", path
    )
    assert [call[0] for call in runner.calls] == [
        ["git", "worktree", "prune"],
        ["git", "worktree", "list", "--porcelain"],
        ["git", "rev-parse", "--verify", "--quiet", "refs/heads/gdw/issue-9"],
        ["git", "worktree", "add", str(path), "gdw/issue-9"],
    ]


def test_ensure_issue_worktree_rejects_an_unregistered_existing_path(
    tmp_path: Path,
) -> None:
    runner = ScriptedRun(
        [
            _completed([], stdout=""),
            _completed([], stdout=""),
        ]
    )
    with pytest.raises(gdw.WorkflowError, match="not a registered git worktree"):
        gdw.GitRepository(tmp_path, runner).ensure_issue_worktree(
            "gdw/issue-9", "master", tmp_path
        )


def test_github_owns_markdown_comments_and_pull_requests(tmp_path: Path) -> None:
    bodies: list[str] = []

    def run(args: list[str], **_kwargs):
        if "--body-file" in args:
            body_path = Path(args[args.index("--body-file") + 1])
            bodies.append(body_path.read_text(encoding="utf-8"))
        if args[1:3] == ["repo", "view"]:
            stdout = json.dumps({"defaultBranchRef": {"name": "trunk"}})
        elif args[1:3] == ["issue", "view"]:
            stdout = json.dumps(
                {
                    "number": 9,
                    "comments": [
                        {"body": "<!-- gdw:9:existing -->\nalready here"},
                        "noise",
                    ],
                }
            )
        elif args[1:3] == ["pr", "create"]:
            stdout = "https://example.test/pr/9\n"
        else:
            stdout = "ok\n"
        return _completed(args, stdout=stdout)

    github = gdw.GitHub(
        tmp_path,
        "owner/repo",
        executable=Path("/usr/bin/gh"),
        run=run,
        environment={"GH_TOKEN": "secret"},
    )
    assert github.default_branch() == "trunk"
    assert github.issue(9)["number"] == 9
    github.comment_once(9, "existing", "Skipped", {})
    assert bodies == []
    github.comment_once(
        9,
        "new",
        "New",
        {"answer": 1},
        attribution="\n---\n\nbackend: `grok`\n",
    )
    github.comment_once(9, "new", "New", {"answer": 1})
    assert len(bodies) == 1
    assert "<!-- gdw:9:new -->" in bodies[0]
    assert "### Answer\n\n1" in bodies[0]
    assert "backend: `grok`" in bodies[0]
    assert "```json" not in bodies[0]
    assert (
        github.create_pr(
            base="trunk", branch="feature", title="Title", body="PR body", draft=True
        )
        == "https://example.test/pr/9"
    )
    assert bodies[-1] == "PR body"
    github.update_pr(9, body="Updated body")
    assert bodies[-1] == "Updated body"

    nondraft_bodies: list[str] = []

    def nondraft_run(args: list[str], **_kwargs):
        assert "--draft" not in args
        path = Path(args[args.index("--body-file") + 1])
        nondraft_bodies.append(path.read_text(encoding="utf-8"))
        return _completed(args, stdout="url")

    nondraft = gdw.GitHub(tmp_path, executable=Path("/usr/bin/gh"), run=nondraft_run)
    assert (
        nondraft.create_pr(
            base="main", branch="feature", title="T", body="B", draft=False
        )
        == "url"
    )
    assert nondraft_bodies == ["B"]


def test_github_reports_missing_invalid_and_failed_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gdw.shutil, "which", lambda _name: None)
    with pytest.raises(gdw.WorkflowError, match="not installed"):
        gdw.GitHub(tmp_path)

    invalid_branch = gdw.GitHub(
        tmp_path,
        executable=Path("/gh"),
        run=ScriptedRun([_completed([], stdout='{"defaultBranchRef": null}')]),
    )
    with pytest.raises(gdw.WorkflowError, match="default branch"):
        invalid_branch.default_branch()
    odd_comments = gdw.GitHub(
        tmp_path,
        executable=Path("/gh"),
        run=ScriptedRun([_completed([], stdout='{"comments": {}}')]),
    )
    assert odd_comments.issue(1) == {"comments": {}}
    odd_markers = gdw.GitHub(
        tmp_path,
        executable=Path("/gh"),
        run=ScriptedRun([_completed([], stdout='{"comments": {}}')]),
    )
    odd_markers.markers = {"keep"}
    odd_markers.collect_markers(8)
    assert odd_markers.markers == {"keep"}
    pr_markers = gdw.GitHub(
        tmp_path,
        executable=Path("/gh"),
        run=ScriptedRun(
            [
                _completed(
                    [],
                    stdout=json.dumps(
                        {
                            "comments": [
                                {"body": "<!-- gdw:1:implementation -->\nimpl"},
                                "noise",
                                {"body": 12},
                            ]
                        }
                    ),
                )
            ]
        ),
    )
    pr_markers.markers = {"<!-- gdw:42:specification -->"}
    pr_markers.collect_markers(1)
    assert pr_markers.markers == {
        "<!-- gdw:42:specification -->",
        "<!-- gdw:1:implementation -->",
    }
    invalid_json = gdw.GitHub(
        tmp_path,
        executable=Path("/gh"),
        run=ScriptedRun([_completed([], stdout="nope")]),
    )
    with pytest.raises(gdw.WorkflowError, match="invalid JSON"):
        invalid_json.issue(1)
    nonobject = gdw.GitHub(
        tmp_path,
        executable=Path("/gh"),
        run=ScriptedRun([_completed([], stdout="[]")]),
    )
    with pytest.raises(gdw.WorkflowError, match="not an object"):
        nonobject.issue(1)
    failed = gdw.GitHub(
        tmp_path,
        executable=Path("/gh"),
        run=ScriptedRun([_completed([], 1, stderr="denied")]),
    )
    with pytest.raises(gdw.WorkflowError, match="denied"):
        failed.issue(1)


def test_github_app_client_owns_app_auth_comments_and_pull_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_at = datetime(2026, 8, 20, tzinfo=UTC)
    comments = [
        SimpleNamespace(
            user=SimpleNamespace(login="alice"),
            body="hello",
            created_at=created_at,
            html_url="https://example.test/comments/1",
        ),
        SimpleNamespace(
            user=None,
            body="<!-- gdw:9:existing -->\nalready posted",
            created_at=created_at,
            html_url="https://example.test/comments/2",
        ),
    ]
    issue = SimpleNamespace(
        number=9,
        title="Issue",
        body=None,
        html_url="https://example.test/issues/9",
        labels=[SimpleNamespace(name="bug")],
        state="open",
        get_comments=lambda: comments,
        create_comment=MagicMock(),
    )
    pull = SimpleNamespace(html_url="https://example.test/pulls/9")
    repository = SimpleNamespace(
        default_branch="trunk",
        get_issue=MagicMock(return_value=issue),
        create_pull=MagicMock(return_value=pull),
    )
    github_session = SimpleNamespace(close=MagicMock())
    integration_session = SimpleNamespace(close=MagicMock())
    client = app_github.GitHubAppClient(
        cast(app_github.Repository, repository),
        cast(app_github.Github, github_session),
        cast(app_github.GithubIntegration, integration_session),
    )

    assert client.default_branch == "trunk"
    payload = client.issue(9)
    assert payload["body"] == ""
    assert payload["comments"][0]["author"] == "alice"
    assert payload["comments"][1]["author"] is None
    assert payload["labels"] == ["bug"]

    client.comment_once(9, "existing", "Skipped", {})
    issue.create_comment.assert_not_called()
    client.comment_once(
        9,
        "new",
        "New",
        {"answer": 1},
        attribution="\n---\n\nbackend: `grok`\n",
    )
    client.comment_once(9, "new", "New", {"answer": 1})
    posted = issue.create_comment.call_args.args[0]
    assert "<!-- gdw:9:new -->" in posted
    assert "### Answer\n\n1" in posted
    assert "backend: `grok`" in posted
    assert issue.create_comment.call_count == 1

    assert (
        client.create_pr(
            base="trunk", branch="feature", title="Title", body="Body", draft=True
        )
        == pull.html_url
    )
    repository.create_pull.assert_called_once_with(
        base="trunk", head="feature", title="Title", body="Body", draft=True
    )
    client.close()
    github_session.close.assert_called_once_with()
    integration_session.close.assert_called_once_with()
    app_github.GitHubAppClient(cast(app_github.Repository, repository)).close()

    app_auth = MagicMock()
    app_auth.get_installation_auth.return_value = "installation-auth"
    app_auth_factory = MagicMock(return_value=app_auth)
    integration = MagicMock()
    integration.get_repo_installation.return_value.id = 77
    integration_factory = MagicMock(return_value=integration)
    github = MagicMock()
    github.get_repo.return_value = repository
    github_factory = MagicMock(return_value=github)
    monkeypatch.setattr(app_github.Auth, "AppAuth", app_auth_factory)
    monkeypatch.setattr(app_github, "GithubIntegration", integration_factory)
    monkeypatch.setattr(app_github, "Github", github_factory)

    connected = app_github.GitHubAppClient.connect(12, "private-key", "owner/repo")
    assert connected.repository is repository
    app_auth_factory.assert_called_once_with(12, "private-key")
    integration.get_repo_installation.assert_called_once_with("owner", "repo")
    app_auth.get_installation_auth.assert_called_once_with(77)
    github_factory.assert_called_once_with(auth="installation-auth", per_page=100)


def test_github_app_client_collects_pr_markers_and_updates_the_body() -> None:
    issue = SimpleNamespace(
        get_comments=lambda: [
            SimpleNamespace(body="<!-- gdw:1:implementation -->\nalready on the PR"),
            SimpleNamespace(body=None),
        ]
    )
    pull = SimpleNamespace(edit=MagicMock())
    repository = SimpleNamespace(
        get_issue=MagicMock(return_value=issue),
        get_pull=MagicMock(return_value=pull),
    )
    client = app_github.GitHubAppClient(cast(app_github.Repository, repository))

    client.collect_markers(1)
    client.update_pr(9, body="Updated")

    assert "<!-- gdw:1:implementation -->" in client.markers
    repository.get_pull.assert_called_once_with(9)
    pull.edit.assert_called_once_with(body="Updated")


def test_agent_gateway_validates_prompt_and_removes_github_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[list[str], dict]] = []

    def fake_run(args: list[str], **kwargs):
        calls.append((args, kwargs))
        environment = kwargs["env"]
        assert [environment.get(key) for key in gdw.GITHUB_TOKEN_NAMES] == [
            None,
            None,
            None,
            None,
            None,
            None,
        ]
        assert environment["GH_CONFIG_DIR"].endswith("empty-gh-config")
        blocker = Path(environment["PATH"].split(os.pathsep)[0]) / "gh"
        assert "owned by the GDW driver" in blocker.read_text(encoding="utf-8")
        return _completed(
            args,
            stdout=f"[gdw-3-expander session=s1]\n{json.dumps(_expansion())}\n",
        )

    monkeypatch.setenv("GH_TOKEN", "secret")
    original_path = os.environ["PATH"]
    root = Path(gdw.__file__).parent
    gateway = gdw.AgentGateway(
        roles={
            "expander": SimpleNamespace(
                backend="codex", model=None, reasoning_effort=None
            )
        },
        issue=3,
        state_file=tmp_path / "agents.json",
        example_root=root,
        workdir=tmp_path,
        run=fake_run,
    )
    result = gateway.ask(
        role="expander",
        prompt_name="expand",
        schema_name="expansion",
        values={"ISSUE_CONTEXT_JSON": "{}"},
        timeout=17,
    )
    assert result == _expansion()
    assert calls[0][0][:10] == [
        "orchestrator",
        "talk",
        "gdw-3-expander",
        "--backend",
        "codex",
        "--schema",
        str(root / "validations" / "expansion.json"),
        "--timeout",
        "17",
        "--prompt",
    ]
    assert len(calls) == 1
    assert calls[0][1]["env"]["AGENTS_ARMY_STATE_FILE"] == str(tmp_path / "agents.json")
    assert calls[0][1]["timeout"] == 22
    assert calls[0][1]["cwd"] == str(tmp_path)
    assert calls[0][1]["env"]["AGENTS_ARMY_HOME"] == str(tmp_path)
    assert os.environ["GH_TOKEN"] == "secret"
    assert os.environ["PATH"] == original_path


def test_agent_turns_run_in_the_worktree_not_the_directory_the_driver_started_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Agents develop in the tree git and CI use, whatever the process's cwd.

    Regression: the gateway used to hand the orchestrator `Path.cwd()` and set
    no AGENTS_ARMY_HOME, so every agent edited the checkout the driver was
    started from while `make ci`, `git add` and `commit` ran in the issue
    worktree. Nothing failed until `commit` found an unchanged tree.
    """
    driver_cwd = tmp_path / "checkout"
    worktree = tmp_path / "checkout" / ".git" / "gdw" / "issue-3" / "worktree"
    worktree.mkdir(parents=True)
    monkeypatch.chdir(driver_cwd)
    calls: list[dict] = []

    def fake_run(args: list[str], **kwargs):
        calls.append(kwargs)
        return _completed(
            args,
            stdout=f"[gdw-3-expander session=s1]\n{json.dumps(_expansion())}\n",
        )

    gateway = gdw.AgentGateway(
        roles={
            "expander": SimpleNamespace(
                backend="codex", model=None, reasoning_effort=None
            )
        },
        issue=3,
        state_file=worktree.parent / "agents.json",
        example_root=Path(gdw.__file__).parent,
        workdir=worktree,
        run=fake_run,
    )
    gateway.ask(
        role="expander",
        prompt_name="expand",
        schema_name="expansion",
        values={"ISSUE_CONTEXT_JSON": "{}"},
    )

    assert calls[0]["cwd"] == str(worktree)
    assert calls[0]["env"]["AGENTS_ARMY_HOME"] == str(worktree)
    assert calls[0]["cwd"] != str(driver_cwd)
    assert calls[0]["env"]["AGENTS_ARMY_HOME"] != str(driver_cwd)


def test_agent_gateway_rejects_bad_prompts_backend_and_reply(tmp_path: Path) -> None:
    example_root = tmp_path / "example"
    (example_root / "prompts").mkdir(parents=True)
    (example_root / "validations").mkdir()
    schema = Path(gdw.__file__).parent / "validations" / "expansion.json"
    (example_root / "validations" / "expansion.json").write_text(
        schema.read_text(encoding="utf-8"), encoding="utf-8"
    )

    replies = deque(
        [
            _completed([], returncode=1, stderr="already uses backend/model/effort"),
            _completed([], stdout="[gdw-1-role session=s1]\nnull\n"),
        ]
    )

    def fake_run(args: list[str], **_kwargs):
        reply = replies.popleft()
        return _completed(args, reply.returncode, reply.stdout, reply.stderr)

    gateway = gdw.AgentGateway(
        roles={
            "role": SimpleNamespace(backend="codex", model=None, reasoning_effort=None)
        },
        issue=1,
        state_file=tmp_path / "state.json",
        example_root=example_root,
        workdir=tmp_path,
        run=fake_run,
    )
    with pytest.raises(gdw.WorkflowError, match="cannot read prompt"):
        gateway.ask(
            role="role",
            prompt_name="missing",
            schema_name="expansion",
            values={},
        )
    (example_root / "prompts" / "bad.md").write_text("{{MISSING}}", encoding="utf-8")
    with pytest.raises(gdw.WorkflowError, match="unresolved placeholders"):
        gateway.ask(role="role", prompt_name="bad", schema_name="expansion", values={})
    (example_root / "prompts" / "ok.md").write_text("ok", encoding="utf-8")
    with pytest.raises(gdw.WorkflowError, match="already uses backend/model/effort"):
        gateway.ask(role="role", prompt_name="ok", schema_name="expansion", values={})
    gateway.roles["role"] = SimpleNamespace(
        backend="claude", model=None, reasoning_effort=None
    )
    with pytest.raises(gdw.WorkflowError, match="no structured response"):
        gateway.ask(role="role", prompt_name="ok", schema_name="expansion", values={})


def test_agent_gateway_fills_prompts_with_brace_bearing_values(tmp_path: Path) -> None:
    """Braces in a substituted value are text, not an unfilled placeholder."""

    example_root = tmp_path / "example"
    (example_root / "prompts").mkdir(parents=True)
    (example_root / "validations").mkdir()
    schema = Path(gdw.__file__).parent / "validations" / "expansion.json"
    (example_root / "validations" / "expansion.json").write_text(
        schema.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (example_root / "prompts" / "grill.md").write_text(
        "review:\n{{EXPANSION_JSON}}\ncomments:\n{{LATEST_COMMENTS_JSON}}\n",
        encoding="utf-8",
    )
    # What an agent actually wrote back on issue #28: a dict comprehension
    # closing on `}}`, and prose naming another placeholder.
    expansion = "{role: f'{role}-x' for role in ROLES}} and {{LATEST_COMMENTS_JSON}}"

    sent: list[str] = []

    def fake_run(args: list[str], **_kwargs):
        sent.append(args[-1])
        return _completed(args, stdout='[gdw-28-role session=s1]\n{"ok": true}\n')

    gateway = gdw.AgentGateway(
        roles={
            "role": SimpleNamespace(backend="codex", model=None, reasoning_effort=None)
        },
        issue=28,
        state_file=tmp_path / "state.json",
        example_root=example_root,
        workdir=tmp_path,
        run=fake_run,
    )

    reply = gateway.ask(
        role="role",
        prompt_name="grill",
        schema_name="expansion",
        values={"EXPANSION_JSON": expansion, "LATEST_COMMENTS_JSON": "[]"},
    )

    assert reply == {"ok": True}
    assert sent == [f"review:\n{expansion}\ncomments:\n[]\n"]


def test_agent_gateway_names_the_placeholder_no_value_was_given_for(
    tmp_path: Path,
) -> None:
    example_root = tmp_path / "example"
    (example_root / "prompts").mkdir(parents=True)
    (example_root / "validations").mkdir()
    (example_root / "prompts" / "grill.md").write_text(
        "{{EXPANSION_JSON}} {{GRILL_JSON}} {{EXPANSION_JSON}}", encoding="utf-8"
    )

    gateway = gdw.AgentGateway(
        roles={
            "role": SimpleNamespace(backend="codex", model=None, reasoning_effort=None)
        },
        issue=28,
        state_file=tmp_path / "state.json",
        example_root=example_root,
        workdir=tmp_path,
        run=lambda args, **_kwargs: _completed(args),
    )

    with pytest.raises(
        gdw.WorkflowError, match=r"unresolved placeholders: EXPANSION_JSON, GRILL_JSON$"
    ):
        gateway.ask(
            role="role",
            prompt_name="grill",
            schema_name="expansion",
            values={},
        )


@pytest.mark.parametrize(
    ("turn_stdout", "message"),
    [
        ("header only", "no structured response"),
        ("[agent session=s1]\nnot-json\n", "invalid structured JSON"),
    ],
)
def test_agent_gateway_rejects_malformed_cli_output(
    tmp_path: Path, turn_stdout: str, message: str
) -> None:
    replies = deque(
        [
            _completed([], stdout=turn_stdout),
        ]
    )

    def fake_run(args: list[str], **_kwargs):
        reply = replies.popleft()
        return _completed(args, reply.returncode, reply.stdout, reply.stderr)

    gateway = gdw.AgentGateway(
        roles={
            "expander": SimpleNamespace(
                backend="codex", model=None, reasoning_effort=None
            )
        },
        issue=1,
        state_file=tmp_path / "state.json",
        example_root=Path(gdw.__file__).parent,
        workdir=tmp_path,
        run=fake_run,
    )
    with pytest.raises(gdw.WorkflowError, match=message):
        gateway.ask(
            role="expander",
            prompt_name="expand",
            schema_name="expansion",
            values={"ISSUE_CONTEXT_JSON": "{}"},
        )


@pytest.mark.parametrize(
    "failure",
    [OSError("missing executable"), subprocess.TimeoutExpired("orchestrator", 3)],
)
def test_agent_gateway_reports_cli_launch_failures(
    tmp_path: Path, failure: BaseException
) -> None:
    def failing_run(_args: list[str], **_kwargs):
        raise failure

    gateway = gdw.AgentGateway(
        roles={},
        issue=1,
        state_file=tmp_path / "state.json",
        example_root=Path(gdw.__file__).parent,
        workdir=tmp_path,
        run=failing_run,
    )
    with pytest.raises(gdw.WorkflowError, match="orchestrator CLI failed"):
        gateway._run_cli(["list"], {}, timeout=3)


def test_agent_gateway_reports_cli_exit_without_stderr(tmp_path: Path) -> None:
    gateway = gdw.AgentGateway(
        roles={},
        issue=1,
        state_file=tmp_path / "state.json",
        example_root=Path(gdw.__file__).parent,
        workdir=tmp_path,
        run=lambda args, **_kwargs: _completed(args, returncode=9),
    )
    with pytest.raises(gdw.WorkflowError, match="exited 9"):
        gateway._run_cli(["list"], {}, timeout=3)


def test_agent_gateway_uses_each_roles_backend_model_and_effort(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    configured = RoleConfig(
        backend="grok",
        model="grok-code-test",
        reasoning_effort="xhigh",
        github_app={"app_id": 1, "private_key": "key"},
    )

    def fake_run(args: list[str], **_kwargs):
        calls.append(args)
        return _completed(
            args,
            stdout=f"[gdw-5-expander session=s1]\n{json.dumps(_expansion())}\n",
        )

    gateway = gdw.AgentGateway(
        roles={"expander": configured},
        issue=5,
        state_file=tmp_path / "agents.json",
        example_root=Path(gdw.__file__).parent,
        workdir=tmp_path,
        run=fake_run,
    )
    assert gateway.options("expander") is configured
    with pytest.raises(gdw.WorkflowError, match="role 'missing' is not configured"):
        gateway.options("missing")
    gateway.ask(
        role="expander",
        prompt_name="expand",
        schema_name="expansion",
        values={"ISSUE_CONTEXT_JSON": "{}"},
    )
    expected_prompt = gateway._prompt("expand", {"ISSUE_CONTEXT_JSON": "{}"})
    assert calls[0] == [
        "orchestrator",
        "talk",
        "gdw-5-expander",
        "--backend",
        "grok",
        "--model",
        "grok-code-test",
        "--reasoning-effort",
        "xhigh",
        "--schema",
        str(Path(gdw.__file__).parent / "validations" / "expansion.json"),
        "--timeout",
        str(gdw.DEFAULT_AGENT_TIMEOUT),
        "--prompt",
        expected_prompt,
    ]
    with pytest.raises(gdw.WorkflowError, match="role 'missing' is not configured"):
        gateway.ask(
            role="missing",
            prompt_name="expand",
            schema_name="expansion",
            values={"ISSUE_CONTEXT_JSON": "{}"},
        )


def test_agent_gateway_attaches_skills_only_when_given(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs):
        calls.append(args)
        return _completed(
            args,
            stdout=f"[gdw-7-implementer session=s1]\n{json.dumps(_implementation())}\n",
        )

    gateway = gdw.AgentGateway(
        roles={
            "implementer": SimpleNamespace(
                backend="codex", model=None, reasoning_effort=None
            )
        },
        issue=7,
        state_file=tmp_path / "agents.json",
        example_root=Path(gdw.__file__).parent,
        workdir=tmp_path,
        run=fake_run,
    )
    gateway.ask(
        role="implementer",
        prompt_name="implement",
        schema_name="implementation",
        values={"SPECIFICATION_JSON": "{}"},
        skills=("code-simplification",),
    )
    assert "--skill" in calls[0]
    assert calls[0][calls[0].index("--skill") + 1] == "code-simplification"

    gateway.ask(
        role="implementer",
        prompt_name="implement",
        schema_name="implementation",
        values={"SPECIFICATION_JSON": "{}"},
    )
    assert "--skill" not in calls[1]


def test_all_workflow_schemas_are_strict_and_prompts_resolve(tmp_path: Path) -> None:
    example_root = Path(gdw.__file__).parent
    schema_names = (
        "expansion",
        "grill",
        "specification",
        "implementation",
        "documentation",
        "review",
    )
    for name in schema_names:
        schema = load_schema(example_root / "validations" / f"{name}.json")
        assert schema.path.is_absolute()

    gateway = gdw.AgentGateway(
        roles={},
        issue=1,
        state_file=tmp_path / "agents.json",
        example_root=example_root,
        workdir=tmp_path,
    )
    values = {
        "ISSUE_CONTEXT_JSON": "{}",
        "LATEST_COMMENTS_JSON": "[]",
        "EXPANSION_JSON": "{}",
        "GRILL_JSON": "{}",
        "SPECIFICATION_JSON": "{}",
        "FAILURE_EVIDENCE": "{}",
        "CI_SUMMARY": "{}",
    }
    prompt_names = (
        "expand",
        "grill",
        "revise",
        "specify",
        "implement",
        "repair",
        "document",
        "review-specification",
        "review-quality",
    )
    for name in prompt_names:
        prompt = gateway._prompt(name, values)
        assert "{{" not in prompt
        assert "external" in prompt
        assert "services" in prompt
        assert "use `gh`" not in prompt


def test_workflow_happy_path_and_completed_resume(tmp_path: Path) -> None:
    replies = [
        _expansion(),
        _grill(),
        _specification(),
        _implementation(),
        _documentation(),
        _review(),
        _review(),
    ]
    workflow, github, repository, agents = _workflow(tmp_path, replies)
    assert workflow.run() == "https://example.test/pr/1"
    assert repository.start_commits == [
        ("Start work on #42: Implement the thing with detail", "base-sha")
    ]
    assert repository.commits == [
        ("Implement #42: Implement the thing with detail", "base-sha")
    ]
    assert repository.pushes == ["feature", "feature"]
    assert github.pr_calls[0]["draft"] is True
    opening = github.pr_calls[0]["body"]
    assert "Closes #42" in opening
    assert "### Problem Statement\n\nproblem" in opening
    assert "review comments follow on this pull request" in opening
    assert github.pr_updates[0]["number"] == 1
    pr_body = github.pr_updates[0]["body"]
    assert "Closes #42" in pr_body
    assert "### Problem Statement\n\nproblem" in pr_body
    assert "### Specification\n\n#### Verdict\n\napprove" in pr_body
    assert "```json" not in pr_body
    assert len(agents.calls) == 7
    assert workflow.run() == "https://example.test/pr/1"
    assert len(agents.calls) == 7


def test_each_stage_comments_as_its_configured_github_app(tmp_path: Path) -> None:
    replies = [
        _expansion(),
        _grill(),
        _specification(),
        _implementation(),
        _documentation(),
        _review(),
        _review(),
    ]
    roles = {
        role: gdw._StaticRoleOptions(
            f"backend-{role}", f"model-{role}", f"effort-{role}"
        )
        for role in REQUIRED_ROLES
    }
    workflow, driver_github, _repository, _agents = _workflow(
        tmp_path, replies, {"agent_roles": roles}
    )
    role_github = {role: FakeGitHub() for role in REQUIRED_ROLES}
    workflow.role_github = cast(dict[str, gdw.GitHubService], role_github)

    workflow.run()

    expected_keys = {
        "expander": "expansion-1",
        "griller": "grill-1",
        "specifier": "specification",
        "implementer": "implementation",
        "documenter": "documentation",
        "reviewer-specification": "review-1-specification",
        "reviewer-quality": "review-1-quality",
    }
    assert {
        role: client.comments[0][1] for role, client in role_github.items()
    } == expected_keys
    assert all(len(client.comments) == 1 for client in role_github.values())
    role_skills = {
        "expander": (),
        "griller": (),
        "specifier": (),
        "implementer": ("code-simplification", "caveman"),
        "documenter": ("caveman",),
        "reviewer-specification": (),
        "reviewer-quality": ("code-review-and-quality", "code-simplification"),
    }
    assert {role: client.attributions[0] for role, client in role_github.items()} == {
        role: gdw._attribution(options, role_skills[role])
        for role, options in roles.items()
    }
    issue_roles = gdw.ISSUE_COMMENT_ROLES
    for role, client in role_github.items():
        number = client.comments[0][0]
        if role in issue_roles:
            assert number == 42
        else:
            assert number == 1
    assert {comment[1] for comment in driver_github.comments} == {"ci-implementation-1"}
    assert driver_github.comments[0][0] == 1
    assert driver_github.attributions == [""]


def test_specification_is_the_last_issue_comment_and_later_bots_post_on_the_pr(
    tmp_path: Path,
) -> None:
    replies = [
        _expansion(),
        _grill(),
        _specification(),
        _implementation(),
        _documentation(),
        _review(),
        _review(),
    ]
    workflow, github, _repository, _agents = _workflow(tmp_path, replies)

    workflow.run()

    issue_keys = [
        key for number, key, _title, _payload in github.comments if number == 42
    ]
    pr_keys = [key for number, key, _title, _payload in github.comments if number == 1]
    assert issue_keys == ["expansion-1", "grill-1", "specification"]
    assert pr_keys == [
        "implementation",
        "documentation",
        "ci-implementation-1",
        "review-1-specification",
        "review-1-quality",
    ]


def test_open_pull_request_reuses_a_recorded_development_pr(tmp_path: Path) -> None:
    workflow, github, repository, _agents = _workflow(tmp_path, [])
    workflow.store.record_development_pr(7, "https://example.test/pr/7")

    assert workflow.open_pull_request(_specification()) == "https://example.test/pr/7"
    assert github.pr_calls == []
    assert repository.pushes == []


def test_open_pull_request_rejects_a_url_without_a_number(tmp_path: Path) -> None:
    workflow, *_ = _workflow(
        tmp_path, [], {"github": FakeGitHub("https://example.test/pulls")}
    )
    with pytest.raises(gdw.WorkflowError, match="without a number"):
        workflow.open_pull_request(_specification())


def test_resumed_issue_load_collects_pull_request_comment_markers(
    tmp_path: Path,
) -> None:
    workflow, github, _repository, _agents = _workflow(tmp_path, [])
    workflow.store.record_development_pr(1, "https://example.test/pr/1")

    workflow.load_issue()

    assert github.collected_markers == [1]


def test_comment_number_stays_on_the_issue_until_a_pull_request_exists(
    tmp_path: Path,
) -> None:
    workflow, *_ = _workflow(tmp_path, [])
    assert workflow._comment_number("specifier") == 42
    assert workflow._comment_number("implementer") == 42
    assert workflow._comment_number() == 42
    workflow.store.record_development_pr(9, "https://example.test/pr/9")
    assert workflow._comment_number("specifier") == 42
    assert workflow._comment_number("implementer") == 9
    assert workflow._comment_number() == 9


def test_workflow_sends_the_body_once_and_only_five_latest_comments(
    tmp_path: Path,
) -> None:
    class ContextGitHub(FakeGitHub):
        def issue(self, number: int) -> dict:
            comments: list[object] = [
                {"author": f"user-{index}", "body": f"comment-{index}"}
                for index in range(7)
            ]
            comments.extend(
                [
                    "noise",
                    {"body": 12},
                    {"body": "<!-- gdw:42:old -->\nworkflow output"},
                ]
            )
            return {
                "number": number,
                "title": "Raw issue",
                "body": "Original issue body",
                "comments": comments,
                "url": "https://example.test/issues/42",
                "state": "open",
            }

    github = ContextGitHub()
    workflow, _github, _repository, agents = _workflow(
        tmp_path,
        [_expansion(), _grill("revise"), _expansion(), _grill()],
        {"github": github},
    )
    issue_context = workflow.load_issue()
    workflow.clarify(issue_context)

    assert [comment["body"] for comment in issue_context["latest_comments"]] == [
        "comment-2",
        "comment-3",
        "comment-4",
        "comment-5",
        "comment-6",
    ]
    initial = agents.calls[0]["values"]
    assert "Original issue body" in initial["ISSUE_CONTEXT_JSON"]
    assert "comment-0" not in initial["ISSUE_CONTEXT_JSON"]
    for call in agents.calls[1:]:
        serialized_values = json.dumps(call["values"])
        assert "Original issue body" not in serialized_values
        assert "comment-2" in serialized_values


def test_workflow_revises_repairs_ci_and_repairs_review(tmp_path: Path) -> None:
    replies = [
        _expansion(),
        _grill("revise"),
        _expansion(),
        _grill(),
        _specification(),
        _implementation(),
        _documentation(),
        _implementation(),
        _review("changes_requested"),
        _review(),
        _implementation(),
        _review(),
        _review(),
    ]
    repository = FakeRepository(
        [
            gdw.CommandResult(1, "CI failed"),
            gdw.CommandResult(0, "CI fixed"),
            gdw.CommandResult(0, "review repair green"),
        ]
    )
    workflow, _github, _repository, agents = _workflow(
        tmp_path, replies, {"repository": repository}
    )
    assert workflow.run().endswith("/1")
    prompts = [call["prompt_name"] for call in agents.calls]
    assert prompts.count("revise") == 1
    assert prompts.count("repair") == 2
    assert prompts.count("review-specification") == 2
    assert prompts.count("review-quality") == 2


def test_clarification_continues_until_both_agents_report_convergence(
    tmp_path: Path,
) -> None:
    first = _expansion(needs_another_round=True)
    first["open_questions"] = ["first unresolved question"]
    second = _expansion(needs_another_round=True)
    second["open_questions"] = ["different unresolved question"]
    final = _expansion()
    workflow, _github, _repository, agents = _workflow(
        tmp_path,
        [first, _grill(), second, _grill(), final, _grill()],
    )

    assert workflow.clarify({"initial": {}, "latest_comments": []}) == final
    assert [call["prompt_name"] for call in agents.calls] == [
        "expand",
        "grill",
        "revise",
        "grill",
        "revise",
        "grill",
    ]


def test_review_continues_until_every_reviewer_reports_convergence(
    tmp_path: Path,
) -> None:
    workflow, _github, repository, agents = _workflow(
        tmp_path,
        [
            _review("approve", needs_another_round=True),
            _review(),
            _implementation(),
            _review(),
            _review(),
        ],
        {"repository": FakeRepository([gdw.CommandResult(0, "green")])},
    )

    result = workflow.review(_specification(), {"returncode": 0, "output": "green"})
    assert all(not review["needs_another_round"] for review in result.values())
    assert [call["prompt_name"] for call in agents.calls] == [
        "review-specification",
        "review-quality",
        "repair",
        "review-specification",
        "review-quality",
    ]
    assert repository.ci == deque()


def test_repeated_unresolved_states_stop_stalled_processes(tmp_path: Path) -> None:
    clarification, *_ = _workflow(
        tmp_path / "clarification",
        [
            _expansion(needs_another_round=True),
            _grill("revise"),
            _expansion(needs_another_round=True),
            _grill("revise"),
        ],
    )
    with pytest.raises(gdw.WorkflowStopped, match="clarification stalled"):
        clarification.clarify({"initial": {}, "latest_comments": []})

    ci, *_ = _workflow(
        tmp_path / "ci",
        [_implementation(), _implementation()],
        {"repository": FakeRepository([gdw.CommandResult(1, "same failure")] * 2)},
    )
    with pytest.raises(gdw.WorkflowStopped, match="CI repair stalled"):
        ci.stabilize(_specification())

    review, *_ = _workflow(
        tmp_path / "review",
        [
            _review("changes_requested"),
            _review(),
            _implementation(),
            _review("changes_requested"),
            _review(),
        ],
        {"repository": FakeRepository([gdw.CommandResult(0, "green")])},
    )
    with pytest.raises(gdw.WorkflowStopped, match="review stalled"):
        review.review(_specification(), {"returncode": 0, "output": "green"})


@pytest.mark.parametrize(
    ("replies", "message"),
    [
        ([_expansion("stop")], "proposal"),
        ([_expansion(), _grill("reject")], "review"),
        (
            [_expansion(), _grill(), _specification(), _implementation("blocked")],
            "implementation blocked",
        ),
    ],
)
def test_workflow_deliberate_stop_conditions(
    tmp_path: Path, replies: list[dict], message: str
) -> None:
    workflow, _github, _repository, _agents = _workflow(tmp_path, replies)
    with pytest.raises(gdw.WorkflowStopped, match=message):
        workflow.run()


def test_workflow_stops_on_blocked_repairs_and_empty_pr_url(tmp_path: Path) -> None:
    ci_blocked = [
        _expansion(),
        _grill(),
        _specification(),
        _implementation(),
        _documentation(),
        _implementation("blocked"),
    ]
    workflow, *_ = _workflow(
        tmp_path / "ci",
        ci_blocked,
        {"repository": FakeRepository([gdw.CommandResult(1, "bad")])},
    )
    with pytest.raises(gdw.WorkflowStopped, match="CI repair blocked"):
        workflow.run()

    review_blocked = [
        _expansion(),
        _grill(),
        _specification(),
        _implementation(),
        _documentation(),
        _review("changes_requested"),
        _review(),
        _implementation("blocked"),
    ]
    workflow, *_ = _workflow(tmp_path / "review", review_blocked)
    with pytest.raises(gdw.WorkflowStopped, match="review repair blocked"):
        workflow.run()

    happy = [
        _expansion(),
        _grill(),
        _specification(),
        _implementation(),
        _documentation(),
        _review(),
        _review(),
    ]
    workflow, *_ = _workflow(tmp_path / "empty", happy, {"github": FakeGitHub("")})
    with pytest.raises(gdw.WorkflowError, match="empty pull-request URL"):
        workflow.run()


def test_cached_stage_does_not_call_agent(tmp_path: Path) -> None:
    workflow, github, _repository, agents = _workflow(tmp_path, [])
    workflow.store.save("cached", {"value": 1})
    assert workflow._stage(gdw.Stage("cached", "T", "r", "p", "s", {})) == {"value": 1}
    assert agents.calls == []
    assert github.comments == []


def test_positive_parser_and_main_success_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    assert gdw._positive("2") == 2
    with pytest.raises(Exception, match="expected 1 or more"):
        gdw._positive("0")
    assert gdw._parser().parse_args(["4", "--ready"]).ready is True
    assert gdw._parser().parse_args(["4", "--backend", "opencode"]).backend == (
        "opencode"
    )

    class MainGitHub(FakeGitHub):
        def __init__(self, _root: Path, _repo: str | None) -> None:
            super().__init__()

        def default_branch(self) -> str:
            return "master"

    class MainRepository(FakeRepository):
        def __init__(self, _root: Path) -> None:
            super().__init__()

        def prepare(self, _base: str, _resuming: bool) -> tuple[str, str]:
            return "feature", "base"

    class MainStore:
        initialized = False

        def __init__(self, root: Path) -> None:
            self.root = root

        def initialize(self, *_args) -> None:
            return None

    class MainWorkflow:
        should_fail = False

        def __init__(self, _options, _services) -> None:
            return None

        def run(self) -> str:
            if self.should_fail:
                raise gdw.WorkflowStopped("halt")
            return "https://example.test/pr/main"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gdw, "GitHub", MainGitHub)
    monkeypatch.setattr(gdw, "GitRepository", MainRepository)
    monkeypatch.setattr(gdw, "ArtifactStore", MainStore)
    monkeypatch.setattr(gdw, "AgentGateway", lambda **_kwargs: object())
    monkeypatch.setattr(gdw, "DevelopmentWorkflow", MainWorkflow)
    assert gdw.main(["4", "--backend", "codex"]) == 0
    assert "https://example.test/pr/main" in capsys.readouterr().out
    MainWorkflow.should_fail = True
    assert gdw.main(["4", "--base", "trunk"]) == 1
    assert "workflow stopped: halt" in capsys.readouterr().err


def test_role_config_accepts_opencode() -> None:
    config = RoleConfig(
        backend=" OpenCode ",
        github_app={"app_id": 1, "private_key": "key"},
    )
    assert config.backend == "opencode"


def test_prepare_simple_workflow_checks_tools_and_builds_services(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config = _workflow_config()
    monkeypatch.setattr(simple_setup.shutil, "which", lambda _name: None)
    with pytest.raises(gdw.WorkflowError, match="git is not installed"):
        simple_setup.prepare_workflow(7, config)

    monkeypatch.setattr(
        simple_setup.shutil,
        "which",
        lambda name: (
            f"/bin/{name}" if name in {"git", "make", "uv", "orchestrator"} else None
        ),
    )
    with pytest.raises(gdw.WorkflowError, match="configured agent backend 'codex'"):
        simple_setup.prepare_workflow(7, config)

    observed: dict = {}

    class SetupStore:
        initialized = False

        def __init__(self, root: Path) -> None:
            self.root = root

        def initialize(self, issue: int, branch: str, base_sha: str) -> None:
            observed["initialize"] = (issue, branch, base_sha)

    class SetupRepository:
        def __init__(self, root: Path) -> None:
            self.root = root
            observed.setdefault("repository_roots", []).append(root)

        def ensure_issue_worktree(
            self, branch: str, base_branch: str, path: Path
        ) -> None:
            observed["ensure_issue_worktree"] = (self.root, branch, base_branch, path)

        def prepare(self, base: str, resuming: bool) -> tuple[str, str]:
            observed["prepare"] = (self.root, base, resuming)
            return "feature", "base-sha"

    class SetupWorkflow:
        def __init__(self, options, services) -> None:
            self.options = options
            self.services = services

    github = SimpleNamespace(default_branch="trunk")
    connect = MagicMock(return_value=github)
    gateway = object()
    gateway_factory = MagicMock(return_value=gateway)
    monkeypatch.setattr(simple_setup.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(simple_setup.GitHubAppClient, "connect", connect)
    monkeypatch.setattr(simple_setup, "ArtifactStore", SetupStore)
    monkeypatch.setattr(simple_setup, "GitRepository", SetupRepository)
    monkeypatch.setattr(simple_setup, "AgentGateway", gateway_factory)
    monkeypatch.setattr(simple_setup, "DevelopmentWorkflow", SetupWorkflow)

    workflow = simple_setup.prepare_workflow(7, config)
    assert isinstance(workflow, SetupWorkflow)
    assert connect.call_count == len(REQUIRED_ROLES)
    assert {
        (call.args[0], call.args[1], call.args[2]) for call in connect.call_args_list
    } == {
        (
            role_config.github_app.app_id,
            role_config.github_app.private_key.get_secret_value(),
            "owner/project",
        )
        for role_config in config.roles.values()
    }
    issue_root = tmp_path / ".git" / "gdw" / "issue-7"
    worktree_path = issue_root / "worktree"
    assert observed["repository_roots"] == [tmp_path, worktree_path]
    assert observed["ensure_issue_worktree"] == (
        tmp_path,
        "gdw/issue-7",
        "trunk",
        worktree_path,
    )
    assert observed["prepare"] == (worktree_path, "trunk", False)
    assert observed["initialize"] == (7, "feature", "base-sha")
    assert workflow.options == gdw.WorkflowOptions(7, "trunk", "feature", False)
    assert workflow.services.store.root == issue_root
    assert workflow.services.github is github
    assert workflow.services.repository.__class__ is SetupRepository
    assert workflow.services.repository.root == worktree_path
    assert workflow.services.agents is gateway
    assert workflow.services.role_github == dict.fromkeys(REQUIRED_ROLES, github)
    gateway_factory.assert_called_once_with(
        roles=config.roles,
        issue=7,
        state_file=tmp_path / ".git" / "gdw" / "issue-7" / "agents.json",
        example_root=Path(simple_setup.__file__).resolve().parent,
        # The agents develop in the same tree git and CI use. Asserting the
        # repository root here instead is what let them diverge.
        workdir=worktree_path,
    )


def test_simple_entrypoint_shows_the_main_flow_and_resumes_completed_work(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    fresh = EntryWorkflow(None)
    resumed = EntryWorkflow("https://example.test/pulls/existing")
    prepare = MagicMock(side_effect=[fresh, resumed])
    monkeypatch.setattr(simple_setup, "prepare_workflow", prepare)
    load = MagicMock(
        return_value=_workflow_config(repository="gabepsilva/agents-army-2")
    )
    monkeypatch.setattr(workflow_config, "load_config", load)
    script = Path(simple_setup.__file__).with_name("simple_development_workflow.py")

    monkeypatch.setattr(sys, "argv", [str(script), "42"])
    # The script has to exit with main()'s status: a stopped workflow that
    # returns 1 through a discarded return value looks like success to the
    # shell, cron job, or CI step that started it.
    with pytest.raises(SystemExit) as exited:
        runpy.run_path(str(script), run_name="__main__")
    assert exited.value.code == 0
    assert capsys.readouterr().out.strip() == "https://example.test/pulls/new"
    assert fresh.calls == [
        "completed",
        "load",
        ("clarify", {"issue": 1}),
        ("specify", {"issue": 1}, {"proposal": 1}),
        ("open_pull_request", {"specification": 1}),
        ("implement", {"specification": 1}),
        ("stabilize", {"specification": 1}),
        ("review", {"specification": 1}, {"ci": "green"}),
        ("publish", {"specification": 1}, {"reviews": "approved"}),
    ]

    with pytest.raises(SystemExit) as resumed_exit:
        runpy.run_path(str(script), run_name="__main__")
    assert resumed_exit.value.code == 0
    assert capsys.readouterr().out.strip() == "https://example.test/pulls/existing"
    assert resumed.calls == ["completed"]
    assert [call.args[0] for call in prepare.call_args_list] == [42, 42]
    assert [call.args[1].repository for call in prepare.call_args_list] == [
        "gabepsilva/agents-army-2",
        "gabepsilva/agents-army-2",
    ]
    assert [call.args[0] for call in load.call_args_list] == [
        Path(simple_entrypoint.__file__).with_name("workflow.local"),
        Path(simple_entrypoint.__file__).with_name("workflow.local"),
    ]


def test_simple_entrypoint_defaults_to_local_workflow_config() -> None:
    options = simple_entrypoint._parser().parse_args(["42"])

    assert options.config == Path(simple_entrypoint.__file__).with_name(
        "workflow.local"
    )


def test_configure_logging_writes_one_stderr_handler_at_the_requested_level(
    capsys: pytest.CaptureFixture,
) -> None:
    gdw.configure_logging()
    assert gdw.LOGGER.level == logging.INFO
    assert len(gdw.LOGGER.handlers) == 1

    gdw.configure_logging(verbose=True)
    assert gdw.LOGGER.level == logging.DEBUG
    assert len(gdw.LOGGER.handlers) == 1

    gdw.LOGGER.debug("hello from the workflow")
    captured = capsys.readouterr()
    assert "hello from the workflow" in captured.err
    assert captured.out == ""


def test_outcome_digests_structured_replies_and_tail_bounds_evidence() -> None:
    assert gdw._outcome(_grill()) == "verdict=ready, needs_another_round=False"
    assert gdw._outcome(_implementation()) == "status=complete"
    assert gdw._outcome({}) == "no outcome fields"
    assert gdw._tail("a\nb\nc\nd", lines=2) == "c\nd"
    assert gdw._tail("a\nb") == "a\nb"


def test_workflow_logs_progress_checkpoint_reuse_and_ci_failures(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    replies = [
        _expansion(),
        _grill(),
        _specification(),
        _implementation(),
        _documentation(),
        _implementation(),
        _review(),
        _review(),
    ]
    repository = FakeRepository(
        [gdw.CommandResult(1, "boom\nfailing line"), gdw.CommandResult(0, "green")]
    )
    workflow, _github, _repository, _agents = _workflow(
        tmp_path, replies, {"repository": repository}
    )
    with caplog.at_level(logging.DEBUG, logger="gdw"):
        workflow.run()
    messages = [record.getMessage() for record in caplog.records]

    assert "workflow: issue #42, branch feature onto master, draft=True" in messages
    assert "clarification: round 1" in messages
    assert "clarification: converged after 1 round(s)" in messages
    assert "stage expansion-1: asking expander" in messages
    assert any(
        message.startswith("stage expansion-1: expander answered in")
        and message.endswith("(decision=proceed, needs_another_round=False)")
        for message in messages
    )
    assert "ci: implementation attempt 1" in messages
    assert any(
        "ci: implementation failed on attempt 1 (exit 1)" in message
        and "failing line" in message
        for message in messages
    )
    assert "ci: implementation green on attempt 2" in messages
    assert "review: round 1" in messages
    assert "review: approved after 1 round(s)" in messages
    assert "workflow: pull request created at https://example.test/pr/1" in messages

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="gdw"):
        workflow.run()
    assert "workflow: already completed, pull request https://example.test/pr/1" in [
        record.getMessage() for record in caplog.records
    ]


def test_cached_stage_logs_that_it_reused_the_checkpoint(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    workflow, github, _repository, agents = _workflow(tmp_path, [])
    workflow.store.save("specification", _specification())
    stage = gdw.Stage("specification", "Specification", "specifier", "specify", "s", {})

    with caplog.at_level(logging.INFO, logger="gdw"):
        workflow._stage(stage)

    assert agents.calls == []
    assert (
        "stage specification: reusing checkpoint from specifier (no outcome fields)"
        in [record.getMessage() for record in caplog.records]
    )
    assert github.comments == []
    assert github.attributions == []


def test_simple_entrypoint_reports_a_stopped_workflow_and_how_to_resume(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def explode(issue: int, config: WorkflowConfig) -> None:
        raise gdw.WorkflowStopped("clarification stalled")

    monkeypatch.setattr(simple_entrypoint, "prepare_workflow", explode)
    monkeypatch.setattr(
        simple_entrypoint,
        "load_config",
        MagicMock(return_value=_workflow_config(repository="gabepsilva/agents-army-2")),
    )

    with caplog.at_level(logging.INFO, logger="gdw"):
        assert simple_entrypoint.main(["42"]) == 1

    assert "workflow stopped: clarification stalled" in capsys.readouterr().err
    messages = [record.getMessage() for record in caplog.records]
    assert any("clarification stalled" in message for message in messages)
    assert (
        "workflow: rerun the same command to resume from the last checkpoint"
        in messages
    )


def test_progress_redraws_are_collapsed_out_of_ci_evidence() -> None:
    """mutmut repaints one status line per mutant, and only the tail of the
    output is kept: verbatim frames push the real errors out of the evidence."""
    spinner = "\n".join(f"{glyph} 1719/1719  killed 1659" for glyph in "⠦⠧⠇⠏⠋")
    noisy = f"ruff: would reformat orchestrator/__init__.py\n{spinner}\nError 1"

    assert gdw._readable(noisy) == (
        "ruff: would reformat orchestrator/__init__.py\n1719/1719  killed 1659\nError 1"
    )
    assert gdw._readable("same\rsame") == "same"
    assert gdw._readable("keep\nboth") == "keep\nboth"


def test_a_gate_reason_comes_from_its_own_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Separate stdout/stderr pipes dump every gate's diagnostics after the
    last announcement. Merging the streams at spawn keeps a failing gate's
    stderr inside its own block."""
    monkeypatch.delenv("MAKEFLAGS", raising=False)
    monkeypatch.delenv("MFLAGS", raising=False)
    monkeypatch.delenv("MAKELEVEL", raising=False)
    monkeypatch.setenv("JOBS", "1")
    (tmp_path / "Makefile").write_text(
        "\n".join(
            [
                ".PHONY: ci lint types",
                "MAKEFLAGS += -k",
                "ifneq ($(filter output-sync,$(.FEATURES)),)",
                "MAKEFLAGS += --output-sync=target",
                "endif",
                "gate = @printf '\\n=== gate: %s ===\\n' $@",
                "ci-gates:",
                "\t@printf '%s\\n' lint types",
                "lint:",
                "\t$(gate)",
                "\t@echo uv run ruff check",
                "\t@echo Found 12 errors. >&2",
                "\t@false",
                "types:",
                "\t$(gate)",
                "\t@echo uv run ty check",
                "ci: lint types",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = gdw.GitRepository(tmp_path).run_ci(timeout=30)

    lint = next(gate for gate in result.gates if gate.name == "lint")
    types = next(gate for gate in result.gates if gate.name == "types")
    assert lint.status == "failed"
    assert "Found 12 errors." in lint.reason
    assert types.status == "passed"
    assert "Found 12 errors." not in types.reason


def test_run_ci_reports_the_readable_evidence_not_the_raw_redraws(
    tmp_path: Path,
) -> None:
    frames = "\n".join(f"{glyph} working" for glyph in "⠦⠧⠇")
    run = ScriptedRun(
        [
            _completed([], stdout="lint\n"),
            _completed([], 2, stdout=f"boom\n{frames}", stderr=""),
        ]
    )
    repository = gdw.GitRepository(tmp_path, run)

    result = repository.run_ci()

    assert result.returncode == 2
    assert result.output == "boom\nworking"


def test_a_blocked_stage_names_the_checkpoint_to_delete(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The blocked reply is checkpointed, so a plain resume replays it."""
    workflow, _github, _repository, _agents = _workflow(tmp_path, [])
    blocked = _implementation(status="blocked")

    with (
        caplog.at_level(logging.ERROR, logger="gdw"),
        pytest.raises(gdw.WorkflowStopped, match="CI repair blocked"),
    ):
        workflow._require_complete(blocked, "CI repair", "repair-implementation-2")

    assert (
        f"stage repair-implementation-2 reported itself blocked; delete "
        f"{tmp_path / 'state' / 'artifacts' / 'repair-implementation-2.json'} "
        f"to ask again" in [record.getMessage() for record in caplog.records]
    )


@pytest.mark.parametrize(
    ("replies", "key", "verdict"),
    [
        ([_expansion("stop")], "expansion-1", "stop"),
        ([_expansion(), _grill("reject")], "grill-1", "reject"),
    ],
)
def test_a_refusing_stage_names_the_checkpoint_to_delete(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    replies: list[dict],
    key: str,
    verdict: str,
) -> None:
    """A refusal is checkpointed too, so a plain resume replays it forever.

    Regression: `stop` and `reject` raised without naming the artifact, so the
    operator was told only to rerun the command — which replays the same
    refusal and can never move the run forward.
    """
    workflow, _github, _repository, _agents = _workflow(tmp_path, replies)

    with (
        caplog.at_level(logging.ERROR, logger="gdw"),
        pytest.raises(gdw.WorkflowStopped),
    ):
        workflow.run()

    assert (
        f"stage {key} returned '{verdict}'; delete "
        f"{tmp_path / 'state' / 'artifacts' / key}.json to ask again"
        in [record.getMessage() for record in caplog.records]
    )


def test_ci_signature_ignores_run_to_run_noise_but_not_a_moved_score() -> None:
    """`make -j` interleaves differently every run, so comparing the evidence
    itself never reports a stall on a repair that achieved nothing."""
    failure = (
        "mutation score 95.4% (1671 killed, 78 survived, 1751 mutants) floor 98.0%\n"
        "make: *** [Makefile:85: mutation] Error 1\n"
    )
    noisier = f"unrelated gate chatter\n{failure}"
    moved = failure.replace("95.4", "96.1").replace("1671", "1685")

    signature = gdw._ci_signature({"returncode": 2, "output": failure})

    assert signature == gdw._ci_signature({"returncode": 2, "output": noisier})
    assert signature != gdw._ci_signature({"returncode": 2, "output": moved})
    assert json.loads(signature) == {
        "returncode": 2,
        "failed_targets": ["mutation"],
        "verdicts": [
            "mutation score 95.4% (1671 killed, 78 survived, 1751 mutants) floor 98.0%"
        ],
    }
    assert json.loads(gdw._ci_signature({"returncode": 0})) == {
        "returncode": 0,
        "failed_targets": [],
        "verdicts": [],
    }


def test_ci_signature_follows_gate_reasons_not_the_interleaved_log() -> None:
    """A coverage or pytest number that moved is progress, even when the same
    target still fails and the log around it is noise."""
    first = {
        "returncode": 2,
        "output": "whatever interleaving",
        "gates": [
            {
                "name": "test-coverage",
                "status": "failed",
                "reason": "=== 10 failed, 0 passed in 5.2s ===",
            }
        ],
    }
    moved = {
        "returncode": 2,
        "output": "whatever interleaving",
        "gates": [
            {
                "name": "test-coverage",
                "status": "failed",
                "reason": "=== 1 failed, 9 passed in 5.1s ===",
            }
        ],
    }
    noisier = {
        "returncode": 2,
        "output": "different make -j ordering",
        "gates": first["gates"],
    }

    assert gdw._ci_signature(first) == gdw._ci_signature(noisier)
    assert gdw._ci_signature(first) != gdw._ci_signature(moved)
    assert json.loads(gdw._ci_signature(first)) == {
        "returncode": 2,
        "failed": [
            {
                "name": "test-coverage",
                "reason": "=== 10 failed, 0 passed in 5.2s ===",
            }
        ],
    }


def test_ci_signature_sees_a_moved_coverage_failure_in_the_log() -> None:
    """When make never named its gates, the fallback still has to see the
    numbers coverage actually prints, not a regex that never matches them."""
    first = (
        "error: mod.py: 90.0% is below its floor of 100.0%.\n"
        "1 per-file coverage failure(s).\n"
        "make: *** [Makefile:1: test-coverage] Error 1\n"
    )
    moved = first.replace("90.0", "96.0")

    assert gdw._ci_signature({"returncode": 2, "output": first}) != gdw._ci_signature(
        {"returncode": 2, "output": moved}
    )


CI_LOG = """
=== gate: lint ===
uv run ruff check .
All checks passed!

=== gate: types ===
uv run ty check
error[invalid-assignment] orchestrator/state.py:41: not assignable
Found 3 diagnostics.
make: *** [Makefile:88: types] Error 1
make: *** Waiting for unfinished jobs....

=== gate: mutation ===
mutation score 95.4% (1671 killed, 78 survived, 1751 mutants) floor 98.0%
make: *** [Makefile:110: mutation] Error 1
"""


def test_each_gate_is_reported_as_passed_failed_or_never_started() -> None:
    """A gate that never announced itself never ran, which is not passing."""
    gates = gdw._gate_results(("lint", "types", "mutation", "secrets"), CI_LOG)

    assert [gate.as_json() for gate in gates] == [
        {"name": "lint", "status": "passed", "reason": ""},
        {
            "name": "types",
            "status": "failed",
            "reason": (
                "uv run ty check error[invalid-assignment] "
                "orchestrator/state.py:41: not assignable Found 3 diagnostics."
            ),
        },
        {
            "name": "mutation",
            "status": "failed",
            "reason": (
                "mutation score 95.4% (1671 killed, 78 survived, 1751 mutants) "
                "floor 98.0%"
            ),
        },
        {"name": "secrets", "status": "not run", "reason": "not run"},
    ]
    assert gdw._gate_checklist(gates) == [
        "✅ lint",
        "❌ types — uv run ty check error[invalid-assignment] "
        "orchestrator/state.py:41: not assignable Found 3 diagnostics.",
        "❌ mutation — mutation score 95.4% (1671 killed, 78 survived, "
        "1751 mutants) floor 98.0%",
        "⚪ secrets — not run",
    ]


def test_a_gate_the_makefile_never_advertised_is_still_reported() -> None:
    """The advertised list is the Makefile's; a run that contradicts it is
    evidence about the run, not a reason to drop a gate from the report."""
    gates = gdw._gate_results((), CI_LOG)

    assert [(gate.name, gate.status) for gate in gates] == [
        ("lint", "passed"),
        ("mutation", "failed"),
        ("types", "failed"),
    ]


def test_a_failure_reason_is_a_headline_not_a_log() -> None:
    wordy = "=== gate: lint ===\n" + " ".join(f"word{index}" for index in range(40))
    wordy += "\nmake: *** [Makefile:1: lint] Error 1\n"

    reason = gdw._gate_results(("lint",), wordy)[0].reason

    assert reason.split() == [f"word{index}" for index in range(15)] + ["…"]


def test_a_gate_that_failed_silently_says_so() -> None:
    silent = "=== gate: secrets ===\nmake: *** [Makefile:1: secrets] Error 1\n"

    assert gdw._gate_results(("secrets",), silent) == (
        gdw.GateResult("secrets", "failed", "failed without saying anything"),
    )


def test_a_run_whose_gates_are_unknown_still_reports_a_verdict() -> None:
    """An older make cannot list its gates; the run still owes the issue an
    answer, so it reports as the one command it was."""
    assert gdw.CommandResult(0, "green").checklist() == ["✅ make ci"]
    assert gdw.CommandResult(2, "boom").checklist() == [
        "❌ make ci — exit 2, no gate named itself"
    ]


def test_gates_are_read_from_the_whole_log_not_the_bounded_tail(
    tmp_path: Path,
) -> None:
    """Only the tail of a long run is kept as evidence. Reading the gates from
    that tail would report every early gate as never started."""
    # Distinct lines: identical consecutive ones collapse as progress redraws.
    filler = "\n".join(f"noise {index}" for index in range(gdw.CI_EVIDENCE_CHARS))
    log = f"=== gate: lint ===\nok\n{filler}\n=== gate: secrets ===\nclean\n"
    run = ScriptedRun(
        [
            _completed([], stdout="lint\nsecrets\n"),
            _completed([], stdout=log, stderr=""),
        ]
    )

    result = gdw.GitRepository(tmp_path, run).run_ci()

    assert "=== gate: lint ===" not in result.output
    assert result.checklist() == ["✅ lint", "✅ secrets"]


def test_a_make_that_cannot_list_its_gates_does_not_stop_ci(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    run = ScriptedRun(
        [
            _completed([], 2, stdout="", stderr="No rule to make target 'ci-gates'"),
            _completed([], 1, stdout="boom", stderr=""),
        ]
    )

    with caplog.at_level(logging.WARNING, logger="gdw"):
        result = gdw.GitRepository(tmp_path, run).run_ci()

    assert result.gates == ()
    assert result.checklist() == ["❌ make ci — exit 1, no gate named itself"]
    assert "unstarted gates will go unreported" in caplog.text


def test_ci_posts_a_checklist_and_keeps_the_log_off_github(
    tmp_path: Path,
) -> None:
    """The pull request gets the verdict; the evidence stays in the checkpoint
    and in the repair agent's prompt, where it is actually read."""
    failing = gdw.CommandResult(
        2,
        "uv run ruff check .\nFound 12 errors.",
        (
            gdw.GateResult("lint", "failed", "Found 12 errors."),
            gdw.GateResult("mutation", "not run", "not run"),
        ),
    )
    repository = FakeRepository([failing, gdw.CommandResult(0, "green")])
    workflow, github, _repository, agents = _workflow(
        tmp_path, [_implementation()], {"repository": repository}
    )

    workflow.stabilize(_specification())

    titles = [comment[2] for comment in github.comments]
    payloads = [comment[3] for comment in github.comments]
    assert github.comments[0][0] == 1
    assert titles[0] == "CI checks for implementation, attempt 1"
    assert payloads[0] == [
        "❌ lint — Found 12 errors.",
        "⚪ mutation — not run",
    ]
    assert all("uv run ruff check" not in gdw._json(payload) for payload in payloads)
    assert "uv run ruff check ." in agents.calls[0]["values"]["FAILURE_EVIDENCE"]
    checkpoint = json.loads(
        (tmp_path / "state" / "artifacts" / "ci-implementation-1.json").read_text(
            encoding="utf-8"
        )
    )
    assert checkpoint["output"] == "uv run ruff check .\nFound 12 errors."
    assert checkpoint["gates"][0]["name"] == "lint"


def test_a_coverage_failure_that_moved_is_not_a_stall(tmp_path: Path) -> None:
    """Two test-coverage failures that still moved (10 tests then 1; 90% then
    96%) must not stop the repair loop as stalled."""
    first = gdw.CommandResult(
        2,
        "=== 10 failed, 0 passed in 5.2s ===\n"
        "error: mod.py: 90.0% is below its floor of 100.0%.",
        (
            gdw.GateResult(
                "test-coverage",
                "failed",
                "=== 10 failed, 0 passed in 5.2s === error: mod.py: 90.0% is "
                "below its floor of 100.0%.",
            ),
        ),
    )
    second = gdw.CommandResult(
        2,
        "=== 1 failed, 9 passed in 5.1s ===\n"
        "error: mod.py: 96.0% is below its floor of 100.0%.",
        (
            gdw.GateResult(
                "test-coverage",
                "failed",
                "=== 1 failed, 9 passed in 5.1s === error: mod.py: 96.0% is "
                "below its floor of 100.0%.",
            ),
        ),
    )
    repository = FakeRepository([first, second, gdw.CommandResult(0, "green")])
    workflow, _github, _repository, agents = _workflow(
        tmp_path,
        [_implementation(), _implementation()],
        {"repository": repository},
    )

    assert workflow.stabilize(_specification())["returncode"] == 0
    assert len(agents.calls) == 2


def test_a_ci_failure_that_never_moves_is_reported_as_stalled(
    tmp_path: Path,
) -> None:
    """The loop this replaces ran forever: the agent reworded its repair every
    round, so the compared state never repeated even though CI never moved."""
    failure = (
        "mutation score 95.4% (1671 killed, 78 survived, 1751 mutants) floor 98.0%\n"
        "make: *** [Makefile:85: mutation] Error 1\n"
    )
    repository = FakeRepository(
        [
            gdw.CommandResult(2, f"first ordering\n{failure}"),
            gdw.CommandResult(2, f"second ordering, other gates chatter\n{failure}"),
        ]
    )
    repairs = [
        _implementation() | {"summary": "tightened the assertions"},
        _implementation() | {"summary": "rewrote them again, differently"},
    ]
    workflow, _github, _repository, agents = _workflow(
        tmp_path, repairs, {"repository": repository}
    )

    with pytest.raises(gdw.WorkflowStopped, match="CI repair stalled"):
        workflow.stabilize(_specification())

    assert len(agents.calls) == 2


def test_every_role_learns_which_stages_were_already_commented(
    tmp_path: Path,
) -> None:
    """Only the client that reads the issue sees its comments. Without sharing
    them, each other role repeats every stage it owns on a resumed run —
    issue #22 carried two 'specification' comments because of exactly this."""
    reader = FakeGitHub()
    specifier_app = FakeGitHub()
    reader.markers = {"specification", "expansion-1"}
    workflow, _github, _repository, _agents = _workflow(
        tmp_path,
        [_specification()],
        {"github": reader},
    )
    # The reader is itself a role client: it must not be handed its own set.
    workflow.role_github = {"specifier": specifier_app, "implementer": reader}

    workflow.load_issue()
    workflow._stage(
        gdw.Stage("specification", "Specification", "specifier", "specify", "s", {})
    )

    assert specifier_app.markers == {"specification", "expansion-1"}
    assert reader.markers == {"specification", "expansion-1"}
    assert specifier_app.comments == [], (
        "the specifier reposted a stage already on the issue"
    )


def test_both_github_clients_adopt_markers(tmp_path: Path) -> None:
    """Each client keeps its own set, so a resumed run needs both to learn."""
    gh_cli = gdw.GitHub(
        tmp_path,
        "owner/repo",
        executable=tmp_path / "gh",
        run=ScriptedRun([]),
    )
    gh_cli.markers = {"already"}
    gh_cli.adopt_markers({"expansion-1"})

    app = app_github.GitHubAppClient(cast(app_github.Repository, SimpleNamespace()))
    app.adopt_markers({"grill-1"})
    app.adopt_markers({"grill-2"})

    assert gh_cli.markers == {"already", "expansion-1"}
    assert app.markers == {"grill-1", "grill-2"}
