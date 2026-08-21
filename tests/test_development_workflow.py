"""Behavior tests for the raw-issue-to-PR workflow example."""

from __future__ import annotations

import json
import logging
import os
import runpy
import subprocess
import sys
from collections import deque
from collections.abc import Mapping
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
        return _completed(args, reply.returncode, reply.stdout, reply.stderr)


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
            "claude, codex, grok",
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
        self.pr_calls: list[dict] = []
        self.pr_url = pr_url
        self.markers: set[str] = set()

    def issue(self, number: int) -> dict:
        return {"number": number, "title": "Raw issue", "body": "Please build it"}

    def adopt_markers(self, markers: set[str]) -> None:
        self.markers |= markers

    def comment_once(self, number: int, key: str, title: str, payload: object) -> None:
        if key in self.markers:
            return
        self.comments.append((number, key, title, payload))
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


class FakeRepository:
    def __init__(self, ci: list[gdw.CommandResult] | None = None) -> None:
        self.ci = deque(ci or [gdw.CommandResult(0, "green")])
        self.commits: list[tuple[str, str]] = []
        self.pushes: list[str] = []

    def run_ci(self) -> gdw.CommandResult:
        return self.ci.popleft()

    def commit(self, message: str, base_sha: str) -> None:
        self.commits.append((message, base_sha))

    def push(self, branch: str) -> None:
        self.pushes.append(branch)


class FakeAgents:
    def __init__(self, replies: list[dict]) -> None:
        self.replies = deque(replies)
        self.calls: list[dict] = []

    def ask(
        self,
        *,
        role: str,
        prompt_name: str,
        schema_name: str,
        values: Mapping[str, str],
        timeout: int = gdw.DEFAULT_AGENT_TIMEOUT,
    ) -> dict:
        self.calls.append(
            {
                "role": role,
                "prompt_name": prompt_name,
                "schema_name": schema_name,
                "values": values,
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
    }
    settings.update(overrides or {})
    store = gdw.ArtifactStore(tmp_path / "state")
    store.initialize(42, "feature", "base-sha")
    github = settings["github"] or FakeGitHub()
    repository = settings["repository"] or FakeRepository()
    agents = FakeAgents(agent_replies)
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
    assert gdw.CommandResult(0, "ok").succeeded is True
    assert gdw.CommandResult(1, "bad").succeeded is False
    assert gdw.CommandResult(1, "bad").as_json() == {
        "returncode": 1,
        "output": "bad",
    }


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
    }
    repository.commit("message", "abc")
    repository.push("feature")
    commands = [call[0] for call in runner.calls]
    assert ["git", "add", "--", "file", "new.py"] in commands
    assert ["git", "commit", "-m", "message"] in commands
    assert commands[-1] == ["git", "push", "--set-upstream", "origin", "feature"]
    ci_kwargs = runner.calls[3][1]
    assert ci_kwargs["timeout"] == gdw.DEFAULT_CI_TIMEOUT
    assert ci_kwargs["stdin"] == subprocess.DEVNULL

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
    github.comment_once(9, "new", "New", {"answer": 1})
    github.comment_once(9, "new", "New", {"answer": 1})
    assert len(bodies) == 1
    assert "<!-- gdw:9:new -->" in bodies[0]
    assert "### Answer\n\n1" in bodies[0]
    assert "```json" not in bodies[0]
    assert (
        github.create_pr(
            base="trunk", branch="feature", title="Title", body="PR body", draft=True
        )
        == "https://example.test/pr/9"
    )
    assert bodies[-1] == "PR body"

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
    client.comment_once(9, "new", "New", {"answer": 1})
    client.comment_once(9, "new", "New", {"answer": 1})
    posted = issue.create_comment.call_args.args[0]
    assert "<!-- gdw:9:new -->" in posted
    assert "### Answer\n\n1" in posted
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
        if args[1] == "ensure":
            return _completed(args, stdout="created agent\n")
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
    assert calls[0][0] == [
        "orchestrator",
        "ensure",
        "gdw-3-expander",
        "--backend",
        "codex",
    ]
    assert calls[1][0][:8] == [
        "orchestrator",
        "--agent",
        "gdw-3-expander",
        "--validate-schema",
        str(root / "validations" / "expansion.json"),
        "--timeout",
        "17",
        "--prompt",
    ]
    assert calls[0][1]["env"]["AGENTS_ARMY_STATE_FILE"] == str(tmp_path / "agents.json")
    assert calls[1][1]["timeout"] == 22
    assert os.environ["GH_TOKEN"] == "secret"
    assert os.environ["PATH"] == original_path


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
            _completed([], stdout="reused agent\n"),
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
            _completed([], stdout="reused agent\n"),
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
        if args[1] == "ensure":
            return _completed(args, stdout="created agent\n")
        return _completed(
            args,
            stdout=f"[gdw-5-expander session=s1]\n{json.dumps(_expansion())}\n",
        )

    gateway = gdw.AgentGateway(
        roles={"expander": configured},
        issue=5,
        state_file=tmp_path / "agents.json",
        example_root=Path(gdw.__file__).parent,
        run=fake_run,
    )
    gateway.ask(
        role="expander",
        prompt_name="expand",
        schema_name="expansion",
        values={"ISSUE_CONTEXT_JSON": "{}"},
    )
    assert calls[0] == [
        "orchestrator",
        "ensure",
        "gdw-5-expander",
        "--backend",
        "grok",
        "--model",
        "grok-code-test",
        "--reasoning-effort",
        "xhigh",
    ]
    with pytest.raises(gdw.WorkflowError, match="role 'missing' is not configured"):
        gateway.ask(
            role="missing",
            prompt_name="expand",
            schema_name="expansion",
            values={"ISSUE_CONTEXT_JSON": "{}"},
        )


def test_all_workflow_schemas_are_strict_and_prompts_resolve(tmp_path: Path) -> None:
    example_root = Path(gdw.__file__).parent
    schema_names = ("expansion", "grill", "specification", "implementation", "review")
    for name in schema_names:
        schema = load_schema(example_root / "validations" / f"{name}.json")
        assert schema.path.is_absolute()

    gateway = gdw.AgentGateway(
        roles={},
        issue=1,
        state_file=tmp_path / "agents.json",
        example_root=example_root,
    )
    values = {
        "ISSUE_CONTEXT_JSON": "{}",
        "LATEST_COMMENTS_JSON": "[]",
        "EXPANSION_JSON": "{}",
        "GRILL_JSON": "{}",
        "SPECIFICATION_JSON": "{}",
        "FAILURE_EVIDENCE": "{}",
        "REVIEW_KIND": "quality",
        "CI_SUMMARY": "{}",
    }
    prompt_names = (
        "expand",
        "grill",
        "revise",
        "specify",
        "implement",
        "repair",
        "review",
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
        _review(),
        _review(),
    ]
    workflow, github, repository, agents = _workflow(tmp_path, replies)
    assert workflow.run() == "https://example.test/pr/1"
    assert repository.commits == [
        ("Implement #42: Implement the thing with detail", "base-sha")
    ]
    assert repository.pushes == ["feature"]
    assert github.pr_calls[0]["draft"] is True
    pr_body = github.pr_calls[0]["body"]
    assert "Closes #42" in pr_body
    assert "### Problem Statement\n\nproblem" in pr_body
    assert "### Specification\n\n#### Verdict\n\napprove" in pr_body
    assert "```json" not in pr_body
    assert len(agents.calls) == 6
    assert workflow.run() == "https://example.test/pr/1"
    assert len(agents.calls) == 6


def test_each_stage_comments_as_its_configured_github_app(tmp_path: Path) -> None:
    replies = [
        _expansion(),
        _grill(),
        _specification(),
        _implementation(),
        _review(),
        _review(),
    ]
    workflow, driver_github, _repository, _agents = _workflow(tmp_path, replies)
    role_github = {role: FakeGitHub() for role in REQUIRED_ROLES}
    workflow.role_github = cast(dict[str, gdw.GitHubService], role_github)

    workflow.run()

    expected_keys = {
        "expander": "expansion-1",
        "griller": "grill-1",
        "specifier": "specification",
        "implementer": "implementation",
        "reviewer-specification": "review-1-specification",
        "reviewer-quality": "review-1-quality",
    }
    assert {
        role: client.comments[0][1] for role, client in role_github.items()
    } == expected_keys
    assert all(len(client.comments) == 1 for client in role_github.values())
    assert {comment[1] for comment in driver_github.comments} == {
        "ci-implementation-1",
        "pull-request",
    }


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
    assert prompts.count("review") == 4


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
        "review",
        "review",
        "repair",
        "review",
        "review",
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
            observed["repository_root"] = root

        def prepare(self, base: str, resuming: bool) -> tuple[str, str]:
            observed["prepare"] = (base, resuming)
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
    assert observed["prepare"] == ("trunk", False)
    assert observed["initialize"] == (7, "feature", "base-sha")
    assert workflow.options == gdw.WorkflowOptions(7, "trunk", "feature", False)
    assert workflow.services.store.root == tmp_path / ".git" / "gdw" / "issue-7"
    assert workflow.services.github is github
    assert workflow.services.repository.__class__ is SetupRepository
    assert workflow.services.agents is gateway
    assert workflow.services.role_github == dict.fromkeys(REQUIRED_ROLES, github)
    gateway_factory.assert_called_once_with(
        roles=config.roles,
        issue=7,
        state_file=tmp_path / ".git" / "gdw" / "issue-7" / "agents.json",
        example_root=Path(simple_setup.__file__).resolve().parent,
    )


def test_simple_entrypoint_shows_the_main_flow_and_resumes_completed_work(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
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
    workflow, _github, _repository, agents = _workflow(tmp_path, [])
    workflow.store.save("specification", _specification())
    stage = gdw.Stage("specification", "Specification", "specifier", "specify", "s", {})

    with caplog.at_level(logging.INFO, logger="gdw"):
        workflow._stage(stage)

    assert agents.calls == []
    assert (
        "stage specification: reusing checkpoint from specifier (no outcome fields)"
        in [record.getMessage() for record in caplog.records]
    )


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


def test_run_ci_reports_the_readable_evidence_not_the_raw_redraws(
    tmp_path: Path,
) -> None:
    frames = "\n".join(f"{glyph} working" for glyph in "⠦⠧⠇")
    run = ScriptedRun([_completed([], 2, stdout=f"boom\n{frames}", stderr="")])
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
