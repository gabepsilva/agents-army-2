"""Reading and updating one repository as the installed GitHub App."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest

from examples.gabriels_workflow_v2 import errors
from examples.gabriels_workflow_v2 import github_app as app_github


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


def test_github_app_pull_request_context_normalizes_empty_and_populated_sources() -> (
    None
):
    created_at = datetime(2026, 8, 20, tzinfo=UTC)
    issue = SimpleNamespace(
        number=9,
        title="Title",
        body="Body",
        get_comments=lambda: [
            SimpleNamespace(
                user=SimpleNamespace(login="alice"),
                body="discussion",
                created_at=created_at,
                html_url="comment",
            )
        ],
    )
    pull = SimpleNamespace(
        html_url="https://example.test/pr/9",
        get_reviews=lambda: [
            SimpleNamespace(
                user=SimpleNamespace(login="bob"),
                state="APPROVED",
                body="review",
                submitted_at=created_at,
                html_url="review",
            )
        ],
        get_review_comments=lambda: [
            SimpleNamespace(
                user=SimpleNamespace(login="carol"),
                body="inline",
                path="main.py",
                line=12,
                created_at=created_at,
                html_url="inline",
            )
        ],
    )
    repository = SimpleNamespace(
        get_issue=MagicMock(return_value=issue),
        get_pull=MagicMock(return_value=pull),
    )
    context = app_github.GitHubAppClient(
        cast(app_github.Repository, repository)
    ).pull_request_context(9)
    assert context["comments"][0]["author"] == "alice"
    assert context["reviews"][0]["state"] == "APPROVED"
    assert context["review_comments"][0]["line"] == 12


def test_github_app_pull_request_context_wraps_read_failures() -> None:
    repository = SimpleNamespace(
        get_issue=MagicMock(side_effect=RuntimeError("unreadable")),
        get_pull=MagicMock(),
    )
    client = app_github.GitHubAppClient(cast(app_github.Repository, repository))
    with pytest.raises(errors.WorkflowError, match="cannot read pull-request context"):
        client.pull_request_context(9)


def test_github_app_pull_request_context_rejects_malformed_scalar_fields() -> None:
    assert app_github.GitHubAppClient._author(None) is None
    with pytest.raises(errors.WorkflowError, match="invalid pull-request author"):
        app_github.GitHubAppClient._author(SimpleNamespace(login=9))

    assert app_github.GitHubAppClient._text(None, "body", allow_none=True) == ""
    with pytest.raises(errors.WorkflowError, match="invalid body"):
        app_github.GitHubAppClient._text(9, "body")

    with pytest.raises(errors.WorkflowError, match="invalid timestamp"):
        app_github.GitHubAppClient._timestamp(None, "timestamp")
    with pytest.raises(errors.WorkflowError, match="invalid timestamp"):
        app_github.GitHubAppClient._timestamp(
            SimpleNamespace(isoformat=lambda: ""), "timestamp"
        )
    with pytest.raises(errors.WorkflowError, match="invalid review comment line"):
        app_github.GitHubAppClient._line(True)


@pytest.mark.parametrize(
    ("issue_number", "title", "url"),
    [
        (True, "Title", "https://example.test/pr/9"),
        (9, "", "https://example.test/pr/9"),
        (9, "Title", "https://example.test/pr/10"),
    ],
)
def test_github_app_pull_request_context_rejects_mismatched_identity(
    issue_number: object, title: str, url: str
) -> None:
    issue = SimpleNamespace(
        number=issue_number,
        title=title,
        body="Body",
        get_comments=lambda: [],
    )
    pull = SimpleNamespace(
        html_url=url,
        get_reviews=lambda: [],
        get_review_comments=lambda: [],
    )
    repository = SimpleNamespace(
        get_issue=MagicMock(return_value=issue),
        get_pull=MagicMock(return_value=pull),
    )
    client = app_github.GitHubAppClient(cast(app_github.Repository, repository))

    with pytest.raises(errors.WorkflowError, match="cannot read pull-request context"):
        client.pull_request_context(9)


def test_render_comment_only_appends_a_nonempty_attribution() -> None:
    base = "<!-- marker -->\n## GDW — Title\n\n### Answer\n\n1\n"
    footer = (
        "\n---\n\nbackend: `codex`  \nmodel: _unset_  \n"
        "reasoning_effort: _unset_  \n"
        "task_duration: `0.0s`  \nskills: _none_  \n"
        "worktree: `worktree` - `/tmp/worktree`"
    )

    assert app_github._render_comment("<!-- marker -->", "Title", {"answer": 1}) == base
    assert (
        app_github._render_comment("<!-- marker -->", "Title", {"answer": 1}, "")
        == base
    )
    assert (
        app_github._render_comment("<!-- marker -->", "Title", {"answer": 1}, footer)
        == base + footer
    )


def test_a_comment_renders_markdown_and_a_pull_request_url_yields_its_number() -> None:
    rendered = app_github._render_comment(
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

    assert app_github.pull_request_number("https://github.com/o/r/pull/12") == 12
    assert app_github.pull_request_number("https://example.test/pr/1/") == 1
    with pytest.raises(errors.WorkflowError, match="without a number"):
        app_github.pull_request_number("https://example.test/pulls")
