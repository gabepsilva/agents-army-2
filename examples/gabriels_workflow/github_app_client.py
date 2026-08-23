"""GitHub App communication for the linear development-workflow example."""

from __future__ import annotations

import logging
from typing import Any

from github import Auth, Github, GithubIntegration
from github.Repository import Repository

from examples.gabriels_workflow.development_workflow import (
    WorkflowError,
    _pull_request_number,
)

LOGGER = logging.getLogger("gdw")


class GitHubAppClient:
    """Read and update one repository as its installed GitHub App."""

    # A subclass that publishes a different workflow's comments overrides
    # these so its markers never collide with, and are never adopted by,
    # another workflow reading the same issue.
    marker_prefix = "gdw"
    comment_heading = "GDW"

    def __init__(
        self,
        repository: Repository,
        github: Github | None = None,
        integration: GithubIntegration | None = None,
    ) -> None:
        self.repository = repository
        self.github = github
        self.integration = integration
        self.markers: set[str] = set()

    @classmethod
    def connect(
        cls,
        app_id: int,
        private_key: str,
        repository_name: str,
    ) -> GitHubAppClient:
        owner, name = repository_name.split("/", 1)
        app_auth = Auth.AppAuth(app_id, private_key)
        integration = GithubIntegration(auth=app_auth, per_page=100)
        installation = integration.get_repo_installation(owner, name)
        installation_auth = app_auth.get_installation_auth(installation.id)
        github = Github(auth=installation_auth, per_page=100)
        repository = github.get_repo(repository_name)
        LOGGER.debug(
            "github-app: app %s connected to %s as installation %s",
            app_id,
            repository_name,
            installation.id,
        )
        return cls(repository, github, integration)

    @property
    def default_branch(self) -> str:
        return self.repository.default_branch

    def issue(self, number: int) -> dict[str, Any]:
        issue = self.repository.get_issue(number)
        comments = [
            {
                "author": comment.user.login if comment.user is not None else None,
                "body": comment.body,
                "createdAt": comment.created_at.isoformat(),
                "url": comment.html_url,
            }
            for comment in issue.get_comments()
        ]
        self.markers = {
            line
            for comment in comments
            for line in comment["body"].splitlines()
            if line.startswith("<!-- gdw:")
        }
        return {
            "number": issue.number,
            "title": issue.title,
            "body": issue.body or "",
            "comments": comments,
            "url": issue.html_url,
            "labels": [label.name for label in issue.labels],
            "state": issue.state,
        }

    def adopt_markers(self, markers: set[str]) -> None:
        """Learn which stages another role's client already commented.

        Only the client that read the issue saw its comments. Every other role
        posts through its own app, so without this each of them repeats every
        stage it owns on a resumed run.
        """
        self.markers |= markers

    def collect_markers(self, number: int) -> None:
        """Union this client's markers from comments on an issue or pull request."""

        prefix = f"<!-- {self.marker_prefix}:"
        issue = self.repository.get_issue(number)
        self.markers |= {
            line
            for comment in issue.get_comments()
            for line in (comment.body or "").splitlines()
            if line.startswith(prefix)
        }

    def comment(self, number: int, body: str) -> None:
        self.repository.get_issue(number).create_comment(body)

    def comment_once(
        self,
        number: int,
        key: str,
        title: str,
        payload: object,
        *,
        attribution: str = "",
    ) -> None:
        marker = f"<!-- {self.marker_prefix}:{number}:{key} -->"
        if marker in self.markers:
            LOGGER.info("github-app: comment '%s' already posted, skipping", key)
            return
        from examples.gabriels_workflow.development_workflow import _render_comment

        LOGGER.info("github-app: commenting '%s' on #%s", key, number)
        self.comment(
            number,
            _render_comment(
                marker, title, payload, attribution, heading=self.comment_heading
            ),
        )
        self.markers.add(marker)

    def create_pr(
        self,
        *,
        base: str,
        branch: str,
        title: str,
        body: str,
        draft: bool,
    ) -> str:
        LOGGER.info(
            "github-app: creating %s pull request %s -> %s",
            "draft" if draft else "ready",
            branch,
            base,
        )
        pull_request = self.repository.create_pull(
            base=base,
            head=branch,
            title=title,
            body=body,
            draft=draft,
        )
        return pull_request.html_url

    def update_pr(self, number: int, *, body: str) -> None:
        LOGGER.info("github-app: updating pull request #%s", number)
        self.repository.get_pull(number).edit(body=body)

    def pull_request_context(self, number: int) -> dict[str, Any]:
        try:
            issue = self.repository.get_issue(number)
            pull = self.repository.get_pull(number)
            if (
                isinstance(issue.number, bool)
                or not isinstance(issue.number, int)
                or issue.number != number
            ):
                raise WorkflowError(
                    f"GitHub returned pull-request identity that does not match #{number}"
                )
            title = self._text(issue.title, "pull-request title")
            if not title:
                raise WorkflowError("GitHub returned an empty pull-request title")
            url = self._text(pull.html_url, "pull-request URL")
            if not url or _pull_request_number(url) != number:
                raise WorkflowError(
                    f"GitHub returned pull-request identity that does not match #{number}"
                )
            comments = [
                self._comment_context(comment) for comment in issue.get_comments()
            ]
            reviews = [self._review_context(review) for review in pull.get_reviews()]
            review_comments = [
                self._review_comment_context(comment)
                for comment in pull.get_review_comments()
            ]
            return {
                "number": issue.number,
                "title": title,
                "body": self._text(issue.body, "pull-request body", allow_none=True),
                "url": url,
                "comments": comments,
                "reviews": reviews,
                "review_comments": review_comments,
            }
        except Exception as exc:
            raise WorkflowError(
                f"cannot read pull-request context #{number}: {exc}"
            ) from exc

    @staticmethod
    def _author(value: object) -> object:
        if value is None:
            return None
        login = getattr(value, "login", None)
        if not isinstance(login, str):
            raise WorkflowError("GitHub returned an invalid pull-request author")
        return login

    @staticmethod
    def _text(value: object, field: str, *, allow_none: bool = False) -> str:
        if value is None and allow_none:
            return ""
        if not isinstance(value, str):
            raise WorkflowError(f"GitHub returned an invalid {field}")
        return value

    @classmethod
    def _timestamp(cls, value: object, field: str) -> str:
        isoformat = getattr(value, "isoformat", None)
        if not callable(isoformat):
            raise WorkflowError(f"GitHub returned an invalid {field}")
        timestamp = isoformat()
        if not isinstance(timestamp, str) or not timestamp:
            raise WorkflowError(f"GitHub returned an invalid {field}")
        return timestamp

    @staticmethod
    def _line(value: object) -> int | None:
        if value is None or (isinstance(value, int) and not isinstance(value, bool)):
            return value
        raise WorkflowError("GitHub returned an invalid review comment line")

    @classmethod
    def _comment_context(cls, comment: object) -> dict[str, Any]:
        return {
            "author": cls._author(getattr(comment, "user", None)),
            "body": cls._text(
                getattr(comment, "body", None), "comment body", allow_none=True
            ),
            "createdAt": cls._timestamp(
                getattr(comment, "created_at", None), "comment timestamp"
            ),
            "url": cls._text(getattr(comment, "html_url", None), "comment URL"),
        }

    @classmethod
    def _review_context(cls, review: object) -> dict[str, Any]:
        return {
            "author": cls._author(getattr(review, "user", None)),
            "state": cls._text(getattr(review, "state", None), "review state"),
            "body": cls._text(
                getattr(review, "body", None), "review body", allow_none=True
            ),
            "submittedAt": cls._timestamp(
                getattr(review, "submitted_at", None), "review timestamp"
            ),
            "url": cls._text(getattr(review, "html_url", None), "review URL"),
        }

    @classmethod
    def _review_comment_context(cls, comment: object) -> dict[str, Any]:
        return {
            "author": cls._author(getattr(comment, "user", None)),
            "body": cls._text(
                getattr(comment, "body", None), "review comment body", allow_none=True
            ),
            "path": cls._text(getattr(comment, "path", None), "review comment path"),
            "line": cls._line(getattr(comment, "line", None)),
            "createdAt": cls._timestamp(
                getattr(comment, "created_at", None), "review comment timestamp"
            ),
            "url": cls._text(getattr(comment, "html_url", None), "review comment URL"),
        }

    def close(self) -> None:
        if self.github is not None:
            self.github.close()
        if self.integration is not None:
            self.integration.close()
