"""GitHub App communication for the linear development-workflow example."""

from __future__ import annotations

import logging
from typing import Any

from github import Auth, Github, GithubIntegration
from github.Repository import Repository

LOGGER = logging.getLogger("gdw")


class GitHubAppClient:
    """Read and update one repository as its installed GitHub App."""

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
        """Union gdw markers from comments on this issue or pull request."""

        issue = self.repository.get_issue(number)
        self.markers |= {
            line
            for comment in issue.get_comments()
            for line in (comment.body or "").splitlines()
            if line.startswith("<!-- gdw:")
        }

    def comment(self, number: int, body: str) -> None:
        self.repository.get_issue(number).create_comment(body)

    def comment_once(
        self,
        number: int,
        key: str,
        title: str,
        payload: object,
    ) -> None:
        marker = f"<!-- gdw:{number}:{key} -->"
        if marker in self.markers:
            LOGGER.info("github-app: comment '%s' already posted, skipping", key)
            return
        from examples.gabriels_workflow.development_workflow import _render_comment

        LOGGER.info("github-app: commenting '%s' on #%s", key, number)
        self.comment(number, _render_comment(marker, title, payload))
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

    def close(self) -> None:
        if self.github is not None:
            self.github.close()
        if self.integration is not None:
            self.integration.close()
