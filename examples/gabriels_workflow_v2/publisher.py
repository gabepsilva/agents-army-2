"""Milestone-only GitHub projection for Gabriel's workflow V2."""

from __future__ import annotations

from examples.gabriels_workflow.development_workflow import WorkflowError
from examples.gabriels_workflow.github_app_client import GitHubAppClient


class GitHubPublisher(GitHubAppClient):
    """Publish summaries without using GitHub comments as execution state.

    V2 keeps its execution state in the local checkpoint store, so GitHub
    carries only milestones. Its own marker prefix keeps those milestones
    from being read as V1 stage comments, and keeps a V1 run on the same
    issue from suppressing them.
    """

    marker_prefix = "gdw-v2"
    comment_heading = "GDW V2"

    def create_or_find_pr(
        self,
        *,
        base: str,
        branch: str,
        title: str,
        body: str,
        draft: bool,
    ) -> str:
        """Return the branch's open pull request, opening one only if absent.

        A resumed run whose checkpoint store was discarded would otherwise
        try to open a second pull request for a branch that already has one.
        """

        owner = getattr(getattr(self.repository, "owner", None), "login", None)
        if not isinstance(owner, str) or not owner:
            raise WorkflowError("GitHub did not report the repository owner")
        existing = list(
            self.repository.get_pulls(state="open", base=base, head=f"{owner}:{branch}")
        )
        if len(existing) > 1:
            raise WorkflowError(f"multiple open pull requests use branch {branch}")
        if existing:
            url = getattr(existing[0], "html_url", None)
            if not isinstance(url, str) or not url:
                raise WorkflowError("GitHub returned an invalid existing pull request")
            return url
        return self.create_pr(
            base=base,
            branch=branch,
            title=title,
            body=body,
            draft=draft,
        )
