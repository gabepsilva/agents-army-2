"""Milestone-only GitHub projection for Gabriel's workflow V2."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from examples.gabriels_workflow.development_workflow import WorkflowError
from examples.gabriels_workflow.github_app_client import GitHubAppClient

LOGGER = logging.getLogger("gdw-v2")
CHECK_CONCLUSIONS = frozenset({"success", "failure", "neutral"})


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

    def publish_checks(
        self, head_sha: str, ledger: Sequence[Mapping[str, Any]]
    ) -> None:
        """Publish one check run per stage, so the Checks tab lists the fleet.

        Best effort. The ledger comment carries the same record, so an App
        without `checks:write` — or a GitHub that rejects one row — loses a
        convenience, not evidence, and must not fail a run that has already
        committed, pushed, and opened its pull request.
        """

        for entry in ledger:
            try:
                self._check_run(head_sha, entry)
            except Exception as exc:
                LOGGER.warning(
                    "github-app: check run for '%s' not published: %s",
                    entry.get("stage"),
                    exc,
                )
                return

    def _check_run(self, head_sha: str, entry: Mapping[str, Any]) -> None:
        started = self._started(entry)
        duration = entry.get("duration_seconds")
        completed = started + timedelta(
            seconds=float(duration) if isinstance(duration, int | float) else 0.0
        )
        conclusion = entry.get("conclusion")
        role = entry.get("role", "unknown")
        self.repository.create_check_run(
            name=f"gdw-v2 / {entry.get('stage')}",
            head_sha=head_sha,
            status="completed",
            conclusion=(conclusion if conclusion in CHECK_CONCLUSIONS else "neutral"),
            started_at=started,
            completed_at=completed,
            output={
                "title": f"{role} - {entry.get('outcome') or 'no outcome fields'}",
                "summary": self._check_summary(entry),
            },
        )

    @staticmethod
    def _started(entry: Mapping[str, Any]) -> datetime:
        raw = entry.get("started_at")
        if isinstance(raw, str):
            try:
                return datetime.fromisoformat(raw)
            except ValueError:
                pass
        return datetime.now(UTC)

    @staticmethod
    def _check_summary(entry: Mapping[str, Any]) -> str:
        def field(value: object) -> str:
            text = str(value).strip() if value is not None else ""
            return f"`{text}`" if text else "_unset_"

        skills = ", ".join(entry.get("skills") or [])
        duration = entry.get("duration_seconds")
        return (
            f"backend: {field(entry.get('backend'))}  \n"
            f"model: {field(entry.get('model'))}  \n"
            f"reasoning_effort: {field(entry.get('reasoning_effort'))}  \n"
            f"task_duration: {f'`{duration}s`' if duration is not None else '_unset_'}"
            "  \n"
            f"skills: {field(skills) if skills else '_none_'}  \n"
            f"source: {field(entry.get('source'))}\n\n"
            f"{entry.get('summary') or '_no summary recorded_'}\n"
        )
