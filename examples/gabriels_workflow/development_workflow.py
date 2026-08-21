#!/usr/bin/env python3
"""Drive a raw GitHub issue through implementation and into a draft PR.

Agents receive repository context and strict output contracts. This process owns
GitHub, full CI, commits, pushes, workflow state, and all stage transitions.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

GITHUB_TOKEN_NAMES = (
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "GH_ENTERPRISE_TOKEN",
    "GITHUB_ENTERPRISE_TOKEN",
    "GITHUB_APP_ID",
    "GITHUB_APP_PRIVATE_KEY",
)
CI_EVIDENCE_CHARS = 20_000
# The braille block, which every spinner this project's gates print draws from.
SPINNER_FRAME = re.compile(r"^[\u2800-\u28ff]+\s*")
# `make: *** [Makefile:85: mutation] Error 1` — which gate failed, not where.
FAILED_TARGET = re.compile(r"^make: \*\*\* \[[^\]]*: (\S+)\] Error", re.MULTILINE)
# The numbers a gate reports when it refuses: a score that has not moved is a
# repair that achieved nothing, however differently the agent described it.
GATE_VERDICT = re.compile(
    r"^(?:mutation score .*|.*Coverage failure.*|Found \d+ (?:error|diagnostic)s?\.)$",
    re.MULTILINE,
)
DEFAULT_AGENT_TIMEOUT = 3_600
DEFAULT_CI_TIMEOUT = 7_200
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(message)s"
LOG_TAIL_LINES = 40
LOGGER = logging.getLogger("gdw")
AGENT_ROLES = frozenset(
    {
        "expander",
        "griller",
        "specifier",
        "implementer",
        "reviewer-specification",
        "reviewer-quality",
    }
)


def configure_logging(verbose: bool = False) -> None:
    """Send timestamped progress to stderr so stdout stays the result channel.

    Only this workflow's own logger is touched: the root logger is left to
    whatever embeds the example, and a second call replaces the handler rather
    than printing every line twice.
    """

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    LOGGER.handlers.clear()
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.DEBUG if verbose else logging.INFO)


class WorkflowError(RuntimeError):
    """A workflow failure with a concise user-facing message."""


class WorkflowStopped(WorkflowError):
    """A deliberate terminal outcome rather than an infrastructure failure."""


@dataclass(frozen=True)
class CommandResult:
    """The bounded evidence retained from a subprocess."""

    returncode: int
    output: str

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0

    def as_json(self) -> dict:
        return {"returncode": self.returncode, "output": self.output}


class GitHubService(Protocol):
    def issue(self, number: int) -> dict: ...

    def comment_once(
        self, number: int, key: str, title: str, payload: object
    ) -> None: ...

    def create_pr(
        self,
        *,
        base: str,
        branch: str,
        title: str,
        body: str,
        draft: bool,
    ) -> str: ...


class RepositoryService(Protocol):
    def run_ci(self) -> CommandResult: ...

    def commit(self, message: str, base_sha: str) -> None: ...

    def push(self, branch: str) -> None: ...


class AgentService(Protocol):
    def ask(
        self,
        *,
        role: str,
        prompt_name: str,
        schema_name: str,
        values: Mapping[str, str],
        timeout: int = DEFAULT_AGENT_TIMEOUT,
    ) -> dict: ...


class RoleOptions(Protocol):
    @property
    def backend(self) -> str: ...

    @property
    def model(self) -> str | None: ...

    @property
    def reasoning_effort(self) -> str | None: ...


@dataclass(frozen=True)
class _StaticRoleOptions:
    backend: str
    model: str | None = None
    reasoning_effort: str | None = None


@dataclass(frozen=True)
class WorkflowOptions:
    issue_number: int
    base_branch: str
    branch: str
    draft: bool


@dataclass(frozen=True)
class WorkflowServices:
    store: ArtifactStore
    github: GitHubService
    repository: RepositoryService
    agents: AgentService
    role_github: Mapping[str, GitHubService] | None = None


@dataclass(frozen=True)
class Stage:
    key: str
    title: str
    role: str
    prompt: str
    schema: str
    values: Mapping[str, str]


def _json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


def _readable(text: str) -> str:
    """Collapse in-place progress redraws so the real errors survive bounding.

    `make ci` runs mutmut, whose spinner repaints one status line once per
    mutant. Kept verbatim those frames were 369 of 416 lines of a failure this
    workflow captured, and because only the tail is retained they pushed the
    ruff, ty and pytest messages out of the evidence entirely: the agent asked
    to repair a failure could no longer see it.

    splitlines() already breaks each carriage-return repaint onto its own
    line, so a frame differs from the one before it only by its spinner glyph.
    Dropping that glyph leaves consecutive duplicates, which collapse.
    """

    kept: list[str] = []
    for raw in text.splitlines():
        line = SPINNER_FRAME.sub("", raw)
        if kept and kept[-1] == line:
            continue
        kept.append(line)
    return "\n".join(kept)


def _tail(text: str, lines: int = LOG_TAIL_LINES) -> str:
    """The last few lines of long evidence, for a readable failure log."""

    return "\n".join(text.splitlines()[-lines:])


def _ci_signature(result: Mapping[str, object]) -> str:
    """What a CI failure is, stripped of everything that varies between runs.

    `make -j` interleaves its gates differently every time, so two runs of one
    unchanged failure differ by thousands of characters and comparing the
    evidence itself never reports a stall. Which targets failed, and the
    numbers they refused on, do not vary — and when those repeat, the repair
    between them changed nothing that CI can see.
    """

    output = str(result.get("output", ""))
    return _json(
        {
            "returncode": result.get("returncode"),
            "failed_targets": sorted(set(FAILED_TARGET.findall(output))),
            "verdicts": sorted(set(GATE_VERDICT.findall(output))),
        }
    )


def _outcome(result: Mapping[str, object]) -> str:
    """A one-line digest of a structured reply, for the progress log."""

    fields = [
        f"{key}={result[key]}"
        for key in ("decision", "verdict", "status", "needs_another_round")
        if key in result
    ]
    return ", ".join(fields) if fields else "no outcome fields"


def _bounded(text: str, limit: int = CI_EVIDENCE_CHARS) -> str:
    if len(text) <= limit:
        return text
    return f"… output truncated …\n{text[-limit:]}"


def _markdown_label(value: object) -> str:
    return str(value).replace("_", " ").strip().title()


def _markdown_scalar(value: object) -> str:
    if value is None:
        return "_None._"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    rendered = str(value)
    return rendered if rendered.strip() else "_None._"


def _markdown(value: object, heading_level: int = 3) -> str:
    if isinstance(value, Mapping):
        sections = []
        for key, item in value.items():
            heading = "#" * min(heading_level, 6)
            sections.append(
                f"{heading} {_markdown_label(key)}\n\n"
                f"{_markdown(item, heading_level + 1)}"
            )
        return "\n\n".join(sections) if sections else "_None._"
    if isinstance(value, list):
        if not value:
            return "_None._"
        if all(not isinstance(item, (Mapping, list)) for item in value):
            return "\n".join(
                f"- {_markdown_scalar(item).replace(chr(10), chr(10) + '  ')}"
                for item in value
            )
        sections = []
        for index, item in enumerate(value, start=1):
            heading = "#" * min(heading_level, 6)
            sections.append(
                f"{heading} Item {index}\n\n{_markdown(item, heading_level + 1)}"
            )
        return "\n\n".join(sections)
    return _markdown_scalar(value)


def _render_comment(marker: str, title: str, payload: object) -> str:
    return f"{marker}\n## GDW — {title}\n\n{_markdown(payload)}\n"


class ArtifactStore:
    """Durable, atomic checkpoints kept under .git so they are never committed."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.artifacts = root / "artifacts"
        self.metadata_path = root / "workflow.json"

    @property
    def initialized(self) -> bool:
        return self.metadata_path.exists()

    def initialize(self, issue: int, branch: str, base_sha: str) -> None:
        if self.initialized:
            LOGGER.info("state: resuming from %s", self.metadata_path)
            metadata = self._read(self.metadata_path)
            expected = {"issue": issue, "branch": branch}
            actual = {name: metadata.get(name) for name in expected}
            if actual != expected:
                raise WorkflowError(
                    f"workflow state belongs to {actual}, not {expected}"
                )
            return
        LOGGER.info(
            "state: initializing %s for issue #%s on %s at %s",
            self.root,
            issue,
            branch,
            base_sha,
        )
        self._write(
            self.metadata_path,
            {"issue": issue, "branch": branch, "base_sha": base_sha, "pr_url": None},
        )

    @property
    def metadata(self) -> dict:
        if not self.initialized:
            raise WorkflowError("workflow state has not been initialized")
        return self._read(self.metadata_path)

    def has(self, name: str) -> bool:
        return self.artifact_path(name).exists()

    def load(self, name: str) -> dict:
        return self._read(self.artifact_path(name))

    def save(self, name: str, payload: dict) -> None:
        self._write(self.artifact_path(name), payload)

    def record_pr(self, url: str) -> None:
        metadata = self.metadata
        metadata["pr_url"] = url
        self._write(self.metadata_path, metadata)

    def artifact_path(self, name: str) -> Path:
        return self.artifacts / f"{name}.json"

    @staticmethod
    def _read(path: Path) -> dict:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise WorkflowError(f"cannot read workflow state {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise WorkflowError(f"workflow state {path} is not a JSON object")
        return value

    @staticmethod
    def _write(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(_json(payload) + "\n", encoding="utf-8")
        temporary.replace(path)


class GitRepository:
    """Deterministic git and CI operations that do not require model judgment."""

    def __init__(
        self,
        root: Path,
        run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self.root = root
        self._run_process = run

    def prepare(self, base_branch: str, resuming: bool) -> tuple[str, str]:
        branch = self._call("branch", "--show-current").strip()
        if not branch:
            raise WorkflowError("the workflow requires a named git branch")
        if branch == base_branch:
            raise WorkflowError(
                f"refusing to develop directly on protected base branch '{branch}'"
            )
        if not resuming and self._call("status", "--porcelain").strip():
            raise WorkflowError("start the workflow from a clean worktree")
        head = self._call("rev-parse", "HEAD").strip()
        LOGGER.info(
            "git: branch=%s base=%s head=%s resuming=%s",
            branch,
            base_branch,
            head,
            resuming,
        )
        return branch, head

    def run_ci(self, timeout: int = DEFAULT_CI_TIMEOUT) -> CommandResult:
        LOGGER.info("ci: running 'make ci' in %s (timeout %ss)", self.root, timeout)
        started = time.monotonic()
        proc = self._run_process(
            ["make", "ci"],
            cwd=str(self.root),
            capture_output=True,
            text=True,
            check=False,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
        )
        combined = f"{proc.stdout}\n{proc.stderr}".strip()
        evidence = _readable(combined)
        LOGGER.info(
            "ci: 'make ci' exited %s after %.1fs with %s chars of output "
            "(%s after collapsing progress redraws)",
            proc.returncode,
            time.monotonic() - started,
            len(combined),
            len(evidence),
        )
        return CommandResult(proc.returncode, _bounded(evidence))

    def commit(self, message: str, base_sha: str) -> None:
        tracked = self._call("diff", "--no-renames", "--name-only", "-z", "HEAD").split(
            "\0"
        )
        untracked = self._call(
            "ls-files", "--others", "--exclude-standard", "-z"
        ).split("\0")
        changed_paths = [path for path in (*tracked, *untracked) if path]
        LOGGER.info("git: %s path(s) changed since HEAD", len(changed_paths))
        LOGGER.debug("git: changed paths %s", changed_paths)
        if changed_paths:
            self._call("add", "--", *changed_paths)
            self._call("commit", "-m", message)
            LOGGER.info("git: committed %r", message)
        commits = self._call("rev-list", "--count", f"{base_sha}..HEAD").strip()
        LOGGER.info("git: %s commit(s) ahead of %s", commits, base_sha)
        if commits == "0":
            raise WorkflowStopped("implementation produced no commits")

    def push(self, branch: str) -> None:
        LOGGER.info("git: pushing %s to origin", branch)
        self._call("push", "--set-upstream", "origin", branch)
        LOGGER.info("git: pushed %s", branch)

    def _call(self, *args: str) -> str:
        proc = self._run_process(
            ["git", *args],
            cwd=str(self.root),
            capture_output=True,
            text=True,
            check=False,
            stdin=subprocess.DEVNULL,
        )
        if proc.returncode != 0:
            command = " ".join(("git", *args))
            LOGGER.error("git: %s failed: %s", command, proc.stderr.strip())
            raise WorkflowError(f"{command} failed: {proc.stderr.strip()}")
        return proc.stdout


class GitHub:
    """The only component allowed to possess GitHub credentials or call gh."""

    def __init__(
        self,
        root: Path,
        repository: str | None = None,
        executable: Path | None = None,
        run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        discovered = shutil.which("gh") if executable is None else str(executable)
        if discovered is None:
            raise WorkflowError("gh is not installed or is not on PATH")
        self.executable = str(Path(discovered).resolve())
        self.root = root
        self.repository = repository
        self._run_process = run
        self._environment = dict(os.environ if environment is None else environment)
        self._markers: set[str] = set()

    def default_branch(self) -> str:
        payload = self._json_call("repo", "view", "--json", "defaultBranchRef")
        branch = payload.get("defaultBranchRef")
        if not isinstance(branch, dict) or not isinstance(branch.get("name"), str):
            raise WorkflowError("gh repo view did not report a default branch")
        return branch["name"]

    def issue(self, number: int) -> dict:
        LOGGER.info("github: loading issue #%s", number)
        payload = self._json_call(
            "issue",
            "view",
            str(number),
            "--json",
            "number,title,body,comments,url,labels,state",
        )
        comments = payload.get("comments", [])
        if isinstance(comments, list):
            self._markers = {
                line
                for comment in comments
                if isinstance(comment, dict)
                for line in str(comment.get("body", "")).splitlines()
                if line.startswith("<!-- gdw:")
            }
        return payload

    def comment_once(self, number: int, key: str, title: str, payload: object) -> None:
        marker = f"<!-- gdw:{number}:{key} -->"
        if marker in self._markers:
            LOGGER.info("github: comment '%s' already posted, skipping", key)
            return
        LOGGER.info("github: commenting '%s' on issue #%s", key, number)
        self._body_call(
            _render_comment(marker, title, payload),
            "issue",
            "comment",
            str(number),
            "--body-file",
        )
        self._markers.add(marker)

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
            "github: creating %s pull request %s -> %s",
            "draft" if draft else "ready",
            branch,
            base,
        )
        args = ["pr", "create", "--base", base, "--head", branch, "--title", title]
        if draft:
            args.append("--draft")
        return self._body_call(body, *args, "--body-file").strip()

    def _repo_args(self) -> list[str]:
        return [] if self.repository is None else ["--repo", self.repository]

    def _json_call(self, *args: str) -> dict:
        raw = self._call(*args)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise WorkflowError(f"gh returned invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise WorkflowError("gh returned JSON that was not an object")
        return payload

    def _body_call(self, body: str, *args: str) -> str:
        with tempfile.TemporaryDirectory(prefix="gdw-gh-") as directory:
            body_file = Path(directory) / "body.md"
            body_file.write_text(body, encoding="utf-8")
            return self._call(*args, str(body_file))

    def _call(self, *args: str) -> str:
        proc = self._run_process(
            [self.executable, *args, *self._repo_args()],
            cwd=str(self.root),
            env=self._environment,
            capture_output=True,
            text=True,
            check=False,
            stdin=subprocess.DEVNULL,
        )
        if proc.returncode != 0:
            LOGGER.error("gh: %s failed: %s", args[0], proc.stderr.strip())
            raise WorkflowError(f"gh {' '.join(args)} failed: {proc.stderr.strip()}")
        return proc.stdout


class AgentGateway:
    """Structured turns through the public orchestrator CLI."""

    def __init__(
        self,
        *,
        roles: Mapping[str, RoleOptions],
        issue: int,
        state_file: Path,
        example_root: Path,
        run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self.roles = dict(roles)
        self.issue = issue
        self.state_file = state_file
        self.prompts = example_root / "prompts"
        self.validations = example_root / "validations"
        self._run_process = run

    def ask(
        self,
        *,
        role: str,
        prompt_name: str,
        schema_name: str,
        values: Mapping[str, str],
        timeout: int = DEFAULT_AGENT_TIMEOUT,
    ) -> dict:
        prompt = self._prompt(prompt_name, values)
        schema = self.validations / f"{schema_name}.json"
        role_options = self.roles.get(role)
        if role_options is None:
            raise WorkflowError(f"agent role '{role}' is not configured")
        backend = role_options.backend
        model = role_options.model
        reasoning_effort = role_options.reasoning_effort
        agent_name = f"gdw-{self.issue}-{role}"
        LOGGER.info(
            "agent %s: ensuring '%s' (backend=%s model=%s effort=%s)",
            role,
            agent_name,
            backend,
            model,
            reasoning_effort,
        )
        ensure_args = ["ensure", agent_name, "--backend", backend]
        if model is not None:
            ensure_args += ["--model", model]
        if reasoning_effort is not None:
            ensure_args += ["--reasoning-effort", reasoning_effort]
        LOGGER.info(
            "agent %s: turn starting (prompt=%s '%s' of %s chars, schema=%s, "
            "timeout=%ss)",
            role,
            prompt_name,
            schema_name,
            len(prompt),
            schema.name,
            timeout,
        )
        LOGGER.debug("agent %s: prompt sent\n%s", role, prompt)
        started = time.monotonic()
        with self._without_github_access() as environment:
            self._run_cli(ensure_args, environment, timeout=30)
            turn = self._run_cli(
                [
                    "--agent",
                    agent_name,
                    "--validate-schema",
                    str(schema),
                    "--timeout",
                    str(timeout),
                    "--prompt",
                    prompt,
                ],
                environment,
                timeout=timeout + 5,
            )
        LOGGER.info(
            "agent %s: turn finished in %.1fs with %s chars of stdout",
            role,
            time.monotonic() - started,
            len(turn.stdout),
        )
        LOGGER.debug("agent %s: stdout\n%s", role, turn.stdout)
        _header, separator, payload = turn.stdout.partition("\n")
        if not separator:
            raise WorkflowError(f"agent '{role}' returned no structured response")
        try:
            structured = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise WorkflowError(
                f"agent '{role}' returned invalid structured JSON: {exc}"
            ) from exc
        if not isinstance(structured, dict):
            raise WorkflowError(f"agent '{role}' returned no structured response")
        return structured

    def _run_cli(
        self,
        args: list[str],
        environment: Mapping[str, str],
        *,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        LOGGER.debug("orchestrator: invoking '%s' (timeout %ss)", args[0], timeout)
        try:
            result = self._run_process(
                ["orchestrator", *args],
                cwd=str(Path.cwd()),
                env=environment,
                capture_output=True,
                text=True,
                check=False,
                stdin=subprocess.DEVNULL,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            LOGGER.error("orchestrator: '%s' failed to run: %s", args[0], exc)
            raise WorkflowError(f"orchestrator CLI failed: {exc}") from exc
        if result.returncode != 0:
            message = result.stderr.strip() or f"exited {result.returncode}"
            LOGGER.error("orchestrator: '%s' failed: %s", args[0], message)
            raise WorkflowError(f"orchestrator CLI failed: {message}")
        return result

    def _prompt(self, name: str, values: Mapping[str, str]) -> str:
        path = self.prompts / f"{name}.md"
        try:
            prompt = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise WorkflowError(f"cannot read prompt {path}: {exc}") from exc
        for key, value in values.items():
            prompt = prompt.replace(f"{{{{{key}}}}}", value)
        if "{{" in prompt or "}}" in prompt:
            raise WorkflowError(f"prompt {path} has unresolved placeholders")
        return prompt

    @contextmanager
    def _without_github_access(self) -> Iterator[dict[str, str]]:
        original = dict(os.environ)
        with tempfile.TemporaryDirectory(prefix="gdw-agent-") as directory:
            isolation = Path(directory)
            blocker = isolation / "gh"
            blocker.write_text(
                "#!/bin/sh\necho 'gh is owned by the GDW driver' >&2\nexit 126\n",
                encoding="utf-8",
            )
            blocker.chmod(0o700)
            sanitized = dict(original)
            sanitized["PATH"] = f"{isolation}{os.pathsep}{original.get('PATH', '')}"
            sanitized["GH_CONFIG_DIR"] = str(isolation / "empty-gh-config")
            sanitized["AGENTS_ARMY_STATE_FILE"] = str(self.state_file)
            for name in GITHUB_TOKEN_NAMES:
                sanitized.pop(name, None)
            yield sanitized


class DevelopmentWorkflow:
    """The explicit state machine from issue expansion to pull request."""

    def __init__(
        self,
        options: WorkflowOptions,
        services: WorkflowServices,
    ) -> None:
        self.issue_number = options.issue_number
        self.base_branch = options.base_branch
        self.branch = options.branch
        self.store = services.store
        self.github = services.github
        self.repository = services.repository
        self.agents = services.agents
        self.role_github = (
            {} if services.role_github is None else dict(services.role_github)
        )
        self.draft = options.draft

    def run(self) -> str:
        existing_url = self.completed_url()
        if existing_url is not None:
            return existing_url
        LOGGER.info(
            "workflow: issue #%s, branch %s onto %s, draft=%s",
            self.issue_number,
            self.branch,
            self.base_branch,
            self.draft,
        )
        issue_context = self.load_issue()
        expansion = self.clarify(issue_context)
        specification = self.specify(issue_context, expansion)
        self.implement(specification)
        ci_summary = self.stabilize(specification)
        final_reviews = self.review(specification, ci_summary)
        return self.publish(specification, final_reviews)

    def completed_url(self) -> str | None:
        existing_url = self.store.metadata.get("pr_url")
        if isinstance(existing_url, str) and existing_url:
            LOGGER.info("workflow: already completed, pull request %s", existing_url)
            return existing_url
        return None

    def load_issue(self) -> dict:
        issue = self.github.issue(self.issue_number)
        comments = issue.get("comments", [])
        latest_comments = [
            comment
            for comment in comments
            if isinstance(comment, Mapping)
            and isinstance(comment.get("body"), str)
            and "<!-- gdw:" not in comment["body"]
        ][-5:]
        LOGGER.info(
            "workflow: issue #%s %r loaded with %s of %s comment(s) forwarded",
            issue.get("number"),
            issue.get("title"),
            len(latest_comments),
            len(comments) if isinstance(comments, list) else 0,
        )
        return {
            "initial": {
                "number": issue.get("number"),
                "title": issue.get("title"),
                "body": issue.get("body"),
                "latest_comments": latest_comments,
            },
            "latest_comments": latest_comments,
        }

    def clarify(self, issue_context: dict) -> dict:
        return self._reach_agreement(issue_context)

    def specify(self, issue_context: dict, expansion: dict) -> dict:
        return self._stage(
            Stage(
                "specification",
                "Specification",
                "specifier",
                "specify",
                "specification",
                {
                    "LATEST_COMMENTS_JSON": _json(issue_context["latest_comments"]),
                    "EXPANSION_JSON": _json(expansion),
                },
            )
        )

    def implement(self, specification: dict) -> dict:
        implementation = self._stage(
            Stage(
                "implementation",
                "Implementation report",
                "implementer",
                "implement",
                "implementation",
                {"SPECIFICATION_JSON": _json(specification)},
            )
        )
        self._require_complete(implementation, "implementation", "implementation")
        return implementation

    def stabilize(self, specification: dict) -> dict:
        return self._ci_until_green("implementation", specification)

    def review(self, specification: dict, ci_summary: dict) -> dict[str, dict]:
        return self._review_until_approved(specification, ci_summary)

    def publish(self, specification: dict, final_reviews: dict[str, dict]) -> str:
        base_sha = str(self.store.metadata["base_sha"])
        commit_title = str(specification["title"]).replace("\n", " ")[:72]
        self.repository.commit(
            f"Implement #{self.issue_number}: {commit_title}", base_sha
        )
        self.repository.push(self.branch)
        body = self._pr_body(specification, final_reviews)
        url = self.github.create_pr(
            base=self.base_branch,
            branch=self.branch,
            title=commit_title,
            body=body,
            draft=self.draft,
        )
        if not url:
            raise WorkflowError("GitHub returned an empty pull-request URL")
        LOGGER.info("workflow: pull request created at %s", url)
        self.store.record_pr(url)
        self.github.comment_once(
            self.issue_number,
            "pull-request",
            "Pull request created",
            {"url": url, "draft": self.draft},
        )
        return url

    def _reach_agreement(self, issue_context: dict) -> dict:
        expansion: dict | None = None
        grill: dict | None = None
        previous_unresolved: str | None = None
        round_number = 1
        while True:
            LOGGER.info("clarification: round %s", round_number)
            if expansion is None:
                prompt = "expand"
                values = {"ISSUE_CONTEXT_JSON": _json(issue_context["initial"])}
            else:
                prompt = "revise"
                values = {
                    "LATEST_COMMENTS_JSON": _json(issue_context["latest_comments"]),
                    "EXPANSION_JSON": _json(expansion),
                    "GRILL_JSON": _json(grill),
                }
            expansion = self._stage(
                Stage(
                    f"expansion-{round_number}",
                    f"Expansion round {round_number}",
                    "expander",
                    prompt,
                    "expansion",
                    values,
                )
            )
            if expansion["decision"] == "stop":
                raise WorkflowStopped(str(expansion["summary"]))
            grill = self._stage(
                Stage(
                    f"grill-{round_number}",
                    f"Ambiguity review round {round_number}",
                    "griller",
                    "grill",
                    "grill",
                    {
                        "LATEST_COMMENTS_JSON": _json(issue_context["latest_comments"]),
                        "EXPANSION_JSON": _json(expansion),
                    },
                )
            )
            if grill["verdict"] == "reject":
                raise WorkflowStopped(str(grill["summary"]))
            if (
                not expansion["needs_another_round"]
                and not grill["needs_another_round"]
                and grill["verdict"] == "ready"
            ):
                LOGGER.info("clarification: converged after %s round(s)", round_number)
                return expansion
            unresolved = _json(
                {
                    "expander_needs_another_round": expansion["needs_another_round"],
                    "open_questions": expansion["open_questions"],
                    "griller_needs_another_round": grill["needs_another_round"],
                    "questions": grill["questions"],
                    "required_changes": grill["required_changes"],
                }
            )
            self._require_progress(previous_unresolved, unresolved, "clarification")
            previous_unresolved = unresolved
            round_number += 1

    def _ci_until_green(self, prefix: str, specification: dict) -> dict:
        previous_unresolved: str | None = None
        attempt = 1
        while True:
            key = f"ci-{prefix}-{attempt}"
            LOGGER.info("ci: %s attempt %s", prefix, attempt)
            ci = self.repository.run_ci()
            result = ci.as_json()
            self.store.save(key, result)
            self.github.comment_once(
                self.issue_number,
                key,
                f"CI result for {prefix}, attempt {attempt}",
                result,
            )
            if result["returncode"] == 0:
                LOGGER.info("ci: %s green on attempt %s", prefix, attempt)
                return result
            LOGGER.warning(
                "ci: %s failed on attempt %s (exit %s); last %s lines:\n%s",
                prefix,
                attempt,
                result["returncode"],
                LOG_TAIL_LINES,
                _tail(str(result["output"])),
            )
            repair = self._stage(
                Stage(
                    f"repair-{prefix}-{attempt}",
                    f"Repair report for {prefix}, attempt {attempt}",
                    "implementer",
                    "repair",
                    "implementation",
                    {
                        "SPECIFICATION_JSON": _json(specification),
                        "FAILURE_EVIDENCE": _json(result),
                    },
                )
            )
            self._require_complete(repair, "CI repair", f"repair-{prefix}-{attempt}")
            unresolved = _ci_signature(result)
            LOGGER.info("ci: %s unresolved as %s", prefix, unresolved)
            self._require_progress(previous_unresolved, unresolved, "CI repair")
            previous_unresolved = unresolved
            attempt += 1

    def _review_until_approved(
        self, specification: dict, ci_summary: dict
    ) -> dict[str, dict]:
        final: dict[str, dict] = {}
        previous_unresolved: str | None = None
        round_number = 1
        while True:
            LOGGER.info("review: round %s", round_number)
            final = {}
            for kind in ("specification", "quality"):
                final[kind] = self._stage(
                    Stage(
                        f"review-{round_number}-{kind}",
                        f"{kind.title()} review round {round_number}",
                        f"reviewer-{kind}",
                        "review",
                        "review",
                        {
                            "REVIEW_KIND": kind,
                            "SPECIFICATION_JSON": _json(specification),
                            "CI_SUMMARY": _json(ci_summary),
                        },
                    )
                )
            if all(
                review["verdict"] == "approve" and not review["needs_another_round"]
                for review in final.values()
            ):
                LOGGER.info("review: approved after %s round(s)", round_number)
                return final
            unresolved = _json(
                {
                    kind: {
                        "verdict": review["verdict"],
                        "needs_another_round": review["needs_another_round"],
                        "findings": review["findings"],
                    }
                    for kind, review in final.items()
                }
            )
            self._require_progress(previous_unresolved, unresolved, "review")
            previous_unresolved = unresolved
            repair = self._stage(
                Stage(
                    f"review-repair-{round_number}",
                    f"Review repair round {round_number}",
                    "implementer",
                    "repair",
                    "implementation",
                    {
                        "SPECIFICATION_JSON": _json(specification),
                        "FAILURE_EVIDENCE": _json(final),
                    },
                )
            )
            self._require_complete(
                repair, "review repair", f"review-repair-{round_number}"
            )
            ci_summary = self._ci_until_green(f"review-{round_number}", specification)
            round_number += 1

    def _stage(self, stage: Stage) -> dict:
        if self.store.has(stage.key):
            cached = self.store.load(stage.key)
            LOGGER.info(
                "stage %s: reusing checkpoint from %s (%s)",
                stage.key,
                stage.role,
                _outcome(cached),
            )
            return cached
        LOGGER.info("stage %s: asking %s", stage.key, stage.role)
        started = time.monotonic()
        result = self.agents.ask(
            role=stage.role,
            prompt_name=stage.prompt,
            schema_name=stage.schema,
            values=stage.values,
        )
        LOGGER.info(
            "stage %s: %s answered in %.1fs (%s)",
            stage.key,
            stage.role,
            time.monotonic() - started,
            _outcome(result),
        )
        LOGGER.debug("stage %s: reply\n%s", stage.key, _json(result))
        self.store.save(stage.key, result)
        role_github = self.role_github.get(stage.role, self.github)
        role_github.comment_once(self.issue_number, stage.key, stage.title, result)
        return result

    def _require_complete(self, result: dict, stage: str, key: str) -> None:
        if result["status"] != "complete":
            blockers = "; ".join(str(item) for item in result["blockers"])
            # The blocked reply is already checkpointed, so resuming replays it
            # rather than asking again. Say which file to remove to retry.
            LOGGER.error(
                "stage %s reported itself blocked; delete %s to ask again",
                key,
                self.store.artifact_path(key),
            )
            raise WorkflowStopped(f"{stage} blocked: {blockers}")

    @staticmethod
    def _require_progress(
        previous_unresolved: str | None, unresolved: str, process: str
    ) -> None:
        LOGGER.debug("%s: unresolved state\n%s", process, unresolved)
        if unresolved == previous_unresolved:
            raise WorkflowStopped(f"{process} stalled with the same unresolved state")

    def _pr_body(self, specification: dict, reviews: dict[str, dict]) -> str:
        return (
            f"Closes #{self.issue_number}\n\n"
            "## Validated specification\n\n"
            f"{_markdown(specification)}\n\n"
            "## Final independent reviews\n\n"
            f"{_markdown(reviews)}\n\n"
            "Generated by Gabriel's development workflow. Agents could inspect "
            "and edit the repository; the driver owned GitHub, CI, commits, and push.\n"
        )


def _positive(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected 1 or more, got {value}")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Expand a GitHub issue, implement it, and open a draft PR."
    )
    parser.add_argument("issue", type=_positive)
    parser.add_argument("--repo", help="GitHub OWNER/REPO; defaults to this checkout")
    parser.add_argument(
        "--backend", default="claude", choices=("claude", "codex", "grok")
    )
    parser.add_argument("--base", help="PR base branch; defaults to repository default")
    parser.add_argument(
        "--ready", action="store_true", help="open a ready PR instead of a draft"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="log every prompt, reply, and subprocess at DEBUG level",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    opts = _parser().parse_args(argv)
    configure_logging(opts.verbose)
    root = Path.cwd().resolve()
    example_root = Path(__file__).resolve().parent
    github = GitHub(root, opts.repo)
    base_branch = opts.base or github.default_branch()
    store = ArtifactStore(root / ".git" / "gdw" / f"issue-{opts.issue}")
    repository = GitRepository(root)
    branch, base_sha = repository.prepare(base_branch, store.initialized)
    store.initialize(opts.issue, branch, base_sha)
    agents = AgentGateway(
        roles={role: _StaticRoleOptions(opts.backend) for role in AGENT_ROLES},
        issue=opts.issue,
        state_file=store.root / "agents.json",
        example_root=example_root,
    )
    workflow = DevelopmentWorkflow(
        WorkflowOptions(
            opts.issue,
            base_branch,
            branch,
            not opts.ready,
        ),
        WorkflowServices(store, github, repository, agents),
    )
    try:
        url = workflow.run()
    except WorkflowError as exc:
        LOGGER.error("workflow stopped: %s", exc)
        LOGGER.error(
            "workflow: state kept in %s; rerun the same command to resume",
            store.root,
        )
        print(f"workflow stopped: {exc}", file=sys.stderr)
        return 1
    print(url)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
