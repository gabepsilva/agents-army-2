"""Behavior tests for the raw-issue-to-PR workflow example."""

from __future__ import annotations

import json
import os
import subprocess
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples import gabriel_development_workflow as gdw
from orchestrator.schema import load_schema


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


def _expansion(decision: str = "proceed") -> dict:
    return {
        "decision": decision,
        "summary": "proposal",
        "current_state": ["current"],
        "proposed_changes": ["change"],
        "out_of_scope": [],
        "risks": [],
        "open_questions": [],
    }


def _grill(verdict: str = "ready") -> dict:
    return {
        "verdict": verdict,
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


def _review(verdict: str = "approve") -> dict:
    return {
        "verdict": verdict,
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

    def issue(self, number: int) -> dict:
        return {"number": number, "title": "Raw issue", "body": "Please build it"}

    def comment_once(self, number: int, key: str, title: str, payload: object) -> None:
        self.comments.append((number, key, title, payload))

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
        "clarification_rounds": 2,
        "repair_rounds": 1,
        "review_rounds": 2,
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
            settings["clarification_rounds"],
            settings["repair_rounds"],
            settings["review_rounds"],
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


def test_agent_gateway_validates_prompt_and_removes_github_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict = {}

    class FakeOrchestrator:
        def __init__(self, state_file: Path) -> None:
            observed["state_file"] = state_file
            self.agent = SimpleNamespace(
                name="gdw-3-expander", backend=SimpleNamespace(name="codex")
            )

        def ensure(self, name: str, backend: str):
            observed["ensure"] = (name, backend)
            return self.agent, True

        def talk(self, name: str, prompt: str, **kwargs):
            observed["talk"] = (name, prompt, kwargs)
            observed["tokens"] = [os.environ.get(key) for key in gdw.GITHUB_TOKEN_NAMES]
            observed["config"] = os.environ["GH_CONFIG_DIR"]
            blocker = Path(os.environ["PATH"].split(os.pathsep)[0]) / "gh"
            observed["blocker"] = blocker.read_text(encoding="utf-8")
            return SimpleNamespace(structured=_expansion())

    monkeypatch.setenv("GH_TOKEN", "secret")
    original_path = os.environ["PATH"]
    root = Path(gdw.__file__).parent
    gateway = gdw.AgentGateway(
        backend="codex",
        issue=3,
        state_file=tmp_path / "agents.json",
        example_root=root,
        orchestrator_factory=FakeOrchestrator,
    )
    result = gateway.ask(
        role="expander",
        prompt_name="expand",
        schema_name="expansion",
        values={"ISSUE_JSON": "{}"},
        timeout=17,
    )
    assert result == _expansion()
    assert observed["ensure"] == ("gdw-3-expander", "codex")
    assert observed["tokens"] == [None, None, None, None]
    assert observed["config"].endswith("empty-gh-config")
    assert "owned by the GDW driver" in observed["blocker"]
    assert observed["talk"][2]["timeout"] == 17
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

    class FakeOrchestrator:
        def __init__(self, _state_file: Path) -> None:
            self.agent = SimpleNamespace(
                name="gdw-1-role", backend=SimpleNamespace(name="claude")
            )

        def ensure(self, _name: str, _backend: str):
            return self.agent, False

        def talk(self, *_args, **_kwargs):
            return SimpleNamespace(structured=None)

    gateway = gdw.AgentGateway(
        backend="codex",
        issue=1,
        state_file=tmp_path / "state.json",
        example_root=example_root,
        orchestrator_factory=FakeOrchestrator,
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
    with pytest.raises(gdw.WorkflowError, match="already uses backend"):
        gateway.ask(role="role", prompt_name="ok", schema_name="expansion", values={})
    gateway.backend = "claude"
    with pytest.raises(gdw.WorkflowError, match="no structured response"):
        gateway.ask(role="role", prompt_name="ok", schema_name="expansion", values={})


def test_all_workflow_schemas_are_strict_and_prompts_resolve(tmp_path: Path) -> None:
    example_root = Path(gdw.__file__).parent
    schema_names = ("expansion", "grill", "specification", "implementation", "review")
    for name in schema_names:
        schema = load_schema(example_root / "validations" / f"{name}.json")
        assert schema.path.is_absolute()

    gateway = gdw.AgentGateway(
        backend="codex",
        issue=1,
        state_file=tmp_path / "agents.json",
        example_root=example_root,
    )
    values = {
        "ISSUE_JSON": "{}",
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
        assert "use `gh`" in prompt


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


@pytest.mark.parametrize(
    ("replies", "options", "message"),
    [
        ([_expansion("stop")], {}, "proposal"),
        ([_expansion(), _grill("reject")], {}, "review"),
        (
            [_expansion(), _grill("revise")],
            {"clarification_rounds": 1},
            "clarification",
        ),
        (
            [_expansion(), _grill(), _specification(), _implementation("blocked")],
            {},
            "implementation blocked",
        ),
        (
            [_expansion(), _grill(), _specification(), _implementation()],
            {
                "repair_rounds": 0,
                "repository": FakeRepository([gdw.CommandResult(1, "bad")]),
            },
            "CI did not pass",
        ),
        (
            [
                _expansion(),
                _grill(),
                _specification(),
                _implementation(),
                _review("changes_requested"),
                _review(),
            ],
            {"review_rounds": 1},
            "review rounds exhausted",
        ),
    ],
)
def test_workflow_deliberate_stop_conditions(
    tmp_path: Path, replies: list[dict], options: dict, message: str
) -> None:
    workflow, _github, _repository, _agents = _workflow(tmp_path, replies, options)
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
    with pytest.raises(gdw.WorkflowError, match="empty URL"):
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
    assert gdw._nonnegative("0") == 0
    with pytest.raises(Exception, match="expected 0 or more"):
        gdw._nonnegative("-1")
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
