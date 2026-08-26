"""Unit tests for agent backend interface, implementations, and orchestrator."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import tomllib
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

import orchestrator
from backends.base import (
    DEFAULT_TURN_TIMEOUT,
    AgentBackend,
    OutputSchema,
    TurnError,
    TurnResult,
    describe_command,
    json_objects,
    reply_text,
    stdout_for_error,
    structured_reply,
)
from backends.claude import (
    OPT_IN_REQUIRED_REASON,
    PERMISSION_MODE,
    ClaudeBackend,
    ClaudeTurnError,
    parse_claude_stdout,
)
from backends.claude import SCHEMA_FLAG as CLAUDE_SCHEMA_FLAG
from backends.codex import SCHEMA_FLAG as CODEX_SCHEMA_FLAG
from backends.codex import CodexBackend, CodexTurnError
from backends.grok import (
    ALWAYS_APPROVE_FLAG,
    PROMPT_FLAG,
    GrokBackend,
    GrokTurnError,
    parse_grok_stdout,
)
from backends.grok import SCHEMA_FLAG as GROK_SCHEMA_FLAG
from backends.opencode import OpenCodeBackend, OpenCodeTurnError
from backends.registry import (
    UnknownBackendError,
    get_backend,
    list_backends,
    register_backend,
)
from orchestrator import (
    Agent,
    Orchestrator,
    main,
)
from orchestrator import (
    cmd_create as _cmd_create,
)
from orchestrator import (
    cmd_delete as _cmd_delete,
)
from orchestrator import (
    cmd_list as _cmd_list,
)
from orchestrator import (
    cmd_talk as _cmd_talk,
)
from orchestrator.schema import compose_schema_prompt


def _talk_options(argv: list[str]) -> argparse.Namespace:
    separator = argv.index("--") if "--" in argv else None
    head = argv if separator is None else argv[:separator]
    tail = [] if separator is None else argv[separator + 1 :]
    options = orchestrator._build_parser().parse_args(head)
    orchestrator._resolve_talk_prompt(options, tail, separator is not None)
    return options


def _options(argv: list[str]) -> argparse.Namespace:
    return orchestrator._build_parser().parse_args(argv)


def run_version(argv: list[str] | None = None) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"] if argv is None else argv)
    assert excinfo.value.code == 0


def _flock_is_held(path: Path) -> bool:
    """Is someone holding this lock file?

    flock is owned by the open file description, not the process, so a second
    handle here contends with the orchestrator's exactly as another process
    would — the lock this asks about is the real one, not a stand-in.
    """
    with path.open("a+", encoding="utf-8") as probe:
        try:
            fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(probe.fileno(), fcntl.LOCK_UN)
        return False


# What each tool prints for `--version` on a machine that has all of them.
# The spellings are the real ones: uv leads with its own name, jq glues its
# name on with a hyphen, claude prints a bare number, and codex names the
# package rather than the command.
ALL_TOOLS_PRESENT = {
    "uv": "uv 0.4.18",
    "claude": "2.1.234 (Claude Code)",
    "codex": "codex-cli 0.147.0",
    "grok": "grok 1.0.5",
    "opencode": "1.18.21",
    "jq": "jq-1.7",
}


def _running_python() -> str:
    """The interpreter version the report is expected to name."""
    return ".".join(str(part) for part in sys.version_info[:3])


def _fake_dependency_env(
    monkeypatch: pytest.MonkeyPatch, versions: dict[str, str]
) -> list[list[str]]:
    """Stand in for PATH and for every `<tool> --version` process.

    Only the tools named in `versions` are on PATH; the rest are absent. The
    returned list records each command actually run, so a test can assert on
    the invocation and not merely that something was spawned.
    """
    calls: list[list[str]] = []

    def which(tool: str) -> str | None:
        return f"/usr/bin/{tool}" if tool in versions else None

    def run(args, **kwargs):
        calls.append(args)
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["check"] is False
        assert kwargs["timeout"] == 5
        assert kwargs["stdin"] == subprocess.DEVNULL
        return subprocess.CompletedProcess(args, 0, stdout=versions[args[0]], stderr="")

    monkeypatch.setattr(orchestrator.shutil, "which", which)
    monkeypatch.setattr(orchestrator.subprocess, "run", run)
    return calls


def _messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Log lines are asserted verbatim: a wrong one is a wrong diagnostic."""
    return [record.getMessage() for record in caplog.records]


def _reported_seconds(message: str, pattern: str) -> float:
    """Return the duration a log line claims, so an implausible one fails.

    Matching the shape of the number is not enough: computing the elapsed time
    with the wrong sign still prints a well-formed float. Only its magnitude —
    process uptime rather than a turn duration — distinguishes the two.
    """
    match = re.fullmatch(pattern, message)
    assert match is not None, f"unexpected log line: {message!r}"
    return float(match.group(1))


class EchoBackend(AgentBackend):
    """A backend that answers without a CLI, for tests about everything else."""

    @property
    def name(self) -> str:
        return "echo"

    def run_turn(
        self,
        prompt: str,
        session_id: str | None,
        cwd: Path,
        timeout: int = DEFAULT_TURN_TIMEOUT,
        schema: OutputSchema | None = None,
    ) -> TurnResult:
        return TurnResult(session_id="echo-sid", reply=f"echo:{prompt}", raw="")


@pytest.fixture(autouse=True)
def register_echo_backend() -> None:
    """Registered for every test, not just the class that introduced it.

    The registry is module-level state, so a class relying on another class
    having registered it first passes or fails on test ordering — which xdist
    is free to change.
    """
    register_backend("echo", EchoBackend)


def _gate_backend(
    name: str, entered: threading.Event, release: threading.Event
) -> type[AgentBackend]:
    """A backend class whose turn signals `entered`, then blocks on `release`.

    A factory rather than one shared class: each caller closes over its own
    pair of events, and `.name` must equal the string it gets registered
    under — `_persist`/`_reload` round-trip a spawned agent's backend through
    `agent.backend.name`, not through the registry key `spawn` was called
    with, so the two have to match.
    """

    class Gate(AgentBackend):
        @property
        def name(self) -> str:
            return name

        def run_turn(
            self,
            prompt: str,
            session_id: str | None,
            cwd: Path,
            timeout: int = DEFAULT_TURN_TIMEOUT,
            schema: OutputSchema | None = None,
        ) -> TurnResult:
            entered.set()
            release.wait(timeout=5)
            return TurnResult(session_id="s1", reply="ok", raw="")

    return Gate


def _pause_first_flock(
    monkeypatch: pytest.MonkeyPatch, thread_name: str
) -> tuple[threading.Event, threading.Event]:
    """Pause `thread_name`'s first `fcntl.flock()` call just before it runs.

    Drives a real `open()`-then-`flock()` TOCTOU window on demand: the
    caller can act between a thread's `_flock` opening a path and that same
    call's `flock()` actually being granted. Returns `(opened, proceed)` —
    `opened` fires once the call is paused there, and the caller sets
    `proceed` to let it continue. Only the first call is paused, so the
    same thread's later `flock()` calls (e.g. the matching unlock) run
    unpatched.
    """
    opened = threading.Event()
    proceed = threading.Event()
    real_flock = fcntl.flock
    paused = False

    def patched(fd: int, op: int) -> None:
        nonlocal paused
        if threading.current_thread().name == thread_name and not paused:
            paused = True
            opened.set()
            proceed.wait(timeout=5)
        real_flock(fd, op)

    monkeypatch.setattr(fcntl, "flock", patched)
    return opened, proceed


class _OverlapState:
    """Results from `_overlap_recorder`, read after every thread joins."""

    def __init__(self) -> None:
        self.overlapped = False
        self.inodes: dict[str, int] = {}


def _overlap_recorder(
    lock_path: Path,
) -> tuple[Callable[..., None], _OverlapState]:
    """Build a `record(who, entered=None)` that marks `who` active for a
    beat, capturing which inode `lock_path` currently names.

    Shared by the two concurrency tests below, each of which calls `record`
    from more than one thread contending for the same agent lock:
    `state.overlapped` is true if two callers were ever active at once, and
    `state.inodes[who]` is what each one saw. A closure rather than a class
    with one method, so each test gets its own `active`/guard/`inodes`
    without instantiating anything.
    """
    active: list[str] = []
    active_guard = threading.Lock()
    state = _OverlapState()

    def record(who: str, entered: threading.Event | None = None) -> None:
        with active_guard:
            active.append(who)
            if len(active) > 1:
                state.overlapped = True
        if entered is not None:
            entered.set()
        state.inodes[who] = os.stat(lock_path).st_ino
        time.sleep(0.05)
        with active_guard:
            active.remove(who)

    return record, state


def _assert_subprocess_kwargs(
    kwargs: dict,
    cwd: Path,
    expected_stdin: object = subprocess.DEVNULL,
    expected_input: str | None = None,
) -> None:
    """Every backend must run its subprocess the same disciplined way."""
    assert kwargs["cwd"] == str(cwd)
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert kwargs["check"] is False
    assert kwargs["timeout"] == 3600
    # Not a detail: a CLI whose stdin is an inherited pipe rather than a tty
    # blocks until it is killed. `codex exec "reply ok" --json` under a pipe
    # returns nothing after 25s and exits 124, and claude and grok are given
    # no chance to do the same. Asserted for every backend, in the one helper
    # every backend test already calls, so a new backend cannot skip it.
    if expected_input is None:
        assert kwargs["stdin"] == expected_stdin
    else:
        assert kwargs["input"] == expected_input


def _completed(returncode: int, stdout: str, stderr: str = "") -> Callable:
    """A `subprocess.run` stand-in for a test that only cares what came back."""

    def run(args, **kwargs):
        return subprocess.CompletedProcess(
            args, returncode, stdout=stdout, stderr=stderr
        )

    return run


# A schema as the adapters receive it: already loaded, in both spellings. The
# text is what claude and grok take inline; the path is what codex is handed.
SCHEMA = OutputSchema(
    text='{"type":"object","additionalProperties":false,"properties":{}}',
    path=Path("/schemas/reply.json"),
)


# The same schema as the orchestrator loads it from a file that declares its
# dialect. Its keys are deliberately unsorted, so a re-serialisation that lost
# the canonical ordering would be visible.
DIALECT_SCHEMA = OutputSchema(
    text=(
        '{"type":"object","$schema":"https://json-schema.org/draft/2020-12/schema"'
        ',"additionalProperties":false,"properties":{}}'
    ),
    path=Path("/schemas/reply.json"),
)


class TestAgentBackendInterface:
    def test_claude_name(self) -> None:
        assert ClaudeBackend().name == "claude"

    def test_codex_name(self) -> None:
        assert CodexBackend().name == "codex"

    def test_grok_name(self) -> None:
        assert GrokBackend().name == "grok"

    def test_opencode_name(self) -> None:
        assert OpenCodeBackend().name == "opencode"
        assert "opencode" in list_backends()

    def test_backend_turn_errors_share_the_orchestrator_type(self) -> None:
        """cmd_talk catches TurnError, not a per-CLI tuple that grows."""
        assert issubclass(ClaudeTurnError, TurnError)
        assert issubclass(CodexTurnError, TurnError)
        assert issubclass(GrokTurnError, TurnError)
        assert issubclass(OpenCodeTurnError, TurnError)

    def test_schema_enforcement_defaults_and_opencode_override(self) -> None:
        assert ClaudeBackend.enforces_schema is True
        assert CodexBackend.enforces_schema is True
        assert GrokBackend.enforces_schema is True
        assert OpenCodeBackend.enforces_schema is False

    def test_get_backend_resolves_grok(self) -> None:
        backend = get_backend("grok", model="grok-test", reasoning_effort="high")
        assert isinstance(backend, GrokBackend)
        assert backend.name == "grok"
        assert backend.model == "grok-test"
        assert backend.reasoning_effort == "high"

    def test_custom_backend_registration(self, tmp_path: Path) -> None:
        class CustomBackend(AgentBackend):
            @property
            def name(self) -> str:
                return "custom"

            def run_turn(
                self,
                prompt: str,
                session_id: str | None,
                cwd: Path,
                timeout: int = DEFAULT_TURN_TIMEOUT,
                schema: OutputSchema | None = None,
            ) -> TurnResult:
                return TurnResult(session_id="custom-sid", reply=prompt, raw="")

        register_backend("custom", CustomBackend)
        assert "custom" in list_backends()

        backend = get_backend("custom")
        assert isinstance(backend, CustomBackend)
        assert backend.run_turn("hi", None, tmp_path).reply == "hi"

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend") as excinfo:
            get_backend("nonexistent")
        assert str(excinfo.value) == (
            f"Unknown backend 'nonexistent'. Available backends: "
            f"{', '.join(list_backends())}"
        )


class TestClaudeRunTurn:
    def test_permission_mode_is_the_noninteractive_opt_in(self) -> None:
        assert PERMISSION_MODE == "bypassPermissions"

    def test_null_result_is_an_empty_reply_not_a_crash(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """An explicit null must not reach len() in the debug log."""
        backend = ClaudeBackend()
        payload = json.dumps({"type": "result", "session_id": "s1", "result": None})

        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = backend.run_turn("x", None, tmp_path)
        assert result.reply == ""
        assert result.session_id == "s1"

    def test_model_and_effort_are_forwarded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = ClaudeBackend(model="sonnet", reasoning_effort="high")
        payload = json.dumps({"session_id": "s1", "result": "done"})

        def fake_run(args, **_kwargs):
            assert args == [
                "claude",
                "--print",
                "--output-format",
                "json",
                "--permission-mode",
                PERMISSION_MODE,
                "--model",
                "sonnet",
                "--effort",
                "high",
                "-p",
                "work",
            ]
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert backend.run_turn("work", None, tmp_path).reply == "done"

    def test_new_turn_parses_session_and_result(
        self, tmp_path: Path, monkeypatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        backend = ClaudeBackend()
        payload = json.dumps({"is_error": False, "session_id": "s1", "result": "hi"})

        def fake_run(args, **kwargs):
            assert args == [
                "claude",
                "--print",
                "--output-format",
                "json",
                "--permission-mode",
                PERMISSION_MODE,
                "-p",
                "hello",
            ]
            _assert_subprocess_kwargs(kwargs, tmp_path)
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with caplog.at_level("DEBUG"):
            result = backend.run_turn("hello", None, tmp_path)
        assert result.session_id == "s1"
        assert result.reply == "hi"
        assert result.raw == payload
        messages = _messages(caplog)
        assert messages[0] == (
            f"claude turn: cwd={tmp_path} resume=False prompt_chars=5 timeout=3600s"
        )
        assert messages[1] == (
            "claude turn: invoking "
            "claude --print --output-format json --permission-mode "
            f"{PERMISSION_MODE} -p <prompt:5chars>"
        )
        assert (
            _reported_seconds(
                messages[2],
                rf"claude turn: exited 0 after (\d+\.\d)s with {len(payload)} "
                rf"chars of stdout",
            )
            < 60
        )
        assert messages[3] == "claude turn: parsed session=s1 reply_chars=2"

    def test_resume_turn_passes_resume_flag(
        self, tmp_path: Path, monkeypatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        backend = ClaudeBackend()

        def fake_run(args, **kwargs):
            assert args == [
                "claude",
                "--print",
                "--output-format",
                "json",
                "--permission-mode",
                PERMISSION_MODE,
                "--resume",
                "s1",
                "-p",
                "again",
            ]
            _assert_subprocess_kwargs(kwargs, tmp_path)
            payload = json.dumps(
                {"is_error": False, "session_id": "s1", "result": "still here"}
            )
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with caplog.at_level("DEBUG"):
            result = backend.run_turn("again", "s1", tmp_path)
        assert result.session_id == "s1"
        assert result.reply == "still here"
        messages = _messages(caplog)
        assert messages[0] == (
            f"claude turn: cwd={tmp_path} resume=True prompt_chars=5 timeout=3600s"
        )
        # The resumed session id stays readable; only the prompt is summarised.
        assert messages[1] == (
            "claude turn: invoking "
            "claude --print --output-format json --permission-mode "
            f"{PERMISSION_MODE} --resume s1 -p <prompt:5chars>"
        )
        assert messages[3] == "claude turn: parsed session=s1 reply_chars=10"

    def test_error_reply_raises(self, tmp_path: Path, monkeypatch) -> None:
        backend = ClaudeBackend()

        def fake_run(args, **kwargs):
            _assert_subprocess_kwargs(kwargs, tmp_path)
            # Session id present, so is_error is the only thing making this a
            # failure: a check that stopped reading the flag would return a
            # perfectly ordinary reply here.
            payload = json.dumps(
                {"is_error": True, "session_id": "s1", "result": "boom"}
            )
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(ClaudeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == "claude reported an error: boom"

    def test_result_defaults_to_empty_reply(self, tmp_path: Path, monkeypatch) -> None:
        backend = ClaudeBackend()

        def fake_run(args, **kwargs):
            payload = json.dumps({"is_error": False, "session_id": "s1"})
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = backend.run_turn("x", None, tmp_path)
        assert result.reply == ""

    def test_nonzero_exit_raises(self, tmp_path: Path, monkeypatch) -> None:
        backend = ClaudeBackend()
        stderr = "s" * 500 + "M" + "e" * 1999  # 2500 chars total

        def fake_run(args, **kwargs):
            _assert_subprocess_kwargs(kwargs, tmp_path)
            return subprocess.CompletedProcess(args, 1, stdout="", stderr=stderr)

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(ClaudeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == (
            f"claude exited 1\nstderr: {stderr[-2000:]}\nstdout: "
        )

    def test_malformed_json_raises(self, tmp_path: Path, monkeypatch) -> None:
        backend = ClaudeBackend()
        stdout = "s" * 500 + "M" + "e" * 1999  # 2500 chars total, not valid JSON

        def fake_run(args, **kwargs):
            _assert_subprocess_kwargs(kwargs, tmp_path)
            return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(ClaudeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == (
            f"claude output was not JSON\nstdout: {stdout[:400]}\n…\n{stdout[-1600:]}"
        )
        assert stdout[:400] in str(excinfo.value)
        assert stdout[-1600:] in str(excinfo.value)

    def test_text_prefix_then_json_envelope_is_parsed(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Regression: Claude print mode wrote the reply, then the JSON."""
        backend = ClaudeBackend()
        envelope = {
            "type": "result",
            "subtype": "success",
            "session_id": "s1",
            "result": "I've read both skills",
        }
        stdout = "I've read both skills\n" + json.dumps(envelope)

        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = backend.run_turn("x", None, tmp_path)
        assert result.session_id == "s1"
        assert result.reply == "I've read both skills"

    def test_opt_in_reason_is_a_failure_not_a_reply(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A turn that ran with no tools exits 0 and says so only here."""
        backend = ClaudeBackend()
        payload = json.dumps(
            {
                "type": "result",
                "reason": OPT_IN_REQUIRED_REASON,
                "session_id": "s1",
                "result": "I could not run that command",
            }
        )

        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(ClaudeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == (
            f"claude ran without tools: reason={OPT_IN_REQUIRED_REASON}. "
            f"--permission-mode {PERMISSION_MODE} did not take effect."
        )

    def test_another_reason_is_still_a_normal_reply(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        backend = ClaudeBackend()
        payload = json.dumps(
            {"type": "result", "reason": "stop", "session_id": "s1", "result": "done"}
        )

        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert backend.run_turn("x", None, tmp_path).reply == "done"

    def test_missing_session_id_raises_rather_than_returning_none(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """None here would be persisted over the id the agent already has."""
        backend = ClaudeBackend()
        payload = json.dumps({"type": "system", "result": "hi"})

        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(ClaudeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == (
            f"claude did not report a session_id\nstdout: {payload}"
        )

    def test_blank_session_id_raises(self, tmp_path: Path, monkeypatch) -> None:
        backend = ClaudeBackend()
        payload = json.dumps({"session_id": "", "result": "hi"})

        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(ClaudeTurnError, match="did not report a session_id"):
            backend.run_turn("x", None, tmp_path)

    def test_non_string_session_id_raises(self, tmp_path: Path, monkeypatch) -> None:
        backend = ClaudeBackend()
        payload = json.dumps({"session_id": 17, "result": "hi"})

        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(ClaudeTurnError, match="did not report a session_id"):
            backend.run_turn("x", None, tmp_path)

    def test_schema_is_passed_inline_and_parsed_from_its_own_field(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = ClaudeBackend()
        # The reply text is deliberately not the object: claude's own parse is
        # preferred, and parsing `result` is only the fallback below.
        payload = json.dumps(
            {
                "session_id": "s1",
                "result": "see structured_output",
                "structured_output": {"verdict": "pass"},
            }
        )

        def fake_run(args, **kwargs):
            assert args == [
                "claude",
                "--print",
                "--output-format",
                "json",
                "--permission-mode",
                PERMISSION_MODE,
                CLAUDE_SCHEMA_FLAG,
                SCHEMA.text,
                "-p",
                "hello",
            ]
            _assert_subprocess_kwargs(kwargs, tmp_path)
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = backend.run_turn("hello", None, tmp_path, schema=SCHEMA)
        assert result.structured == {"verdict": "pass"}
        assert result.reply == "see structured_output"

    def test_the_dialect_keyword_is_dropped_from_the_inline_schema(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`--json-schema` is checked by a draft-07 ajv: a `$schema` naming a
        newer dialect makes the CLI exit 1 before the turn runs."""
        backend = ClaudeBackend()
        payload = json.dumps(
            {"session_id": "s1", "result": "{}", "structured_output": {}}
        )
        seen: list[str] = []

        def fake_run(args, **kwargs):
            seen.append(args[args.index(CLAUDE_SCHEMA_FLAG) + 1])
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        backend.run_turn("hello", None, tmp_path, schema=DIALECT_SCHEMA)
        backend.run_turn("hello", None, tmp_path, schema=SCHEMA)

        assert seen == [
            '{"additionalProperties":false,"properties":{},"type":"object"}',
            SCHEMA.text,
        ]

    def test_structured_falls_back_to_parsing_the_result_text(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A CLI that stops publishing its own parse still has the reply."""
        backend = ClaudeBackend()
        payload = json.dumps({"session_id": "s1", "result": '{"verdict":"pass"}'})
        monkeypatch.setattr(subprocess, "run", _completed(0, payload, ""))
        result = backend.run_turn("hello", None, tmp_path, schema=SCHEMA)
        assert result.structured == {"verdict": "pass"}

    def test_a_reply_that_is_not_an_object_is_not_a_failed_turn(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The validator retries that; raising here would deny it the chance."""
        backend = ClaudeBackend()
        payload = json.dumps({"session_id": "s1", "result": "here you go:"})
        monkeypatch.setattr(subprocess, "run", _completed(0, payload, ""))
        result = backend.run_turn("hello", None, tmp_path, schema=SCHEMA)
        assert result.structured is None
        assert result.reply == "here you go:"

    def test_no_schema_leaves_structured_unset_even_for_a_json_reply(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = ClaudeBackend()
        payload = json.dumps({"session_id": "s1", "result": '{"verdict":"pass"}'})

        def fake_run(args, **kwargs):
            assert CLAUDE_SCHEMA_FLAG not in args
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert backend.run_turn("hello", None, tmp_path).structured is None

    def test_nonzero_exit_reports_the_error_envelope_over_the_exit_code(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The envelope names the failure; the exit code only counts it."""
        backend = ClaudeBackend()
        stdout = json.dumps(
            {"type": "result", "is_error": True, "result": "credit balance too low"}
        )

        def fake_run(args, **kwargs):
            _assert_subprocess_kwargs(kwargs, tmp_path)
            return subprocess.CompletedProcess(args, 1, stdout=stdout, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(ClaudeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == "claude reported an error: credit balance too low"

    def test_nonzero_exit_keeps_stdout_when_stderr_is_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The failure that prompted this: exit 1, stderr empty, and the only
        thing the CLI said sitting unread on stdout."""
        backend = ClaudeBackend()
        stdout = "Invalid API key · Please run /login"

        def fake_run(args, **kwargs):
            _assert_subprocess_kwargs(kwargs, tmp_path)
            return subprocess.CompletedProcess(args, 1, stdout=stdout, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(ClaudeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == (f"claude exited 1\nstderr: \nstdout: {stdout}")

    def test_nonzero_exit_bounds_a_long_stdout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both ends of the dump, the same bound the parse failure uses."""
        backend = ClaudeBackend()
        stdout = "s" * 500 + "M" + "e" * 1999  # 2500 chars, not an envelope

        def fake_run(args, **kwargs):
            _assert_subprocess_kwargs(kwargs, tmp_path)
            return subprocess.CompletedProcess(args, 2, stdout=stdout, stderr="boom")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(ClaudeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == (
            f"claude exited 2\nstderr: boom\nstdout: {stdout_for_error(stdout)}"
        )

    def test_nonzero_exit_ignores_a_non_error_envelope(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`is_error` false with a non-zero exit is not a reported error, so the
        exit code stays the headline rather than `result` being read as one."""
        backend = ClaudeBackend()
        stdout = json.dumps({"type": "result", "is_error": False, "result": "hi"})

        def fake_run(args, **kwargs):
            _assert_subprocess_kwargs(kwargs, tmp_path)
            return subprocess.CompletedProcess(args, 3, stdout=stdout, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(ClaudeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == f"claude exited 3\nstderr: \nstdout: {stdout}"


class TestParseClaudeStdout:
    def test_plain_object(self) -> None:
        payload = {"session_id": "s1", "result": "hi"}
        assert parse_claude_stdout(json.dumps(payload)) == payload

    def test_empty_is_an_error(self) -> None:
        with pytest.raises(ClaudeTurnError) as excinfo:
            parse_claude_stdout("   ")
        assert str(excinfo.value) == "claude output was not JSON\nstdout: "

    def test_picks_the_type_result_object(self) -> None:
        result_obj = json.dumps(
            {"type": "result", "session_id": "keep", "result": "done"}
        )
        later = json.dumps({"type": "system", "session_id": "later"})
        parsed = parse_claude_stdout(f"{result_obj}\n{later}")
        assert parsed["session_id"] == "keep"
        assert parsed["result"] == "done"

    def test_falls_back_to_the_last_object(self) -> None:
        parsed = parse_claude_stdout(
            'noise {"other": 1} then {"result": "ok", "session_id": "s"}'
        )
        assert parsed == {"result": "ok", "session_id": "s"}

    def test_short_garbage_keeps_the_whole_dump(self) -> None:
        with pytest.raises(ClaudeTurnError) as excinfo:
            parse_claude_stdout("not json")
        assert str(excinfo.value) == "claude output was not JSON\nstdout: not json"

    def test_skips_a_broken_brace_and_keeps_the_next_object(self) -> None:
        parsed = parse_claude_stdout('{{"result": "ok"}')
        assert parsed == {"result": "ok"}

    def test_last_object_when_none_look_like_a_result(self) -> None:
        assert parse_claude_stdout('{"a": 1}{"b": 2}{"c": 3}') == {"c": 3}


class TestStdoutForError:
    def test_keeps_a_short_dump(self) -> None:
        assert stdout_for_error("short") == "short"

    def test_keeps_exactly_2000_chars(self) -> None:
        text = "x" * 2000
        assert stdout_for_error(text) == text

    def test_splits_a_2001_char_dump(self) -> None:
        text = ("H" * 400) + "M" + ("T" * 1600)
        assert stdout_for_error(text) == f"{'H' * 400}\n…\n{'T' * 1600}"


class TestCodexRunTurn:
    def test_model_and_effort_are_forwarded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = CodexBackend(model="gpt-test", reasoning_effort="xhigh")
        stdout = (
            '{"type":"thread.started","thread_id":"t1"}\n'
            '{"type":"item.completed","item":'
            '{"type":"agent_message","text":"done"}}\n'
        )

        def fake_run(args, **_kwargs):
            assert args == [
                "codex",
                "exec",
                "--model",
                "gpt-test",
                "--config",
                'model_reasoning_effort="xhigh"',
                "work",
                "--json",
                "--skip-git-repo-check",
            ]
            return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert backend.run_turn("work", None, tmp_path).reply == "done"

    def test_new_turn_parses_thread_and_reply(
        self, tmp_path: Path, monkeypatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        backend = CodexBackend()

        def fake_run(args, **kwargs):
            assert args == ["codex", "exec", "hello", "--json", "--skip-git-repo-check"]
            _assert_subprocess_kwargs(kwargs, tmp_path)
            stdout = (
                '{"type":"thread.started","thread_id":"t1"}\n'
                '{"type":"item.completed","item":{"type":"agent_message","text":"yo"}}\n'
            )
            return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with caplog.at_level("DEBUG"):
            result = backend.run_turn("hello", None, tmp_path)
        assert result.session_id == "t1"
        assert result.reply == "yo"
        messages = _messages(caplog)
        assert messages[0] == (
            f"codex turn: cwd={tmp_path} resume=False prompt_chars=5 timeout=3600s"
        )
        # The prompt sits mid-argv for codex, so the summary must find it there.
        assert messages[1] == (
            "codex turn: invoking "
            "codex exec <prompt:5chars> --json --skip-git-repo-check"
        )
        assert (
            _reported_seconds(
                messages[2],
                r"codex turn: exited 0 after (\d+\.\d)s with \d+ chars of stdout",
            )
            < 60
        )
        assert messages[3] == "codex turn: parsed session=t1 messages=1 reply_chars=2"

    def test_resume_turn_uses_thread_id(
        self, tmp_path: Path, monkeypatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        backend = CodexBackend()

        def fake_run(args, **kwargs):
            assert args == [
                "codex",
                "exec",
                "resume",
                "t1",
                "again",
                "--json",
                "--skip-git-repo-check",
            ]
            _assert_subprocess_kwargs(kwargs, tmp_path)
            stdout = (
                '{"type":"thread.started","thread_id":"t1"}\n'
                '{"type":"item.completed","item":{"type":"agent_message","text":"back"}}\n'
            )
            return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with caplog.at_level("DEBUG"):
            result = backend.run_turn("again", "t1", tmp_path)
        assert result.session_id == "t1"
        assert result.reply == "back"
        messages = _messages(caplog)
        assert messages[0] == (
            f"codex turn: cwd={tmp_path} resume=True prompt_chars=5 timeout=3600s"
        )
        assert messages[1] == (
            "codex turn: invoking "
            "codex exec resume t1 <prompt:5chars> --json --skip-git-repo-check"
        )
        assert messages[3] == "codex turn: parsed session=t1 messages=1 reply_chars=4"

    def test_no_thread_id_raises(self, tmp_path: Path, monkeypatch) -> None:
        backend = CodexBackend()

        def fake_run(args, **kwargs):
            _assert_subprocess_kwargs(kwargs, tmp_path)
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(CodexTurnError, match="thread_id"):
            backend.run_turn("x", None, tmp_path)

    def test_no_thread_id_error_keeps_the_tail_of_long_output(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        backend = CodexBackend()
        stdout = "o" * 2500  # not JSON, so session_id stays None
        stderr = "e" * 2500

        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr=stderr)

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(CodexTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == (
            f"codex did not report a thread_id\nstdout: {stdout[-2000:]}"
            f"\nstderr: {stderr[-2000:]}"
        )

    def test_nonzero_exit_raises(self, tmp_path: Path, monkeypatch) -> None:
        backend = CodexBackend()
        stderr = "e" * 2500

        def fake_run(args, **kwargs):
            _assert_subprocess_kwargs(kwargs, tmp_path)
            return subprocess.CompletedProcess(args, 1, stdout="", stderr=stderr)

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(CodexTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == f"codex exited 1\nstderr: {stderr[-2000:]}"

    def test_parse_skips_blank_and_malformed_lines_and_other_events(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        backend = CodexBackend()
        stdout = "\n".join(
            [
                "",
                "not json",
                '{"type":"thread.started","thread_id":"t1"}',
                '{"type":"other.event"}',
                '{"type":"item.completed","item":{"type":"reasoning"}}',
                '{"type":"item.completed","item":{"type":"agent_message"}}',
                '{"type":"item.completed","item":{"type":"agent_message","text":"hi"}}',
                '{"type":"item.completed","item":{"type":"agent_message","text":"there"}}',
            ]
        )

        def fake_run(args, **kwargs):
            _assert_subprocess_kwargs(kwargs, tmp_path)
            return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = backend.run_turn("x", None, tmp_path)
        assert result.session_id == "t1"
        assert result.reply == "hi\nthere"
        assert result.raw == stdout

    def test_schema_is_passed_as_a_file_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """codex reads the schema itself, so it gets the path, not the text."""
        backend = CodexBackend()
        stdout = (
            '{"type":"thread.started","thread_id":"t1"}\n'
            '{"type":"item.completed","item":{"type":"agent_message",'
            '"text":"{\\"verdict\\":\\"pass\\"}"}}\n'
        )

        def fake_run(args, **kwargs):
            assert args == [
                "codex",
                "exec",
                "hello",
                "--json",
                "--skip-git-repo-check",
                CODEX_SCHEMA_FLAG,
                str(SCHEMA.path),
            ]
            assert SCHEMA.text not in args
            _assert_subprocess_kwargs(kwargs, tmp_path)
            return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = backend.run_turn("hello", None, tmp_path, schema=SCHEMA)
        # No pre-parsed field anywhere in the stream: the object is the text.
        assert result.structured == {"verdict": "pass"}

    def test_prose_around_the_object_leaves_structured_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = CodexBackend()
        stdout = (
            '{"type":"thread.started","thread_id":"t1"}\n'
            '{"type":"item.completed","item":{"type":"agent_message",'
            '"text":"Sure! {\\"verdict\\":\\"pass\\"}"}}\n'
        )
        monkeypatch.setattr(subprocess, "run", _completed(0, stdout, ""))
        assert backend.run_turn("hi", None, tmp_path, schema=SCHEMA).structured is None

    def test_no_schema_omits_the_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = CodexBackend()
        stdout = '{"type":"thread.started","thread_id":"t1"}\n'

        def fake_run(args, **kwargs):
            assert CODEX_SCHEMA_FLAG not in args
            return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert backend.run_turn("hi", None, tmp_path).structured is None

    def test_nonzero_exit_reports_the_api_message_not_the_stderr_tail(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A rejected schema is the case: the API says which part it refused,
        while stderr holds only the CLI's stdin notice."""
        backend = CodexBackend()
        stdout = (
            '{"type":"thread.started","thread_id":"t1"}\n'
            '{"type":"error","message":"Invalid schema for response_format: '
            "In context=('properties', 'detail'), 'additionalProperties' is "
            'required to be supplied and to be false."}\n'
        )
        monkeypatch.setattr(
            subprocess,
            "run",
            _completed(1, stdout, "Reading additional input from stdin..."),
        )
        with pytest.raises(CodexTurnError) as excinfo:
            backend.run_turn("hi", None, tmp_path, schema=SCHEMA)
        assert "In context=('properties', 'detail')" in str(excinfo.value)
        assert "stdin" not in str(excinfo.value)

    def test_an_error_before_a_later_event_is_still_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The failure is not always the last line: codex keeps narrating."""
        backend = CodexBackend()
        stdout = (
            '{"type":"thread.started","thread_id":"t1"}\n'
            '{"type":"error","message":"the real failure"}\n'
            '{"type":"item.completed","item":{"type":"agent_message","text":""}}\n'
        )
        monkeypatch.setattr(subprocess, "run", _completed(1, stdout, "noise"))
        with pytest.raises(CodexTurnError) as excinfo:
            backend.run_turn("hi", None, tmp_path)
        assert str(excinfo.value) == "codex reported an error: the real failure"

    def test_nonzero_exit_reads_a_failed_turn_event(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = CodexBackend()
        stdout = '{"type":"turn.failed","error":{"message":"usage limit reached"}}\n'
        monkeypatch.setattr(subprocess, "run", _completed(1, stdout, "noise"))
        with pytest.raises(
            CodexTurnError, match="codex reported an error: usage limit"
        ):
            backend.run_turn("hi", None, tmp_path)

    def test_nonzero_exit_without_a_message_keeps_the_exit_and_stderr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every shape that carries no message to report: an error event with
        no message, one with an empty message, and a failed turn whose error is
        not an object. An empty message is not a message: reporting it would
        print `codex reported an error: ` and nothing else."""
        backend = CodexBackend()
        stdout = (
            '{"type":"thread.started","thread_id":"t1"}\n'
            '{"type":"error"}\n'
            '{"type":"error","message":""}\n'
            '{"type":"turn.failed","error":"boom"}\n'
        )
        monkeypatch.setattr(subprocess, "run", _completed(3, stdout, "segfault"))
        with pytest.raises(CodexTurnError) as excinfo:
            backend.run_turn("hi", None, tmp_path)
        assert str(excinfo.value) == "codex exited 3\nstderr: segfault"

    def test_the_last_error_reported_is_the_one_raised(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An earlier event can be one the CLI recovered from; the last one is
        what it gave up on."""
        backend = CodexBackend()
        stdout = (
            '{"type":"error","message":"retrying after a stream hiccup"}\n'
            '{"type":"error","message":"the real failure"}\n'
        )
        monkeypatch.setattr(subprocess, "run", _completed(1, stdout, ""))
        with pytest.raises(CodexTurnError) as excinfo:
            backend.run_turn("hi", None, tmp_path)
        assert str(excinfo.value) == "codex reported an error: the real failure"


class TestGrokRunTurn:
    def test_model_and_effort_are_forwarded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = GrokBackend(model="grok-test", reasoning_effort="high")
        payload = json.dumps({"sessionId": "s1", "text": "done"})

        def fake_run(args, **_kwargs):
            assert args == [
                "grok",
                "--output-format",
                "json",
                "--always-approve",
                "--model",
                "grok-test",
                "--reasoning-effort",
                "high",
                "--single=work",
            ]
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert backend.run_turn("work", None, tmp_path).reply == "done"

    def test_new_turn_parses_session_and_text(
        self, tmp_path: Path, monkeypatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        backend = GrokBackend()
        payload = json.dumps(
            {"sessionId": "s1", "text": "hi", "stopReason": "end_turn"}
        )

        def fake_run(args, **kwargs):
            assert args == [
                "grok",
                "--output-format",
                "json",
                "--always-approve",
                "--single=hello",
            ]
            _assert_subprocess_kwargs(kwargs, tmp_path)
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with caplog.at_level("DEBUG"):
            result = backend.run_turn("hello", None, tmp_path)
        assert result.session_id == "s1"
        assert result.reply == "hi"
        assert result.raw == payload
        messages = _messages(caplog)
        assert messages[0] == (
            f"grok turn: cwd={tmp_path} resume=False prompt_chars=5 timeout=3600s"
        )
        assert messages[1] == (
            "grok turn: invoking "
            "grok --output-format json --always-approve --single=<prompt:5chars>"
        )
        assert (
            _reported_seconds(
                messages[2],
                rf"grok turn: exited 0 after (\d+\.\d)s with {len(payload)} "
                rf"chars of stdout",
            )
            < 60
        )
        assert messages[3] == "grok turn: parsed session=s1 reply_chars=2"

    def test_resume_uses_resume_flag_not_session_id(
        self, tmp_path: Path, monkeypatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """`--session-id` names a new session and errors if that id exists."""
        backend = GrokBackend()

        def fake_run(args, **kwargs):
            assert args == [
                "grok",
                "--output-format",
                "json",
                "--always-approve",
                "--resume",
                "s1",
                "--single=again",
            ]
            assert "--session-id" not in args
            assert "-s" not in args
            _assert_subprocess_kwargs(kwargs, tmp_path)
            payload = json.dumps({"sessionId": "s1", "text": "still here"})
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with caplog.at_level("DEBUG"):
            result = backend.run_turn("again", "s1", tmp_path)
        assert result.session_id == "s1"
        assert result.reply == "still here"
        messages = _messages(caplog)
        assert messages[0] == (
            f"grok turn: cwd={tmp_path} resume=True prompt_chars=5 timeout=3600s"
        )
        assert messages[1] == (
            "grok turn: invoking "
            "grok --output-format json --always-approve --resume s1 "
            "--single=<prompt:5chars>"
        )
        assert messages[3] == "grok turn: parsed session=s1 reply_chars=10"

    def test_always_approve_is_the_noninteractive_opt_in(self) -> None:
        assert ALWAYS_APPROVE_FLAG == "--always-approve"

    def test_prompt_flag_is_the_attached_form(self) -> None:
        assert PROMPT_FLAG == "--single"

    def test_hyphen_leading_prompt_is_attached_to_its_flag(
        self, tmp_path: Path, monkeypatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A bare `--fix ...` argument is read by grok's parser as a flag.

        `talk` builds the prompt from argparse.REMAINDER, so a prompt opening
        with a dash reaches the CLI verbatim and the run dies at argv parsing
        with a usage dump instead of answering.
        """
        backend = GrokBackend()
        prompt = "--fix the parser"

        def fake_run(args, **kwargs):
            assert prompt not in args
            assert args[-1] == f"--single={prompt}"
            _assert_subprocess_kwargs(kwargs, tmp_path)
            payload = json.dumps({"sessionId": "s1", "text": "ok"})
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with caplog.at_level("DEBUG"):
            result = backend.run_turn(prompt, None, tmp_path)
        assert result.reply == "ok"
        # The flag stays readable; only the prompt is collapsed.
        assert _messages(caplog)[1] == (
            "grok turn: invoking "
            "grok --output-format json --always-approve --single=<prompt:16chars>"
        )

    def test_null_text_is_an_empty_reply_not_a_crash(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """An explicit null must not reach len() in the debug log."""
        backend = GrokBackend()
        payload = json.dumps({"sessionId": "s1", "text": None})

        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = backend.run_turn("x", None, tmp_path)
        assert result.reply == ""
        assert result.session_id == "s1"

    def test_reply_comes_from_text_not_result(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Claude's field names must not be read as Grok's envelope."""
        backend = GrokBackend()
        payload = json.dumps(
            {
                "sessionId": "s1",
                "text": "grok-reply",
                "result": "claude-field",
                "session_id": "wrong",
            }
        )

        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = backend.run_turn("x", None, tmp_path)
        assert result.session_id == "s1"
        assert result.reply == "grok-reply"

    def test_claude_shaped_envelope_is_not_a_session(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        backend = GrokBackend()
        payload = json.dumps({"session_id": "s1", "result": "hi"})

        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(GrokTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == (
            f"grok did not report a sessionId\nstdout: {payload}"
        )

    def test_error_envelope_raises(self, tmp_path: Path, monkeypatch) -> None:
        backend = GrokBackend()

        def fake_run(args, **kwargs):
            _assert_subprocess_kwargs(kwargs, tmp_path)
            payload = json.dumps(
                {"type": "error", "message": "boom", "sessionId": "s1"}
            )
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(GrokTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == "grok reported an error: boom"

    def test_nonzero_exit_prefers_the_json_error_message(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        backend = GrokBackend()
        payload = json.dumps({"type": "error", "message": "auth failed"})

        def fake_run(args, **kwargs):
            _assert_subprocess_kwargs(kwargs, tmp_path)
            return subprocess.CompletedProcess(
                args, 1, stdout=payload, stderr="ignored"
            )

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(GrokTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == "grok reported an error: auth failed"

    def test_nonzero_exit_without_error_object_uses_stderr(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        backend = GrokBackend()
        stderr = "s" * 500 + "M" + "e" * 1999  # 2500 chars total

        def fake_run(args, **kwargs):
            _assert_subprocess_kwargs(kwargs, tmp_path)
            return subprocess.CompletedProcess(args, 1, stdout="", stderr=stderr)

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(GrokTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == f"grok exited 1\nstderr: {stderr[-2000:]}"

    def test_nonzero_exit_with_a_success_envelope_still_fails(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Exit 0 is part of success; a payload is not enough on its own."""
        backend = GrokBackend()
        payload = json.dumps({"sessionId": "s1", "text": "hi"})
        stderr = "e" * 2500

        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 2, stdout=payload, stderr=stderr)

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(GrokTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == f"grok exited 2\nstderr: {stderr[-2000:]}"

    def test_text_defaults_to_empty_reply(self, tmp_path: Path, monkeypatch) -> None:
        backend = GrokBackend()

        def fake_run(args, **kwargs):
            payload = json.dumps({"sessionId": "s1"})
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = backend.run_turn("x", None, tmp_path)
        assert result.reply == ""

    def test_malformed_json_raises(self, tmp_path: Path, monkeypatch) -> None:
        backend = GrokBackend()
        stdout = "s" * 500 + "M" + "e" * 1999  # 2500 chars total, not valid JSON

        def fake_run(args, **kwargs):
            _assert_subprocess_kwargs(kwargs, tmp_path)
            return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(GrokTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == (
            f"grok output was not JSON\nstdout: {stdout[:400]}\n…\n{stdout[-1600:]}"
        )
        assert stdout[:400] in str(excinfo.value)
        assert stdout[-1600:] in str(excinfo.value)

    def test_text_prefix_then_json_envelope_is_parsed(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        backend = GrokBackend()
        envelope = {"sessionId": "s1", "text": "I've read both skills"}
        stdout = "I've read both skills\n" + json.dumps(envelope)

        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = backend.run_turn("x", None, tmp_path)
        assert result.session_id == "s1"
        assert result.reply == "I've read both skills"

    def test_another_type_is_still_a_normal_reply(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        backend = GrokBackend()
        payload = json.dumps(
            {"type": "end", "sessionId": "s1", "text": "done", "stopReason": "end_turn"}
        )

        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert backend.run_turn("x", None, tmp_path).reply == "done"

    def test_missing_session_id_raises_rather_than_returning_none(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        backend = GrokBackend()
        payload = json.dumps({"text": "hi"})

        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(GrokTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == (
            f"grok did not report a sessionId\nstdout: {payload}"
        )

    def test_blank_session_id_raises(self, tmp_path: Path, monkeypatch) -> None:
        backend = GrokBackend()
        payload = json.dumps({"sessionId": "", "text": "hi"})

        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(GrokTurnError, match="did not report a sessionId"):
            backend.run_turn("x", None, tmp_path)

    def test_non_string_session_id_raises(self, tmp_path: Path, monkeypatch) -> None:
        backend = GrokBackend()
        payload = json.dumps({"sessionId": 17, "text": "hi"})

        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(GrokTurnError, match="did not report a sessionId"):
            backend.run_turn("x", None, tmp_path)

    def test_schema_is_passed_inline_as_its_own_argument(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Its own argument, not glued on like --single=: the value starts with
        `{`, which no parser reads as a flag."""
        backend = GrokBackend()
        # As with claude: the text is not the object, so this can only pass by
        # reading grok's own parse.
        payload = json.dumps(
            {
                "sessionId": "s1",
                "text": "see structuredOutput",
                "structuredOutput": {"verdict": "pass"},
            }
        )

        def fake_run(args, **kwargs):
            assert args == [
                "grok",
                "--output-format",
                "json",
                ALWAYS_APPROVE_FLAG,
                GROK_SCHEMA_FLAG,
                SCHEMA.text,
                "--single=hello",
            ]
            _assert_subprocess_kwargs(kwargs, tmp_path)
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = backend.run_turn("hello", None, tmp_path, schema=SCHEMA)
        assert result.structured == {"verdict": "pass"}

    def test_structured_falls_back_to_parsing_the_text(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = GrokBackend()
        payload = json.dumps({"sessionId": "s1", "text": '{"verdict":"pass"}'})
        monkeypatch.setattr(subprocess, "run", _completed(0, payload, ""))
        result = backend.run_turn("hello", None, tmp_path, schema=SCHEMA)
        assert result.structured == {"verdict": "pass"}

    def test_no_schema_omits_the_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = GrokBackend()
        payload = json.dumps({"sessionId": "s1", "text": '{"verdict":"pass"}'})

        def fake_run(args, **kwargs):
            assert GROK_SCHEMA_FLAG not in args
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert backend.run_turn("hello", None, tmp_path).structured is None


class TestParseGrokStdout:
    def test_plain_object(self) -> None:
        payload = {"sessionId": "s1", "text": "hi"}
        assert parse_grok_stdout(json.dumps(payload)) == payload

    def test_empty_is_an_error(self) -> None:
        with pytest.raises(GrokTurnError) as excinfo:
            parse_grok_stdout("   ")
        assert str(excinfo.value) == "grok output was not JSON\nstdout: "

    def test_picks_the_session_envelope(self) -> None:
        noise = json.dumps({"progress": 1})
        envelope = json.dumps({"sessionId": "keep", "text": "done"})
        parsed = parse_grok_stdout(f"{noise}\n{envelope}")
        assert parsed["sessionId"] == "keep"
        assert parsed["text"] == "done"

    def test_picks_the_session_envelope_ahead_of_later_noise(self) -> None:
        """Last-object fallback would take the trailing progress event."""
        envelope = json.dumps({"sessionId": "keep", "text": "done"})
        later = json.dumps({"progress": 1})
        parsed = parse_grok_stdout(f"{envelope}\n{later}")
        assert parsed["sessionId"] == "keep"

    def test_picks_an_error_object(self) -> None:
        noise = json.dumps({"progress": 1})
        error = json.dumps({"type": "error", "message": "nope"})
        parsed = parse_grok_stdout(f"{noise}\n{error}")
        assert parsed == {"type": "error", "message": "nope"}

    def test_picks_an_error_object_ahead_of_later_noise(self) -> None:
        error = json.dumps({"type": "error", "message": "nope"})
        later = json.dumps({"progress": 1})
        parsed = parse_grok_stdout(f"{error}\n{later}")
        assert parsed == {"type": "error", "message": "nope"}

    def test_picks_a_text_only_object_ahead_of_later_noise(self) -> None:
        envelope = json.dumps({"text": "done"})
        later = json.dumps({"progress": 1})
        parsed = parse_grok_stdout(f"{envelope}\n{later}")
        assert parsed == {"text": "done"}

    def test_picks_a_session_id_only_object_ahead_of_later_noise(self) -> None:
        """A success envelope can omit text; sessionId alone must still win."""
        envelope = json.dumps({"sessionId": "keep"})
        later = json.dumps({"progress": 1})
        parsed = parse_grok_stdout(f"{envelope}\n{later}")
        assert parsed == {"sessionId": "keep"}

    def test_a_trailing_tip_does_not_outrank_the_real_envelope(self) -> None:
        """`text` alone is a weak marker: tips and warnings carry one too."""
        envelope = json.dumps({"sessionId": "keep", "text": "done"})
        tip = json.dumps({"type": "tip", "text": "try --help"})
        assert parse_grok_stdout(f"{envelope}\n{tip}") == {
            "sessionId": "keep",
            "text": "done",
        }

    def test_a_trailing_tip_does_not_outrank_an_error_envelope(self) -> None:
        error = json.dumps({"type": "error", "message": "nope"})
        tip = json.dumps({"type": "tip", "text": "try --help"})
        assert parse_grok_stdout(f"{error}\n{tip}") == {
            "type": "error",
            "message": "nope",
        }

    def test_falls_back_to_the_last_object(self) -> None:
        parsed = parse_grok_stdout('noise {"other": 1} then {"later": 2}')
        assert parsed == {"later": 2}

    def test_short_garbage_keeps_the_whole_dump(self) -> None:
        with pytest.raises(GrokTurnError) as excinfo:
            parse_grok_stdout("not json")
        assert str(excinfo.value) == "grok output was not JSON\nstdout: not json"

    def test_objects_inside_a_json_array_are_still_found(self) -> None:
        parsed = parse_grok_stdout('[{"a": 1}, {"sessionId": "s", "text": "hi"}]')
        assert parsed == {"sessionId": "s", "text": "hi"}

    def test_skips_a_broken_brace_and_keeps_the_next_object(self) -> None:
        parsed = parse_grok_stdout('{{"sessionId": "s"}')
        assert parsed == {"sessionId": "s"}


class TestOpenCodeRunTurn:
    def test_fresh_turn_uses_exact_argv_and_verbatim_input(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        backend = OpenCodeBackend(model="provider/model", reasoning_effort="high")
        prompt = '- leading "quote"\\path\nsecond line — café'
        stdout = "\n".join(
            [
                json.dumps({"type": "step-start", "sessionID": "s1"}),
                json.dumps(
                    {
                        "type": "text",
                        "sessionID": "s1",
                        "part": {"id": "p1", "text": "done"},
                    }
                ),
            ]
        )
        calls: list[tuple[list[str], dict]] = []

        def fake_run(args, **kwargs):
            calls.append((args, kwargs))
            return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with caplog.at_level("DEBUG", logger="backends.opencode"):
            result = backend.run_turn(prompt, None, tmp_path)

        assert calls[0][0] == [
            "opencode",
            "run",
            "--format",
            "json",
            "--auto",
            "--dir",
            str(tmp_path),
            "--model",
            "provider/model",
            "--variant",
            "high",
        ]
        assert prompt not in calls[0][0]
        _assert_subprocess_kwargs(
            calls[0][1], tmp_path, expected_stdin=None, expected_input=prompt
        )
        assert result == TurnResult("s1", "done", stdout)
        messages = _messages(caplog)
        assert messages[0] == (
            f"opencode turn: cwd={tmp_path} resume=False prompt_chars={len(prompt)} "
            "timeout=3600s"
        )
        assert messages[1] == (
            "opencode turn: invoking opencode run --format json --auto --dir "
            f"{tmp_path} --model provider/model --variant high"
        )
        assert (
            _reported_seconds(
                messages[2],
                rf"opencode turn: exited 0 after (\d+\.\d)s with {len(stdout)} "
                rf"chars of stdout",
            )
            < 60
        )
        assert messages[3] == "opencode turn: parsed session=s1 parts=1 reply_chars=4"

    def test_resumed_turn_adds_session_flag(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        backend = OpenCodeBackend()
        stdout = json.dumps({"type": "step-finish", "sessionID": "s1"})

        def fake_run(args, **kwargs):
            assert args == [
                "opencode",
                "run",
                "--format",
                "json",
                "--auto",
                "--dir",
                str(tmp_path),
                "--session",
                "s1",
            ]
            _assert_subprocess_kwargs(
                kwargs, tmp_path, expected_stdin=None, expected_input="again"
            )
            return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with caplog.at_level("DEBUG", logger="backends.opencode"):
            assert backend.run_turn("again", "s1", tmp_path).session_id == "s1"
        assert _messages(caplog)[0] == (
            f"opencode turn: cwd={tmp_path} resume=True prompt_chars=5 timeout=3600s"
        )

    def test_noisy_events_are_combined_and_structured_is_parsed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = OpenCodeBackend()
        reply = '{"verdict":\n"pass"}'
        stdout = "\nnot json\n" + "\n".join(
            [
                json.dumps({"type": "step-start", "sessionID": "s1"}),
                json.dumps(
                    {
                        "type": "text",
                        "sessionID": "s1",
                        "part": {"id": "a", "text": '{"verdict":'},
                    }
                ),
                json.dumps(
                    {
                        "type": "text",
                        "sessionID": "s1",
                        "part": {"id": "b", "text": '"pass"}'},
                    }
                ),
            ]
        )
        monkeypatch.setattr(subprocess, "run", _completed(0, stdout))
        result = backend.run_turn("x", None, tmp_path, schema=SCHEMA)
        assert result.session_id == "s1"
        assert result.reply == reply
        assert result.raw == stdout
        assert result.structured == {"verdict": "pass"}

    def test_duplicate_parts_keep_last_text_in_first_seen_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = OpenCodeBackend()
        events = [
            {"type": "text", "sessionID": "s1", "part": {"id": "a", "text": "old"}},
            {"type": "text", "sessionID": "s1", "part": {"id": "b", "text": "two"}},
            {"type": "text", "sessionID": "s1", "part": {"id": "a", "text": "new"}},
        ]
        stdout = "\n".join(json.dumps(event) for event in events)
        monkeypatch.setattr(subprocess, "run", _completed(0, stdout))
        assert backend.run_turn("x", None, tmp_path).reply == "new\ntwo"

    @pytest.mark.parametrize("session_id", [None, "", " \t\n", 17])
    def test_invalid_session_id_raises(
        self,
        session_id: object,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        backend = OpenCodeBackend()
        payload = {"type": "step-start"}
        if session_id is not None:
            payload["sessionID"] = session_id
        stdout = json.dumps(payload)
        monkeypatch.setattr(subprocess, "run", _completed(0, stdout))
        with pytest.raises(OpenCodeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == (
            f"opencode did not report a sessionID\nstdout: {stdout}\nstderr: "
        )

    def test_error_event_on_zero_exit_raises(self, tmp_path: Path, monkeypatch) -> None:
        backend = OpenCodeBackend()
        stdout = json.dumps(
            {
                "type": "error",
                "sessionID": "s1",
                "error": {"data": {"message": "bad request"}},
            }
        )
        monkeypatch.setattr(subprocess, "run", _completed(0, stdout))
        with pytest.raises(OpenCodeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == "opencode reported an error: bad request"

    def test_nonzero_exit_prefers_data_message(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = OpenCodeBackend()
        stdout = json.dumps(
            {
                "type": "error",
                "sessionID": "s1",
                "error": {"name": "ProviderError", "data": {"message": "auth failed"}},
            }
        )
        monkeypatch.setattr(subprocess, "run", _completed(2, stdout, "ignored"))
        with pytest.raises(OpenCodeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == "opencode reported an error: auth failed"

    def test_error_detail_requires_string_message_before_using_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = OpenCodeBackend()
        stdout = json.dumps(
            {
                "type": "error",
                "sessionID": "s1",
                "error": {
                    "name": "ProviderError",
                    "data": {"message": 17},
                },
            }
        )
        monkeypatch.setattr(subprocess, "run", _completed(2, stdout))
        with pytest.raises(OpenCodeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == "opencode reported an error: ProviderError"

    def test_error_detail_requires_string_name_before_using_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = OpenCodeBackend()
        stdout = json.dumps(
            {
                "type": "error",
                "sessionID": "s1",
                "error": {"name": 17},
            }
        )
        stderr = "stderr detail"
        monkeypatch.setattr(subprocess, "run", _completed(2, stdout, stderr))
        with pytest.raises(OpenCodeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == f"opencode exited 2\nstderr: {stderr}"

    def test_nonzero_exit_uses_last_error_event(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = OpenCodeBackend()
        stdout = "\n".join(
            [
                json.dumps(
                    {
                        "type": "error",
                        "sessionID": "s1",
                        "error": {"data": {"message": "old"}},
                    }
                ),
                json.dumps(
                    {
                        "type": "error",
                        "sessionID": "s1",
                        "error": {"data": {"message": "latest"}},
                    }
                ),
                json.dumps({"type": "step-start", "sessionID": "s1"}),
            ]
        )
        monkeypatch.setattr(subprocess, "run", _completed(2, stdout))
        with pytest.raises(OpenCodeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == "opencode reported an error: latest"

    def test_nonzero_exit_uses_error_name_when_message_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = OpenCodeBackend()
        stdout = json.dumps(
            {
                "type": "error",
                "sessionID": "s1",
                "error": {"name": "ProviderError", "data": {}},
            }
        )
        monkeypatch.setattr(subprocess, "run", _completed(2, stdout, "ignored"))
        with pytest.raises(OpenCodeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == "opencode reported an error: ProviderError"

    def test_nonzero_exit_falls_back_to_stderr_tail(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = OpenCodeBackend()
        stderr = "s" * 2500
        monkeypatch.setattr(subprocess, "run", _completed(3, "not json", stderr))
        with pytest.raises(OpenCodeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == f"opencode exited 3\nstderr: {stderr[-2000:]}"

    def test_error_event_without_detail_on_zero_exit_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = OpenCodeBackend()
        stdout = "\n".join(
            [
                json.dumps({"type": "error", "sessionID": "s1", "error": "bad"}),
                json.dumps({"type": "error", "sessionID": "s1", "error": {}}),
                json.dumps({"type": "step-start", "sessionID": "s1"}),
            ]
        )
        monkeypatch.setattr(subprocess, "run", _completed(0, stdout))
        with pytest.raises(OpenCodeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == "opencode reported an error event"

    def test_non_dict_and_empty_stream_events_are_ignored_or_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = OpenCodeBackend()
        stdout = '["progress"]\n{"type":"step-start","sessionID":"s1"}'
        monkeypatch.setattr(subprocess, "run", _completed(0, stdout))
        assert backend.run_turn("x", None, tmp_path).session_id == "s1"

        monkeypatch.setattr(subprocess, "run", _completed(0, "not json"))
        with pytest.raises(OpenCodeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == (
            "opencode did not report a sessionID\nstdout: not json\nstderr: "
        )

    def test_malformed_text_part_is_ignored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = OpenCodeBackend()
        stdout = "\n".join(
            [
                json.dumps({"type": "step-start", "sessionID": "s1"}),
                json.dumps({"type": "text", "sessionID": "s1", "part": None}),
                json.dumps(
                    {
                        "type": "text",
                        "sessionID": "s1",
                        "part": {"id": "a", "text": "one"},
                    }
                ),
                json.dumps(
                    {
                        "type": "text",
                        "sessionID": "s1",
                        "part": {"id": "missing"},
                    }
                ),
                json.dumps(
                    {
                        "type": "text",
                        "sessionID": "s1",
                        "part": {"id": "b", "text": "two"},
                    }
                ),
            ]
        )
        monkeypatch.setattr(subprocess, "run", _completed(0, stdout))
        assert backend.run_turn("x", None, tmp_path).reply == "one\ntwo"

    def test_missing_session_error_preserves_both_output_tails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = OpenCodeBackend()
        stdout = json.dumps({"type": "text"}) + "\n" + "o" * 2500
        stderr = "e" * 2500
        monkeypatch.setattr(subprocess, "run", _completed(0, stdout, stderr))
        with pytest.raises(OpenCodeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == (
            "opencode did not report a sessionID\n"
            f"stdout: {stdout[-2000:]}\nstderr: {stderr[-2000:]}"
        )

    def test_empty_event_stream_preserves_both_output_tails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = OpenCodeBackend()
        stdout = "o" * 2500
        stderr = "e" * 2500
        monkeypatch.setattr(subprocess, "run", _completed(0, stdout, stderr))
        with pytest.raises(OpenCodeTurnError) as excinfo:
            backend.run_turn("x", None, tmp_path)
        assert str(excinfo.value) == (
            "opencode did not report a sessionID\n"
            f"stdout: {stdout[-2000:]}\nstderr: {stderr[-2000:]}"
        )

    def test_no_schema_does_not_parse_json_reply_into_structured_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = OpenCodeBackend()
        stdout = "\n".join(
            [
                json.dumps({"type": "step-start", "sessionID": "s1"}),
                json.dumps(
                    {
                        "type": "text",
                        "sessionID": "s1",
                        "part": {"id": "p1", "text": "{}"},
                    }
                ),
            ]
        )
        monkeypatch.setattr(subprocess, "run", _completed(0, stdout))
        assert backend.run_turn("x", None, tmp_path).structured is None

    def test_two_completed_blocks_are_separated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A turn that speaks either side of a tool call emits two completed
        parts. Concatenating them ran the sentences together in a live turn."""
        backend = OpenCodeBackend()
        events = [
            {
                "type": "text",
                "sessionID": "s1",
                "part": {"id": "a", "text": "I'll read marker.txt."},
            },
            {
                "type": "tool_use",
                "sessionID": "s1",
                "part": {"id": "t", "tool": "read"},
            },
            {"type": "text", "sessionID": "s1", "part": {"id": "b", "text": "rhubarb"}},
        ]
        stdout = "\n".join(json.dumps(event) for event in events)
        monkeypatch.setattr(subprocess, "run", _completed(0, stdout))
        assert backend.run_turn("x", None, tmp_path).reply == (
            "I'll read marker.txt.\nrhubarb"
        )

    def test_a_fenced_reply_still_yields_the_structured_object(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The measured 1.18.21 behaviour: asked for JSON, it answers in a
        ```json fence. `json.loads` of that whole reply fails, so without the
        scan every schema turn burns its retries and then fails."""
        backend = OpenCodeBackend()
        reply = '```json\n{"verdict":"pass"}\n```'
        stdout = json.dumps(
            {"type": "text", "sessionID": "s1", "part": {"id": "a", "text": reply}}
        )
        monkeypatch.setattr(subprocess, "run", _completed(0, stdout))
        result = backend.run_turn("x", None, tmp_path, schema=SCHEMA)
        assert result.reply == reply
        assert result.structured == {"verdict": "pass"}

    def test_an_object_introduced_by_prose_is_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = OpenCodeBackend()
        reply = 'Here is the result:\n{"verdict":"pass"}'
        stdout = json.dumps(
            {"type": "text", "sessionID": "s1", "part": {"id": "a", "text": reply}}
        )
        monkeypatch.setattr(subprocess, "run", _completed(0, stdout))
        assert backend.run_turn("x", None, tmp_path, schema=SCHEMA).structured == {
            "verdict": "pass"
        }

    def test_the_last_object_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A model that shows its working puts the answer last, and the schema
        check downstream is what decides whether it is the right one."""
        backend = OpenCodeBackend()
        reply = 'Not this: {"verdict":"draft"}\nFinal: {"verdict":"pass"}'
        stdout = json.dumps(
            {"type": "text", "sessionID": "s1", "part": {"id": "a", "text": reply}}
        )
        monkeypatch.setattr(subprocess, "run", _completed(0, stdout))
        assert backend.run_turn("x", None, tmp_path, schema=SCHEMA).structured == {
            "verdict": "pass"
        }

    def test_a_reply_with_no_object_is_none_not_an_exception(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Not an object is a contract failure the validator retries, not a
        parse error for the adapter to raise on."""
        backend = OpenCodeBackend()
        stdout = json.dumps(
            {
                "type": "text",
                "sessionID": "s1",
                "part": {"id": "a", "text": "Sure! Here you go:"},
            }
        )
        monkeypatch.setattr(subprocess, "run", _completed(0, stdout))
        assert backend.run_turn("x", None, tmp_path, schema=SCHEMA).structured is None

    def test_a_fenced_reply_stays_unparsed_without_a_schema(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The scan must not turn an ordinary reply that happens to quote JSON
        into a structured object nobody asked for."""
        backend = OpenCodeBackend()
        stdout = json.dumps(
            {
                "type": "text",
                "sessionID": "s1",
                "part": {"id": "a", "text": '```json\n{"verdict":"pass"}\n```'},
            }
        )
        monkeypatch.setattr(subprocess, "run", _completed(0, stdout))
        assert backend.run_turn("x", None, tmp_path).structured is None


class TestReplyText:
    def test_returns_the_string(self) -> None:
        assert reply_text({"text": "hi"}, "text") == "hi"

    def test_missing_key_is_empty(self) -> None:
        assert reply_text({"sessionId": "s1"}, "text") == ""

    def test_explicit_null_is_empty(self) -> None:
        """`.get(key, "")` would hand None straight to len()."""
        assert reply_text({"text": None}, "text") == ""

    def test_non_string_is_empty(self) -> None:
        assert reply_text({"text": ["block"]}, "text") == ""

    def test_empty_string_is_kept(self) -> None:
        assert reply_text({"text": ""}, "text") == ""

    def test_reads_the_key_it_was_given(self) -> None:
        payload = {"text": "grok", "result": "claude"}
        assert reply_text(payload, "result") == "claude"


class TestJsonObjects:
    def test_finds_each_object_in_order(self) -> None:
        assert json_objects('{"a": 1} noise {"b": 2}') == [{"a": 1}, {"b": 2}]

    def test_empty_when_there_is_no_object(self) -> None:
        assert json_objects("not json") == []

    def test_skips_a_broken_brace(self) -> None:
        assert json_objects('{{"ok": true}') == [{"ok": True}]


class TestStructuredReply:
    """The one place backend envelopes turn into one object."""

    def test_no_schema_means_no_object_however_json_the_reply_is(self) -> None:
        assert structured_reply(None, '{"a":1}', {"a": 1}) is None

    def test_the_pre_parsed_field_wins(self) -> None:
        assert structured_reply(SCHEMA, "ignored", {"a": 1}) == {"a": 1}

    def test_falls_back_to_parsing_the_reply(self) -> None:
        assert structured_reply(SCHEMA, '{"a":1}') == {"a": 1}

    def test_a_non_dict_pre_parsed_field_is_not_trusted(self) -> None:
        assert structured_reply(SCHEMA, '{"a":1}', "not an object") == {"a": 1}

    def test_unparseable_reply_is_none_not_an_exception(self) -> None:
        assert structured_reply(SCHEMA, "Sure! Here you go:") is None

    def test_a_json_value_that_is_not_an_object_is_none(self) -> None:
        """A schema's reply is one object; a bare list or number is not it."""
        assert structured_reply(SCHEMA, "[1, 2]") is None


class TestOrchestrator:
    def test_validated_opencode_turn_warns_once_about_cli_schema_enforcement(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        class AdvisoryBackend(AgentBackend):
            name = "advisory"
            enforces_schema = False

            def run_turn(
                self,
                prompt: str,
                session_id: str | None,
                cwd: Path,
                timeout: int = DEFAULT_TURN_TIMEOUT,
                schema: OutputSchema | None = None,
            ) -> TurnResult:
                return TurnResult(
                    session_id="s1",
                    reply="{}",
                    raw="",
                    structured={},
                )

        register_backend("advisory", AdvisoryBackend)
        orch = Orchestrator(state_file=tmp_path / "state.json")
        orch.spawn("agent", "advisory")

        with caplog.at_level(logging.WARNING, logger="orchestrator"):
            result = orch.talk("agent", "hello", schema=SCHEMA, retries=0)

        assert result.structured == {}
        assert _messages(caplog) == [
            "backend advisory: schema is enforced via validation/repair, not the CLI"
        ]

    def test_a_non_enforcing_backend_is_sent_the_schema_document(
        self, tmp_path: Path
    ) -> None:
        """The CLI has no flag to carry it, so the prompt has to. Without this
        the model is told to conform to a schema it was never shown."""
        seen: list[str] = []

        class AdvisoryBackend(AgentBackend):
            name = "advisory-prompt"
            enforces_schema = False

            def run_turn(
                self,
                prompt: str,
                session_id: str | None,
                cwd: Path,
                timeout: int = DEFAULT_TURN_TIMEOUT,
                schema: OutputSchema | None = None,
            ) -> TurnResult:
                seen.append(prompt)
                return TurnResult(session_id="s1", reply="{}", raw="", structured={})

        register_backend("advisory-prompt", AdvisoryBackend)
        orch = Orchestrator(state_file=tmp_path / "state.json")
        orch.spawn("agent", "advisory-prompt")
        orch.talk("agent", "hello", schema=SCHEMA, retries=0)

        assert seen == [compose_schema_prompt("hello", SCHEMA)]
        assert SCHEMA.text in seen[0]

    def test_an_enforcing_backend_is_not_sent_the_schema_document(
        self, tmp_path: Path
    ) -> None:
        seen: list[str] = []

        class EnforcingBackend(AgentBackend):
            name = "enforcing-prompt"

            def run_turn(
                self,
                prompt: str,
                session_id: str | None,
                cwd: Path,
                timeout: int = DEFAULT_TURN_TIMEOUT,
                schema: OutputSchema | None = None,
            ) -> TurnResult:
                seen.append(prompt)
                return TurnResult(session_id="s1", reply="{}", raw="", structured={})

        register_backend("enforcing-prompt", EnforcingBackend)
        orch = Orchestrator(state_file=tmp_path / "state.json")
        orch.spawn("agent", "enforcing-prompt")
        orch.talk("agent", "hello", schema=SCHEMA, retries=0)

        assert seen == [compose_schema_prompt("hello")]
        assert SCHEMA.text not in seen[0]

    def test_schema_warning_is_absent_without_schema_or_for_enforcing_backend(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        class EnforcingBackend(AgentBackend):
            name = "enforcing"

            def run_turn(
                self,
                prompt: str,
                session_id: str | None,
                cwd: Path,
                timeout: int = DEFAULT_TURN_TIMEOUT,
                schema: OutputSchema | None = None,
            ) -> TurnResult:
                return TurnResult(
                    session_id="s1",
                    reply="{}",
                    raw="",
                    structured={},
                )

        register_backend("enforcing", EnforcingBackend)
        orch = Orchestrator(state_file=tmp_path / "state.json")
        orch.spawn("echo", "enforcing")

        with caplog.at_level(logging.WARNING, logger="orchestrator"):
            orch.talk("echo", "plain")
            orch.talk("echo", "structured", schema=SCHEMA, retries=0)

        assert _messages(caplog) == []

    def test_spawn_talk_persists_and_resumes(self, tmp_path: Path, monkeypatch) -> None:
        state_file = tmp_path / "state.json"
        seen_session_ids: list[str | None] = []
        # A literal, not orchestrator.WORKDIR: comparing the module constant
        # against itself would pass whatever it happened to be set to.
        workdir = tmp_path / "workdir"
        monkeypatch.setattr(orchestrator, "WORKDIR", workdir)

        def fake_backend_run(args, **kwargs):
            _assert_subprocess_kwargs(kwargs, workdir)
            return subprocess.CompletedProcess(
                args,
                0,
                stdout=json.dumps(
                    {"is_error": False, "session_id": "persist-me", "result": "reply"}
                ),
                stderr="",
            )

        class RecordingBackend(AgentBackend):
            @property
            def name(self) -> str:
                return "recording"

            def run_turn(
                self,
                prompt: str,
                session_id: str | None,
                cwd: Path,
                timeout: int = DEFAULT_TURN_TIMEOUT,
                schema: OutputSchema | None = None,
            ) -> TurnResult:
                seen_session_ids.append(session_id)
                return TurnResult(session_id="persist-me", reply="reply", raw="")

        register_backend("recording", RecordingBackend)
        monkeypatch.setattr(subprocess, "run", fake_backend_run)

        orch = Orchestrator(state_file=state_file)
        agent = orch.spawn("a1", "claude")
        assert agent.session_id is None

        orch2 = Orchestrator(state_file=state_file)
        agent2 = orch2.spawn("a2", "recording")
        assert agent2.name == "a2"
        orch2.talk("a2", "first")
        orch2.talk("a2", "second")
        assert seen_session_ids == [None, "persist-me"]

        result = orch.talk("a1", "first")
        assert result.reply == "reply"
        assert orch.agents["a1"].session_id == "persist-me"

        orch3 = Orchestrator(state_file=state_file)
        assert "a1" in orch3.agents
        assert orch3.agents["a1"].name == "a1"
        assert orch3.agents["a1"].session_id == "persist-me"
        assert orch3.talk("a1", "second").reply == "reply"

    def test_persist_writes_sorted_indented_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state_file = tmp_path / "state.json"
        monkeypatch.setattr(orchestrator, "_utcnow", lambda: "2026-08-25T00:00:00Z")
        orch = Orchestrator(state_file=state_file)
        orch.spawn("b", "codex")
        orch.spawn("a", "claude")
        assert state_file.read_text(encoding="utf-8") == (
            "{\n"
            '  "a": {\n'
            '    "backend": "claude",\n'
            '    "created_at": "2026-08-25T00:00:00Z",\n'
            '    "session_id": null,\n'
            '    "turns": 0\n'
            "  },\n"
            '  "b": {\n'
            '    "backend": "codex",\n'
            '    "created_at": "2026-08-25T00:00:00Z",\n'
            '    "session_id": null,\n'
            '    "turns": 0\n'
            "  }\n"
            "}\n"
        )

    def test_spawn_defaults_to_claude_backend(self, tmp_path: Path) -> None:
        orch = Orchestrator(state_file=tmp_path / "s.json")
        agent = orch.spawn("a1")
        assert agent.backend.name == "claude"

    def test_spawn_persists_a_grok_agent(self, tmp_path: Path) -> None:
        state_file = tmp_path / "s.json"
        orch = Orchestrator(state_file=state_file)
        agent = orch.spawn("g", "grok")
        assert agent.backend.name == "grok"
        loaded = Orchestrator(state_file=state_file)
        assert loaded.agents["g"].backend.name == "grok"
        assert loaded.agents["g"].session_id is None

    def test_spawn_persists_model_and_reasoning_effort(self, tmp_path: Path) -> None:
        state_file = tmp_path / "s.json"
        orch = Orchestrator(state_file=state_file)
        agent = orch.spawn(
            "configured",
            "codex",
            model="gpt-test",
            reasoning_effort="high",
        )
        assert agent.backend.model == "gpt-test"
        assert agent.backend.reasoning_effort == "high"

        loaded = Orchestrator(state_file=state_file).agents["configured"].backend
        assert loaded.name == "codex"
        assert loaded.model == "gpt-test"
        assert loaded.reasoning_effort == "high"

    def test_spawn_duplicate_raises(self, tmp_path: Path) -> None:
        orch = Orchestrator(state_file=tmp_path / "s.json")
        orch.spawn("a1", "claude")
        with pytest.raises(ValueError, match="already exists"):
            orch.spawn("a1", "claude")

    def test_ensure_creates_a_missing_agent(self, tmp_path: Path) -> None:
        state_file = tmp_path / "s.json"
        orch = Orchestrator(state_file=state_file)
        agent, created = orch.ensure("fresh", "echo")
        assert created is True
        assert agent.name == "fresh"
        assert agent.backend.name == "echo"
        assert Orchestrator(state_file=state_file).list_agents() == ["fresh"]

    def test_ensure_defaults_to_the_default_backend(self, tmp_path: Path) -> None:
        orch = Orchestrator(state_file=tmp_path / "s.json")
        agent, _created = orch.ensure("fresh")
        assert agent.backend.name == orchestrator.DEFAULT_BACKEND

    def test_ensure_returns_an_existing_agent_untouched(self, tmp_path: Path) -> None:
        """An existing agent keeps its backend and its session, not a new one."""
        state_file = tmp_path / "s.json"
        orch = Orchestrator(state_file=state_file)
        orch.spawn("a", "echo")
        orch.talk("a", "hi")
        agent, created = orch.ensure("a", "codex")
        assert created is False
        assert agent.backend.name == "echo"
        assert agent.session_id == "echo-sid"

    def test_ensure_takes_an_exclusive_lock(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The lookup and the create are one critical section, not two."""
        seen: list[int] = []

        def fake_flock(_fd: int, op: int) -> None:
            seen.append(op)

        monkeypatch.setattr(fcntl, "flock", fake_flock)
        Orchestrator(state_file=tmp_path / "s.json").ensure("a", "echo")
        assert fcntl.LOCK_EX in seen
        assert fcntl.LOCK_UN in seen

    def test_talk_unknown_raises(self, tmp_path: Path) -> None:
        orch = Orchestrator(state_file=tmp_path / "s.json")
        with pytest.raises(KeyError, match="no agent named"):
            orch.talk("nope", "hi")

    def test_list_agents(self, tmp_path: Path) -> None:
        orch = Orchestrator(state_file=tmp_path / "s.json")
        orch.spawn("b", "codex")
        orch.spawn("a", "claude")
        assert orch.list_agents() == ["a", "b"]

    def test_delete_agent(self, tmp_path: Path, monkeypatch) -> None:
        state_file = tmp_path / "state.json"
        # A literal, not orchestrator.WORKDIR: comparing the module constant
        # against itself would pass whatever it happened to be set to.
        workdir = tmp_path / "workdir"
        monkeypatch.setattr(orchestrator, "WORKDIR", workdir)

        def fake_backend_run(args, **kwargs):
            _assert_subprocess_kwargs(kwargs, workdir)
            return subprocess.CompletedProcess(
                args,
                0,
                stdout=json.dumps(
                    {"is_error": False, "session_id": "s1", "result": "reply"}
                ),
                stderr="",
            )

        monkeypatch.setattr(subprocess, "run", fake_backend_run)

        orch = Orchestrator(state_file=state_file)
        orch.spawn("a", "claude")
        orch.spawn("b", "codex")
        orch.delete("a")
        assert orch.list_agents() == ["b"]

        orch2 = Orchestrator(state_file=state_file)
        assert orch2.list_agents() == ["b"]

    def test_delete_unknown_raises(self, tmp_path: Path) -> None:
        orch = Orchestrator(state_file=tmp_path / "s.json")
        with pytest.raises(KeyError, match="no agent named"):
            orch.delete("nope")

    def test_reloaded_agent_keeps_the_name_from_state(self, tmp_path: Path) -> None:
        state_file = tmp_path / "s.json"
        Orchestrator(state_file=state_file).spawn("named", "echo")
        loaded = Orchestrator(state_file=state_file)
        assert loaded.agents["named"].name == "named"

    def test_agent_talk_resumes_on_the_same_instance(self, tmp_path: Path) -> None:
        seen: list[str | None] = []

        class Rec(AgentBackend):
            @property
            def name(self) -> str:
                return "rec"

            def run_turn(
                self,
                prompt: str,
                session_id: str | None,
                cwd: Path,
                timeout: int = DEFAULT_TURN_TIMEOUT,
                schema: OutputSchema | None = None,
            ) -> TurnResult:
                seen.append(session_id)
                return TurnResult(session_id="sid", reply="r", raw="")

        register_backend("rec", Rec)
        orch = Orchestrator(state_file=tmp_path / "s.json")
        orch.spawn("a", "rec")
        agent = orch.agents["a"]
        agent.talk("one")
        agent.talk("two")
        assert seen == [None, "sid"]

    def test_spawn_takes_an_exclusive_lock(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[int] = []

        def fake_flock(_fd: int, op: int) -> None:
            seen.append(op)

        monkeypatch.setattr(fcntl, "flock", fake_flock)
        Orchestrator(state_file=tmp_path / "s.json").spawn("a", "echo")
        assert fcntl.LOCK_EX in seen
        assert fcntl.LOCK_UN in seen
        assert (tmp_path / "s.json.lock").is_file()

    def test_talk_keeps_agents_spawned_during_the_turn(self, tmp_path: Path) -> None:
        """A long talk must not clobber a concurrent spawn (issue #2)."""
        state_file = tmp_path / "state.json"

        class MidTurnSpawn(AgentBackend):
            @property
            def name(self) -> str:
                return "midturn"

            def run_turn(
                self,
                prompt: str,
                session_id: str | None,
                cwd: Path,
                timeout: int = DEFAULT_TURN_TIMEOUT,
                schema: OutputSchema | None = None,
            ) -> TurnResult:
                Orchestrator(state_file=state_file).spawn("b", "echo")
                return TurnResult(session_id="s1", reply="ok", raw="")

        register_backend("midturn", MidTurnSpawn)
        orch = Orchestrator(state_file=state_file)
        orch.spawn("a", "midturn")
        orch.talk("a", "hi")
        reloaded = Orchestrator(state_file=state_file)
        assert reloaded.list_agents() == ["a", "b"]
        assert reloaded.agents["a"].session_id == "s1"

    def test_talk_holds_a_per_agent_lock_for_the_whole_turn(
        self, tmp_path: Path
    ) -> None:
        """Two turns on one agent must not both resume its session."""
        state_file = tmp_path / "state.json"
        held: list[tuple[bool, bool]] = []

        class ProbeLock(AgentBackend):
            @property
            def name(self) -> str:
                return "probelock"

            def run_turn(
                self,
                prompt: str,
                session_id: str | None,
                cwd: Path,
                timeout: int = DEFAULT_TURN_TIMEOUT,
                schema: OutputSchema | None = None,
            ) -> TurnResult:
                probe = Orchestrator(state_file=state_file)
                held.append(
                    (
                        _flock_is_held(probe._agent_lock_path("a")),
                        _flock_is_held(probe._agent_lock_path("other")),
                    )
                )
                return TurnResult(session_id="s1", reply="ok", raw="")

        register_backend("probelock", ProbeLock)
        orch = Orchestrator(state_file=state_file)
        orch.spawn("a", "probelock")
        orch.talk("a", "hi")
        # Locked for this agent only, and released once the turn is over.
        assert held == [(True, False)]
        assert _flock_is_held(orch._agent_lock_path("a")) is False

    def test_agent_lock_paths_sit_in_the_locks_dir(self, tmp_path: Path) -> None:
        orch = Orchestrator(state_file=tmp_path / "state.json")
        first = orch._agent_lock_path("a")
        assert first.parent == tmp_path / "state.json.locks"
        assert first.name == hashlib.sha256(b"a").hexdigest()
        assert first != orch._agent_lock_path("b")
        assert first != orch._lock_path()

    def test_a_backend_reporting_no_session_keeps_the_stored_one(
        self, tmp_path: Path
    ) -> None:
        """Overwriting a live session id with None restarts the conversation."""
        state_file = tmp_path / "state.json"
        replies = iter([TurnResult("s1", "first", ""), TurnResult(None, "second", "")])

        class Forgetful(AgentBackend):
            @property
            def name(self) -> str:
                return "forgetful"

            def run_turn(
                self,
                prompt: str,
                session_id: str | None,
                cwd: Path,
                timeout: int = DEFAULT_TURN_TIMEOUT,
                schema: OutputSchema | None = None,
            ) -> TurnResult:
                return next(replies)

        register_backend("forgetful", Forgetful)
        orch = Orchestrator(state_file=state_file)
        orch.spawn("a", "forgetful")
        orch.talk("a", "one")
        assert orch.talk("a", "two").reply == "second"
        assert orch.agents["a"].session_id == "s1"
        assert Orchestrator(state_file=state_file).agents["a"].session_id == "s1"

    def test_corrupt_state_names_the_file(self, tmp_path: Path) -> None:
        state_file = tmp_path / "state.json"
        state_file.write_text("{", encoding="utf-8")
        with pytest.raises(orchestrator.StateError) as excinfo:
            Orchestrator(state_file=state_file)
        assert str(excinfo.value).startswith(f"{state_file} is not valid JSON: ")

    def test_state_entry_without_a_backend_is_reported(self, tmp_path: Path) -> None:
        state_file = tmp_path / "state.json"
        state_file.write_text('{"a": {"session_id": "s1"}}', encoding="utf-8")
        with pytest.raises(orchestrator.StateError) as excinfo:
            Orchestrator(state_file=state_file)
        assert excinfo.value.args[0] == f"{state_file}: agent 'a' has no backend"

    def test_state_naming_an_unknown_backend_is_reported(self, tmp_path: Path) -> None:
        state_file = tmp_path / "state.json"
        state_file.write_text('{"a": {"backend": "ghost"}}', encoding="utf-8")
        with pytest.raises(UnknownBackendError, match="Unknown backend 'ghost'"):
            Orchestrator(state_file=state_file)

    def test_talk_fails_if_the_agent_is_deleted_during_the_turn(
        self, tmp_path: Path
    ) -> None:
        state_file = tmp_path / "state.json"

        class DeleteDuring(AgentBackend):
            @property
            def name(self) -> str:
                return "delduring"

            def run_turn(
                self,
                prompt: str,
                session_id: str | None,
                cwd: Path,
                timeout: int = DEFAULT_TURN_TIMEOUT,
                schema: OutputSchema | None = None,
            ) -> TurnResult:
                Orchestrator(state_file=state_file).delete("a")
                return TurnResult(session_id="s1", reply="ok", raw="")

        register_backend("delduring", DeleteDuring)
        orch = Orchestrator(state_file=state_file)
        orch.spawn("a", "delduring")
        with pytest.raises(KeyError, match="no agent named 'a'"):
            orch.talk("a", "hi")
        assert Orchestrator(state_file=state_file).list_agents() == []

    def test_delete_on_an_idle_agent_unlinks_its_lock_file(
        self, tmp_path: Path
    ) -> None:
        state_file = tmp_path / "state.json"
        orch = Orchestrator(state_file=state_file)
        orch.spawn("a", "echo")
        orch.talk("a", "hi")
        lock_path = orch._agent_lock_path("a")
        assert lock_path.is_file()

        orch.delete("a")
        assert not lock_path.exists()

    def test_delete_during_its_own_turn_keeps_the_lock_file(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        state_file = tmp_path / "state.json"
        entered = threading.Event()
        release = threading.Event()

        register_backend("gate-self", _gate_backend("gate-self", entered, release))
        orch = Orchestrator(state_file=state_file)
        orch.spawn("a", "gate-self")

        def run_turn() -> None:
            # Deleting the agent mid-turn (below) makes the registry entry
            # this turn resumes into vanish before it persists, which is the
            # pre-existing AgentNotFoundError path this test isn't about —
            # the reclaim-on-that-path is covered separately, below.
            with pytest.raises(orchestrator.AgentNotFoundError):
                orch.talk("a", "hi")

        # daemon=True on both threads below: several asserts run before
        # their matching join(), and one firing must not leave a thread
        # running past this test.
        turn_thread = threading.Thread(target=run_turn, daemon=True)
        turn_thread.start()
        assert entered.wait(timeout=5)
        lock_path = orch._agent_lock_path("a")
        assert lock_path.is_file()

        # The probe loses to the in-flight turn and backs off rather than
        # refusing: `delete` never fails and never changes its exit code.
        # Run on its own thread and asserted to complete quickly, not
        # called synchronously here: a probe that regressed into blocking
        # (dropping delete's LOCK_NB) would deadlock this test against
        # `release` below instead of failing it — the whole point of a
        # non-blocking probe is that `delete` never waits for someone
        # else's turn, so that has to be checked, not assumed.
        results: list[Agent] = []
        delete_thread = threading.Thread(
            target=lambda: results.append(orch.delete("a")), daemon=True
        )
        with caplog.at_level("DEBUG", logger="orchestrator"):
            delete_thread.start()
            delete_thread.join(timeout=5)
        assert not delete_thread.is_alive(), "delete blocked instead of backing off"
        agent = results[0]
        assert agent.name == "a"
        assert lock_path.is_file()
        assert "agent 'a': lock file in use, not reclaiming" in [
            r.getMessage() for r in caplog.records
        ]

        release.set()
        turn_thread.join(timeout=5)
        # Once the turn ends, its own AgentNotFoundError handler reclaims
        # what delete's probe could not — see the next test.
        assert not lock_path.exists()

    def test_a_turn_reclaims_its_lock_when_deleted_mid_turn(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        state_file = tmp_path / "state.json"

        class DeleteDuring(AgentBackend):
            @property
            def name(self) -> str:
                return "delduring-reclaim"

            def run_turn(
                self,
                prompt: str,
                session_id: str | None,
                cwd: Path,
                timeout: int = DEFAULT_TURN_TIMEOUT,
                schema: OutputSchema | None = None,
            ) -> TurnResult:
                # Runs with the outer talk() holding the agent lock, so this
                # delete's own reclaim probe backs off (BlockingIOError,
                # caught inside _reclaim_agent_lock) rather than blocking on
                # a lock this very thread already holds through a different
                # fd — flock() isn't reentrant, so a probe that regressed
                # into blocking would deadlock this thread against itself,
                # not just wait. Run on its own thread and joined with a
                # timeout so that shows up as a clean failure instead; the
                # file is left for talk()'s AgentNotFoundError handler to
                # clean up.
                delete_thread = threading.Thread(
                    target=lambda: Orchestrator(state_file=state_file).delete("a"),
                    daemon=True,
                )
                delete_thread.start()
                delete_thread.join(timeout=5)
                assert not delete_thread.is_alive(), (
                    "delete blocked instead of backing off"
                )
                return TurnResult(session_id="s1", reply="ok", raw="")

        register_backend("delduring-reclaim", DeleteDuring)
        orch = Orchestrator(state_file=state_file)
        orch.spawn("a", "delduring-reclaim")
        with (
            caplog.at_level("DEBUG", logger="orchestrator"),
            pytest.raises(orchestrator.AgentNotFoundError),
        ):
            orch.talk("a", "hi")
        assert list(orch._locks_dir().iterdir()) == []
        # Asserted through the log line, not just the empty directory: an
        # agent that was simply never talked to also leaves it empty.
        assert "agent 'a': reclaimed lock file, no such agent" in [
            r.getMessage() for r in caplog.records
        ]

    def test_delete_during_a_sibling_agents_turn_still_reclaims(
        self, tmp_path: Path
    ) -> None:
        """Rules out a home-wide lock: only the deleted agent's own turn
        should block a reclaim, not an unrelated agent's."""
        state_file = tmp_path / "state.json"
        entered = threading.Event()
        release = threading.Event()

        register_backend(
            "gate-sibling", _gate_backend("gate-sibling", entered, release)
        )
        orch = Orchestrator(state_file=state_file)
        orch.spawn("bob", "gate-sibling")
        orch.spawn("alice", "echo")
        orch.talk("alice", "warm up")

        # daemon=True: the asserts below run before this thread's own
        # join(), and one firing must not leave it running past this test.
        turn_thread = threading.Thread(
            target=lambda: orch.talk("bob", "hi"), daemon=True
        )
        turn_thread.start()
        assert entered.wait(timeout=5)

        alice_lock = orch._agent_lock_path("alice")
        assert alice_lock.is_file()
        orch.delete("alice")
        assert not alice_lock.exists()

        release.set()
        turn_thread.join(timeout=5)

    def test_a_reclaim_racing_its_own_toctou_does_not_evict_a_fresh_acquirer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression for a TOCTOU in `_reclaim_agent_lock`.

        `_flock`'s open() and its flock() are two syscalls, not one: a probe
        can end up holding a lock on an inode a *different*, legitimate
        reclaim already orphaned in between, while a live acquirer has since
        put a fresh file at the same path. Unlinking blindly at that point
        destroys the live acquirer's file instead of the dead one the probe
        actually holds.

        Exercises the real `_reclaim_agent_lock`, not a hand-rolled stand-in:
        driving `_flock` directly instead misses this entirely, since
        nothing about a bare `_flock` call reproduces the probe's own
        open()-then-flock() window.
        """
        orch = Orchestrator(state_file=tmp_path / "state.json")
        lock_path = orch._agent_lock_path("bob")
        record, state = _overlap_recorder(lock_path)

        # Pauses the reclaimer thread's first flock() call, after its
        # `_flock` has already open()ed the path — the window the bug lives
        # in — so a second, unpaused reclaim can legitimately acquire,
        # unlink, and release that same inode before this one's flock()
        # finally runs and is granted on the now-dead inode.
        reclaimer_opened, let_reclaimer_flock = _pause_first_flock(
            monkeypatch, "reclaimer"
        )
        waiter_entered = threading.Event()

        def reclaimer() -> None:
            orch._reclaim_agent_lock("bob")

        def waiter() -> None:
            with orch._agent_lock("bob"):
                record("waiter", entered=waiter_entered)

        def late_arrival() -> None:
            with orch._agent_lock("bob"):
                record("late")

        # daemon=True: several asserts below run before their matching
        # join(); one firing must not leave a thread running past this test
        # (a paused one, via _pause_first_flock, would otherwise survive it
        # by however long its own wait(timeout=...) takes to give up).
        t_reclaim = threading.Thread(target=reclaimer, name="reclaimer", daemon=True)
        t_reclaim.start()
        assert reclaimer_opened.wait(timeout=5)

        # A second, unpaused reclaim on the same (still-existing) inode the
        # paused thread already opened: it acquires, finds it live, unlinks,
        # and releases — a plain `delete` on an idle agent.
        orch._reclaim_agent_lock("bob")

        t_wait = threading.Thread(target=waiter, daemon=True)
        t_wait.start()
        assert waiter_entered.wait(timeout=5)

        let_reclaimer_flock.set()
        t_reclaim.join(timeout=5)

        t_late = threading.Thread(target=late_arrival, daemon=True)
        t_late.start()
        t_wait.join(timeout=5)
        t_late.join(timeout=5)

        assert not state.overlapped
        assert state.inodes["waiter"] == state.inodes["late"]

    def test_a_waiter_blocked_on_a_reclaimed_inode_reacquires_without_overlap(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression for the original race in the PR description: a
        distinct mechanism from the TOCTOU above, and from a genuinely
        distinct bug. Replacing this test with the TOCTOU one above,
        instead of adding it alongside, would leave `_agent_lock`'s
        `revalidate=True` completely untested.

        `_agent_lock`'s `flock()` can be granted on an inode a reclaimer has
        already unlinked and released out from under it — `flock` binds to
        the inode, not the path. `_flock`'s `revalidate=True` loop is what
        notices via `_is_live` and re-acquires instead of proceeding on a
        dead lock; disabling `revalidate`, or making `_is_live` treat a
        vanished path as still live, both make this test fail.
        """
        orch = Orchestrator(state_file=tmp_path / "state.json")
        lock_path = orch._agent_lock_path("bob")
        record, state = _overlap_recorder(lock_path)

        # Pauses the waiter's first flock() call, after its `_agent_lock`
        # has already open()ed the still-existing path — so it shares
        # whatever inode the reclaim below is about to unlink and release.
        waiter_opened, let_waiter_flock = _pause_first_flock(monkeypatch, "waiter")
        waiter_entered = threading.Event()

        def waiter() -> None:
            with orch._agent_lock("bob"):
                record("waiter", entered=waiter_entered)

        def late_arrival() -> None:
            with orch._agent_lock("bob"):
                record("late")

        # daemon=True: see the same note in the TOCTOU test above.
        t_wait = threading.Thread(target=waiter, name="waiter", daemon=True)
        t_wait.start()
        assert waiter_opened.wait(timeout=5)

        # A plain reclaim on the same (still-existing) inode the paused
        # waiter already opened: it acquires, finds it live, unlinks, and
        # releases — the file the waiter is about to be granted a lock on.
        orch._reclaim_agent_lock("bob")
        let_waiter_flock.set()
        assert waiter_entered.wait(timeout=5)

        t_late = threading.Thread(target=late_arrival, daemon=True)
        t_late.start()
        t_wait.join(timeout=5)
        t_late.join(timeout=5)

        assert not state.overlapped
        assert state.inodes["waiter"] == state.inodes["late"]

    def test_talk_on_an_unknown_agent_leaves_no_lock_file(self, tmp_path: Path) -> None:
        """Written against `Orchestrator.talk` directly: `cmd_talk` creates
        the agent first, so `main(["talk", ...])` would assert something
        false here."""
        orch = Orchestrator(state_file=tmp_path / "state.json")
        with pytest.raises(orchestrator.AgentNotFoundError):
            orch.talk("ghost", "hi")
        assert list(orch._locks_dir().iterdir()) == []

    def test_locks_dir_holds_one_file_per_live_agent(self, tmp_path: Path) -> None:
        state_file = tmp_path / "state.json"
        orch = Orchestrator(state_file=state_file)
        orch.spawn("a", "echo")
        orch.spawn("b", "echo")
        orch.talk("a", "hi")
        orch.talk("b", "hi")

        locks_dir = orch._locks_dir()
        assert sorted(p.name for p in locks_dir.iterdir()) == sorted(
            path.name
            for path in (orch._agent_lock_path("a"), orch._agent_lock_path("b"))
        )

        orch.delete("a")
        orch.delete("b")
        assert list(locks_dir.iterdir()) == []


class TestCLI:
    """cmd_* dispatch and printed output, backed by a fake CLI-free backend."""

    def test_cmd_create_prints_confirmation(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        orch = Orchestrator(state_file=tmp_path / "s.json")
        _cmd_create(
            orch,
            _options(
                [
                    "create",
                    "a",
                    "-b",
                    "echo",
                    "--model",
                    "test-model",
                    "--reasoning-effort",
                    "high",
                ]
            ),
        )
        assert capsys.readouterr().out == "created agent 'a' backend=echo\n"
        assert orch.agents["a"].backend.model == "test-model"
        assert orch.agents["a"].backend.reasoning_effort == "high"

    def test_cmd_create_defaults_to_claude_backend(self, tmp_path: Path) -> None:
        orch = Orchestrator(state_file=tmp_path / "s.json")
        _cmd_create(orch, _options(["create", "a"]))
        assert orch.agents["a"].backend.name == "claude"

    def test_cmd_create_follows_default_backend(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DEFAULT_BACKEND documents itself as the one `create` uses.

        An argparse default of its own would leave `create` on claude however
        that constant changed, so the two would silently disagree.
        """
        monkeypatch.setattr(orchestrator, "DEFAULT_BACKEND", "codex")
        orch = Orchestrator(state_file=tmp_path / "s.json")
        _cmd_create(orch, _options(["create", "a"]))
        assert orch.agents["a"].backend.name == "codex"

    def test_cmd_create_rejects_unknown_backend(self, tmp_path: Path) -> None:
        orch = Orchestrator(state_file=tmp_path / "s.json")
        with pytest.raises(SystemExit, match="2"):
            _cmd_create(orch, _options(["create", "a", "-b", "not-a-backend"]))

    def test_cmd_create_missing_name_reports_its_own_prog(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        orch = Orchestrator(state_file=tmp_path / "s.json")
        with pytest.raises(SystemExit, match="2"):
            _cmd_create(orch, _options(["create"]))
        assert capsys.readouterr().err.startswith("usage: orchestrator create ")

    def test_cmd_talk_prints_reply(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        orch = Orchestrator(state_file=tmp_path / "s.json")
        orch.spawn("a", "echo")
        _cmd_talk(orch, _talk_options(["talk", "a", "--", "hello", "there"]))
        out = capsys.readouterr().out
        assert "session=echo-sid" in out
        assert "echo:hello there" in out

    def test_cmd_talk_creates_a_missing_agent(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Talking to a name that does not exist spawns it and runs the turn."""
        monkeypatch.setattr(orchestrator, "DEFAULT_BACKEND", "echo")
        state_file = tmp_path / "s.json"
        orch = Orchestrator(state_file=state_file)
        _cmd_talk(orch, _talk_options(["talk", "fresh", "--", "hello"]))
        captured = capsys.readouterr()
        assert captured.err == "created agent 'fresh' backend=echo\n"
        assert "echo:hello" in captured.out
        assert Orchestrator(state_file=state_file).list_agents() == ["fresh"]

    def test_cmd_talk_says_nothing_extra_for_an_existing_agent(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        orch = Orchestrator(state_file=tmp_path / "s.json")
        orch.spawn("a", "echo")
        _cmd_talk(orch, _talk_options(["talk", "a", "--", "hello"]))
        assert capsys.readouterr().err == ""

    def test_cmd_talk_flags_create_exact_configuration(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        orch = Orchestrator(state_file=tmp_path / "s.json")
        _cmd_talk(
            orch,
            _talk_options(
                [
                    "talk",
                    "-b",
                    "echo",
                    "--model",
                    "m",
                    "--reasoning-effort",
                    "high",
                    "fresh",
                    "-p",
                    "hello",
                ],
            ),
        )
        assert capsys.readouterr().err == "created agent 'fresh' backend=echo\n"
        assert orchestrator._agent_config(orch.agents["fresh"]) == ("echo", "m", "high")

    def test_cmd_talk_matching_flags_reuse_silently(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        orch = Orchestrator(state_file=tmp_path / "s.json")
        orch.spawn("a", "echo", model="m", reasoning_effort="high")
        _cmd_talk(
            orch,
            _talk_options(
                [
                    "talk",
                    "-b",
                    "echo",
                    "--model",
                    "m",
                    "--reasoning-effort",
                    "high",
                    "a",
                    "-p",
                    "hi",
                ]
            ),
        )
        assert capsys.readouterr().err == ""

    def test_cmd_talk_mismatch_reports_the_exact_configuration_tuple(
        self, tmp_path: Path
    ) -> None:
        orch = Orchestrator(state_file=tmp_path / "s.json")
        orch.spawn("a", "echo", model="first", reasoning_effort="high")
        with pytest.raises(
            orchestrator.OrchestratorError,
            match=r"agent 'a' already uses backend/model/effort \('echo', 'first', 'high'\); configured \('codex', None, None\)",
        ):
            _cmd_talk(orch, _talk_options(["talk", "-b", "codex", "a", "-p", "hi"]))

    def test_cmd_talk_omitting_model_still_asserts_the_exact_tuple(
        self, tmp_path: Path
    ) -> None:
        orch = Orchestrator(state_file=tmp_path / "s.json")
        orch.spawn("a", "echo", model="stored")
        with pytest.raises(
            orchestrator.OrchestratorError,
            match=r"configured \('echo', None, None\)$",
        ):
            _cmd_talk(orch, _talk_options(["talk", "-b", "echo", "a", "-p", "hi"]))

    def test_cmd_talk_without_config_flags_reuses_any_stored_config(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        orch = Orchestrator(state_file=tmp_path / "s.json")
        orch.spawn("a", "echo", model="stored", reasoning_effort="high")
        _cmd_talk(orch, _talk_options(["talk", "a", "--", "hi"]))
        assert capsys.readouterr().err == ""

    def test_cmd_talk_rejects_unknown_backend(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit, match="2"):
            _talk_options(["talk", "-b", "unknown", "a", "-p", "hi"])

    def test_talk_flags_after_name_are_rejected_before_agent_creation(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        state_file = tmp_path / "s.json"
        monkeypatch.setattr(orchestrator, "STATE_FILE", state_file)
        monkeypatch.setattr(
            orchestrator,
            "Orchestrator",
            lambda: (_ for _ in ()).throw(AssertionError("constructed")),
        )
        with pytest.raises(SystemExit, match="2"):
            main(["talk", "a", "hi", "--timeout", "5"])
        assert "usage: orchestrator talk " in capsys.readouterr().err
        assert not state_file.exists()

    def test_cmd_talk_empty_prompt_creates_nothing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A rejected invocation must not leave an agent behind."""
        state_file = tmp_path / "s.json"
        with pytest.raises(SystemExit, match="2"):
            _talk_options(["talk", "fresh", "-p", "   "])
        assert Orchestrator(state_file=state_file).list_agents() == []

    def test_cmd_talk_empty_prompt_exits_nonzero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Exit 0 here reads as a turn that ran to a caller under `set -e`."""
        orch = Orchestrator(state_file=tmp_path / "s.json")
        orch.spawn("a", "echo")
        with pytest.raises(SystemExit, match="2"):
            _talk_options(["talk", "a", "-p", "   "])
        captured = capsys.readouterr()
        assert "usage: orchestrator talk " in captured.err
        assert "talk prompt must not be empty" in captured.err
        assert captured.out == ""

    def test_cmd_talk_missing_name_reports_its_own_prog(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit, match="2"):
            _options(["talk"])
        assert capsys.readouterr().err.startswith("usage: orchestrator talk ")

    def test_cmd_talk_prints_backend_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        class BoomBackend(AgentBackend):
            @property
            def name(self) -> str:
                return "boom"

            def run_turn(
                self,
                prompt: str,
                session_id: str | None,
                cwd: Path,
                timeout: int = DEFAULT_TURN_TIMEOUT,
                schema: OutputSchema | None = None,
            ) -> TurnResult:
                raise ClaudeTurnError("claude output was not JSON")

        register_backend("boom", BoomBackend)
        orch = Orchestrator(state_file=tmp_path / "s.json")
        orch.spawn("b", "boom")
        with pytest.raises(SystemExit, match="1"):
            _cmd_talk(orch, _talk_options(["talk", "b", "-p", "hi"]))
        captured = capsys.readouterr()
        assert captured.err == "claude output was not JSON\n"
        assert captured.out == ""

    def test_cmd_talk_prints_codex_backend_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        class BoomCodex(AgentBackend):
            @property
            def name(self) -> str:
                return "boomcodex"

            def run_turn(
                self,
                prompt: str,
                session_id: str | None,
                cwd: Path,
                timeout: int = DEFAULT_TURN_TIMEOUT,
                schema: OutputSchema | None = None,
            ) -> TurnResult:
                raise CodexTurnError("codex did not report a thread_id")

        register_backend("boomcodex", BoomCodex)
        orch = Orchestrator(state_file=tmp_path / "s.json")
        orch.spawn("c", "boomcodex")
        with pytest.raises(SystemExit, match="1"):
            _cmd_talk(orch, _talk_options(["talk", "c", "-p", "hi"]))
        assert capsys.readouterr().err == "codex did not report a thread_id\n"

    def test_cmd_talk_prints_any_turn_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The CLI catches TurnError, so a new backend does not need a new except."""

        class BoomAny(AgentBackend):
            @property
            def name(self) -> str:
                return "boomany"

            def run_turn(
                self,
                prompt: str,
                session_id: str | None,
                cwd: Path,
                timeout: int = DEFAULT_TURN_TIMEOUT,
                schema: OutputSchema | None = None,
            ) -> TurnResult:
                raise TurnError("cli failed")

        register_backend("boomany", BoomAny)
        orch = Orchestrator(state_file=tmp_path / "s.json")
        orch.spawn("d", "boomany")
        with pytest.raises(SystemExit, match="1"):
            _cmd_talk(orch, _talk_options(["talk", "d", "-p", "hi"]))
        captured = capsys.readouterr()
        assert captured.err == "cli failed\n"
        assert captured.out == ""

    def test_cmd_talk_prints_grok_backend_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        class BoomGrok(AgentBackend):
            @property
            def name(self) -> str:
                return "boomgrok"

            def run_turn(
                self,
                prompt: str,
                session_id: str | None,
                cwd: Path,
                timeout: int = DEFAULT_TURN_TIMEOUT,
                schema: OutputSchema | None = None,
            ) -> TurnResult:
                raise GrokTurnError("grok did not report a sessionId")

        register_backend("boomgrok", BoomGrok)
        orch = Orchestrator(state_file=tmp_path / "s.json")
        orch.spawn("g", "boomgrok")
        with pytest.raises(SystemExit, match="1"):
            _cmd_talk(orch, _talk_options(["talk", "g", "-p", "hi"]))
        assert capsys.readouterr().err == "grok did not report a sessionId\n"

    def test_cmd_create_accepts_grok(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        orch = Orchestrator(state_file=tmp_path / "s.json")
        _cmd_create(orch, _options(["create", "a", "-b", "grok"]))
        assert capsys.readouterr().out == "created agent 'a' backend=grok\n"
        assert orch.agents["a"].backend.name == "grok"

    def test_cmd_list_empty(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        state_file = tmp_path / "s.json"
        orch = Orchestrator(state_file=state_file)
        _cmd_list(orch, _options(["list"]))
        assert capsys.readouterr().out == f"registry: {state_file}\nno agents\n"

    def test_cmd_list_prints_each_agent(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        orch = Orchestrator(state_file=tmp_path / "s.json")
        orch.spawn("a", "echo")
        _cmd_list(orch, _options(["list"]))
        out = capsys.readouterr().out
        assert "a" in out
        assert "backend=echo" in out
        assert "session=-" in out

    def test_cmd_list_rejects_unexpected_arg_reporting_its_own_prog(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit, match="2"):
            _options(["list", "unexpected"])
        assert capsys.readouterr().err.startswith("usage: orchestrator list ")

    def test_cmd_delete_prints_confirmation(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        orch = Orchestrator(state_file=tmp_path / "s.json")
        orch.spawn("a", "echo")
        _cmd_delete(orch, _options(["delete", "a"]))
        assert "deleted agent 'a' backend=echo" in capsys.readouterr().out

    def test_cmd_delete_missing_name_reports_its_own_prog(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # `name` is nargs="?" now (`--team T` alone tears a team down), so
        # argparse itself no longer rejects a bare `delete`; main() does.
        with pytest.raises(SystemExit, match="2"):
            main(["delete"])
        assert capsys.readouterr().err.startswith("usage: orchestrator delete ")

    def test_main_no_args_prints_usage_and_exits(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit, match="2"):
            main([])
        captured = capsys.readouterr()
        assert captured.err.startswith("usage: orchestrator ")
        assert "the following arguments are required: <verb>" in captured.err
        assert captured.out == ""

    @pytest.mark.parametrize("flags", [[], ["-v"], ["--verbose"], ["-vv"], ["-vvv"]])
    def test_main_version_is_clean_and_early(
        self,
        flags: list[str],
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        def fail_orchestrator() -> None:
            raise AssertionError("version must not construct Orchestrator")

        monkeypatch.setattr(orchestrator, "Orchestrator", fail_orchestrator)
        run_version([*flags, "--version", "ignored"])
        captured = capsys.readouterr()
        assert captured.out == "0.1.0\n"
        assert captured.err == ""

    def test_main_version_prefers_checkout_metadata(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(
            orchestrator.importlib.metadata,
            "version",
            lambda _: "installed-version",
        )
        run_version()
        assert capsys.readouterr().out == "0.1.0\n"

    def test_main_version_uses_installed_metadata_as_fallback(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(orchestrator, "_project_version", lambda: None)
        monkeypatch.setattr(
            orchestrator.importlib.metadata,
            "version",
            lambda _: "installed-version",
        )
        run_version()
        assert capsys.readouterr().out == "installed-version\n"

    def test_main_version_rejects_missing_installed_metadata(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(orchestrator, "_project_version", lambda: None)
        monkeypatch.setattr(orchestrator.importlib.metadata, "version", lambda _: None)
        with pytest.raises(SystemExit, match="1"):
            main(["--version"])
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == "unable to determine agents-army version\n"

    def test_project_version_ignores_unreadable_metadata(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def unreadable(*args: object, **kwargs: object) -> object:
            raise OSError("metadata unavailable")

        monkeypatch.setattr(Path, "open", unreadable)
        assert orchestrator._project_version() is None

    def test_main_version_falls_back_for_invalid_utf8_metadata(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def invalid_metadata(*_args: object, **_kwargs: object) -> io.BytesIO:
            return io.BytesIO(b"\xff")

        monkeypatch.setattr(Path, "open", invalid_metadata)
        monkeypatch.setattr(
            orchestrator.importlib.metadata,
            "version",
            lambda _: "installed-version",
        )
        run_version()
        assert capsys.readouterr().out == "installed-version\n"

    def test_main_version_reports_unavailable_without_traceback(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(orchestrator, "_project_version", lambda: None)

        def missing(_: str) -> str:
            raise orchestrator.importlib.metadata.PackageNotFoundError

        monkeypatch.setattr(orchestrator.importlib.metadata, "version", missing)
        with pytest.raises(SystemExit, match="1"):
            main(["--version"])
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == "unable to determine agents-army version\n"
        assert "Traceback" not in captured.err

    def test_main_version_after_command_remains_prompt_data(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(orchestrator, "STATE_FILE", tmp_path / "s.json")
        monkeypatch.setattr(orchestrator, "DEFAULT_BACKEND", "echo")
        run_version(["talk", "agent", "a", "--version"])
        assert capsys.readouterr().out == "0.1.0\n"

    @pytest.mark.parametrize("flags", [[], ["-v"], ["--verbose"], ["-vv"], ["-vvv"]])
    def test_main_dependency_check_is_clean_and_early(
        self,
        flags: list[str],
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        def fail_orchestrator() -> None:
            raise AssertionError("dependency check must not construct Orchestrator")

        monkeypatch.setattr(orchestrator, "Orchestrator", fail_orchestrator)
        _fake_dependency_env(monkeypatch, ALL_TOOLS_PRESENT)
        main([*flags, "doctor"])
        captured = capsys.readouterr()
        assert captured.out == (
            f"\u2713 Python {_running_python()}\n"
            "\u2713 uv 0.4.18\n"
            "\u2713 claude 2.1.234 (Claude Code)\n"
            "\u2713 codex-cli 0.147.0\n"
            "\u2713 grok 1.0.5\n"
            "\u2713 opencode 1.18.21\n"
            "\u25cb jq 1.7 (optional)\n"
        )
        assert captured.err == ""

    def test_dependency_check_probes_each_tool_with_its_own_version_flag(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        calls = _fake_dependency_env(monkeypatch, ALL_TOOLS_PRESENT)
        main(["doctor"])
        assert calls == [
            ["uv", "--version"],
            ["claude", "--version"],
            ["codex", "--version"],
            ["grok", "--version"],
            ["opencode", "--version"],
            ["jq", "--version"],
        ]
        assert capsys.readouterr().err == ""

    def test_dependency_check_reports_every_missing_agent_cli_and_still_exits_zero(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Zero usable backends is a report, not a failure."""
        _fake_dependency_env(monkeypatch, {"uv": "uv 0.4.18", "jq": "jq-1.7"})
        main(["doctor"])
        captured = capsys.readouterr()
        assert captured.out == (
            f"\u2713 Python {_running_python()}\n"
            "\u2713 uv 0.4.18\n"
            "\u2717 claude (not found)\n"
            "\u2717 codex (not found)\n"
            "\u2717 grok (not found)\n"
            "\u2717 opencode (not found)\n"
            "\u25cb jq 1.7 (optional)\n"
        )
        assert captured.err == ""

    def test_dependency_check_exits_zero_with_nothing_installed(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _fake_dependency_env(monkeypatch, {})
        main(["doctor"])
        captured = capsys.readouterr()
        assert captured.out == (
            f"\u2713 Python {_running_python()}\n"
            "\u2717 uv (not found)\n"
            "\u2717 claude (not found)\n"
            "\u2717 codex (not found)\n"
            "\u2717 grok (not found)\n"
            "\u2717 opencode (not found)\n"
            "\u2717 jq (not found, optional)\n"
        )
        assert captured.err == ""

    def test_dependency_check_marks_absent_jq_optional_and_leaves_the_rest_alone(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        present = {
            tool: version for tool, version in ALL_TOOLS_PRESENT.items() if tool != "jq"
        }
        _fake_dependency_env(monkeypatch, present)
        main(["doctor"])
        captured = capsys.readouterr()
        assert captured.out == (
            f"\u2713 Python {_running_python()}\n"
            "\u2713 uv 0.4.18\n"
            "\u2713 claude 2.1.234 (Claude Code)\n"
            "\u2713 codex-cli 0.147.0\n"
            "\u2713 grok 1.0.5\n"
            "\u2713 opencode 1.18.21\n"
            "\u2717 jq (not found, optional)\n"
        )
        assert captured.err == ""

    def test_dependency_check_keeps_a_silent_tool_on_its_own_line(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Empty stdout means installed, version unknown — not missing."""
        _fake_dependency_env(monkeypatch, {"uv": "", "jq": "\n"})
        main(["doctor"])
        captured = capsys.readouterr()
        assert captured.out == (
            f"\u2713 Python {_running_python()}\n"
            "\u2713 uv (version unknown)\n"
            "\u2717 claude (not found)\n"
            "\u2717 codex (not found)\n"
            "\u2717 grok (not found)\n"
            "\u2717 opencode (not found)\n"
            "\u25cb jq (version unknown, optional)\n"
        )
        assert captured.err == ""

    def test_dependency_check_reads_only_the_first_line_of_a_version(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _fake_dependency_env(monkeypatch, {"uv": "  uv 0.4.18  \nbuild deadbeef\n"})
        main(["doctor"])
        assert "\u2713 uv 0.4.18\n" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "failure",
        [
            OSError("no such file"),
            subprocess.TimeoutExpired(cmd=["uv", "--version"], timeout=5),
            subprocess.SubprocessError("spawn failed"),
        ],
    )
    def test_dependency_check_survives_a_failing_version_probe(
        self,
        failure: Exception,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            orchestrator.shutil, "which", lambda tool: f"/usr/bin/{tool}"
        )

        def explode(args, **kwargs):
            raise failure

        monkeypatch.setattr(orchestrator.subprocess, "run", explode)
        main(["doctor"])
        captured = capsys.readouterr()
        assert captured.out == (
            f"\u2713 Python {_running_python()}\n"
            "\u2713 uv (version unknown)\n"
            "\u2713 claude (version unknown)\n"
            "\u2713 codex (version unknown)\n"
            "\u2713 grok (version unknown)\n"
            "\u2713 opencode (version unknown)\n"
            "\u25cb jq (version unknown, optional)\n"
        )
        assert captured.err == ""

    def test_dependency_check_ignores_a_version_probe_that_exits_nonzero(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(orchestrator.shutil, "which", lambda _tool: "/usr/bin/uv")

        def refuse(args, **kwargs):
            return subprocess.CompletedProcess(args, 1, stdout="uv 0.4.18", stderr="")

        monkeypatch.setattr(orchestrator.subprocess, "run", refuse)
        main(["doctor"])
        captured = capsys.readouterr()
        assert "\u2713 uv (version unknown)\n" in captured.out
        assert "0.4.18" not in captured.out
        assert captured.err == ""

    def test_dependency_check_reports_an_interpreter_below_the_floor(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _fake_dependency_env(monkeypatch, {})
        monkeypatch.setattr(orchestrator.sys, "version_info", (3, 10, 2, "final", 0))
        main(["doctor"])
        assert capsys.readouterr().out.startswith(
            "\u2717 Python 3.10.2 (needs 3.11+)\n"
        )

    def test_dependency_check_accepts_an_interpreter_exactly_at_the_floor(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """3.11.0 is the floor, not the first version above it."""
        _fake_dependency_env(monkeypatch, {})
        monkeypatch.setattr(orchestrator.sys, "version_info", (3, 11, 0, "final", 0))
        main(["doctor"])
        assert capsys.readouterr().out.startswith("\u2713 Python 3.11.0\n")

    def test_dependency_check_reports_a_later_major_as_satisfying_the_floor(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _fake_dependency_env(monkeypatch, {})
        monkeypatch.setattr(orchestrator.sys, "version_info", (4, 0, 1, "final", 0))
        main(["doctor"])
        assert capsys.readouterr().out.startswith("\u2713 Python 4.0.1\n")

    def test_only_a_space_or_hyphen_separates_a_name_from_its_version(self) -> None:
        assert orchestrator.NAME_SEPARATORS == (" ", "-")

    def test_dependency_floor_matches_the_packaged_requires_python(self) -> None:
        required = tomllib.loads(
            Path(__file__)
            .resolve()
            .parents[1]
            .joinpath("pyproject.toml")
            .read_text(encoding="utf-8")
        )["project"]["requires-python"]
        parsed = re.fullmatch(r">=(\d+)\.(\d+)", required)
        assert parsed is not None, required
        assert tuple(int(part) for part in parsed.groups()) == orchestrator.MIN_PYTHON

    def test_only_jq_is_optional(self) -> None:
        assert orchestrator.DEPENDENCY_TOOLS == (
            ("uv", False),
            ("claude", False),
            ("codex", False),
            ("grok", False),
            ("opencode", False),
            ("jq", True),
        )

    @pytest.mark.parametrize(
        ("tool", "reported", "expected"),
        [
            ("uv", "uv 0.4.18", "uv 0.4.18"),
            ("jq", "jq-1.7", "jq 1.7"),
            ("claude", "2.1.234 (Claude Code)", "claude 2.1.234 (Claude Code)"),
            ("codex", "codex-cli 0.147.0", "codex-cli 0.147.0"),
            ("jq", "jq", "jq"),
            ("grok", "unreleased build", "grok unreleased build"),
            ("uv", "uv  0.4.18", "uv  0.4.18"),
            ("jq", "jqX1.7", "jqX1.7"),
        ],
    )
    def test_describe_version_names_each_tool_exactly_once(
        self, tool: str, reported: str, expected: str
    ) -> None:
        assert orchestrator._describe_version(tool, reported) == expected

    def test_main_dependency_check_after_command_remains_prompt_data(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(orchestrator, "STATE_FILE", tmp_path / "s.json")
        monkeypatch.setattr(orchestrator, "DEFAULT_BACKEND", "echo")
        with pytest.raises(SystemExit, match="2"):
            main(["talk", "agent", "a", "--dependency-check"])
        assert capsys.readouterr().out == ""

    def test_main_reads_sys_argv_when_none_given(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr("sys.argv", ["orchestrator", "bogus"])
        with pytest.raises(SystemExit, match="2"):
            main()
        assert "usage: orchestrator " in capsys.readouterr().err

    def test_main_unknown_command_exits(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit, match="2"):
            main(["bogus"])
        assert "usage: orchestrator " in capsys.readouterr().err

    @pytest.mark.parametrize(
        ("argv", "usage"),
        [
            (["talk", "a", "hi", "--timeout", "5"], "usage: orchestrator talk "),
            (["doctor", "ignored"], "usage: orchestrator doctor "),
        ],
    )
    def test_trailing_argument_errors_use_the_selected_verb_usage(
        self,
        argv: list[str],
        usage: str,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            orchestrator,
            "Orchestrator",
            lambda: (_ for _ in ()).throw(AssertionError("constructed")),
        )
        with pytest.raises(SystemExit, match="2"):
            main(argv)
        assert capsys.readouterr().err.startswith(usage)

    @pytest.mark.parametrize("removed", ["spawn", "ensure"])
    def test_removed_lifecycle_commands_are_not_dispatchable(
        self, removed: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit, match="2"):
            main([removed])
        captured = capsys.readouterr()
        assert "unrecognized" not in captured.err
        assert "invalid choice" in captured.err

    def test_main_dispatches_using_sys_argv_when_none_given(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr("sys.argv", ["orchestrator", "create", "a", "-b", "echo"])
        monkeypatch.setattr(
            orchestrator,
            "Orchestrator",
            lambda: Orchestrator(state_file=tmp_path / "s.json"),
        )
        main()
        assert "created agent 'a' backend=echo" in capsys.readouterr().out

    def test_main_dispatches_to_command(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(orchestrator, "STATE_FILE", tmp_path / "s.json")
        monkeypatch.setattr(
            orchestrator,
            "Orchestrator",
            lambda: Orchestrator(state_file=tmp_path / "s.json"),
        )
        main(["create", "a", "-b", "echo"])
        assert "created agent 'a' backend=echo" in capsys.readouterr().out

    def test_main_talk_creates_a_missing_agent(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """`talk <new-name>` creates it and runs the turn, not an error."""
        monkeypatch.setattr(orchestrator, "STATE_FILE", tmp_path / "s.json")
        monkeypatch.setattr(orchestrator, "DEFAULT_BACKEND", "echo")
        main(["talk", "nope", "-p", "hi"])
        captured = capsys.readouterr()
        assert captured.err == "created agent 'nope' backend=echo\n"
        assert "echo:hi" in captured.out

    def test_main_duplicate_create_is_one_line(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(orchestrator, "STATE_FILE", tmp_path / "s.json")
        main(["create", "a", "-b", "echo"])
        capsys.readouterr()
        with pytest.raises(SystemExit, match="1"):
            main(["create", "a", "-b", "echo"])
        captured = capsys.readouterr()
        assert captured.err == "agent 'a' already exists\n"
        assert "Traceback" not in captured.err

    def test_main_unknown_delete_is_one_line(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(orchestrator, "STATE_FILE", tmp_path / "s.json")
        with pytest.raises(SystemExit, match="1"):
            main(["delete", "nope"])
        assert capsys.readouterr().err == "no agent named 'nope'\n"

    @pytest.mark.parametrize("flag", ["-h", "--help"])
    def test_main_help_describes_the_whole_cli(
        self, flag: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Routing every dash token to the skill parser hid the commands."""
        with pytest.raises(SystemExit) as excinfo:
            main([flag])
        assert excinfo.value.code == 0
        captured = capsys.readouterr()
        assert "usage: orchestrator " in captured.out
        with pytest.raises(SystemExit) as talk_help:
            main(["talk", "-h"])
        assert talk_help.value.code == 0
        assert "--skill" in capsys.readouterr().out
        assert captured.err == ""

    def test_main_lets_an_internal_error_surface(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A bug inside a backend must not print as a one-word usage error."""

        class BuggyBackend(AgentBackend):
            @property
            def name(self) -> str:
                return "buggy"

            def run_turn(
                self,
                prompt: str,
                session_id: str | None,
                cwd: Path,
                timeout: int = DEFAULT_TURN_TIMEOUT,
                schema: OutputSchema | None = None,
            ) -> TurnResult:
                raise KeyError("some internal dict key")

        register_backend("buggy", BuggyBackend)
        monkeypatch.setattr(orchestrator, "STATE_FILE", tmp_path / "s.json")
        main(["create", "b", "-b", "buggy"])
        capsys.readouterr()
        with pytest.raises(KeyError, match="some internal dict key"):
            main(["talk", "b", "-p", "hi"])

    def test_main_corrupt_state_is_one_line(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        state = tmp_path / "s.json"
        state.write_text("{", encoding="utf-8")
        monkeypatch.setattr(orchestrator, "STATE_FILE", state)
        with pytest.raises(SystemExit, match="1"):
            main(["list"])
        err = capsys.readouterr().err
        assert err.endswith("\n")
        assert "Traceback" not in err
        assert err.count("\n") == 1


class TestDescribeCommand:
    def test_prompt_is_replaced_by_its_size(self) -> None:
        assert describe_command(["claude", "-p", "hello"], "hello") == (
            "claude -p <prompt:5chars>"
        )

    def test_prompt_in_the_middle_is_found(self) -> None:
        assert describe_command(["codex", "exec", "hi", "--json"], "hi") == (
            "codex exec <prompt:2chars> --json"
        )

    def test_output_does_not_grow_with_the_prompt(self) -> None:
        """The whole point: a 10k prompt must not produce a 10k log line."""
        short = describe_command(["claude", "-p", "x"], "x")
        long = describe_command(["claude", "-p", "x" * 10_000], "x" * 10_000)
        assert len(long) - len(short) == len("10000") - len("1")

    def test_flags_are_left_intact(self) -> None:
        args = ["claude", "--resume", "s1", "-p", "hi"]
        assert describe_command(args, "hi") == "claude --resume s1 -p <prompt:2chars>"

    def test_only_the_prompt_slot_is_replaced_when_it_matches_a_flag(self) -> None:
        args = ["codex", "exec", "resume", "t1", "resume", "--json"]
        assert describe_command(args, "resume") == (
            "codex exec resume t1 <prompt:6chars> --json"
        )

    def test_prompt_attached_to_its_flag_is_replaced(self) -> None:
        """Grok takes `--single=<prompt>`; the flag must stay readable."""
        args = ["grok", "--always-approve", "--single=hello"]
        assert describe_command(args, "hello") == (
            "grok --always-approve --single=<prompt:5chars>"
        )

    def test_attached_prompt_does_not_grow_the_line(self) -> None:
        short = describe_command(["grok", "--single=x"], "x")
        long = describe_command(["grok", f"--single={'x' * 10_000}"], "x" * 10_000)
        assert len(long) - len(short) == len("10000") - len("1")

    def test_attached_form_keeps_a_hyphen_leading_prompt_out_of_the_log(
        self,
    ) -> None:
        args = ["grok", "--single=--fix the parser"]
        assert describe_command(args, "--fix the parser") == (
            "grok --single=<prompt:16chars>"
        )

    def test_a_standalone_prompt_wins_over_an_earlier_attached_one(self) -> None:
        args = ["grok", "--single=hi", "hi"]
        assert describe_command(args, "hi") == "grok --single=hi <prompt:2chars>"

    def test_an_empty_prompt_matches_no_attached_flag(self) -> None:
        """Every argument ends with "=" + "", so the attached form must not
        fire and swallow the flag it is glued to."""
        assert describe_command(["grok", "--single="], "") == "grok --single="

    def test_unmatched_prompt_leaves_args_unchanged(self) -> None:
        assert describe_command(["claude", "-p", "hello"], "other") == "claude -p hello"

    def test_empty_args_render_as_empty(self) -> None:
        assert describe_command([], "hello") == ""

    def test_prompt_in_the_first_two_slots_is_still_replaced(self) -> None:
        assert describe_command(["hello"], "hello") == "<prompt:5chars>"
        assert describe_command(["codex", "hi"], "hi") == "codex <prompt:2chars>"


class TestVerboseFlag:
    @pytest.fixture(autouse=True)
    def isolated_orchestrator(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            orchestrator,
            "Orchestrator",
            lambda: Orchestrator(state_file=tmp_path / "s.json"),
        )
        for name in ("orchestrator", "backends", "third_party"):
            logging.getLogger(name).setLevel(logging.NOTSET)

    @pytest.mark.parametrize(
        ("flag", "expected"),
        [
            ("-v", logging.DEBUG),
            ("--verbose", logging.DEBUG),
            ("-vv", orchestrator.TRACE),
            ("-vvv", orchestrator.TRACE),
        ],
    )
    def test_each_flag_selects_its_level(self, flag: str, expected: int) -> None:
        main([flag, "list"])
        assert logging.getLogger("orchestrator").level == expected
        assert logging.getLogger("backends").level == expected

    @pytest.mark.parametrize("flag", ["-v", "--verbose", "-vv", "-vvv"])
    def test_no_flag_leaks_third_party_debug_output(self, flag: str) -> None:
        """A dependency's debug output would bury the signal being asked for."""
        main([flag, "list"])
        assert logging.getLogger("third_party").getEffectiveLevel() > logging.DEBUG

    @pytest.mark.parametrize("flag", ["-v", "--verbose", "-vv", "-vvv"])
    def test_flag_is_not_treated_as_a_command(self, flag: str, capsys) -> None:
        main([flag, "list"])
        out = capsys.readouterr().out
        assert out.startswith("registry: ")
        assert out.endswith("no agents\n")

    def test_the_loudest_flag_given_wins(self) -> None:
        main(["-v", "-vv", "list"])
        assert logging.getLogger("orchestrator").level == orchestrator.TRACE

    def test_verbosity_counts_before_and_after_the_verb(self) -> None:
        main(["-v", "list", "-v"])
        assert logging.getLogger("orchestrator").level == orchestrator.TRACE
        assert logging.getLogger("backends").level == orchestrator.TRACE

    def test_without_a_flag_our_loggers_stay_quiet(self) -> None:
        main(["list"])
        assert logging.getLogger("orchestrator").getEffectiveLevel() > logging.DEBUG

    def test_no_arguments_at_all_stays_quiet(self, capsys) -> None:
        """The empty argv path still picks a verbosity before printing usage."""
        with pytest.raises(SystemExit, match="2"):
            main([])
        assert logging.getLogger("orchestrator").getEffectiveLevel() > logging.DEBUG

    def test_trace_sits_below_debug(self) -> None:
        """Above DEBUG it would fire at -v, dumping transcripts unasked."""
        assert orchestrator.TRACE < logging.DEBUG
        assert logging.getLevelName(orchestrator.TRACE) == "TRACE"

    def test_prompt_token_equal_to_a_verbose_flag_is_kept(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        orch = Orchestrator(state_file=tmp_path / "s.json")
        orch.spawn("a", "echo")
        monkeypatch.setattr(orchestrator, "Orchestrator", lambda: orch)
        main(["talk", "a", "--", "add", "-v", "please"])
        assert "echo:add -v please" in capsys.readouterr().out
        assert logging.getLogger("orchestrator").getEffectiveLevel() > logging.DEBUG

    def test_leading_flag_does_not_strip_the_same_token_from_the_prompt(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        orch = Orchestrator(state_file=tmp_path / "s.json")
        orch.spawn("a", "echo")
        monkeypatch.setattr(orchestrator, "Orchestrator", lambda: orch)
        main(["-v", "talk", "a", "--", "add", "-v", "please"])
        assert "echo:add -v please" in capsys.readouterr().out
        assert logging.getLogger("orchestrator").level == logging.DEBUG


class TestStepLogging:
    def test_state_load_and_write_are_logged(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        state_file = tmp_path / "s.json"
        with caplog.at_level("DEBUG", logger="orchestrator"):
            orch = Orchestrator(state_file=state_file)
            orch.spawn("a", "echo")
        messages = _messages(caplog)
        assert f"state: loaded 0 agent(s) from {state_file}" in messages
        assert f"state: wrote 1 agent(s) to {state_file}" in messages

    def test_turn_start_and_duration_are_logged(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        orch = Orchestrator(state_file=tmp_path / "s.json")
        orch.spawn("a", "echo")
        with caplog.at_level("DEBUG", logger="orchestrator"):
            orch.talk("a", "hi")
        messages = _messages(caplog)
        assert "agent 'a' (echo): starting turn, resume=False" in messages
        durations = [
            _reported_seconds(m, r"agent 'a': turn finished in (\d+\.\d)s")
            for m in messages
            if m.startswith("agent 'a': turn finished")
        ]
        assert len(durations) == 1
        assert durations[0] < 60

    def test_prompt_and_reply_are_logged_in_full_at_trace(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        orch = Orchestrator(state_file=tmp_path / "s.json")
        orch.spawn("a", "echo")
        prompt = "line one\nline two"
        with caplog.at_level(orchestrator.TRACE, logger="orchestrator"):
            orch.talk("a", prompt)
        messages = _messages(caplog)
        # Multi-line prompts must survive intact; this is the level you turn on
        # precisely to read exactly what went in and came back.
        assert f"agent 'a' prompt in:\n{prompt}" in messages
        assert f"agent 'a' reply out:\necho:{prompt}" in messages

    def test_transcripts_stay_out_of_the_step_level_view(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """-v is the level meant to be readable and safe to paste."""
        orch = Orchestrator(state_file=tmp_path / "s.json")
        orch.spawn("a", "echo")
        with caplog.at_level(logging.DEBUG, logger="orchestrator"):
            orch.talk("a", "secret prompt")
        assert not any("secret prompt" in message for message in _messages(caplog))

    def test_a_resumed_turn_is_logged_as_such(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """resume= must reflect the session, not be hardcoded by either turn."""
        orch = Orchestrator(state_file=tmp_path / "s.json")
        orch.spawn("a", "echo")
        orch.talk("a", "first")
        with caplog.at_level("DEBUG", logger="orchestrator"):
            orch.talk("a", "second")
        assert "agent 'a' (echo): starting turn, resume=True" in _messages(caplog)

    def test_cli_argument_shape_and_dispatch_are_logged(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(
            orchestrator,
            "Orchestrator",
            lambda: Orchestrator(state_file=tmp_path / "s.json"),
        )
        with caplog.at_level("DEBUG", logger="orchestrator"):
            main(["-v", "create", "a", "-b", "echo"])
        messages = _messages(caplog)
        # Four arguments remain once -v is stripped.
        assert "cli: 5 argument(s) after flag splitting" in messages
        assert "cli: dispatching 'create'" in messages


class TestLoggingConfiguration:
    """basicConfig() only acts on a root logger that has no handlers yet.

    pytest installs its own capture handler around every test phase, so the
    handlers have to be cleared inside the test body — clearing them in a
    fixture is undone before the body runs, and every assertion here would
    then be measuring pytest's logging setup rather than this project's.
    """

    @pytest.fixture(autouse=True)
    def restore_root(self) -> Iterator[None]:
        root = logging.getLogger()
        handlers, level = root.handlers[:], root.level
        yield
        root.handlers[:] = handlers
        root.setLevel(level)

    def test_root_is_pinned_to_warning(self) -> None:
        root = logging.getLogger()
        root.handlers.clear()
        # Start from the level the call must overwrite: WARNING is also the
        # default, so starting there would pass without the level ever applying.
        root.setLevel(logging.DEBUG)

        orchestrator._configure_logging(False)

        assert root.level == logging.WARNING

    def test_records_carry_time_level_and_logger_name(self) -> None:
        root = logging.getLogger()
        root.handlers.clear()

        orchestrator._configure_logging(False)

        record = logging.LogRecord(
            "backends.claude", logging.WARNING, "p", 1, "msg", None, None
        )
        assert re.fullmatch(
            r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} "
            r"WARNING backends\.claude: msg",
            root.handlers[0].format(record),
        )


def test_parser_exposes_the_complete_new_surface() -> None:
    parser = orchestrator._build_parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )

    assert set(subparsers.choices) == {"create", "talk", "list", "delete", "doctor"}
    child_args = {
        "create": ["a"],
        "talk": ["a", "-p", "x"],
        "list": [],
        "delete": ["a"],
        "doctor": [],
    }
    for verb in subparsers.choices:
        child = subparsers.choices[verb]
        assert child.prog == f"orchestrator {verb}"
        assert child.get_default("_parser") is child
        assert child.parse_args(child_args[verb]).verbosity_after == 0

    talk = subparsers.choices["talk"]
    assert (
        "prompt source: orchestrator talk NAME "
        "[-p TEXT | --prompt-file PATH | -- PROMPT...]" in talk.epilog
    )
    assert (
        talk.parse_args(
            [
                "a",
                "-s",
                "review",
                "--schema",
                "out.json",
                "--retries",
                "3",
                "--timeout",
                "7",
                "-p",
                "x",
            ]
        ).skill
        == "review"
    )

    for action in (parser._actions[1],):
        assert action.option_strings == ["--version"]
        assert action.nargs == 0
        assert action.default is argparse.SUPPRESS
        assert action.help == "show the installed version"

    verbosity = next(action for action in parser._actions if action.dest == "verbosity")
    assert verbosity.option_strings == ["-v", "--verbose"]
    assert verbosity.default == 0
    assert (
        verbosity.help == "log each step and how long it took; repeat for full prompts"
    )


def test_agent_config_short_options_are_forwarded_by_create_and_talk() -> None:
    parser = orchestrator._build_parser()
    create = parser.parse_args(["create", "a", "-m", "model", "-e", "high"])
    talk = parser.parse_args(["talk", "a", "-m", "model", "-e", "high", "-p", "x"])
    assert (create.model, create.reasoning_effort) == ("model", "high")
    assert (talk.model, talk.reasoning_effort) == ("model", "high")


def test_version_fallback_uses_the_distribution_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(orchestrator, "_project_version", lambda: None)
    seen: list[str] = []

    def version(package: str) -> str:
        seen.append(package)
        return "9.8.7"

    monkeypatch.setattr(orchestrator.importlib.metadata, "version", version)
    assert orchestrator._resolve_version() == "9.8.7"
    assert seen == ["agents-army"]


def test_version_fallback_rejects_an_empty_metadata_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(orchestrator, "_project_version", lambda: None)
    monkeypatch.setattr(orchestrator.importlib.metadata, "version", lambda _: "")
    with pytest.raises(ValueError, match=r"^$"):
        orchestrator._resolve_version()


def test_project_version_rejects_a_non_string_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        orchestrator.tomllib,
        "load",
        lambda _: {"project": {"version": 123}},
    )
    monkeypatch.setattr(Path, "open", lambda *_: io.BytesIO(b"project = {}"))
    assert orchestrator._project_version() is None
    monkeypatch.setattr(orchestrator.tomllib, "load", lambda _: {})
    assert orchestrator._project_version() is None


def test_prompt_errors_have_exact_messages_and_strip_tail(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = orchestrator._build_parser()
    options = parser.parse_args(["talk", "a"])
    with pytest.raises(SystemExit) as missing:
        orchestrator._resolve_talk_prompt(options, [], False)
    assert missing.value.code == 2
    assert (
        "orchestrator talk: error: talk requires exactly one prompt source"
        in capsys.readouterr().err
    )

    options = parser.parse_args(["talk", "a", "-p", " "])
    with pytest.raises(SystemExit) as empty:
        orchestrator._resolve_talk_prompt(options, [], False)
    assert empty.value.code == 2
    assert (
        "orchestrator talk: error: talk prompt must not be empty"
        in capsys.readouterr().err
    )

    options = parser.parse_args(["talk", "a"])
    orchestrator._resolve_talk_prompt(options, ["  one", "two  "], True)
    assert options.prompt == "one two"

    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("  one\n two  \n", encoding="utf-8")
    options = parser.parse_args(["talk", "a", "--prompt-file", str(prompt_file)])
    orchestrator._resolve_talk_prompt(options, [], False)
    assert options.prompt == "one\n two"


@pytest.mark.parametrize("kind", ["missing", "directory", "binary", "empty"])
def test_prompt_file_errors_are_exact_and_prevent_side_effects(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prompt_file = tmp_path / "prompt.txt"
    if kind == "directory":
        prompt_file.mkdir()
    elif kind == "binary":
        prompt_file.write_bytes(b"\xff\xfe")
    elif kind == "empty":
        prompt_file.write_text(" \t\n", encoding="utf-8")

    monkeypatch.setattr(
        orchestrator,
        "Orchestrator",
        lambda: (_ for _ in ()).throw(AssertionError("constructed")),
    )
    with pytest.raises(SystemExit) as excinfo:
        main(["talk", "a", "--prompt-file", str(prompt_file)])
    assert excinfo.value.code == 2
    error = capsys.readouterr().err
    if kind == "empty":
        assert "orchestrator talk: error: talk prompt must not be empty" in error
    else:
        try:
            prompt_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            expected = f"cannot read prompt file {prompt_file.resolve()}: {exc}"
        else:
            raise AssertionError("expected prompt file read to fail")
        assert expected in error


def test_prompt_file_errors_have_exact_messages(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    parser = orchestrator._build_parser()

    missing_path = tmp_path / "missing.txt"
    options = parser.parse_args(["talk", "a", "--prompt-file", str(missing_path)])
    with pytest.raises(SystemExit) as missing:
        orchestrator._resolve_talk_prompt(options, [], False)
    assert missing.value.code == 2
    resolved_missing = missing_path.resolve()
    assert (
        f"orchestrator talk: error: cannot read prompt file {resolved_missing}: "
        in capsys.readouterr().err
    )

    directory_path = tmp_path / "adir"
    directory_path.mkdir()
    options = parser.parse_args(["talk", "a", "--prompt-file", str(directory_path)])
    with pytest.raises(SystemExit) as is_dir:
        orchestrator._resolve_talk_prompt(options, [], False)
    assert is_dir.value.code == 2
    resolved_dir = directory_path.resolve()
    assert (
        f"orchestrator talk: error: cannot read prompt file {resolved_dir}: "
        in capsys.readouterr().err
    )

    binary_path = tmp_path / "binary.txt"
    binary_path.write_bytes(b"\xff\xfe\x00not utf-8")
    options = parser.parse_args(["talk", "a", "--prompt-file", str(binary_path)])
    with pytest.raises(SystemExit) as bad_decode:
        orchestrator._resolve_talk_prompt(options, [], False)
    assert bad_decode.value.code == 2
    resolved_binary = binary_path.resolve()
    assert (
        f"orchestrator talk: error: cannot read prompt file {resolved_binary}: "
        in capsys.readouterr().err
    )

    blank_path = tmp_path / "blank.txt"
    blank_path.write_text(" \n\t \n", encoding="utf-8")
    options = parser.parse_args(["talk", "a", "--prompt-file", str(blank_path)])
    with pytest.raises(SystemExit) as blank:
        orchestrator._resolve_talk_prompt(options, [], False)
    assert blank.value.code == 2
    assert (
        "orchestrator talk: error: talk prompt must not be empty"
        in capsys.readouterr().err
    )


def test_prompt_file_strips_outer_whitespace_and_keeps_interior_newlines(
    tmp_path: Path,
) -> None:
    parser = orchestrator._build_parser()
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("  line one\nline two  \n", encoding="utf-8")
    options = parser.parse_args(["talk", "a", "--prompt-file", str(prompt_path)])
    orchestrator._resolve_talk_prompt(options, [], False)
    assert options.prompt == "line one\nline two"


def test_separator_error_and_cli_error_without_arguments_are_exact(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as separator:
        main(["create", "a", "--", "extra"])
    assert separator.value.code == 2
    assert (
        "orchestrator create: error: the -- separator is only valid for talk"
        in capsys.readouterr().err
    )

    monkeypatch.setattr(
        orchestrator,
        "Orchestrator",
        lambda: (_ for _ in ()).throw(orchestrator.OrchestratorError()),
    )
    with pytest.raises(SystemExit) as empty_error:
        main(["list"])
    assert empty_error.value.code == 1
    assert capsys.readouterr().err == "\n"


def test_cli_log_counts_head_and_tail_arguments(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setitem(orchestrator.VERBS, "talk", lambda _orch, _opts: None)
    monkeypatch.setattr(orchestrator, "Orchestrator", lambda: object())
    caplog.set_level(logging.DEBUG, logger="orchestrator")
    main(["-v", "talk", "a", "--", "one", "two"])
    assert "cli: 5 argument(s) after flag splitting" in caplog.text
