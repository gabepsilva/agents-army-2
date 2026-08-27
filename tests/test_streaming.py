"""Behavioral tests for the opt-in CLI streaming transport."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import backends.base as base_backend
import backends.claude as claude_backend
import backends.codex as codex_backend
import backends.grok as grok_backend
import backends.opencode as opencode_backend
import orchestrator
from backends.base import (
    DEFAULT_TURN_TIMEOUT,
    AgentBackend,
    OutputSchema,
    TurnResult,
    run_cli_turn,
    structured_reply,
)
from backends.registry import register_backend
from orchestrator import Orchestrator
from tests.path_helpers import runtime_paths


class _StderrProbe:
    """Capture writes and expose the first flush to the test thread."""

    def __init__(self) -> None:
        self.text = ""
        self.writes: list[str] = []
        self.flushes = 0
        self.flushed = threading.Event()

    def write(self, text: str) -> int:
        self.text += text
        self.writes.append(text)
        return len(text)

    def flush(self) -> None:
        self.flushes += 1
        self.flushed.set()


def _child(script: str) -> list[str]:
    return [sys.executable, "-c", script]


def test_streaming_echoes_a_complete_line_before_the_child_exits(
    tmp_path: Path, monkeypatch
) -> None:
    probe = _StderrProbe()
    monkeypatch.setattr(sys, "stderr", probe)
    result: list[subprocess.CompletedProcess[str]] = []
    failure: list[BaseException] = []

    def run() -> None:
        try:
            result.append(
                run_cli_turn(
                    "demo",
                    _child(
                        "import sys, time; "
                        "sys.stdout.write('event\\nsecond\\n'); sys.stdout.flush(); "
                        "time.sleep(0.4)"
                    ),
                    prompt="",
                    session_id=None,
                    cwd=tmp_path,
                    timeout=2,
                    stream=True,
                )
            )
        except BaseException as exc:  # pragma: no cover - surfaced below
            failure.append(exc)

    worker = threading.Thread(target=run)
    worker.start()
    assert probe.flushed.wait(timeout=1)
    assert worker.is_alive()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert failure == []
    assert result[0].stdout == "event\nsecond\n"
    assert result[0].stderr == ""
    assert probe.text == "event\nsecond\n"
    assert probe.writes == ["event\n", "second\n"]
    assert probe.flushes == 2


def test_streaming_drains_a_large_prompt_and_child_stderr(
    tmp_path: Path, capsys
) -> None:
    prompt = "p" * (256 * 1024)
    child_stderr = "d" * (256 * 1024)
    result = run_cli_turn(
        "demo",
        _child(
            "import sys; "
            "received = sys.stdin.buffer.read(); "
            f"sys.stderr.buffer.write(b'd' * {len(child_stderr)}); "
            "sys.stderr.flush(); "
            "sys.stdout.write(str(len(received)) + '\\n'); sys.stdout.flush()"
        ),
        prompt=prompt,
        session_id=None,
        cwd=tmp_path,
        timeout=2,
        prompt_on_stdin=True,
        stream=True,
    )

    assert result.stdout == f"{len(prompt.encode())}\n"
    assert result.stderr == child_stderr
    assert capsys.readouterr().err == f"{len(prompt.encode())}\n"


def test_streaming_raw_matches_text_mode_across_chunks_and_newlines(
    tmp_path: Path, capsys
) -> None:
    command = _child(
        r"""
import os
import time
os.write(1, b"alpha \xc3")
time.sleep(0.1)
os.write(1, b"\xa9\r\nbeta\rfinal")
"""
    )
    normal = run_cli_turn(
        "demo",
        command,
        prompt="",
        session_id=None,
        cwd=tmp_path,
        timeout=2,
    )
    streamed = run_cli_turn(
        "demo",
        command,
        prompt="",
        session_id=None,
        cwd=tmp_path,
        timeout=2,
        stream=True,
    )

    assert normal.stdout == "alpha é\nbeta\nfinal"
    assert streamed.stdout == normal.stdout
    assert streamed.stderr == normal.stderr == ""
    assert capsys.readouterr().err == "alpha é\nbeta\n"


def test_streaming_timeout_kills_and_reaps_a_child_after_output(
    tmp_path: Path, capsys
) -> None:
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired) as excinfo:
        run_cli_turn(
            "demo",
            _child(
                "import os, sys, time; "
                "print(os.getpid(), flush=True); "
                "print('diagnostic', file=sys.stderr, flush=True); "
                "time.sleep(10)"
            ),
            prompt="",
            session_id=None,
            cwd=tmp_path,
            timeout=1,
            stream=True,
        )
    assert time.monotonic() - started < 3

    assert excinfo.value.cmd[0] == sys.executable
    assert excinfo.value.timeout == 1
    assert isinstance(excinfo.value.output, bytes)
    pid = int(excinfo.value.output.decode().strip())
    assert excinfo.value.stderr == b"diagnostic\n"
    for _ in range(20):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.01)
    else:
        raise AssertionError(f"streaming child {pid} was not reaped")
    assert capsys.readouterr().err == f"{pid}\n"


def test_streaming_closes_an_empty_prompt_pipe(tmp_path: Path, capsys) -> None:
    result = run_cli_turn(
        "demo",
        _child("import sys; print(sys.stdin.read(), end='ok\\n', flush=True)"),
        prompt="",
        session_id=None,
        cwd=tmp_path,
        timeout=2,
        prompt_on_stdin=True,
        stream=True,
    )

    assert result.stdout == "ok\n"
    assert capsys.readouterr().err == "ok\n"


def test_streaming_retries_if_select_is_interrupted(
    tmp_path: Path, monkeypatch
) -> None:
    real_select = base_backend.select.select
    calls = 0

    def interrupted_once(readable, writable, exceptional, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise InterruptedError
        return real_select(readable, writable, exceptional, timeout)

    monkeypatch.setattr(base_backend.select, "select", interrupted_once)
    result = run_cli_turn(
        "demo",
        _child("print('ok', flush=True)"),
        prompt="",
        session_id=None,
        cwd=tmp_path,
        timeout=2,
        stream=True,
    )

    assert result.stdout == "ok\n"
    assert calls >= 2


def test_streaming_retries_if_read_is_interrupted(tmp_path: Path, monkeypatch) -> None:
    real_read = base_backend.os.read
    real_popen = base_backend.subprocess.Popen
    calls = 0

    def interrupted_once(file_descriptor, size):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise InterruptedError
        return real_read(file_descriptor, size)

    def start_process(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        monkeypatch.setattr(base_backend.os, "read", interrupted_once)
        return process

    monkeypatch.setattr(base_backend.subprocess, "Popen", start_process)
    result = run_cli_turn(
        "demo",
        _child("print('ok', flush=True)"),
        prompt="",
        session_id=None,
        cwd=tmp_path,
        timeout=2,
        stream=True,
    )

    assert result.stdout == "ok\n"
    assert calls >= 2


def test_streaming_rejects_an_incomplete_utf8_sequence(tmp_path: Path) -> None:
    with pytest.raises(UnicodeDecodeError):
        run_cli_turn(
            "demo",
            _child("import os; os.write(1, b'\\xc3')"),
            prompt="",
            session_id=None,
            cwd=tmp_path,
            timeout=2,
            stream=True,
        )


@pytest.mark.parametrize(
    ("script", "expected_stdout", "expected_stderr", "expected_echo"),
    [
        (
            "import os, time; os.close(2); time.sleep(0.1); os.write(1, b'out\\n')",
            "out\n",
            "",
            "out\n",
        ),
        (
            "import os, time; os.close(1); time.sleep(0.1); os.write(2, b'err\\n')",
            "",
            "err\n",
            "",
        ),
    ],
    ids=["stdout-after-stderr-eof", "stderr-after-stdout-eof"],
)
def test_streaming_keeps_draining_after_the_other_output_pipe_closes(
    script: str,
    expected_stdout: str,
    expected_stderr: str,
    expected_echo: str,
    tmp_path: Path,
    capsys,
) -> None:
    result = run_cli_turn(
        "demo",
        _child(script),
        prompt="",
        session_id=None,
        cwd=tmp_path,
        timeout=2,
        stream=True,
    )

    assert result.stdout == expected_stdout
    assert result.stderr == expected_stderr
    assert capsys.readouterr().err == expected_echo


def test_streaming_uses_the_child_working_directory(tmp_path: Path) -> None:
    result = run_cli_turn(
        "demo",
        _child("import os; print(os.getcwd(), flush=True)"),
        prompt="",
        session_id=None,
        cwd=tmp_path,
        timeout=2,
        stream=True,
    )

    assert result.stdout == f"{tmp_path}\n"


def test_streaming_preserves_process_metadata_and_exit_logging(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    command = _child("print('ok', flush=True)")
    with caplog.at_level("DEBUG", logger="backends.base"):
        result = run_cli_turn(
            "demo",
            command,
            prompt="",
            session_id=None,
            cwd=tmp_path,
            timeout=2,
            stream=True,
        )

    assert result.args == command
    assert result.returncode == 0
    exit_messages = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("demo turn: exited ")
    ]
    assert len(exit_messages) == 1
    assert exit_messages[0].endswith("s with 3 chars of stdout")
    elapsed = float(exit_messages[0].split(" after ", 1)[1].split("s ", 1)[0])
    assert 0 <= elapsed < 2


def test_streaming_enforces_the_deadline_after_output_pipes_close(
    tmp_path: Path,
) -> None:
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired) as excinfo:
        run_cli_turn(
            "demo",
            _child("import os, time; os.close(1); os.close(2); time.sleep(10)"),
            prompt="",
            session_id=None,
            cwd=tmp_path,
            timeout=1,
            stream=True,
        )

    assert time.monotonic() - started < 3
    assert excinfo.value.output is None
    assert excinfo.value.stderr is None


def test_streaming_waits_for_a_child_that_closes_pipes_before_the_deadline(
    tmp_path: Path,
) -> None:
    result = run_cli_turn(
        "demo",
        _child(
            "import os, time; "
            "time.sleep(1.2); os.close(1); os.close(2); time.sleep(0.1)"
        ),
        prompt="",
        session_id=None,
        cwd=tmp_path,
        timeout=2,
        stream=True,
    )

    assert result.returncode == 0
    assert result.stdout == result.stderr == ""


def test_talk_without_stream_passes_the_default_false_to_the_backend(
    tmp_path: Path,
) -> None:
    calls: list[bool] = []

    class RecordingBackend(AgentBackend):
        name = "streaming-default"

        def run_turn(
            self,
            prompt: str,
            session_id: str | None,
            cwd: Path,
            timeout: int = DEFAULT_TURN_TIMEOUT,
            schema: OutputSchema | None = None,
            *,
            resume_as_fork: bool = False,
            stream: bool = False,
        ) -> TurnResult:
            calls.append(stream)
            return TurnResult(session_id="sid", reply=prompt, raw=prompt)

    register_backend("streaming-default", RecordingBackend)
    orchestrator = Orchestrator(
        runtime_paths(tmp_path, state_file=tmp_path / "state.json")
    )
    orchestrator.spawn("agent", "streaming-default")

    assert orchestrator.talk("agent", "answer").reply == "answer"
    assert calls == [False]


def test_agent_talk_defaults_streaming_off(tmp_path: Path) -> None:
    calls: list[bool] = []

    class RecordingBackend(AgentBackend):
        name = "agent-streaming-default"

        def run_turn(
            self,
            prompt: str,
            session_id: str | None,
            cwd: Path,
            timeout: int = DEFAULT_TURN_TIMEOUT,
            schema: OutputSchema | None = None,
            *,
            resume_as_fork: bool = False,
            stream: bool = False,
        ) -> TurnResult:
            calls.append(stream)
            return TurnResult(session_id="sid", reply=prompt, raw=prompt)

    agent = orchestrator.Agent("agent", RecordingBackend(), workdir=tmp_path)

    assert agent.talk("answer").reply == "answer"
    assert calls == [False]


def test_talk_stream_keeps_structured_result_on_stdout(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    backend_name = "streaming-contract"

    class CliBackend(AgentBackend):
        @property
        def name(self) -> str:
            return backend_name

        def run_turn(
            self,
            prompt: str,
            session_id: str | None,
            cwd: Path,
            timeout: int = DEFAULT_TURN_TIMEOUT,
            schema: OutputSchema | None = None,
            *,
            resume_as_fork: bool = False,
            stream: bool = False,
        ) -> TurnResult:
            proc = run_cli_turn(
                "demo",
                _child(
                    "import sys; "
                    "sys.stdout.write('event\\n{\"ok\":true}\\n'); "
                    "sys.stdout.flush()"
                ),
                prompt=prompt,
                session_id=session_id,
                cwd=cwd,
                timeout=timeout,
                stream=stream,
            )
            reply = proc.stdout.splitlines()[-1]
            return TurnResult(
                session_id="sid",
                reply=reply,
                raw=proc.stdout,
                structured=structured_reply(schema, reply),
            )

    register_backend(backend_name, CliBackend)
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(
        '{"type":"object","additionalProperties":false,'
        '"properties":{"ok":{"type":"boolean"}},"required":["ok"]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(orchestrator, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(orchestrator, "WORKDIR", tmp_path)

    orchestrator.main(["create", "agent", "-b", backend_name])
    capsys.readouterr()
    orchestrator.main(
        [
            "talk",
            "agent",
            "--stream",
            "--schema",
            str(schema_path),
            "-p",
            "answer",
        ]
    )
    captured = capsys.readouterr()

    assert captured.out == '[agent session=sid]\n{\n  "ok": true\n}\n'
    assert captured.err == 'event\n{"ok":true}\n'
    assert "event" not in captured.out


def test_streaming_is_inherited_by_schema_repair_attempts(tmp_path: Path) -> None:
    calls: list[bool] = []
    replies = iter(("not json", '{"ok":true}'))

    class ScriptedBackend(AgentBackend):
        name = "streaming-scripted"

        def run_turn(
            self,
            prompt: str,
            session_id: str | None,
            cwd: Path,
            timeout: int = DEFAULT_TURN_TIMEOUT,
            schema: OutputSchema | None = None,
            *,
            resume_as_fork: bool = False,
            stream: bool = False,
        ) -> TurnResult:
            calls.append(stream)
            reply = next(replies)
            return TurnResult(
                session_id="sid",
                reply=reply,
                raw=reply,
                structured=structured_reply(schema, reply),
            )

    register_backend("streaming-scripted", ScriptedBackend)
    schema = OutputSchema(
        text=(
            '{"type":"object","additionalProperties":false,'
            '"properties":{"ok":{"type":"boolean"}},"required":["ok"]}'
        ),
        path=tmp_path / "schema.json",
    )
    orchestrator = Orchestrator(
        runtime_paths(tmp_path, state_file=tmp_path / "state.json")
    )
    orchestrator.spawn("agent", "streaming-scripted")

    result = orchestrator.talk(
        "agent", "answer", schema=schema, retries=1, timeout=2, stream=True
    )

    assert calls == [True, True]
    assert result.structured == {"ok": True}


@pytest.mark.parametrize(
    ("backend_module", "backend_cls", "stdout", "expected_argv"),
    [
        pytest.param(
            claude_backend,
            claude_backend.ClaudeBackend,
            '{"type":"result","session_id":"sid","result":"reply"}',
            [
                "claude",
                "--print",
                "--output-format",
                "json",
                "--permission-mode",
                "bypassPermissions",
                "-p",
                "hello",
            ],
            id="claude",
        ),
        pytest.param(
            codex_backend,
            codex_backend.CodexBackend,
            (
                '{"type":"thread.started","thread_id":"sid"}\n'
                '{"type":"item.completed","item":{"type":"agent_message",'
                '"text":"reply"}}'
            ),
            [
                "codex",
                "exec",
                "--yolo",
                "hello",
                "--json",
                "--skip-git-repo-check",
            ],
            id="codex",
        ),
        pytest.param(
            grok_backend,
            grok_backend.GrokBackend,
            '{"sessionId":"sid","text":"reply"}',
            [
                "grok",
                "--output-format",
                "json",
                "--always-approve",
                "--single=hello",
            ],
            id="grok",
        ),
        pytest.param(
            opencode_backend,
            opencode_backend.OpenCodeBackend,
            '{"type":"text","sessionID":"sid","part":{"id":"p","text":"reply"}}',
            [
                "opencode",
                "run",
                "--format",
                "json",
                "--auto",
                "--dir",
                "PLACEHOLDER",
            ],
            id="opencode",
        ),
    ],
)
@pytest.mark.parametrize("stream", [False, True], ids=["default", "stream"])
def test_each_backend_forwards_stream_without_changing_argv(
    backend_module,
    backend_cls,
    stdout: str,
    expected_argv: list[str],
    stream: bool,
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[tuple[list[str], dict]] = []

    def fake_run(
        name: str, args: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(backend_module, "run_cli_turn", fake_run)
    result = backend_cls().run_turn("hello", None, tmp_path, stream=stream)

    expected = [
        str(tmp_path.absolute()) if arg == "PLACEHOLDER" else arg
        for arg in expected_argv
    ]
    assert calls[0][0] == expected
    assert calls[0][1]["stream"] is stream
    assert result.reply == "reply"
