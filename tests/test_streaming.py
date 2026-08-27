from __future__ import annotations

import io
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import cast

import pytest

import backends.base as base
import backends.claude as claude
import backends.codex as codex
import backends.grok as grok
import backends.opencode as opencode
import orchestrator
from backends.base import run_cli_turn
from backends.registry import get_backend, register_backend
from orchestrator import Orchestrator, main
from tests.path_helpers import runtime_paths


def _python(code: str, *arguments: str) -> list[str]:
    return [sys.executable, "-c", textwrap.dedent(code), *arguments]


def _scripted_backend(name: str, replies: list[str]) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []
    queued = list(replies)

    class ScriptedBackend(base.AgentBackend):
        @property
        def name(self) -> str:
            return name

        def run_turn(  # noqa: PLR0913 - test doubles mirror AgentBackend.run_turn public seam
            self,
            prompt: str,
            session_id: str | None,
            cwd: Path,
            timeout: int = base.DEFAULT_TURN_TIMEOUT,
            schema: base.OutputSchema | None = None,
            *,
            resume_as_fork: bool = False,
            stream: bool = False,
        ) -> base.TurnResult:
            calls.append(
                {
                    "prompt": prompt,
                    "session_id": session_id,
                    "cwd": cwd,
                    "timeout": timeout,
                    "resume_as_fork": resume_as_fork,
                    "stream": stream,
                    "schema": schema,
                }
            )
            reply = queued.pop(0)
            return base.TurnResult(
                session_id="sid",
                reply=reply,
                raw=reply,
                structured=base.structured_reply(schema, reply),
            )

    register_backend(name, ScriptedBackend)
    return calls


class _TrackingStderr(io.StringIO):
    def __init__(self, child_finished: Path) -> None:
        super().__init__()
        self.child_finished = child_finished
        self.wrote_before_child_exit = False
        self.writes: list[str] = []

    def write(self, text: str) -> int:
        if not self.child_finished.exists():
            self.wrote_before_child_exit = True
        self.writes.append(text)
        return super().write(text)


def test_streaming_runner_echoes_complete_lines_before_child_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child_finished = tmp_path / "child-finished"
    stderr = _TrackingStderr(child_finished)
    monkeypatch.setattr(sys, "stderr", stderr)

    args = _python(
        """
        import pathlib
        import sys
        import time

        finished = pathlib.Path(sys.argv[1])
        sys.stdout.write("event one\\nevent two\\n")
        sys.stdout.flush()
        time.sleep(0.2)
        finished.write_text("done")
        """,
        str(child_finished),
    )
    proc = run_cli_turn(
        "fixture",
        args,
        prompt="ignored",
        session_id=None,
        cwd=tmp_path,
        timeout=5,
        stream=True,
    )

    assert proc.args == args
    assert proc.returncode == 0
    assert proc.stdout == "event one\nevent two\n"
    assert proc.stderr == ""
    assert stderr.getvalue() == "event one\nevent two\n"
    assert stderr.writes == ["event one\n", "event two\n"]
    assert stderr.wrote_before_child_exit


def test_streaming_runner_matches_subprocess_text_decoding(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = _python(
        r"""
        import os
        import time

        for chunk in (b"\xcf", b"\x80\r", b"\nsecond\nfinal"):
            os.write(1, chunk)
            time.sleep(0.04)
        for chunk in (b"err\xe2", b"\x98", b"\x83\r", b"\npartial"):
            os.write(2, chunk)
            time.sleep(0.04)
        """
    )
    baseline = run_cli_turn(
        "fixture",
        args,
        prompt="ignored",
        session_id=None,
        cwd=tmp_path,
        timeout=5,
    )
    capsys.readouterr()

    streamed = run_cli_turn(
        "fixture",
        args,
        prompt="ignored",
        session_id=None,
        cwd=tmp_path,
        timeout=5,
        stream=True,
    )
    captured = capsys.readouterr()

    assert baseline.stdout == "π\nsecond\nfinal"
    assert baseline.stderr == "err☃\npartial"
    assert streamed.stdout == baseline.stdout
    assert streamed.stderr == baseline.stderr
    assert captured.out == ""
    assert captured.err == "π\nsecond\n"


def test_streaming_runner_drains_large_stdin_and_stderr(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prompt = "p" * (256 * 1024)
    stderr_size = 256 * 1024
    proc = run_cli_turn(
        "fixture",
        _python(
            """
            import sys

            prompt = sys.stdin.buffer.read()
            sys.stderr.buffer.write(b"e" * int(sys.argv[1]))
            sys.stderr.buffer.flush()
            sys.stdout.write(f"{len(prompt)}\\n")
            sys.stdout.flush()
            """,
            str(stderr_size),
        ),
        prompt=prompt,
        session_id=None,
        cwd=tmp_path,
        timeout=5,
        prompt_on_stdin=True,
        stream=True,
    )
    captured = capsys.readouterr()

    assert proc.returncode == 0
    assert proc.stdout == f"{len(prompt)}\n"
    assert proc.stderr == "e" * stderr_size
    assert captured.err == f"{len(prompt)}\n"


def test_streaming_runner_kills_and_reaps_a_child_at_the_deadline(
    tmp_path: Path,
) -> None:
    pid_file = tmp_path / "child-pid"
    args = _python(
        """
        import os
        import pathlib
        import sys
        import time

        pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))
        sys.stdout.write("before\\n")
        sys.stdout.flush()
        sys.stderr.write("diagnostic\\n")
        sys.stderr.flush()
        time.sleep(30)
        """,
        str(pid_file),
    )

    with pytest.raises(subprocess.TimeoutExpired) as excinfo:
        run_cli_turn(
            "fixture",
            args,
            prompt="ignored",
            session_id=None,
            cwd=tmp_path,
            timeout=1,
            stream=True,
        )

    assert excinfo.value.cmd == args
    assert excinfo.value.timeout == 1
    assert excinfo.value.output == b"before\n"
    assert excinfo.value.stderr == b"diagnostic\n"
    child_pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


def test_stream_wait_honors_an_expired_deadline_without_selecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = ["fixture", "--stream"]
    monkeypatch.setattr(base.time, "monotonic", lambda: 10.0)

    def unexpected_select(*args: object, **kwargs: object) -> None:
        raise AssertionError("an expired deadline must not call select")

    monkeypatch.setattr(base.select, "select", unexpected_select)

    with pytest.raises(subprocess.TimeoutExpired) as excinfo:
        base._wait_for_streams({}, None, args=args, deadline=10.0, timeout=9)

    assert excinfo.value.cmd == args
    assert excinfo.value.timeout == 9


def test_streaming_runner_closes_empty_stdin(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    proc = run_cli_turn(
        "fixture",
        _python(
            """
            import sys

            data = sys.stdin.buffer.read()
            print(len(data))
            """
        ),
        prompt="",
        session_id=None,
        cwd=tmp_path,
        timeout=5,
        prompt_on_stdin=True,
        stream=True,
    )

    assert proc.stdout == "0\n"
    assert capsys.readouterr().err == "0\n"


def test_streaming_runner_flushes_a_final_carriage_return(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    proc = run_cli_turn(
        "fixture",
        _python(
            """
            import os
            import sys

            os.write(1, b"partial\\r")
            os.close(sys.stdout.fileno())
            os.close(sys.stderr.fileno())
            """
        ),
        prompt="ignored",
        session_id=None,
        cwd=tmp_path,
        timeout=5,
        stream=True,
    )

    assert proc.stdout == "partial\n"
    assert proc.stderr == ""
    assert capsys.readouterr().err == "partial\n"


def test_streaming_runner_honors_the_requested_working_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    proc = run_cli_turn(
        "fixture",
        _python("import pathlib; print(pathlib.Path.cwd())"),
        prompt="ignored",
        session_id=None,
        cwd=tmp_path,
        timeout=5,
        stream=True,
    )

    assert proc.stdout == f"{tmp_path}\n"
    assert capsys.readouterr().err == f"{tmp_path}\n"


def test_streaming_runner_uses_the_locale_text_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[bool] = []

    def preferred_encoding(do_setlocale: bool) -> str:
        calls.append(do_setlocale)
        return "utf-8"

    monkeypatch.setattr(base.locale, "getpreferredencoding", preferred_encoding)
    proc = run_cli_turn(
        "fixture",
        _python('print("encoded")'),
        prompt="ignored",
        session_id=None,
        cwd=tmp_path,
        timeout=5,
        stream=True,
    )

    assert proc.stdout == "encoded\n"
    assert calls == [False, False, False]


def test_streaming_runner_uses_bounded_pipe_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_select = base.select.select
    real_read = base.os.read
    select_returned = False
    read_sizes: list[int] = []

    def select_and_enable_recording(
        rlist: list[int],
        wlist: list[int],
        xlist: list[int],
        timeout: float,
    ) -> tuple[list[int], list[int], list[int]]:
        nonlocal select_returned
        select_returned = True
        return real_select(rlist, wlist, xlist, timeout)

    def read_and_record_size(fd: int, size: int) -> bytes:
        if select_returned:
            read_sizes.append(size)
        return real_read(fd, size)

    monkeypatch.setattr(base.select, "select", select_and_enable_recording)
    monkeypatch.setattr(base.os, "read", read_and_record_size)
    proc = run_cli_turn(
        "fixture",
        _python("print('bounded')"),
        prompt="ignored",
        session_id=None,
        cwd=tmp_path,
        timeout=5,
        stream=True,
    )

    assert proc.stdout == "bounded\n"
    assert read_sizes
    assert set(read_sizes) == {65536}


def test_streaming_runner_handles_a_child_that_closes_stdin(
    tmp_path: Path,
) -> None:
    proc = run_cli_turn(
        "fixture",
        _python(
            """
            import os
            import sys
            import time

            os.close(sys.stdin.fileno())
            time.sleep(0.2)
            print("closed")
            """
        ),
        prompt="p" * (256 * 1024),
        session_id=None,
        cwd=tmp_path,
        timeout=5,
        prompt_on_stdin=True,
        stream=True,
    )

    assert proc.stdout == "closed\n"


def test_streaming_runner_retries_transient_pipe_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_select = base.select.select
    real_read = base.os.read
    select_interrupted = False
    allow_read_race = False
    read_raced = False

    def select_with_one_interrupt(
        rlist: list[int],
        wlist: list[int],
        xlist: list[int],
        timeout: float,
    ) -> tuple[list[int], list[int], list[int]]:
        nonlocal allow_read_race, select_interrupted
        if not select_interrupted:
            select_interrupted = True
            raise InterruptedError
        allow_read_race = True
        return real_select(rlist, wlist, xlist, timeout)

    def read_with_one_race(fd: int, size: int) -> bytes:
        nonlocal read_raced
        if allow_read_race and not read_raced:
            read_raced = True
            raise BlockingIOError
        return real_read(fd, size)

    monkeypatch.setattr(base.select, "select", select_with_one_interrupt)
    monkeypatch.setattr(base.os, "read", read_with_one_race)

    proc = run_cli_turn(
        "fixture",
        _python('print("ready")'),
        prompt="ignored",
        session_id=None,
        cwd=tmp_path,
        timeout=5,
        stream=True,
    )

    assert proc.stdout == "ready\n"
    assert select_interrupted
    assert read_raced


def test_streaming_runner_times_out_before_waiting_when_budget_is_spent(
    tmp_path: Path,
) -> None:
    with pytest.raises(subprocess.TimeoutExpired) as excinfo:
        run_cli_turn(
            "fixture",
            _python("import time; time.sleep(30)"),
            prompt="ignored",
            session_id=None,
            cwd=tmp_path,
            timeout=0,
            stream=True,
        )

    assert excinfo.value.timeout == 0


def test_streaming_runner_applies_deadline_to_process_wait(
    tmp_path: Path,
) -> None:
    with pytest.raises(subprocess.TimeoutExpired) as excinfo:
        run_cli_turn(
            "fixture",
            _python(
                """
                import os
                import time

                os.close(1)
                os.close(2)
                time.sleep(30)
                """
            ),
            prompt="ignored",
            session_id=None,
            cwd=tmp_path,
            timeout=1,
            stream=True,
        )

    assert excinfo.value.output is None
    assert excinfo.value.stderr is None


def test_streaming_runner_reaps_a_child_when_decoding_fails(
    tmp_path: Path,
) -> None:
    pid_file = tmp_path / "child-pid"
    with pytest.raises(UnicodeDecodeError):
        run_cli_turn(
            "fixture",
            _python(
                """
                import os
                import pathlib
                import sys
                import time

                pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))
                os.write(1, b"\\xff")
                time.sleep(30)
                """,
                str(pid_file),
            ),
            prompt="ignored",
            session_id=None,
            cwd=tmp_path,
            timeout=5,
            stream=True,
        )

    child_pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


@pytest.mark.parametrize(
    ("backend_cls", "module", "stdout"),
    [
        (
            claude.ClaudeBackend,
            claude,
            '{"type":"result","session_id":"s1","result":"reply"}',
        ),
        (
            codex.CodexBackend,
            codex,
            '{"type":"thread.started","thread_id":"s1"}\n'
            '{"type":"item.completed","item":{"type":"agent_message",'
            '"text":"reply"}}\n',
        ),
        (grok.GrokBackend, grok, '{"sessionId":"s1","text":"reply"}'),
        (
            opencode.OpenCodeBackend,
            opencode,
            '{"type":"text","sessionID":"s1","part":{"id":"p1","text":"reply"}}\n',
        ),
    ],
)
def test_each_backend_forwards_stream_to_the_shared_runner(
    backend_cls: type[base.AgentBackend],
    module: object,
    stdout: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_cli_turn(
        *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(kwargs)
        return subprocess.CompletedProcess(
            cast(list[str], args[1]), 0, stdout=stdout, stderr=""
        )

    monkeypatch.setattr(module, "run_cli_turn", fake_run_cli_turn)
    result = backend_cls().run_turn("prompt", None, tmp_path, stream=True)

    assert result.reply == "reply"
    assert calls[0]["stream"] is True


def test_agent_forwards_turn_arguments_and_keeps_stream_opt_in(
    tmp_path: Path,
) -> None:
    calls = _scripted_backend("agent-streaming", ["reply"])
    agent = orchestrator.Agent(
        "agent", get_backend("agent-streaming"), workdir=tmp_path
    )
    agent.pending_fork_from = "source-session"

    result = agent.talk("prompt", timeout=7, stream=True)

    assert result.reply == "reply"
    assert calls == [
        {
            "prompt": "prompt",
            "session_id": "source-session",
            "cwd": tmp_path,
            "timeout": 7,
            "resume_as_fork": True,
            "stream": True,
            "schema": None,
        }
    ]

    default_calls = _scripted_backend("agent-default", ["reply"])
    default_agent = orchestrator.Agent(
        "agent", get_backend("agent-default"), workdir=tmp_path
    )
    default_agent.talk("default prompt")

    assert default_calls[0]["stream"] is False


def test_orchestrator_forwards_stream_on_plain_turns(tmp_path: Path) -> None:
    calls = _scripted_backend("plain-streaming", ["reply"])
    orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "state.json"))
    orch.spawn("agent", "plain-streaming")

    result = orch.talk("agent", "prompt", stream=True)

    assert result.reply == "reply"
    assert calls[0]["prompt"] == "prompt"
    assert calls[0]["stream"] is True


def test_orchestrator_forwards_stream_through_schema_repair_attempts(
    tmp_path: Path,
) -> None:
    schema = base.OutputSchema(
        text=(
            '{"type":"object","additionalProperties":false,"required":[],'
            '"properties":{}}'
        ),
        path=tmp_path / "schema.json",
    )
    calls = _scripted_backend("streaming-scripted", ["not json", "{}"])
    orch = Orchestrator(runtime_paths(tmp_path, state_file=tmp_path / "state.json"))
    orch.spawn("agent", "streaming-scripted")

    result = orch.talk("agent", "prompt", schema=schema, retries=1, stream=True)

    assert result.structured == {}
    assert all(isinstance(call["prompt"], str) for call in calls)
    assert [call["stream"] for call in calls] == [True, True]


def test_talk_stream_flag_reaches_the_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeOrchestrator:
        def __init__(self) -> None:
            self.calls: list[bool] = []

        def ensure(self, *args: object, **kwargs: object) -> tuple[object, bool]:
            return object(), False

        def talk(self, *args: object, **kwargs: object) -> base.TurnResult:
            self.calls.append(cast(bool, kwargs["stream"]))
            return base.TurnResult(session_id="sid", reply="reply", raw="reply")

    fake = FakeOrchestrator()
    monkeypatch.setattr(orchestrator, "Orchestrator", lambda *_: fake)

    main(["talk", "agent", "--stream", "-p", "prompt"])
    main(["talk", "agent", "-p", "prompt"])

    assert fake.calls == [True, False]
    assert (
        capsys.readouterr().out
        == "[agent session=sid]\nreply\n[agent session=sid]\nreply\n"
    )
