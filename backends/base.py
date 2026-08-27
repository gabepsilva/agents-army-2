"""Abstract base interface for coding-agent CLI backends."""

from __future__ import annotations

import codecs
import io
import json
import locale
import logging
import os
import select
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, cast

# One turn's wall-clock ceiling, shared by every backend and by the
# orchestrator that budgets a retry loop against it. A per-backend literal
# would let one CLI drift away from the number the orchestrator plans with.
#
# TODO(pipeline): one number for every stage is the wrong shape. A grill round
# answers in a minute; an `implement` turn against a real spec can run for
# most of an hour, and when it overruns the reply is lost even though the
# session survives. 3600 is the interim ceiling picked for the longest stage,
# which means every short stage now waits an hour before it gives up. Replace
# it with a per-turn budget the caller sets -- a `--timeout` flag on the CLI,
# and a per-stage value in the workflow driver -- rather than raising this
# again.
DEFAULT_TURN_TIMEOUT = 3600

log = logging.getLogger(__name__)


class TurnError(RuntimeError):
    """A CLI turn returned something the orchestrator cannot use.

    Backend-specific subclasses keep tests and logs precise. The CLI
    catches this base type so a new backend is not a new except-clause.
    """


def describe_command(args: list[str], prompt: str) -> str:
    """Render a CLI invocation for logs with the prompt replaced by its size.

    The prompt is the one unbounded argument, and a verbose run that echoes it
    buries the flags — the part worth reading — under the whole request.

    A backend that has to glue the prompt onto its flag (`--single=<prompt>`,
    so an argument parser cannot read a leading `-` as a flag) is handled too:
    the flag stays readable and only the value is replaced.
    """
    rendered = list(args)
    placeholder = f"<prompt:{len(prompt)}chars>"
    attached = f"={prompt}"
    for i in range(len(rendered) - 1, -1, -1):
        arg = rendered[i]
        if arg == prompt:
            rendered[i] = placeholder
            break
        # `prompt and` guards the empty prompt: every argument ends with
        # "=" + "", and slicing one off by length would eat the flag too.
        if prompt and arg.endswith(attached):
            rendered[i] = f"{arg.removesuffix(prompt)}{placeholder}"
            break
    return " ".join(rendered)


# The argument count is the boundary's own: every value below is something
# subprocess.run or the log line needs, and bundling them into an options
# object would add a type for four call sites to construct and nothing else.
# Everything after the argv is keyword-only instead: `prompt` and
# `session_id` are adjacent and `str` is assignable to `str | None`, so a
# positional swap would type-check and surface only as a prompt sent as a
# resume id.
def run_cli_turn(  # noqa: PLR0913 - flat process arguments, see above
    name: str,
    args: list[str],
    *,
    prompt: str,
    session_id: str | None,
    cwd: Path,
    timeout: int,
    prompt_on_stdin: bool = False,
    stream: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run one backend's already-built argv and log the turn around it.

    Every adapter runs its CLI the same disciplined way, and the discipline is
    what a fourth backend would silently drop: captured text output, no
    `check`, the turn timeout, and a stdin that is never the inherited pipe.
    `session_id` is here only so the log can say whether the turn resumed.

    The two stdin arms are the only real difference. By default stdin is
    closed: a CLI reading a non-tty stdin blocks until it is killed, so a run
    from cron, CI or any host script that is not a terminal would spend the
    whole timeout and return nothing. With `prompt_on_stdin` the prompt is
    written there instead, for OpenCode, which joins positional arguments
    before sending them to the model: omitting that argument makes its
    no-value fallback read stdin verbatim, preserving spaces, quotes, and
    newlines. With `stream`, the same captured streams are read through
    nonblocking pipes and complete stdout lines are echoed to this process's
    flushed stderr while the child runs.
    """
    log.debug(
        "%s turn: cwd=%s resume=%s prompt_chars=%d timeout=%ds",
        name,
        cwd,
        bool(session_id),
        len(prompt),
        timeout,
    )
    log.debug("%s turn: invoking %s", name, describe_command(args, prompt))
    if stream:
        started = time.monotonic()
        proc = _run_streaming_cli_turn(
            args,
            prompt=prompt,
            cwd=cwd,
            timeout=timeout,
            prompt_on_stdin=prompt_on_stdin,
        )
    else:
        # The non-streaming arm intentionally remains the subprocess.run path:
        # its text-mode decoding, input handling, and TimeoutExpired behavior
        # are the compatibility contract for every existing backend.
        # The two arms differ only in the stdin kwarg, spelled out at each call
        # because subprocess.run is overloaded on it and an unpacked mapping
        # leaves the type checker no overload to pick.
        started = time.monotonic()
        if prompt_on_stdin:
            proc = subprocess.run(
                args,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
                input=prompt,
            )
        else:
            proc = subprocess.run(
                args,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
                stdin=subprocess.DEVNULL,
            )
    log.debug(
        "%s turn: exited %d after %.1fs with %d chars of stdout",
        name,
        proc.returncode,
        time.monotonic() - started,
        len(proc.stdout),
    )
    return proc


def _new_text_decoder() -> io.IncrementalNewlineDecoder:
    """Build the incremental equivalent of subprocess text-mode decoding."""
    encoding = locale.getpreferredencoding(False)
    decoder = codecs.getincrementaldecoder(encoding)()
    return io.IncrementalNewlineDecoder(decoder, translate=True)


def _close_pipe(pipe: BinaryIO | None) -> None:
    """Close one Popen pipe, tolerating the already-drained case."""
    if pipe is not None:
        pipe.close()


def _kill_and_reap(process: subprocess.Popen[bytes]) -> None:
    """Kill a timed-out child and wait for its process entry to disappear."""
    with suppress(ProcessLookupError):
        process.kill()
    process.wait()


class _StreamingReader:
    """One nonblocking child output pipe, decoded with text-mode semantics."""

    def __init__(self, pipe: BinaryIO, *, echo_lines: bool) -> None:
        self.pipe = pipe
        self.fd = pipe.fileno()
        self.decoder = _new_text_decoder()
        self.echo_lines = echo_lines
        self.raw = bytearray()
        self.parts: list[str] = []
        self.pending_line = ""

    @property
    def text(self) -> str:
        return "".join(self.parts)

    def read(self) -> bool:
        """Read one ready chunk; return false after reaching EOF."""
        try:
            chunk = os.read(self.fd, 65536)
        except BlockingIOError:
            return True
        if not chunk:
            self._decode(b"", final=True)
            self.pipe.close()
            return False
        self.raw.extend(chunk)
        self._decode(chunk)
        return True

    def _decode(self, chunk: bytes, *, final: bool = False) -> None:
        decoded = self.decoder.decode(chunk, final=final)
        if not decoded:
            return
        self.parts.append(decoded)
        if not self.echo_lines:
            return
        self.pending_line += decoded
        while "\n" in self.pending_line:
            line, self.pending_line = self.pending_line.split("\n", 1)
            sys.stderr.write(f"{line}\n")
            sys.stderr.flush()

    def close(self) -> None:
        self.pipe.close()


class _StreamingInput:
    """A nonblocking stdin pipe that drains a prompt in bounded writes."""

    def __init__(self, pipe: BinaryIO, data: bytes) -> None:
        self.pipe = pipe
        self.fd = pipe.fileno()
        self.pending = memoryview(data)
        if not self.pending:
            self.close()

    @property
    def open(self) -> bool:
        return not self.pipe.closed

    def write(self) -> bool:
        """Write one ready chunk; return false after closing stdin."""
        try:
            written = os.write(self.fd, self.pending)
        except (BrokenPipeError, ConnectionResetError):
            self.close()
            return False
        self.pending = self.pending[written:]
        if not self.pending:
            self.close()
            return False
        return True

    def close(self) -> None:
        self.pipe.close()


def _wait_for_streams(
    readers: dict[int, _StreamingReader],
    input_stream: _StreamingInput | None,
    *,
    args: list[str],
    deadline: float,
    timeout: int,
) -> tuple[list[int], list[int]]:
    """Wait for at least one pipe event against the shared deadline."""
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(args, timeout)
        write_fds = [input_stream.fd] if input_stream is not None else []
        try:
            readable, writable, _ = select.select(
                list(readers), write_fds, [], remaining
            )
        except InterruptedError:
            continue
        if readable or writable:
            return readable, writable
        raise subprocess.TimeoutExpired(args, timeout)


def _read_ready_pipes(
    readable: list[int], readers: dict[int, _StreamingReader]
) -> None:
    """Drain one nonblocking chunk from every output pipe selected as ready."""
    for fd in readable:
        reader = readers[fd]
        if not reader.read():
            del readers[fd]


def _write_ready_input(
    input_stream: _StreamingInput | None, writable: list[int]
) -> _StreamingInput | None:
    """Write one ready stdin chunk and clear the writer after EOF."""
    if (
        input_stream is not None
        and input_stream.fd in writable
        and not input_stream.write()
    ):
        return None
    return input_stream


def _run_streaming_cli_turn(
    args: list[str],
    *,
    prompt: str,
    cwd: Path,
    timeout: int,
    prompt_on_stdin: bool,
) -> subprocess.CompletedProcess[str]:
    """Run a CLI with nonblocking pipes and one absolute deadline.

    Binary reads are decoded locally so a UTF-8 character or CRLF pair split
    across pipe reads has the same text-mode result as ``subprocess.run``.
    Keeping the byte buffers too lets a timeout expose the same raw captured
    values as the standard subprocess boundary.
    """
    started = time.monotonic()
    deadline = started + timeout
    encoding = locale.getpreferredencoding(False)
    input_bytes = prompt.encode(encoding) if prompt_on_stdin else None
    process = subprocess.Popen(
        args,
        cwd=str(cwd),
        stdin=subprocess.PIPE if prompt_on_stdin else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    stdout_reader = _StreamingReader(cast(BinaryIO, process.stdout), echo_lines=True)
    stderr_reader = _StreamingReader(cast(BinaryIO, process.stderr), echo_lines=False)
    stdin_pipe = cast(BinaryIO | None, process.stdin)
    input_stream = (
        _StreamingInput(stdin_pipe, input_bytes)
        if prompt_on_stdin and input_bytes is not None and stdin_pipe is not None
        else None
    )
    if input_stream is not None and not input_stream.open:
        input_stream = None
    for fd in (stdout_reader.fd, stderr_reader.fd):
        os.set_blocking(fd, False)
    if input_stream is not None:
        os.set_blocking(input_stream.fd, False)
    readers = {stdout_reader.fd: stdout_reader, stderr_reader.fd: stderr_reader}

    try:
        while readers or input_stream is not None:
            readable, writable = _wait_for_streams(
                readers,
                input_stream,
                args=args,
                deadline=deadline,
                timeout=timeout,
            )
            _read_ready_pipes(readable, readers)
            input_stream = _write_ready_input(input_stream, writable)

        try:
            returncode = process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            raise subprocess.TimeoutExpired(args, timeout) from None
    except subprocess.TimeoutExpired:
        _kill_and_reap(process)
        raise subprocess.TimeoutExpired(
            args,
            timeout,
            output=bytes(stdout_reader.raw) or None,
            stderr=bytes(stderr_reader.raw) or None,
        ) from None
    finally:
        _close_pipe(stdin_pipe)
        stdout_reader.close()
        stderr_reader.close()
        if process.poll() is None:
            _kill_and_reap(process)

    return subprocess.CompletedProcess(
        args,
        returncode,
        stdout=stdout_reader.text,
        stderr=stderr_reader.text,
    )


def stdout_for_error(stdout: str) -> str:
    """Keep both ends of a long dump: the parse error is at char 0, the
    envelope is usually at the tail."""
    if len(stdout) <= 2000:
        return stdout
    return f"{stdout[:400]}\n…\n{stdout[-1600:]}"


def json_objects(text: str) -> list[dict]:
    """Scan `text` for every top-level JSON object, in order of appearance.

    CLIs that promise a single envelope still sometimes write a text prefix
    first. json.loads of the whole buffer then fails even though a valid
    object is sitting in the stream.
    """
    decoder = json.JSONDecoder()
    found: list[dict] = []
    idx = 0
    while idx < len(text):
        start = text.find("{", idx)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            idx = start + 1
            continue
        found.append(obj)
        idx = end
    return found


def reply_text(payload: dict, key: str) -> str:
    """The turn's reply as text, whatever the CLI actually put under `key`.

    A missing key and an explicit null both mean the same thing — the turn
    produced no assistant text — and neither is worth failing the turn over:
    the session id it did report is what the next turn needs, and raising
    here would throw that away before the orchestrator could persist it.
    """
    value = payload.get(key)
    return value if isinstance(value, str) else ""


@dataclass(frozen=True)
class OutputSchema:
    """A JSON Schema for a turn's reply, in both forms the CLIs want.

    claude and grok take the document inline as a `--json-schema` argument;
    codex takes a `--output-schema` path and reads the file itself. Carrying
    both spellings of one schema keeps the choice inside each adapter, where
    the rest of that CLI's dialect already lives.
    """

    text: str
    path: Path


@dataclass
class TurnResult:
    """Outcome of one non-interactive turn against a CLI session."""

    session_id: str | None
    reply: str
    raw: str
    # None whenever no schema was asked for, and also when a schema was asked
    # for and the reply was not a JSON object: that is a reply that failed the
    # contract, which the validator retries rather than a parse the adapter
    # should raise on.
    structured: dict | None = None


def structured_reply(
    schema: OutputSchema | None, reply: str, pre_parsed: object = None
) -> dict | None:
    """The reply as the JSON object a schema asked for, or None.

    `pre_parsed` is the CLI's own parse of that object where it publishes one
    (claude's `structured_output`, grok's `structuredOutput`); codex has no
    such field and passes nothing, so its reply text is parsed here instead.

    Everything that is not an object — a reply the model wrapped in prose, a
    bare string, a turn that asked for no schema at all — comes back as None
    rather than raising. Whether that breaks the contract is the validator's
    call, and a raise here would deny it the retry it is entitled to.
    """
    if schema is None:
        return None
    if isinstance(pre_parsed, dict):
        return pre_parsed
    try:
        value = json.loads(reply)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


class AgentBackend(ABC):
    """Abstract interface defining interaction with coding-agent CLIs.

    Different CLIs (Claude, Codex, Grok, OpenCode, etc.) have different flag conventions,
    session lifecycle mechanisms, and execution requirements. Concrete
    subclasses encapsulate these differences.
    """

    enforces_schema: bool = True

    # Whether this CLI can resume a session into a *copy* of it, leaving the
    # original untouched. Declared on the class, like `enforces_schema`, so
    # `fork` can refuse before it creates anything. Every shipped backend
    # answers True; the default stays False so a third-party backend has to
    # opt in once it can emit a fork of its own.
    supports_fork: bool = False

    # Whether this CLI can hand an existing session to a human at a terminal.
    # The default is deliberately conservative: a third-party backend must
    # opt in after checking that its interactive resume keeps the same session
    # id, rather than silently risking a stale registry.
    supports_chat: bool = False

    def __init__(
        self,
        *,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier name for this backend (e.g., 'claude', 'codex')."""
        ...

    @abstractmethod
    def run_turn(
        self,
        prompt: str,
        session_id: str | None,
        cwd: Path,
        timeout: int = DEFAULT_TURN_TIMEOUT,
        schema: OutputSchema | None = None,
        *,
        resume_as_fork: bool = False,
    ) -> TurnResult:
        """Start (session_id=None) or resume (session_id set) a CLI session with
        `prompt` in directory `cwd` and return the model's text reply along with
        the session id to use for the next turn.

        `timeout` and `schema` are declared here rather than left to each
        subclass: a parameter that exists only as a subclass default is one a
        fourth backend can silently drop, and a type checker cannot see the
        omission because Liskov is only checked against a declared base
        signature. When `schema` is set the reply must be a JSON object
        satisfying it, and `TurnResult.structured` carries that object.

        `resume_as_fork` resumes `session_id` into a copy: the turn continues
        that conversation but lands in a new session, whose id the result
        reports, and the session named by `session_id` is left as it was. It
        is keyword-only and defaults off so the ordinary turn — every call
        site but the first turn of a forked agent — reads unchanged. Every
        shipped backend forks; a third-party backend that leaves
        `supports_fork` False owes callers its own `TurnError` here rather
        than a silently unforked turn.
        """
        ...

    def chat_argv(self, session_id: str, cwd: Path) -> list[str]:
        """Build the interactive resume command for ``session_id``.

        The default backend has no interactive contract. Keeping this method
        concrete preserves the extension point for existing third-party
        backends: they can remain unsupported without implementing a method
        they will never call.
        """
        raise NotImplementedError(
            f"backend '{self.name}' has no interactive chat command for {cwd}"
        )
