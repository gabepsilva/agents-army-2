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

_STREAM_CHUNK_SIZE = 64 * 1024


def _default_text_encoding() -> str:
    """Return the encoding used by ``subprocess``'s implicit text mode."""
    return "utf-8" if sys.flags.utf8_mode else locale.getencoding()


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


class _StreamCapture:
    """Decode one nonblocking child pipe with text-mode newline semantics."""

    def __init__(self, *, echo_lines: bool) -> None:
        decoder = codecs.getincrementaldecoder(_default_text_encoding())(
            errors="strict"
        )
        self._decoder = io.IncrementalNewlineDecoder(decoder, translate=True)
        self._echo_lines = echo_lines
        self._raw_chunks: list[bytes] = []
        self._chunks: list[str] = []
        self._line_buffer = ""

    def feed(self, data: bytes) -> None:
        self._raw_chunks.append(data)
        self._append(self._decoder.decode(data, final=False))

    def finish(self) -> str:
        self._append(self._decoder.decode(b"", final=True))
        return self.text

    @property
    def text(self) -> str:
        return "".join(self._chunks)

    @property
    def raw(self) -> bytes | None:
        """Return captured bytes in the shape used by timeout exceptions."""
        return b"".join(self._raw_chunks) or None

    def _append(self, text: str) -> None:
        if not text:
            return
        self._chunks.append(text)
        if not self._echo_lines:
            return
        self._line_buffer += text
        while "\n" in self._line_buffer:
            line, self._line_buffer = self._line_buffer.split("\n", 1)
            sys.stderr.write(f"{line}\n")
            sys.stderr.flush()


def _drain_pipe(pipe: BinaryIO, capture: _StreamCapture) -> bool:
    """Read all currently available bytes, returning whether EOF was seen."""
    file_descriptor = pipe.fileno()
    while True:
        try:
            data = os.read(file_descriptor, _STREAM_CHUNK_SIZE)
        except BlockingIOError:
            return False
        except InterruptedError:
            continue
        if not data:
            return True
        capture.feed(data)


def _reap_after_failure(proc: subprocess.Popen[bytes]) -> None:
    """Stop a streaming child and wait for it, including after a timeout."""
    if proc.poll() is None:
        with suppress(ProcessLookupError):
            proc.kill()
    proc.wait()


def _stream_cli_turn(
    name: str,
    args: list[str],
    *,
    prompt: str,
    cwd: Path,
    timeout: int,
    prompt_on_stdin: bool,
) -> subprocess.CompletedProcess[str]:
    """Run a child while draining all pipes against one absolute deadline."""
    return _StreamingTurn(
        name,
        args,
        prompt=prompt,
        cwd=cwd,
        timeout=timeout,
        prompt_on_stdin=prompt_on_stdin,
    ).run()


class _StreamingTurn:
    """Own a streaming child and the state needed to pump its three pipes."""

    def __init__(
        self,
        name: str,
        args: list[str],
        *,
        prompt: str,
        cwd: Path,
        timeout: int,
        prompt_on_stdin: bool,
    ) -> None:
        self.name = name
        self.args = args
        self.timeout = timeout
        self.started = time.monotonic()
        self.deadline = self.started + timeout
        self.proc = subprocess.Popen(
            args,
            cwd=str(cwd),
            stdin=subprocess.PIPE if prompt_on_stdin else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        stdin_pipe = self.proc.stdin
        stdout_pipe = self.proc.stdout
        stderr_pipe = self.proc.stderr
        if stdout_pipe is None or stderr_pipe is None:
            _reap_after_failure(self.proc)
            raise RuntimeError("streaming child did not expose output pipes")
        self.stdin_pipe: BinaryIO | None = cast(BinaryIO, stdin_pipe)
        self.stdout_pipe: BinaryIO = cast(BinaryIO, stdout_pipe)
        self.stderr_pipe: BinaryIO = cast(BinaryIO, stderr_pipe)
        self.stdout_capture = _StreamCapture(echo_lines=True)
        self.stderr_capture = _StreamCapture(echo_lines=False)
        self.stdout_fd = self._nonblocking(self.stdout_pipe)
        self.stderr_fd = self._nonblocking(self.stderr_pipe)
        self.pending_input = prompt.encode(_default_text_encoding())
        self.input_offset = 0
        self.stdin_fd: int | None = None
        if self.stdin_pipe is not None:
            self.stdin_fd = self._nonblocking(self.stdin_pipe)
            if not self.pending_input:
                self._close_stdin()

    @staticmethod
    def _nonblocking(pipe: BinaryIO) -> int:
        file_descriptor = pipe.fileno()
        os.set_blocking(file_descriptor, False)
        return file_descriptor

    def run(self) -> subprocess.CompletedProcess[str]:
        try:
            self._pump()
            return self._complete()
        except BaseException:
            _reap_after_failure(self.proc)
            raise
        finally:
            for pipe in (self.stdin_pipe, self.stdout_pipe, self.stderr_pipe):
                if pipe is not None:
                    pipe.close()

    def _pump(self) -> None:
        while self._has_open_pipes():
            readable, writable = self._wait_for_ready()
            self._read_ready(readable)
            self._write_ready(writable)

    def _has_open_pipes(self) -> bool:
        return (
            self.stdout_fd is not None
            or self.stderr_fd is not None
            or self.stdin_fd is not None
        )

    def _wait_for_ready(self) -> tuple[list[int], list[int]]:
        while True:
            remaining = self._remaining()
            if remaining <= 0:
                raise self._timeout()
            readable = [fd for fd in (self.stdout_fd, self.stderr_fd) if fd is not None]
            writable = [self.stdin_fd] if self.stdin_fd is not None else []
            try:
                ready_read, ready_write, _ = select.select(
                    readable, writable, [], remaining
                )
            except InterruptedError:
                continue
            return ready_read, ready_write

    def _read_ready(self, readable: list[int]) -> None:
        if (
            self.stdout_fd is not None
            and self.stdout_fd in readable
            and _drain_pipe(self.stdout_pipe, self.stdout_capture)
        ):
            self._close_stdout()
        if (
            self.stderr_fd is not None
            and self.stderr_fd in readable
            and _drain_pipe(self.stderr_pipe, self.stderr_capture)
        ):
            self._close_stderr()

    def _write_ready(self, writable: list[int]) -> None:
        if self.stdin_fd is None or self.stdin_fd not in writable:
            return
        try:
            written = os.write(self.stdin_fd, self.pending_input[self.input_offset :])
        except (BrokenPipeError, ConnectionResetError):
            self._close_stdin()
            return
        self.input_offset += written
        if self.input_offset == len(self.pending_input):
            self._close_stdin()

    def _complete(self) -> subprocess.CompletedProcess[str]:
        remaining = self._remaining()
        if remaining <= 0:
            raise self._timeout()
        try:
            returncode = self.proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            raise self._timeout() from None
        stdout = self.stdout_capture.finish()
        stderr = self.stderr_capture.finish()
        log.debug(
            "%s turn: exited %d after %.1fs with %d chars of stdout",
            self.name,
            returncode,
            time.monotonic() - self.started,
            len(stdout),
        )
        return subprocess.CompletedProcess(
            self.args, returncode, stdout=stdout, stderr=stderr
        )

    def _remaining(self) -> float:
        return self.deadline - time.monotonic()

    def _timeout(self) -> subprocess.TimeoutExpired:
        return subprocess.TimeoutExpired(
            self.args,
            self.timeout,
            output=self.stdout_capture.raw,
            stderr=self.stderr_capture.raw,
        )

    def _close_stdin(self) -> None:
        if self.stdin_pipe is not None:
            self.stdin_pipe.close()
        self.stdin_fd = None

    def _close_stdout(self) -> None:
        self.stdout_pipe.close()
        self.stdout_fd = None

    def _close_stderr(self) -> None:
        self.stderr_pipe.close()
        self.stderr_fd = None


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
    newlines.

    With `stream`, the child is drained through nonblocking pipes and complete
    stdout lines are copied to the parent's flushed stderr as they arrive.
    The returned text remains captured and is decoded with the same text-mode
    newline rules as the non-streaming path.
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
        return _stream_cli_turn(
            name,
            args,
            prompt=prompt,
            cwd=cwd,
            timeout=timeout,
            prompt_on_stdin=prompt_on_stdin,
        )
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
    def run_turn(  # noqa: PLR0913 - flat backend interface, see run_cli_turn
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

        `stream` opts into forwarding complete stdout lines to the parent
        process's stderr while the child is still running. It defaults off so
        the established `subprocess.run` transport remains unchanged.
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
