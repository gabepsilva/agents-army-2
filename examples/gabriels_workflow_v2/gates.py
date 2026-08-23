"""What one `make ci` run means, read back out of its own output.

Under `make -j` the gates finish interleaved and make names only the target
that failed, so the gates' own `=== gate: NAME ===` banners are the only
thing tying a line of output to the gate that wrote it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

CI_EVIDENCE_CHARS = 20_000

# The braille block, which every spinner this project's gates print draws from.
SPINNER_FRAME = re.compile(r"^[\u2800-\u28ff]+\s*")

# `make: *** [Makefile:85: mutation] Error 1` — which gate failed, not where.
FAILED_TARGET = re.compile(r"^make: \*\*\* \[[^\]]*: (\S+)\] Error", re.MULTILINE)

# `=== gate: lint ===`, printed by the Makefile's own `gate` macro before a
# gate's first command. Under `make -j` this is the only thing tying a line of
# output to the gate that wrote it.
GATE_ANNOUNCE = re.compile(r"^=== gate: (\S+) ===$", re.MULTILINE)

GATE_MARKS = {"passed": "✅", "failed": "❌", "not run": "⚪"}

# A headline, not a log: the reason is there to say what broke, and the
# evidence the repair agent works from is kept whole in the checkpoint.
GATE_REASON_WORDS = 15

GATE_NOT_RUN = "not run"


@dataclass(frozen=True)
class GateResult:
    """How one CI gate ended, and in one line, why it refused."""

    name: str
    status: str
    reason: str = ""

    def as_json(self) -> dict:
        return {"name": self.name, "status": self.status, "reason": self.reason}


@dataclass(frozen=True)
class CommandResult:
    """The bounded evidence retained from a subprocess."""

    returncode: int
    output: str
    gates: tuple[GateResult, ...] = ()

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0

    def checklist(self) -> list[str]:
        """One check per gate, for a reader who wants the verdict not the log.

        A run whose gates could not be identified still owes that reader a
        verdict, so it reports as the single command it actually was.
        """

        if self.gates:
            return _gate_checklist(self.gates)
        whole = GateResult(
            "make ci",
            "passed" if self.succeeded else "failed",
            "" if self.succeeded else f"exit {self.returncode}, no gate named itself",
        )
        return _gate_checklist((whole,))

    def as_json(self) -> dict:
        return {
            "returncode": self.returncode,
            "output": self.output,
            "gates": [gate.as_json() for gate in self.gates],
        }


def readable(text: str) -> str:
    """Collapse in-place progress redraws so the real errors survive bounding.

    mutmut repaints one status line per mutant; kept verbatim those frames
    push the diagnostics out of the retained tail. splitlines() already breaks
    each carriage-return repaint onto its own line, so dropping the spinner
    glyph leaves consecutive duplicates, which collapse.
    """

    kept: list[str] = []
    for raw in text.splitlines():
        line = SPINNER_FRAME.sub("", raw)
        if kept and kept[-1] == line:
            continue
        kept.append(line)
    return "\n".join(kept)


def _gate_blocks(output: str) -> dict[str, str]:
    """The output each gate produced, split apart at the gates' own headers."""

    blocks: dict[str, str] = {}
    headers = list(GATE_ANNOUNCE.finditer(output))
    for index, header in enumerate(headers):
        following = (
            headers[index + 1].start() if index + 1 < len(headers) else len(output)
        )
        name = header.group(1)
        blocks[name] = blocks.get(name, "") + output[header.end() : following]
    return blocks


def _gate_reason(block: str) -> str:
    """The last thing a failing gate said, cut down to a headline.

    Read backwards, because a gate states its verdict last, but take whole
    lines until there are enough words to say something: the very last line is
    often the advice that follows the failure rather than the failure itself.
    make's own `*** [Error]` bookkeeping is skipped — it says nothing a reader
    cannot already see from the red cross.
    """

    tail: list[str] = []
    spoken = 0
    for line in reversed(block.splitlines()):
        words = line.split()
        if not words or words[0].startswith("make"):
            continue
        tail.append(line.strip())
        spoken += len(words)
        if spoken >= GATE_REASON_WORDS:
            break
    if not tail:
        return "failed without saying anything"
    reason = " ".join(reversed(tail)).split()
    if len(reason) <= GATE_REASON_WORDS:
        return " ".join(reason)
    return " ".join(reason[:GATE_REASON_WORDS]) + " …"


def gate_results(expected: Sequence[str], output: str) -> tuple[GateResult, ...]:
    """Read each gate's verdict back out of one interleaved CI log.

    A gate that announced itself and was never reported as failing passed:
    make waits for its running jobs before giving up, so a gate that started
    also finished. One that never announced never started, which is a
    different thing from passing and is reported as such.
    """

    blocks = _gate_blocks(output)
    failed = set(FAILED_TARGET.findall(output))
    surprises = sorted((set(blocks) | failed).difference(expected))
    results = []
    for name in [*expected, *surprises]:
        if name in failed:
            results.append(
                GateResult(name, "failed", _gate_reason(blocks.get(name, "")))
            )
        elif name in blocks:
            results.append(GateResult(name, "passed"))
        else:
            results.append(GateResult(name, GATE_NOT_RUN, GATE_NOT_RUN))
    return tuple(results)


def _gate_checklist(gates: Sequence[GateResult]) -> list[str]:
    return [
        f"{GATE_MARKS[gate.status]} {gate.name}"
        + (f" — {gate.reason}" if gate.reason else "")
        for gate in gates
    ]


def bounded(text: str, limit: int = CI_EVIDENCE_CHARS) -> str:
    if len(text) <= limit:
        return text
    return f"… output truncated …\n{text[-limit:]}"
