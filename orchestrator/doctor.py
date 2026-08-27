"""Report the project version and local dependency availability."""

from __future__ import annotations

import importlib.metadata
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path


def _project_version() -> str | None:
    """Read the version from the checkout containing this package, if valid."""
    project_file = Path(__file__).resolve().parent.parent / "pyproject.toml"
    try:
        with project_file.open("rb") as stream:
            document = tomllib.load(stream)
    except (OSError, ValueError, AttributeError, TypeError):
        return None
    project = document.get("project")
    if not isinstance(project, dict):
        return None
    version = project.get("version")
    return version if isinstance(version, str) and version else None


def _resolve_version() -> str:
    """Resolve the distribution version without touching CLI runtime state."""
    version = _project_version()
    if version is not None:
        return version
    try:
        installed_version = importlib.metadata.version("agents-army")
    except (importlib.metadata.PackageNotFoundError, ValueError, TypeError):
        raise ValueError from None
    if not isinstance(installed_version, str) or not installed_version:
        raise ValueError
    return installed_version


def _print_version() -> None:
    try:
        version = _resolve_version()
    except (ValueError, TypeError):
        print("unable to determine agents-army version", file=sys.stderr)
        raise SystemExit(1) from None
    print(version)


# The interpreter floor from pyproject's requires-python. Duplicated as a
# tuple because sys.version_info is what the running process can be compared
# against, and parsing the specifier back out of the metadata would report on
# the checkout rather than on the interpreter actually executing this.
MIN_PYTHON = (3, 11)

# Every tool `doctor` reports, in the order it prints them, paired
# with whether its absence is fine. Only jq is optional: agent CLIs
# are listed separately rather than collapsed into one "at least one" line, so
# the report says which backends this machine can actually run.
DEPENDENCY_TOOLS: tuple[tuple[str, bool], ...] = (
    ("uv", False),
    ("claude", False),
    ("codex", False),
    ("grok", False),
    ("opencode", False),
    ("jq", True),
)

# Present and required, present and optional, absent.
FOUND = "\u2713"
FOUND_OPTIONAL = "\u25cb"
NOT_FOUND = "\u2717"

# What a CLI may put between its own name and its version number, when it
# prints the name at all: `uv 0.4.18` against `jq-1.7`.
NAME_SEPARATORS = (" ", "-")

# A version probe is a courtesy, not the check: a CLI that hangs on --version
# must not hang the report, so it gets seconds rather than the turn timeout.
VERSION_PROBE_TIMEOUT = 5


def _tool_version(tool: str) -> str | None:
    """The first line of `<tool> --version`, or None if it cannot be had.

    Every failure mode is the same answer — the tool is installed and its
    version is unknown — so a CLI that is missing its runtime, hangs, exits
    non-zero, or prints nothing degrades the line instead of the command.
    """
    try:
        proc = subprocess.run(
            [tool, "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=VERSION_PROBE_TIMEOUT,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    lines = proc.stdout.splitlines()
    first_line = lines[0].strip() if lines else ""
    return first_line or None


def _describe_version(tool: str, reported: str) -> str:
    """`<tool> <version>`, without repeating a name the tool printed itself.

    The CLIs disagree about their own version line: `uv --version` prints
    "uv 0.4.18", `jq --version` prints "jq-1.7", `claude --version` prints
    a bare number, and `codex --version` prints "codex-cli 0.147.0". A leading
    copy of the tool's name is dropped only when a version number is what
    follows it, so codex keeps the product name it actually reports instead
    of being rewritten into "codex cli".
    """
    remainder = reported.removeprefix(tool)
    version = remainder[1:] if remainder[:1] in NAME_SEPARATORS else remainder
    if version[:1].isdigit():
        return f"{tool} {version}"
    if reported.startswith(tool):
        return reported
    return f"{tool} {reported}"


def _status_line(symbol: str, subject: str, note: str | None, optional: bool) -> str:
    """One report line, with its parenthesised notes rendered at most once."""
    notes = [note] if note is not None else []
    if optional:
        notes.append("optional")
    if not notes:
        return f"{symbol} {subject}"
    return f"{symbol} {subject} ({', '.join(notes)})"


def _python_line() -> str:
    """The running interpreter, checked against the floor this project needs.

    Not routed through `_status_line`: the interpreter is not a PATH lookup
    and can never be the optional half of that signature.
    """
    running = ".".join(str(part) for part in sys.version_info[:3])
    if (sys.version_info[0], sys.version_info[1]) >= MIN_PYTHON:
        return f"{FOUND} Python {running}"
    required = ".".join(str(part) for part in MIN_PYTHON)
    return f"{NOT_FOUND} Python {running} (needs {required}+)"


def _tool_line(tool: str, optional: bool) -> str:
    """One tool's line: found via PATH, with a version where one is available."""
    if shutil.which(tool) is None:
        return _status_line(NOT_FOUND, tool, "not found", optional)
    symbol = FOUND_OPTIONAL if optional else FOUND
    reported = _tool_version(tool)
    if reported is None:
        return _status_line(symbol, tool, "version unknown", optional)
    return _status_line(symbol, _describe_version(tool, reported), None, optional)


def _dependency_report() -> list[str]:
    """Every line of the setup report, in the fixed order it is printed."""
    return [
        _python_line(),
        *(_tool_line(tool, optional) for tool, optional in DEPENDENCY_TOOLS),
    ]


def _print_dependency_check() -> None:
    """Report the setup and stop.

    A status report, not a gate: it exits 0 whether every tool is present or
    none of them are, because which backends are usable is the user's call and
    a missing optional jq is not a failure at all.
    """
    for line in _dependency_report():
        print(line)
