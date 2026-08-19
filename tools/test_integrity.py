#!/usr/bin/env python3
"""Reject tests that cannot fail and skips that hide behavior.

A test written only to satisfy a coverage gate passes without checking
anything, and a skipped test is indistinguishable from a passing one on a
green run. Both are invisible to pytest, ruff, and coverage, so they are
checked here.

This deliberately accepts assertion styles this suite already uses that carry
no bare ``assert``: ``pytest.raises``/``pytest.warns`` context managers, and
fakes that raise ``AssertionError`` when a prohibited path is reached.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import TypeGuard

TESTS_DIR = Path("tests")

# A skip or xfail must name the work that will remove it.
ISSUE_MARKERS = ("#", "http://", "https://")


def _is_test(node: ast.AST) -> TypeGuard[ast.FunctionDef | ast.AsyncFunctionDef]:
    return isinstance(
        node, ast.FunctionDef | ast.AsyncFunctionDef
    ) and node.name.startswith("test_")


def _attr_path(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _has_assertion_signal(test: ast.AST) -> bool:
    for node in ast.walk(test):
        if isinstance(node, ast.Assert):
            return True
        # `with pytest.raises(...)` / `pytest.warns(...)` and bare calls.
        if isinstance(node, ast.Call):
            path = _attr_path(node.func)
            if path.endswith(("pytest.raises", "pytest.warns", "pytest.fail")):
                return True
            # unittest.mock style: mock.assert_called_with(...)
            if path.split(".")[-1].startswith("assert_"):
                return True
        # A fake that raises AssertionError when a prohibited path is taken is
        # an inverted assertion, not an unchecked test.
        if isinstance(node, ast.Raise):
            exc = node.exc
            name = exc.func if isinstance(exc, ast.Call) else exc
            if isinstance(name, ast.Name) and name.id == "AssertionError":
                return True
    return False


def _skip_failures(tree: ast.AST, path: Path) -> list[str]:
    failures = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = _attr_path(node.func)
        if not any(
            target.endswith(marker)
            for marker in ("pytest.skip", "mark.skip", "mark.skipif", "mark.xfail")
        ):
            continue
        reasons = [
            kw.value.value
            for kw in node.keywords
            if kw.arg == "reason"
            and isinstance(kw.value, ast.Constant)
            and isinstance(kw.value.value, str)
        ]
        if not reasons or not any(
            marker in reason for reason in reasons for marker in ISSUE_MARKERS
        ):
            failures.append(
                f"{path}:{node.lineno}: {target} needs reason= naming an issue "
                f"or URL; a skipped test reads as a passing one."
            )
    # Bare decorator form: @pytest.mark.xfail with no call at all.
    for node in ast.walk(tree):
        if _is_test(node):
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Attribute) and decorator.attr in {
                    "skip",
                    "xfail",
                }:
                    failures.append(
                        f"{path}:{decorator.lineno}: bare @...{decorator.attr} "
                        f"on {node.name} needs reason= naming an issue or URL."
                    )
    return failures


def main() -> int:
    failures: list[str] = []
    for path in sorted(TESTS_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        failures.extend(_skip_failures(tree, path))
        for node in ast.walk(tree):
            if _is_test(node) and not _has_assertion_signal(node):
                failures.append(
                    f"{path}:{node.lineno}: {node.name} has no assertion; a test "
                    f"that cannot fail does not protect the behavior it names."
                )

    for failure in failures:
        print(f"error: {failure}")
    if failures:
        print(f"\n{len(failures)} test-integrity failure(s).")
        return 1

    print("test integrity: every test can fail; no unexplained skips.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
