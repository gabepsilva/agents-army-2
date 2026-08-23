#!/usr/bin/env python3
"""Reject skips and expected failures that hide behavior indefinitely.

An assertion-presence scan cannot distinguish a behavioral test from
``assert True`` and rejects valid tests that fail through an exception or a
purpose-built verifier. Test quality belongs in behavioral assertions,
regression proof, and mutation testing—not in a syntactic assertion quota.
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

    for failure in failures:
        print(f"error: {failure}")
    if failures:
        print(f"\n{len(failures)} test-integrity failure(s).")
        return 1

    print("test integrity: no unexplained skips or expected failures.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
