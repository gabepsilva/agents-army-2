#!/usr/bin/env python3
"""Run Gabriel's local-first development workflow V2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from examples.gabriels_workflow.development_workflow import (
    WorkflowError,
    configure_logging,
)
from examples.gabriels_workflow_v2.config import DEFAULT_CONFIG_PATH, load_config
from examples.gabriels_workflow_v2.setup import prepare_workflow


def _positive(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected 1 or more, got {value}")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Gabriel's budgeted local-first development workflow V2."
    )
    parser.add_argument("issue", type=_positive, help="GitHub issue number")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"workflow configuration (default: {DEFAULT_CONFIG_PATH.name})",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="log every prompt, reply, and subprocess",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    options = _parser().parse_args(argv)
    configure_logging(options.verbose)
    try:
        workflow = prepare_workflow(options.issue, load_config(options.config))
        url = workflow.run()
    except WorkflowError as exc:
        print(f"V2 workflow stopped: {exc}", file=sys.stderr)
        return 1
    print(url)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
