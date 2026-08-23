#!/usr/bin/env python3
"""Reclaim disk from closed V2 runs under `$GIT_COMMON_DIR/gdw-v2`."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from examples.gabriels_workflow_v2.config import DEFAULT_CONFIG_PATH, load_config
from examples.gabriels_workflow_v2.errors import WorkflowError, configure_logging
from examples.gabriels_workflow_v2.retention import prune_issue_state
from examples.gabriels_workflow_v2.workflow import RelayRepository


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remove completed and stale V2 issue state directories."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"workflow configuration (default: {DEFAULT_CONFIG_PATH.name})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list what would be removed without deleting anything",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="log every filesystem and subprocess step",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Report every directory reclaimed, on the failing exit path too.

    A prune that steps over one wedged directory still returns the ones it
    freed, and an error raised after that must not swallow them: a caller
    piping stdout into a cleanup log would otherwise be told nothing was
    freed when directories were.
    """

    options = _parser().parse_args(argv)
    configure_logging(options.verbose)
    removed: list[Path] = []
    status = 0
    try:
        config = load_config(options.config)
        repository = RelayRepository(Path.cwd().resolve())
        gdw_root = repository.common_git_dir() / "gdw-v2"
        removed = prune_issue_state(
            gdw_root,
            config.retention,
            repository,
            now=datetime.now(UTC),
            dry_run=options.dry_run,
        )
    except WorkflowError as exc:
        print(f"V2 prune stopped: {exc}", file=sys.stderr)
        status = 1
    for path in removed:
        print(path)
    return status


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
