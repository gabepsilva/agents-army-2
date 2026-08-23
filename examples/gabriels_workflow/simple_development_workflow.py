#!/usr/bin/env python3
"""Turn one GitHub issue into a reviewed draft pull request.

The issue conversation ends at the specification. After that, bot comments
belong on the pull request the implementer opens.
"""

import argparse
import sys
import time
from pathlib import Path

from examples.gabriels_workflow.config import DEFAULT_CONFIG_PATH, load_config
from examples.gabriels_workflow.development_workflow import (
    LOGGER,
    WorkflowError,
    configure_logging,
)
from examples.gabriels_workflow.setup import prepare_workflow


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Turn one GitHub issue into a reviewed draft pull request."
    )
    parser.add_argument("issue", type=int, help="GitHub issue number")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"workflow YAML (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="log every prompt, reply, and subprocess at DEBUG level",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    options = _parser().parse_args(argv)
    configure_logging(options.verbose)
    started = time.monotonic()
    LOGGER.info("workflow: starting issue #%s with %s", options.issue, options.config)

    try:
        workflow = prepare_workflow(options.issue, load_config(options.config))

        pull_request_url = workflow.completed_url()

        if pull_request_url is None and workflow._publication_complete():
            pull_request_url = workflow.finalize_from_checkpoints()

        if pull_request_url is None:
            issue = workflow.load_issue()
            proposal = workflow.clarify(issue)
            specification = workflow.specify(issue, proposal)
            workflow.open_pull_request(specification)
            workflow.implement(specification)
            passing_ci = workflow.stabilize(specification)
            approvals = workflow.review(specification, passing_ci)
            pull_request_url = workflow.publish(specification, approvals)
            pull_request_url = workflow.finalize(issue)
    except WorkflowError as exc:
        LOGGER.error(
            "workflow: stopped after %.1fs: %s", time.monotonic() - started, exc
        )
        LOGGER.error(
            "workflow: rerun the same command to resume from the last checkpoint"
        )
        print(f"workflow stopped: {exc}", file=sys.stderr)
        return 1

    LOGGER.info("workflow: finished in %.1fs", time.monotonic() - started)
    print(pull_request_url)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
