#!/usr/bin/env python3
"""Turn one GitHub issue into a reviewed draft pull request."""

import argparse
from pathlib import Path

from examples.gabriels_workflow.config import DEFAULT_CONFIG_PATH, load_config
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
    return parser


def main(argv: list[str] | None = None) -> int:
    options = _parser().parse_args(argv)
    workflow = prepare_workflow(options.issue, load_config(options.config))

    pull_request_url = workflow.completed_url()

    if pull_request_url is None:
        issue = workflow.load_issue()
        proposal = workflow.clarify(issue)
        specification = workflow.specify(issue, proposal)
        workflow.implement(specification)
        passing_ci = workflow.stabilize(specification)
        approvals = workflow.review(specification, passing_ci)
        pull_request_url = workflow.publish(specification, approvals)

    print(pull_request_url)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    main()
