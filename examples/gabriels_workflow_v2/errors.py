"""The failure vocabulary and progress channel every V2 module shares.

Errors and logging live together because everything else in this package
imports both and neither is large enough to earn a file of its own. Keeping
them in one leaf module with no intra-package imports also means no other
module can create an import cycle by reaching for them.
"""

from __future__ import annotations

import logging
import sys

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(message)s"

LOGGER = logging.getLogger("gdw")


def configure_logging(verbose: bool = False) -> None:
    """Send timestamped progress to stderr so stdout stays the result channel.

    Only this workflow's own logger is touched: the root logger is left to
    whatever embeds the example, and a second call replaces the handler rather
    than printing every line twice.
    """

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    LOGGER.handlers.clear()
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.DEBUG if verbose else logging.INFO)


class WorkflowError(RuntimeError):
    """A workflow failure with a concise user-facing message."""


class WorkflowStopped(WorkflowError):
    """A deliberate terminal outcome rather than an infrastructure failure."""
