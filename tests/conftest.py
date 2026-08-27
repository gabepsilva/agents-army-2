"""Fixtures every test file gets.

Deliberately thin: helpers belong next to the tests that use them. What lives
here is process-global state the code under test mutates and never restores.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest

import orchestrator.cli as cli


@pytest.fixture(autouse=True)
def _restore_own_logger_levels() -> Iterator[None]:
    """Put this project's logger levels back after every test.

    `_configure_logging` raises them by name for `-v`/`-vv` and nothing ever
    lowers them again, so a single `main(["-v", ...])` leaves every later
    test in the same process logging at DEBUG. Tests asserting an exact set
    of records then fail on records the code was right to emit — and only
    when pytest-randomly happens to order the two that way, which is exactly
    the shape of flake that reads as a mutant killed by luck.
    """
    saved = {name: logging.getLogger(name).level for name in cli.OWN_LOGGERS}
    try:
        yield
    finally:
        for name, level in saved.items():
            logging.getLogger(name).setLevel(level)
