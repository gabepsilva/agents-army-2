"""Fixtures every test file gets.

Deliberately thin: helpers belong next to the tests that use them. What lives
here is process-global state the code under test mutates and never restores.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest

import orchestrator.cli as cli
from backends.registry import register_backend
from tests.backend_helpers import EchoBackend


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


@pytest.fixture(autouse=True)
def register_echo_backend() -> None:
    """Registered for every test, not just the class that introduced it.

    The registry is module-level state, so a class relying on another class
    having registered it first passes or fails on test ordering — which xdist
    is free to change.
    """
    register_backend("echo", EchoBackend)
