"""Stop mutmut's timeouts from dumping core.

mutmut bounds each mutant with RLIMIT_CPU and its watchdog sends SIGXCPU
(see mutmut/__main__.py and mutmut/threading/timeout.py). SIGXCPU's default
action is a core dump, so every timed-out mutant lands in systemd-coredump --
on a desktop that is a "process crashed" toast per timeout, all run long.

Clearing PR_SET_DUMPABLE makes the kernel skip the dump entirely. mutmut's
children are forked, never exec'd, so they inherit the flag, and they still
die from SIGXCPU: the timeout classification is unchanged. Setting RLIMIT_CORE
to 0 does not work here -- systemd-coredump still journals the event.

Python imports sitecustomize at startup, so putting this directory on
PYTHONPATH covers mutmut without touching the gate's command line.
"""

import contextlib
import ctypes

PR_SET_DUMPABLE = 4

# Non-Linux, or no prctl: the core dumps are a desktop annoyance, never a
# correctness concern, so a failure here must not take the gate down.
with contextlib.suppress(OSError, AttributeError):
    ctypes.CDLL(None).prctl(PR_SET_DUMPABLE, 0, 0, 0, 0)
