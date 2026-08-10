"""Fail-closed wall-clock deadlines for synchronous local adapter calls."""

from __future__ import annotations

import signal
import threading
from collections.abc import Callable
from typing import TypeVar

from ragkit.domain import OperationTimeoutError, UnsupportedCapabilityError

_T = TypeVar("_T")


def run_with_deadline(operation: Callable[[], _T], seconds: float, label: str) -> _T:
    """Run one synchronous operation under a process interruptible deadline.

    The reference CLI is synchronous and runs adapters on the main thread. Other
    execution models must provide process-level supervision instead of silently
    running without an enforceable deadline.
    """

    if threading.current_thread() is not threading.main_thread() or not hasattr(
        signal, "setitimer"
    ):
        raise UnsupportedCapabilityError(
            "local adapter deadlines require a POSIX main-thread execution boundary",
            capability="operation_deadline",
        )

    def expired(signum: int, frame: object) -> None:
        del signum, frame
        raise OperationTimeoutError(f"{label} exceeded its configured deadline")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return operation()
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)
