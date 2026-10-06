"""Rolling-window circuit breaker.

CLOSED     normal operation; outcomes recorded in a rolling window.
OPEN       failure rate crossed the threshold; calls are rejected instantly so
           the router can go straight to a fallback instead of waiting out a
           timeout on a provider that is known to be down.
HALF_OPEN  after ``open_timeout`` seconds one probe call is let through.
           Success closes the breaker; failure re-opens it.
"""

from __future__ import annotations

import time
from collections import deque
from enum import Enum
from typing import Any, Callable


class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Raised (or reported) when a call is rejected by an open breaker."""


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        *,
        failure_threshold: float = 0.5,
        window_size: int = 20,
        min_calls: int = 5,
        open_timeout: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        if not 0 < failure_threshold <= 1:
            raise ValueError("failure_threshold must be in (0, 1]")
        self.name = name
        self.failure_threshold = failure_threshold
        self.min_calls = min_calls
        self.open_timeout = open_timeout
        self._clock = clock
        self._window: deque[bool] = deque(maxlen=window_size)  # True = failure
        self._state = BreakerState.CLOSED
        self._opened_at = 0.0
        self._probe_in_flight = False

    @property
    def state(self) -> BreakerState:
        if self._state is BreakerState.OPEN and self._clock() - self._opened_at >= self.open_timeout:
            self._state = BreakerState.HALF_OPEN
            self._probe_in_flight = False
        return self._state

    def allow(self) -> bool:
        """Whether a call may proceed right now. Claims the probe slot in HALF_OPEN."""
        state = self.state
        if state is BreakerState.CLOSED:
            return True
        if state is BreakerState.HALF_OPEN and not self._probe_in_flight:
            self._probe_in_flight = True
            return True
        return False

    def record_success(self) -> None:
        if self.state is BreakerState.HALF_OPEN:
            self._close()
            return
        self._window.append(False)

    def record_failure(self) -> None:
        if self.state is BreakerState.HALF_OPEN:
            self._open()
            return
        self._window.append(True)
        if len(self._window) >= self.min_calls and self.failure_rate >= self.failure_threshold:
            self._open()

    @property
    def failure_rate(self) -> float:
        return sum(self._window) / len(self._window) if self._window else 0.0

    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_rate": round(self.failure_rate, 3),
            "calls_in_window": len(self._window),
        }

    def _open(self) -> None:
        self._state = BreakerState.OPEN
        self._opened_at = self._clock()
        self._probe_in_flight = False

    def _close(self) -> None:
        self._state = BreakerState.CLOSED
        self._window.clear()
        self._probe_in_flight = False
