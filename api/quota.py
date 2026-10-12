"""In-memory daily cap on runs that use the server's own API keys.

Enough for a portfolio demo: one process, resets at midnight (server time).
``limit=None`` means unlimited (local development). ``until`` is the last day
server-key runs are allowed; after it the quota is zero, so a deploy can stop
spending the owner's keys without anyone touching it.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from datetime import date


class DailyQuota:
    def __init__(
        self,
        limit: int | None,
        today: Callable[[], date] = date.today,
        until: date | None = None,
    ) -> None:
        self._limit = limit
        self._until = until
        self._today = today
        self._day = today()
        self._used = 0
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls) -> DailyQuota:
        raw = os.environ.get("PUBLIC_DAILY_RUNS", "").strip()
        until = os.environ.get("PUBLIC_RUNS_UNTIL", "").strip()
        return cls(int(raw) if raw else None, until=date.fromisoformat(until) if until else None)

    def _expired(self) -> bool:
        return self._until is not None and self._today() > self._until

    def _roll(self) -> None:
        today = self._today()
        if today != self._day:
            self._day, self._used = today, 0

    @property
    def remaining(self) -> int | None:
        if self._expired():
            return 0
        if self._limit is None:
            return None
        with self._lock:
            self._roll()
            return max(0, self._limit - self._used)

    def has_remaining(self) -> bool:
        remaining = self.remaining
        return remaining is None or remaining > 0

    def try_consume(self) -> bool:
        if self._expired():
            return False
        if self._limit is None:
            return True
        with self._lock:
            self._roll()
            if self._used >= self._limit:
                return False
            self._used += 1
            return True
