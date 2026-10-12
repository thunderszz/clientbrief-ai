"""Daily quota and demo replay pacing."""

from __future__ import annotations

import asyncio
from datetime import date

from api.demos import replay
from api.quota import DailyQuota


def test_unlimited_quota():
    quota = DailyQuota(None)

    assert all(quota.try_consume() for _ in range(100))
    assert quota.remaining is None


def test_quota_counts_down_and_blocks():
    quota = DailyQuota(2)

    assert quota.try_consume() is True
    assert quota.remaining == 1
    assert quota.try_consume() is True
    assert quota.try_consume() is False
    assert quota.remaining == 0


def test_quota_resets_next_day():
    today = [date(2026, 10, 12)]
    quota = DailyQuota(1, today=lambda: today[0])

    assert quota.try_consume() is True
    assert quota.try_consume() is False
    today[0] = date(2026, 10, 13)
    assert quota.try_consume() is True


def _collect(events, **kwargs):
    delays: list[float] = []

    async def fake_sleep(seconds):
        delays.append(seconds)

    async def run():
        return [e async for e in replay(events, sleep=fake_sleep, **kwargs)]

    return asyncio.run(run()), delays


def test_replay_keeps_order_and_compresses_long_runs():
    events = [{"type": "a", "t": 0.0}, {"type": "b", "t": 60.0}, {"type": "c", "t": 120.0}]

    out, delays = _collect(events, target_seconds=24)

    assert [e["type"] for e in out] == ["a", "b", "c"]
    assert sum(delays) == 24
    assert delays == [12.0, 12.0]


def test_replay_does_not_slow_down_short_runs():
    events = [{"type": "a", "t": 0.0}, {"type": "b", "t": 2.0}]

    _, delays = _collect(events, target_seconds=24)

    assert delays == [2.0]


def test_replay_speed_multiplier():
    events = [{"type": "a", "t": 0.0}, {"type": "b", "t": 10.0}]

    _, delays = _collect(events, target_seconds=24, speed=10)

    assert delays == [1.0]
