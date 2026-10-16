from __future__ import annotations

"""Client self-check: can this process measure the load it is generating?

Two numbers matter. Event-loop lag is how late a timer callback runs; every
client timestamp can be late by up to that much. The sustainable event rate
is how many SSE events per second one Python thread can decode and parse; if
the run's peak observed rate approaches it, reads queue up in the socket
buffer and coalesce, which inflates TTFT and hides inter-token gaps.
"""

import asyncio
import json
import time
from collections import Counter
from typing import Any, Callable, Iterable

from .stats import describe

LAG_INTERVAL_SECONDS = 0.01
LAG_WARN_P99_MS = 5.0
EVENT_RATE_WARN_FRACTION = 0.5


def measure_event_capacity(
    parse: Callable[[bytes], Any],
    seconds: float = 0.25,
) -> float:
    """Return SSE events/s that ``parse`` sustains on a realistic chunk."""
    event = {
        "id": "chatcmpl-selfcheck",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "selfcheck",
        "choices": [{"index": 0, "delta": {"content": " token"}, "finish_reason": None}],
        "usage": {"prompt_tokens": 512, "completion_tokens": 100, "total_tokens": 612},
    }
    payload = f"data: {json.dumps(event)}\n\n".encode()
    batch = 200
    count = 0
    started = time.perf_counter()
    deadline = started + seconds
    while time.perf_counter() < deadline:
        for _ in range(batch):
            parse(payload)
        count += batch
    elapsed = time.perf_counter() - started
    return count / elapsed if elapsed > 0 else 0.0


class LagMonitor:
    """Samples event-loop lag while the run is in progress."""

    def __init__(self, interval: float = LAG_INTERVAL_SECONDS) -> None:
        self.interval = interval
        self.lags_ms: list[float] = []
        self._task: asyncio.Task[None] | None = None
        self._stop = False

    async def _loop(self) -> None:
        while not self._stop:
            expected = time.monotonic() + self.interval
            await asyncio.sleep(self.interval)
            self.lags_ms.append(max(0.0, (time.monotonic() - expected) * 1000.0))

    def start(self) -> None:
        self._stop = False
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._stop = True
        if self._task is not None:
            await self._task
            self._task = None


def peak_event_rate(read_times_and_events: Iterable[tuple[float, int]]) -> float:
    """Largest number of SSE events observed within any one-second bucket."""
    buckets: Counter[int] = Counter()
    for at, events in read_times_and_events:
        buckets[int(at)] += events
    return float(max(buckets.values())) if buckets else 0.0


def build_report(
    capacity_events_per_second: float,
    lags_ms: list[float],
    peak_events_per_second: float,
) -> dict[str, Any]:
    lag = describe(lags_ms, (50, 99), digits=3)
    warnings: list[str] = []
    if lag["p99"] is not None and lag["p99"] > LAG_WARN_P99_MS:
        warnings.append(
            f"event-loop lag p99 {lag['p99']:.1f} ms exceeds {LAG_WARN_P99_MS} ms; "
            "client timestamps may be late by that much"
        )
    utilisation = (
        peak_events_per_second / capacity_events_per_second
        if capacity_events_per_second
        else None
    )
    if utilisation is not None and utilisation > EVENT_RATE_WARN_FRACTION:
        warnings.append(
            f"peak {peak_events_per_second:.0f} events/s is {utilisation:.0%} of this "
            "client's parse capacity; split load across client processes or hosts"
        )
    return {
        "max_events_per_second": round(capacity_events_per_second, 1),
        "peak_observed_events_per_second": round(peak_events_per_second, 1),
        "peak_utilisation": round(utilisation, 4) if utilisation is not None else None,
        "event_loop_lag_ms": lag,
        "lag_sample_interval_ms": LAG_INTERVAL_SECONDS * 1000,
        "verdict": "ok" if not warnings else "client_limited",
        "warnings": warnings,
    }
