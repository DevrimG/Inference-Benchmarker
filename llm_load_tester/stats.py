from __future__ import annotations

"""Statistics used by run summaries.

Every percentile is reported with its sample size and a 95 % confidence
interval. Small samples use a seeded percentile bootstrap; large samples use
the distribution-free order-statistic interval, which is cheap and close to
the bootstrap at that size.
"""

import math
import statistics
from typing import Iterable, Sequence

import numpy as np

DEFAULT_PERCENTILES: tuple[float, ...] = (5, 50, 90, 95, 99)
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_MAX_N = 5000
_Z95 = 1.959963984540054


def percentile(values: Sequence[float], q: float) -> float | None:
    if not len(values):
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q))


def _order_statistic_ci(sorted_values: np.ndarray, q: float) -> tuple[float, float]:
    n = len(sorted_values)
    p = q / 100.0
    half = _Z95 * math.sqrt(n * p * (1 - p))
    lo = max(0, int(math.floor(n * p - half)) - 1)
    hi = min(n - 1, int(math.ceil(n * p + half)))
    return float(sorted_values[lo]), float(sorted_values[hi])


def percentile_ci(
    values: Sequence[float],
    percentiles: Iterable[float],
    *,
    seed: int = 0,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[dict[str, list[float] | None], str]:
    """Return 95 % intervals per percentile and the method used."""
    data = np.asarray(values, dtype=float)
    qs = list(percentiles)
    n = len(data)
    if n < 2:
        return {f"p{q:g}": None for q in qs}, "insufficient_samples"
    if n > BOOTSTRAP_MAX_N:
        ordered = np.sort(data)
        return (
            {f"p{q:g}": list(_order_statistic_ci(ordered, q)) for q in qs},
            "order_statistic",
        )
    rng = np.random.default_rng(seed)
    samples = rng.choice(data, size=(resamples, n), replace=True)
    estimates = np.percentile(samples, qs, axis=1)
    intervals: dict[str, list[float] | None] = {}
    for q, row in zip(qs, estimates):
        intervals[f"p{q:g}"] = [float(np.percentile(row, 2.5)), float(np.percentile(row, 97.5))]
    return intervals, "bootstrap"


def describe(
    values: Sequence[float],
    percentiles: Iterable[float] = DEFAULT_PERCENTILES,
    *,
    seed: int = 0,
    digits: int = 6,
) -> dict:
    """Summarise a sample: n, mean, min, max, percentiles and their 95 % CIs."""
    data = [float(v) for v in values if v is not None and math.isfinite(v)]
    qs = list(percentiles)
    out: dict = {"n": len(data)}
    if not data:
        out.update({"mean": None, "min": None, "max": None})
        out.update({f"p{q:g}": None for q in qs})
        out["ci95"] = {f"p{q:g}": None for q in qs}
        out["ci_method"] = "insufficient_samples"
        return out

    def r(value: float) -> float:
        return round(value, digits)

    out["mean"] = r(statistics.fmean(data))
    out["min"] = r(min(data))
    out["max"] = r(max(data))
    for q in qs:
        out[f"p{q:g}"] = r(float(np.percentile(data, q)))
    intervals, method = percentile_ci(data, qs, seed=seed)
    out["ci95"] = {
        key: [r(bound) for bound in interval] if interval else None
        for key, interval in intervals.items()
    }
    out["ci_method"] = method
    return out


def coefficient_of_variation(values: Sequence[float | None]) -> float | None:
    """Sample standard deviation over mean; None for fewer than two values."""
    data = [float(v) for v in values if v is not None]
    if len(data) < 2:
        return None
    mean = statistics.fmean(data)
    if mean == 0:
        return None
    return statistics.stdev(data) / abs(mean)


# -- in-flight timeline -------------------------------------------------------

Interval = tuple[float, float]


def inflight_steps(intervals: Sequence[Interval]) -> list[tuple[float, int]]:
    """Step function of requests in flight: (time, count after this time)."""
    events: list[tuple[float, int]] = []
    for start, end in intervals:
        events.append((start, 1))
        events.append((end, -1))
    # Ends sort before starts at the same instant so a replacement request
    # does not show a transient C+1.
    events.sort(key=lambda item: (item[0], item[1]))
    steps: list[tuple[float, int]] = []
    level = 0
    for at, delta in events:
        level += delta
        if steps and steps[-1][0] == at:
            steps[-1] = (at, level)
        else:
            steps.append((at, level))
    return steps


def steady_state_window(
    intervals: Sequence[Interval], target: int
) -> tuple[float, float] | None:
    """Span from the first moment ``target`` requests are in flight to the
    last moment that is still true (when the drain begins)."""
    if target <= 0:
        return None
    steps = inflight_steps(intervals)
    start = next((at for at, level in steps if level >= target), None)
    if start is None:
        return None
    end = None
    for index in range(len(steps) - 1, -1, -1):
        if steps[index][1] >= target:
            end = steps[index + 1][0] if index + 1 < len(steps) else steps[index][0]
            break
    if end is None or end <= start:
        return None
    return start, end


def time_weighted_inflight(
    intervals: Sequence[Interval], window: tuple[float, float]
) -> float | None:
    start, end = window
    if end <= start:
        return None
    total = 0.0
    for a, b in intervals:
        overlap = min(b, end) - max(a, start)
        if overlap > 0:
            total += overlap
    return total / (end - start)


def littles_law(
    intervals: Sequence[Interval], window: tuple[float, float]
) -> dict:
    """Compare observed mean in-flight L with λ·W inside ``window``.

    λ is completions per second in the window and W the mean latency of those
    requests. A ratio far from 1 means the client did not hold the load it
    claims to have offered, or the window is too short to be stationary.
    """
    start, end = window
    duration = end - start
    completed = [(a, b) for a, b in intervals if start <= b <= end]
    observed = time_weighted_inflight(intervals, window)
    if duration <= 0 or not completed or observed is None:
        return {"available": False}
    arrival_rate = len(completed) / duration
    mean_latency = statistics.fmean(b - a for a, b in completed)
    predicted = arrival_rate * mean_latency
    return {
        "available": True,
        "window_seconds": round(duration, 6),
        "observed_mean_in_flight": round(observed, 4),
        "completion_rate_per_second": round(arrival_rate, 4),
        "mean_latency_seconds": round(mean_latency, 6),
        "predicted_in_flight": round(predicted, 4),
        "ratio_predicted_to_observed": round(predicted / observed, 4) if observed else None,
    }
