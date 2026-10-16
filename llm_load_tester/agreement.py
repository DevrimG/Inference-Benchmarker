from __future__ import annotations

"""Client/server agreement checks.

These compare what the client measured with what the server's own
Prometheus counters and histograms say happened over the same interval. They
are only meaningful when this client is the only traffic on the server while
the run is measured.
"""

import math
from typing import Any, Iterable, Sequence

from . import stats

QUANTILES = (0.5, 0.9, 0.99)


def _bucket_for_quantile(
    buckets: Sequence[dict[str, Any]], quantile: float
) -> tuple[float, float] | None:
    """(lower, upper] bounds of the bucket holding the server's quantile."""
    parsed: list[tuple[float, float]] = []
    for bucket in buckets:
        raw = bucket.get("le")
        bound = math.inf if raw in ("+Inf", "Inf") else float(raw)
        parsed.append((bound, float(bucket.get("count_delta") or 0.0)))
    parsed.sort()
    if not parsed:
        return None
    total = parsed[-1][1]
    if total <= 0:
        return None
    target = quantile * total
    lower = 0.0
    for bound, cumulative in parsed:
        if cumulative >= target:
            return lower, bound
        lower = bound
    return None


def bucket_agreement(
    client_values: Sequence[float],
    histogram: dict[str, Any] | None,
    quantiles: Iterable[float] = QUANTILES,
) -> dict[str, Any]:
    if not histogram or not histogram.get("buckets"):
        return {"available": False, "reason": "server histogram not available"}
    if not client_values:
        return {"available": False, "reason": "no client samples"}
    checks: dict[str, Any] = {}
    for q in quantiles:
        bounds = _bucket_for_quantile(histogram["buckets"], q)
        client = stats.percentile(client_values, q * 100)
        within = bounds is not None and client is not None and bounds[0] < client <= bounds[1]
        checks[f"p{q * 100:g}"] = {
            "client": client,
            "server_bucket": [bounds[0], "+Inf" if math.isinf(bounds[1]) else bounds[1]]
            if bounds else None,
            "within": within,
        }
    return {
        "available": True,
        "client_samples": len(client_values),
        "server_samples": histogram.get("count_delta"),
        "source_metrics": histogram.get("source_metrics"),
        "quantiles": checks,
        "all_within": all(check["within"] for check in checks.values()),
    }


def server_agreement(request_metrics: Sequence[Any], server_metrics: dict[str, Any] | None) -> dict[str, Any]:
    """Compare client totals and percentiles with one server's /metrics deltas."""
    if not server_metrics or not server_metrics.get("available"):
        return {"available": False, "reason": "server metrics not collected"}
    summary = server_metrics.get("summary") or {}
    counters = summary.get("counters") or {}
    histograms = summary.get("histograms") or {}
    measured = [m for m in request_metrics if not getattr(m, "is_warmup", False)]
    ok = [m for m in measured if m.error.value == "none"]

    client_tokens = sum(m.tokens_generated for m in measured)
    server_tokens = (counters.get("generation_tokens") or {}).get("delta")
    all_reported = all(m.token_count_source == "provider_usage" for m in measured)
    tokens = {
        "client_completion_tokens": client_tokens,
        "server_generation_tokens_delta": server_tokens,
        "difference": (client_tokens - server_tokens) if server_tokens is not None else None,
        "match": server_tokens is not None and int(round(server_tokens)) == client_tokens,
        "client_counts_all_server_reported": all_reported,
    }

    ttft_values = [m.ttft for m in ok if m.ttft is not None]
    # vLLM's inter_token_latency histogram has one sample per decode step,
    # which on the client is one output-bearing read: compare chunk gaps.
    gap_values = [gap for m in ok for gap in m.chunk_gaps]
    return {
        "available": True,
        "tokens": tokens,
        "ttft": bucket_agreement(ttft_values, histograms.get("ttft")),
        "itl": bucket_agreement(gap_values, histograms.get("itl")),
        "note": (
            "Valid only when this client was the server's sole traffic during the run. "
            "Client TTFT includes network and proxy time, so it may sit one bucket above."
        ),
    }
