from __future__ import annotations

"""Prometheus-compatible scenario metrics collection.

The collector deliberately keeps server-side telemetry separate from client-side
request timing. Prometheus counters and histograms are cumulative for the life of
the server, so scenario values are calculated from a scrape immediately before
and immediately after the measured load interval. Gauges are sampled throughout
the interval to retain queue and cache pressure peaks.
"""

import asyncio
import math
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

import aiohttp

try:
    from prometheus_client.parser import text_string_to_metric_families
except ModuleNotFoundError:  # Optional until server metrics are explicitly enabled.
    text_string_to_metric_families = None


SeriesKey = tuple[str, tuple[tuple[str, str], ...]]


@dataclass(frozen=True)
class PrometheusSample:
    """One labeled sample from a Prometheus exposition."""

    name: str
    labels: dict[str, str]
    value: float
    metric_type: str

    @property
    def key(self) -> SeriesKey:
        return self.name, tuple(sorted(self.labels.items()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "labels": self.labels,
            "value": self.value,
            "metric_type": self.metric_type,
        }


@dataclass
class PrometheusSnapshot:
    """All selected samples observed during one scrape."""

    phase: str
    observed_at: datetime
    monotonic_time: float
    samples: dict[SeriesKey, PrometheusSample]

    def to_dict(self, start_monotonic: float | None = None) -> dict[str, Any]:
        elapsed = None
        if start_monotonic is not None:
            elapsed = self.monotonic_time - start_monotonic
        return {
            "phase": self.phase,
            "observed_at": self.observed_at.isoformat(),
            "elapsed_seconds": elapsed,
            "samples": [
                sample.to_dict()
                for sample in sorted(
                    self.samples.values(),
                    key=lambda item: (item.name, sorted(item.labels.items())),
                )
            ],
        }


@dataclass
class PrometheusMetricsConfig:
    """Configuration for optional scenario-level server metrics."""

    url: str
    scrape_interval_seconds: float = 1.0
    timeout_seconds: float = 5.0
    api_key: str | None = None
    strict: bool = False
    metric_prefixes: tuple[str, ...] = ("vllm:", "vllm_", "infer_service_")


def parse_prometheus_text(
    text: str,
    metric_prefixes: tuple[str, ...] = (),
) -> dict[SeriesKey, PrometheusSample]:
    """Parse Prometheus text into a stable labeled-series mapping."""

    if text_string_to_metric_families is None:
        raise RuntimeError(
            "Prometheus metrics collection requires the 'prometheus-client' package. "
            "Install project dependencies with: python3.12 -m pip install -r requirements.txt"
        )

    parsed: dict[SeriesKey, PrometheusSample] = {}
    for family in text_string_to_metric_families(text):
        for raw_sample in family.samples:
            name = raw_sample.name
            if metric_prefixes and not name.startswith(metric_prefixes):
                continue
            sample = PrometheusSample(
                name=name,
                labels={str(k): str(v) for k, v in raw_sample.labels.items()},
                value=float(raw_sample.value),
                metric_type=family.type,
            )
            parsed[sample.key] = sample
    return parsed


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _series_delta(before: float, after: float) -> tuple[float, bool]:
    """Return a cumulative-series delta and whether a reset was detected."""

    if after >= before:
        return after - before, False
    return after, True


def _histogram_quantile(buckets: list[tuple[float, float]], quantile: float) -> float | None:
    """Approximate a quantile from cumulative Prometheus histogram buckets."""

    finite = sorted((bound, count) for bound, count in buckets if not math.isnan(bound))
    if not finite:
        return None
    total = next((count for bound, count in finite if math.isinf(bound)), finite[-1][1])
    if total <= 0:
        return None
    target = total * quantile
    previous_bound = 0.0
    previous_count = 0.0
    for bound, count in finite:
        if count < target:
            if not math.isinf(bound):
                previous_bound = bound
            previous_count = count
            continue
        if math.isinf(bound):
            return previous_bound
        bucket_count = count - previous_count
        if bucket_count <= 0:
            return bound
        fraction = (target - previous_count) / bucket_count
        return previous_bound + (bound - previous_bound) * fraction
    return finite[-1][0]


def _base_histogram_name(sample_name: str) -> str | None:
    for suffix in ("_bucket", "_count", "_sum"):
        if sample_name.endswith(suffix):
            return sample_name[: -len(suffix)]
    return None


CANONICAL_COUNTERS: dict[str, tuple[str, ...]] = {
    "prompt_tokens": ("vllm:prompt_tokens_total",),
    "generation_tokens": ("vllm:generation_tokens_total",),
    "successful_requests": ("vllm:request_success_total",),
    "preemptions": ("vllm:num_preemptions_total",),
    "prefix_cache_hits": ("vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits_total"),
    "prefix_cache_queries": ("vllm:prefix_cache_queries_total",),
}

CANONICAL_GAUGES: dict[str, tuple[str, ...]] = {
    "running_requests": ("vllm:num_requests_running",),
    "waiting_requests": ("vllm:num_requests_waiting",),
    "kv_cache_usage": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
}

CANONICAL_HISTOGRAMS: dict[str, tuple[str, ...]] = {
    "ttft": ("vllm:time_to_first_token_seconds", "infer_service_first_token_cost"),
    "ttfb": ("vllm:time_to_first_byte_seconds", "infer_service_first_byte_cost"),
    "tpot": (
        "vllm:request_time_per_output_token_seconds",
        "vllm:time_per_output_token_seconds",
        "vllm:inter_token_latency_seconds",
        "infer_service_per_token_cost",
    ),
    "e2e_latency": ("vllm:e2e_request_latency_seconds", "infer_service_request_cost"),
    "queue_time": ("vllm:request_queue_time_seconds",),
    "prefill_time": ("vllm:request_prefill_time_seconds",),
    "decode_time": ("vllm:request_decode_time_seconds",),
    "inference_time": ("vllm:request_inference_time_seconds",),
    "prompt_tokens_per_request": (
        "vllm:request_prompt_tokens",
        "infer_service_input_token_quantity",
    ),
    "generation_tokens_per_request": (
        "vllm:request_generation_tokens",
        "infer_service_output_token_quantity",
    ),
}

# vLLM latency histograms are already seconds. ModelArts infer_service cost
# histograms are documented in milliseconds and are normalized here so the
# canonical summary can safely compare providers.
HISTOGRAM_VALUE_SCALE: dict[str, float] = {
    "infer_service_first_token_cost": 0.001,
    "infer_service_first_byte_cost": 0.001,
    "infer_service_per_token_cost": 0.001,
    "infer_service_request_cost": 0.001,
}


def _matching_names(samples: Iterable[PrometheusSample], aliases: tuple[str, ...]) -> set[str]:
    names = {sample.name for sample in samples}
    selected = next((alias for alias in aliases if alias in names), None)
    return {selected} if selected else set()


def summarize_snapshots(
    snapshots: list[PrometheusSnapshot],
) -> tuple[dict[str, Any], list[str]]:
    """Build canonical scenario summaries from before/during/after scrapes."""

    before = next((snapshot for snapshot in snapshots if snapshot.phase == "before"), None)
    after = next(
        (snapshot for snapshot in reversed(snapshots) if snapshot.phase == "after"),
        None,
    )
    if before is None or after is None:
        return {}, [
            "Scenario deltas are unavailable because both before and after scrapes are required."
        ]
    warnings: list[str] = []
    all_samples = [sample for snap in snapshots for sample in snap.samples.values()]

    counter_summary: dict[str, Any] = {}
    for canonical_name, aliases in CANONICAL_COUNTERS.items():
        boundary_samples = [*before.samples.values(), *after.samples.values()]
        matching = _matching_names(boundary_samples, aliases)
        if not matching:
            continue
        total_delta = 0.0
        reset_series = 0
        matched_series = 0
        source_names: set[str] = set()
        for key, before_sample in before.samples.items():
            if before_sample.name not in matching or key not in after.samples:
                continue
            delta, reset = _series_delta(before_sample.value, after.samples[key].value)
            total_delta += delta
            reset_series += int(reset)
            matched_series += 1
            source_names.add(before_sample.name)
        if matched_series == 0:
            continue
        if reset_series:
            warnings.append(
                f"Detected {reset_series} reset series while calculating {canonical_name}."
            )
        counter_summary[canonical_name] = {
            "delta": total_delta,
            "source_metrics": sorted(source_names or matching),
            "counter_resets": reset_series,
        }

    gauge_summary: dict[str, Any] = {}
    for canonical_name, aliases in CANONICAL_GAUGES.items():
        matching = _matching_names(all_samples, aliases)
        if not matching:
            continue
        per_scrape: list[float] = []
        for snapshot in snapshots:
            values = [
                sample.value
                for sample in snapshot.samples.values()
                if sample.name in matching and math.isfinite(sample.value)
            ]
            if not values:
                continue
            if canonical_name in {"running_requests", "waiting_requests"}:
                per_scrape.append(sum(values))
            else:
                per_scrape.append(max(values))
        if not per_scrape:
            continue
        gauge_summary[canonical_name] = {
            "min": min(per_scrape),
            "mean": statistics.mean(per_scrape),
            "p95": _percentile(per_scrape, 95),
            "max": max(per_scrape),
            "samples": len(per_scrape),
            "source_metrics": sorted(matching),
        }

    histogram_summary: dict[str, Any] = {}
    before_histograms: dict[tuple[str, tuple[tuple[str, str], ...]], dict[str, float]] = {}
    after_histograms: dict[tuple[str, tuple[tuple[str, str], ...]], dict[str, float]] = {}

    def collect_histograms(
        snapshot: PrometheusSnapshot,
        target: dict[tuple[str, tuple[tuple[str, str], ...]], dict[str, float]],
    ) -> None:
        for sample in snapshot.samples.values():
            base = _base_histogram_name(sample.name)
            if base is None:
                continue
            labels = dict(sample.labels)
            suffix = sample.name[len(base) :]
            if suffix == "_bucket":
                bucket = labels.pop("le", "+Inf")
                field_name = f"bucket:{bucket}"
            else:
                field_name = suffix[1:]
            key = base, tuple(sorted(labels.items()))
            target.setdefault(key, {})[field_name] = sample.value

    collect_histograms(before, before_histograms)
    collect_histograms(after, after_histograms)

    for canonical_name, aliases in CANONICAL_HISTOGRAMS.items():
        available_bases = {
            base for base, _ in set(before_histograms) & set(after_histograms)
        }
        selected_base = next((alias for alias in aliases if alias in available_bases), None)
        if selected_base is None:
            continue
        matching_bases = {selected_base}
        total_count = 0.0
        total_sum = 0.0
        combined_buckets: dict[float, float] = {}
        reset_series = 0
        matched_histogram = False
        for key, before_values in before_histograms.items():
            base, _ = key
            if base not in matching_bases or key not in after_histograms:
                continue
            value_scale = HISTOGRAM_VALUE_SCALE.get(base, 1.0)
            after_values = after_histograms[key]
            matched_histogram = True
            for field_name, before_value in before_values.items():
                if field_name not in after_values:
                    continue
                delta, reset = _series_delta(before_value, after_values[field_name])
                reset_series += int(reset)
                if field_name == "count":
                    total_count += delta
                elif field_name == "sum":
                    total_sum += delta * value_scale
                elif field_name.startswith("bucket:"):
                    raw_bound = field_name.split(":", 1)[1]
                    bound = math.inf if raw_bound in {"+Inf", "Inf"} else float(raw_bound)
                    if math.isfinite(bound):
                        bound *= value_scale
                    combined_buckets[bound] = combined_buckets.get(bound, 0.0) + delta
        if not matched_histogram:
            continue
        if reset_series:
            warnings.append(
                f"Detected {reset_series} reset histogram series while calculating {canonical_name}."
            )
        buckets = sorted(combined_buckets.items(), key=lambda item: item[0])
        histogram_summary[canonical_name] = {
            "count_delta": total_count,
            "sum_delta": total_sum,
            "mean": total_sum / total_count if total_count > 0 else None,
            "p50": _histogram_quantile(buckets, 0.50),
            "p95": _histogram_quantile(buckets, 0.95),
            "p99": _histogram_quantile(buckets, 0.99),
            "buckets": [
                {"le": "+Inf" if math.isinf(bound) else bound, "count_delta": count}
                for bound, count in buckets
            ],
            "source_metrics": sorted(matching_bases),
            "unit": (
                "tokens"
                if canonical_name in {
                    "prompt_tokens_per_request",
                    "generation_tokens_per_request",
                }
                else "seconds/token"
                if canonical_name == "tpot"
                else "seconds"
            ),
            "counter_resets": reset_series,
        }

    return {
        "counters": counter_summary,
        "gauges": gauge_summary,
        "histograms": histogram_summary,
    }, warnings


@dataclass
class ScenarioMetricsCollector:
    """Scrape server metrics before, during, and after one measured scenario."""

    config: PrometheusMetricsConfig
    snapshots: list[PrometheusSnapshot] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    started_at: datetime | None = None
    ended_at: datetime | None = None
    _start_monotonic: float | None = None
    _stop_event: asyncio.Event | None = None
    _sampler_task: asyncio.Task[None] | None = None

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "text/plain; version=0.0.4, application/openmetrics-text"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    async def scrape(self, session: aiohttp.ClientSession, phase: str) -> PrometheusSnapshot | None:
        try:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            async with session.get(self.config.url, headers=self._headers(), timeout=timeout) as response:
                body = await response.text()
                if response.status != 200:
                    raise RuntimeError(f"metrics endpoint returned HTTP {response.status}: {body[:160]}")
                snapshot = PrometheusSnapshot(
                    phase=phase,
                    observed_at=datetime.now(timezone.utc),
                    monotonic_time=time.monotonic(),
                    samples=parse_prometheus_text(body, self.config.metric_prefixes),
                )
                self.snapshots.append(snapshot)
                return snapshot
        except Exception as exc:
            message = f"{phase} scrape failed: {exc}"
            self.errors.append(message)
            if self.config.strict:
                raise
            return None

    async def start(self, session: aiohttp.ClientSession) -> None:
        self.started_at = datetime.now(timezone.utc)
        self._start_monotonic = time.monotonic()
        self._stop_event = asyncio.Event()
        await self.scrape(session, "before")
        self._sampler_task = asyncio.create_task(self._sample_loop(session))

    async def _sample_loop(self, session: aiohttp.ClientSession) -> None:
        assert self._stop_event is not None
        interval = max(0.05, self.config.scrape_interval_seconds)
        while True:
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval)
                return
            except asyncio.TimeoutError:
                await self.scrape(session, "during")

    async def stop(self, session: aiohttp.ClientSession) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._sampler_task is not None:
            await self._sampler_task
        await self.scrape(session, "after")
        self.ended_at = datetime.now(timezone.utc)

    def to_dict(
        self,
        elapsed_reference_monotonic: float | None = None,
    ) -> dict[str, Any]:
        summary, summary_warnings = summarize_snapshots(self.snapshots)
        warnings = [*self.warnings, *summary_warnings]
        has_before = any(snapshot.phase == "before" for snapshot in self.snapshots)
        has_after = any(snapshot.phase == "after" for snapshot in self.snapshots)
        has_data = any(snapshot.samples for snapshot in self.snapshots)
        if has_before and has_after and not has_data:
            warnings.append("Metrics scrapes succeeded but no selected metric series were found.")
        return {
            "enabled": True,
            "url": self.config.url,
            "available": has_before and has_after,
            "has_data": has_data,
            "scrape_interval_seconds": self.config.scrape_interval_seconds,
            "scrape_count": len(self.snapshots),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "errors": self.errors,
            "warnings": warnings,
            "elapsed_reference": (
                "measured_scenario_start"
                if elapsed_reference_monotonic is not None
                else "metrics_collection_start"
            ),
            "summary": summary,
            "snapshots": [
                snapshot.to_dict(
                    elapsed_reference_monotonic
                    if elapsed_reference_monotonic is not None
                    else self._start_monotonic
                )
                for snapshot in self.snapshots
            ],
        }
