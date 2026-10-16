from __future__ import annotations

"""Lab orchestration: fan-out, repeats, concurrency ladder, soak and burst.

Every run goes through :func:`run_fanout`, so a single endpoint is simply a
fan-out of one. Endpoints share one :class:`StartGate`, which gives them a
common start instant and a common time origin; their request records are
merged into one :class:`BenchmarkResult` with an ``endpoint`` column.
"""

import asyncio
import json
import math
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

from . import stats
from .benchmarker import LLMBenchmarker, LoadTestConfig, StartGate
from .metrics import BenchmarkResult, ErrorCategory
from .selfcheck import build_report, peak_event_rate


@dataclass
class EndpointSpec:
    label: str
    endpoint: str
    metrics_url: str | None = None
    weight: float = 1.0
    api_key: str | None = None
    metrics_api_key: str | None = None
    tls_verify: bool = True
    tls_ca_file: str | None = None
    tls_check_hostname: bool = True


def split_concurrency(total: int, weights: Sequence[float]) -> list[int]:
    """Largest-remainder split of ``total`` by ``weights``; at least 1 each."""
    if not weights:
        return []
    if total < len(weights):
        raise ValueError(f"concurrency {total} cannot be split across {len(weights)} endpoints")
    weight_sum = float(sum(weights))
    shares = [total * w / weight_sum for w in weights]
    counts = [max(1, math.floor(share)) for share in shares]
    while sum(counts) > total:
        index = max(range(len(counts)), key=lambda i: counts[i] - shares[i])
        counts[index] -= 1
    order = sorted(range(len(shares)), key=lambda i: shares[i] - counts[i], reverse=True)
    for index in order[: total - sum(counts)]:
        counts[index] += 1
    return counts


def _split_replay(path: str, parts: int, out_dir: Path) -> list[str]:
    rows = [line for line in Path(path).read_text().splitlines()
            if line.strip() and not line.startswith('{"schedule"')]
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for part in range(parts):
        target = out_dir / f"replay_part{part}.jsonl"
        target.write_text("\n".join(rows[part::parts]) + "\n")
        paths.append(str(target))
    return paths


# -- fan-out -----------------------------------------------------------------

async def run_fanout(
    base: LoadTestConfig,
    endpoints: Sequence[EndpointSpec],
    *,
    schedule_dir: Path | None = None,
) -> BenchmarkResult:
    """Run ``base`` against every endpoint at once and merge the results."""
    if not endpoints:
        raise ValueError("at least one endpoint is required")
    weights = [e.weight for e in endpoints]
    concurrency = split_concurrency(base.concurrency, weights)
    weight_sum = float(sum(weights))
    replay_parts = (
        _split_replay(base.replay_file, len(endpoints), schedule_dir or Path("."))
        if base.arrival_mode == "replay" and base.replay_file and len(endpoints) > 1
        else None
    )
    gate = StartGate(len(endpoints))
    benchmarkers: list[LLMBenchmarker] = []
    for index, spec in enumerate(endpoints):
        share = spec.weight / weight_sum
        config = replace(
            base,
            endpoint=spec.endpoint,
            api_key=spec.api_key if spec.api_key is not None else base.api_key,
            metrics_url=spec.metrics_url,
            metrics_api_key=(
                spec.metrics_api_key if spec.metrics_api_key is not None else base.metrics_api_key
            ),
            concurrency=concurrency[index],
            total_requests=(
                0 if base.arrival_mode == "replay"
                else max(1, round(base.total_requests * share))
            ),
            warmup_requests=min(base.warmup_requests, concurrency[index]),
            target_rps=base.target_rps * share,
            arrival_rate=base.arrival_rate * share if base.arrival_rate else None,
            arrival_seed=None if base.arrival_seed is None else base.arrival_seed + index,
            replay_file=replay_parts[index] if replay_parts else base.replay_file,
            endpoint_label=spec.label,
            tls_verify=spec.tls_verify,
            tls_ca_file=spec.tls_ca_file,
            tls_check_hostname=spec.tls_check_hostname,
            start_gate=gate,
            schedule_log_path=(
                str(schedule_dir / f"schedule_{spec.label}.jsonl") if schedule_dir else None
            ),
        )
        benchmarkers.append(LLMBenchmarker(config))
    results = await asyncio.gather(*(b.run() for b in benchmarkers))
    if len(results) == 1:
        return results[0]
    return merge_results(base, endpoints, results, gate, concurrency)


def merge_results(
    base: LoadTestConfig,
    endpoints: Sequence[EndpointSpec],
    results: Sequence[BenchmarkResult],
    gate: StartGate,
    concurrency: Sequence[int],
) -> BenchmarkResult:
    first = results[0]
    merged = BenchmarkResult(
        modality=first.modality,
        model=first.model,
        endpoint="fanout:" + ",".join(e.label for e in endpoints),
        concurrency=sum(r.concurrency for r in results),
        target_rps=sum(r.target_rps for r in results),
        total_requests=sum(r.total_requests for r in results),
        warmup_requests=sum(r.warmup_requests for r in results),
        start_time=min(r.start_time for r in results),
        end_time=max((r.end_time for r in results if r.end_time), default=None),
        input_tokens=first.input_tokens,
        output_tokens=first.output_tokens,
        output_token_parameter=first.output_token_parameter,
        thinking_mode=first.thinking_mode,
        image_directory=first.image_directory,
        audio_directory=first.audio_directory,
        audio_file=first.audio_file,
    )
    merged.scenario_start_monotonic = gate.start_monotonic
    merged.arrival_mode = first.arrival_mode
    merged.slo = first.slo
    for result in results:
        for metric in result.request_metrics:
            merged.add_request(metric)
    merged.arrival_schedule = {
        "mode": first.arrival_mode,
        "per_endpoint": {r.endpoint_label: r.arrival_schedule for r in results},
    }
    merged.server_metrics = {
        "available": any((r.server_metrics or {}).get("available") for r in results),
        "per_endpoint": {r.endpoint_label: r.server_metrics for r in results if r.server_metrics},
    }
    merged.manifest = {
        "fanout": True,
        "start_epoch": gate.start_epoch,
        "concurrency_split": dict(zip((e.label for e in endpoints), concurrency)),
        "per_endpoint": {r.endpoint_label: r.manifest for r in results},
    }
    checks = [r.client_self_check for r in results if r.client_self_check]
    if checks:
        # One process drives every endpoint: judge the client on combined load.
        merged.client_self_check = build_report(
            min(c["max_events_per_second"] for c in checks),
            [],
            peak_event_rate(
                (record.t, record.events)
                for metric in merged.request_metrics
                for record in metric.chunk_records
            ),
        )
        merged.client_self_check["event_loop_lag_ms"] = {
            r.endpoint_label: r.client_self_check["event_loop_lag_ms"]
            for r in results if r.client_self_check
        }
    return merged


# -- repeats -----------------------------------------------------------------

KEY_METRICS = (
    ("steady_output_tokens_per_second", lambda s: s["steady_state"].get("output_tokens_per_second")),
    ("full_output_tokens_per_second", lambda s: s["full_window"].get("output_tokens_per_second")),
    ("ttft_p50_s", lambda s: s["distributions"]["ttft_s"]["p50"]),
    ("ttft_p95_s", lambda s: s["distributions"]["ttft_s"]["p95"]),
    ("itl_p50_s", lambda s: s["distributions"]["itl_s"]["p50"]),
    ("itl_p99_s", lambda s: s["distributions"]["itl_s"]["p99"]),
    ("decode_p5_tok_s", lambda s: s["distributions"]["decode_tokens_per_second"]["p5"]),
    ("decode_p50_tok_s", lambda s: s["distributions"]["decode_tokens_per_second"]["p50"]),
    ("max_stall_p99_s", lambda s: s["distributions"]["max_stall_s"]["p99"]),
    ("error_rate_percent", lambda s: s["error_rate_percent"]),
    ("goodput_requests_per_second",
     lambda s: (s.get("goodput") or {}).get("goodput_requests_per_second")),
)


def key_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    return {name: extract(summary) for name, extract in KEY_METRICS}


def repeat_summary(summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = [key_metrics(s) for s in summaries]
    out: dict[str, Any] = {"repeats": len(rows), "per_repeat": rows, "metrics": {}}
    for name, _ in KEY_METRICS:
        values = [row[name] for row in rows if row[name] is not None]
        out["metrics"][name] = {
            "mean": sum(values) / len(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "cv": stats.coefficient_of_variation(values),
        }
    return out


# -- concurrency ladder ------------------------------------------------------

@dataclass
class LadderRules:
    max_error_rate_percent: float = 2.0
    ttft_p95_limit_s: float | None = None
    min_throughput_gain_percent: float = 5.0
    min_p5_decode_tok_s: float = 25.0
    consecutive_levels: int = 2


def level_flags(
    level_metrics: dict[str, Any],
    previous: dict[str, Any] | None,
    rules: LadderRules,
) -> list[str]:
    flags: list[str] = []
    errors = level_metrics.get("error_rate_percent")
    if errors is not None and errors > rules.max_error_rate_percent:
        flags.append("error_rate")
    ttft = level_metrics.get("ttft_p95_s")
    if rules.ttft_p95_limit_s is not None and ttft is not None and ttft > rules.ttft_p95_limit_s:
        flags.append("ttft_p95")
    throughput = level_metrics.get("throughput_tok_s")
    if previous is not None and throughput is not None and previous.get("throughput_tok_s"):
        gain = (throughput / previous["throughput_tok_s"] - 1) * 100
        if gain < rules.min_throughput_gain_percent:
            flags.append("throughput_gain")
    decode = level_metrics.get("decode_p5_tok_s")
    if decode is not None and decode < rules.min_p5_decode_tok_s:
        flags.append("p5_decode")
    return flags


def evaluate_ladder(levels: Sequence[dict[str, Any]], rules: LadderRules) -> dict[str, Any]:
    """Apply the stop rules to the levels run so far.

    A rule stops the ladder once it has fired on ``consecutive_levels``
    consecutive levels. The knee is the last level before that streak.
    """
    streaks: dict[str, int] = {}
    evaluated = []
    for index, level in enumerate(levels):
        previous = levels[index - 1] if index else None
        flags = level_flags(level, previous, rules)
        for rule in ("error_rate", "ttft_p95", "throughput_gain", "p5_decode"):
            streaks[rule] = streaks.get(rule, 0) + 1 if rule in flags else 0
        evaluated.append({**level, "flags": flags})
        stopping = [rule for rule, n in streaks.items() if n >= rules.consecutive_levels]
        if stopping:
            knee_index = index - rules.consecutive_levels
            return {
                "levels": evaluated,
                "stopped": True,
                "stop_rules": stopping,
                "stopped_at_concurrency": level["concurrency"],
                "knee_concurrency": levels[knee_index]["concurrency"] if knee_index >= 0 else None,
            }
    return {
        "levels": evaluated,
        "stopped": False,
        "stop_rules": [],
        "stopped_at_concurrency": None,
        "knee_concurrency": levels[-1]["concurrency"] if levels else None,
    }


def ladder_level_metrics(concurrency: int, summary: dict[str, Any]) -> dict[str, Any]:
    metrics = key_metrics(summary)
    throughput = metrics["steady_output_tokens_per_second"] or metrics["full_output_tokens_per_second"]
    return {"concurrency": concurrency, "throughput_tok_s": throughput, **metrics}


async def run_ladder(
    levels: Sequence[int],
    rules: LadderRules,
    run_level: Callable[[int], Awaitable[dict[str, Any]]],
) -> dict[str, Any]:
    """Run levels in order until a stop rule fires; ``run_level`` returns a summary."""
    measured: list[dict[str, Any]] = []
    decision: dict[str, Any] = evaluate_ladder([], rules)
    for concurrency in levels:
        summary = await run_level(concurrency)
        measured.append(ladder_level_metrics(concurrency, summary))
        decision = evaluate_ladder(measured, rules)
        if decision["stopped"]:
            break
    decision["rules"] = rules.__dict__
    return decision


# -- soak --------------------------------------------------------------------

def soak_windows(result: BenchmarkResult, window_seconds: float = 300.0) -> list[dict[str, Any]]:
    """Per-window statistics for a long run (default five-minute windows)."""
    start = result.scenario_start_monotonic
    measured = [m for m in result.request_metrics if m.end_time is not None]
    if start is None or not measured:
        return []
    last = max(m.end_time for m in measured)
    windows = []
    count = max(1, math.ceil((last - start) / window_seconds))
    for index in range(count):
        lo = start + index * window_seconds
        hi = min(lo + window_seconds, last)
        done = [m for m in measured if lo <= m.end_time < hi or (index == count - 1 and m.end_time == hi)]
        ok = [m for m in done if m.error == ErrorCategory.NONE]
        tokens = sum(
            n for m in measured for at, n in result._token_arrivals(m) if lo <= at < hi
        )
        windows.append({
            "window": index,
            "start_s": round(lo - start, 3),
            "end_s": round(hi - start, 3),
            # The last window is usually a short drain after the deadline.
            "partial": (hi - lo) < window_seconds - 1e-9,
            "requests_completed": len(done),
            "error_rate_percent": round(100 * (len(done) - len(ok)) / len(done), 3) if done else None,
            "output_tokens_per_second": round(tokens / (hi - lo), 3) if hi > lo else None,
            "ttft_s": stats.describe([m.ttft for m in ok if m.ttft is not None], (50, 95)),
            "decode_tokens_per_second": stats.describe(
                [v for m in ok if (v := m.decode_tokens_per_second) is not None], (5, 50)
            ),
            "max_stall_s": max((s for m in ok if (s := m.max_stall) is not None), default=None),
        })
    return windows


# -- output ------------------------------------------------------------------

def write_run_outputs(result: BenchmarkResult, run_dir: Path, exports: Sequence[str]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    if result.manifest is not None:
        (run_dir / "manifest.json").write_text(json.dumps(result.manifest, indent=2, default=str))
    if "json" in exports:
        result.export_json(run_dir / "results.json")
    if "csv" in exports:
        result.export_csv(run_dir / "requests.csv")
    if "xlsx" in exports:
        result.export_xlsx(run_dir / "results.xlsx")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")
