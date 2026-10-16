#!/usr/bin/env python3
"""Recompute a run's summary from its saved results.json (raw reads included).

Useful after a fix to the derivations: no requests are re-sent.

    python tools/recompute_results.py outputs/.../r1/results.json [--out recomputed.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_load_tester.metrics import (  # noqa: E402
    BenchmarkResult,
    ChunkRecord,
    ErrorCategory,
    RequestMetrics,
)

TIME_FIELDS = (
    "headers_received_time", "first_byte_time", "first_event_time", "first_output_time",
    "first_token_time", "first_reasoning_time", "first_text_time", "first_tool_call_time",
    "first_audio_time", "end_time",
)


def load_result(path: Path) -> BenchmarkResult:
    data = json.loads(path.read_text())
    summary = data["summary"]
    result = BenchmarkResult(
        modality=summary["modality"], model=summary["model"], endpoint=summary["endpoint"],
        concurrency=summary["concurrency"], target_rps=summary["target_rps"],
        total_requests=summary["total_requests"], warmup_requests=summary["warmup_requests"],
        start_time=datetime.fromisoformat(summary["start_time"]),
        end_time=datetime.fromisoformat(summary["end_time"]) if summary["end_time"] else None,
    )
    result.arrival_mode = summary.get("arrival_mode", "closed")
    result.arrival_schedule = summary.get("arrival_schedule")
    result.client_self_check = summary.get("client_self_check")
    result.slo = (summary.get("goodput") or {}).get("slo")
    result.stream_usage_effective = summary.get("stream_usage_effective")
    result.server_metrics = data.get("server_metrics")
    result.manifest = data.get("manifest")
    starts = []
    for raw in data["raw_metrics"]:
        metric = RequestMetrics(request_id=raw["request_id"], start_time=raw["start_time"])
        for name in TIME_FIELDS:
            setattr(metric, name, raw.get(name))
        for name in ("tokens_generated", "input_tokens", "token_count_source", "provider_usage",
                     "reasoning_content", "response_content", "finish_reason", "fixture_id",
                     "scheduled_time", "provider_response_id", "status_code", "response_mode"):
            if name in raw:
                setattr(metric, name, raw[name])
        metric.endpoint_label = raw.get("endpoint")
        metric.error = ErrorCategory(raw["error"])
        metric.chunk_records = [
            ChunkRecord(metric.start_time + t, nbytes, events, token_events, cumulative)
            for t, nbytes, events, token_events, cumulative in raw.get("chunk_records", [])
        ]
        result.add_request(metric)
        starts.append(metric.scheduled_time or metric.start_time)
    # The scenario start is not exported; the earliest send is within ms of it.
    result.scenario_start_monotonic = min(starts) if starts else None
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recompute a summary from saved results.json")
    parser.add_argument("results")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    result = load_result(Path(args.results))
    summary = result.get_summary()
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2, default=str))
    else:
        print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
