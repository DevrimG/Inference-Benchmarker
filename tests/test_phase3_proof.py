"""Phase 3: external proof that the client's numbers are right.

Timing proofs run the mock server in a separate process, as a real server
would be, so neither side's scheduling delays the other. The server records
when it handed each chunk to the transport on the system-wide monotonic
clock, which the client shares. Two claims are checked:

* instrument error: client gaps match the server's actual write gaps;
* reproduction: the configured cadence (median gap), chunking (tokens per
  read) and stall positions show up in what the client measured. Individual
  gaps are not compared with the configuration, because the OS can overshoot
  one of the mock's sleeps by 10 ms; that is the mock deviating, and the
  client is judged against what was actually written.

Typical on the development MacBook: single-stream per-gap error p95 0.25 ms,
max 1.1 ms; 16 streams read delay p99 3.5 ms. The bounds below leave room for
a loaded laptop (worst p95 seen: 2.0 ms) and are not the tool's precision.
"""

from __future__ import annotations

import importlib.util
import json
import statistics
from pathlib import Path

import pytest

from llm_load_tester.mock_server import (
    MockStreamConfig,
    ProcessMockServer,
    expected_chunk_gaps,
    expected_decode_tokens_per_second,
)

from .conftest import run_benchmark

# Client gap vs server write gap. No max bound: a single OS stall of 10-20 ms
# can hit any gap on a laptop, and the run's self-check reports such lag.
GAP_ERROR_P95_S = 0.003
MEDIAN_GAP_ERROR_S = 0.001  # median client gap vs configured gap


def _load_tool(name: str):
    path = Path(__file__).resolve().parent.parent / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


async def run_against_process(cfg: MockStreamConfig, **overrides):
    async with ProcessMockServer(cfg) as server:
        result = await run_benchmark(server, show_progress=False, **overrides)
        records = await server.records()
    served = {f"mock-{r.index}": r for r in records["requests"]}
    return result, served, records


SEQUENTIAL_REQUESTS = 5  # pooled so the p95 bound has ~70+ gaps behind it


def assert_gaps_match(result, cfg: MockStreamConfig, served) -> None:
    """Pooled over every request: client gaps vs server writes, median vs config."""
    expected = expected_chunk_gaps(cfg)
    errors = []
    client_gaps: list[float] = []
    for metric in result.request_metrics:
        writes = served[metric.provider_response_id].chunk_times
        server_gaps = [b - a for a, b in zip(writes, writes[1:])]
        assert len(metric.chunk_gaps) == len(expected) == len(server_gaps)
        errors += [abs(c - g) for c, g in zip(metric.chunk_gaps, server_gaps)]
        client_gaps += metric.chunk_gaps
    # Cadence: pooled median. Stall placement is checked by its own test.
    assert statistics.median(client_gaps) == pytest.approx(
        statistics.median(expected), abs=MEDIAN_GAP_ERROR_S
    )
    errors.sort()
    assert errors[int(0.95 * (len(errors) - 1))] <= GAP_ERROR_P95_S, errors


async def test_reproduces_per_token_cadence() -> None:
    cfg = MockStreamConfig(output_tokens=15, ttft_s=0.04, token_interval_s=0.02)
    result, served, _ = await run_against_process(cfg, total_requests=SEQUENTIAL_REQUESTS)
    assert_gaps_match(result, cfg, served)
    metric = result.request_metrics[0]
    truth = served[metric.provider_response_id]
    # TTFT: client send-to-read vs server receive-to-write, same clock.
    assert metric.ttft == pytest.approx(truth.first_chunk_at - truth.received_at, abs=0.003)
    assert statistics.median(metric.itl_values) == pytest.approx(0.02, abs=0.001)
    assert statistics.median(
        m.decode_tokens_per_second for m in result.request_metrics
    ) == pytest.approx(expected_decode_tokens_per_second(cfg), rel=0.03)


async def test_reproduces_multi_token_chunking() -> None:
    """MTP-style chunks: 3 tokens per chunk every 30 ms is still 100 tok/s."""
    cfg = MockStreamConfig(output_tokens=24, tokens_per_chunk=3, ttft_s=0.03, token_interval_s=0.01)
    result, served, _ = await run_against_process(cfg, total_requests=SEQUENTIAL_REQUESTS)
    assert_gaps_match(result, cfg, served)
    metric = result.request_metrics[0]
    assert [c for _, c in metric.token_timeline()] == list(range(3, 25, 3))
    assert statistics.median(metric.itl_values) == pytest.approx(0.01, abs=0.001)
    assert statistics.median(m.tpot for m in result.request_metrics) == pytest.approx(
        0.01, abs=0.0005
    )
    assert statistics.median(
        m.decode_tokens_per_second for m in result.request_metrics
    ) == pytest.approx(100, rel=0.03)


async def test_reproduces_injected_stalls() -> None:
    cfg = MockStreamConfig(output_tokens=12, ttft_s=0.02, token_interval_s=0.01,
                           stalls={4: 0.25, 9: 0.4})
    result, served, _ = await run_against_process(cfg, total_requests=SEQUENTIAL_REQUESTS)
    assert_gaps_match(result, cfg, served)
    metric = result.request_metrics[0]
    assert statistics.median(m.max_stall for m in result.request_metrics) == pytest.approx(
        0.41, abs=MEDIAN_GAP_ERROR_S
    )
    big = sorted(range(len(metric.chunk_gaps)), key=metric.chunk_gaps.__getitem__)[-2:]
    assert sorted(big) == [3, 8]  # gaps before chunk 4 and chunk 9


async def test_reproduction_holds_under_concurrency() -> None:
    cfg = MockStreamConfig(output_tokens=20, tokens_per_chunk=2, ttft_s=0.03, token_interval_s=0.01)
    result, served, records = await run_against_process(cfg, concurrency=16, total_requests=48)
    assert records["max_running"] == 16
    read_delays = []
    for metric in result.request_metrics:
        writes = served[metric.provider_response_id].chunk_times
        reads = [t for t, _ in metric.token_timeline()]
        assert len(reads) == len(writes)  # no coalesced chunks
        read_delays += [read - write for read, write in zip(reads, writes)]
    # Instrument error: client read time minus the server's write time.
    assert min(read_delays) >= 0
    assert statistics.median(read_delays) < 0.001
    gaps = [g for m in result.request_metrics for g in m.chunk_gaps]
    assert statistics.median(gaps) == pytest.approx(0.02, abs=0.001)
    # Either the tail is accurate or the run flags itself as client-limited.
    check = result.get_summary()["client_self_check"]
    p99_delay = statistics.quantiles(read_delays, n=100)[98]
    assert check["peak_utilisation"] < 0.5
    if check["verdict"] == "ok":
        assert p99_delay < 0.010, (p99_delay, check)
    else:
        assert check["event_loop_lag_ms"]["p99"] > 5.0


async def test_client_agrees_with_server_counters_and_histograms() -> None:
    # TTFT 30 ms sits inside vLLM's (0.02, 0.04] bucket, chunk gaps of 15 ms
    # inside (0.01, 0.025], away from bucket edges.
    cfg = MockStreamConfig(output_tokens=10, ttft_s=0.03, token_interval_s=0.015)
    async with ProcessMockServer(cfg) as server:
        result = await run_benchmark(
            server, concurrency=4, total_requests=20, warmup_requests=4, show_progress=False,
            metrics_url=f"{server.url}/metrics", metrics_scrape_interval_seconds=0.05,
        )
    agreement = result.get_summary()["server_agreement"]
    assert agreement["tokens"]["match"], agreement["tokens"]
    assert agreement["tokens"]["client_completion_tokens"] == 200  # warmup excluded
    assert agreement["ttft"]["all_within"], agreement["ttft"]
    assert agreement["itl"]["all_within"], agreement["itl"]
    # 20 TTFT samples support p50 only; 180 gaps support p50 and p90.
    assert agreement["ttft"]["compared_quantiles"] == ["p50"]
    assert agreement["itl"]["compared_quantiles"] == ["p50", "p90"]


def test_bucket_agreement_needs_tail_samples() -> None:
    from llm_load_tester.agreement import bucket_agreement

    histogram = {"count_delta": 600, "buckets": [
        {"le": 0.02, "count_delta": 0}, {"le": 0.04, "count_delta": 590},
        {"le": 0.06, "count_delta": 600}, {"le": "+Inf", "count_delta": 600},
    ]}
    small = bucket_agreement([0.03] * 20, histogram)
    assert small["quantiles"]["p99"]["within"] is None
    large = bucket_agreement([0.03] * 590 + [0.05] * 10, histogram)
    assert large["compared_quantiles"] == ["p50", "p90", "p99"]
    assert large["quantiles"]["p99"]["within"] and large["all_within"]


async def test_agreement_flags_an_estimated_token_count(mock_server) -> None:
    """Without usage, 2-token chunks are counted as 1 token each: the check must catch it."""
    mock_server.config = MockStreamConfig(
        output_tokens=10, tokens_per_chunk=2, ttft_s=0.02, token_interval_s=0.01, usage_mode="never"
    )
    result = await run_benchmark(
        mock_server, total_requests=2, metrics_url=f"{mock_server.url}/metrics",
    )
    tokens = result.get_summary()["server_agreement"]["tokens"]
    assert not tokens["match"]
    assert tokens["client_completion_tokens"] == 10 and tokens["server_generation_tokens_delta"] == 20
    assert not tokens["client_counts_all_server_reported"]


async def test_littles_law_holds_in_closed_and_open_loop(mock_server) -> None:
    mock_server.config = MockStreamConfig(output_tokens=8, ttft_s=0.02, token_interval_s=0.01)
    closed = (await run_benchmark(mock_server, concurrency=4, total_requests=40)).get_summary()
    assert closed["littles_law"]["observed_mean_in_flight"] == pytest.approx(4, rel=0.1)
    assert closed["littles_law"]["ratio_predicted_to_observed"] == pytest.approx(1, rel=0.1)
    opened = (await run_benchmark(
        mock_server, arrival_mode="poisson", arrival_rate=40, arrival_seed=5, total_requests=80
    )).get_summary()
    assert opened["littles_law"]["ratio_predicted_to_observed"] == pytest.approx(1, rel=0.25)


async def test_cross_tool_report_against_server_truth(tmp_path) -> None:
    """Feed the comparison script a vLLM-format result built from server records."""
    cfg = MockStreamConfig(output_tokens=12, ttft_s=0.03, token_interval_s=0.015)
    result, served_by_id, records = await run_against_process(cfg, concurrency=4, total_requests=24)
    ours = tmp_path / "results.json"
    result.export_json(ours)

    served = records["requests"]
    ttfts = [r.first_chunk_at - r.received_at for r in served]
    gaps = [b - a for r in served for a, b in zip(r.chunk_times, r.chunk_times[1:])]
    duration = max(r.finished_at for r in served) - min(r.received_at for r in served)
    truth = {
        "completed": len(served),
        "total_output_tokens": records["generation_tokens_total"],
        "output_throughput": records["generation_tokens_total"] / duration,
        "request_throughput": len(served) / duration,
        "mean_ttft_ms": statistics.fmean(ttfts) * 1000,
        "median_ttft_ms": statistics.median(ttfts) * 1000,
        "median_itl_ms": statistics.median(gaps) * 1000,
    }
    theirs = tmp_path / "vllm.json"
    theirs.write_text(json.dumps(truth))
    tool = _load_tool("cross_tool_agreement")
    out = tmp_path / "report.json"
    code = tool.main(["--ours", str(ours), "--vllm-result", str(theirs), "--out", str(out)])
    rows = {r["metric"]: r for r in json.loads(out.read_text())["comparisons"]["vllm"]}
    assert code == 0, rows
    assert rows["total_output_tokens"]["diff_percent"] == 0
    assert rows["ttft_p50_s"]["within"] and rows["chunk_itl_p50_s"]["within"]


def test_cross_tool_dry_run_builds_matching_vllm_command(tmp_path) -> None:
    tool = _load_tool("cross_tool_agreement")
    fixtures = tmp_path / "fx.jsonl"
    fixtures.write_text(
        '{"fixture_id": "a", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 64}\n'
        '{"fixture_id": "b", "prompt": "yo", "max_tokens": 128}\n'
    )
    ours = tmp_path / "ours.json"
    ours.write_text(json.dumps({"summary": {}}))
    out = tmp_path / "report.json"
    code = tool.main([
        "--ours", str(ours), "--base-url", "http://h:8000", "--model", "m",
        "--fixtures", str(fixtures), "--concurrency", "8", "--dry-run", "--out", str(out),
    ])
    command = json.loads(out.read_text())["vllm_command"]
    assert code == 0
    assert command[:3] == ["vllm", "bench", "serve"]
    assert command[command.index("--max-concurrency") + 1] == "8"
    assert command[command.index("--custom-output-len") + 1] == "128"
    dataset = Path(command[command.index("--dataset-path") + 1])
    assert [json.loads(line)["prompt"] for line in dataset.read_text().splitlines()] == ["hi", "yo"]
