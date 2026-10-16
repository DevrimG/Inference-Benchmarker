"""Phase 3: external proof that the client's numbers are right.

The mock server writes every chunk on an absolute schedule and shares the
client's monotonic clock, so the client must reproduce the configured
cadence, chunking and stalls to within a few milliseconds of loopback and
event-loop jitter.
"""

from __future__ import annotations

import importlib.util
import json
import statistics
from pathlib import Path

import pytest

from llm_load_tester.mock_server import (
    MockStreamConfig,
    ThreadedMockServer,
    expected_chunk_gaps,
    expected_decode_tokens_per_second,
)

from .conftest import run_benchmark

JITTER_S = 0.006  # per-gap tolerance on loopback


def _load_tool(name: str):
    path = Path(__file__).resolve().parent.parent / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def assert_gaps_match(metric, cfg: MockStreamConfig) -> None:
    expected = expected_chunk_gaps(cfg)
    assert len(metric.chunk_gaps) == len(expected)
    for got, want in zip(metric.chunk_gaps, expected):
        assert got == pytest.approx(want, abs=JITTER_S)


async def test_reproduces_per_token_cadence(mock_server) -> None:
    cfg = MockStreamConfig(output_tokens=15, ttft_s=0.04, token_interval_s=0.02)
    mock_server.config = cfg
    metric = (await run_benchmark(mock_server)).request_metrics[0]
    assert_gaps_match(metric, cfg)
    served = mock_server.requests[0]
    assert metric.ttft == pytest.approx(served.first_chunk_at - served.received_at, abs=0.005)
    assert statistics.median(metric.itl_values) == pytest.approx(0.02, abs=0.002)
    assert metric.decode_tokens_per_second == pytest.approx(
        expected_decode_tokens_per_second(cfg), rel=0.05
    )


async def test_reproduces_multi_token_chunking(mock_server) -> None:
    """MTP-style chunks: 3 tokens per chunk every 30 ms is still 100 tok/s."""
    cfg = MockStreamConfig(output_tokens=24, tokens_per_chunk=3, ttft_s=0.03, token_interval_s=0.01)
    mock_server.config = cfg
    metric = (await run_benchmark(mock_server)).request_metrics[0]
    assert_gaps_match(metric, cfg)
    assert [c for _, c in metric.token_timeline()] == list(range(3, 25, 3))
    assert statistics.median(metric.itl_values) == pytest.approx(0.01, abs=0.002)
    assert metric.tpot == pytest.approx(0.01, abs=0.001)
    assert metric.decode_tokens_per_second == pytest.approx(100, rel=0.05)


async def test_reproduces_injected_stalls(mock_server) -> None:
    cfg = MockStreamConfig(output_tokens=12, ttft_s=0.02, token_interval_s=0.01,
                           stalls={4: 0.25, 9: 0.4})
    mock_server.config = cfg
    metric = (await run_benchmark(mock_server)).request_metrics[0]
    assert_gaps_match(metric, cfg)
    assert metric.max_stall == pytest.approx(0.41, abs=JITTER_S)
    big = sorted(range(len(metric.chunk_gaps)), key=metric.chunk_gaps.__getitem__)[-2:]
    assert sorted(big) == [3, 8]  # gaps before chunk 4 and chunk 9


async def test_reproduction_holds_under_concurrency() -> None:
    cfg = MockStreamConfig(output_tokens=20, tokens_per_chunk=2, ttft_s=0.03, token_interval_s=0.01)
    with ThreadedMockServer(cfg) as server:
        result = await run_benchmark(server, concurrency=16, total_requests=48)
    served = {f"mock-{r.index}": r for r in server.requests}
    read_delays = []
    for metric in result.request_metrics:
        writes = served[metric.provider_response_id].chunk_times
        reads = [t for t, _ in metric.token_timeline()]
        assert len(reads) == len(writes)
        read_delays += [read - write for read, write in zip(reads, writes)]
    # Instrument error: client read time minus the server's write time.
    assert min(read_delays) >= 0
    assert statistics.quantiles(read_delays, n=100)[98] < 0.005
    gaps = [g for m in result.request_metrics for g in m.chunk_gaps]
    assert statistics.median(gaps) == pytest.approx(0.02, abs=0.001)
    assert result.get_summary()["client_self_check"]["verdict"] == "ok"


async def test_client_agrees_with_server_counters_and_histograms(mock_server) -> None:
    # TTFT 30 ms sits inside vLLM's (0.02, 0.04] bucket, chunk gaps of 15 ms
    # inside (0.01, 0.025], away from bucket edges.
    mock_server.config = MockStreamConfig(output_tokens=10, ttft_s=0.03, token_interval_s=0.015)
    result = await run_benchmark(
        mock_server, concurrency=4, total_requests=20, warmup_requests=4,
        metrics_url=f"{mock_server.url}/metrics", metrics_scrape_interval_seconds=0.05,
    )
    agreement = result.get_summary()["server_agreement"]
    assert agreement["tokens"]["match"], agreement["tokens"]
    assert agreement["tokens"]["client_completion_tokens"] == 200  # warmup excluded
    assert agreement["ttft"]["all_within"], agreement["ttft"]
    assert agreement["itl"]["all_within"], agreement["itl"]


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


async def test_cross_tool_report_against_server_truth(mock_server, tmp_path) -> None:
    """Feed the comparison script a vLLM-format result built from server records."""
    mock_server.config = MockStreamConfig(output_tokens=12, ttft_s=0.03, token_interval_s=0.015)
    result = await run_benchmark(mock_server, concurrency=4, total_requests=24)
    ours = tmp_path / "results.json"
    result.export_json(ours)

    served = mock_server.requests
    ttfts = [r.first_chunk_at - r.received_at for r in served]
    gaps = [b - a for r in served for a, b in zip(r.chunk_times, r.chunk_times[1:])]
    duration = max(r.finished_at for r in served) - min(r.received_at for r in served)
    truth = {
        "completed": len(served),
        "total_output_tokens": mock_server.generation_tokens_total,
        "output_throughput": mock_server.generation_tokens_total / duration,
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
