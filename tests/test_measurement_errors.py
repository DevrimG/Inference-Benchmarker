"""Phase 0: tests that expose measurement errors in the original client.

Each test states the correct behaviour against the deterministic mock server.
They were written before the fixes and failed on commit 1fe82fe.
"""

from __future__ import annotations

import pytest

from llm_load_tester.mock_server import MockStreamConfig

from .conftest import run_benchmark


async def test_closed_loop_rate_cap_reports_coordinated_omission(mock_server) -> None:
    """A stall must not hide the queueing delay of requests scheduled behind it.

    At 20 req/s with one worker, request 1 stalls for 1 s. Requests 2..10 were
    due every 50 ms but could only be sent after the stall. Latency measured
    from the actual send hides that; latency from the intended start must not.
    """
    mock_server.config = MockStreamConfig(
        output_tokens=2, ttft_s=0.005, token_interval_s=0.002, request_stalls={1: 1.0}
    )
    result = await run_benchmark(mock_server, concurrency=1, target_rps=20, total_requests=10)
    summary = result.get_summary()
    lag = summary["distributions"]["schedule_lag_s"]
    corrected = summary["distributions"]["latency_from_intended_s"]
    assert summary["latency_p50"] < 0.1  # what the uncorrected number shows
    assert lag["max"] > 0.8
    assert corrected["p50"] > 0.4
    assert summary["coordinated_omission"]["affected_requests"] >= 8


async def test_throughput_steady_window_excludes_ramp_and_drain(mock_server) -> None:
    cfg = MockStreamConfig(output_tokens=16, ttft_s=0.03, token_interval_s=0.01,
                           request_stalls={12: 0.6})
    mock_server.config = cfg
    result = await run_benchmark(mock_server, concurrency=4, total_requests=12)
    summary = result.get_summary()
    per_request_s = cfg.chunk_offsets()[-1]
    expected = 4 * cfg.output_tokens / per_request_s
    steady = summary["steady_state"]["output_tokens_per_second"]
    full = summary["full_window"]["output_tokens_per_second"]
    assert steady == pytest.approx(expected, rel=0.15)
    assert full < steady * 0.8  # the long single-request drain drags the full window down


async def test_tpot_excludes_first_token_and_hidden_reasoning(mock_server) -> None:
    """TPOT is decode time over tokens decoded after the first visible chunk.

    The original code divided (latency - TTFT) by every completion token,
    including 50 reasoning tokens produced before the first visible token.
    """
    mock_server.config = MockStreamConfig(
        output_tokens=20, hidden_reasoning_tokens=50, ttft_s=0.02, token_interval_s=0.01
    )
    result = await run_benchmark(mock_server)
    metric = result.request_metrics[0]
    assert metric.tokens_generated == 70
    assert metric.tpot == pytest.approx(0.01, rel=0.15)
    assert metric.decode_tokens_per_second == pytest.approx(100, rel=0.15)


async def test_token_count_without_usage_is_not_chars_div_4(mock_server) -> None:
    mock_server.config = MockStreamConfig(output_tokens=10, usage_mode="never", token_text=" a")
    result = await run_benchmark(mock_server)
    metric = result.request_metrics[0]
    assert metric.tokens_generated == 10  # chars/4 gives 5
    assert "estimate" in metric.token_count_source
    assert result.get_summary()["token_count_estimated_requests"] == 1


async def test_requests_reuse_connections_by_default(mock_server) -> None:
    mock_server.config = MockStreamConfig(output_tokens=2, ttft_s=0.005, token_interval_s=0.002)
    await run_benchmark(mock_server, concurrency=2, total_requests=10)
    assert mock_server.connection_count <= 2


async def test_cold_connection_option_opens_one_connection_per_request(mock_server) -> None:
    mock_server.config = MockStreamConfig(output_tokens=2, ttft_s=0.005, token_interval_s=0.002)
    await run_benchmark(mock_server, concurrency=2, total_requests=6, cold_connections=True)
    assert mock_server.connection_count == 6


async def test_every_sse_read_is_recorded_with_cumulative_tokens(mock_server) -> None:
    mock_server.config = MockStreamConfig(output_tokens=8, ttft_s=0.02, token_interval_s=0.02)
    result = await run_benchmark(mock_server)
    metric = result.request_metrics[0]
    token_reads = [r for r in metric.chunk_records if r.token_events]
    assert len(token_reads) == 8
    assert all(r.nbytes > 0 for r in metric.chunk_records)
    assert [r.cumulative_tokens for r in token_reads] == list(range(1, 9))
    assert metric.token_count_source == "provider_usage"
    assert metric.token_timeline_source == "continuous_usage"


async def test_stream_usage_and_continuous_stats_are_requested(mock_server) -> None:
    await run_benchmark(mock_server)
    options = mock_server.requests[0].body["stream_options"]
    assert options == {"include_usage": True, "continuous_usage_stats": True}


async def test_warmup_in_flight_is_capped_at_concurrency(mock_server) -> None:
    mock_server.config = MockStreamConfig(output_tokens=4, ttft_s=0.02, token_interval_s=0.01)
    await run_benchmark(mock_server, concurrency=2, total_requests=2, warmup_requests=8)
    assert mock_server.max_running <= 2


async def test_client_self_check_is_reported(mock_server) -> None:
    result = await run_benchmark(mock_server)
    check = result.get_summary()["client_self_check"]
    assert check["max_events_per_second"] > 1000
    assert check["event_loop_lag_ms"]["n"] > 0
    assert check["event_loop_lag_ms"]["p99"] is not None
    assert "verdict" in check


async def test_server_rejecting_continuous_usage_falls_back_to_final_usage(mock_server) -> None:
    mock_server.config = MockStreamConfig(
        output_tokens=4, ttft_s=0.005, token_interval_s=0.002, reject_continuous_usage=True
    )
    result = await run_benchmark(mock_server, total_requests=3, warmup_requests=1)
    assert all(m.error.value == "none" for m in result.request_metrics)
    assert all(m.token_count_source == "provider_usage" for m in result.request_metrics)
    assert result.get_summary()["stream_usage_effective"].startswith("final")
