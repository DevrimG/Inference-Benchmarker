from __future__ import annotations

import time
import unittest
from datetime import datetime, timezone

from llm_load_tester.server_metrics import (
    PrometheusSnapshot,
    parse_prometheus_text,
    summarize_snapshots,
)


BEFORE = """
# HELP vllm:generation_tokens_total Generated tokens.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model_name="mock"} 100
# HELP vllm:num_requests_running Running requests.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="mock"} 0
# HELP vllm:time_to_first_token_seconds Time to first token.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{model_name="mock",le="0.1"} 4
vllm:time_to_first_token_seconds_bucket{model_name="mock",le="+Inf"} 5
vllm:time_to_first_token_seconds_sum{model_name="mock"} 0.5
vllm:time_to_first_token_seconds_count{model_name="mock"} 5
# HELP infer_service_first_byte_cost Time to first byte.
# TYPE infer_service_first_byte_cost histogram
infer_service_first_byte_cost_bucket{model_name="mock",le="200"} 3
infer_service_first_byte_cost_bucket{model_name="mock",le="+Inf"} 5
infer_service_first_byte_cost_sum{model_name="mock"} 700
infer_service_first_byte_cost_count{model_name="mock"} 5
"""

AFTER = """
# HELP vllm:generation_tokens_total Generated tokens.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model_name="mock"} 124
# HELP vllm:num_requests_running Running requests.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="mock"} 0
# HELP vllm:time_to_first_token_seconds Time to first token.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{model_name="mock",le="0.1"} 6
vllm:time_to_first_token_seconds_bucket{model_name="mock",le="+Inf"} 7
vllm:time_to_first_token_seconds_sum{model_name="mock"} 0.6
vllm:time_to_first_token_seconds_count{model_name="mock"} 7
# HELP infer_service_first_byte_cost Time to first byte.
# TYPE infer_service_first_byte_cost histogram
infer_service_first_byte_cost_bucket{model_name="mock",le="200"} 5
infer_service_first_byte_cost_bucket{model_name="mock",le="+Inf"} 7
infer_service_first_byte_cost_sum{model_name="mock"} 900
infer_service_first_byte_cost_count{model_name="mock"} 7
"""


class ServerMetricsTests(unittest.TestCase):
    def _snapshot(self, phase: str, body: str, offset: float) -> PrometheusSnapshot:
        return PrometheusSnapshot(
            phase=phase,
            observed_at=datetime.now(timezone.utc),
            monotonic_time=time.monotonic() + offset,
            samples=parse_prometheus_text(body, ("vllm:", "infer_service_")),
        )

    def test_prometheus_parser_preserves_names_labels_and_types(self) -> None:
        samples = parse_prometheus_text(BEFORE, ("vllm:",))
        generation = next(
            sample
            for sample in samples.values()
            if sample.name == "vllm:generation_tokens_total"
        )
        self.assertEqual(generation.labels, {"model_name": "mock"})
        self.assertEqual(generation.value, 100.0)
        self.assertEqual(generation.metric_type, "counter")

    def test_scenario_summary_uses_deltas_and_histogram_buckets(self) -> None:
        before = self._snapshot("before", BEFORE, 0.0)
        during_body = BEFORE.replace(
            'vllm:num_requests_running{model_name="mock"} 0',
            'vllm:num_requests_running{model_name="mock"} 2',
        )
        during = self._snapshot("during", during_body, 0.05)
        after = self._snapshot("after", AFTER, 0.1)

        summary, warnings = summarize_snapshots([before, during, after])

        self.assertEqual(warnings, [])
        self.assertEqual(summary["counters"]["generation_tokens"]["delta"], 24.0)
        self.assertEqual(summary["gauges"]["running_requests"]["max"], 2.0)
        ttft = summary["histograms"]["ttft"]
        self.assertEqual(ttft["count_delta"], 2.0)
        self.assertAlmostEqual(ttft["mean"], 0.05)
        self.assertLessEqual(ttft["p95"], 0.1)
        ttfb = summary["histograms"]["ttfb"]
        self.assertEqual(ttfb["count_delta"], 2.0)
        self.assertAlmostEqual(ttfb["mean"], 0.1)
        self.assertEqual(ttfb["unit"], "seconds")

    def test_scenario_deltas_require_before_and_after_boundaries(self) -> None:
        during = self._snapshot("during", BEFORE, 0.05)
        after = self._snapshot("after", AFTER, 0.1)

        summary, warnings = summarize_snapshots([during, after])

        self.assertEqual(summary, {})
        self.assertIn("both before and after scrapes", warnings[0])


if __name__ == "__main__":
    unittest.main()
