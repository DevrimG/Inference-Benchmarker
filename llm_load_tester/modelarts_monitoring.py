from __future__ import annotations

"""Huawei ModelArts MaaS monitoring through the Cloud Eye API.

MaaS built-in-service metrics are Cloud Eye metrics in the SYS.MaaS namespace.
They are aggregated at one-minute granularity and use IAM authentication, so
they intentionally remain separate from request timings and vLLM /metrics data.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import aiohttp


DEFAULT_MAAS_METRICS = (
    "rpm",
    "tpm",
    "req_count",
    "req_count_2xx",
    "req_count_400",
    "req_count_401",
    "req_count_403",
    "req_count_404",
    "req_count_413",
    "req_count_429",
    "req_count_4xx",
    "req_count_500",
    "req_count_503",
    "req_count_504",
    "req_count_5xx",
    "req_count_error",
    "req_error_rate",
    "req_error_4xx_rate",
    "req_error_5xx_rate",
    "prompt_tokens",
    "prompt_tokens_avg",
    "prompt_tokens_p50",
    "prompt_tokens_p80",
    "prompt_tokens_p90",
    "prompt_tokens_p99",
    "prompt_tokens_max",
    "completion_tokens",
    "completion_tokens_avg",
    "completion_tokens_p50",
    "completion_tokens_p80",
    "completion_tokens_p90",
    "completion_tokens_p99",
    "completion_tokens_max",
    "total_tokens",
    "ttft",
    "ttft_p50",
    "ttft_p80",
    "ttft_p90",
    "ttft_p99",
    "ttft_max",
    "tpot",
    "tpot_p50",
    "tpot_p80",
    "tpot_p90",
    "tpot_p99",
    "tpot_max",
    "latency_avg",
)

LATENCY_METRICS = {
    "ttft",
    "ttft_p50",
    "ttft_p80",
    "ttft_p90",
    "ttft_p99",
    "ttft_max",
    "tpot",
    "tpot_p50",
    "tpot_p80",
    "tpot_p90",
    "tpot_p99",
    "tpot_max",
    "latency_avg",
}


@dataclass
class ModelArtsCloudEyeConfig:
    """Cloud Eye query settings for one MaaS benchmark scenario."""

    endpoint: str
    project_id: str
    iam_token: str
    dimensions: tuple[tuple[str, str], ...]
    metric_names: tuple[str, ...] = DEFAULT_MAAS_METRICS
    period: str = "1"
    filter: str = "average"
    timeout_seconds: float = 15.0
    ingestion_wait_seconds: float = 0.0
    query_padding_seconds: float = 60.0
    strict: bool = False


@dataclass
class ModelArtsCloudEyeCollector:
    """Query scenario-adjacent MaaS metrics from Cloud Eye after a benchmark."""

    config: ModelArtsCloudEyeConfig
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def _url(self) -> str:
        return (
            f"{self.config.endpoint.rstrip('/')}/V1.0/"
            f"{self.config.project_id}/batch-query-metric-data"
        )

    def _request_body(self, start_epoch_ms: int, end_epoch_ms: int) -> dict[str, Any]:
        padding_ms = int(max(0.0, self.config.query_padding_seconds) * 1000)
        dimensions = [
            {"name": name, "value": value} for name, value in self.config.dimensions
        ]
        return {
            "from": max(0, start_epoch_ms - padding_ms),
            "to": end_epoch_ms + padding_ms,
            "period": self.config.period,
            "filter": self.config.filter,
            "metrics": [
                {
                    "namespace": "SYS.MaaS",
                    "metric_name": metric_name,
                    "dimensions": dimensions,
                }
                for metric_name in self.config.metric_names
            ],
        }

    def _normalize_response_metric(self, metric: dict[str, Any]) -> dict[str, Any]:
        metric_name = str(metric.get("metric_name", ""))
        source_unit = metric.get("unit")
        value_scale = 0.001 if metric_name in LATENCY_METRICS else 1.0
        normalized_points: list[dict[str, Any]] = []
        for point in metric.get("datapoints", []):
            if not isinstance(point, dict):
                continue
            normalized: dict[str, Any] = {"timestamp": point.get("timestamp")}
            for key, value in point.items():
                if key == "timestamp":
                    continue
                normalized[key] = (
                    float(value) * value_scale
                    if isinstance(value, (int, float))
                    else value
                )
            normalized_points.append(normalized)
        normalized_points.sort(key=lambda point: point.get("timestamp") or 0)
        return {
            "metric_name": metric_name,
            "dimensions": metric.get("dimensions", []),
            "source_unit": source_unit,
            "unit": "seconds" if metric_name in LATENCY_METRICS else source_unit,
            "datapoints": normalized_points,
            "latest": normalized_points[-1] if normalized_points else None,
        }

    async def collect(
        self,
        session: aiohttp.ClientSession,
        start_epoch_ms: int,
        end_epoch_ms: int,
    ) -> dict[str, Any]:
        wait_seconds = max(0.0, self.config.ingestion_wait_seconds)
        if wait_seconds:
            await asyncio.sleep(wait_seconds)

        queried_at = datetime.now(timezone.utc)
        body = self._request_body(start_epoch_ms, end_epoch_ms)
        metrics: list[dict[str, Any]] = []
        try:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            headers = {
                "X-Auth-Token": self.config.iam_token,
                "Content-Type": "application/json",
            }
            async with session.post(
                self._url(), headers=headers, json=body, timeout=timeout
            ) as response:
                response_body = await response.text()
                if response.status != 200:
                    raise RuntimeError(
                        f"Cloud Eye returned HTTP {response.status}: {response_body[:180]}"
                    )
                payload = await response.json(content_type=None)
                metrics = [
                    self._normalize_response_metric(metric)
                    for metric in payload.get("metrics", [])
                    if isinstance(metric, dict)
                ]
        except Exception as exc:
            self.errors.append(str(exc))
            if self.config.strict:
                raise

        has_data = any(metric.get("datapoints") for metric in metrics)
        self.warnings.append(
            "Cloud Eye MaaS metrics use one-minute aggregation and may include traffic "
            "outside this benchmark when dimensions are shared."
        )
        if not has_data and not self.errors:
            self.warnings.append(
                "No Cloud Eye datapoints were available yet; metric ingestion can lag the request run."
            )
        return {
            "provider": "modelarts_cloud_eye",
            "namespace": "SYS.MaaS",
            "available": has_data,
            "queried_at": queried_at.isoformat(),
            "query_window": {
                "scenario_start_epoch_ms": start_epoch_ms,
                "scenario_end_epoch_ms": end_epoch_ms,
                "padding_seconds": self.config.query_padding_seconds,
                "period": self.config.period,
                "filter": self.config.filter,
            },
            "dimensions": [
                {"name": name, "value": value}
                for name, value in self.config.dimensions
            ],
            "errors": self.errors,
            "warnings": self.warnings,
            "metrics": metrics,
        }
