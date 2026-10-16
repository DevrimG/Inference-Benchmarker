from __future__ import annotations

from typing import Any

import pytest

from llm_load_tester.benchmarker import LLMBenchmarker, LoadTestConfig
from llm_load_tester.metrics import BenchmarkResult
from llm_load_tester.mock_server import MockOpenAIServer, MockStreamConfig
from llm_load_tester.modalities import ModalityHandler, PayloadResult


class StaticChatHandler(ModalityHandler):
    """Fixed streaming chat payload; the mock server decides the output shape."""

    async def prepare_payload(self, config: dict) -> PayloadResult:
        return PayloadResult(
            payload={
                "model": config["model"],
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 64,
                "stream": True,
            },
            metadata={"input_tokens": 8, "fixture_id": "static-hello"},
        )

    def get_presets(self) -> dict:
        return {}


@pytest.fixture
async def mock_server():
    """A started deterministic mock server; tests may edit ``server.config``."""
    server = MockOpenAIServer(MockStreamConfig())
    await server.start()
    try:
        yield server
    finally:
        await server.stop()


def make_config(server: MockOpenAIServer, **overrides: Any) -> LoadTestConfig:
    values: dict[str, Any] = {
        "endpoint": server.url,
        "api_route": "v1/chat/completions",
        "api_key": None,
        "model": "mock-model",
        "concurrency": 1,
        "target_rps": 0,
        "total_requests": 1,
        "warmup_requests": 0,
        "modality_handler": StaticChatHandler(),
        "modality_config": {"modality": "text", "model": "mock-model"},
        "timeout_seconds": 10.0,
    }
    values.update(overrides)
    return LoadTestConfig(**values)


async def run_benchmark(server: MockOpenAIServer, **overrides: Any) -> BenchmarkResult:
    benchmarker = LLMBenchmarker(make_config(server, **overrides))
    return await benchmarker.run()
