from __future__ import annotations

"""Deterministic OpenAI-compatible streaming server for validating the client.

The server is the reference instrument for the benchmark tool: every chunk is
written on an absolute schedule derived from :class:`MockStreamConfig`, and
the ``/metrics`` endpoint exposes vLLM-named counters and histograms computed
from exactly what was served. Tests compare the client's measurements against
both the configured schedule and the server-side observations.

Timing uses ``time.monotonic()``, so when the server and client share a
process their timestamps are directly comparable.
"""

import asyncio
import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from aiohttp import web


# Bucket bounds copied from vLLM's default histogram definitions.
TTFT_BUCKETS = (
    0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75,
    1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0, 2560.0,
)
ITL_BUCKETS = (
    0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75,
    1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0,
)
E2E_BUCKETS = (
    0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0,
    40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0,
)


@dataclass
class MockStreamConfig:
    """Shape of every streamed response.

    Timeline for one request, measured from when the server receives it
    (after a late write, the next chunk keeps at least half its gap):

    * ``request_stalls[n]`` extra seconds before headers for the n-th request
      the server receives (1-based), imitating a queued or slow request.
    * The first visible chunk is written at
      ``ttft_s + hidden_reasoning_tokens * token_interval_s``.
    * Every later chunk follows the previous one by
      ``tokens_in_chunk * token_interval_s`` plus ``stalls.get(chunk_index)``.

    ``tokens_per_chunk > 1`` imitates multi-token prediction (MTP) or other
    speculative decoding that emits several tokens per step.
    """

    output_tokens: int = 16
    tokens_per_chunk: int = 1
    ttft_s: float = 0.03
    token_interval_s: float = 0.01
    stalls: dict[int, float] = field(default_factory=dict)
    request_stalls: dict[int, float] = field(default_factory=dict)
    hidden_reasoning_tokens: int = 0
    streamed_reasoning_tokens: int = 0
    prompt_tokens: int = 8
    cached_prompt_tokens: int = 0
    # "auto" follows the request's stream_options like vLLM; "never" omits usage.
    usage_mode: str = "auto"
    # Answer 400 to continuous_usage_stats, like servers that reject unknown fields.
    reject_continuous_usage: bool = False
    token_text: str = " a"
    model: str = "mock-model"
    server_version: str = "mock-vllm-0.0.1"

    def chunk_sizes(self) -> list[int]:
        """Return the visible tokens carried by each chunk."""
        visible = self.streamed_reasoning_tokens + self.output_tokens
        step = max(1, self.tokens_per_chunk)
        sizes = [step] * (visible // step)
        if visible % step:
            sizes.append(visible % step)
        return sizes

    def chunk_offsets(self) -> list[float]:
        """Return each chunk's write time relative to the start of streaming."""
        offsets: list[float] = []
        at = self.ttft_s + self.hidden_reasoning_tokens * self.token_interval_s
        for index, size in enumerate(self.chunk_sizes()):
            if index > 0:
                at += size * self.token_interval_s
            at += self.stalls.get(index, 0.0)
            offsets.append(at)
        return offsets


@dataclass
class _Histogram:
    bounds: tuple[float, ...]
    counts: list[int] = field(default_factory=list)
    total: float = 0.0
    n: int = 0

    def __post_init__(self) -> None:
        self.counts = [0] * len(self.bounds)

    def observe(self, value: float) -> None:
        self.n += 1
        self.total += value
        for index, bound in enumerate(self.bounds):
            if value <= bound:
                self.counts[index] += 1

    def render(self, name: str, labels: str) -> list[str]:
        lines = [f"# TYPE {name} histogram"]
        for bound, count in zip(self.bounds, self.counts):
            lines.append(f'{name}_bucket{{{labels},le="{bound}"}} {count}')
        lines.append(f'{name}_bucket{{{labels},le="+Inf"}} {self.n}')
        lines.append(f"{name}_count{{{labels}}} {self.n}")
        lines.append(f"{name}_sum{{{labels}}} {self.total}")
        return lines


@dataclass
class ServedRequest:
    """Server-side record of one served request."""

    index: int
    received_at: float
    peer: str
    first_chunk_at: float | None = None
    chunk_times: list[float] = field(default_factory=list)
    completion_tokens: int = 0
    finished_at: float | None = None
    body: dict[str, Any] = field(default_factory=dict)


class MockOpenAIServer:
    """aiohttp application serving ``/v1/chat/completions`` and ``/metrics``."""

    def __init__(self, config: MockStreamConfig | None = None) -> None:
        self.config = config or MockStreamConfig()
        self.requests: list[ServedRequest] = []
        self.peers: set[str] = set()
        self.running = 0
        self.max_running = 0
        self.generation_tokens_total = 0
        self.prompt_tokens_total = 0
        self.success_total = 0
        self.prefix_cache_queries_total = 0
        self.prefix_cache_hits_total = 0
        self.spec_drafts_total = 0
        self.spec_draft_tokens_total = 0
        self.spec_accepted_tokens_total = 0
        self.ttft = _Histogram(TTFT_BUCKETS)
        self.itl = _Histogram(ITL_BUCKETS)
        self.e2e = _Histogram(E2E_BUCKETS)
        self._runner: web.AppRunner | None = None
        self.url = ""

    # -- lifecycle ---------------------------------------------------------
    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self._chat)
        app.router.add_get("/v1/models", self._models)
        app.router.add_get("/version", self._version)
        app.router.add_get("/metrics", self._metrics)
        app.router.add_get("/_mock/records", self._records)
        return app

    async def start(self, ssl_context: Any = None) -> str:
        self._runner = web.AppRunner(self.app())
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0, ssl_context=ssl_context)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        scheme = "https" if ssl_context is not None else "http"
        self.url = f"{scheme}://127.0.0.1:{port}"
        return self.url

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def __aenter__(self) -> "MockOpenAIServer":
        await self.start()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.stop()

    @property
    def connection_count(self) -> int:
        """Distinct TCP connections (client ip:port pairs) seen so far."""
        return len(self.peers)

    # -- handlers ----------------------------------------------------------
    async def _models(self, request: web.Request) -> web.Response:
        return web.json_response(
            {"object": "list", "data": [{"id": self.config.model, "object": "model"}]}
        )

    async def _version(self, request: web.Request) -> web.Response:
        return web.json_response({"version": self.config.server_version})

    async def _chat(self, request: web.Request) -> web.StreamResponse:
        received = time.monotonic()
        peername = request.transport.get_extra_info("peername") if request.transport else None
        peer = f"{peername[0]}:{peername[1]}" if peername else "unknown"
        self.peers.add(peer)
        body = await request.json()
        if self.config.reject_continuous_usage and (
            (body.get("stream_options") or {}).get("continuous_usage_stats")
        ):
            return web.json_response(
                {"error": "Unrecognized field: stream_options.continuous_usage_stats"},
                status=400,
            )
        served = ServedRequest(
            index=len(self.requests) + 1, received_at=received, peer=peer, body=body
        )
        self.requests.append(served)
        cfg = self.config
        stream_options = body.get("stream_options") or {}
        if cfg.usage_mode == "never":
            include_usage = continuous = False
        else:
            include_usage = bool(stream_options.get("include_usage"))
            continuous = bool(stream_options.get("continuous_usage_stats"))

        self.running += 1
        self.max_running = max(self.max_running, self.running)
        try:
            stall = cfg.request_stalls.get(served.index, 0.0)
            if stall:
                await asyncio.sleep(stall)
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            stream_start = time.monotonic()

            def usage(completion: int) -> dict[str, Any]:
                payload: dict[str, Any] = {
                    "prompt_tokens": cfg.prompt_tokens,
                    "completion_tokens": completion,
                    "total_tokens": cfg.prompt_tokens + completion,
                }
                reasoning = cfg.hidden_reasoning_tokens + cfg.streamed_reasoning_tokens
                if reasoning:
                    payload["completion_tokens_details"] = {"reasoning_tokens": reasoning}
                if cfg.cached_prompt_tokens:
                    payload["prompt_tokens_details"] = {
                        "cached_tokens": cfg.cached_prompt_tokens
                    }
                return payload

            async def send(event: dict[str, Any]) -> None:
                await response.write(f"data: {json.dumps(event)}\n\n".encode())

            base = {"id": f"mock-{served.index}", "object": "chat.completion.chunk",
                    "created": int(time.time()), "model": cfg.model}
            role_event = {**base, "choices": [{"index": 0, "delta": {"role": "assistant"}}]}
            if continuous:
                role_event["usage"] = usage(0)
            await send(role_event)

            completion = cfg.hidden_reasoning_tokens
            sizes = cfg.chunk_sizes()
            offsets = cfg.chunk_offsets()
            previous_chunk_at: float | None = None
            emitted_visible = 0
            for index, (size, offset) in enumerate(zip(sizes, offsets)):
                # Absolute schedule keeps the average cadence exact. After a
                # late write, catch up but keep at least half a gap between
                # writes so two chunks never coalesce into one client read.
                due = stream_start + offset
                if previous_chunk_at is not None:
                    due = max(due, previous_chunk_at + 0.5 * (offset - offsets[index - 1]))
                delay = due - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                reasoning_part = max(
                    0, min(size, cfg.streamed_reasoning_tokens - emitted_visible)
                )
                text_part = size - reasoning_part
                delta: dict[str, Any] = {}
                if reasoning_part:
                    delta["reasoning_content"] = cfg.token_text * reasoning_part
                if text_part:
                    delta["content"] = cfg.token_text * text_part
                emitted_visible += size
                completion += size
                last = index == len(sizes) - 1
                event = {**base, "choices": [{
                    "index": 0,
                    "delta": delta,
                    "finish_reason": "length" if last else None,
                }]}
                if continuous:
                    event["usage"] = usage(completion)
                # Timestamp before handing bytes to the transport: the client
                # can read them before this coroutine resumes after the write.
                now = time.monotonic()
                await send(event)
                served.chunk_times.append(now)
                if previous_chunk_at is None:
                    served.first_chunk_at = now
                    self.ttft.observe(now - received)
                else:
                    self.itl.observe(now - previous_chunk_at)
                    if size > 1:
                        self.spec_drafts_total += 1
                        self.spec_draft_tokens_total += size - 1
                        self.spec_accepted_tokens_total += size - 1
                previous_chunk_at = now

            if include_usage:
                await send({**base, "choices": [], "usage": usage(completion)})
            await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
            served.completion_tokens = completion
            served.finished_at = time.monotonic()
            self.generation_tokens_total += completion
            self.prompt_tokens_total += cfg.prompt_tokens
            self.prefix_cache_queries_total += cfg.prompt_tokens
            self.prefix_cache_hits_total += cfg.cached_prompt_tokens
            self.success_total += 1
            self.e2e.observe(served.finished_at - received)
            return response
        finally:
            self.running -= 1

    async def _records(self, request: web.Request) -> web.Response:
        """Server-side ground truth, for a client in another process."""
        return web.json_response({
            "requests": [asdict(r) for r in self.requests],
            "connection_count": self.connection_count,
            "max_running": self.max_running,
            "generation_tokens_total": self.generation_tokens_total,
        })

    async def _metrics(self, request: web.Request) -> web.Response:
        labels = f'model_name="{self.config.model}"'
        lines = [
            "# TYPE vllm:generation_tokens_total counter",
            f"vllm:generation_tokens_total{{{labels}}} {self.generation_tokens_total}",
            "# TYPE vllm:prompt_tokens_total counter",
            f"vllm:prompt_tokens_total{{{labels}}} {self.prompt_tokens_total}",
            "# TYPE vllm:request_success_total counter",
            f'vllm:request_success_total{{{labels},finished_reason="length"}} {self.success_total}',
            "# TYPE vllm:prefix_cache_queries_total counter",
            f"vllm:prefix_cache_queries_total{{{labels}}} {self.prefix_cache_queries_total}",
            "# TYPE vllm:prefix_cache_hits_total counter",
            f"vllm:prefix_cache_hits_total{{{labels}}} {self.prefix_cache_hits_total}",
            "# TYPE vllm:spec_decode_num_drafts_total counter",
            f"vllm:spec_decode_num_drafts_total{{{labels}}} {self.spec_drafts_total}",
            "# TYPE vllm:spec_decode_num_draft_tokens_total counter",
            f"vllm:spec_decode_num_draft_tokens_total{{{labels}}} {self.spec_draft_tokens_total}",
            "# TYPE vllm:spec_decode_num_accepted_tokens_total counter",
            f"vllm:spec_decode_num_accepted_tokens_total{{{labels}}} {self.spec_accepted_tokens_total}",
            "# TYPE vllm:num_requests_running gauge",
            f"vllm:num_requests_running{{{labels}}} {self.running}",
            "# TYPE vllm:num_requests_waiting gauge",
            f"vllm:num_requests_waiting{{{labels}}} 0",
            "# TYPE vllm:kv_cache_usage_perc gauge",
            f"vllm:kv_cache_usage_perc{{{labels}}} {min(1.0, self.running / 100):.4f}",
        ]
        lines += self.ttft.render("vllm:time_to_first_token_seconds", labels)
        lines += self.itl.render("vllm:inter_token_latency_seconds", labels)
        lines += self.e2e.render("vllm:e2e_request_latency_seconds", labels)
        return web.Response(text="\n".join(lines) + "\n", content_type="text/plain")


def expected_chunk_gaps(config: MockStreamConfig) -> list[float]:
    """Configured gaps between consecutive visible chunks."""
    offsets = config.chunk_offsets()
    return [b - a for a, b in zip(offsets, offsets[1:])]


def expected_decode_tokens_per_second(config: MockStreamConfig) -> float:
    """Configured decode rate between the first and last visible chunk."""
    offsets = config.chunk_offsets()
    sizes = config.chunk_sizes()
    span = offsets[-1] - offsets[0]
    return (sum(sizes) - sizes[0]) / span if span > 0 else math.inf


def _config_from_json(text: str) -> MockStreamConfig:
    data = json.loads(text)
    for key in ("stalls", "request_stalls"):
        if key in data:
            data[key] = {int(k): float(v) for k, v in data[key].items()}
    return MockStreamConfig(**data)


class ProcessMockServer:
    """Run the mock in a separate Python process.

    This is the closest local stand-in for a real server: its sleeps and
    writes are not delayed by the client's work, and vice versa.
    ``time.monotonic()`` is system-wide on Linux and macOS, so the server's
    timestamps are comparable with the client's. Ground truth is fetched
    from ``/_mock/records``.
    """

    def __init__(self, config: MockStreamConfig | None = None) -> None:
        self.config = config or MockStreamConfig()
        self.url = ""
        self._process: Any = None

    async def __aenter__(self) -> "ProcessMockServer":
        import sys

        self._process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "llm_load_tester.mock_server",
            "--config-json", json.dumps(asdict(self.config)),
            stdout=asyncio.subprocess.PIPE,
        )
        assert self._process.stdout is not None
        line = await asyncio.wait_for(self._process.stdout.readline(), timeout=20)
        self.url = line.decode().strip()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._process is not None:
            self._process.terminate()
            await self._process.wait()

    async def records(self) -> dict[str, Any]:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.url}/_mock/records") as response:
                data = await response.json()
        data["requests"] = [ServedRequest(**r) for r in data["requests"]]
        return data


async def _serve_forever(config: MockStreamConfig) -> None:
    server = MockOpenAIServer(config)
    print(await server.start(), flush=True)
    await asyncio.Event().wait()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deterministic OpenAI streaming mock server")
    parser.add_argument("--config-json", default="{}", help="MockStreamConfig fields as JSON")
    asyncio.run(_serve_forever(_config_from_json(parser.parse_args().config_json)))
