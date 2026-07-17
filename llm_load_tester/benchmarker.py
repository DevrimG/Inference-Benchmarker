from __future__ import annotations

"""
Core benchmark engine for asynchronous LLM load testing.
"""

import asyncio
import codecs
import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import aiohttp
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.panel import Panel

from .metrics import BenchmarkResult, ErrorCategory, RequestMetrics
from .modelarts_monitoring import ModelArtsCloudEyeCollector, ModelArtsCloudEyeConfig
from .modalities import ModalityHandler, PayloadResult
from .server_metrics import (
    PrometheusMetricsConfig,
    ScenarioMetricsCollector,
)


@dataclass
class LoadTestConfig:
    """Configuration for a load test run."""
    endpoint: str
    api_route: str
    api_key: str | None
    model: str
    concurrency: int
    target_rps: float
    total_requests: int
    warmup_requests: int
    modality_handler: ModalityHandler
    modality_config: dict[str, Any]
    timeout_seconds: float = 600.0
    metrics_url: str | None = None
    metrics_scrape_interval_seconds: float = 1.0
    metrics_timeout_seconds: float = 5.0
    metrics_api_key: str | None = None
    metrics_strict: bool = False
    modelarts_cloud_eye: ModelArtsCloudEyeConfig | None = None


class SSEDecoder:
    """Incrementally decode SSE events without losing first-byte timing."""

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._buffer = ""
        self._data_lines: list[str] = []

    def feed(self, chunk: bytes) -> list[str]:
        self._buffer += self._decoder.decode(chunk)
        return self._drain_complete_lines()

    def finalize(self) -> list[str]:
        self._buffer += self._decoder.decode(b"", final=True)
        events = self._drain_complete_lines()
        if self._buffer:
            events.extend(self._process_line(self._buffer.rstrip("\r")))
            self._buffer = ""
        if self._data_lines:
            events.append("\n".join(self._data_lines))
            self._data_lines = []
        return events

    def _drain_complete_lines(self) -> list[str]:
        events: list[str] = []
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            events.extend(self._process_line(line.rstrip("\r")))
        return events

    def _process_line(self, line: str) -> list[str]:
        if line == "":
            if not self._data_lines:
                return []
            event = "\n".join(self._data_lines)
            self._data_lines = []
            return [event]
        if line.startswith(":"):
            return []
        if line.startswith("data:"):
            self._data_lines.append(line[5:].lstrip())
        return []


@dataclass
class OutputFragments:
    """Output-bearing fields extracted from one response object."""

    text: str = ""
    reasoning: str = ""
    tool_call: str = ""
    audio: str = ""

    @property
    def has_output(self) -> bool:
        return any((self.text, self.reasoning, self.tool_call, self.audio))

    @property
    def first_kind(self) -> str | None:
        if self.reasoning:
            return "reasoning"
        if self.text:
            return "text"
        if self.tool_call:
            return "tool_call"
        if self.audio:
            return "audio"
        return None

    @property
    def token_text(self) -> str:
        return "".join((self.reasoning, self.text, self.tool_call))


class AsyncRateLimiter:
    """Token bucket rate limiter for controlling RPS."""
    
    def __init__(self, target_rps: float):
        self.target_rps = max(0.0, target_rps)
        self.unlimited = self.target_rps == 0
        self.tokens = self.target_rps
        self.last_update = time.monotonic()
        self.lock: asyncio.Lock | None = None

    def _ensure_lock(self) -> asyncio.Lock:
        """Create the rate limiter lock inside a running event loop."""
        if self.lock is None:
            self.lock = asyncio.Lock()
        return self.lock
    
    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary to maintain target RPS."""
        if self.unlimited:
            return

        async with self._ensure_lock():
            now = time.monotonic()
            elapsed = now - self.last_update
            self.tokens = min(self.target_rps, self.tokens + elapsed * self.target_rps)
            self.last_update = now
            
            if self.tokens < 1:
                # Need to wait for a token
                wait_time = (1 - self.tokens) / self.target_rps
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1


class LLMBenchmarker:
    """Main benchmark engine for LLM API load testing."""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.console = Console()
        self.result = BenchmarkResult(
            modality=config.modality_config.get("modality", "unknown"),
            model=config.model,
            endpoint=f"{config.endpoint}/{config.api_route}",
            concurrency=config.concurrency,
            target_rps=config.target_rps,
            total_requests=config.total_requests,
            warmup_requests=config.warmup_requests,
            start_time=datetime.now(),
            # Text modality config
            input_tokens=config.modality_config.get("input_tokens"),
            output_tokens=config.modality_config.get("output_tokens"),
            output_token_parameter=config.modality_config.get("output_token_parameter"),
            thinking_mode=config.modality_config.get("thinking_mode"),
            # Image modality config
            image_directory=config.modality_config.get("image_directory"),
            # Voice modality config
            audio_directory=config.modality_config.get("audio_directory"),
            audio_file=config.modality_config.get("audio_file")
        )
        self.rate_limiter = AsyncRateLimiter(config.target_rps)
        self.request_counter = 0
        self.counter_lock: asyncio.Lock | None = None
        self.semaphore: asyncio.Semaphore | None = None

    def _ensure_async_primitives(self) -> None:
        """Create asyncio synchronization primitives inside a running event loop."""
        if self.counter_lock is None:
            self.counter_lock = asyncio.Lock()
        if self.semaphore is None:
            self.semaphore = asyncio.Semaphore(self.config.concurrency)
        
    async def _get_next_request_id(self) -> int:
        """Get the next request ID atomically."""
        self._ensure_async_primitives()
        async with self.counter_lock:
            self.request_counter += 1
            return self.request_counter

    async def _claim_request_id(self) -> int | None:
        """Reserve the next benchmark request ID or return None when complete."""
        self._ensure_async_primitives()
        async with self.counter_lock:
            if self.request_counter >= self.config.total_requests:
                return None
            self.request_counter += 1
            return self.request_counter
    
    def _parse_sse_line(self, line: str) -> dict[str, Any] | None:
        """Parse a Server-Sent Events line."""
        if line.startswith(":"):
            return None

        if line.startswith("data:"):
            data = line[5:].lstrip()
            if not data or data == "[DONE]":
                return None
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return None
        return None

    def _parse_sse_data(self, data: str) -> dict[str, Any] | None:
        """Parse the data payload from a complete SSE event."""
        if not data or data == "[DONE]":
            return None
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _extract_output_fragments(self, chunk: dict[str, Any]) -> OutputFragments:
        """Extract text, reasoning, tool-call, and audio output fragments."""
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_parts: list[str] = []
        audio_parts: list[str] = []

        for choice in chunk.get("choices", []):
            output = choice.get("delta") or choice.get("message") or {}
            content = output.get("content")
            if isinstance(content, str) and content:
                text_parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        text_parts.append(part["text"])

            for key in ("reasoning_content", "reasoning"):
                reasoning = output.get(key)
                if isinstance(reasoning, str) and reasoning:
                    reasoning_parts.append(reasoning)

            tool_calls = output.get("tool_calls")
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        continue
                    function = tool_call.get("function") or {}
                    for key in ("name", "arguments"):
                        value = function.get(key)
                        if isinstance(value, str) and value:
                            tool_parts.append(value)
            function_call = output.get("function_call")
            if isinstance(function_call, dict):
                for key in ("name", "arguments"):
                    value = function_call.get(key)
                    if isinstance(value, str) and value:
                        tool_parts.append(value)

            audio = output.get("audio")
            if isinstance(audio, str) and audio:
                audio_parts.append(audio)
            elif isinstance(audio, dict):
                for key in ("data", "transcript"):
                    value = audio.get(key)
                    if isinstance(value, str) and value:
                        audio_parts.append(value)

        return OutputFragments(
            text="".join(text_parts),
            reasoning="".join(reasoning_parts),
            tool_call="".join(tool_parts),
            audio="".join(audio_parts),
        )

    def _extract_usage(self, chunk: dict[str, Any]) -> tuple[int | None, int | None]:
        """Extract provider-reported prompt and completion token counts."""
        usage = chunk.get("usage")
        if not isinstance(usage, dict):
            return None, None
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        return (
            prompt_tokens if isinstance(prompt_tokens, int) else None,
            completion_tokens if isinstance(completion_tokens, int) else None,
        )
    
    def _count_tokens_from_chunk(self, chunk: dict[str, Any]) -> int:
        """Count tokens from a streaming chunk."""
        fragments = self._extract_output_fragments(chunk)
        return max(1, len(fragments.token_text) // 4) if fragments.token_text else 0
    
    def _extract_content_from_chunk(self, chunk: dict[str, Any]) -> str:
        """Extract content text from a streaming chunk."""
        return self._extract_output_fragments(chunk).text
    
    async def _make_request(
        self,
        session: aiohttp.ClientSession,
        request_id: int | None = None,
        is_warmup: bool = False
    ) -> RequestMetrics:
        """Execute a single request and collect metrics."""
        if request_id is None:
            request_id = await self._get_next_request_id()
        url = f"{self.config.endpoint}/{self.config.api_route}"
        
        # Generate a fresh unique payload for this request
        payload_result = await self.config.modality_handler.prepare_payload(
            {
                **self.config.modality_config,
                "endpoint": self.config.endpoint,
                "api_route": self.config.api_route,
            }
        )
        
        metrics = RequestMetrics(
            request_id=request_id,
            start_time=time.monotonic(),
            input_tokens=payload_result.metadata.get("input_tokens", 0)
        )

        try:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            headers = {"Content-Type": "application/json"}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            async with session.post(
                url,
                json=payload_result.payload,
                headers=headers,
                timeout=timeout
            ) as response:
                metrics.headers_received_time = time.monotonic()
                metrics.status_code = response.status
                metrics.upstream_request_id = self._extract_upstream_request_id(response.headers)
                metrics.upstream_span_id = self._extract_upstream_span_id(response.headers)
                metrics.response_content_type = response.headers.get("Content-Type", "")

                raw_response = bytearray()
                if response.status != 200:
                    async for raw_chunk in response.content.iter_any():
                        if not raw_chunk:
                            continue
                        if metrics.first_byte_time is None:
                            metrics.first_byte_time = time.monotonic()
                        raw_response.extend(raw_chunk)
                    metrics.error = ErrorCategory.from_status_code(response.status)
                    metrics.error_message = raw_response.decode("utf-8", errors="replace")[:200]
                    metrics.end_time = time.monotonic()
                    return metrics

                decoder = SSEDecoder()
                saw_sse_event = False
                saw_output = False
                response_content_parts: list[str] = []
                reasoning_content_parts: list[str] = []
                token_text_parts: list[str] = []
                reported_prompt_tokens: int | None = None
                reported_completion_tokens: int | None = None

                def capture_provider_metadata(chunk: dict[str, Any]) -> None:
                    if metrics.provider_response_id is None and chunk.get("id") is not None:
                        metrics.provider_response_id = str(chunk["id"])
                    if metrics.provider_model is None and chunk.get("model") is not None:
                        metrics.provider_model = str(chunk["model"])
                    if metrics.provider_object is None and chunk.get("object") is not None:
                        metrics.provider_object = str(chunk["object"])
                    if metrics.provider_service_tier is None and chunk.get("service_tier") is not None:
                        metrics.provider_service_tier = str(chunk["service_tier"])
                    created = chunk.get("created")
                    if metrics.provider_created is None and isinstance(created, (int, float)):
                        metrics.provider_created = float(created)
                    return_time = chunk.get("first_token_return_time")
                    if isinstance(return_time, (int, float)):
                        value = float(return_time)
                        if metrics.provider_first_token_return_time is None:
                            metrics.provider_first_token_return_time = value
                        metrics.provider_last_token_return_time = value
                    usage = chunk.get("usage")
                    if isinstance(usage, dict):
                        metrics.provider_usage = usage
                    for choice in chunk.get("choices", []):
                        if not isinstance(choice, dict):
                            continue
                        finish_reason = choice.get("finish_reason")
                        if finish_reason:
                            metrics.finish_reason = str(finish_reason)

                def process_event(data: str, observed_time: float) -> None:
                    nonlocal saw_sse_event, saw_output
                    nonlocal reported_prompt_tokens, reported_completion_tokens
                    saw_sse_event = True
                    if metrics.first_event_time is None:
                        metrics.first_event_time = observed_time
                    chunk = self._parse_sse_data(data)
                    if chunk is None:
                        return
                    capture_provider_metadata(chunk)

                    prompt_tokens, completion_tokens = self._extract_usage(chunk)
                    if prompt_tokens is not None:
                        reported_prompt_tokens = prompt_tokens
                    if completion_tokens is not None:
                        reported_completion_tokens = completion_tokens

                    fragments = self._extract_output_fragments(chunk)
                    if not fragments.has_output:
                        return
                    saw_output = True
                    if metrics.first_output_time is None:
                        metrics.first_output_time = observed_time
                        metrics.first_output_kind = fragments.first_kind
                    if fragments.token_text and metrics.first_token_time is None:
                        metrics.first_token_time = observed_time
                    if fragments.reasoning and metrics.first_reasoning_time is None:
                        metrics.first_reasoning_time = observed_time
                    if fragments.text and metrics.first_text_time is None:
                        metrics.first_text_time = observed_time
                    if fragments.tool_call and metrics.first_tool_call_time is None:
                        metrics.first_tool_call_time = observed_time
                    if fragments.audio and metrics.first_audio_time is None:
                        metrics.first_audio_time = observed_time

                    if fragments.text:
                        response_content_parts.append(fragments.text)
                    if fragments.reasoning:
                        reasoning_content_parts.append(fragments.reasoning)
                    if fragments.token_text:
                        token_text_parts.append(fragments.token_text)

                async for raw_chunk in response.content.iter_any():
                    if not raw_chunk:
                        continue
                    observed_time = time.monotonic()
                    if metrics.first_byte_time is None:
                        metrics.first_byte_time = observed_time
                    raw_response.extend(raw_chunk)
                    for event_data in decoder.feed(raw_chunk):
                        process_event(event_data, observed_time)

                final_observed_time = time.monotonic()
                for event_data in decoder.finalize():
                    process_event(event_data, final_observed_time)

                if saw_sse_event:
                    metrics.response_mode = "streaming"
                else:
                    metrics.response_mode = "buffered"
                    try:
                        buffered_chunk = json.loads(raw_response.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        buffered_chunk = None
                    if isinstance(buffered_chunk, dict):
                        capture_provider_metadata(buffered_chunk)
                        prompt_tokens, completion_tokens = self._extract_usage(buffered_chunk)
                        reported_prompt_tokens = prompt_tokens
                        reported_completion_tokens = completion_tokens
                        fragments = self._extract_output_fragments(buffered_chunk)
                        if fragments.has_output:
                            saw_output = True
                            metrics.first_output_time = final_observed_time
                            metrics.first_output_kind = fragments.first_kind
                            if fragments.reasoning:
                                metrics.first_reasoning_time = final_observed_time
                                reasoning_content_parts.append(fragments.reasoning)
                            if fragments.text:
                                metrics.first_text_time = final_observed_time
                                response_content_parts.append(fragments.text)
                            if fragments.tool_call:
                                metrics.first_tool_call_time = final_observed_time
                            if fragments.audio:
                                metrics.first_audio_time = final_observed_time
                            if fragments.token_text:
                                token_text_parts.append(fragments.token_text)

                metrics.response_content = "".join(response_content_parts)
                metrics.reasoning_content = "".join(reasoning_content_parts)
                if reported_prompt_tokens is not None:
                    metrics.input_tokens = reported_prompt_tokens
                if reported_completion_tokens is not None:
                    metrics.tokens_generated = reported_completion_tokens
                    metrics.token_count_source = "provider_usage"
                else:
                    combined_output = "".join(token_text_parts)
                    metrics.tokens_generated = (
                        max(1, len(combined_output) // 4) if combined_output else 0
                    )
                    metrics.token_count_source = "character_estimate"

                if not saw_output:
                    metrics.error = ErrorCategory.UNKNOWN
                    metrics.error_message = (
                        "HTTP 200 response completed without identifiable text, reasoning, "
                        "tool-call, or audio output."
                    )
                metrics.end_time = time.monotonic()

        except asyncio.TimeoutError:
            metrics.error = ErrorCategory.TIMEOUT
            metrics.error_message = "Request timed out"
            metrics.end_time = time.monotonic()
        except aiohttp.ClientError as e:
            metrics.error = ErrorCategory.CONNECTION_ERROR
            metrics.error_message = str(e)[:200]
            metrics.end_time = time.monotonic()
        except Exception as e:
            metrics.error = ErrorCategory.UNKNOWN
            metrics.error_message = str(e)[:200]
            metrics.end_time = time.monotonic()
        
        return metrics

    def _extract_upstream_request_id(self, headers: Any) -> str | None:
        """Return a provider request id from common response header names."""
        for header_name in (
            "X-Request-Id",
            "X-Request-ID",
            "x-request-id",
            "X-Amzn-RequestId",
            "x-amzn-requestid",
            "X-Amz-Request-Id",
            "x-amz-request-id",
            "Trace-Id",
            "trace-id",
        ):
            value = headers.get(header_name)
            if value:
                return str(value)
        return None

    def _extract_upstream_span_id(self, headers: Any) -> str | None:
        """Return a provider span/trace correlation id from common headers."""
        for header_name in (
            "X-Span-Id",
            "x-span-id",
            "Span-Id",
            "span-id",
            "X-Trace-Id",
            "x-trace-id",
        ):
            value = headers.get(header_name)
            if value:
                return str(value)
        return None
    
    async def _execute_warmup(self, session: aiohttp.ClientSession) -> None:
        """Execute warmup requests to initialize the model."""
        self.console.print()
        self.console.print(Panel(
            f"[yellow]Warm-up Phase: Sending {self.config.warmup_requests} requests to initialize model...[/yellow]",
            title="Warm-up",
            border_style="yellow"
        ))
        
        warmup_tasks = []
        for _ in range(self.config.warmup_requests):
            # Each warmup request gets its own unique payload
            task = self._make_request(session, is_warmup=True)
            warmup_tasks.append(task)
        
        # Execute warmup requests
        completed = 0
        with Progress() as progress:
            task = progress.add_task("[yellow]Warming up...", total=self.config.warmup_requests)
            
            for coro in asyncio.as_completed(warmup_tasks):
                await coro
                completed += 1
                progress.update(task, advance=1)
        
        self.console.print("[green]✓ Warm-up complete![/green]")
        await asyncio.sleep(1)  # Brief pause after warmup
    
    async def _worker(
        self,
        session: aiohttp.ClientSession,
        progress: Progress,
        progress_task: TaskID
    ) -> None:
        """Worker coroutine that runs requests with rate limiting."""
        while True:
            request_id = await self._claim_request_id()
            if request_id is None:
                break

            # Rate limiting
            await self.rate_limiter.acquire()

            # Execute request with concurrency control (each gets unique payload)
            async with self.semaphore:
                metrics = await self._make_request(session, request_id=request_id)
                self.result.add_request(metrics)
                progress.update(progress_task, advance=1)
    
    async def run(self) -> BenchmarkResult:
        """Execute the full benchmark."""
        url = f"{self.config.endpoint}/{self.config.api_route}"
        self.console.print(Panel(
            f"[cyan]Target: {url}\n"
            f"Model: {self.config.model}\n"
            f"Concurrency: {self.config.concurrency}\n"
            f"Target RPS: {'Unlimited' if self.config.target_rps <= 0 else self.config.target_rps}\n"
            f"Total Requests: {self.config.total_requests}[/cyan]",
            title="Benchmark Configuration",
            border_style="cyan"
        ))
        
        # Create HTTP session
        connector = aiohttp.TCPConnector(
            limit=self.config.concurrency * 2,
            limit_per_host=self.config.concurrency * 2,
            enable_cleanup_closed=True,
            force_close=True
        )
        
        async with aiohttp.ClientSession(connector=connector) as session:
            # Warmup phase
            if self.config.warmup_requests > 0:
                await self._execute_warmup(session)
            
            # Reset counter for actual benchmark
            self.request_counter = 0

            server_collector: ScenarioMetricsCollector | None = None
            if self.config.metrics_url:
                server_collector = ScenarioMetricsCollector(
                    PrometheusMetricsConfig(
                        url=self.config.metrics_url,
                        scrape_interval_seconds=self.config.metrics_scrape_interval_seconds,
                        timeout_seconds=self.config.metrics_timeout_seconds,
                        api_key=self.config.metrics_api_key,
                        strict=self.config.metrics_strict,
                    )
                )
                await server_collector.start(session)

            # Only the measured request interval contributes to aggregate throughput.
            self.result.start_time = datetime.now()
            scenario_start_epoch_ms = int(time.time() * 1000)
            
            self.console.print()
            self.console.print(Panel(
                "[green]Starting benchmark...[/green]",
                title="Benchmark",
                border_style="green"
            ))
            
            # Main benchmark with progress tracking
            with Progress() as progress:
                task = progress.add_task(
                    "[green]Executing requests...",
                    total=self.config.total_requests
                )
                
                # Create worker tasks
                workers = [
                    self._worker(session, progress, task)
                    for _ in range(self.config.concurrency)
                ]
                
                try:
                    # Run all workers
                    await asyncio.gather(*workers)
                finally:
                    self.result.end_time = datetime.now()
                    scenario_end_epoch_ms = int(time.time() * 1000)
                    if server_collector is not None:
                        await server_collector.stop(session)
                        self.result.server_metrics = server_collector.to_dict()
                    if self.config.modelarts_cloud_eye is not None:
                        cloud_eye_collector = ModelArtsCloudEyeCollector(
                            self.config.modelarts_cloud_eye
                        )
                        self.result.provider_monitoring = await cloud_eye_collector.collect(
                            session,
                            scenario_start_epoch_ms,
                            scenario_end_epoch_ms,
                        )
        
        return self.result
