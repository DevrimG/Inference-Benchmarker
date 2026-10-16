from __future__ import annotations

"""
Core benchmark engine for asynchronous LLM load testing.
"""

import asyncio
import codecs
import json
import time
from dataclasses import dataclass, fields
from datetime import datetime
from typing import Any

import aiohttp
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.panel import Panel

from .arrival import ArrivalSchedule, build_schedule
from .fixtures import FixtureHandler, payload_fingerprint
from .manifest import build_manifest, probe_server
from .metrics import BenchmarkResult, ChunkRecord, ErrorCategory, RequestMetrics
from .selfcheck import LagMonitor, build_report, measure_event_capacity, peak_event_rate
from .modelarts_monitoring import ModelArtsCloudEyeCollector, ModelArtsCloudEyeConfig
from .modalities import ModalityHandler, PayloadResult
from .server_metrics import (
    PrometheusMetricsConfig,
    ScenarioMetricsCollector,
)


_REDACTED_HEADER_VALUE = "[REDACTED]"
_SENSITIVE_RESPONSE_HEADER_NAMES = frozenset({
    "authorization",
    "proxy-authorization",
    "cookie",
    "cookie2",
    "set-cookie",
    "set-cookie2",
    "api-key",
    "apikey",
    "x-api-key",
    "x-apikey",
    "api-token",
    "x-api-token",
    "access-token",
    "x-access-token",
    "refresh-token",
    "x-refresh-token",
    "auth-token",
    "x-auth-token",
    "client-key",
    "x-client-key",
    "client-secret",
    "x-client-secret",
})


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
    # Arrival process; see arrival.py. "closed" keeps the original behaviour.
    arrival_mode: str = "closed"
    arrival_rate: float | None = None
    arrival_shape: float = 1.0
    arrival_seed: int | None = None
    replay_file: str | None = None
    # Run for this long instead of a fixed request count (soak mode).
    duration_seconds: float | None = None
    # Open a new TCP/TLS connection for every request.
    cold_connections: bool = False
    # "continuous": include_usage + continuous_usage_stats; "final": usage only
    # at the end; "off": leave the payload's stream_options untouched.
    stream_usage: str = "continuous"
    slo: dict[str, Any] | None = None
    endpoint_label: str | None = None
    tokenizer: str | None = None
    image_digest: str | None = None
    schedule_log_path: str | None = None
    self_check: bool = True
    show_progress: bool = True
    # Shared StartGate for multi-endpoint fan-out.
    start_gate: Any = None

    def snapshot(self) -> dict[str, Any]:
        """Config values safe to persist: no credentials, no live objects."""
        skip = {"api_key", "metrics_api_key", "modality_handler", "start_gate",
                "modelarts_cloud_eye"}
        out: dict[str, Any] = {}
        for item in fields(self):
            if item.name in skip:
                continue
            value = getattr(self, item.name)
            out[item.name] = json.loads(json.dumps(value, default=str))
        out["modality_handler"] = type(self.modality_handler).__name__
        out["api_key_set"] = bool(self.api_key)
        return out


class StartGate:
    """Releases every participant at one common monotonic instant.

    Each endpoint's benchmarker finishes its warmup and its "before" metrics
    scrape, then waits here. When the last one arrives the gate picks a start
    time slightly in the future so all participants begin together and share
    one time origin.
    """

    def __init__(self, parties: int, lead_seconds: float = 0.05) -> None:
        self.parties = parties
        self.lead_seconds = lead_seconds
        self.arrived = 0
        self.start_monotonic: float | None = None
        self.start_epoch: float | None = None
        self._event: asyncio.Event | None = None

    async def wait(self) -> float:
        if self._event is None:
            self._event = asyncio.Event()
        self.arrived += 1
        if self.arrived >= self.parties:
            self.start_monotonic = time.monotonic() + self.lead_seconds
            self.start_epoch = time.time() + self.lead_seconds
            self._event.set()
        await self._event.wait()
        assert self.start_monotonic is not None
        delay = self.start_monotonic - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        return self.start_monotonic


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
                # The wait has already paid for this token. Do not credit its
                # elapsed time again on the next acquisition.
                self.last_update = time.monotonic()
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
        self.result.slo = config.slo
        self.result.endpoint_label = config.endpoint_label
        self.rate_limiter = AsyncRateLimiter(config.target_rps)
        self.request_counter = 0
        self.counter_lock: asyncio.Lock | None = None
        self.schedule: ArrivalSchedule | None = None
        self._tokenizer: Any = None
        self._tokenizer_failed = False
        self.stream_usage = config.stream_usage
        self.result.stream_usage_effective = config.stream_usage
        if config.stream_usage not in {"continuous", "final", "off"}:
            raise ValueError("stream_usage must be 'continuous', 'final' or 'off'")

    def _ensure_async_primitives(self) -> None:
        """Create asyncio synchronization primitives inside a running event loop."""
        if self.counter_lock is None:
            self.counter_lock = asyncio.Lock()
        
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
            if (
                not self.config.duration_seconds
                and self.request_counter >= self.config.total_requests
            ):
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
        is_warmup: bool = False,
        scheduled_time: float | None = None,
        fixture_id: str | None = None,
    ) -> RequestMetrics:
        """Execute a single request and collect metrics."""
        if request_id is None:
            request_id = await self._get_next_request_id()
        url = f"{self.config.endpoint}/{self.config.api_route}"
        
        # Generate a fresh unique payload for this request
        handler_config = {
            **self.config.modality_config,
            "endpoint": self.config.endpoint,
            "api_route": self.config.api_route,
        }
        if fixture_id is not None:
            handler_config["fixture_id"] = fixture_id
        payload_result = await self.config.modality_handler.prepare_payload(handler_config)
        payload = self._apply_stream_usage(payload_result.payload)
        
        metrics = RequestMetrics(
            request_id=request_id,
            start_time=time.monotonic(),
            input_tokens=payload_result.metadata.get("input_tokens", 0),
            fixture_id=payload_result.metadata.get("fixture_id") or payload_fingerprint(payload),
            endpoint_label=self.config.endpoint_label,
            scheduled_time=scheduled_time,
            is_warmup=is_warmup,
        )

        try:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            headers = {"Content-Type": "application/json"}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=timeout
            ) as response:
                metrics.headers_received_time = time.monotonic()
                metrics.status_code = response.status
                metrics.upstream_request_id = self._extract_upstream_request_id(response.headers)
                metrics.upstream_span_id = self._extract_upstream_span_id(response.headers)
                metrics.response_content_type = response.headers.get("Content-Type", "")
                (
                    metrics.response_headers,
                    metrics.response_header_redactions,
                ) = self._capture_response_headers(response)

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
                    if (
                        response.status == 400
                        and self.stream_usage == "continuous"
                        and "continuous_usage_stats" in raw_response.decode("utf-8", "replace")
                    ):
                        # The server rejects running usage; keep final usage only.
                        self.stream_usage = "final"
                        self.result.stream_usage_effective = "final (continuous rejected by server)"
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

                def process_event(data: str, observed_time: float) -> tuple[bool, int | None]:
                    """Handle one SSE event; return (token-bearing, usage completion tokens)."""
                    nonlocal saw_sse_event, saw_output
                    nonlocal reported_prompt_tokens, reported_completion_tokens
                    saw_sse_event = True
                    if metrics.first_event_time is None:
                        metrics.first_event_time = observed_time
                    chunk = self._parse_sse_data(data)
                    if chunk is None:
                        return False, None
                    capture_provider_metadata(chunk)

                    prompt_tokens, completion_tokens = self._extract_usage(chunk)
                    if prompt_tokens is not None:
                        reported_prompt_tokens = prompt_tokens
                    if completion_tokens is not None:
                        reported_completion_tokens = completion_tokens

                    fragments = self._extract_output_fragments(chunk)
                    if not fragments.has_output:
                        return False, completion_tokens
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
                    return bool(fragments.token_text), completion_tokens

                def record_read(events: list[str], observed_time: float, nbytes: int) -> None:
                    token_events = 0
                    cumulative: int | None = None
                    for event_data in events:
                        token_bearing, usage_tokens = process_event(event_data, observed_time)
                        token_events += int(token_bearing)
                        if usage_tokens is not None:
                            cumulative = usage_tokens
                    metrics.chunk_records.append(
                        ChunkRecord(observed_time, nbytes, len(events), token_events, cumulative)
                    )

                async for raw_chunk in response.content.iter_any():
                    if not raw_chunk:
                        continue
                    observed_time = time.monotonic()
                    if metrics.first_byte_time is None:
                        metrics.first_byte_time = observed_time
                    raw_response.extend(raw_chunk)
                    record_read(decoder.feed(raw_chunk), observed_time, len(raw_chunk))

                final_observed_time = time.monotonic()
                final_events = decoder.finalize()
                if final_events:
                    record_read(final_events, final_observed_time, 0)

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
                    tokenizer_count = self._tokenizer_count(combined_output)
                    timeline = metrics.token_timeline()
                    if tokenizer_count is not None:
                        metrics.tokens_generated = tokenizer_count
                        metrics.token_count_source = "tokenizer_estimate"
                    elif timeline:
                        # One output-bearing SSE event per token; undercounts
                        # multi-token (MTP/speculative) chunks.
                        metrics.tokens_generated = timeline[-1][1]
                        metrics.token_count_source = "sse_event_count_estimate"
                    else:
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

    def _apply_stream_usage(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Ask streaming servers for usage (and running usage) in the stream."""
        if self.stream_usage == "off" or not payload.get("stream"):
            return payload
        options = dict(payload.get("stream_options") or {})
        options.setdefault("include_usage", True)
        if self.stream_usage == "continuous":
            options.setdefault("continuous_usage_stats", True)
        return {**payload, "stream_options": options}

    def _tokenizer_count(self, text: str) -> int | None:
        """Count tokens with the configured Hugging Face tokenizer, if any."""
        if not self.config.tokenizer or self._tokenizer_failed or not text:
            return None
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer  # type: ignore

                self._tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer)
            except Exception as exc:  # pragma: no cover - optional dependency
                self._tokenizer_failed = True
                self.console.print(f"[yellow]Tokenizer unavailable ({exc}); using estimates.[/yellow]")
                return None
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def _selfcheck_parser(self):
        decoder = SSEDecoder()

        def parse(raw: bytes) -> None:
            for event in decoder.feed(raw):
                chunk = self._parse_sse_data(event)
                if chunk is not None:
                    self._extract_output_fragments(chunk)
                    self._extract_usage(chunk)

        return parse

    @staticmethod
    def _decode_raw_header_part(value: Any) -> str:
        """Decode one raw HTTP header part without changing its spelling."""
        if isinstance(value, (bytes, bytearray, memoryview)):
            return bytes(value).decode("latin-1", errors="replace")
        return str(value)

    @classmethod
    def _is_sensitive_response_header(cls, name: str) -> bool:
        """Return whether a response header value contains credential material."""
        normalized = name.casefold().replace("_", "-")
        if normalized in _SENSITIVE_RESPONSE_HEADER_NAMES:
            return True
        if normalized.startswith("set-cookie"):
            return True
        return normalized.endswith((
            "-api-key",
            "-apikey",
            "-auth-token",
            "-access-token",
            "-refresh-token",
            "-client-secret",
        ))

    @classmethod
    def _capture_response_headers(
        cls,
        response: Any,
    ) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
        """Capture raw response headers while redacting credential values.

        ``aiohttp`` exposes ``raw_headers`` as an ordered sequence of byte
        pairs.  Reading that sequence instead of ``response.headers`` keeps
        duplicate fields and the server's original header-name casing.
        """
        captured: list[dict[str, str]] = []
        redactions: list[dict[str, Any]] = []
        for index, pair in enumerate(getattr(response, "raw_headers", ())):
            try:
                raw_name, raw_value = pair
            except (TypeError, ValueError):
                continue
            name = cls._decode_raw_header_part(raw_name)
            value = cls._decode_raw_header_part(raw_value)
            if cls._is_sensitive_response_header(name):
                value = _REDACTED_HEADER_VALUE
                redactions.append({
                    "index": index,
                    "name": name,
                    "reason": "sensitive credential/cookie header",
                    "replacement": _REDACTED_HEADER_VALUE,
                })
            captured.append({"name": name, "value": value})
        return captured, redactions

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
        """Execute warmup requests with at most ``concurrency`` in flight."""
        self.console.print()
        self.console.print(Panel(
            f"[yellow]Warm-up Phase: Sending {self.config.warmup_requests} requests to initialize model...[/yellow]",
            title="Warm-up",
            border_style="yellow"
        ))
        remaining = self.config.warmup_requests

        with Progress(disable=not self.config.show_progress) as progress:
            task = progress.add_task("[yellow]Warming up...", total=self.config.warmup_requests)

            async def warmup_worker() -> None:
                nonlocal remaining
                while remaining > 0:
                    remaining -= 1
                    await self._make_request(session, is_warmup=True)
                    progress.update(task, advance=1)

            workers = min(self.config.concurrency, self.config.warmup_requests)
            await asyncio.gather(*(warmup_worker() for _ in range(workers)))

        self.console.print("[green]✓ Warm-up complete![/green]")
        await asyncio.sleep(1)  # Brief pause after warmup

    def _build_schedule(self) -> ArrivalSchedule:
        return build_schedule(
            self.config.arrival_mode,
            requests=self.config.total_requests,
            concurrency=self.config.concurrency,
            rate=self.config.arrival_rate,
            shape=self.config.arrival_shape,
            seed=self.config.arrival_seed,
            duration_seconds=self.config.duration_seconds,
            replay_file=self.config.replay_file,
        )

    async def _closed_worker(
        self,
        session: aiohttp.ClientSession,
        scenario_start: float,
        deadline: float | None,
        progress: Progress,
        progress_task: TaskID,
    ) -> None:
        """One of C closed-loop workers: send the next request when the last ends."""
        rate = self.config.target_rps
        while deadline is None or time.monotonic() < deadline:
            request_id = await self._claim_request_id()
            if request_id is None:
                break
            # With a rate cap the intended send time is known, so the delay a
            # slow response imposes on later requests can be measured.
            scheduled = scenario_start + (request_id - 1) / rate if rate > 0 else None
            await self.rate_limiter.acquire()
            metrics = await self._make_request(
                session, request_id=request_id, scheduled_time=scheduled
            )
            self.result.add_request(metrics)
            progress.update(progress_task, advance=1)

    async def _run_open_loop(
        self,
        session: aiohttp.ClientSession,
        schedule: ArrivalSchedule,
        scenario_start: float,
        progress: Progress,
        progress_task: TaskID,
    ) -> None:
        """Send each request at its precomputed time regardless of earlier ones."""

        async def send(index: int, scheduled: float) -> None:
            fixture_id = schedule.fixture_ids[index] if schedule.fixture_ids else None
            metrics = await self._make_request(
                session, request_id=index + 1, scheduled_time=scheduled, fixture_id=fixture_id
            )
            self.result.add_request(metrics)
            progress.update(progress_task, advance=1)

        tasks: list[asyncio.Task[None]] = []
        for index, offset in enumerate(schedule.offsets):
            scheduled = scenario_start + offset
            delay = scheduled - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(send(index, scheduled)))
        await asyncio.gather(*tasks)

    def _make_connector(self, open_loop: bool) -> aiohttp.TCPConnector:
        # Keep-alive by default: a pool sized to the closed-loop concurrency
        # (unbounded for open loop, which must never queue in the client).
        limit = 0 if open_loop else max(1, self.config.concurrency)
        return aiohttp.TCPConnector(
            limit=limit,
            limit_per_host=limit,
            force_close=self.config.cold_connections,
        )

    async def run(self) -> BenchmarkResult:
        """Execute the full benchmark."""
        url = f"{self.config.endpoint}/{self.config.api_route}"
        schedule = self._build_schedule()
        self.schedule = schedule
        self.result.arrival_mode = schedule.mode
        self.result.arrival_schedule = schedule.to_dict()
        if self.config.schedule_log_path:
            from pathlib import Path

            schedule.write_jsonl(Path(self.config.schedule_log_path))
        if schedule.mode == "burst":
            self.result.total_requests = len(schedule)
        self.console.print(Panel(
            f"[cyan]Target: {url}\n"
            f"Model: {self.config.model}\n"
            f"Arrival: {schedule.mode}"
            + (f" @ {schedule.rate}/s" if schedule.rate else "")
            + f"\nConcurrency: {self.config.concurrency}\n"
            f"Target RPS: {'Unlimited' if self.config.target_rps <= 0 else self.config.target_rps}\n"
            f"Total Requests: "
            f"{len(schedule) if schedule.open_loop else self.config.total_requests}"
            + (f"\nDuration: {self.config.duration_seconds}s" if self.config.duration_seconds else "")
            + "[/cyan]",
            title="Benchmark Configuration",
            border_style="cyan"
        ))

        capacity = (
            measure_event_capacity(self._selfcheck_parser()) if self.config.self_check else None
        )
        connector = self._make_connector(schedule.open_loop)

        # Monitoring (probes, /metrics, Cloud Eye) uses its own pool so it never
        # waits behind, or takes a connection from, the measured requests.
        async with aiohttp.ClientSession(connector=connector) as session, \
                aiohttp.ClientSession() as monitor_session:
            server_probe = await probe_server(
                monitor_session, self.config.endpoint, self.config.api_key
            )
            handler = self.config.modality_handler
            fixture_file = str(handler.path) if isinstance(handler, FixtureHandler) else None
            self.result.manifest = build_manifest(
                endpoint=self.config.endpoint,
                api_route=self.config.api_route,
                model=self.config.model,
                server_probe=server_probe,
                schedule=schedule.to_dict(),
                fixture_file=fixture_file,
                fixture_manifest_sha256=(
                    handler.manifest_sha256() if isinstance(handler, FixtureHandler) else None
                ),
                tokenizer=self.config.tokenizer,
                image_digest=self.config.image_digest,
                config_snapshot=self.config.snapshot(),
                endpoint_label=self.config.endpoint_label,
            )

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
                await server_collector.start(monitor_session)

            lag_monitor = LagMonitor()
            if self.config.start_gate is not None:
                scenario_start = await self.config.start_gate.wait()
            else:
                scenario_start = time.monotonic()
            lag_monitor.start()

            # Only the measured request interval contributes to aggregate throughput.
            self.result.scenario_start_monotonic = scenario_start
            self.result.start_time = datetime.now()
            scenario_start_epoch_ms = int(time.time() * 1000)
            deadline = (
                scenario_start + self.config.duration_seconds
                if self.config.duration_seconds else None
            )
            
            self.console.print()
            self.console.print(Panel(
                "[green]Starting benchmark...[/green]",
                title="Benchmark",
                border_style="green"
            ))
            
            # Main benchmark with progress tracking
            with Progress(disable=not self.config.show_progress) as progress:
                task = progress.add_task(
                    "[green]Executing requests...",
                    total=(
                        len(schedule) if schedule.open_loop
                        else None if deadline else self.config.total_requests
                    ),
                )
                
                try:
                    if schedule.open_loop:
                        await self._run_open_loop(session, schedule, scenario_start, progress, task)
                    else:
                        await asyncio.gather(*(
                            self._closed_worker(session, scenario_start, deadline, progress, task)
                            for _ in range(self.config.concurrency)
                        ))
                finally:
                    self.result.end_time = datetime.now()
                    scenario_end_epoch_ms = int(time.time() * 1000)
                    await lag_monitor.stop()
                    if capacity is not None:
                        self.result.client_self_check = build_report(
                            capacity,
                            lag_monitor.lags_ms,
                            peak_event_rate(
                                (record.t, record.events)
                                for metric in self.result.request_metrics
                                for record in metric.chunk_records
                            ),
                        )
                    if server_collector is not None:
                        await server_collector.stop(monitor_session)
                        self.result.server_metrics = server_collector.to_dict(
                            self.result.scenario_start_monotonic
                        )
                    if self.config.modelarts_cloud_eye is not None:
                        cloud_eye_collector = ModelArtsCloudEyeCollector(
                            self.config.modelarts_cloud_eye
                        )
                        self.result.provider_monitoring = await cloud_eye_collector.collect(
                            monitor_session,
                            scenario_start_epoch_ms,
                            scenario_end_epoch_ms,
                        )
        
        return self.result
