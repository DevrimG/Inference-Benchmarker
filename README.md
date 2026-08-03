# Multi-Modal-LLM-Benchmark

An asynchronous CLI benchmarker for OpenAI-compatible LLM APIs across text, image, and voice workflows.

Python 3.10+ is required. The examples below use Python 3.12.

## Features

- **Multiple Modalities**: Supports Text, Image (multimodal), and Voice inputs
- **Dynamic Responsiveness Metrics**: Measures true body-byte TTFB and reports TTFT only when token-bearing streamed output is observable
- **vLLM Metrics Capture**: Optionally samples a Prometheus-compatible `/metrics` endpoint before, during, and after each measured scenario
- **ModelArts MaaS Support**: Captures GLM reasoning/token metadata and can query aggregated `SYS.MaaS` metrics through Cloud Eye
- **Configurable Load**: Adjustable concurrency, RPS targeting, and request counts
- **Warm-up Phase**: Pre-test requests to eliminate cold-start latency
- **Rich Output**: Beautiful terminal tables with detailed metrics
- **Data Export**: Export results to JSON or one formatted Excel workbook

## Installation

```bash
# Create a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Interactive Mode

Run the tool and follow the prompts:

```bash
# Run from project root
python -m llm_load_tester

# Or use the convenience script
./run.sh
```

### Configuration Options

The interactive CLI will guide you through:

1. **Modality Selection**: Text, Image, or Voice
2. **API Configuration**: Endpoint, route, and model name (with presets)
3. **Modality-specific Settings**:
   - Text: Input/output token lengths
   - Image: Directory path containing images
   - Voice: Directory path containing audio files
4. **Load Parameters**: Concurrency, target RPS, total requests
5. **Optional Monitoring**: vLLM `/metrics`, or Cloud Eye for a ModelArts MaaS endpoint

## Metrics Collected

- **TTFB** (Time To First Byte): Client request start until the first non-empty response-body byte
- **TTFT** (Time To First Token): Client request start until the first identifiable streamed text, reasoning, or tool-call token
- **First Visible Text**: Kept separate from reasoning-token TTFT for reasoning models
- **TPOT** (Time Per Output Token): Average generation speed per token
- **End-to-End Latency**: Total request completion time (p50, p95, p99)
- **Throughput**: Tokens per second (overall and per-request averages)
- **Error Rates**: Categorized by HTTP status codes

The primary responsiveness metric is selected from evidence in each response:
streamed token-bearing output uses TTFT, while a buffered response uses TTFB and
does not invent a TTFT value. Both raw measurements remain available when both
can be observed.

For reasoning models, TTFT is the arrival of the first token-bearing output,
including `reasoning_content`. Time to first visible answer text is retained as a
separate measurement. `X-Request-Id` and `X-Span-Id` are also captured when the
provider returns them.

## GLM and ModelArts MaaS Responses

For ModelArts MaaS text requests, the interactive CLI supports both output-limit
contracts:

- `max_completion_tokens` limits reasoning tokens plus visible answer tokens;
- `max_tokens` limits the visible answer and excludes chain-of-thought tokens.

Thinking can use the provider default or be explicitly enabled/disabled. The
runner requests streamed usage data and preserves response metadata including
model, service tier, finish reason, detailed reasoning/cached token usage, and
the provider's `first_token_return_time` values.

`first_token_return_time` is stored as provider metadata. It is an absolute
per-chunk return timestamp, not a TTFT duration, so it is never substituted for
the client-observed TTFT or TTFB measurements.

## Optional vLLM `/metrics` Collection

Enable server metrics in the interactive prompt and point it at the API server's
Prometheus endpoint, usually `http://host:port/metrics`. The collector:

- takes a baseline after warm-up and before measured traffic;
- samples gauges such as running/waiting requests and KV-cache usage during load;
- takes a final scrape and calculates scenario deltas for counters and histograms;
- stores the original labeled samples plus canonical summaries for TTFT, TPOT,
  queue time, token counts, request counts, preemptions, and cache pressure.

Server-internal metrics remain separate from client-observed timings. Standard
vLLM exposes a TTFT histogram; provider-specific TTFB histograms are reported as
TTFB only when the endpoint actually exposes one.

JSON exports include a `server_metrics` section. Excel exports keep everything
in one `.xlsx` file with `Benchmark Summary`, `Requests`, `vLLM Summary`,
`vLLM Timeline`, and `Request Context` sheets. The request-context sheet aligns
each client request window with nearby server gauge scrapes. These values are
contextual observations, not exact Prometheus attribution to an individual
request. Exact request-level queue/prefill/decode attribution requires logs or
traces carrying request IDs.

The benchmark summary includes an explicit unit beside every numeric metric,
such as `seconds`, `percent`, `tokens`, or `tokens/second`. JSON exports provide
the same additive metadata in `summary_units` while preserving existing keys.

The programmatic `BenchmarkResult.export_csv()` method remains available for
backward compatibility and continues to create vLLM metric sidecars.

## Optional ModelArts Cloud Eye Collection

Public ModelArts MaaS endpoints do not expose the serving engine's Prometheus
`/metrics` endpoint. When a `modelarts-maas.com` endpoint is selected, the CLI
can instead query Huawei Cloud Eye after the measured scenario.

This requires a temporary IAM token with `ces:metricData:list`, the regional
Cloud Eye endpoint, project ID, and either a `maas_api_id` or
`maas_service_name` dimension. The inference API key and Cloud Eye IAM token are
different credentials. Neither credential is written to benchmark exports.

Cloud Eye results stay in a separate `provider_monitoring` section because they
are one-minute aggregates and can include traffic outside the benchmark when a
dimension is shared. Latency values are normalized from milliseconds to seconds.
Excel exports add a `ModelArts Metrics` sheet to the same workbook. The legacy
programmatic CSV export continues to create ModelArts sidecars.

## Testing Without a GPU

The test suite starts a CPU-only local HTTP server that emulates OpenAI-compatible
streaming, buffered responses, GLM metadata, a changing vLLM-style `/metrics`
endpoint, and the Cloud Eye batch-query contract:

```bash
python -m unittest discover -v
```

This validates collection and timing semantics without loading a model. A short
smoke run against real vLLM is still recommended once GPU access is available,
because metric availability can vary by vLLM/provider version.

The base benchmark can still run when the optional Prometheus parser is absent.
To enable `/metrics` collection, install the full project requirements first:

```bash
python3.12 -m pip install -r requirements.txt
```

## Project Structure

```
.
├── benchmarks/       # Export destination for benchmark results
├── images/           # Local image inputs (gitignored except .gitkeep)
├── llm_load_tester/  # Python package
├── README.md
├── QUICKSTART.md
├── requirements.txt
├── run.sh
└── sound/            # Local audio inputs (gitignored except .gitkeep)
```

## Example Workflow

```bash
# 1. Start the tool
python -m llm_load_tester

# 2. Select modality (e.g., Text)
# 3. Choose API preset or enter custom endpoint
# 4. Select or enter model name
# 5. Set token lengths (e.g., 512 input, 1024 output)
# 6. Configure load (e.g., 8 concurrent, 5 RPS, 100 requests)
# 7. Wait for warm-up and benchmark completion
# 8. View results table in terminal
# 9. Optionally export to JSON or Excel
```

## API Compatibility

This tool uses the OpenAI-compatible chat completions API format:
- Streaming via `stream: true`
- SSE (Server-Sent Events) response parsing
- Standard message format for text
- Base64-encoded media for image/voice

Compatible with:
- vLLM
- Text Generation Inference (TGI)
- TensorRT-LLM
- LMStudio
- Ollama
- Any OpenAI-compatible endpoint

## Notes

- The tool generates dummy text for text modality testing
- For image testing, place files in `images/` or another directory and the runner will cycle through a shuffled pool
- For voice testing, place files in `sound/` or another directory and the runner will cycle through a shuffled pool
- Warm-up requests (5) are sent before timing to initialize model/VRAM
- Local assets and generated benchmark results are intentionally ignored by git
