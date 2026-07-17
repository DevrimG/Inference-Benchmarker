# Quick Start Guide

## Installation

```bash
# 1. Navigate to the project directory
cd /path/to/Multi-Modal-LLM-Benchmark

# 2. Create and activate virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Running the Tool

```bash
# Option 1: Run as a module
python -m llm_load_tester

# Option 2: Use the runner script
./run.sh
```

## Example Session

```
┌───────────────────────────────────────────────────────────┐
│        Multi-Modal LLM Benchmark                          │
│  Asynchronous benchmarking tool for LLM APIs              │
└───────────────────────────────────────────────────────────┘

Select Test Modality:
  1. Text - Text-based chat completions
  2. Image - Vision/multimodal with images
  3. Voice - Audio/voice inputs

Enter choice [1]: 1

Select API Configuration Preset:
  1. OpenAI Compatible (Local) (http://localhost:8000)
  2. vLLM Default (http://localhost:8000)
  3. TGI (Text Generation Inference) (http://localhost:8080)
  ...
  7. Custom

Enter choice: 7

Enter custom API configuration:
Endpoint IP:Port [localhost:8000]: 192.0.2.10:8080
API Route [v1/chat/completions]: v1/chat/completions

Select Model:
  1. gpt-3.5-turbo
  2. gpt-4
  ...
  13. custom

Enter choice: 13
Enter model name: meta-llama/Llama-2-7b-chat-hf

┌───────────────────────────────────────────────────────────┐
│  Text Modality Configuration                              │
└───────────────────────────────────────────────────────────┘

Select Input Token Length:
  1. 256 tokens
  2. 512 tokens
  3. 1024 tokens
  4. 2048 tokens
  5. 4096 tokens
  6. 8192 tokens
  7. Custom

Enter choice: 2

Select Output Token Length (max_tokens):
  1. 256 tokens
  ...

Enter choice: 3

┌───────────────────────────────────────────────────────────┐
│  Load Testing Parameters                                  │
└───────────────────────────────────────────────────────────┘

Select Concurrency Level:
  (Number of simultaneous connections)
  1. 1 concurrent
  2. 2 concurrent
  ...
  7. Custom

Enter choice: 4

Select Target RPS:
  (Requests Per Second - use 0 for unlimited)
  1. 0.5 RPS
  ...
  8. Unlimited (0)

Enter choice: 4

Select Total Request Count:
  1. 10 requests
  ...
  7. Custom

Enter choice: 3

┌───────────────────────────────────────────────────────────┐
│  Benchmark Configuration                                  │
│  Target: http://192.0.2.10:8080/v1/chat/completions       │
│  Model: meta-llama/Llama-2-7b-chat-hf                     │
│  Concurrency: 8                                           │
│  Target RPS: 5.0                                          │
│  Total Requests: 100                                      │
└───────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────┐
│  Warm-up                                                  │
│  Warm-up Phase: Sending 5 requests to initialize model... │
└───────────────────────────────────────────────────────────┘
Warming up... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%
✓ Warm-up complete!

┌───────────────────────────────────────────────────────────┐
│  Benchmark                                                │
│  Starting benchmark...                                    │
└───────────────────────────────────────────────────────────┘
Executing requests... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%

┌────────────────────────────────────────────────────────────┐
│ Benchmark Results                                          │
├────────────────────────────────────────────────────────────┤
│ Metric                    │ Value                          │
├───────────────────────────┼────────────────────────────────┤
│ Configuration             │                                │
│   Modality                │ text                           │
│   Model                   │ meta-llama/Llama-2-7b-chat-hf  │
│   ...                     │                                │
│ Summary                   │                                │
│   Successful Requests     │ 100                            │
│   Failed Requests         │ 0                              │
│   Error Rate              │ 0.00%                          │
├───────────────────────────┼────────────────────────────────┤
│ Time To First Token       │                                │
│   Mean                    │ 0.234s                         │
│   p95                     │ 0.312s                         │
│   p99                     │ 0.345s                         │
├───────────────────────────┼────────────────────────────────┤
│ End-to-End Latency        │                                │
│   Mean                    │ 2.456s                         │
│   p95                     │ 3.123s                         │
│   p99                     │ 3.456s                         │
├───────────────────────────┼────────────────────────────────┤
│ Throughput                │                                │
│   Overall                 │ 234.56 tok/s                   │
│   Per Request Mean        │ 45.23 tok/s                    │
└───────────────────────────┴────────────────────────────────┘

Would you like to export the results to a file? [y/n]: y

Select Export Format:
  1. JSON (includes full raw metrics)
  2. CSV (raw request metrics only)

Enter choice [1]: 1
Enter filename for export [/path/to/Multi-Modal-LLM-Benchmark/benchmarks/llm_benchmark_20240218_120530.json]: 
✓ Results exported to: /path/to/Multi-Modal-LLM-Benchmark/benchmarks/llm_benchmark_20240218_120530.json

┌───────────────────────────────────────────────────────────┐
│  Benchmark complete!                                      │
└───────────────────────────────────────────────────────────┘
```

## Tips

1. **Warm-up is crucial**: The 5 warm-up requests ensure the model is loaded and VRAM is allocated before timing starts.

2. **RPS vs Concurrency**: 
   - Concurrency = max parallel requests
   - RPS = target requests per second (0 = unlimited)

3. **Token estimation**: For text mode, tokens are roughly 4 characters each.

4. **Image/Voice**: Put your test files in `images/` and `sound/` or point the tool at other folders. Files are reused from a shuffled pool when requests exceed file count.

5. **Responsiveness metric**: Streaming token-bearing responses use TTFT as the primary metric. Buffered responses use TTFB; the tool does not treat a completed buffered JSON response as a first token.

6. **Optional `/metrics` capture**: Enable it when prompted to sample vLLM/Prometheus metrics before, during, and after the measured requests. Warm-up activity is excluded from the server-metric deltas.

7. **GLM/ModelArts**: For a ModelArts MaaS endpoint, prefer `max_completion_tokens` when the limit should include reasoning. TTFT begins at the first reasoning or visible-text token; time to first visible text remains separate. Provider timestamps are metadata, not TTFT durations.

8. **ModelArts monitoring**: MaaS has no public vLLM `/metrics` endpoint. The optional Cloud Eye adapter needs a separate IAM token, project ID, and MaaS dimension, and returns one-minute aggregated metrics.

9. **Export**: JSON includes the summary, per-request metrics, and optional monitoring. CSV exports add `.vllm_metrics.*` and/or `.modelarts_metrics.*` sidecars when collection is enabled.
