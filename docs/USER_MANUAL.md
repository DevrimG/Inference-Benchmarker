# Multi-Modal LLM Benchmark: User Manual

This manual covers running benchmarks against OpenAI-compatible endpoints
(ModelArts MaaS, ModelArts Real-Time Inference, self-hosted vLLM), reading the
results, and checking that the numbers are right.

- [1. Quick start](#1-quick-start)
- [2. What the tool measures](#2-what-the-tool-measures)
- [3. Writing a config file](#3-writing-a-config-file)
- [4. Choosing how requests arrive](#4-choosing-how-requests-arrive)
- [5. Run modes: fixed, ladder, soak, burst](#5-run-modes-fixed-ladder-soak-burst)
- [6. Several endpoints at once](#6-several-endpoints-at-once)
- [7. Reading the results](#7-reading-the-results)
- [8. Checking the numbers](#8-checking-the-numbers)
- [9. Endpoints with self-signed certificates](#9-endpoints-with-self-signed-certificates)
- [10. Troubleshooting](#10-troubleshooting)
- [11. Reference](#11-reference)

---

## 1. Quick start

![Workflow: install, configure, dry run, run, read](images/workflow.svg)

```bash
python3.12 -m venv venv && source venv/bin/activate
python -m pip install -r requirements.txt
```

**Interactive mode** asks questions step by step and is the quickest way to try
an endpoint:

```bash
./run.sh
```

**Config-file mode** is repeatable and is what lab runs should use. Copy the
example, edit it, check it, then run it:

```bash
cp examples/lab.example.toml my_run.toml
python -m llm_load_tester --config my_run.toml --dry-run
export BENCH_API_KEY='your-key'
python -m llm_load_tester --config my_run.toml
```

`--dry-run` prints the resolved settings without sending any requests. API
keys are read from the environment variable named by `api_key_env`. Never put
a key in the config file.

A minimal config for one endpoint:

```toml
mode = "fixed"
model = "Qwen3-30B-A3B"
api_route = "v1/chat/completions"
api_key_env = "BENCH_API_KEY"
modality = "text"
concurrency = 4
total_requests = 16
warmup_requests = 2
repeats = 2
exports = ["json", "csv", "xlsx"]

[modality_config]
input_tokens = 512
output_tokens = 256

[[endpoints]]
label = "my-deployment"
url = "https://my-endpoint.example.com"
metrics_url = "https://my-endpoint.example.com/metrics"   # optional, vLLM only
```

The request URL is `url` + `/` + `api_route`. MaaS uses
`api_route = "v2/chat/completions"`; vLLM and most Real-Time Inference
deployments use `v1/chat/completions`.

---

## 2. What the tool measures

![Timeline of one streamed request](images/request-timeline.svg)

The tool streams every response and stores each network read with its arrival
time, size and the server's running token count. Every timing below is derived
from that record.

| Metric | Meaning |
|---|---|
| **TTFT** | Request sent → first read that carries output (reasoning counts as output). Includes network, gateway and queueing. |
| **Chunk gap** | Time between two consecutive output reads. |
| **Per-token ITL** | Chunk gap ÷ tokens that read carried. With multi-token chunks (MTP, speculative decoding) this is smaller than the chunk gap. |
| **Max stall** | The largest chunk gap in a request. |
| **Decode tok/s** | Tokens after the first read ÷ time from first to last read. This is the speed one user sees. |
| **TPOT** | The inverse of decode tok/s: seconds per token after the first read. |
| **Latency** | Request sent → stream ends. |

**Where token counts come from.** The tool asks the server for running usage
(`stream_options.continuous_usage_stats`). Each record's
`token_timeline_source` says what it got:

| Source | Meaning | Trust |
|---|---|---|
| `continuous_usage` | The server sent a running count on every chunk (vLLM does). | Exact |
| `sse_event_count_scaled_to_final_usage` | Only a final total arrived (MaaS does this); event counts are scaled to that total. | Good estimate; assumes tokens spread evenly over events |
| `sse_event_count_estimate` | No usage at all; one token per event. | Undercounts multi-token chunks |

---

## 3. Writing a config file

Configs are TOML (or JSON with the same keys).
[`examples/lab.example.toml`](../examples/lab.example.toml) lists every key
with a comment. The settings you will change most:

| Key | Default | What it does |
|---|---|---|
| `mode` | `fixed` | `fixed`, `ladder`, `soak` or `burst` (§5) |
| `model` | (required) | Model name sent in each request |
| `api_route` | `v1/chat/completions` | Appended to each endpoint's `url` |
| `api_key_env` | none | Name of the environment variable holding the key |
| `modality` | `text` | `text`, `image`, `voice` or `fixture` |
| `concurrency` | 1 | Requests in flight (closed loop); burst size; split across endpoints |
| `total_requests` | 0 | Measured requests (closed loop); cap for open loop and replay (0 = all) |
| `warmup_requests` | 0 | Sent first and not measured; at most `concurrency` in flight |
| `repeats` | 1 | Run the whole test r1..rk; the summary reports the CV of each key metric |
| `timeout_seconds` | 600 | Per request |
| `exports` | `["json", "csv"]` | Any of `json`, `csv`, `xlsx` |
| `stream_usage` | `continuous` | `continuous`, `final` or `off` (§10) |
| `record_request_payload` | true | Store each sent request body in the exports |
| `cold_connections` | false | Open a new TCP/TLS connection for every request |

**Text workloads:** `[modality_config]` sets `input_tokens` and
`output_tokens` (sent as `max_tokens`). Prompts are generated per request.

**Fixed prompts:** use `modality = "fixture"` and a JSONL file of fixtures,
each with a stable `fixture_id`. Every result record carries the fixture it
used, so runs can be compared prompt by prompt:

```json
{"fixture_id": "chat-001", "messages": [{"role": "user", "content": "..."}], "max_tokens": 256}
{"fixture_id": "chat-002", "prompt": "Summarise ...", "max_tokens": 128}
```

**SLO and goodput:** with an `[slo]` block, each run reports how many
requests met all three limits and whether the run passed as a whole.

```toml
[slo]
ttft_p95_s = 2.0        # each request's TTFT; the run's p95
decode_p5_tok_s = 25    # each request's decode speed; the run's p5
max_stall_s = 1.0       # each request's max stall; the run's p99
```

---

## 4. Choosing how requests arrive

![Arrival modes](images/arrival-modes.svg)

```toml
[arrival]
mode = "closed"      # closed | poisson | gamma | burst | replay
rate = 4.0           # requests/s for poisson and gamma
shape = 1.0          # gamma only: < 1 burstier, > 1 smoother
seed = 42            # same seed, same schedule
replay_file = ""     # replay only
```

- **closed** keeps exactly `concurrency` requests in flight. Use it to find
  capacity. When the server slows down, fewer requests are sent, so latency
  under real traffic is understated.
- **poisson / gamma** send at a target rate, whatever the server does. Use them
  to measure latency at a given load. `shape` matches `vllm bench serve --burstiness`.
- **burst** sends exactly `concurrency` requests at once (e.g. PRB-REF-1N).
- **replay** sends at the times in a JSONL trace: one line per request with
  `timestamp` (milliseconds, the AIPerf/Mooncake format) or `offset_s`, and
  optionally `fixture_id`.

The schedule is generated before the run and saved as
`schedule_<endpoint>.jsonl` next to the results. That file is itself a valid
replay trace, so any run can be replayed exactly.

---

## 5. Run modes: fixed, ladder, soak, burst

| Mode | Use it to | Output |
|---|---|---|
| `fixed` | Measure one workload, repeated `repeats` times | `r1/`, `r2/` … plus mean and CV per metric |
| `ladder` | Find the concurrency where the service stops scaling | `c0001/`, `c0002/` … plus the stop decision |
| `soak` | Check stability over a long run | Per-window statistics (default 5 minutes) |
| `burst` | Measure a sudden spike of exactly C requests | `r1/` … like `fixed` |

**Ladder.** It runs each level in `[ladder].levels` in order and stops when
one rule fires on two consecutive levels:

| Rule | Fires when |
|---|---|
| `error_rate` | Errors > 2 % (`max_error_rate_percent`) |
| `ttft_p95` | TTFT p95 > the workload's limit (`[workloads.<name>].ttft_p95_limit_s`) |
| `throughput_gain` | Less than 5 % more throughput than the previous level |
| `p5_decode` | p5 per-user decode speed < 25 tok/s |

The **knee** is the last level before the stopping streak. That is the
highest concurrency worth running.

**Soak.** Set `[soak] duration_s` and `window_s`. Each window reports
completed requests, error rate, throughput, TTFT p50/p95, decode p5/p50 and
max stall. The last window is marked `partial` when requests still in flight
at the deadline finish after it.

You can override the mode, concurrency and repeats on the command line
without editing the file:

```bash
python -m llm_load_tester --config my_run.toml --mode ladder
python -m llm_load_tester --config my_run.toml --concurrency 16 --repeats 3
```

---

## 6. Several endpoints at once

![Fan-out across endpoints](images/fanout.svg)

Add one `[[endpoints]]` block per endpoint. `concurrency` is the total and is
split by `weight`. All endpoints finish their warmup, then start at the same
instant on a shared clock. The merged results have an `endpoint` column, and
server checks are reported per endpoint.

```toml
concurrency = 8

[[endpoints]]
label = "node1"
url = "http://10.0.0.11:8000"
metrics_url = "http://10.0.0.11:8000/metrics"
weight = 2            # gets 5 of 8

[[endpoints]]
label = "node2"
url = "http://10.0.0.12:8000"
metrics_url = "http://10.0.0.12:8000/metrics"
weight = 1            # gets 3 of 8
```

---

## 7. Reading the results

```text
outputs/lab/fixed-20261016-150436/
├── plan.json              resolved config (no secrets)
├── lab_summary.json       per-repeat key metrics, mean, CV; ladder or soak decision
├── r1/
│   ├── results.json       summary + every request + every network read
│   ├── results.xlsx       Benchmark Summary, Requests, vLLM Summary, vLLM Timeline, Request Context
│   ├── requests.csv       one row per request
│   ├── manifest.json      git SHA, endpoint, served model, server version, host, schedule hash
│   └── schedule_<label>.jsonl
└── r2/ …
```

Start with `lab_summary.json`. To dig into a run, open `results.xlsx` or
the `summary` block of `results.json`.

### Throughput: steady state vs full window

![Requests in flight with the steady window shaded](images/steady-window.svg)

The **full window** runs from the first send to the last token, so it includes
ramp-up and the drain at the end, when fewer requests are in flight. The
**steady state** window counts only the time all `concurrency` slots are busy.
Quote steady state for capacity, and the full window for "how long did 16
requests take".

| Window | Duration | Output throughput |
|---|---|---|
| Steady state (in flight = 4) | 20.2 s | 197 tok/s |
| Full window | 22.6 s | 181 tok/s |

*Qwen3-30B-A3B on ModelArts Real-Time Inference, run r1, 2026-10-16. The brief dips to 3 are the
moments between one request ending and its worker sending the next.*

### Per-request behaviour

![Cumulative tokens over time for three requests](images/token-timeline.svg)

Each line is one request's running token count. A steady slope means smooth
streaming; flat steps are stalls between reads. The same data is in
`results.json` → `raw_metrics[i].chunk_records`, as
`[seconds since send, bytes, events, token events, cumulative tokens]`.

| Request | TTFT | Decode | Max stall |
|---|---|---|---|
| smoothest | 1.09 s | 59.8 tok/s | 0.06 s |
| median | 1.07 s | 55.9 tok/s | 0.31 s |
| largest stall | 0.65 s | 40.3 tok/s | 0.84 s |

### Key summary fields

| Field | Meaning |
|---|---|
| `distributions.<metric>` | `n`, mean, min, max, p5/p50/p90/p95/p99 and `ci95` per percentile |
| `steady_state`, `full_window` | Throughput, duration and completed requests for each window |
| `goodput` | Requests meeting the SLO, `goodput_requests_per_second`, `run_slo_pass` |
| `server_agreement` | Client vs server checks (§8) |
| `littles_law` | Observed in-flight vs arrival rate × latency (§8) |
| `client_self_check` | Whether this machine kept up (§8) |
| `coordinated_omission` | How late requests left compared with their intended time |
| `token_count_sources` | How each request's tokens were counted (§2) |
| `stream_usage_effective` | Whether running usage was accepted |

Each percentile comes with its sample size and a 95 % confidence interval. Treat
p95 and p99 from small runs with care: with 16 requests, "p99" is close to
the slowest request.

Every request record also has `request_payload`: the exact JSON body that was
sent. Embedded base64 images and audio are replaced by
`<media sha256=... chars=...>`. Request headers, and therefore API keys, are
never stored.

### Recomputing a run without re-sending it

If the tool's calculations change, rebuild a summary from saved data:

```bash
python tools/recompute_results.py outputs/.../r1/results.json --out summary_recomputed.json
```

---

## 8. Checking the numbers

### Against the server's own counters

When `metrics_url` points at a vLLM `/metrics` endpoint, each run summary
includes `server_agreement`:

- **tokens:** the client's total completion tokens should equal the change in
  `vllm:generation_tokens_total`. `match: false` means either another client
  was using the server, or token counts were estimated (see
  `client_counts_all_server_reported`).
- **ttft / itl:** the client's percentile should fall in the same vLLM
  histogram bucket as the server's. A quantile is only compared when at least
  5 samples lie beyond it (p99 needs ≥ 500 samples).

![Client vs server inter-token gaps per histogram bucket](images/itl-buckets.svg)

On the Qwen3 run, the client and the server agree at the median: 86.5 % of
client gaps and 99.4 % of server samples fall in the 10–25 ms bucket. Gaps
over 50 ms make up 8.3 % of client gaps but only 0.5 % of server samples. That
extra delay is added between the server and the client (network, gateway),
and real users see it too. Client gaps under 10 ms are reads arriving close together
after such a delay. A client TTFT above the server's (0.56 s vs 0.23 s here)
is the same effect.

| Bucket upper bound (ms) | Client | Server |
|---|---|---|
| 10 | 1.9 % | 0 % |
| 25 | 86.5 % | 99.4 % |
| 50 | 3.3 % | 0.1 % |
| 75 to 1000 | 8.3 % in total | 0.5 % in total |

### Against other tools

`tools/cross_tool_agreement.py` runs `vllm bench serve` on the same fixtures,
endpoint and arrival process, and reports the difference for each metric:

```bash
python tools/cross_tool_agreement.py --ours outputs/.../r1/results.json \
    --base-url http://10.0.0.11:8000 --model Qwen3-30B-A3B \
    --fixtures examples/chat_fixtures.jsonl --concurrency 8 --num-prompts 200
```

Add `--dry-run` to print the vLLM command without running it, or
`--vllm-result file.json` to compare a result you already have. AISBench can
be run through `--aisbench-cmd`.

### Little's law

`littles_law.ratio_predicted_to_observed` should be close to 1 in a stable
run. Values around 0.8 usually mean the steady window is short compared with
request latency, which is common in small tests. Run more requests before
reading anything into it.

### Client self-check

`client_self_check.verdict` is `ok` or `client_limited`. It is
`client_limited` when the event loop lagged more than 5 ms (p99), or the peak
event rate exceeded half of what this machine can parse. In that case, split
the load across more client processes or machines before trusting tail
latencies.

---

## 9. Endpoints with self-signed certificates

IP-addressed ModelArts Real-Time Inference endpoints often present a
self-signed certificate. The tool rejects these by default. There are two
opt-in settings, both per endpoint and both recorded in the manifest.

**Pin the certificate (recommended).** Save the server's certificate once and
trust only that one:

```bash
mkdir -p certs
openssl s_client -connect 203.0.113.10:443 </dev/null 2>/dev/null | openssl x509 > certs/rti.pem
openssl x509 -in certs/rti.pem -noout -fingerprint -sha256
```

```toml
[[endpoints]]
label = "rti"
url = "https://203.0.113.10/v2/infer/<service-id>"
metrics_url = "https://203.0.113.10/v2/infer/<service-id>/metrics"
tls_ca_file = "certs/rti.pem"
tls_check_hostname = false     # the certificate names no host or IP
```

An impostor server is then rejected. The weak point is the first download: if
you can get the fingerprint from somewhere else (console, support ticket),
compare it. Save the certificate again after a redeployment if ModelArts
issues a new one.

**Turn verification off.** `tls_verify = false` accepts any certificate. The
tool prints a warning on every run. Anyone able to intercept the traffic can
read the key and prompts, so use it only with short-lived keys.

---

## 10. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Every request fails with `connection_error` mentioning a certificate | Self-signed certificate (§9). |
| `stream_usage_effective` says `continuous rejected` | The server answered 400 to `continuous_usage_stats`. The tool switched to final usage on its own. Set `stream_usage = "final"` to avoid the first failed request. |
| `token_timeline_source` is `sse_event_count_scaled_to_final_usage` | Normal for MaaS: only a final total is sent. Decode speed is estimated. |
| Decode speed looks ~2–3× too low on an older run | Runs before commit `a284e13` counted one token per event. Re-run `tools/recompute_results.py`. |
| `completion_tokens` far above `output_tokens` | Reasoning models count thinking tokens. MaaS `deepseek-v4.1-flash` used ~3,100 reasoning tokens per request on top of `max_tokens`. |
| `server_agreement.tokens.match` is false | Another client used the server during the run, or counts were estimated. |
| `client_self_check.verdict` is `client_limited` | This machine is the bottleneck (§8). |
| `littles_law` ratio well below 1 | Steady window too short; run more requests. |
| `served_models` doesn't list your `model` | The name you send differs from what the endpoint advertises; check which model actually answered (`provider_model`). |
| Excel cell ends with `[truncated …]` | Excel's 32,767-character cell limit; the full value is in `results.json`. |

---

## 11. Reference

| Command | Purpose |
|---|---|
| `./run.sh` or `python -m llm_load_tester` | Interactive mode |
| `python -m llm_load_tester --config FILE [--mode M] [--concurrency N] [--repeats N] [--output-dir DIR] [--dry-run]` | Config-file mode |
| `python tools/run_fixed_benchmark.py --config FILE` | Same as above, always `fixed` |
| `python tools/cross_tool_agreement.py --ours results.json …` | Compare with `vllm bench serve` / AISBench |
| `python tools/recompute_results.py results.json` | Rebuild a summary from saved data |
| `python -m llm_load_tester.mock_server --config-json '{...}'` | Local mock server for practice runs |
| `python -m pytest` | Test suite (needs `requirements-dev.txt`) |
| `python docs/build_figures.py results.json` | Rebuild this manual's figures from a run |

Every config key is listed in [`examples/lab.example.toml`](../examples/lab.example.toml).
The measurement design is described in the [README](../README.md).
