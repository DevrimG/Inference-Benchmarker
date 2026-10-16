#!/usr/bin/env python3
"""Run `vllm bench serve` (and optionally AISBench) on the same fixtures and
endpoint as one of our runs, and report how far the numbers differ.

    python tools/cross_tool_agreement.py \
        --ours outputs/lab/fixed-.../r1/results.json \
        --base-url http://10.0.0.11:8000 --model Qwen3-32B \
        --fixtures examples/chat_fixtures.jsonl --concurrency 8 --num-prompts 200

Use the same arrival process on both sides: our closed loop at C matches
`--max-concurrency C --request-rate inf`; our Poisson/gamma at rate R and
shape k matches `--request-rate R --burstiness k`.

Metric mapping. vLLM's ITL is the gap between streamed chunks, so it is
compared with our chunk_gap_s (our per-token itl_s divides each gap by the
tokens it carried). vLLM's output throughput uses the whole benchmark
duration, so it is compared with our full-window throughput.

`--vllm-result` skips running vLLM and compares a saved result file.
AISBench has no stable machine-readable output we could verify here: pass the
command with `--aisbench-cmd` and, if you have it, a flat JSON of the same
metric names with `--aisbench-result`.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from tabulate import tabulate

# metric -> (our extractor, vLLM result key, unit scale vLLM->ours, tolerance %)
METRICS: dict[str, tuple[Any, str, float, float]] = {
    "completed_requests": (lambda s: s["successful_requests"], "completed", 1.0, 0.0),
    "total_output_tokens": (lambda s: s["total_output_tokens"], "total_output_tokens", 1.0, 0.5),
    "output_tokens_per_second": (
        lambda s: s["full_window"]["output_tokens_per_second"], "output_throughput", 1.0, 5.0),
    "requests_per_second": (
        lambda s: s["full_window"]["requests_per_second"], "request_throughput", 1.0, 5.0),
    "ttft_mean_s": (lambda s: s["distributions"]["ttft_s"]["mean"], "mean_ttft_ms", 1e-3, 10.0),
    "ttft_p50_s": (lambda s: s["distributions"]["ttft_s"]["p50"], "median_ttft_ms", 1e-3, 10.0),
    "ttft_p99_s": (lambda s: s["distributions"]["ttft_s"]["p99"], "p99_ttft_ms", 1e-3, 15.0),
    "tpot_p50_s": (lambda s: s["distributions"]["tpot_s"]["p50"], "median_tpot_ms", 1e-3, 10.0),
    "tpot_p99_s": (lambda s: s["distributions"]["tpot_s"]["p99"], "p99_tpot_ms", 1e-3, 15.0),
    "chunk_itl_p50_s": (
        lambda s: s["distributions"]["chunk_gap_s"]["p50"], "median_itl_ms", 1e-3, 10.0),
    "chunk_itl_p99_s": (
        lambda s: s["distributions"]["chunk_gap_s"]["p99"], "p99_itl_ms", 1e-3, 15.0),
    "e2e_p50_s": (lambda s: s["distributions"]["latency_s"]["p50"], "median_e2el_ms", 1e-3, 10.0),
}


def fixtures_to_vllm_custom(fixture_path: Path, out_path: Path) -> int:
    """Write vLLM's `custom` dataset format: one {"prompt": ...} per line."""
    rows = 0
    with open(fixture_path) as src, open(out_path, "w") as dst:
        for line in src:
            if not line.strip():
                continue
            record = json.loads(line)
            if isinstance(record.get("payload"), dict):
                messages = record["payload"].get("messages") or []
            else:
                messages = record.get("messages") or [{"role": "user", "content": record.get("prompt", "")}]
            user = [m for m in messages if m.get("role") == "user"]
            content = (user[-1] if user else messages[-1]).get("content", "")
            if isinstance(content, list):
                content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
            dst.write(json.dumps({"prompt": content}) + "\n")
            rows += 1
    return rows


def fixture_output_len(fixture_path: Path) -> int | None:
    lengths = []
    for line in Path(fixture_path).read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            source = record.get("payload", record)
            value = source.get("max_tokens") or source.get("max_completion_tokens")
            if value:
                lengths.append(int(value))
    return max(lengths) if lengths else None


def build_vllm_command(args: argparse.Namespace, dataset: Path, result_dir: Path) -> list[str]:
    command = [
        *shlex.split(args.vllm_bin), "bench", "serve",
        "--backend", "openai-chat",
        "--base-url", args.base_url,
        "--endpoint", "/" + args.api_route.lstrip("/"),
        "--model", args.model,
        "--dataset-name", "custom",
        "--dataset-path", str(dataset),
        "--num-prompts", str(args.num_prompts),
        "--request-rate", str(args.request_rate),
        "--burstiness", str(args.burstiness),
        "--seed", str(args.seed),
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "50,95,99",
        "--save-result", "--save-detailed",
        "--result-dir", str(result_dir),
        "--result-filename", "vllm_bench.json",
    ]
    if args.concurrency:
        command += ["--max-concurrency", str(args.concurrency)]
    if args.output_len:
        command += ["--custom-output-len", str(args.output_len)]
    if args.tokenizer:
        command += ["--tokenizer", args.tokenizer]
    if args.vllm_extra:
        command += shlex.split(args.vllm_extra)
    return command


def compare(ours: dict[str, Any], theirs: dict[str, Any], tool: str) -> list[dict[str, Any]]:
    rows = []
    for name, (extract, key, scale, tolerance) in METRICS.items():
        try:
            mine = extract(ours)
        except (KeyError, TypeError):
            mine = None
        raw = theirs.get(key, theirs.get(name))
        other = raw * scale if isinstance(raw, (int, float)) and key in theirs else raw
        diff_pct = None
        if isinstance(mine, (int, float)) and isinstance(other, (int, float)) and other:
            diff_pct = (mine - other) / other * 100
        rows.append({
            "metric": name,
            "ours": mine,
            tool: other,
            "diff_percent": round(diff_pct, 3) if diff_pct is not None else None,
            "tolerance_percent": tolerance,
            "within": None if diff_pct is None else abs(diff_pct) <= tolerance,
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cross-tool agreement against vllm bench serve / AISBench")
    parser.add_argument("--ours", required=True, help="our results.json for the same workload")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--api-route", default="v1/chat/completions")
    parser.add_argument("--fixtures", help="our fixture JSONL (converted for vLLM)")
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=0)
    parser.add_argument("--request-rate", default="inf")
    parser.add_argument("--burstiness", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-len", type=int)
    parser.add_argument("--tokenizer")
    parser.add_argument("--vllm-bin", default="vllm")
    parser.add_argument("--vllm-extra", default="", help="extra flags, e.g. '--ignore-eos'")
    parser.add_argument("--vllm-result", help="compare an existing vllm bench result JSON")
    parser.add_argument("--aisbench-cmd", help="AISBench command to run (as you would type it)")
    parser.add_argument("--aisbench-result", help="flat JSON with our metric names from AISBench")
    parser.add_argument("--out", default="cross_tool_agreement.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    ours = json.loads(Path(args.ours).read_text())["summary"]
    report: dict[str, Any] = {"ours": args.ours, "comparisons": {}}
    work = Path(tempfile.mkdtemp(prefix="cross_tool_"))

    theirs = None
    if args.vllm_result:
        theirs = json.loads(Path(args.vllm_result).read_text())
    elif args.base_url and args.model and args.fixtures:
        dataset = work / "vllm_custom.jsonl"
        fixtures_to_vllm_custom(Path(args.fixtures), dataset)
        args.output_len = args.output_len or fixture_output_len(Path(args.fixtures))
        command = build_vllm_command(args, dataset, work)
        report["vllm_command"] = command
        print("vllm:", shlex.join(command))
        if not args.dry_run:
            if not shutil.which(shlex.split(args.vllm_bin)[0]):
                print(f"'{args.vllm_bin}' not found; install vLLM on the client or pass --vllm-result")
                return 2
            subprocess.run(command, check=True)
            theirs = json.loads((work / "vllm_bench.json").read_text())
    if theirs is not None:
        rows = compare(ours, theirs, "vllm")
        report["comparisons"]["vllm"] = rows
        print(tabulate(rows, headers="keys", floatfmt=".4g"))

    if args.aisbench_cmd:
        report["aisbench_command"] = args.aisbench_cmd
        print("aisbench:", args.aisbench_cmd)
        if not args.dry_run:
            completed = subprocess.run(shlex.split(args.aisbench_cmd), capture_output=True, text=True)
            (work / "aisbench_stdout.txt").write_text(completed.stdout + completed.stderr)
            report["aisbench_exit_code"] = completed.returncode
            report["aisbench_output"] = str(work / "aisbench_stdout.txt")
    elif shutil.which("ais_bench"):
        report["aisbench_note"] = "ais_bench is installed; pass --aisbench-cmd to run it."
    if args.aisbench_result:
        rows = compare(ours, json.loads(Path(args.aisbench_result).read_text()), "aisbench")
        report["comparisons"]["aisbench"] = rows
        print(tabulate(rows, headers="keys", floatfmt=".4g"))

    Path(args.out).write_text(json.dumps(report, indent=2))
    failed = [r["metric"] for rows in report["comparisons"].values() for r in rows if r["within"] is False]
    if failed:
        print("outside tolerance:", ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
