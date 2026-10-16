from __future__ import annotations

"""Non-interactive benchmark CLI driven by a TOML or JSON config file.

    python -m llm_load_tester --config lab.toml [--mode ladder] [--dry-run]

See ``examples/lab.example.toml`` for every key. Credentials are never read
from the config file: give the name of an environment variable instead
(``api_key_env``, ``metrics_api_key_env``).
"""

import argparse
import asyncio
import json
import os
import sys
import tomllib
from dataclasses import replace
from pathlib import Path
from typing import Any

from rich.console import Console

from .benchmarker import LoadTestConfig
from .fixtures import FixtureHandler
from .lab import (
    EndpointSpec,
    LadderRules,
    repeat_summary,
    run_fanout,
    run_ladder,
    soak_windows,
    timestamp,
    write_run_outputs,
)
from .modalities import get_handler

MODES = ("fixed", "ladder", "soak", "burst")
console = Console()


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def _env(name: str | None) -> str | None:
    return os.environ.get(name) if name else None


def build_base_config(cfg: dict[str, Any]) -> tuple[LoadTestConfig, list[EndpointSpec]]:
    model = cfg["model"]
    modality = cfg.get("modality", "text")
    modality_config = dict(cfg.get("modality_config") or {})
    modality_config.setdefault("model", model)
    if modality == "fixture":
        if not cfg.get("fixtures"):
            raise ValueError("modality = 'fixture' requires 'fixtures' (a JSONL path)")
        handler = FixtureHandler(cfg["fixtures"])
        modality_config.setdefault("modality", "fixture")
    else:
        handler = get_handler(modality)
        modality_config.setdefault("modality", modality)

    endpoints_cfg = cfg.get("endpoints") or []
    if not endpoints_cfg:
        raise ValueError("config needs at least one [[endpoints]] entry with a url")
    endpoints = [
        EndpointSpec(
            label=str(e.get("label") or f"ep{i}"),
            endpoint=str(e["url"]).rstrip("/"),
            metrics_url=e.get("metrics_url"),
            weight=float(e.get("weight", 1.0)),
            api_key=_env(e.get("api_key_env")),
            metrics_api_key=_env(e.get("metrics_api_key_env")),
        )
        for i, e in enumerate(endpoints_cfg)
    ]
    labels = [e.label for e in endpoints]
    if len(set(labels)) != len(labels):
        raise ValueError(f"endpoint labels must be unique: {labels}")

    arrival = cfg.get("arrival") or {}
    metrics = cfg.get("metrics") or {}
    workload = (cfg.get("workloads") or {}).get(cfg.get("workload", ""), {})
    slo = {**(cfg.get("slo") or {}), **(workload.get("slo") or {})} or None
    base = LoadTestConfig(
        endpoint=endpoints[0].endpoint,
        api_route=cfg.get("api_route", "v1/chat/completions"),
        api_key=_env(cfg.get("api_key_env")),
        model=model,
        concurrency=int(cfg.get("concurrency", 1)),
        target_rps=float(cfg.get("target_rps", 0)),
        total_requests=int(cfg.get("total_requests", 0)),
        warmup_requests=int(cfg.get("warmup_requests", 0)),
        modality_handler=handler,
        modality_config=modality_config,
        timeout_seconds=float(cfg.get("timeout_seconds", 600)),
        metrics_scrape_interval_seconds=float(metrics.get("scrape_interval_s", 1.0)),
        metrics_strict=bool(metrics.get("strict", False)),
        metrics_api_key=_env(cfg.get("metrics_api_key_env")),
        arrival_mode=arrival.get("mode", "closed"),
        arrival_rate=arrival.get("rate"),
        arrival_shape=float(arrival.get("shape", 1.0)),
        arrival_seed=arrival.get("seed"),
        replay_file=arrival.get("replay_file") or None,
        cold_connections=bool(cfg.get("cold_connections", False)),
        stream_usage=cfg.get("stream_usage", "continuous"),
        slo=slo,
        tokenizer=cfg.get("tokenizer") or None,
        image_digest=cfg.get("image_digest") or None,
        show_progress=bool(cfg.get("show_progress", True)),
    )
    return base, endpoints


def ladder_rules(cfg: dict[str, Any]) -> LadderRules:
    ladder = cfg.get("ladder") or {}
    workload = (cfg.get("workloads") or {}).get(cfg.get("workload", ""), {})
    return LadderRules(
        max_error_rate_percent=float(ladder.get("max_error_rate_percent", 2.0)),
        ttft_p95_limit_s=workload.get("ttft_p95_limit_s", ladder.get("ttft_p95_limit_s")),
        min_throughput_gain_percent=float(ladder.get("min_throughput_gain_percent", 5.0)),
        min_p5_decode_tok_s=float(ladder.get("min_p5_decode_tok_s", 25.0)),
        consecutive_levels=int(ladder.get("consecutive_levels", 2)),
    )


async def _run_repeats(
    base: LoadTestConfig,
    endpoints: list[EndpointSpec],
    repeats: int,
    out_dir: Path,
    exports: list[str],
    cooldown_s: float = 0.0,
) -> dict[str, Any]:
    summaries = []
    for rep in range(1, repeats + 1):
        run_dir = out_dir / f"r{rep}"
        result = await run_fanout(base, endpoints, schedule_dir=run_dir)
        write_run_outputs(result, run_dir, exports)
        summaries.append(result.get_summary())
        console.print(f"[green]✓ {run_dir}[/green]")
        if cooldown_s and rep < repeats:
            await asyncio.sleep(cooldown_s)
    return repeat_summary(summaries)


async def run_from_config(cfg: dict[str, Any], out_root: Path) -> dict[str, Any]:
    mode = cfg.get("mode", "fixed")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    base, endpoints = build_base_config(cfg)
    exports = list(cfg.get("exports", ["json", "csv"]))
    repeats = int(cfg.get("repeats", 1))
    out_dir = out_root / f"{mode}-{timestamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = {**cfg, "resolved": base.snapshot(), "endpoints": [
        {"label": e.label, "url": e.endpoint, "metrics_url": e.metrics_url, "weight": e.weight}
        for e in endpoints
    ]}
    (out_dir / "plan.json").write_text(json.dumps(plan, indent=2, default=str))

    summary: dict[str, Any] = {"mode": mode, "output_dir": str(out_dir)}
    if mode == "fixed":
        summary["repeats"] = await _run_repeats(base, endpoints, repeats, out_dir, exports)
    elif mode == "burst":
        burst = cfg.get("burst") or {}
        base = replace(base, arrival_mode="burst", warmup_requests=base.warmup_requests)
        summary["repeats"] = await _run_repeats(
            base, endpoints, repeats, out_dir, exports, float(burst.get("cooldown_s", 0))
        )
    elif mode == "soak":
        soak = cfg.get("soak") or {}
        duration = float(soak.get("duration_s", 3600))
        window = float(soak.get("window_s", 300))
        base = replace(base, duration_seconds=duration)
        result = await run_fanout(base, endpoints, schedule_dir=out_dir)
        write_run_outputs(result, out_dir, exports)
        summary["soak"] = {"duration_s": duration, "window_s": window,
                           "windows": soak_windows(result, window)}
    elif mode == "ladder":
        ladder = cfg.get("ladder") or {}
        levels = [int(level) for level in ladder.get("levels", [1, 2, 4, 8, 16, 32, 64])]
        requests_per_level = ladder.get("requests_per_concurrency")
        rules = ladder_rules(cfg)

        async def run_level(concurrency: int) -> dict[str, Any]:
            total = (
                int(requests_per_level * concurrency) if requests_per_level
                else base.total_requests
            )
            level_cfg = replace(
                base, concurrency=concurrency, total_requests=total,
                warmup_requests=min(base.warmup_requests, concurrency),
            )
            level_dir = out_dir / f"c{concurrency:04d}"
            result = await run_fanout(level_cfg, endpoints, schedule_dir=level_dir)
            write_run_outputs(result, level_dir, exports)
            return result.get_summary()

        summary["ladder"] = await run_ladder(levels, rules, run_level)
    (out_dir / "lab_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Non-interactive benchmark CLI driven by a TOML or JSON config file."
    )
    parser.add_argument("--config", required=True, help="TOML or JSON config file")
    parser.add_argument("--mode", choices=MODES, help="override the config's mode")
    parser.add_argument("--output-dir", help="override output_dir")
    parser.add_argument("--concurrency", type=int, help="override concurrency")
    parser.add_argument("--repeats", type=int, help="override repeats")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate the config and print the resolved plan without sending requests")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.mode:
        cfg["mode"] = args.mode
    if args.concurrency:
        cfg["concurrency"] = args.concurrency
    if args.repeats:
        cfg["repeats"] = args.repeats
    out_root = Path(args.output_dir or cfg.get("output_dir", "outputs/lab"))
    if args.dry_run:
        base, endpoints = build_base_config(cfg)
        console.print_json(json.dumps({
            "mode": cfg.get("mode", "fixed"),
            "config": base.snapshot(),
            "endpoints": [e.label for e in endpoints],
        }, default=str))
        return 0
    summary = asyncio.run(run_from_config(cfg, out_root))
    console.print_json(json.dumps(summary, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
