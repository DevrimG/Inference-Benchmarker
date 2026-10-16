from __future__ import annotations

"""Arrival schedules, generated in full before a run starts.

Modes:

* ``closed``  – C workers, each sends its next request when the previous one
  finishes (optionally capped by the legacy ``target_rps`` token bucket).
  There is no precomputed schedule.
* ``poisson`` – open loop, exponential inter-arrival times at ``rate``.
* ``gamma``   – open loop, gamma inter-arrival times at ``rate`` with shape
  ``shape`` (``shape < 1`` is burstier than Poisson, ``> 1`` smoother); this
  matches ``vllm bench serve --burstiness``.
* ``burst``   – exactly C requests released at the same instant.
* ``replay``  – offsets read from a JSONL trace. Each line carries
  ``timestamp`` (milliseconds, the AIPerf/Mooncake trace convention) or
  ``offset_s`` (seconds), and optionally ``fixture_id``.
"""

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ARRIVAL_MODES = ("closed", "poisson", "gamma", "burst", "replay")
OPEN_LOOP_MODES = ("poisson", "gamma", "burst", "replay")


@dataclass
class ArrivalSchedule:
    mode: str
    offsets: list[float] = field(default_factory=list)
    fixture_ids: list[str | None] = field(default_factory=list)
    rate: float | None = None
    shape: float | None = None
    seed: int | None = None
    source: str | None = None

    @property
    def open_loop(self) -> bool:
        return self.mode in OPEN_LOOP_MODES

    def __len__(self) -> int:
        return len(self.offsets)

    def digest(self) -> str:
        blob = json.dumps(
            {"mode": self.mode, "offsets": [round(o, 9) for o in self.offsets],
             "fixture_ids": self.fixture_ids},
            sort_keys=True,
        ).encode()
        return hashlib.sha256(blob).hexdigest()

    def to_dict(self, include_offsets: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "mode": self.mode,
            "requests": len(self.offsets) if self.open_loop else None,
            "rate": self.rate,
            "shape": self.shape,
            "seed": self.seed,
            "source": self.source,
            "duration_seconds": self.offsets[-1] if self.offsets else None,
            "sha256": self.digest() if self.open_loop else None,
        }
        if include_offsets:
            data["offsets"] = self.offsets
        return data

    def write_jsonl(self, path: Path) -> None:
        """Log the schedule; the file is itself a valid replay trace."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            header = {"schedule": self.to_dict()}
            handle.write(json.dumps(header) + "\n")
            for index, offset in enumerate(self.offsets):
                row: dict[str, Any] = {"index": index, "offset_s": offset}
                fixture_id = self.fixture_ids[index] if self.fixture_ids else None
                if fixture_id is not None:
                    row["fixture_id"] = fixture_id
                handle.write(json.dumps(row) + "\n")


def _interarrivals(rng: np.random.Generator, n: int, rate: float, shape: float) -> np.ndarray:
    # Gamma(k, θ) with θ = 1 / (k·rate) has mean 1 / rate for any shape k.
    return rng.gamma(shape, 1.0 / (shape * rate), size=n)


def load_replay(path: Path) -> ArrivalSchedule:
    offsets: list[float] = []
    fixture_ids: list[str | None] = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "schedule" in row and len(row) == 1:
                continue  # header written by write_jsonl
            if "offset_s" in row:
                offsets.append(float(row["offset_s"]))
            elif "timestamp" in row:
                offsets.append(float(row["timestamp"]) / 1000.0)
            else:
                raise ValueError(f"replay row lacks 'timestamp' or 'offset_s': {row}")
            fixture_ids.append(row.get("fixture_id"))
    if not offsets:
        raise ValueError(f"replay file {path} has no arrivals")
    order = sorted(range(len(offsets)), key=offsets.__getitem__)
    base = offsets[order[0]]
    return ArrivalSchedule(
        mode="replay",
        offsets=[offsets[i] - base for i in order],
        fixture_ids=[fixture_ids[i] for i in order],
        source=str(path),
    )


def build_schedule(
    mode: str,
    *,
    requests: int = 0,
    concurrency: int = 1,
    rate: float | None = None,
    shape: float = 1.0,
    seed: int | None = None,
    duration_seconds: float | None = None,
    replay_file: str | Path | None = None,
) -> ArrivalSchedule:
    """Build the full schedule up front so it can be logged and replayed."""
    if mode not in ARRIVAL_MODES:
        raise ValueError(f"arrival mode must be one of {ARRIVAL_MODES}, got {mode!r}")
    if mode == "closed":
        return ArrivalSchedule(mode="closed", seed=seed)
    if mode == "burst":
        return ArrivalSchedule(mode="burst", offsets=[0.0] * concurrency, seed=seed)
    if mode == "replay":
        if not replay_file:
            raise ValueError("replay mode requires replay_file")
        schedule = load_replay(Path(replay_file))
        if requests and requests < len(schedule):
            schedule.offsets = schedule.offsets[:requests]
            schedule.fixture_ids = schedule.fixture_ids[:requests]
        return schedule

    if not rate or rate <= 0:
        raise ValueError(f"{mode} arrivals require a positive rate")
    effective_shape = 1.0 if mode == "poisson" else float(shape)
    if effective_shape <= 0:
        raise ValueError("gamma shape must be positive")
    rng = np.random.default_rng(seed)
    if duration_seconds:
        # Draw in blocks until the schedule covers the duration.
        offsets: list[float] = []
        at = 0.0
        block = max(16, int(math.ceil(rate * duration_seconds * 1.2)))
        while True:
            gaps = _interarrivals(rng, block, rate, effective_shape)
            for gap in gaps:
                if offsets:
                    at += float(gap)
                if at > duration_seconds:
                    break
                offsets.append(at)
            else:
                continue
            break
    else:
        if requests <= 0:
            raise ValueError("open-loop arrivals need requests or duration_seconds")
        gaps = _interarrivals(rng, requests, rate, effective_shape)
        gaps[0] = 0.0  # first request at t=0
        offsets = np.cumsum(gaps).tolist()
    return ArrivalSchedule(
        mode=mode, offsets=offsets, rate=rate, shape=effective_shape, seed=seed
    )
