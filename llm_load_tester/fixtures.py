from __future__ import annotations

"""Fixed request fixtures loaded from JSONL.

Each line is one fixture with a stable ``fixture_id`` and either a complete
``payload`` (sent as-is apart from ``model``/``stream``) or chat fields:

    {"fixture_id": "chat-001", "messages": [...], "max_tokens": 256}
    {"fixture_id": "chat-002", "prompt": "Summarise ...", "max_tokens": 128}

Optional ``prompt_tokens`` is recorded as the request's input token count
until the server's usage replaces it.
"""

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from .modalities import ModalityHandler, PayloadResult


def payload_fingerprint(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "gen-" + hashlib.sha256(blob).hexdigest()[:16]


class FixtureHandler(ModalityHandler):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.fixtures: list[dict[str, Any]] = []
        with open(self.path) as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "fixture_id" not in record:
                    raise ValueError(f"{self.path}:{line_number} lacks fixture_id")
                self.fixtures.append(record)
        if not self.fixtures:
            raise ValueError(f"fixture file {self.path} is empty")
        self.by_id = {str(record["fixture_id"]): record for record in self.fixtures}
        self._index = 0
        self._lock: asyncio.Lock | None = None

    def manifest_sha256(self) -> str:
        """Hash of the parsed fixtures, independent of file formatting."""
        blob = json.dumps(self.fixtures, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()

    def get_presets(self) -> dict[str, Any]:
        return {}

    def _build(self, record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        if isinstance(record.get("payload"), dict):
            payload = dict(record["payload"])
        else:
            messages = record.get("messages")
            if messages is None:
                messages = [{"role": "user", "content": record.get("prompt", "")}]
            payload = {"messages": messages}
            for key in ("max_tokens", "max_completion_tokens", "temperature", "top_p",
                        "ignore_eos", "seed", "stop"):
                if key in record:
                    payload[key] = record[key]
        payload["model"] = config.get("model", payload.get("model"))
        payload["stream"] = True
        return payload

    async def prepare_payload(self, config: dict[str, Any]) -> PayloadResult:
        requested = config.get("fixture_id")
        if requested is not None and str(requested) in self.by_id:
            record = self.by_id[str(requested)]
        else:
            if self._lock is None:
                self._lock = asyncio.Lock()
            async with self._lock:
                record = self.fixtures[self._index % len(self.fixtures)]
                self._index += 1
        return PayloadResult(
            payload=self._build(record, config),
            metadata={
                "fixture_id": str(record["fixture_id"]),
                "input_tokens": record.get("prompt_tokens", 0),
                "modality": "fixture",
            },
        )
