from __future__ import annotations

"""Run manifest: everything needed to say which numbers came from what."""

import hashlib
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp

from . import __version__

_REPO_ROOT = Path(__file__).resolve().parent.parent


def sha256_file(path: str | Path | None) -> str | None:
    if not path:
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tool_git_state() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(_REPO_ROOT), *args],
                capture_output=True, text=True, timeout=5, check=True,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip()

    status = git("status", "--porcelain", "--untracked-files=no")
    return {
        "sha": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def client_host() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "aiohttp": aiohttp.__version__,
        "cpu_count": os.cpu_count(),
    }


async def _get_json(session: aiohttp.ClientSession, url: str, headers: dict[str, str]) -> Any:
    try:
        timeout = aiohttp.ClientTimeout(total=5)
        async with session.get(url, headers=headers, timeout=timeout) as response:
            if response.status != 200:
                return None
            return await response.json(content_type=None)
    except Exception:
        return None


async def probe_server(
    session: aiohttp.ClientSession, endpoint: str, api_key: str | None
) -> dict[str, Any]:
    """Best-effort served model and server version (vLLM exposes both)."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    base = endpoint.rstrip("/")
    models = await _get_json(session, f"{base}/v1/models", headers)
    version = await _get_json(session, f"{base}/version", headers)
    served = None
    if isinstance(models, dict) and isinstance(models.get("data"), list):
        served = [m.get("id") for m in models["data"] if isinstance(m, dict)]
    return {
        "served_models": served,
        "server_version": version.get("version") if isinstance(version, dict) else None,
    }


def build_manifest(
    *,
    endpoint: str,
    api_route: str,
    model: str,
    server_probe: dict[str, Any] | None,
    schedule: dict[str, Any] | None,
    fixture_file: str | None,
    fixture_manifest_sha256: str | None,
    tokenizer: str | None,
    image_digest: str | None,
    config_snapshot: dict[str, Any],
    endpoint_label: str | None = None,
) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tool": {"name": "llm_load_tester", "version": __version__, **tool_git_state()},
        "endpoint": endpoint,
        "endpoint_label": endpoint_label,
        "api_route": api_route,
        "requested_model": model,
        "served_models": (server_probe or {}).get("served_models"),
        "server_version": (server_probe or {}).get("server_version"),
        "image_digest": image_digest or os.environ.get("BENCH_IMAGE_DIGEST"),
        "tokenizer": tokenizer,
        "fixture_file": fixture_file,
        "fixture_file_sha256": sha256_file(fixture_file),
        "fixture_manifest_sha256": fixture_manifest_sha256,
        "arrival_schedule": schedule,
        "client_host": client_host(),
        "config": config_snapshot,
    }
