"""
Telemetry — wraps every model call to log token usage, latency, and image count.
Writes to a local JSONL file for later operational analysis.
"""

import json
import time
from pathlib import Path
from typing import Any, Optional

TELEMETRY_FILE = Path(__file__).resolve().parent / "telemetry.jsonl"


def log_call(
    stage: str,         # "L2" or "L3"
    claim_id: str,
    tokens_in: int,
    tokens_out: int,
    image_count: int,
    latency_ms: float,
    cached: bool = False,
    model: str = "",
    extra: Optional[dict] = None,
) -> None:
    entry = {
        "stage": stage,
        "claim_id": claim_id,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "image_count": image_count,
        "latency_ms": round(latency_ms, 1),
        "cached": cached,
        "model": model,
        **(extra or {}),
    }
    with open(TELEMETRY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_telemetry() -> list[dict]:
    if not TELEMETRY_FILE.exists():
        return []
    entries = []
    with open(TELEMETRY_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


def summarize_telemetry() -> dict[str, Any]:
    entries = load_telemetry()
    if not entries:
        return {}
    total_calls = len(entries)
    cached = sum(1 for e in entries if e.get("cached"))
    uncached = total_calls - cached
    total_tokens_in = sum(e.get("tokens_in", 0) for e in entries)
    total_tokens_out = sum(e.get("tokens_out", 0) for e in entries)
    total_images = sum(e.get("image_count", 0) for e in entries)
    total_latency = sum(e.get("latency_ms", 0) for e in entries if not e.get("cached"))
    avg_latency = total_latency / uncached if uncached > 0 else 0

    by_stage: dict[str, Any] = {}
    for e in entries:
        s = e.get("stage", "unknown")
        if s not in by_stage:
            by_stage[s] = {"calls": 0, "tokens_in": 0, "tokens_out": 0, "images": 0}
        by_stage[s]["calls"] += 1
        by_stage[s]["tokens_in"] += e.get("tokens_in", 0)
        by_stage[s]["tokens_out"] += e.get("tokens_out", 0)
        by_stage[s]["images"] += e.get("image_count", 0)

    return {
        "total_calls": total_calls,
        "cached_calls": cached,
        "uncached_calls": uncached,
        "total_tokens_in": total_tokens_in,
        "total_tokens_out": total_tokens_out,
        "total_images_processed": total_images,
        "avg_latency_ms_uncached": round(avg_latency, 1),
        "by_stage": by_stage,
    }
