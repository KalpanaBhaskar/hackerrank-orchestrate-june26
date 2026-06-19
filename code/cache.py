"""
Cache module — disk-based JSON cache for L2 and L3 responses.
Key = sha256(claim_text + sorted image file hashes + prompt version).
Prevents repeated API calls during iteration.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

CACHE_VERSION = "v1"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / ".cache"


class DiskCache:
    def __init__(self, cache_dir: str | Path | None = None):
        env_dir = os.environ.get("CACHE_DIR")
        if cache_dir:
            self.dir = Path(cache_dir)
        elif env_dir:
            self.dir = Path(env_dir)
        else:
            self.dir = DEFAULT_CACHE_DIR
        self.dir.mkdir(parents=True, exist_ok=True)

    def _key_path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> Optional[Any]:
        p = self._key_path(key)
        if p.exists():
            try:
                with open(p, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def set(self, key: str, value: Any) -> None:
        p = self._key_path(key)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)

    def make_key(self, *parts: str) -> str:
        combined = CACHE_VERSION + "|".join(str(p) for p in parts)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def hash_file(path: Path) -> str:
    """SHA-256 of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_images(image_paths: list[Path]) -> str:
    """Stable hash of a sorted list of image files."""
    hashes = sorted(hash_file(p) for p in image_paths if p.exists())
    return hashlib.sha256("|".join(hashes).encode()).hexdigest()
