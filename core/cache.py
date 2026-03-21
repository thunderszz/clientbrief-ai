"""File-based cache for ClientBrief AI research results.

Cache key:  SHA-256(domain + meeting_type + normalized_function + stakeholder_role)
Path:       .clientbrief_cache/{hash}.json
TTL:        24 hours
Stored:     Full BriefingState minus ``final_brief``
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional

# Fields that are NOT cached (they are recomposed each run).
_EXCLUDED_FIELDS: frozenset[str] = frozenset({"final_brief"})

CACHE_DIR = Path(".clientbrief_cache")
CACHE_TTL_SECONDS: int = 24 * 60 * 60  # 24 hours


def _cache_key(
    domain: str,
    meeting_type: str,
    normalized_function: str,
    stakeholder_role: str,
) -> str:
    """Compute the deterministic SHA-256 cache key."""
    payload = "|".join([
        domain.lower().strip(),
        meeting_type.lower().strip(),
        normalized_function.lower().strip(),
        stakeholder_role.lower().strip(),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def _ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def cache_read(
    domain: str,
    meeting_type: str,
    normalized_function: str,
    stakeholder_role: str,
) -> Optional[dict[str, Any]]:
    """Load a cached state dict if it exists and is within TTL.

    Returns ``None`` on miss (file absent, expired, or corrupt).
    """
    key = _cache_key(domain, meeting_type, normalized_function, stakeholder_role)
    path = _cache_path(key)

    if not path.exists():
        return None

    try:
        raw = path.read_text(encoding="utf-8")
        envelope = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return None

    written_at = envelope.get("_written_at", 0)
    if time.time() - written_at > CACHE_TTL_SECONDS:
        # Expired — remove stale file silently
        path.unlink(missing_ok=True)
        return None

    data: dict[str, Any] = envelope.get("state", {})
    return data if data else None


def cache_write(
    domain: str,
    meeting_type: str,
    normalized_function: str,
    stakeholder_role: str,
    state: dict[str, Any],
) -> None:
    """Persist ``state`` to the file cache, excluding ``final_brief``."""
    _ensure_cache_dir()

    filtered = {k: v for k, v in state.items() if k not in _EXCLUDED_FIELDS}

    key = _cache_key(domain, meeting_type, normalized_function, stakeholder_role)
    envelope = {
        "_written_at": time.time(),
        "state": filtered,
    }

    path = _cache_path(key)
    path.write_text(json.dumps(envelope, default=str, indent=2), encoding="utf-8")


def cache_clear() -> int:
    """Remove all cache files. Returns the number of files deleted."""
    if not CACHE_DIR.exists():
        return 0
    count = 0
    for f in CACHE_DIR.glob("*.json"):
        f.unlink(missing_ok=True)
        count += 1
    return count
