"""Shared paths and reproducible artifact utilities; no network side effects."""
from __future__ import annotations

import hashlib
import json
import os
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
TARGETS = ("target_survival_5y", "target_recurrence")
TARGET_LABELS = {
    "target_survival_5y": "Five-year mortality",
    "target_recurrence": "Locoregional recurrence",
}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    def clean(item):
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(v) for v in item]
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item
    temporary.write_text(json.dumps(clean(value), indent=2, default=str, allow_nan=False))
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text())


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:20]


def load_token() -> str:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    return os.environ.get("TABPFN_TOKEN", "").strip()


def safe_error(exc: Exception) -> str:
    """Never persist a credential accidentally included by an SDK exception."""
    message = f"{type(exc).__name__}: {exc}"
    token = os.environ.get("TABPFN_TOKEN", "")
    return message.replace(token, "[REDACTED]")[:1200] if token else message[:1200]
