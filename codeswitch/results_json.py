"""Serialize experiment metrics to JSON (numpy-safe)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def to_json_serializable(obj: Any) -> Any:
    """Recursively convert numpy scalars / arrays and other non-JSON types."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (float, int)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_serializable(v) for v in obj]
    if hasattr(obj, "item") and callable(obj.item) and hasattr(obj, "shape") and obj.shape == ():
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def save_results_json(path: str | Path, payload: dict[str, Any], *, indent: int = 2) -> None:
    """Write ``payload`` to ``path`` as UTF-8 JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(
            to_json_serializable(payload),
            f,
            indent=indent,
            ensure_ascii=False,
        )
