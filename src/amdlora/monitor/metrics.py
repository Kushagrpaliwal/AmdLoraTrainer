"""Step 10: write training numbers to metrics.jsonl (one JSON object per line)."""
from __future__ import annotations
import json
import time
from dataclasses import asdict
from pathlib import Path
import yaml


class MetricsLogger:
    """Every log() call writes one line and flushes it, so the file can be read while
    training runs and stays valid if training crashes. A new run starts a fresh file."""

    def __init__(self, out_dir, filename: str = "metrics.jsonl"):
        self.path = Path(out_dir) / filename
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", encoding="utf-8")

    def log(self, **fields) -> None:
        row = {"time": round(time.time(), 3), **fields}
        self._file.write(json.dumps(row) + "\n")
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()


def save_config_copy(cfg, out_dir) -> Path:
    """Saves the exact settings of this run next to its outputs (loadable with load_config)."""
    path = Path(out_dir) / "config_used.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False, allow_unicode=True)
    return path
