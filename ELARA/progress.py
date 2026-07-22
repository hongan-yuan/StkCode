from __future__ import annotations

import json
import os
import time
from pathlib import Path


class ProgressReporter:
    """Atomically publish progress for an external launcher."""

    def __init__(self, path: Path | None, total: int, unit: str = "episodes"):
        self.path = path
        self.total = max(0, int(total))
        self.unit = str(unit)
        self.started_at = time.time()

    def update(
        self,
        completed: int,
        status: str = "running",
        item_count: int | None = None,
    ) -> None:
        if self.path is None:
            return
        completed = min(max(0, int(completed)), self.total)
        now = time.time()
        elapsed = max(1.0e-9, now - self.started_at)
        rate = completed / elapsed if completed > 0 and elapsed > 0.0 else 0.0
        eta = (self.total - completed) / rate if rate > 0.0 else None
        payload = {
            "completed": completed,
            "total": self.total,
            "unit": self.unit,
            "item_count": item_count,
            "fraction": completed / max(1, self.total),
            "elapsed_s": elapsed,
            "eta_s": eta,
            "status": status,
            "updated_at": now,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(temporary, self.path)
