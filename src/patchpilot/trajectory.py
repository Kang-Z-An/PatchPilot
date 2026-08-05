from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


class TrajectoryWriter:
    def __init__(self, directory: Path, metadata: dict[str, Any]) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = directory / f"{stamp}-{uuid4().hex[:8]}.jsonl"
        self._write({"event": "run_started", **metadata})

    def record(self, event: str, **payload: Any) -> None:
        self._write({"event": event, **payload})

    def _write(self, value: dict[str, Any]) -> None:
        envelope = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **value,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(envelope, ensure_ascii=False) + "\n")
