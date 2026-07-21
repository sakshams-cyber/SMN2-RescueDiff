"""JSONL event logging for live docking visualization."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .simulation_state import write_state


class LiveLogger:
    """Write live docking events and current state."""

    def __init__(self, log_file: str | Path, state_file: str | Path, enabled: bool = True) -> None:
        self.log_file = Path(log_file)
        self.state_file = Path(state_file)
        self.enabled = enabled
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        if self.enabled:
            self.log_file.write_text("", encoding="utf-8")

    def event(self, event_type: str, **kwargs: Any) -> None:
        """Append one JSON event."""
        if not self.enabled:
            return
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "run_mode": kwargs.pop("run_mode", None),
            "molecule_id": kwargs.pop("molecule_id", None),
            "canonical_smiles": kwargs.pop("canonical_smiles", None),
            "target_type": kwargs.pop("target_type", None),
            "target_id": kwargs.pop("target_id", None),
            "decoy_category": kwargs.pop("decoy_category", None),
            "job_index": kwargs.pop("job_index", None),
            "total_jobs": kwargs.pop("total_jobs", None),
            "status": kwargs.pop("status", None),
            "current_score": kwargs.pop("current_score", None),
            "best_target_score": kwargs.pop("best_target_score", None),
            "best_selectivity_gap": kwargs.pop("best_selectivity_gap", None),
            "message": kwargs.pop("message", ""),
        }
        payload.update(kwargs)
        with self.log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def state(self, **updates: Any) -> dict[str, Any]:
        """Update current state JSON."""
        if not self.enabled:
            return {}
        return write_state(self.state_file, updates)

