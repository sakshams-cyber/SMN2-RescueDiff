"""Live simulation state helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_STATE = {
    "run_status": "not_started",
    "run_mode": "dry_run",
    "current_molecule_id": None,
    "current_smiles": None,
    "current_target_id": None,
    "current_target_type": None,
    "current_decoy_category": None,
    "jobs_completed": 0,
    "total_jobs": 0,
    "percent_complete": 0.0,
    "latest_score": None,
    "best_target_score_so_far": None,
    "best_selectivity_gap_so_far": None,
    "last_message": "",
}


def write_state(path: str | Path, updates: dict[str, Any]) -> dict[str, Any]:
    """Merge and write current run state."""
    state = DEFAULT_STATE.copy()
    p = Path(path)
    if p.exists():
        try:
            state.update(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    state.update(updates)
    total = max(int(state.get("total_jobs") or 0), 0)
    completed = max(int(state.get("jobs_completed") or 0), 0)
    state["percent_complete"] = round(100.0 * completed / total, 3) if total else 0.0
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state

