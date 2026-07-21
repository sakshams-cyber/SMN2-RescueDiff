"""Common docking result helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DockingResult:
    """Normalized docking result used by real and simulated engines."""

    status: str
    delta_g: float | str
    best_pose_rank: int | str
    num_poses: int | str
    pose_file: str
    engine: str
    run_mode: str
    is_dry_run_score: bool
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "dock_status": self.status,
            "delta_g": self.delta_g,
            "best_pose_rank": self.best_pose_rank,
            "num_poses": self.num_poses,
            "pose_file": self.pose_file,
            "engine": self.engine,
            "run_mode": self.run_mode,
            "is_dry_run_score": self.is_dry_run_score,
            "notes": self.notes,
        }

