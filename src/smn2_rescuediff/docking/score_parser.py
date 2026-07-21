"""Parse docking engine score output."""

from __future__ import annotations

import re


def parse_vina_best_score(stdout: str) -> float | None:
    """Parse the best affinity from AutoDock Vina stdout."""
    pattern = re.compile(r"^\s*1\s+(-?\d+(?:\.\d+)?)\s+", re.MULTILINE)
    match = pattern.search(stdout)
    if match:
        return float(match.group(1))
    fallback = re.search(r"Affinity:\s*(-?\d+(?:\.\d+)?)", stdout, re.IGNORECASE)
    if fallback:
        return float(fallback.group(1))
    return None

