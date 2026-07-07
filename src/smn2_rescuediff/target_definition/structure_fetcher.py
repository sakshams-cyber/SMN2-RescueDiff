"""Optional RCSB PDB structure download helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import requests


def fetch_structures(config: dict[str, Any], output_dir: str | Path) -> pd.DataFrame:
    structure_cfg = config.get("structures", {})
    pdb_ids = structure_cfg.get("pdb_ids", []) or []
    fetch_enabled = bool(structure_cfg.get("fetch_pdb", False))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    rows = []

    for pdb_id in pdb_ids:
        pdb_id = str(pdb_id).upper()
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        destination = Path(output_dir) / f"{pdb_id}.cif"
        if not fetch_enabled:
            rows.append({"pdb_id": pdb_id, "download_status": "skipped", "file_path": "", "source_url": url, "notes": "fetch_pdb is false"})
            continue
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            destination.write_bytes(response.content)
            rows.append({"pdb_id": pdb_id, "download_status": "downloaded", "file_path": str(destination), "source_url": url, "notes": ""})
        except Exception as exc:  # noqa: BLE001 - failures should be reported but non-fatal.
            rows.append({"pdb_id": pdb_id, "download_status": "failed", "file_path": "", "source_url": url, "notes": str(exc)})
    return pd.DataFrame(rows, columns=["pdb_id", "download_status", "file_path", "source_url", "notes"])
