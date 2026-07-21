"""Pose/contact proxy analysis for Module 5."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


CONTACT_COLUMNS = [
    "molecule_id",
    "num_close_contacts",
    "num_contacts_with_target_region",
    "num_contacts_with_beneficial_landmarks",
    "num_contacts_with_caution_landmarks",
    "minimum_distance_to_u1_region",
    "minimum_distance_to_5prime_splice_site",
    "minimum_distance_to_3prime_splice_site",
    "minimum_distance_to_core_donor_region",
    "minimum_distance_to_core_acceptor_region",
    "hydrogen_bond_proxy_count",
    "stacking_proxy_count",
    "pose_support_score",
    "splice_site_occlusion_penalty",
    "pose_analysis_status",
    "notes",
]


def _atom_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    return math.dist((float(a["x"]), float(a["y"]), float(a["z"])), (float(b["x"]), float(b["y"]), float(b["z"])))


def _is_hbond_like(atom_name: str) -> bool:
    name = atom_name.upper()
    return any(name.startswith(prefix) for prefix in ["N", "O", "P", "S"])


def _is_stack_like(atom_name: str) -> bool:
    return atom_name.upper().startswith("C")


def _landmark_presence(landmarks: pd.DataFrame, token: str) -> bool:
    if landmarks.empty:
        return False
    values = " ".join(
        landmarks.get(column, pd.Series(dtype=str)).astype(str).str.lower().str.cat(sep=" ")
        for column in ["landmark_name", "landmark_type", "window_id"]
        if column in landmarks
    )
    return token.lower() in values


def _missing_row(molecule_id: str, status: str, notes: str) -> dict[str, Any]:
    return {
        "molecule_id": molecule_id,
        "num_close_contacts": 0,
        "num_contacts_with_target_region": 0,
        "num_contacts_with_beneficial_landmarks": 0,
        "num_contacts_with_caution_landmarks": 0,
        "minimum_distance_to_u1_region": "NA",
        "minimum_distance_to_5prime_splice_site": "NA",
        "minimum_distance_to_3prime_splice_site": "NA",
        "minimum_distance_to_core_donor_region": "NA",
        "minimum_distance_to_core_acceptor_region": "NA",
        "hydrogen_bond_proxy_count": 0,
        "stacking_proxy_count": 0,
        "pose_support_score": 0.0,
        "splice_site_occlusion_penalty": 0.0,
        "pose_analysis_status": status,
        "notes": notes,
    }


def analyze_contacts(
    candidates: pd.DataFrame,
    pose_parse: pd.DataFrame,
    atom_cache: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]],
    landmarks: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Calculate pose-based contact proxy features."""
    pose_cfg = config.get("pose_analysis", {})
    contact_cutoff = float(pose_cfg.get("contact_distance_angstrom", 4.5))
    hbond_cutoff = float(pose_cfg.get("hydrogen_bond_distance_angstrom", 3.5))
    stacking_cutoff = float(pose_cfg.get("stacking_distance_angstrom", 5.0))
    rows: list[dict[str, Any]] = []
    parse_by_id = pose_parse.set_index("molecule_id").to_dict("index") if not pose_parse.empty else {}
    has_u1 = _landmark_presence(landmarks, "u1")
    has_five = _landmark_presence(landmarks, "five_prime") or _landmark_presence(landmarks, "5prime")
    has_three = _landmark_presence(landmarks, "three_prime") or _landmark_presence(landmarks, "3prime")
    has_donor = _landmark_presence(landmarks, "donor")
    has_acceptor = _landmark_presence(landmarks, "acceptor")

    for _, candidate in candidates.iterrows():
        molecule_id = str(candidate.get("molecule_id", "NA"))
        parsed = parse_by_id.get(molecule_id, {})
        status = str(parsed.get("pose_parse_status", "missing_pose_feature_based_fallback"))
        ligand_atoms, receptor_atoms = atom_cache.get(molecule_id, ([], []))
        if not ligand_atoms:
            rows.append(_missing_row(molecule_id, status, "No parseable ligand pose; using feature-based fallback."))
            continue
        if not receptor_atoms:
            rows.append(
                _missing_row(
                    molecule_id,
                    "parsed_ligand_no_receptor_feature_based_fallback",
                    "Ligand coordinates were present but receptor atoms were unavailable; using feature-based fallback.",
                )
            )
            continue

        distances: list[float] = []
        close_contacts = 0
        hbond_proxy = 0
        stacking_proxy = 0
        for ligand in ligand_atoms:
            for receptor in receptor_atoms:
                distance = _atom_distance(ligand, receptor)
                distances.append(distance)
                if distance <= contact_cutoff:
                    close_contacts += 1
                if distance <= hbond_cutoff and (_is_hbond_like(ligand.get("atom_name", "")) or _is_hbond_like(receptor.get("atom_name", ""))):
                    hbond_proxy += 1
                if distance <= stacking_cutoff and _is_stack_like(ligand.get("atom_name", "")) and _is_stack_like(receptor.get("atom_name", "")):
                    stacking_proxy += 1
        minimum = min(distances) if distances else None
        contact_density = min(1.0, close_contacts / max(1.0, len(ligand_atoms) * 4.0))
        hbond_score = min(1.0, hbond_proxy / max(1.0, len(ligand_atoms) * 0.5))
        stacking_score = min(1.0, stacking_proxy / max(1.0, len(ligand_atoms) * 0.5))
        beneficial_contacts = int(close_contacts * (0.75 if has_u1 or has_five else 0.4))
        caution_contacts = int(close_contacts * (0.35 if has_donor or has_acceptor else 0.1))
        pose_support = max(0.0, min(1.0, 0.45 * contact_density + 0.30 * hbond_score + 0.25 * stacking_score))
        occlusion = max(0.0, min(1.0, caution_contacts / max(1.0, close_contacts)))
        rows.append(
            {
                "molecule_id": molecule_id,
                "num_close_contacts": int(close_contacts),
                "num_contacts_with_target_region": int(close_contacts),
                "num_contacts_with_beneficial_landmarks": int(beneficial_contacts),
                "num_contacts_with_caution_landmarks": int(caution_contacts),
                "minimum_distance_to_u1_region": round(minimum, 4) if minimum is not None and has_u1 else "NA",
                "minimum_distance_to_5prime_splice_site": round(minimum, 4) if minimum is not None and has_five else "NA",
                "minimum_distance_to_3prime_splice_site": round(minimum, 4) if minimum is not None and has_three else "NA",
                "minimum_distance_to_core_donor_region": round(minimum, 4) if minimum is not None and has_donor else "NA",
                "minimum_distance_to_core_acceptor_region": round(minimum, 4) if minimum is not None and has_acceptor else "NA",
                "hydrogen_bond_proxy_count": int(hbond_proxy),
                "stacking_proxy_count": int(stacking_proxy),
                "pose_support_score": round(pose_support, 6),
                "splice_site_occlusion_penalty": round(occlusion, 6),
                "pose_analysis_status": "pose_based_proxy",
                "notes": "Contact features are geometric proxies from parsed pose/receptor coordinates.",
            }
        )
    return pd.DataFrame(rows, columns=CONTACT_COLUMNS)

