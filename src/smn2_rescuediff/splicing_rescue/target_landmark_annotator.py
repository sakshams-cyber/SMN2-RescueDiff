"""Infer SMN2 exon 7 splicing landmarks from Module 1 target outputs."""

from __future__ import annotations

from typing import Any

import pandas as pd


LANDMARK_COLUMNS = [
    "landmark_id",
    "landmark_name",
    "window_id",
    "sequence_rna",
    "start_relative_to_exon7",
    "end_relative_to_exon7",
    "landmark_type",
    "is_beneficial_target_region",
    "is_caution_region",
    "biological_reason",
    "notes",
]


def _infer_type(window_name: str, row: pd.Series) -> tuple[str, str]:
    name = window_name.lower()
    if "five_prime" in name or "5prime" in name or "donor" in name:
        if "core" in name:
            return "core_donor_region", "Core 5 prime splice-site window; useful but sensitive to occlusion."
        return "five_prime_splice_site", "5 prime splice-site-adjacent region relevant to U1-associated splice-site recognition."
    if "three_prime" in name or "3prime" in name or "acceptor" in name:
        if "core" in name:
            return "core_acceptor_region", "Core 3 prime splice-site window; useful but sensitive to occlusion."
        return "three_prime_splice_site", "3 prime splice-site-adjacent region relevant to exon definition."
    if "u1" in name:
        return "u1_snRNA_recognition_region", "Region annotated as U1-recognition related in the target profile."
    if "enhancer" in name or "ese" in name:
        return "enhancer_like_region", "Candidate enhancer-like target window from Module 1 feature annotation."
    if "silencer" in name or "ess" in name:
        return "silencer_like_region", "Candidate silencer-like target window from Module 1 feature annotation."
    if str(row.get("window_type", "")).lower() == "exon" or "exon7" in name or "exon_7" in name:
        return "exon_7", "Annotated SMN2 exon 7 RNA window."
    if bool(row.get("contains_splice_boundary", False)):
        return "exon_intronic_boundary", "Window overlaps a splice boundary according to Module 1."
    return "NA", "Landmark type could not be confidently inferred from available Module 1 fields."


def _relative_value(row: pd.Series, key: str) -> Any:
    value = row.get(key, "NA")
    if pd.isna(value):
        return "NA"
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def annotate_landmarks(target_profile: pd.DataFrame, target_metadata: dict[str, Any], config: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    """Create a conservative target landmark table from Module 1 features."""
    warnings: list[str] = []
    landmark_cfg = config.get("splicing_landmarks", {})
    beneficial = set(landmark_cfg.get("beneficial_landmarks", []))
    caution = set(landmark_cfg.get("caution_landmarks", []))
    rows: list[dict[str, Any]] = []
    if target_profile.empty:
        warnings.append("Module 1 target profile is empty; no landmarks could be inferred.")
        return pd.DataFrame(columns=LANDMARK_COLUMNS), warnings
    for idx, row in target_profile.reset_index(drop=True).iterrows():
        window_name = str(row.get("window_name", row.get("target_id", f"window_{idx + 1}")))
        landmark_type, reason = _infer_type(window_name, row)
        if landmark_type == "NA":
            warnings.append(f"Could not confidently infer landmark type for target window: {window_name}")
        landmark_name = landmark_type
        if landmark_type == "core_donor_region":
            landmark_name = "core_donor_dinucleotide"
        elif landmark_type == "core_acceptor_region":
            landmark_name = "core_acceptor_dinucleotide"
        elif landmark_type == "enhancer_like_region":
            landmark_name = "exonic_splicing_enhancer_like_region"
        elif landmark_type == "silencer_like_region":
            landmark_name = "exonic_splicing_silencer_like_region"
        rows.append(
            {
                "landmark_id": f"SMN2_LANDMARK_{idx + 1:03d}",
                "landmark_name": landmark_name,
                "window_id": row.get("target_id", window_name),
                "sequence_rna": row.get("sequence_rna", "NA"),
                "start_relative_to_exon7": _relative_value(row, "distance_to_exon7_start"),
                "end_relative_to_exon7": _relative_value(row, "distance_to_exon7_end"),
                "landmark_type": landmark_type,
                "is_beneficial_target_region": landmark_name in beneficial or landmark_type in beneficial,
                "is_caution_region": landmark_name in caution or landmark_type in caution,
                "biological_reason": reason,
                "notes": "Coordinates are relative/annotated from Module 1 when available; no exact biological coordinate was invented.",
            }
        )
    # Add a U1-support landmark when the 5 prime splice-site core exists but no
    # explicit U1 row is present. This is labeled conservatively as inferred.
    has_u1 = any(str(item.get("landmark_name", "")).lower() == "u1_snrna_recognition_region" for item in rows)
    donor_rows = [item for item in rows if item["landmark_type"] in {"five_prime_splice_site", "core_donor_region"}]
    if not has_u1 and donor_rows:
        donor = donor_rows[0]
        rows.append(
            {
                "landmark_id": f"SMN2_LANDMARK_{len(rows) + 1:03d}",
                "landmark_name": "u1_snRNA_recognition_region",
                "window_id": donor["window_id"],
                "sequence_rna": donor["sequence_rna"],
                "start_relative_to_exon7": donor["start_relative_to_exon7"],
                "end_relative_to_exon7": donor["end_relative_to_exon7"],
                "landmark_type": "u1_snRNA_recognition_region",
                "is_beneficial_target_region": True,
                "is_caution_region": False,
                "biological_reason": "Inferred from 5 prime splice-site target window as a U1-associated recognition support region.",
                "notes": "Inferred label only; exact U1-contact coordinates require structural/experimental annotation.",
            }
        )
        warnings.append("U1 support landmark was inferred from the 5 prime splice-site window; exact U1 coordinates were not available.")
    if not rows:
        warnings.append(f"No usable landmarks inferred for transcript {target_metadata.get('selected_transcript_id', 'NA')}.")
    return pd.DataFrame(rows, columns=LANDMARK_COLUMNS), warnings

