"""Streamlit explorer for Module 5 final SRSI candidates."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FINAL_TABLE = ROOT / "data" / "processed" / "final_scoring" / "tables" / "final_srsi_ranked_candidates.tsv"
RESCUE_TABLE = ROOT / "data" / "processed" / "final_scoring" / "tables" / "splicing_rescue_scores.tsv"
EVIDENCE_JSON = ROOT / "data" / "processed" / "final_scoring" / "evidence" / "top_candidate_evidence.json"


def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def _molecule_image(smiles: str):
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw

        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        return Draw.MolToImage(mol, size=(320, 240))
    except Exception:
        return None


def main() -> None:
    try:
        import streamlit as st
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise SystemExit("Streamlit is not installed. Install it with: python3 -m pip install streamlit") from exc

    st.set_page_config(page_title="SMN2-RescueDiff Final Candidate Explorer", layout="wide")
    st.title("SMN2-RescueDiff Final Candidate Explorer")
    final = _load_table(FINAL_TABLE)
    rescue = _load_table(RESCUE_TABLE)
    evidence = json.loads(EVIDENCE_JSON.read_text(encoding="utf-8")) if EVIDENCE_JSON.exists() else {}

    if final.empty:
        st.warning("Final SRSI candidate table is missing or empty. Run Module 5 first.")
        st.code("PYTHONPATH=src python3 -m smn2_rescuediff.splicing_rescue.cli --config configs/splicing_rescue_scoring.yaml")
        return

    if final.get("is_dry_run_input", pd.Series(dtype=bool)).astype(str).str.lower().isin({"true", "1"}).any():
        st.error("DRY-RUN / SIMULATED DOCKING INPUTS - NOT SCIENTIFIC RESULTS")

    tiers = ["all", *sorted(final.get("candidate_tier", pd.Series(dtype=str)).dropna().astype(str).unique())]
    tier = st.sidebar.selectbox("Candidate tier", tiers)
    query = st.sidebar.text_input("Molecule ID search", "")
    view = final.copy()
    if tier != "all":
        view = view[view["candidate_tier"].astype(str) == tier]
    if query:
        view = view[view["molecule_id"].astype(str).str.contains(query, case=False, na=False)]

    st.subheader("Top Candidate Leaderboard")
    leaderboard_cols = [
        "final_rank",
        "molecule_id",
        "target_deltaG",
        "weighted_selectivity_gap",
        "off_target_risk_score",
        "predicted_splicing_rescue_score",
        "srsi_score",
        "candidate_tier",
        "confidence_level",
        "recommended_next_step",
    ]
    st.dataframe(view[[column for column in leaderboard_cols if column in view.columns]], use_container_width=True)

    selected_id = st.selectbox("Inspect molecule", view["molecule_id"].astype(str).tolist() if not view.empty else final["molecule_id"].astype(str).tolist())
    selected = final[final["molecule_id"].astype(str) == selected_id].iloc[0]
    left, right = st.columns([1, 2])
    with left:
        st.markdown(f"### {selected_id}")
        st.write(f"Tier: `{selected.get('candidate_tier', 'NA')}`")
        st.write(f"Confidence: `{selected.get('confidence_level', 'NA')}`")
        img = _molecule_image(str(selected.get("canonical_smiles", "")))
        if img is not None:
            st.image(img, caption="2D depiction from RDKit")
        else:
            st.code(str(selected.get("canonical_smiles", "NA")))
    with right:
        score_cols = [
            "target_deltaG",
            "target_binding_strength_score",
            "weighted_selectivity_gap",
            "off_target_risk_score",
            "predicted_splicing_rescue_score",
            "splice_site_occlusion_penalty",
            "qed",
            "pre_docking_candidate_score",
            "srsi_score",
            "srsi_percentile",
        ]
        st.subheader("Score Breakdown")
        st.json({column: selected.get(column, "NA") for column in score_cols if column in selected})

    st.subheader("Splicing-Rescue Score Details")
    detail = rescue[rescue["molecule_id"].astype(str) == selected_id] if not rescue.empty and "molecule_id" in rescue else pd.DataFrame()
    if not detail.empty:
        st.dataframe(detail, use_container_width=True)
    else:
        st.info("No detailed splicing-rescue row available for this molecule.")

    st.subheader("Evidence Summary")
    st.json(evidence.get(selected_id, {"message": "No top-candidate evidence trace was written for this molecule."}))

    st.subheader("Limitations")
    st.markdown(
        """
- Docking scores are computational predictions.
- Predicted splicing-rescue scores are proxy scores, not measured exon 7 inclusion.
- SRSI is a prioritization metric, not proof of biological activity.
- Dry-run results are simulated pipeline tests and are not scientific results.
"""
    )


if __name__ == "__main__":
    main()

