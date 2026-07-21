"""Streamlit dashboard for SMN2-RescueDiff live docking simulation.

Run from the repository root:

    streamlit run dashboard/docking_live_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    import streamlit as st
except Exception as exc:  # pragma: no cover - dashboard runtime helper.
    raise SystemExit("Streamlit is required for the dashboard. Install with: python3 -m pip install streamlit") from exc


ROOT = Path(__file__).resolve().parents[1]
SIM_DIR = ROOT / "data" / "processed" / "docking" / "simulation"
OUTPUT_DIR = ROOT / "data" / "processed" / "docking"
STATE_FILE = SIM_DIR / "current_docking_state.json"
LOG_FILE = SIM_DIR / "live_docking_events.jsonl"


def load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    return json.loads(STATE_FILE.read_text(encoding="utf-8"))


def load_events() -> pd.DataFrame:
    rows = []
    if LOG_FILE.exists():
        for line in LOG_FILE.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


def maybe_show_molecule(smiles: str | None) -> None:
    if not smiles:
        st.info("No current SMILES available yet.")
        return
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.code(smiles)
            return
        st.image(Draw.MolToImage(mol, size=(320, 220)), caption=smiles)
    except Exception:
        st.code(smiles)


def score_plot(events: pd.DataFrame) -> None:
    if events.empty or "current_score" not in events:
        st.info("No score events yet.")
        return
    scored = events[(events["event_type"] == "job_completed") & events["current_score"].notna()].copy()
    if scored.empty:
        st.info("No completed docking jobs with scores yet.")
        return
    scored["current_score"] = pd.to_numeric(scored["current_score"], errors="coerce")
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, group in scored.groupby("target_type"):
        ax.plot(range(len(group)), group["current_score"], marker="o", linestyle="-", label=label)
    ax.set_xlabel("Completed jobs")
    ax.set_ylabel("DeltaG")
    ax.set_title("Live docking scores over time")
    ax.legend()
    st.pyplot(fig)


def target_decoy_scatter() -> None:
    path = OUTPUT_DIR / "molecule_selectivity_scores.tsv"
    if not path.exists():
        st.info("Selectivity scores have not been written yet.")
        return
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        st.info("No selectivity scores available.")
        return
    x = pd.to_numeric(df["target_deltaG"], errors="coerce")
    y = pd.to_numeric(df["mean_decoy_deltaG"], errors="coerce")
    risk = pd.to_numeric(df["off_target_risk_score"], errors="coerce")
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(x, y, c=risk, alpha=0.8)
    ax.set_xlabel("SMN2 target deltaG")
    ax.set_ylabel("Mean decoy deltaG")
    ax.set_title("Target vs decoy binding")
    fig.colorbar(scatter, ax=ax, label="Off-target risk")
    st.pyplot(fig)


def leaderboard() -> None:
    path = OUTPUT_DIR / "molecule_selectivity_scores.tsv"
    if not path.exists():
        st.info("Leaderboard will appear after selectivity scores are written.")
        return
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        st.info("No scored molecules yet.")
        return
    sort_cols = ["advance_to_splicing_rescue", "weighted_selectivity_gap", "target_deltaG"]
    for col in sort_cols:
        if col not in df:
            df[col] = False if col == "advance_to_splicing_rescue" else 0
    df["_gap"] = pd.to_numeric(df["weighted_selectivity_gap"], errors="coerce").fillna(-999)
    df["_target"] = pd.to_numeric(df["target_deltaG"], errors="coerce").fillna(999)
    view = df.sort_values(["advance_to_splicing_rescue", "_gap", "_target"], ascending=[False, False, True]).head(20)
    st.dataframe(
        view[
            [
                "molecule_id",
                "target_deltaG",
                "weighted_selectivity_gap",
                "off_target_risk_score",
                "advance_to_splicing_rescue",
                "notes",
            ]
        ],
        use_container_width=True,
    )


def decoy_risk_view() -> None:
    path = OUTPUT_DIR / "decoy_docking_scores.tsv"
    if not path.exists():
        st.info("Decoy risk view will appear after decoy scores are written.")
        return
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        st.info("No decoy scores available.")
        return
    df["decoy_deltaG"] = pd.to_numeric(df["decoy_deltaG"], errors="coerce")
    grouped = df.groupby("decoy_category")["decoy_deltaG"].mean().dropna().sort_values()
    if grouped.empty:
        st.info("No numeric decoy scores available.")
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(grouped.index.astype(str), grouped.values)
    ax.set_ylabel("Mean decoy deltaG")
    ax.set_title("Average predicted decoy binding by category")
    ax.tick_params(axis="x", rotation=35)
    st.pyplot(fig)


def main() -> None:
    st.set_page_config(page_title="SMN2 Live Docking Simulation", layout="wide")
    state = load_state()
    events = load_events()
    run_mode = state.get("run_mode", "unknown")

    st.title("SMN2-RescueDiff Live Docking Simulation")
    st.caption("Visualizing SMN2 target docking and HuRDA off-target counter-screening")
    if run_mode == "dry_run":
        st.warning("Dry-run simulation mode: displayed scores are simulated and not scientific docking results.")

    stages = "Load molecules -> Prepare ligands -> Validate receptor -> Select decoys -> Dock SMN2 -> Dock decoys -> Calculate selectivity -> Write outputs"
    st.subheader("Pipeline Progress")
    st.code(stages)
    st.progress(float(state.get("percent_complete", 0.0)) / 100.0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Status", state.get("run_status", "not started"))
    col2.metric("Jobs", f"{state.get('jobs_completed', 0)} / {state.get('total_jobs', 0)}")
    col3.metric("Latest score", state.get("latest_score", "NA"))
    col4.metric("Run mode", run_mode)

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Current Docking Job")
        st.write(
            {
                "molecule_id": state.get("current_molecule_id"),
                "target_type": state.get("current_target_type"),
                "target_id": state.get("current_target_id"),
                "decoy_category": state.get("current_decoy_category"),
                "message": state.get("last_message"),
            }
        )
        st.subheader("Molecule Viewer")
        maybe_show_molecule(state.get("current_smiles"))
    with right:
        st.subheader("Target vs Decoy Testing Animation")
        molecule = state.get("current_molecule_id") or "Molecule"
        target = state.get("current_target_id") or "Target"
        st.markdown(
            f"""
            <div style="font-size:24px; padding:20px; border:1px solid #ddd; border-radius:8px;">
            <b>{molecule}</b> &nbsp; -> &nbsp; <b>{target}</b><br/>
            <span style="font-size:14px;">SMN2 target and HuRDA decoy testing visualization</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.subheader("Live Score Chart")
        score_plot(events)

    st.subheader("Target-vs-Decoy Scatter Plot")
    target_decoy_scatter()
    st.subheader("Candidate Leaderboard")
    leaderboard()
    st.subheader("Decoy Category Risk View")
    decoy_risk_view()
    st.subheader("Explanation")
    st.write(
        "This dashboard shows how each generated molecule is tested against the SMN2 target and selected HuRDA RNA decoys. "
        "A strong candidate should bind the SMN2 target more strongly than the decoys. This display is a visualization of "
        "computational docking or dry-run simulation output, not experimental proof of binding or therapeutic activity."
    )


if __name__ == "__main__":
    main()

