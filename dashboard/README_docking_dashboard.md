# SMN2-RescueDiff Live Docking Dashboard

Run from the repository root:

```bash
streamlit run dashboard/docking_live_app.py
```

If Streamlit is missing:

```bash
python3 -m pip install streamlit
```

The dashboard reads:

```text
data/processed/docking/simulation/live_docking_events.jsonl
data/processed/docking/simulation/current_docking_state.json
data/processed/docking/molecule_selectivity_scores.tsv
data/processed/docking/decoy_docking_scores.tsv
```

Dry-run results are simulated for pipeline testing only and are not scientific docking results.
