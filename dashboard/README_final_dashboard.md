# Final Candidate Explorer Dashboard

This optional dashboard explores Module 5 SRSI-ranked computational candidate molecules.

Run Module 5 first:

```bash
PYTHONPATH=src python3 -m smn2_rescuediff.splicing_rescue.cli --config configs/splicing_rescue_scoring.yaml
```

Then launch:

```bash
streamlit run dashboard/final_candidate_explorer.py
```

If Streamlit is not installed:

```bash
python3 -m pip install streamlit
```

The dashboard reads:

```text
data/processed/final_scoring/tables/final_srsi_ranked_candidates.tsv
data/processed/final_scoring/tables/splicing_rescue_scores.tsv
data/processed/final_scoring/evidence/top_candidate_evidence.json
```

The dashboard is for computational review only. Candidate rankings do not prove SMN2 binding, exon 7 inclusion, safety, efficacy, or therapeutic relevance.
