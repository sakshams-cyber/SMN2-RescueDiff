# October 2026 Reviewer Revision Evidence

This directory contains deterministic analyses added in response to the TCBB-style evaluator reports. Historical score tables were read as observed; no docking score was changed, filled, or removed.

## Rebuild

```bash
MPLCONFIGDIR=/tmp/mpl-cache .venv/bin/python scripts/audit_tcbb_reviewer_findings.py
MPLCONFIGDIR=/tmp/mpl-cache .venv/bin/python scripts/compare_module3_generator_baseline.py
```

## Main Findings

- 81 of 97 historical HuRDA receptors had less than 10% of atoms inside the reused target box.
- Box coverage versus receptor mean score: Spearman rho -0.8469, p 8.36e-28.
- Receptor identity explained 98.83% of score variance; ligand identity explained 0.132%.
- 4,815 of 14,259 scores were exactly zero; sampled zero-score poses were usually spatially separated from receptor atoms.
- HuRDA_000041 was the strongest historical decoy for all 147 ligands.
- The historical structural-decoy campaign is preserved for audit but withdrawn from selectivity inference.
- MolGen_038292 descriptors, the 1/97 stronger-decoy count, generation denominators, and size-controlled target metrics were corrected from repository data.
- The 50,000-sample token 3-gram baseline produced 915 valid and 163 filter-pass molecules; the production Transformer/SELFIES run produced 49,660 and 40,477.

`reviewer_revision_metrics.json` and `generator_baseline_metrics.json` provide machine-readable summaries. The TSV files provide the complete receptor-, molecule-, or replicate-level data behind each claim.
