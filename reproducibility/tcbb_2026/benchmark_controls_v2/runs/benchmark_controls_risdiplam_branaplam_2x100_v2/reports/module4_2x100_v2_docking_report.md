# SMN2-RescueDiff - SMN2 Target Docking and HuRDA Off-Target Counter-Screening

## Purpose

This module estimates computational target binding and off-target RNA selectivity for generated candidate molecules. Docking scores are model-based predictions and do not prove experimental binding, splicing rescue, safety, or therapeutic activity.

## Inputs Used

- Module 1 SMN2 target FASTA/profile/metadata
- Module 2 HuRDA decoy atlas and summary
- Module 3 filtered generated molecules
- Optional receptor structures from the configured receptor path and raw structure directory

## Run Mode

Run mode: `target_plus_decoy_docking`

Stage name: `benchmark_controls_risdiplam_branaplam_2x100_v2`

## Receptor Preparation Summary

```json
[
  {
    "receptor_id": "SMN2_target",
    "receptor_type": "SMN2_RNA_target",
    "receptor_name": "SMN2 exon 7 splicing-related RNA target",
    "source_file": "data/processed/docking/receptors/smn2_target_model1.pdbqt",
    "prepared_file": "data/processed/docking/receptors/smn2_target_model1.pdbqt",
    "format": "pdbqt",
    "box_center_x": 24.896,
    "box_center_y": 20.352,
    "box_center_z": 24.973,
    "box_size_x": 44.963,
    "box_size_y": 34.551,
    "box_size_z": 44.652,
    "prep_status": "ready",
    "notes": ""
  }
]
```

## Ligand Preparation Summary

Ligand preparation status counts:

```json
{
  "prepared": 2
}
```

## Decoy Panel Selection Strategy

The HuRDA panel is selected by configured category targets, then prioritized by SMN2 similarity score, biological weight, and bias-match score. If decoy receptor structures are unavailable, the panel remains a sequence-based counter-screening panel for future structural modeling.

```json
{
  "smn2_like_hard_decoy": 76,
  "five_prime_splice_site": 24
}
```

## Docking Engine

Configured engine: `vina`. Real Vina docking requires prepared receptor and ligand PDBQT files plus a confirmed docking box. Dry-run mode uses deterministic simulated scores and is not scientific docking.

## SMN2 Target Docking Summary

Target score summary:

```json
{
  "count": 2,
  "min": -8.591,
  "mean": -8.456
}
```

## Decoy Counter-Screening Summary

Decoy score summary:

```json
{
  "count": 200,
  "min": -10.1,
  "mean": -2.3444757550000004
}
```

## Selectivity Score Definitions

Because more negative docking energy indicates stronger predicted binding, selectivity gaps are calculated as decoy score minus target score. Larger positive gaps indicate stronger predicted target selectivity.

## Candidate Advancement Criteria

Default advancement requires target deltaG at or below the configured threshold, weighted selectivity gap above the configured threshold, off-target risk at or below the configured threshold, and enough valid decoy scores. In dry-run mode, the flag is based on simulated values only.

Advanced candidate count: 0

## Live Simulation Dashboard

The live simulation writes JSONL events and a current-state JSON file. The dashboard can replay or monitor these files:

```bash
streamlit run dashboard/docking_live_app.py
```

## Warnings

- No warnings were recorded.

## Output Files

- `data/processed/docking/selected_molecules_for_docking.tsv`
- `data/processed/docking/selected_molecules_for_decoy_docking.tsv`
- `data/processed/docking/ligand_preparation_summary.tsv`
- `data/processed/docking/receptor_preparation_summary.tsv`
- `data/processed/docking/selected_decoy_docking_panel.tsv`
- `data/processed/docking/smn2_target_docking_scores.tsv`
- `data/processed/docking/decoy_docking_scores.tsv`
- `data/processed/docking/molecule_selectivity_scores.tsv`
- `data/processed/docking/candidates_for_splicing_rescue.tsv`
- `data/processed/docking/docking_summary.json`
- `data/processed/docking/docking_config_used.yaml`
- `data/processed/docking/logs/docking_run_log.txt`
- `data/processed/docking/logs/failed_docking_jobs.tsv`
- `data/processed/docking/simulation/live_docking_events.jsonl`
- `data/processed/docking/simulation/current_docking_state.json`
- `data/processed/docking/simulation/docking_simulation_summary.json`

## Limitations

- Docking scores are model-based computational predictions and do not prove experimental binding.
- Dry-run scores are simulated for pipeline testing and visualization only.
- This module does not prove splicing rescue, safety, therapeutic efficacy, or clinical relevance.
- Sequence-only RNA target windows cannot replace real 3D receptor structures for scientific docking.
- No chemical synthesis or biological testing instructions are provided.

## Next Steps

Module 5 will be Predicted Splicing-Rescue Scoring. Module 4 addresses binding and selectivity; Module 5 will evaluate whether top molecules bind in a position predicted to support SMN2 exon 7 inclusion, U1 snRNA recognition, and splice-site stabilization.
