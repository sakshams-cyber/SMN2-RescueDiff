# SMN2-RescueDiff TCBB Reproducibility Snapshot

This directory freezes the code-adjacent inputs, configurations, score tables, failure records, and validation outputs available for the experiment discussed in peer review. It preserves observed results and does not fabricate or impute missing jobs.

## October 2026 Corrective Audit

The historical 14,259-pair HuRDA structural docking table is no longer used as evidence of RNA selectivity. A coordinate audit showed that all independently positioned decoy receptors were docked with one target-centered box: 81 of 97 receptors had less than 10% of their atoms in that box, box coverage correlated with mean score at Spearman rho -0.847, and receptor identity explained 98.8% of score variance. Exact zero scores occurred in 4,815 jobs and sampled zero-score poses were typically far from receptor atoms. These findings invalidate target-versus-decoy interpretation of that campaign without changing or deleting the historical records.

The correction is implemented in code: decoy jobs can derive a receptor-specific box from each receptor's coordinates, and fixed-box mode can enforce minimum atom coverage. The corrective 150 x 100 redocking preset is included but has not been represented as completed evidence. Until that campaign is run, target-only five-conformer robustness is the primary screen and all historical Stage C/SRSI selectivity values are audit-only.

## Scope Corrections

- The molecular generator is an **unconditional autoregressive SELFIES Transformer decoder**. It is not a diffusion model and is not conditioned on SMN2 sequence or structure.
- The production run requested 50,000 samples, emitted 49,660 unique generated strings after sampler deduplication, retained 40,477 after filtering, and exported 1,000 candidates.
- The historical generation run used ChEMBL and ZINC subset files. Configured RBIND and SMMRNA files contributed zero rows.
- The intended HuRDA experiment was 150 molecules x 100 decoys, but the frozen observed table is **147 molecules x 97 decoys = 14,259 completed real Vina pairs**. There are 741 explicitly recorded missing pairs relative to 15,000. Missing scores were not imputed.
- A newer complete 200 x 100 run is a separate experiment and is not used to fill this snapshot.
- Four-structure redocking completed 225 real Vina jobs and retained 4,488 modes. Top-ranked near-native recovery was mixed and does not validate broad RNA selectivity inference.
- SRSI, contact, target, and decoy scores are computational descriptors. They do not establish binding, exon 7 inclusion, toxicity, safety, or clinical efficacy.

## Contents

- `generation/`: original run summaries, training/source accounting, losses, docking export, and post hoc two-control benchmark annotation.
- `target_docking/`: exact five-conformer score tables, selected molecules, receptor PDBQT files, prepared ligand PDBQT files, configuration, and failure records.
- `decoy_docking/`: the observed 14,259-pair table, missing-pair table, source-run configs/logs, 100 prepared HuRDA receptor PDBQT files, and prepared ligand PDBQT files.
- `final_scoring/`: staged selectivity inputs and Module 5 SRSI/contact/ablation tables for audit, with no claim that these scores are biologically validated.
- `redocking/`: locked protocol, deposited/prepared structures, job/mode scores, RMSD tables, contact comparison, and report.
- `reviewer_revision_2026_10/`: fixed-box geometry audit, exact-zero pose audit, decoy-panel sensitivity, corrected descriptors, size-controlled target analysis, generation accounting, generator baseline comparison, and machine-readable metrics.
- `cross_target_40x20/`: the previously missing 800-job cross-target score table, inputs, receptor files, logs, poses, configuration, and provenance. Protein and RNA Vina scores are not treated as directly commensurate binding energies.
- `benchmark_controls_v2/`: complete risdiplam/branaplam target and 2 x 100 historical decoy run for audit. Its decoy scores share the fixed-box defect and are not used for selectivity inference.
- `generation_model/`: the production model checkpoint, tokenizer/vocabulary, run configuration, training log, source balance, and dataset summary. The complete third-party training corpus is not redistributed because exact upstream license/release metadata could not be recovered reliably.
- `generator_baseline_50k/`: controlled token 3-gram baseline configuration and outcome tables for the same requested sample count.
- `release_validation.json`: machine-readable count and status checks.
- `SOURCE_FILE_MANIFEST.tsv`: source and release hashes, including path-sanitization status.
- `CHECKSUMS.sha256`: hashes of every released file.

Absolute local paths in text tables were replaced with repository-relative paths. `SOURCE_FILE_MANIFEST.tsv` records both the original source hash and released-file hash. Large generated docking pose directories are not duplicated here; the released inputs, protocol, job manifest, and mode-score/RMSD tables support rerunning and auditing the reported validation.

## Interpretation

The release resolves the stale-code contradiction and now also documents the more consequential coordinate-frame failure. It does not make the historical decoy results accurate. Redocking remains mixed (one of four rank-1 recoveries below 2 Angstrom), no experimental exon-inclusion labels are available, and independent RNA-ligand validation remains outstanding. The defensible contribution is the auditable pipeline architecture and its ability to expose and contain failed evidence.
