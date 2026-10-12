# SMN2-RescueDiff TCBB Reproducibility Snapshot

This directory freezes the code-adjacent inputs, configurations, score tables, failure records, and validation outputs available for the experiment discussed in peer review. It preserves observed results and does not fabricate or impute missing jobs.

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
- `release_validation.json`: machine-readable count and status checks.
- `SOURCE_FILE_MANIFEST.tsv`: source and release hashes, including path-sanitization status.
- `CHECKSUMS.sha256`: hashes of every released file.

Absolute local paths in text tables were replaced with repository-relative paths. `SOURCE_FILE_MANIFEST.tsv` records both the original source hash and released-file hash. Large generated docking pose directories are not duplicated here; the released inputs, protocol, job manifest, and mode-score/RMSD tables support rerunning and auditing the reported validation.

## Interpretation

The release resolves the stale-code contradiction by pairing the current real-decoy Vina implementation with the exact structural receptor panel and observed per-pair output. It does not make the docking method more accurate. The redocking result remains mixed, so target/selectivity rankings should be presented as exploratory computational outputs and a methodological caution.
