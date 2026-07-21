# SMN2-RescueDiff

SMN2-RescueDiff is a computational biology project for defining and later analyzing SMN2 exon 7 splicing-related RNA targets. This repository currently implements Module 1: SMN2 Target Definition, Module 2: HuRDA-180K Human RNA Decoy Atlas, Module 3: Deep Neural Molecular Generation Model, and Module 4: SMN2 Docking and HuRDA Off-Target Counter-Screening.

The pipeline defines the SMN2 exon 7 splicing-related RNA target, builds a transcriptome-scale HuRDA decoy atlas, generates computational molecule candidates, evaluates target-vs-decoy docking/selectivity evidence, and produces a final SRSI-ranked candidate table with cautious evidence summaries.

## Folder Structure

```text
SMN2-RescueDiff/
  configs/target_definition.yaml
  data/raw/genome/
  data/raw/annotations/
  data/raw/structures/
  data/processed/target_definition/
  reports/target_definition_report.md
  figures/target_definition/
  src/smn2_rescuediff/target_definition/
  tests/
```

## Installation

From this folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For direct source-tree execution, set:

```bash
export PYTHONPATH="$PWD/src"
```

## Input Data

Place a genome FASTA and matching annotation file in:

```text
data/raw/genome/
data/raw/annotations/
```

Use files from a consistent genome build, such as hg38 with a matching GENCODE, Ensembl, or RefSeq GTF/GFF. Because SMN1 and SMN2 are highly similar paralogs, do not mix genome builds or annotation sources.

Edit `configs/target_definition.yaml`:

```yaml
inputs:
  genome_fasta: data/raw/genome/hg38.fa
  annotation_gtf: data/raw/annotations/gencode.gtf

target:
  gene_symbol: SMN2
  gene_id: null
  transcript_id: null
  exon_number: 7
  manual_exon_coordinates: null
```

If more than one SMN2 transcript contains exon 7 and no single canonical transcript is marked in the annotation, the pipeline stops and asks you to set `target.transcript_id` or provide `target.manual_exon_coordinates`.

## Running

```bash
python -m smn2_rescuediff.target_definition.cli --config configs/target_definition.yaml
```

If the package is not installed, run with `PYTHONPATH`:

```bash
PYTHONPATH=src python -m smn2_rescuediff.target_definition.cli --config configs/target_definition.yaml
```

## Expected Outputs

The pipeline writes:

```text
data/processed/target_definition/smn2_target_windows.fasta
data/processed/target_definition/smn2_target_profile.tsv
data/processed/target_definition/smn2_target_metadata.json
data/processed/target_definition/smn2_transcript_candidates.tsv
data/processed/target_definition/structure_metadata.tsv
reports/target_definition_report.md
figures/target_definition/smn2_target_feature_profile.png
```

The FASTA headers include transcript, window, chromosome, coordinate, and strand metadata. The TSV profile includes RNA sequence features such as nucleotide frequencies, GC content, k-mer counts, exploratory motif density, simple splice-site score, U1 complementarity score, boundary overlap, and distances from exon 7 ends.

## Optional Structures

`configs/target_definition.yaml` includes configurable PDB IDs. When `structures.fetch_pdb` is true, the pipeline attempts to download mmCIF files from RCSB into `data/raw/structures/`. Failed downloads are recorded in `structure_metadata.tsv` and do not stop sequence target definition.

## Module 2: HuRDA-180K Human RNA Decoy Atlas

HuRDA-180K builds a transcriptome-scale human RNA decoy atlas for future SMN2 splicing-modifier counter-screening. The atlas is designed to provide biologically motivated candidate off-target RNA motifs, including splice-site decoys, SMN2-like hard decoys, candidate regulatory windows, SMN2 scrambled and mutated controls, and random bias-matched transcriptome windows.

### Required Inputs

Module 2 uses the Module 1 outputs:

```text
data/processed/target_definition/smn2_target_windows.fasta
data/processed/target_definition/smn2_target_profile.tsv
data/processed/target_definition/smn2_target_metadata.json
```

It also uses the same raw genome and annotation files:

```text
data/raw/genome/hg38.fa
data/raw/annotations/gencode.gtf
```

All paths are configurable in `configs/decoy_atlas.yaml`.

### Running Module 2

```bash
PYTHONPATH=src python3 -m smn2_rescuediff.decoy_atlas.cli --config configs/decoy_atlas.yaml
```

### Expected Module 2 Outputs

```text
data/processed/decoy_atlas/hurda_180k_full.tsv
data/processed/decoy_atlas/hurda_180k_full.fasta
data/processed/decoy_atlas/hurda_train.tsv
data/processed/decoy_atlas/hurda_validation.tsv
data/processed/decoy_atlas/hurda_test.tsv
data/processed/decoy_atlas/hurda_hard_smn2_like_holdout.tsv
data/processed/decoy_atlas/hurda_chromosome_holdout.tsv
data/processed/decoy_atlas/hurda_gene_holdout.tsv
data/processed/decoy_atlas/hurda_summary.json
data/processed/decoy_atlas/hurda_feature_summary.tsv
data/processed/decoy_atlas/hurda_decoy_category_counts.tsv
data/processed/decoy_atlas/hurda_removed_duplicates.tsv
data/processed/decoy_atlas/hurda_generation_log.txt
reports/decoy_atlas_report.md
figures/decoy_atlas/decoy_category_counts.png
figures/decoy_atlas/gc_content_distribution.png
figures/decoy_atlas/smn2_similarity_distribution.png
figures/decoy_atlas/bias_match_score_distribution.png
figures/decoy_atlas/splice_strength_distribution.png
figures/decoy_atlas/u1_complementarity_distribution.png
```

### Module 2 Limitations

HuRDA decoys are candidate RNA motifs for future computational counter-screening. They do not prove off-target binding, toxicity, clinical safety, drug activity, or SMN2 exon 7 rescue. The similarity and bias-match scores are exploratory sequence-derived features that should be complemented by later structural, docking, and experimental evidence.

### Next Module Preview

The next stage can use the SMN2 target definition and HuRDA decoy atlas for molecule generation, chemical filtering, docking counter-screens, and transcriptome-aware SRSI scoring.

## Module 3: Deep Neural Molecular Generation Model

Module 3 trains and samples a neural molecular language model for future SMN2 RNA docking preparation and HuRDA transcriptome-aware off-target counter-screening. The primary path is a PyTorch Transformer decoder with SELFIES support; a GRU/LSTM fallback and a deterministic debug n-gram mode are retained for small hardware and smoke testing. Generated molecules are computational candidates only.

### Required Molecule Input

For real runs, provide one or more SMILES files with one molecule per line:

```text
data/raw/molecules/chembl_training_smiles.smi
```

Recommended real training sources include ChEMBL, ZINC subsets, RNA-binding ligand SMILES if available, known RNA-targeting small molecules, approved drug-like molecule libraries, or project-specific compound collections. The configured input is a list:

```yaml
inputs:
  training_smiles:
    - data/raw/molecules/chembl_training_smiles.smi
  optional_context:
    smn2_target_profile: data/processed/target_definition/smn2_target_profile.tsv
    smn2_target_metadata: data/processed/target_definition/smn2_target_metadata.json
    decoy_atlas_summary: data/processed/decoy_atlas/hurda_summary.json
```

If the configured training file is missing, the module creates `data/raw/molecules/example_training_smiles.smi` as a toy fallback. That file is only for software testing and should not be used for scientific conclusions.

### Running Module 3

```bash
PYTHONPATH=src python3 -m smn2_rescuediff.molecule_generation.cli --config configs/molecule_generation.yaml
```

The default config uses `tokenization.representation: selfies` and `generation.generator_mode: neural_transformer`. If PyTorch is unavailable, install it in the project environment or use the debug path:

```yaml
generation:
  generator_mode: debug_ngram
```

### Runtime Controls

For laptop-scale test runs, reduce:

```yaml
run:
  overwrite_existing: false

generation:
  generator_mode: neural_gru

training:
  epochs: 1
  batch_size: 16

sampling:
  num_raw_to_generate: 1000

analysis:
  calculate_diversity: true
  pairwise_diversity_sample_size: 500
```

### Expected Module 3 Outputs

```text
data/processed/molecule_generation/runs/<run_name>/training_smiles_merged_cleaned.smi
data/processed/molecule_generation/runs/<run_name>/training_dataset_summary.json
data/processed/molecule_generation/runs/<run_name>/training_dataset_rejected.tsv
data/processed/molecule_generation/runs/<run_name>/token_vocabulary.json
data/processed/molecule_generation/runs/<run_name>/tokenizer_metadata.json
data/processed/molecule_generation/runs/<run_name>/generated_strings_raw.tsv
data/processed/molecule_generation/runs/<run_name>/generated_smiles_raw.smi
data/processed/molecule_generation/runs/<run_name>/generated_smiles_raw.tsv
data/processed/molecule_generation/runs/<run_name>/generated_molecules_valid.tsv
data/processed/molecule_generation/runs/<run_name>/generated_molecules_filtered.tsv
data/processed/molecule_generation/runs/<run_name>/generated_molecules_rejected.tsv
data/processed/molecule_generation/runs/<run_name>/generated_molecules_for_docking.tsv
data/processed/molecule_generation/runs/<run_name>/generated_molecule_summary.json
data/processed/molecule_generation/runs/<run_name>/model_training_log.tsv
data/processed/molecule_generation/runs/<run_name>/rl_finetuning_log.tsv
data/processed/molecule_generation/runs/<run_name>/molecule_generation_config_used.yaml
data/processed/molecule_generation/runs/<run_name>/checkpoints/best_neural_generator.pt
reports/molecule_generation_report.md
figures/molecule_generation/training_loss_curve.png
figures/molecule_generation/validation_loss_curve.png
figures/molecule_generation/rl_reward_curve.png
figures/molecule_generation/molecular_weight_distribution.png
figures/molecule_generation/logp_distribution.png
figures/molecule_generation/qed_distribution.png
figures/molecule_generation/rna_ligand_like_score_distribution.png
figures/molecule_generation/pre_docking_candidate_score_distribution.png
figures/molecule_generation/scaffold_frequency_top20.png
figures/molecule_generation/validity_uniqueness_novelty_summary.png
figures/molecule_generation/diversity_distribution.png
```

The most recent run also copies `generated_molecules_filtered.tsv`, `generated_molecules_for_docking.tsv`, and `generated_molecule_summary.json` into `data/processed/molecule_generation/` so Module 4 can consume them without knowing the run name.

### Module 3 Limitations

Generated molecules are computational candidates only. This module does not provide synthesis instructions, does not test molecules in humans, animals, or cells, and does not prove that any molecule binds SMN2, rescues exon 7 inclusion, treats SMA, or is safe. Reward diagnostics and chemical filters are computational screens for future prioritization.

### Next Module Preview

The next module can use the filtered candidate library for SMN2 target docking preparation, followed by HuRDA decoy counter-screening and transcriptome-aware scoring.

## Module 4: SMN2 Docking and HuRDA Off-Target Counter-Screening

Module 4 evaluates generated candidate molecules from Module 3 against the SMN2 exon 7 splicing-related RNA target and a selected HuRDA decoy panel. It prepares ligands, validates receptor inputs, runs dry-run simulation or real docking hooks, calculates target-vs-decoy selectivity gaps, writes live JSONL events, and powers a Streamlit live simulation dashboard.

### Required Inputs

```text
data/processed/molecule_generation/generated_molecules_filtered.tsv
data/processed/molecule_generation/generated_molecule_summary.json
data/processed/target_definition/smn2_target_windows.fasta
data/processed/target_definition/smn2_target_profile.tsv
data/processed/target_definition/smn2_target_metadata.json
data/processed/decoy_atlas/hurda_180k_full.tsv
data/processed/decoy_atlas/hurda_summary.json
```

Real docking also requires a 3D SMN2 receptor structure or prepared receptor file:

```text
data/processed/docking/receptors/smn2_target.pdbqt
```

FASTA sequence windows alone cannot be used for real molecular docking without a 3D receptor structure.

### Run Modes

- `dry_run`: deterministic simulated docking scores for pipeline testing and dashboard visualization only.
- `prepare_only`: prepare ligands, validate receptor configuration, and select the decoy panel without docking.
- `target_only_docking`: run real docking against the SMN2 receptor only.
- `target_plus_decoy_docking`: run real target docking and decoy counter-screening when receptor files are available.
- `analyze_existing`: regenerate summaries, scores, figures, and reports from existing outputs.

### How To Run

Dry-run simulation:

```bash
PYTHONPATH=src python3 -m smn2_rescuediff.docking.cli --config configs/docking_counter_screening.yaml
```

Prepare-only mode:

```yaml
run:
  run_mode: prepare_only
```

Real docking mode requires AutoDock Vina, prepared ligand/receptor PDBQT files, and a confirmed docking box:

```yaml
run:
  run_mode: target_only_docking

receptor:
  smn2_receptor_file: data/processed/docking/receptors/smn2_target.pdbqt
  docking_box:
    center_x: 0.0
    center_y: 0.0
    center_z: 0.0
    size_x: 24.0
    size_y: 24.0
    size_z: 24.0
```

### Expected Module 4 Outputs

```text
data/processed/docking/selected_molecules_for_docking.tsv
data/processed/docking/ligand_preparation_summary.tsv
data/processed/docking/receptor_preparation_summary.tsv
data/processed/docking/selected_decoy_docking_panel.tsv
data/processed/docking/smn2_target_docking_scores.tsv
data/processed/docking/decoy_docking_scores.tsv
data/processed/docking/molecule_selectivity_scores.tsv
data/processed/docking/candidates_for_splicing_rescue.tsv
data/processed/docking/docking_summary.json
data/processed/docking/logs/docking_run_log.txt
data/processed/docking/logs/failed_docking_jobs.tsv
data/processed/docking/simulation/live_docking_events.jsonl
data/processed/docking/simulation/current_docking_state.json
reports/docking_counter_screening_report.md
figures/docking/
```

### Live Dashboard

```bash
streamlit run dashboard/docking_live_app.py
```

If Streamlit is missing:

```bash
python3 -m pip install streamlit
```

The dashboard displays run progress, current molecule/target, dry-run warnings, score traces, target-vs-decoy scatter plots, a candidate leaderboard, and decoy-category risk summaries.

### Module 4 Limitations

Dry-run results are simulated and are not scientific docking results. Real docking scores are still computational predictions and do not prove binding, safety, therapeutic activity, or SMN2 splicing rescue. Module 4 provides binding and selectivity estimates only.

## Module 5: Predicted Splicing-Rescue Scoring and Final SRSI Ranking

Module 5 is the final computational prioritization layer. It integrates SMN2 exon 7 target definition, HuRDA RNA decoy context, generated molecule features, target docking, and decoy selectivity to rank computational candidate molecules with SRSI: Splicing Rescue Selectivity Index.

The module does not claim that any molecule is a drug, treatment, cure, safe compound, or experimentally validated SMN2 splicing modifier. SRSI is a prioritization score for future validation hypotheses.

### Required Inputs

```text
data/processed/target_definition/smn2_target_windows.fasta
data/processed/target_definition/smn2_target_profile.tsv
data/processed/target_definition/smn2_target_metadata.json
data/processed/decoy_atlas/hurda_180k_full.tsv
data/processed/decoy_atlas/hurda_summary.json
data/processed/molecule_generation/generated_molecules_filtered.tsv
data/processed/molecule_generation/generated_molecules_for_docking.tsv
data/processed/molecule_generation/generated_molecule_summary.json
data/processed/docking/smn2_target_docking_scores.tsv
data/processed/docking/decoy_docking_scores.tsv
data/processed/docking/molecule_selectivity_scores.tsv
data/processed/docking/candidates_for_splicing_rescue.tsv
data/processed/docking/docking_summary.json
data/processed/docking/poses/
```

Older Module 3 runs may not include `generated_molecules_for_docking.tsv`; Module 5 can still use `generated_molecules_filtered.tsv` and records a warning.

### How To Run

```bash
PYTHONPATH=src python3 -m smn2_rescuediff.splicing_rescue.cli --config configs/splicing_rescue_scoring.yaml
```

### Scoring Modes

- `pose_based`: uses docking pose/receptor coordinates when available.
- `feature_based`: uses docking score, selectivity, target-window landmarks, and Module 1 target features.
- `hybrid_pose_and_feature`: uses pose-based evidence when possible and feature-based fallback when poses are missing.
- `dry_run_input_analysis`: labels simulated Module 4 dry-run scores as pipeline-test evidence only.

### SRSI Explanation

SRSI rewards strong predicted SMN2 pre-mRNA target binding, positive HuRDA selectivity gap, predicted splicing-rescue proxy score, QED, Lipinski-style quality, and pre-docking candidate score. It penalizes HuRDA off-target risk, PAINS alerts, synthetic-accessibility proxy when available, and potential splice-site occlusion. Because docking energies are more negative for stronger predicted binding, target binding strength is calculated as `-1 * target_deltaG`.

Binding alone is not enough. A strong binder can rank lower if it also binds HuRDA decoys, has poor splicing-rescue proxy support, or appears likely to occlude core splice-site recognition.

### Expected Module 5 Outputs

```text
data/processed/final_scoring/tables/smn2_splicing_landmarks.tsv
data/processed/final_scoring/tables/pose_contact_features.tsv
data/processed/final_scoring/tables/splicing_rescue_scores.tsv
data/processed/final_scoring/tables/final_srsi_ranked_candidates.tsv
data/processed/final_scoring/tables/top_10_final_candidates.tsv
data/processed/final_scoring/tables/ablation_rank_comparison.tsv
data/processed/final_scoring/evidence/top_candidate_evidence.json
data/processed/final_scoring/evidence/scoring_warnings.json
data/processed/final_scoring/final_scoring_summary.json
data/processed/final_scoring/splicing_rescue_scoring_config_used.yaml
data/processed/final_scoring/logs/module5_run_log.txt
reports/splicing_rescue_scoring_report.md
reports/final_candidate_report.md
figures/final_scoring/
```

### How To Interpret Results

Use `final_srsi_ranked_candidates.tsv` as the main ranked output. The top rows are computational candidates prioritized by target binding, HuRDA selectivity, predicted splicing-rescue proxy evidence, chemical quality, confidence level, and caution flags. Dry-run results are clearly labeled and should not be interpreted as scientific docking results.

### Final Dashboard

```bash
streamlit run dashboard/final_candidate_explorer.py
```

If Streamlit is missing:

```bash
python3 -m pip install streamlit
```

### Module 5 Limitations

Docking scores are computational predictions. Predicted splicing-rescue scores are proxy scores. SRSI is a prioritization metric, not proof of biological activity. Top candidates require future validation before biological interpretation. This repository does not provide synthesis instructions or human, animal, or cell testing procedures.

## Tests

The tests use tiny artificial fixture files and do not require hg38 downloads:

```bash
PYTHONPATH=src pytest
```

## Limitations

These modules produce computational target definitions, decoy atlases, molecule candidates, docking/selectivity estimates, and final SRSI rankings. They do not prove splice rescue, safety, therapeutic efficacy, or clinical relevance.

## Next Module Preview

The full pipeline now produces a ranked computational candidate table for future validation hypotheses, with all results interpreted cautiously.
