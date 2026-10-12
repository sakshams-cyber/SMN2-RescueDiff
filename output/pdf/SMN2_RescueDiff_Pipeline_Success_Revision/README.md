# SMN2-RescueDiff manuscript revision

This package contains the revised manuscript:

**SMN2-RescueDiff: An Auditable Autoregressive Generation and RNA-Decoy Docking Pipeline for SMN2 Splicing-Modifier Prioritization**

The revision preserves the original paper's modular pipeline, methods, core results, and overall organization while reframing the central contribution around the pipeline's demonstrated computational successes. It also retains explicit limitations for docking calibration, biological interpretation, and missing experimental validation.

## Package contents

- `main.tex`: complete IEEE-style manuscript source.
- `figures/`: publication figures in PDF and PNG formats.
- `figure_data/`: released tables and audit records used to regenerate the revised figures.
- `make_figures.py`: deterministic figure-generation script.
- `SMN2_RescueDiff_Pipeline_Success_Revision_preview.pdf`: compiled 13-page preview.
- `CHANGELOG_AND_REVIEW_RESPONSE.md`: itemized record of manuscript changes and reviewer criticisms addressed.
- `MANIFEST.sha256`: checksums for the package files.

The exact repository snapshot cited in the manuscript is branch `real-pipeline-rigor`, commit `deacf0343dd0d3b2874e1ed8db1f24da035961ed`.

## Reproduce in Overleaf

1. Create a blank Overleaf project.
2. Upload `main.tex` and the entire `figures/` directory without changing the folder names.
3. Set `main.tex` as the main document.
4. Select `pdfLaTeX` as the compiler.
5. Compile the project.

The manuscript uses only standard TeX packages available in Overleaf. The precompiled figure PDFs are the publication inputs, so Python is not required inside Overleaf.

## Rebuild figures locally

From this package directory, run:

```bash
python make_figures.py
```

The script reads the files in `figure_data/` and replaces the generated figure files in `figures/`. It requires Python, pandas, NumPy, and Matplotlib. The pipeline diagram is generated from the audited counts encoded in the script and cross-checked against `generation_statistics_audit.json` and the released repository records.

## Build locally

With a TeX distribution installed:

```bash
pdflatex main.tex
pdflatex main.tex
```

The delivered preview was independently compiled with Tectonic and visually checked page by page. The two-pass command above resolves cross-references in a conventional Overleaf-compatible workflow.

## Interpretation boundary

The manuscript reports computational prioritization, AutoDock Vina scores, reproducibility checks, matched-control enrichment, SRSI ablation, and native-pose redocking. It does not claim measured RNA binding, exon 7 inclusion, cellular efficacy, toxicity, clinical superiority, or therapeutic efficacy. Risdiplam and branaplam were held-out benchmark controls and were not used for model training.

## Source preservation

The original PDF and original Overleaf source were not modified. This directory is a new revision package. Numerical statements in the revision were checked against released tables, logs, or provenance records; no missing docking value was invented or imputed.
