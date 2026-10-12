# SMN2-RescueDiff TCBB Innovation Revision

This is a self-contained Overleaf-compatible manuscript package. The paper centers the computational-biology contribution on failure-aware evidence contracts, reproducible module interfaces, coordinate-aware structural validation, and honest containment of invalid evidence.

## Contents

- `main.tex`: IEEE journal manuscript source.
- `figures/`: generated PDF and PNG figures.
- `figure_data/`: TSV files used by the figure generator.
- `make_figures.py`: rebuilds all figures from the repository evidence release.
- `CHANGELOG_AND_REVIEW_RESPONSE.md`: paper-level change report and reviewer-criticism disposition.
- `MANIFEST.sha256`: package checksums.
- `SMN2_RescueDiff_TCBB_Innovation_Revision_preview.pdf`: compiled preview.

## Overleaf

Upload the contents of the ZIP file as a new Overleaf project. Set `main.tex` as the main document and compile with pdfLaTeX. The project uses `IEEEtran`, which is available in Overleaf.

## Local Reproduction

From the package directory:

```bash
python make_figures.py
tectonic main.tex
```

The figure script expects to be run from its location inside the SMN2-RescueDiff repository because it reads the pinned release tables under `reproducibility/tcbb_2026`. The ZIP already contains generated figures and can compile independently on Overleaf.

## Interpretation Boundary

The historical HuRDA decoy docking campaign is preserved for audit but withdrawn from selectivity inference because independently positioned receptors were docked with one target-centered box. The corrected receptor-specific redocking code and preset are included in the repository, but no corrected selectivity result is claimed before that computation is completed and quality-controlled.
