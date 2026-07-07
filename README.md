# SMN2-RescueDiff

SMN2-RescueDiff is a computational biology project for defining and later analyzing SMN2 exon 7 splicing-related RNA targets. This repository currently implements only Part 1: SMN2 Target Definition.

This module identifies a configured SMN2 transcript and exon 7 region from user-provided genome and annotation files, extracts exon and splice-site RNA windows, computes exploratory sequence features, optionally stores RCSB PDB structure files, and writes reproducible FASTA, TSV, JSON, figure, and Markdown report outputs.

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

## Tests

The tests use tiny artificial fixture files and do not require hg38 downloads:

```bash
PYTHONPATH=src pytest
```

## Limitations

This module does not generate molecules, dock molecules, perform reinforcement learning, build a decoy atlas, or calculate final splicing-rescue scores. It does not prove splice rescue or therapeutic activity. Its purpose is to create a cautious, reproducible target definition for later modules.

## Next Module Preview

Later modules can consume these target windows for RNA structural context, decoy counter-screening, generative molecule design, docking, motif-aware scoring, and transcriptome-aware ranking.
