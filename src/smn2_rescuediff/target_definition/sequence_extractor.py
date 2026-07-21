"""Genome sequence extraction for SMN2 target windows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from Bio.Seq import Seq
from pyfaidx import Fasta

from .annotation_parser import SelectedTarget


class SequenceExtractionError(ValueError):
    """Raised when a sequence cannot be extracted safely."""


@dataclass(frozen=True)
class TargetWindow:
    target_id: str
    window_name: str
    window_type: str
    chromosome: str
    start: int
    end: int
    strand: str
    sequence_rna: str
    notes: str


def dna_to_rna(dna: str, strand: str) -> str:
    sequence = dna.upper()
    invalid = set(sequence) - set("ACGTN")
    if invalid:
        raise SequenceExtractionError(f"Extracted sequence contains invalid DNA characters: {sorted(invalid)}")
    if strand == "-":
        sequence = str(Seq(sequence).reverse_complement())
    return sequence.replace("T", "U")


def _slice_fasta(genome: Fasta, chrom: str, start: int, end: int, strand: str) -> str:
    if chrom not in genome:
        raise SequenceExtractionError(f"Chromosome {chrom!r} not found in genome FASTA.")
    chrom_len = len(genome[chrom])
    if start < 1 or end > chrom_len or start > end:
        raise SequenceExtractionError(
            f"Requested window {chrom}:{start}-{end} is outside chromosome bounds 1-{chrom_len}."
        )
    dna = str(genome[chrom][start - 1 : end])
    return dna_to_rna(dna, strand)


def _centered_bounds(center: int, size: int) -> tuple[int, int]:
    left = (size - 1) // 2
    right = size - left - 1
    return center - left, center + right


def build_windows(genome_fasta: str | Path, target: SelectedTarget, config: dict[str, Any]) -> list[TargetWindow]:
    genome = Fasta(str(genome_fasta), as_raw=True, sequence_always_upper=True)
    windows_cfg = config.get("windows", {})
    windows: list[TargetWindow] = []

    def add(name: str, window_type: str, start: int, end: int, notes: str) -> None:
        seq = _slice_fasta(genome, target.chromosome, start, end, target.strand)
        windows.append(
            TargetWindow(
                target_id=f"{target.gene_symbol}_{target.transcript_id}_{name}",
                window_name=name,
                window_type=window_type,
                chromosome=target.chromosome,
                start=start,
                end=end,
                strand=target.strand,
                sequence_rna=seq,
                notes=notes,
            )
        )

    exon_start, exon_end = target.start, target.end
    add("exon7", "exon", exon_start, exon_end, "Full annotated exon 7 RNA sequence.")

    ss5 = windows_cfg.get("splice_site_core_5prime", {})
    ss3 = windows_cfg.get("splice_site_core_3prime", {})
    if target.strand == "-":
        add(
            "five_prime_splice_site_core",
            "splice_site",
            exon_start - int(ss5.get("intron_nt", 6)),
            exon_start + int(ss5.get("exon_nt", 3)) - 1,
            "Core 5 prime splice-site window spanning exon-intron boundary.",
        )
        add(
            "three_prime_splice_site_core",
            "splice_site",
            exon_end - int(ss3.get("exon_nt", 3)) + 1,
            exon_end + int(ss3.get("intron_nt", 20)),
            "Core 3 prime splice-site window spanning intron-exon boundary.",
        )
        center = exon_start
    else:
        add(
            "five_prime_splice_site_core",
            "splice_site",
            exon_end - int(ss5.get("exon_nt", 3)) + 1,
            exon_end + int(ss5.get("intron_nt", 6)),
            "Core 5 prime splice-site window spanning exon-intron boundary.",
        )
        add(
            "three_prime_splice_site_core",
            "splice_site",
            exon_start - int(ss3.get("intron_nt", 20)),
            exon_start + int(ss3.get("exon_nt", 3)) - 1,
            "Core 3 prime splice-site window spanning intron-exon boundary.",
        )
        center = exon_end

    for name, size_key in [
        ("core_21nt_approx_5prime_splice_region", "core_window_nt"),
        ("local_51nt_approx_5prime_splice_region", "local_window_nt"),
        ("extended_101nt_approx_5prime_splice_region", "extended_window_nt"),
    ]:
        size = int(windows_cfg.get(size_key, {"core_window_nt": 21, "local_window_nt": 51, "extended_window_nt": 101}[size_key]))
        start, end = _centered_bounds(center, size)
        add(
            name,
            "target_window",
            start,
            end,
            "Computational approximation centered on the exon 7 5 prime splice-site region.",
        )

    genome.close()
    return windows


def write_fasta(windows: list[TargetWindow], target: SelectedTarget, output_path: str | Path) -> None:
    with Path(output_path).open("w", encoding="utf-8") as handle:
        for window in windows:
            header = (
                f">{target.gene_symbol}|transcript_id={target.transcript_id}|window={window.window_name}|"
                f"chrom={window.chromosome}|start={window.start}|end={window.end}|strand={window.strand}"
            )
            handle.write(header + "\n")
            seq = window.sequence_rna
            for i in range(0, len(seq), 80):
                handle.write(seq[i : i + 80] + "\n")
