"""Load Module 1 SMN2 target-definition outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class TargetReference:
    """Compact reference target profile used for decoy similarity scoring."""

    window_name: str
    sequence_rna: str
    sequence_length: int
    gc_content: float
    a_freq: float
    u_freq: float
    g_freq: float
    c_freq: float
    simple_splice_strength_score: float
    u1_complementarity_score: float


def read_fasta(path: str | Path) -> pd.DataFrame:
    """Read a small FASTA file into a DataFrame with header metadata."""
    rows: list[dict[str, str]] = []
    header: str | None = None
    chunks: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    rows.append(_fasta_row(header, "".join(chunks)))
                header = line[1:]
                chunks = []
            else:
                chunks.append(line)
    if header is not None:
        rows.append(_fasta_row(header, "".join(chunks)))
    if not rows:
        raise ValueError(f"No FASTA records were found in {path}")
    return pd.DataFrame(rows)


def _fasta_row(header: str, sequence: str) -> dict[str, str]:
    metadata = {"header": header, "sequence_rna": sequence.upper().replace("T", "U")}
    parts = header.split("|")
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            metadata[key] = value
    metadata["window_name"] = metadata.get("window", parts[0])
    return metadata


def load_target_profile(path: str | Path) -> pd.DataFrame:
    """Load the Module 1 SMN2 target feature profile."""
    profile = pd.read_csv(path, sep="\t")
    required = {"window_name", "sequence_rna", "GC_content", "simple_splice_strength_score", "u1_complementarity_score"}
    missing = required - set(profile.columns)
    if missing:
        raise ValueError(f"SMN2 target profile is missing required columns: {sorted(missing)}")
    return profile


def load_target_metadata(path: str | Path) -> dict:
    """Load Module 1 target metadata JSON."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def choose_reference_target(profile: pd.DataFrame) -> TargetReference:
    """Choose the most useful SMN2 reference window for bias matching."""
    preferences = [
        "local_51nt_approx_5prime_splice_region",
        "core_21nt_approx_5prime_splice_region",
        "five_prime_splice_site_core",
        "exon7",
    ]
    row = None
    for name in preferences:
        matches = profile[profile["window_name"] == name]
        if not matches.empty:
            row = matches.iloc[0]
            break
    if row is None:
        row = profile.iloc[0]
    return TargetReference(
        window_name=str(row["window_name"]),
        sequence_rna=str(row["sequence_rna"]).upper().replace("T", "U"),
        sequence_length=int(row["sequence_length"]),
        gc_content=float(row["GC_content"]),
        a_freq=float(row.get("A_freq", 0.0)),
        u_freq=float(row.get("U_freq", 0.0)),
        g_freq=float(row.get("G_freq", 0.0)),
        c_freq=float(row.get("C_freq", 0.0)),
        simple_splice_strength_score=float(row.get("simple_splice_strength_score", 0.0)),
        u1_complementarity_score=float(row.get("u1_complementarity_score", 0.0)),
    )


def load_smn2_inputs(fasta_path: str | Path, profile_path: str | Path, metadata_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict, TargetReference]:
    """Load all Module 1 target-definition inputs used by HuRDA."""
    fasta = read_fasta(fasta_path)
    profile = load_target_profile(profile_path)
    metadata = load_target_metadata(metadata_path)
    reference = choose_reference_target(profile)
    return fasta, profile, metadata, reference

