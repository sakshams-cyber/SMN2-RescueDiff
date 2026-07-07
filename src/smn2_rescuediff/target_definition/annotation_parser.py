"""GTF/GFF parsing and SMN2 exon selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


class AnnotationError(ValueError):
    """Raised when annotation parsing or target selection fails."""


@dataclass(frozen=True)
class ExonRecord:
    gene_symbol: str
    gene_id: str | None
    transcript_id: str
    chromosome: str
    start: int
    end: int
    strand: str
    exon_number: int | None
    exon_id: str | None = None


@dataclass(frozen=True)
class SelectedTarget:
    gene_symbol: str
    gene_id: str | None
    transcript_id: str
    chromosome: str
    start: int
    end: int
    strand: str
    exon_number: int
    annotation_confidence: str
    notes: str


def parse_attributes(raw: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for part in raw.strip().strip(";").split(";"):
        item = part.strip()
        if not item:
            continue
        if "=" in item and " " not in item.split("=", 1)[0]:
            key, value = item.split("=", 1)
        else:
            pieces = item.split(None, 1)
            if len(pieces) != 2:
                continue
            key, value = pieces
        attrs[key.strip()] = value.strip().strip('"')
    return attrs


def _first(attrs: dict[str, str], names: list[str]) -> str | None:
    for name in names:
        if attrs.get(name):
            return attrs[name]
    return None


def read_exons(annotation_path: str | Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with Path(annotation_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                continue
            chrom, source, feature, start, end, score, strand, frame, attributes = fields
            if feature.lower() != "exon":
                continue
            attrs = parse_attributes(attributes)
            exon_number_raw = _first(attrs, ["exon_number", "exon"])
            try:
                exon_number = int(exon_number_raw) if exon_number_raw is not None else None
            except ValueError:
                exon_number = None
            rows.append(
                {
                    "gene_symbol": _first(attrs, ["gene_name", "gene", "Name"]),
                    "gene_id": _first(attrs, ["gene_id", "geneID", "gene"]),
                    "transcript_id": _first(attrs, ["transcript_id", "transcript", "Parent"]),
                    "transcript_name": _first(attrs, ["transcript_name"]),
                    "feature": feature,
                    "chromosome": chrom,
                    "start": int(start),
                    "end": int(end),
                    "strand": strand,
                    "exon_number": exon_number,
                    "exon_id": _first(attrs, ["exon_id", "ID"]),
                    "tag": attrs.get("tag", ""),
                    "source": source,
                }
            )
    if not rows:
        raise AnnotationError(f"No exon records were found in annotation file: {annotation_path}")
    df = pd.DataFrame(rows)
    if df["transcript_id"].isna().any():
        raise AnnotationError("Some exon records are missing transcript_id. Use a richer GTF/GFF annotation.")
    return infer_missing_exon_numbers(df)


def infer_missing_exon_numbers(exons: pd.DataFrame) -> pd.DataFrame:
    exons = exons.copy()
    for transcript_id, group in exons.groupby("transcript_id"):
        if group["exon_number"].notna().all():
            continue
        strand = group["strand"].iloc[0]
        ascending = strand != "-"
        ordered = group.sort_values("start", ascending=ascending).index
        for number, idx in enumerate(ordered, start=1):
            if pd.isna(exons.at[idx, "exon_number"]):
                exons.at[idx, "exon_number"] = number
    exons["exon_number"] = exons["exon_number"].astype(int)
    return exons


def candidate_transcript_summary(exons: pd.DataFrame, gene_symbol: str, gene_id: str | None, exon_number: int) -> pd.DataFrame:
    subset = exons[exons["gene_symbol"] == gene_symbol]
    if gene_id:
        subset = subset[subset["gene_id"] == gene_id]
    if subset.empty:
        raise AnnotationError(
            f"Could not find gene_symbol={gene_symbol!r} in annotation. "
            "Confirm the annotation source and avoid mixing SMN1/SMN2 identifiers."
        )

    rows = []
    for transcript_id, group in subset.groupby("transcript_id"):
        exon = group[group["exon_number"] == exon_number]
        rows.append(
            {
                "gene_symbol": gene_symbol,
                "gene_id": group["gene_id"].dropna().iloc[0] if group["gene_id"].notna().any() else None,
                "transcript_id": transcript_id,
                "chromosome": group["chromosome"].iloc[0],
                "strand": group["strand"].iloc[0],
                "num_exons": int(group["exon_number"].nunique()),
                "exon7_start": int(exon["start"].iloc[0]) if not exon.empty else None,
                "exon7_end": int(exon["end"].iloc[0]) if not exon.empty else None,
                "exon7_length": int(exon["end"].iloc[0] - exon["start"].iloc[0] + 1) if not exon.empty else None,
                "annotation_confidence": "candidate",
                "notes": "canonical tag present" if group["tag"].astype(str).str.contains("canonical", case=False).any() else "",
            }
        )
    return pd.DataFrame(rows).sort_values(["transcript_id"])


def select_target(exons: pd.DataFrame, config: dict[str, Any]) -> tuple[SelectedTarget, pd.DataFrame]:
    target_cfg = config.get("target", {})
    gene_symbol = target_cfg.get("gene_symbol", "SMN2")
    gene_id = target_cfg.get("gene_id")
    transcript_id = target_cfg.get("transcript_id")
    exon_number = int(target_cfg.get("exon_number", 7))
    manual = target_cfg.get("manual_exon_coordinates")

    summary = candidate_transcript_summary(exons, gene_symbol, gene_id, exon_number)

    if manual:
        required = ["chromosome", "start", "end", "strand"]
        missing = [key for key in required if key not in manual or manual[key] is None]
        if missing:
            raise AnnotationError(f"manual_exon_coordinates is missing: {', '.join(missing)}")
        chosen_transcript = transcript_id or "manual_coordinates"
        return (
            SelectedTarget(
                gene_symbol=gene_symbol,
                gene_id=gene_id,
                transcript_id=chosen_transcript,
                chromosome=str(manual["chromosome"]),
                start=int(manual["start"]),
                end=int(manual["end"]),
                strand=str(manual["strand"]),
                exon_number=exon_number,
                annotation_confidence="manual",
                notes="User supplied manual exon coordinates because annotation selection may be ambiguous.",
            ),
            summary,
        )

    candidates = summary[summary["exon7_start"].notna()]
    if transcript_id:
        candidates = candidates[candidates["transcript_id"] == transcript_id]
        if candidates.empty:
            raise AnnotationError(
                f"transcript_id={transcript_id!r} was not found with exon {exon_number}. "
                "Check the transcript ID or provide manual_exon_coordinates."
            )
    elif len(candidates) > 1:
        canonical = candidates[candidates["notes"].str.contains("canonical", case=False, na=False)]
        if len(canonical) == 1:
            candidates = canonical
        else:
            raise AnnotationError(
                f"Multiple {gene_symbol} transcripts contain exon {exon_number}. "
                "Set target.transcript_id or target.manual_exon_coordinates in the config."
            )
    elif candidates.empty:
        raise AnnotationError(
            f"No {gene_symbol} transcript with exon {exon_number} could be identified. "
            "Specify target.transcript_id or target.manual_exon_coordinates."
        )

    row = candidates.iloc[0]
    return (
        SelectedTarget(
            gene_symbol=gene_symbol,
            gene_id=row["gene_id"],
            transcript_id=row["transcript_id"],
            chromosome=row["chromosome"],
            start=int(row["exon7_start"]),
            end=int(row["exon7_end"]),
            strand=row["strand"],
            exon_number=exon_number,
            annotation_confidence="annotation",
            notes=row["notes"] or "Selected from annotation using configured criteria.",
        ),
        summary,
    )
