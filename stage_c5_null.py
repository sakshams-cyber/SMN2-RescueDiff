#!/usr/bin/env python3
"""Stage C.5 selection-effect null.

Stage C.5 compared the 40 top-ranked candidates against property-matched ligand
decoy controls and found every one of them bound the SMN2 conformer ensemble
more strongly than its matched-control median. Because the lexicographic
ranking that produced those 40 rewards strong target binding, part of that
enrichment could be an artifact of selection rather than a property of the
molecules.

This script quantifies that. It draws random cohorts of 40 from the unselected
pool, matches each member to ten property-matched controls under the identical
Stage C.5 protocol, and reports the distribution of the cohort-level mean target
enrichment gap. The observed Stage C.5 value is then placed against that null.

No docking is performed and no score is imputed. Run from the project root with
the project environment active.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DESCRIPTORS = [
    "molecular_weight", "logp", "hbd", "hba", "tpsa",
    "rotatable_bonds", "formal_charge", "qed",
    "synthetic_accessibility_proxy",
]


def load_pool(args) -> pd.DataFrame:
    """500-molecule population minus the Stage C.5 candidates."""
    for p in (args.selected_molecules, args.target_summary, args.top40):
        if not Path(p).is_file():
            sys.exit(f"missing input: {p}")

    sel = pd.read_csv(args.selected_molecules, sep="\t")
    tgt = pd.read_csv(args.target_summary, sep="\t")
    top40 = pd.read_csv(args.top40, sep="\t")

    missing = [c for c in DESCRIPTORS + ["molecule_id", "canonical_smiles"]
               if c not in sel.columns]
    if missing:
        sys.exit(f"{args.selected_molecules} lacks columns: {missing}")

    df = sel.merge(tgt[["molecule_id", "mean_target_deltaG"]],
                   on="molecule_id", how="inner")
    before = len(df)
    df = df.dropna(subset=DESCRIPTORS + ["mean_target_deltaG", "canonical_smiles"])
    if len(df) < before:
        print(f"dropped {before - len(df)} molecules with missing descriptors "
              f"or target scores", file=sys.stderr)

    pool = df[~df["molecule_id"].isin(set(top40["molecule_id"]))]
    return pool.reset_index(drop=True)


def tanimoto_matrix(smiles: list[str], radius: int, n_bits: int) -> np.ndarray:
    """Full pairwise Tanimoto similarity, computed once.

    Returns NaN for any molecule RDKit could not parse, so the caller can treat
    an unparseable control as unguardable rather than silently similar.
    """
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps, ok = [], []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        fps.append(None if mol is None else gen.GetFingerprint(mol))
        ok.append(mol is not None)

    n = len(fps)
    S = np.full((n, n), np.nan, dtype=float)
    valid = [i for i in range(n) if ok[i]]
    vfps = [fps[i] for i in valid]
    for pos, i in enumerate(valid):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], vfps)
        S[i, valid] = sims
    n_bad = n - len(valid)
    if n_bad:
        print(f"{n_bad} SMILES could not be parsed; those pairs are unguarded",
              file=sys.stderr)
    return S


def cohort_mean_gap(cand_idx, pool_idx, Z, target, S, max_tanimoto, k):
    """Mean enrichment gap over one cohort, or None if a candidate can't match."""
    gaps = []
    for ci in cand_idx:
        d = np.linalg.norm(Z[pool_idx] - Z[ci], axis=1)
        order = np.argsort(d, kind="stable")
        if S is None:
            chosen = pool_idx[order[:k]]
            if len(chosen) < k:
                return None
        else:
            sims = S[ci, pool_idx[order]]
            allowed = ~(sims > max_tanimoto)          # NaN stays allowed
            picked = pool_idx[order][allowed][:k]
            if len(picked) < k:
                return None
            chosen = picked
        gaps.append(np.median(target[chosen]) - target[ci])
    return float(np.mean(gaps))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selected-molecules", required=True,
                   help="selected_molecules.tsv: 500 molecules, SMILES + 9 descriptors")
    p.add_argument("--target-summary", required=True,
                   help="target_conformer_robustness_scores_final.tsv: mean_target_deltaG")
    p.add_argument("--top40", required=True,
                   help="top40_for_ligand_decoy_control_audit.tsv: the Stage C.5 candidates")
    p.add_argument("--observed-mean-gap", type=float, default=0.7292,
                   help="Stage C.5 mean target enrichment gap (default: %(default)s)")
    p.add_argument("--cohort-size", type=int, default=40)
    p.add_argument("--controls-per-candidate", type=int, default=10)
    p.add_argument("--max-tanimoto", type=float, default=0.60)
    p.add_argument("--morgan-radius", type=int, default=2)
    p.add_argument("--morgan-bits", type=int, default=2048)
    p.add_argument("--replicates", type=int, default=2000)
    p.add_argument("--random-seed", type=int, default=61453)
    p.add_argument("--no-tanimoto-guard", action="store_true",
                   help="skip the fingerprint guard (RDKit then not required)")
    p.add_argument("--out", type=Path, default=Path("stage_c5_selection_null.json"))
    args = p.parse_args()

    pool = load_pool(args)
    n = len(pool)
    print(f"null pool: {n} molecules (500-molecule population minus the "
          f"{args.cohort_size} Stage C.5 candidates)")
    if n < args.cohort_size + args.controls_per_candidate + 1:
        sys.exit(f"pool of {n} too small for cohorts of {args.cohort_size}")

    target = pool["mean_target_deltaG"].to_numpy(float)
    X = pool[DESCRIPTORS].to_numpy(float)
    sd = X.std(axis=0, ddof=0)
    zero = sd == 0
    if zero.any():
        print(f"constant descriptors, excluded from the distance: "
              f"{[d for d, z in zip(DESCRIPTORS, zero) if z]}", file=sys.stderr)
        sd[zero] = 1.0
    Z = (X - X.mean(axis=0)) / sd

    S = None
    if not args.no_tanimoto_guard:
        print("computing pairwise Morgan similarities...")
        S = tanimoto_matrix(list(pool["canonical_smiles"]),
                            args.morgan_radius, args.morgan_bits)

    rng = np.random.default_rng(args.random_seed)
    all_idx = np.arange(n)
    draws, failed = [], 0
    for r in range(args.replicates):
        cand = rng.choice(all_idx, size=args.cohort_size, replace=False)
        rest = np.setdiff1d(all_idx, cand, assume_unique=False)
        g = cohort_mean_gap(cand, rest, Z, target, S,
                            args.max_tanimoto, args.controls_per_candidate)
        if g is None:
            failed += 1
        else:
            draws.append(g)
        if (r + 1) % 250 == 0:
            print(f"  {r + 1}/{args.replicates} cohorts", flush=True)

    a = np.array(draws, dtype=float)
    if a.size < 2:
        sys.exit("too few cohorts completed to summarize")

    obs = args.observed_mean_gap
    result = {
        "null_pool_size": int(n),
        "replicates_requested": int(args.replicates),
        "replicates_completed": int(a.size),
        "replicates_failed_to_match": int(failed),
        "cohort_size": int(args.cohort_size),
        "controls_per_candidate": int(args.controls_per_candidate),
        "tanimoto_guard_applied": not args.no_tanimoto_guard,
        "max_tanimoto": args.max_tanimoto,
        "morgan_radius": args.morgan_radius,
        "morgan_bits": args.morgan_bits,
        "matching_descriptors": DESCRIPTORS,
        "null_mean": float(a.mean()),
        "null_sd": float(a.std(ddof=1)),
        "null_min": float(a.min()),
        "null_max": float(a.max()),
        "null_quantiles": {q: float(np.quantile(a, v)) for q, v in
                           [("0.025", .025), ("0.50", .50), ("0.95", .95),
                            ("0.975", .975), ("0.99", .99)]},
        "observed_mean_gap": obs,
        "observed_z_vs_null": float((obs - a.mean()) / a.std(ddof=1)),
        "cohorts_at_or_above_observed": int(np.sum(a >= obs)),
        "empirical_p_observed_ge_null": float((np.sum(a >= obs) + 1) / (a.size + 1)),
        "random_seed": int(args.random_seed),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
