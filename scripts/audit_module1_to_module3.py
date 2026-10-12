from pathlib import Path
import re
import pandas as pd
import yaml

ROOT = Path(".")
CONFIG = Path(
    "configs/presets/module3_final_final_rigorous_x10.yaml"
)

OUT = Path(
    "data/processed/interpretability/module1_to_module3"
)
OUT.mkdir(parents=True, exist_ok=True)

if not CONFIG.exists():
    raise SystemExit(f"Missing config: {CONFIG}")

cfg = yaml.safe_load(CONFIG.read_text())

profile = Path(
    cfg["inputs"]["optional_context"]["smn2_target_profile"]
)

metadata = Path(
    cfg["inputs"]["optional_context"]["smn2_target_metadata"]
)

print("Module 1 target profile:", profile)
print("Exists:", profile.exists())

print("Module 1 metadata:", metadata)
print("Exists:", metadata.exists())

src = Path(
    "src/smn2_rescuediff/molecule_generation"
)

patterns = [
    "smn2_target_profile",
    "smn2_target_metadata",
    "target_profile",
    "target_features",
    "optional_context",
    "context",
    "conditioning",
    "condition",
]

records = []

for pyfile in src.rglob("*.py"):

    try:
        lines = pyfile.read_text(
            errors="ignore"
        ).splitlines()
    except Exception:
        continue

    for lineno, line in enumerate(lines, start=1):

        hits = [
            p for p in patterns
            if p.lower() in line.lower()
        ]

        if hits:

            records.append({
                "source_file": str(pyfile),
                "line_number": lineno,
                "matched_terms": ";".join(hits),
                "source_line": line.strip(),
            })

audit = pd.DataFrame(records)

audit_file = OUT / "module1_module3_static_dependency_audit.tsv"

audit.to_csv(
    audit_file,
    sep="\t",
    index=False,
)

print("\nStatic references found:", len(audit))

if len(audit):
    print(
        audit.to_string(index=False)
    )

# Profile columns
if profile.exists():

    target = pd.read_csv(
        profile,
        sep="\t",
        low_memory=False,
    )

    feature_rows = []

    for col in target.columns:

        numeric = pd.to_numeric(
            target[col],
            errors="coerce",
        )

        feature_rows.append({
            "feature": col,
            "dtype": str(target[col].dtype),
            "numeric_values": int(numeric.notna().sum()),
            "rows": len(target),
            "potential_shap_feature":
                bool(numeric.notna().all()),
        })

    features = pd.DataFrame(feature_rows)

    features.to_csv(
        OUT / "module1_feature_inventory.tsv",
        sep="\t",
        index=False,
    )

    print("\nModule 1 feature inventory:")
    print(features.to_string(index=False))

print("\nWrote:")
print(audit_file)
