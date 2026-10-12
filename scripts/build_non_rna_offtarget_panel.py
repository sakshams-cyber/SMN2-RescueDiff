from pathlib import Path
from datetime import datetime
import urllib.request
import subprocess
import shutil
import json
import math

import pandas as pd


# ============================================================
# FIXED 20-TARGET PANEL
#
# ligand_resname identifies the experimentally observed ligand
# used ONLY to define the binding pocket.
# ============================================================

PANEL = [
    # Nuclear receptors
    ("PROT_001", "AR",       "Androgen receptor",                    "2AM9", "TES", "nuclear_receptor"),
    ("PROT_002", "ESR1",     "Estrogen receptor alpha",              "1A52", "EST", "nuclear_receptor"),
    ("PROT_003", "PGR",      "Progesterone receptor",                 "1A28", "STR", "nuclear_receptor"),
    ("PROT_004", "PPARG",    "PPAR gamma",                            "2PRG", "BRL", "nuclear_receptor"),
    ("PROT_005", "RXRA",     "Retinoid X receptor alpha",             "1FBY", "9CR", "nuclear_receptor"),
    ("PROT_006", "THRB",     "Thyroid hormone receptor beta",         "1NAX", "IH5", "nuclear_receptor"),

    # Enzymes / binding proteins
    ("PROT_007", "ACHE",     "Acetylcholinesterase",                  "4EY7", "E20", "enzyme"),
    ("PROT_008", "HSP90AA1", "Heat shock protein 90",                 "1YET", "GDM", "chaperone"),
    ("PROT_009", "PARP1",    "PARP1 catalytic domain",                "4UND", "2YQ", "enzyme"),
    ("PROT_010", "BACE1",    "Beta-secretase 1",                      "2NTR", "L00", "protease"),
    ("PROT_011", "DPP4",     "Dipeptidyl peptidase 4",               "1X70", "715", "protease"),
    ("PROT_012", "CA2",      "Carbonic anhydrase II",                 "3HS4", "AZM", "enzyme"),
    ("PROT_013", "PDE5A",    "Phosphodiesterase 5A",                  "1UDT", "VIA", "enzyme"),
    ("PROT_014", "ALB",      "Human serum albumin",                   "2BXD", "RWF", "transport_protein"),

    # Signaling / epigenetic targets
    ("PROT_015", "CDK2",     "Cyclin-dependent kinase 2",             "1HCK", "ATP", "kinase"),
    ("PROT_016", "EGFR",     "EGFR kinase domain",                    "1M17", "AQ4", "kinase"),
    ("PROT_017", "MAPK14",   "p38 alpha MAP kinase",                  "1A9U", "SB2", "kinase"),
    ("PROT_018", "ROCK1",    "Rho-associated kinase 1",               "2ETR", "Y27", "kinase"),
    ("PROT_019", "KDR",      "VEGFR2 kinase domain",                  "4AG8", "AXI", "kinase"),
    ("PROT_020", "BRD4",     "BRD4 first bromodomain",                "3MXF", "JQ1", "epigenetic_reader"),
]


ROOT = Path(
    "data/processed/docking/cross_target_offtargets"
)

RAW_DIR = ROOT / "raw_pdb"
PREP_DIR = ROOT / "receptors"
LIG_DIR = ROOT / "reference_ligands"
LOG_DIR = ROOT / "preparation_logs"

MANIFEST = ROOT / "non_rna_receptor_panel_20.tsv"
FAILURES = ROOT / "panel_build_failures.tsv"
PROVENANCE = ROOT / "panel_provenance.json"

for directory in [
    RAW_DIR,
    PREP_DIR,
    LIG_DIR,
    LOG_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# FIND MEEKO
# ============================================================

meeko = (
    shutil.which("mk_prepare_receptor.py")
    or
    shutil.which("mk_prepare_receptor")
)

if meeko is None:
    raise SystemExit(
        "Meeko receptor preparation was not found.\n"
        "Try:\n"
        "python -m pip install -U meeko"
    )

print("Meeko:", meeko)

help_result = subprocess.run(
    [meeko, "-h"],
    capture_output=True,
    text=True,
)

help_text = (
    help_result.stdout
    + "\n"
    + help_result.stderr
)


# ============================================================
# PDB HELPERS
# ============================================================

def download_pdb(pdb_id):

    output = RAW_DIR / f"{pdb_id}.pdb"

    if output.exists() and output.stat().st_size > 1000:
        return output

    url = (
        f"https://files.rcsb.org/download/"
        f"{pdb_id}.pdb"
    )

    print("Downloading", pdb_id)

    urllib.request.urlretrieve(
        url,
        output,
    )

    if (
        not output.exists()
        or output.stat().st_size < 1000
    ):
        raise RuntimeError(
            f"Bad PDB download: {pdb_id}"
        )

    return output


def xyz(line):

    return (
        float(line[30:38]),
        float(line[38:46]),
        float(line[46:54]),
    )


def distance(a, b):

    return math.sqrt(
        (a[0] - b[0]) ** 2
        + (a[1] - b[1]) ** 2
        + (a[2] - b[2]) ** 2
    )


def residue_key(line):

    return (
        line[21:22],
        line[22:26],
        line[26:27],
    )


def resname(line):

    return line[17:20].strip()


# ============================================================
# BUILD EACH RECEPTOR
# ============================================================

records = []
failures = []

metal_resnames = {
    "ZN",
    "MG",
    "MN",
    "FE",
    "CU",
    "CO",
    "NI",
}

for (
    receptor_id,
    gene,
    target_name,
    pdb_id,
    ligand_resname,
    receptor_class,
) in PANEL:

    print("\n" + "=" * 70)
    print(receptor_id, gene, pdb_id)
    print("=" * 70)

    try:

        pdb_path = download_pdb(
            pdb_id
        )

        lines = pdb_path.read_text(
            errors="ignore"
        ).splitlines()


        # ----------------------------------------------------
        # Locate all copies of reference ligand
        # ----------------------------------------------------

        ligand_candidates = {}

        for line in lines:

            if not line.startswith("HETATM"):
                continue

            if resname(line) != ligand_resname:
                continue

            key = residue_key(line)

            ligand_candidates.setdefault(
                key,
                [],
            ).append(line)


        if not ligand_candidates:

            available = sorted(
                {
                    resname(line)
                    for line in lines
                    if line.startswith("HETATM")
                }
            )

            raise RuntimeError(
                f"Reference ligand {ligand_resname} "
                f"not found. HETATM residue names: "
                f"{available}"
            )


        # Use first ligand instance.
        selected_key = sorted(
            ligand_candidates
        )[0]

        ligand_lines = (
            ligand_candidates[
                selected_key
            ]
        )

        ligand_coords = [
            xyz(line)
            for line in ligand_lines
        ]


        # ----------------------------------------------------
        # Determine nearest protein chain
        # ----------------------------------------------------

        chain_atoms = {}

        for line in lines:

            if not line.startswith("ATOM"):
                continue

            chain = (
                line[21:22].strip()
                or "_"
            )

            chain_atoms.setdefault(
                chain,
                [],
            ).append(
                xyz(line)
            )


        if not chain_atoms:
            raise RuntimeError(
                "No protein ATOM records."
            )


        chain_distances = {}

        for chain, atoms in chain_atoms.items():

            min_d = min(
                distance(
                    ligand_atom,
                    protein_atom,
                )
                for ligand_atom
                in ligand_coords
                for protein_atom
                in atoms
            )

            chain_distances[chain] = min_d


        protein_chain = min(
            chain_distances,
            key=chain_distances.get,
        )

        print(
            "Selected protein chain:",
            protein_chain
        )

        print(
            "Ligand-to-chain minimum distance:",
            round(
                chain_distances[
                    protein_chain
                ],
                3,
            ),
        )


        # ----------------------------------------------------
        # Write native reference ligand
        # ----------------------------------------------------

        ligand_file = (
            LIG_DIR
            / f"{receptor_id}_{pdb_id}_{ligand_resname}.pdb"
        )

        ligand_file.write_text(
            "\n".join(
                ligand_lines
                + ["END"]
            )
            + "\n"
        )


        # ----------------------------------------------------
        # Write single-chain protein receptor.
        #
        # Keep nearby structural metal ions.
        # ----------------------------------------------------

        receptor_lines = []

        for line in lines:

            if line.startswith("ATOM"):

                chain = (
                    line[21:22].strip()
                    or "_"
                )

                if chain == protein_chain:
                    receptor_lines.append(
                        line
                    )

            elif line.startswith("HETATM"):

                rn = resname(line)

                if rn not in metal_resnames:
                    continue

                coord = xyz(line)

                min_ligand_distance = min(
                    distance(
                        coord,
                        ligand_coord,
                    )
                    for ligand_coord
                    in ligand_coords
                )

                if min_ligand_distance <= 12.0:

                    receptor_lines.append(
                        line
                    )


        if not receptor_lines:
            raise RuntimeError(
                "No receptor atoms selected."
            )


        receptor_pdb = (
            PREP_DIR
            / f"{receptor_id}_{gene}_{pdb_id}.pdb"
        )

        receptor_pdb.write_text(
            "\n".join(
                receptor_lines
                + ["END"]
            )
            + "\n"
        )


        # ----------------------------------------------------
        # Prepare PDBQT with Meeko
        # ----------------------------------------------------

        basename = (
            PREP_DIR
            / f"{receptor_id}_{gene}_{pdb_id}"
        )

        command = [
            meeko,
            "--read_pdb",
            str(receptor_pdb),
            "-o",
            str(basename),
            "-p",
        ]

        if "--allow_bad_res" in help_text:
            command.append(
                "--allow_bad_res"
            )

        if "--default_altloc" in help_text:
            command += [
                "--default_altloc",
                "A",
            ]


        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=600,
        )


        log_file = (
            LOG_DIR
            / f"{receptor_id}_{gene}_{pdb_id}.log"
        )

        log_file.write_text(
            "COMMAND:\n"
            + " ".join(command)
            + "\n\nSTDOUT:\n"
            + result.stdout
            + "\n\nSTDERR:\n"
            + result.stderr
        )


        receptor_pdbqt = Path(
            str(basename)
            + ".pdbqt"
        )


        # Some versions may write _rigid.pdbqt.
        if not receptor_pdbqt.exists():

            rigid_candidate = Path(
                str(basename)
                + "_rigid.pdbqt"
            )

            if rigid_candidate.exists():
                receptor_pdbqt = (
                    rigid_candidate
                )


        if (
            result.returncode != 0
            or not receptor_pdbqt.exists()
            or receptor_pdbqt.stat().st_size == 0
        ):

            raise RuntimeError(
                "Meeko receptor preparation failed. "
                f"Inspect {log_file}"
            )


        # ----------------------------------------------------
        # Define docking box directly from experimental ligand.
        #
        # 6 A padding on every side.
        # Minimum box dimension 20 A.
        # ----------------------------------------------------

        xs = [
            p[0]
            for p in ligand_coords
        ]

        ys = [
            p[1]
            for p in ligand_coords
        ]

        zs = [
            p[2]
            for p in ligand_coords
        ]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        min_z, max_z = min(zs), max(zs)

        center_x = (
            min_x + max_x
        ) / 2.0

        center_y = (
            min_y + max_y
        ) / 2.0

        center_z = (
            min_z + max_z
        ) / 2.0

        padding = 6.0

        size_x = max(
            20.0,
            max_x - min_x
            + 2 * padding,
        )

        size_y = max(
            20.0,
            max_y - min_y
            + 2 * padding,
        )

        size_z = max(
            20.0,
            max_z - min_z
            + 2 * padding,
        )


        records.append({

            "receptor_id":
                receptor_id,

            "gene":
                gene,

            "target_name":
                target_name,

            "receptor_class":
                receptor_class,

            "pdb_id":
                pdb_id,

            "reference_ligand_resname":
                ligand_resname,

            "protein_chain":
                protein_chain,

            "reference_ligand_file":
                str(ligand_file),

            "receptor_pdb_file":
                str(receptor_pdb),

            "receptor_file":
                str(receptor_pdbqt),

            "center_x":
                round(center_x, 3),

            "center_y":
                round(center_y, 3),

            "center_z":
                round(center_z, 3),

            "size_x":
                round(size_x, 3),

            "size_y":
                round(size_y, 3),

            "size_z":
                round(size_z, 3),

            "preparation_status":
                "completed",

            "preparation_log":
                str(log_file),
        })


        print("Prepared:", receptor_pdbqt)

        print(
            "Box center:",
            round(center_x, 2),
            round(center_y, 2),
            round(center_z, 2),
        )

        print(
            "Box size:",
            round(size_x, 2),
            round(size_y, 2),
            round(size_z, 2),
        )


    except Exception as exc:

        print("FAILED:", exc)

        failures.append({

            "receptor_id":
                receptor_id,

            "gene":
                gene,

            "pdb_id":
                pdb_id,

            "reference_ligand_resname":
                ligand_resname,

            "error":
                str(exc),
        })


# ============================================================
# WRITE RESULTS
# ============================================================

manifest = pd.DataFrame(
    records
)

manifest.to_csv(
    MANIFEST,
    sep="\t",
    index=False,
)

failure_df = pd.DataFrame(
    failures
)

failure_df.to_csv(
    FAILURES,
    sep="\t",
    index=False,
)


print("\n======================================")
print("NON-RNA RECEPTOR PANEL BUILD")
print("======================================")

print(
    "Requested receptors:",
    len(PANEL),
)

print(
    "Successfully prepared:",
    len(manifest),
)

print(
    "Failed:",
    len(failures),
)


if len(manifest):

    print(
        "Unique receptor IDs:",
        manifest[
            "receptor_id"
        ].nunique(),
    )

    print(
        "Unique PDB structures:",
        manifest[
            "pdb_id"
        ].nunique(),
    )


provenance = {

    "created_at":
        datetime.now().isoformat(),

    "panel_name":
        "non_rna_human_offtarget_panel_20_v1",

    "requested_receptors":
        len(PANEL),

    "successfully_prepared":
        len(manifest),

    "failed":
        len(failures),

    "box_definition":
        (
            "Box centered on experimentally observed "
            "co-crystallized ligand. Six Angstrom padding "
            "on each side; minimum dimension 20 Angstrom."
        ),

    "manifest":
        str(MANIFEST),

    "scientific_role":
        (
            "Orthogonal non-RNA macromolecular "
            "off-target docking screen."
        ),
}


with PROVENANCE.open(
    "w"
) as handle:

    json.dump(
        provenance,
        handle,
        indent=2,
    )


if len(failures):

    print("\nPANEL BUILD INCOMPLETE.")
    print("Inspect:")
    print(FAILURES)

    raise SystemExit(1)


if len(manifest) != 20:

    raise SystemExit(
        "Expected exactly 20 receptors."
    )


print("\n======================================")
print("20-RECEPTOR PANEL: PASSED")
print("======================================")

print("\nManifest:")
print(MANIFEST)
