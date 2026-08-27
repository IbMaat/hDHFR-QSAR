# -*- coding: utf-8 -*-
"""
Part 3 - Molecular Fingerprint Calculation
Human Dihydrofolate Reductase (hDHFR)
Target: CHEMBL202

Fingerprints:
1. PubChem
2. Substructure
3. MACCS
"""

# ============================================================
# STEP 1: INSTALL PADELPY
# ============================================================

!pip install -q padelpy


# ============================================================
# STEP 2: IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import os
import glob


# ============================================================
# STEP 3: PREPARE PaDEL FINGERPRINT XML FILES
# ============================================================

!wget -q https://github.com/dataprofessor/padel/raw/main/fingerprints_xml.zip
!unzip -o fingerprints_xml.zip


# List XML files
xml_files = glob.glob("*.xml")
xml_files.sort()

print("XML files found:")
for x in xml_files:
    print(x)


# ============================================================
# STEP 4: MAP FINGERPRINT NAMES TO XML FILES
# ============================================================

fingerprint_names = [
    "AtomPairs2DCount",
    "AtomPairs2D",
    "EState",
    "CDKextended",
    "CDK",
    "CDKgraphonly",
    "KlekotaRothCount",
    "KlekotaRoth",
    "MACCS",
    "PubChem",
    "SubstructureCount",
    "Substructure"
]

fp = dict(zip(fingerprint_names, xml_files))

print("\nAvailable fingerprints:")
print(fp)


# ============================================================
# STEP 5: SELECT THE THREE FINGERPRINTS
# ============================================================

selected_fingerprints = [
    "PubChem",
    "Substructure",
    "MACCS"
]

for fingerprint in selected_fingerprints:
    print(
        fingerprint,
        "->",
        fp[fingerprint]
    )


# ============================================================
# STEP 6: IMPORT hDHFR DATASET
# ============================================================

df = pd.read_csv(
    "/content/hDHFR_dataset_3classes.csv"
)

print("\nDataset shape:")
print(df.shape)

display(df.head())


# ============================================================
# STEP 7: PREPARE INPUT FOR PaDEL
# ============================================================

df_padel = df[
    [
        "canonical_smiles",
        "molecule_chembl_id"
    ]
].copy()

# Remove missing SMILES
df_padel = df_padel.dropna(
    subset=["canonical_smiles"]
).reset_index(drop=True)

# Save as .smi
df_padel.to_csv(
    "/content/molecule.smi",
    sep="\t",
    index=False,
    header=False
)

print(
    "PaDEL input file created:"
    "/content/molecule.smi"
)

print(
    "Number of molecules:",
    len(df_padel)
)


# ============================================================
# STEP 8: CALCULATE FINGERPRINTS
# ============================================================

from padelpy import padeldescriptor


for fingerprint in selected_fingerprints:

    print("\n========================================")
    print("Calculating:", fingerprint)
    print("========================================")

    output_file = (
        "/content/"
        + fingerprint.lower()
        + "_descriptors_3classes.csv"
    )

    descriptor_file = fp[fingerprint]

    padeldescriptor(
        mol_dir="/content/molecule.smi",
        d_file=output_file,
        descriptortypes=descriptor_file,
        detectaromaticity=True,
        standardizenitro=True,
        standardizetautomers=True,
        threads=2,
        removesalt=True,
        log=True,
        fingerprints=True
    )

    print(
        "Saved:",
        output_file
    )


# ============================================================
# STEP 9: LOAD THE THREE FINGERPRINT DATASETS
# ============================================================

pubchem = pd.read_csv(
    "/content/pubchem_descriptors_3classes.csv"
)

substructure = pd.read_csv(
    "/content/substructure_descriptors_3classes.csv"
)

maccs = pd.read_csv(
    "/content/maccs_descriptors_3classes.csv"
)


# ============================================================
# STEP 10: DISPLAY DIMENSIONS
# ============================================================

print("PubChem shape:", pubchem.shape)
print("Substructure shape:", substructure.shape)
print("MACCS shape:", maccs.shape)


# ============================================================
# STEP 11: CHECK FIRST COLUMNS
# ============================================================

print("\nPubChem:")
display(pubchem.head())

print("\nSubstructure:")
display(substructure.head())

print("\nMACCS:")
display(maccs.head())


# ============================================================
# STEP 12: CHECK MOLECULE COUNTS
# ============================================================

print(
    "\nNumber of molecules in original dataset:",
    len(df)
)

print(
    "Number of molecules in PubChem:",
    len(pubchem)
)

print(
    "Number of molecules in Substructure:",
    len(substructure)
)

print(
    "Number of molecules in MACCS:",
    len(maccs)
)


# ============================================================
# STEP 13: CHECK DUPLICATE IDS
# ============================================================

for name, data in [
    ("PubChem", pubchem),
    ("Substructure", substructure),
    ("MACCS", maccs)
]:

    if "Name" in data.columns:

        print(
            f"{name} duplicated IDs:",
            data["Name"].duplicated().sum()
        )


# ============================================================
# STEP 14: SAVE / RENAME FINAL FILES
# ============================================================

pubchem.to_csv(
    "/content/hDHFR_PubChem_3classes.csv",
    index=False
)

substructure.to_csv(
    "/content/hDHFR_Substructure_3classes.csv",
    index=False
)

maccs.to_csv(
    "/content/hDHFR_MACCS_3classes.csv",
    index=False
)


# ============================================================
# STEP 15: FINAL SUMMARY
# ============================================================

print("\n============================================")
print("hDHFR FINGERPRINT CALCULATION COMPLETED")
print("============================================")

print("Target: CHEMBL202")
print("Dataset:", len(df), "molecules")

print("\nFingerprints generated:")

print(
    "1. PubChem      :",
    pubchem.shape
)

print(
    "2. Substructure :",
    substructure.shape
)

print(
    "3. MACCS        :",
    maccs.shape
)

print("\nFiles saved:")
print("hDHFR_PubChem_3classes.csv")
print("hDHFR_Substructure_3classes.csv")
print("hDHFR_MACCS_3classes.csv")


"""
Combining Molecular Fingerprints and Biological Activity
Human Dihydrofolate Reductase (hDHFR)
Target: CHEMBL202

Fingerprints:
- PubChem
- Substructure
- MACCS
"""

# ============================================================
# STEP 1: IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np


# ============================================================
# STEP 2: IMPORT THE BIOACTIVITY DATASET
# ============================================================

df_bio = pd.read_csv(
    "/content/hDHFR_dataset_3classes.csv"
)

print("Bioactivity dataset:")
print(df_bio.shape)

display(df_bio.head())


# ============================================================
# STEP 3: CHECK REQUIRED BIOACTIVITY COLUMNS
# ============================================================

required_columns = [
    "molecule_chembl_id",
    "canonical_smiles",
    "pIC50"
]

missing = [
    col for col in required_columns
    if col not in df_bio.columns
]

if missing:
    raise ValueError(
        f"Missing columns in bioactivity dataset: {missing}"
    )


# ============================================================
# STEP 4: PREPARE BIOACTIVITY DATA
# ============================================================

df_bio = df_bio[
    [
        "molecule_chembl_id",
        "canonical_smiles",
        "pIC50"
    ]
].copy()

# Remove missing values
df_bio = df_bio.dropna(
    subset=["molecule_chembl_id", "pIC50"]
).reset_index(drop=True)

print(
    "Bioactivity dataset after cleaning:",
    df_bio.shape
)


# ============================================================
# STEP 5: FUNCTION TO PREPARE PaDEL DATA
# ============================================================

def prepare_fingerprint_file(
    fingerprint_file,
    fingerprint_name
):

    fp = pd.read_csv(fingerprint_file)

    print("\n======================================")
    print(f"{fingerprint_name} fingerprint")
    print("======================================")

    print("Original shape:", fp.shape)

    display(fp.head())

    # PaDEL normally uses "Name" as molecule identifier
    if "Name" not in fp.columns:

        raise ValueError(
            f"'Name' column not found in {fingerprint_name} file."
        )

    # Rename Name → molecule_chembl_id
    fp = fp.rename(
        columns={
            "Name": "molecule_chembl_id"
        }
    )

    # Remove Unnamed index column if present
    unnamed_columns = [
        col for col in fp.columns
        if col.startswith("Unnamed:")
    ]

    fp = fp.drop(
        columns=unnamed_columns,
        errors="ignore"
    )

    return fp


# ============================================================
# STEP 6: LOAD THE THREE FINGERPRINT DATASETS
# ============================================================

pubchem = prepare_fingerprint_file(
    "/content/pubchem_descritores_3classes.csv",
    "PubChem"
)

substructure = prepare_fingerprint_file(
    "/content/substructure_descritores_3classes.csv",
    "Substructure"
)

maccs = prepare_fingerprint_file(
    "/content/maccs_descritores_3classes.csv",
    "MACCS"
)


# ============================================================
# STEP 7: FUNCTION TO MERGE FINGERPRINTS WITH pIC50
# ============================================================

def combine_fingerprint_with_activity(
    fingerprint_df,
    bioactivity_df,
    fingerprint_name
):

    print("\n======================================")
    print(f"Combining {fingerprint_name} + pIC50")
    print("======================================")

    # Check duplicate molecule IDs
    duplicates = fingerprint_df[
        fingerprint_df["molecule_chembl_id"].duplicated(
            keep=False
        )
    ]

    if len(duplicates) > 0:

        print(
            f"WARNING: {len(duplicates)} duplicated "
            f"rows in {fingerprint_name}"
        )

        display(duplicates)

        fingerprint_df = fingerprint_df.drop_duplicates(
            subset="molecule_chembl_id",
            keep="first"
        )

    # Merge using molecule ID
    combined = pd.merge(
        fingerprint_df,
        bioactivity_df[
            [
                "molecule_chembl_id",
                "pIC50"
            ]
        ],
        on="molecule_chembl_id",
        how="inner"
    )

    # Reset index
    combined = combined.reset_index(drop=True)

    print(
        f"{fingerprint_name} original molecules:",
        len(fingerprint_df)
    )

    print(
        f"Molecules after merging:",
        len(combined)
    )

    print(
        f"Number of fingerprint features:",
        combined.shape[1] - 2
    )

    return combined


# ============================================================
# STEP 8: COMBINE PUBCHEM + pIC50
# ============================================================

df_pubchem = combine_fingerprint_with_activity(
    pubchem,
    df_bio,
    "PubChem"
)


# ============================================================
# STEP 9: COMBINE SUBSTRUCTURE + pIC50
# ============================================================

df_substructure = combine_fingerprint_with_activity(
    substructure,
    df_bio,
    "Substructure"
)


# ============================================================
# STEP 10: COMBINE MACCS + pIC50
# ============================================================

df_maccs = combine_fingerprint_with_activity(
    maccs,
    df_bio,
    "MACCS"
)


# ============================================================
# STEP 11: VERIFY pIC50 ALIGNMENT
# ============================================================

print("\n======================================")
print("ALIGNMENT CHECK")
print("======================================")


for name, data in [
    ("PubChem", df_pubchem),
    ("Substructure", df_substructure),
    ("MACCS", df_maccs)
]:

    print(f"\n{name}")

    print(
        "Rows:",
        len(data)
    )

    print(
        "Unique molecules:",
        data["molecule_chembl_id"].nunique()
    )

    print(
        "Missing pIC50:",
        data["pIC50"].isna().sum()
    )

    print(
        "Duplicate IDs:",
        data["molecule_chembl_id"].duplicated().sum()
    )


# ============================================================
# STEP 12: CHECK THAT THE MOLECULES ARE IDENTICAL
# ============================================================

pubchem_ids = set(
    df_pubchem["molecule_chembl_id"]
)

substructure_ids = set(
    df_substructure["molecule_chembl_id"]
)

maccs_ids = set(
    df_maccs["molecule_chembl_id"]
)

print("\n======================================")
print("MOLECULE SET COMPARISON")
print("======================================")

print(
    "PubChem vs Substructure:",
    pubchem_ids == substructure_ids
)

print(
    "PubChem vs MACCS:",
    pubchem_ids == maccs_ids
)

print(
    "Substructure vs MACCS:",
    substructure_ids == maccs_ids
)


# ============================================================
# STEP 13: SAVE FINAL DATASETS
# ============================================================

df_pubchem.to_csv(
    "/content/hDHFR_PubChem_final.csv",
    index=False
)

df_substructure.to_csv(
    "/content/hDHFR_Substructure_final.csv",
    index=False
)

df_maccs.to_csv(
    "/content/hDHFR_MACCS_final.csv",
    index=False
)


# ============================================================
# STEP 14: DISPLAY FINAL DATASETS
# ============================================================

print("\n======================================")
print("FINAL DATASETS")
print("======================================")

print(
    "PubChem:",
    df_pubchem.shape
)

print(
    "Substructure:",
    df_substructure.shape
)

print(
    "MACCS:",
    df_maccs.shape
)

display(df_pubchem.head())



