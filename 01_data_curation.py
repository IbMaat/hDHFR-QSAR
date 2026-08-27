# -*- coding: utf-8 -*-

# ============================================================
# hDHFR QSAR - ChEMBL Data Collection and Preprocessing
# Target: Human Dihydrofolate Reductase (hDHFR)
# ChEMBL Target ID: CHEMBL202
# ============================================================


# ============================================================
# STEP 1: INSTALLATION
# ============================================================

!pip install -q chembl_webresource_client


# ============================================================
# STEP 2: IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np

from chembl_webresource_client.new_client import new_client


# ============================================================
# STEP 3: DEFINE hDHFR TARGET
# ============================================================

target_chembl_id = "CHEMBL202"

print("Selected target:", target_chembl_id)


# ============================================================
# STEP 4: RETRIEVE IC50 DATA
# ============================================================

activity = new_client.activity

results = (
    activity
    .filter(target_chembl_id=target_chembl_id)
    .filter(standard_type="IC50")
)

df = pd.DataFrame.from_dict(results)

print("Raw activity records:", len(df))

display(df.head())


# ============================================================
# STEP 5: KEEP RELEVANT COLUMNS
# ============================================================

columns_needed = [
    "molecule_chembl_id",
    "canonical_smiles",
    "standard_value",
    "standard_type",
    "standard_units"
]

columns_available = [
    col for col in columns_needed
    if col in df.columns
]

df = df[columns_available].copy()

print("Columns retained:")
print(df.columns.tolist())


# ============================================================
# STEP 6: KEEP IC50 VALUES IN nM
# ============================================================

df["standard_units"] = (
    df["standard_units"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df = df[
    df["standard_units"] == "nm"
].copy()

print("Records with IC50 in nM:", len(df))


# ============================================================
# STEP 7: REMOVE MISSING VALUES
# ============================================================

df = df.dropna(
    subset=[
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_value"
    ]
).copy()

print("After removing missing values:", len(df))


# ============================================================
# STEP 8: CONVERT IC50 TO NUMERIC
# ============================================================

df["standard_value"] = pd.to_numeric(
    df["standard_value"],
    errors="coerce"
)

df = df.dropna(
    subset=["standard_value"]
).copy()

# Keep only positive IC50 values
df = df[
    df["standard_value"] > 0
].copy()

print("After numeric filtering:", len(df))


# ============================================================
# STEP 9: CHECK DUPLICATE CHEMBL IDs
# ============================================================

duplicate_ids = df[
    df.duplicated(
        subset="molecule_chembl_id",
        keep=False
    )
].sort_values("molecule_chembl_id")

print(
    "Number of duplicated ChEMBL ID rows:",
    len(duplicate_ids)
)

print(
    "Number of duplicated ChEMBL IDs:",
    duplicate_ids["molecule_chembl_id"].nunique()
)

display(duplicate_ids)


# ============================================================
# STEP 10: REMOVE DUPLICATE CHEMBL IDs
# ============================================================

df = df.drop_duplicates(
    subset="molecule_chembl_id",
    keep="first"
).reset_index(drop=True)

print(
    "Records after removing duplicate ChEMBL IDs:",
    len(df)
)


# ============================================================
# STEP 11: ASSIGN BIOACTIVITY CLASS
# ============================================================

def assign_bioactivity_class(ic50):

    if ic50 < 1000:
        return "Active"

    elif ic50 >= 10000:
        return "Inactive"

    else:
        return "Intermediate"


df["bioactivity_class"] = (
    df["standard_value"]
    .apply(assign_bioactivity_class)
)


# ============================================================
# STEP 12: CALCULATE pIC50
# ============================================================

# IC50 is expressed in nM.
#
# pIC50 = -log10(IC50 in mol/L)
#
# Therefore:
#
# pIC50 = 9 - log10(IC50 in nM)

df["pIC50"] = (
    9 - np.log10(df["standard_value"])
)


# ============================================================
# STEP 13: FINAL DATAFRAME
# ============================================================

df_final = df[
    [
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_value",
        "bioactivity_class",
        "pIC50"
    ]
].copy()


# ============================================================
# STEP 14: SORT BY pIC50
# ============================================================

df_final = df_final.sort_values(
    by="pIC50",
    ascending=False
).reset_index(drop=True)


# ============================================================
# STEP 15: DATASET SUMMARY
# ============================================================

print("==========================================")
print("FINAL hDHFR DATASET - CHEMBL202")
print("==========================================")

print("Number of molecules:", len(df_final))

print("\nBioactivity class distribution:")
print(
    df_final["bioactivity_class"]
    .value_counts()
)

print("\nFirst molecules:")
display(df_final.head(10))


# ============================================================
# STEP 16: FINAL DUPLICATE CHECK
# ============================================================

print("\nDuplicate ChEMBL IDs:",
      df_final["molecule_chembl_id"].duplicated().sum())

print("Duplicate canonical SMILES:",
      df_final["canonical_smiles"].duplicated().sum())


# ============================================================
# STEP 17: SAVE DATASET
# ============================================================

output_file = "hDHFR_CHEMBL202_preprocessed.csv"

df_final.to_csv(
    output_file,
    index=False
)

print("\nFile saved as:")
print(output_file)
