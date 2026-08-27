# -*- coding: utf-8 -*-
"""
Part 2 - Exploratory Data Analysis
Human Dihydrofolate Reductase (hDHFR)
Target: CHEMBL202
"""

# ============================================================
# PART 2: EXPLORATORY DATA ANALYSIS
# ============================================================

# Tasks:
# 1. Import libraries
# 2. Import the preprocessed hDHFR dataset
# 3. Calculate Lipinski descriptors
# 4. Check pIC50 distribution
# 5. Generate 3-class and 2-class datasets
# 6. Perform exploratory analysis
# 7. Compare active and inactive compounds statistically
# 8. Save results and figures


# ============================================================
# STEP 1: IMPORT LIBRARIES
# ============================================================

!pip install -q rdkit seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

from scipy.stats import mannwhitneyu

sns.set(style="ticks")


# ============================================================
# STEP 2: IMPORT THE PREPROCESSED hDHFR DATABASE
# ============================================================

df = pd.read_csv(
    "/content/hDHFR_CHEMBL202_preprocessed.csv"
)

print("Dataset shape:", df.shape)

display(df.head())


# ============================================================
# STEP 3: CHECK DATA
# ============================================================

print("Columns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nBioactivity classes:")
print(df["bioactivity_class"].value_counts())


# ============================================================
# STEP 4: CALCULATE LIPINSKI DESCRIPTORS
# ============================================================

def calculate_lipinski(smiles_list):

    descriptors = []

    for smiles in smiles_list:

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            descriptors.append([
                np.nan,
                np.nan,
                np.nan,
                np.nan
            ])
            continue

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hdonors = Lipinski.NumHDonors(mol)
        hacceptors = Lipinski.NumHAcceptors(mol)

        descriptors.append([
            mw,
            logp,
            hdonors,
            hacceptors
        ])

    return pd.DataFrame(
        descriptors,
        columns=[
            "MW",
            "LogP",
            "NumHDonors",
            "NumHAcceptors"
        ]
    )


df_lipinski = calculate_lipinski(
    df["canonical_smiles"]
)

display(df_lipinski.head())


# ============================================================
# STEP 5: COMBINE BIOACTIVITY + LIPINSKI DESCRIPTORS
# ============================================================

df_combined = pd.concat(
    [
        df.reset_index(drop=True),
        df_lipinski.reset_index(drop=True)
    ],
    axis=1
)

print("Combined dataset shape:", df_combined.shape)

display(df_combined.head())


# ============================================================
# STEP 6: DESCRIPTIVE STATISTICS
# ============================================================

print("=== IC50 descriptive statistics ===")

display(
    df_combined["standard_value"].describe()
)

print("=== pIC50 descriptive statistics ===")

display(
    df_combined["pIC50"].describe()
)

print("=== Lipinski descriptors ===")

display(
    df_combined[
        [
            "MW",
            "LogP",
            "NumHDonors",
            "NumHAcceptors"
        ]
    ].describe()
)


# ============================================================
# STEP 7: SAVE THREE-CLASS DATASET
# ============================================================

df_3classes = df_combined.copy()

df_3classes.to_csv(
    "/content/hDHFR_dataset_3classes.csv",
    index=False
)

print(
    "3-class dataset saved:"
    "/content/hDHFR_dataset_3classes.csv"
)


# ============================================================
# STEP 8: CREATE TWO-CLASS DATASET
# ============================================================

# Remove intermediate compounds
df_2classes = df_combined[
    df_combined["bioactivity_class"].isin(
        ["Active", "Inactive"]
    )
].copy()

df_2classes = df_2classes.reset_index(drop=True)

print("Two-class dataset shape:", df_2classes.shape)

print("\nTwo-class distribution:")
print(
    df_2classes["bioactivity_class"].value_counts()
)

display(df_2classes.head())


# ============================================================
# STEP 9: SAVE TWO-CLASS DATASET
# ============================================================

df_2classes.to_csv(
    "/content/hDHFR_dataset_2classes.csv",
    index=False
)


# ============================================================
# STEP 10: BIOACTIVITY CLASS DISTRIBUTION
# ============================================================

plt.figure(figsize=(5.5, 5.5))

sns.countplot(
    x="bioactivity_class",
    data=df_2classes,
    edgecolor="black"
)

plt.xlabel(
    "Bioactivity class",
    fontsize=14,
    fontweight="bold"
)

plt.ylabel(
    "Frequency",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    "/content/hDHFR_bioactivity_class.pdf",
    bbox_inches="tight"
)

plt.show()


# ============================================================
# STEP 11: MW vs LogP
# ============================================================

plt.figure(figsize=(6, 5.5))

sns.scatterplot(
    x="MW",
    y="LogP",
    data=df_2classes,
    hue="bioactivity_class",
    size="pIC50",
    edgecolor="black",
    alpha=0.7
)

plt.xlabel(
    "Molecular Weight (g/mol)",
    fontsize=14,
    fontweight="bold"
)

plt.ylabel(
    "LogP",
    fontsize=14,
    fontweight="bold"
)

plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc=2,
    borderaxespad=0
)

plt.tight_layout()

plt.savefig(
    "/content/hDHFR_MW_vs_LogP.pdf",
    bbox_inches="tight"
)

plt.show()


# ============================================================
# STEP 12: pIC50 DISTRIBUTION
# ============================================================

plt.figure(figsize=(5.5, 5.5))

sns.boxplot(
    x="bioactivity_class",
    y="pIC50",
    data=df_2classes
)

plt.xlabel(
    "Bioactivity class",
    fontsize=14,
    fontweight="bold"
)

plt.ylabel(
    "pIC50",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    "/content/hDHFR_pIC50_boxplot.pdf",
    bbox_inches="tight"
)

plt.show()


# ============================================================
# STEP 13: MANN-WHITNEY U TEST
# ============================================================

def mannwhitney_test(
    descriptor,
    data=df_2classes
):

    active = data[
        data["bioactivity_class"] == "Active"
    ][descriptor].dropna()

    inactive = data[
        data["bioactivity_class"] == "Inactive"
    ][descriptor].dropna()

    statistic, p_value = mannwhitneyu(
        active,
        inactive,
        alternative="two-sided"
    )

    alpha = 0.05

    if p_value < alpha:
        interpretation = (
            "Statistically significant difference"
        )
    else:
        interpretation = (
            "No statistically significant difference"
        )

    result = pd.DataFrame({
        "Descriptor": [descriptor],
        "Mann_Whitney_U": [statistic],
        "p_value": [p_value],
        "alpha": [alpha],
        "Interpretation": [interpretation]
    })

    filename = (
        "/content/mannwhitney_"
        + descriptor
        + ".csv"
    )

    result.to_csv(
        filename,
        index=False
    )

    return result


# ============================================================
# STEP 14: pIC50 COMPARISON
# ============================================================

result_pIC50 = mannwhitney_test(
    "pIC50"
)

display(result_pIC50)


# ============================================================
# STEP 15: MOLECULAR WEIGHT
# ============================================================

plt.figure(figsize=(5.5, 5.5))

sns.boxplot(
    x="bioactivity_class",
    y="MW",
    data=df_2classes
)

plt.xlabel(
    "Bioactivity class",
    fontsize=14,
    fontweight="bold"
)

plt.ylabel(
    "Molecular Weight (g/mol)",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    "/content/hDHFR_MW.pdf",
    bbox_inches="tight"
)

plt.show()

result_MW = mannwhitney_test("MW")

display(result_MW)


# ============================================================
# STEP 16: LogP
# ============================================================

plt.figure(figsize=(5.5, 5.5))

sns.boxplot(
    x="bioactivity_class",
    y="LogP",
    data=df_2classes
)

plt.xlabel(
    "Bioactivity class",
    fontsize=14,
    fontweight="bold"
)

plt.ylabel(
    "LogP",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    "/content/hDHFR_LogP.pdf",
    bbox_inches="tight"
)

plt.show()

result_LogP = mannwhitney_test("LogP")

display(result_LogP)


# ============================================================
# STEP 17: H-BOND DONORS
# ============================================================

plt.figure(figsize=(5.5, 5.5))

sns.boxplot(
    x="bioactivity_class",
    y="NumHDonors",
    data=df_2classes
)

plt.xlabel(
    "Bioactivity class",
    fontsize=14,
    fontweight="bold"
)

plt.ylabel(
    "H-bond donors",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    "/content/hDHFR_NumHDonors.pdf",
    bbox_inches="tight"
)

plt.show()

result_HDonors = mannwhitney_test(
    "NumHDonors"
)

display(result_HDonors)


# ============================================================
# STEP 18: H-BOND ACCEPTORS
# ============================================================

plt.figure(figsize=(5.5, 5.5))

sns.boxplot(
    x="bioactivity_class",
    y="NumHAcceptors",
    data=df_2classes
)

plt.xlabel(
    "Bioactivity class",
    fontsize=14,
    fontweight="bold"
)

plt.ylabel(
    "H-bond acceptors",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    "/content/hDHFR_NumHAcceptors.pdf",
    bbox_inches="tight"
)

plt.show()

result_HAcceptors = mannwhitney_test(
    "NumHAcceptors"
)

display(result_HAcceptors)


# ============================================================
# STEP 19: COMBINE MANN-WHITNEY RESULTS
# ============================================================

mannwhitney_results = pd.concat(
    [
        result_pIC50,
        result_MW,
        result_LogP,
        result_HDonors,
        result_HAcceptors
    ],
    ignore_index=True
)

display(mannwhitney_results)

mannwhitney_results.to_csv(
    "/content/hDHFR_mannwhitney_results.csv",
    index=False
)


# ============================================================
# STEP 20: FINAL SUMMARY
# ============================================================

print("============================================")
print("hDHFR EXPLORATORY DATA ANALYSIS")
print("============================================")

print(
    "Target: Human Dihydrofolate Reductase"
)

print(
    "ChEMBL target ID: CHEMBL202"
)

print(
    "\nThree-class dataset:",
    len(df_3classes)
)

print(
    "Two-class dataset:",
    len(df_2classes)
)

print("\nThree-class distribution:")
print(
    df_3classes["bioactivity_class"]
    .value_counts()
)

print("\nTwo-class distribution:")
print(
    df_2classes["bioactivity_class"]
    .value_counts()
)

print("\nMann-Whitney U results:")
display(mannwhitney_results)


# ============================================================
# STEP 21: SAVE FINAL DATASETS
# ============================================================

df_3classes.to_csv(
    "/content/hDHFR_dataset_3classes.csv",
    index=False
)

df_2classes.to_csv(
    "/content/hDHFR_dataset_2classes.csv",
    index=False
)

print("\nFiles successfully saved.")
