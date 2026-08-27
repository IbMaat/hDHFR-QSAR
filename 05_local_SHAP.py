# -*- coding: utf-8 -*-
"""
Part 6 - Local SHAP Analysis for Selected hDHFR Molecules

Target:
Human dihydrofolate reductase (hDHFR)
CHEMBL202

Models:
- PubChem Random Forest
- Substructure Random Forest
- MACCS Random Forest

Purpose:
Local SHAP interpretation of selected molecules.

The user only needs to provide the SMILES of the selected molecules.
"""

# ============================================================
# 1. INSTALL / IMPORT LIBRARIES
# ============================================================

!pip install -q shap padelpy

import os
import glob
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from rdkit import Chem
from padelpy import padeldescriptor


# ============================================================
# 2. PATHS TO THE THREE TRAINED MODELS
# ============================================================

MODEL_PATHS = {

    "PubChem":
        "/content/hDHFR_PubChem_RF.pkl",

    "Substructure":
        "/content/hDHFR_Substructure_RF.pkl",

    "MACCS":
        "/content/hDHFR_MACCS_RF.pkl"
}


# ============================================================
# 3. PUT THE SELECTED MOLECULES HERE
# ============================================================
#
# Replace the SMILES below with your selected molecules.
#
# You can put 1, 2, 5, 10... molecules.
#
# IMPORTANT:
# The name is only an identifier for the molecule.
#
# ============================================================

selected_molecules = {

    "Selected_01":
        "PUT_SMILES_HERE",

    "Selected_02":
        "PUT_SMILES_HERE",

    "Selected_03":
        "PUT_SMILES_HERE"

}


# ============================================================
# 4. CHECK SMILES VALIDITY
# ============================================================

print("=" * 70)
print("CHECKING SELECTED MOLECULES")
print("=" * 70)

valid_molecules = {}

for name, smiles in selected_molecules.items():

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:

        print(f"✗ {name}: INVALID SMILES")

    else:

        valid_molecules[name] = smiles

        print(f"✓ {name}: VALID")


if len(valid_molecules) == 0:

    raise ValueError(
        "No valid molecules were provided."
    )


# ============================================================
# 5. DOWNLOAD / LOCATE PaDEL FINGERPRINT XML FILES
# ============================================================

if not os.path.exists(
    "/content/fingerprints_xml.zip"
):

    !wget -q https://github.com/dataprofessor/padel/raw/main/fingerprints_xml.zip

if not os.path.exists(
    "/content/PubChemFingerprinter.xml"
):

    !unzip -o -q /content/fingerprints_xml.zip -d /content/


# ============================================================
# 6. LOCATE XML FILES
# ============================================================

xml_files = glob.glob(
    "/content/*.xml"
)

xml_files.sort()

print("\nXML files found:")
for f in xml_files:
    print(os.path.basename(f))


# ============================================================
# 7. IDENTIFY THE THREE XML FILES
# ============================================================

def find_xml(pattern):

    matches = [
        f for f in xml_files
        if pattern.lower() in os.path.basename(f).lower()
    ]

    if len(matches) == 0:

        raise FileNotFoundError(
            f"Could not find XML file containing: {pattern}"
        )

    return matches[0]


fingerprint_xml = {

    "PubChem":
        find_xml("PubChem"),

    "Substructure":
        find_xml("Substructure"),

    "MACCS":
        find_xml("MACCS")

}

print("\nSelected XML files:")

for fp, path in fingerprint_xml.items():

    print(
        fp,
        "->",
        os.path.basename(path)
    )


# ============================================================
# 8. CREATE SMILES FILE FOR PaDEL
# ============================================================

smiles_file = "/content/selected_hDHFR_molecules.smi"

with open(
    smiles_file,
    "w"
) as f:

    for name, smiles in valid_molecules.items():

        f.write(
            f"{smiles}\t{name}\n"
        )


print(
    "\nSMILES file created:",
    smiles_file
)


# ============================================================
# 9. CALCULATE FINGERPRINTS WITH PaDEL
# ============================================================

fingerprint_data = {}


for fingerprint_name, xml_file in fingerprint_xml.items():

    print("\n")
    print("=" * 70)
    print(f"CALCULATING {fingerprint_name} FINGERPRINT")
    print("=" * 70)

    output_file = (
        f"/content/selected_{fingerprint_name}.csv"
    )

    padeldescriptor(

        mol_dir=smiles_file,

        d_file=output_file,

        descriptortypes=xml_file,

        detectaromaticity=True,

        standardizenitro=True,

        standardizetautomers=True,

        threads=2,

        removesalt=True,

        fingerprints=True,

        log=True
    )

    fp = pd.read_csv(
        output_file
    )

    fingerprint_data[
        fingerprint_name
    ] = fp

    print(
        "Fingerprint shape:",
        fp.shape
    )


# ============================================================
# 10. LOCAL SHAP ANALYSIS FUNCTION
# ============================================================

def analyze_local_shap(
    molecule_name,
    fingerprint_name,
    fingerprint_df,
    model_path
):

    print("\n")
    print("=" * 70)
    print(
        f"LOCAL SHAP: {molecule_name} - {fingerprint_name}"
    )
    print("=" * 70)


    # --------------------------------------------------------
    # Load model package
    # --------------------------------------------------------

    with open(
        model_path,
        "rb"
    ) as f:

        package = pickle.load(f)


    model = package["model"]

    selected_features = package[
        "selected_features"
    ]


    # --------------------------------------------------------
    # Extract molecule row
    # --------------------------------------------------------

    if "Name" in fingerprint_df.columns:

        row = fingerprint_df[
            fingerprint_df["Name"] == molecule_name
        ]

    else:

        raise ValueError(
            "'Name' column not found in PaDEL output."
        )


    if len(row) == 0:

        raise ValueError(
            f"Molecule {molecule_name} not found."
        )


    # --------------------------------------------------------
    # Prepare fingerprint matrix
    # --------------------------------------------------------

    X = row.drop(
        columns=["Name"],
        errors="ignore"
    )

    # Remove possible index column
    X = X.drop(
        columns=[
            c for c in X.columns
            if c.startswith("Unnamed:")
        ],
        errors="ignore"
    )

    # Ensure numeric
    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.fillna(0)


    # --------------------------------------------------------
    # Make sure all model features exist
    # --------------------------------------------------------

    missing_features = [
        f for f in selected_features
        if f not in X.columns
    ]

    if len(missing_features) > 0:

        print(
            "Missing model features:",
            missing_features
        )

        raise ValueError(
            f"{len(missing_features)} required "
            f"features are missing."
        )


    # Keep EXACTLY the 50 features used during training
    X_selected = X[
        selected_features
    ].copy()


    # --------------------------------------------------------
    # Predict pIC50
    # --------------------------------------------------------

    prediction = model.predict(
        X_selected
    )[0]


    print(
        f"Predicted pIC50: {prediction:.4f}"
    )


    # --------------------------------------------------------
    # SHAP TreeExplainer
    # --------------------------------------------------------

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer.shap_values(
        X_selected
    )


    # --------------------------------------------------------
    # SHAP values for this molecule
    # --------------------------------------------------------

    shap_vector = np.array(
        shap_values
    )[0]


    shap_df = pd.DataFrame({

        "Feature":
            selected_features,

        "Fingerprint":
            X_selected.iloc[0].values,

        "SHAP":
            shap_vector,

        "Absolute_SHAP":
            np.abs(shap_vector)

    })


    # Sort by absolute SHAP
    shap_df = shap_df.sort_values(
        "Absolute_SHAP",
        ascending=False
    ).reset_index(drop=True)


    # --------------------------------------------------------
    # Save local SHAP table
    # --------------------------------------------------------

    output_file = (
        f"/content/"
        f"{molecule_name}_"
        f"{fingerprint_name}_local_SHAP.csv"
    )

    shap_df.to_csv(
        output_file,
        index=False
    )


    # --------------------------------------------------------
    # Display top features
    # --------------------------------------------------------

    print(
        "\nTop SHAP features:"
    )

    display(
        shap_df.head(15)
    )


    # --------------------------------------------------------
    # Local SHAP bar plot
    # --------------------------------------------------------

    top = shap_df.head(15).copy()

    top = top.sort_values(
        "SHAP"
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.barh(
        top["Feature"],
        top["SHAP"]
    )

    plt.axvline(
        0,
        linestyle="--"
    )

    plt.xlabel(
        "SHAP value"
    )

    plt.ylabel(
        "Fingerprint feature"
    )

    plt.title(
        f"{molecule_name} - "
        f"{fingerprint_name} local SHAP"
    )

    plt.tight_layout()

    plt.savefig(
        f"/content/"
        f"{molecule_name}_"
        f"{fingerprint_name}_local_SHAP.pdf",
        bbox_inches="tight"
    )

    plt.show()


    # --------------------------------------------------------
    # SHAP waterfall
    # --------------------------------------------------------

    shap_explanation = shap.Explanation(

        values=shap_vector,

        base_values=explainer.expected_value,

        data=X_selected.iloc[0].values,

        feature_names=selected_features
    )

    shap.plots.waterfall(
        shap_explanation,
        max_display=15
    )

    plt.tight_layout()

    plt.savefig(
        f"/content/"
        f"{molecule_name}_"
        f"{fingerprint_name}_waterfall.pdf",
        bbox_inches="tight"
    )

    plt.show()


    return {

        "Molecule":
            molecule_name,

        "Fingerprint":
            fingerprint_name,

        "Predicted_pIC50":
            prediction,

        "SHAP_table":
            shap_df

    }


# ============================================================
# 11. RUN LOCAL SHAP FOR ALL SELECTED MOLECULES
# ============================================================

all_results = []


for molecule_name in valid_molecules:

    for fingerprint_name in [
        "PubChem",
        "Substructure",
        "MACCS"
    ]:

        result = analyze_local_shap(

            molecule_name=molecule_name,

            fingerprint_name=fingerprint_name,

            fingerprint_df=
                fingerprint_data[fingerprint_name],

            model_path=
                MODEL_PATHS[fingerprint_name]

        )

        all_results.append({

            "Molecule":
                result["Molecule"],

            "Fingerprint":
                result["Fingerprint"],

            "Predicted_pIC50":
                result["Predicted_pIC50"]

        })


# ============================================================
# 12. SUMMARY OF PREDICTIONS
# ============================================================

summary = pd.DataFrame(
    all_results
)

print("\n")
print("=" * 70)
print("FINAL LOCAL SHAP SUMMARY")
print("=" * 70)

display(
    summary
)


# ============================================================
# 13. SAVE SUMMARY
# ============================================================

summary.to_csv(
    "/content/hDHFR_local_SHAP_predictions.csv",
    index=False
)


# ============================================================
# 14. FINISHED
# ============================================================

print("\n")
print("=" * 70)
print("LOCAL SHAP ANALYSIS COMPLETED")
print("=" * 70)

print(
    "\nGenerated files are available in /content/"
)
