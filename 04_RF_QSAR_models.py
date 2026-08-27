# -*- coding: utf-8 -*-
"""
Part 5 - QSAR Random Forest Models
Human Dihydrofolate Reductase (hDHFR)
Target: CHEMBL202

Fingerprints:
- PubChem
- Substructure
- MACCS

Workflow:
RFE -> 50 features -> Train/Test split -> Random Forest
-> Evaluation -> Save models as .pkl
"""

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)


# ============================================================
# 2. GENERAL SETTINGS
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_FEATURES = 50


# ============================================================
# 3. FUNCTION TO BUILD ONE RANDOM FOREST MODEL
# ============================================================

def build_model(
    input_file,
    fingerprint_name,
    model_file,
    n_estimators,
    max_features,
    max_depth
):

    print("\n")
    print("=" * 70)
    print(f" {fingerprint_name} RANDOM FOREST")
    print("=" * 70)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(input_file)

    print("\nOriginal dataset:", df.shape)


    # --------------------------------------------------------
    # Remove non-informative columns
    # --------------------------------------------------------

    columns_to_remove = [
        "Unnamed: 0",
        "molecule_chembl_id",
        "canonical_smiles"
    ]

    df = df.drop(
        columns=[
            c for c in columns_to_remove
            if c in df.columns
        ],
        errors="ignore"
    )


    # --------------------------------------------------------
    # Check pIC50
    # --------------------------------------------------------

    if "pIC50" not in df.columns:
        raise ValueError(
            f"pIC50 not found in {input_file}"
        )


    # --------------------------------------------------------
    # Define X and y
    # --------------------------------------------------------

    X = df.drop(
        columns=["pIC50"]
    )

    y = df["pIC50"]


    # --------------------------------------------------------
    # Convert fingerprints to numeric
    # --------------------------------------------------------

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.fillna(0)


    print(
        "Fingerprint features before filtering:",
        X.shape[1]
    )


    # --------------------------------------------------------
    # Remove zero-variance features
    # --------------------------------------------------------

    variance_selector = VarianceThreshold(
        threshold=0.0
    )

    X_variance = variance_selector.fit_transform(X)

    variance_features = X.columns[
        variance_selector.get_support()
    ]

    X = pd.DataFrame(
        X_variance,
        columns=variance_features,
        index=df.index
    )

    print(
        "Features after variance filtering:",
        X.shape[1]
    )


    # --------------------------------------------------------
    # RFE - select 50 features
    # --------------------------------------------------------

    print(
        f"\nRunning RFE to select {N_FEATURES} features..."
    )

    rfe_estimator = RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    number_to_select = min(
        N_FEATURES,
        X.shape[1]
    )

    rfe = RFE(
        estimator=rfe_estimator,
        n_features_to_select=number_to_select,
        step=0.1
    )

    rfe.fit(X, y)

    selected_features = X.columns[
        rfe.support_
    ].tolist()

    X_selected = X[
        selected_features
    ].copy()

    print(
        "Selected features:",
        len(selected_features)
    )

    print(selected_features)


    # --------------------------------------------------------
    # Save selected features
    # --------------------------------------------------------

    feature_file = (
        f"/content/{fingerprint_name}_selected_features.csv"
    )

    pd.DataFrame({
        "Feature": selected_features
    }).to_csv(
        feature_file,
        index=False
    )


    # --------------------------------------------------------
    # Train/Test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X_selected,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples :", len(X_test))


    # --------------------------------------------------------
    # Random Forest model
    # --------------------------------------------------------

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features=max_features,
        max_depth=max_depth,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    y_train_pred = model.predict(
        X_train
    )

    y_test_pred = model.predict(
        X_test
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    r2_train = r2_score(
        y_train,
        y_train_pred
    )

    r2_test = r2_score(
        y_test,
        y_test_pred
    )

    rmse_train = np.sqrt(
        mean_squared_error(
            y_train,
            y_train_pred
        )
    )

    rmse_test = np.sqrt(
        mean_squared_error(
            y_test,
            y_test_pred
        )
    )

    mae_train = mean_absolute_error(
        y_train,
        y_train_pred
    )

    mae_test = mean_absolute_error(
        y_test,
        y_test_pred
    )


    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n------------------------------------------")
    print(f"{fingerprint_name} MODEL PERFORMANCE")
    print("------------------------------------------")

    print(
        f"R² Train  : {r2_train:.4f}"
    )

    print(
        f"R² Test   : {r2_test:.4f}"
    )

    print(
        f"RMSE Train: {rmse_train:.4f}"
    )

    print(
        f"RMSE Test : {rmse_test:.4f}"
    )

    print(
        f"MAE Train : {mae_train:.4f}"
    )

    print(
        f"MAE Test  : {mae_test:.4f}"
    )


    # --------------------------------------------------------
    # Experimental vs Predicted
    # --------------------------------------------------------

    plt.figure(
        figsize=(6, 6)
    )

    plt.scatter(
        y_test,
        y_test_pred,
        alpha=0.7,
        edgecolors="black"
    )

    min_value = min(
        y_test.min(),
        y_test_pred.min()
    )

    max_value = max(
        y_test.max(),
        y_test_pred.max()
    )

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--"
    )

    plt.xlabel(
        "Experimental pIC50",
        fontsize=13
    )

    plt.ylabel(
        "Predicted pIC50",
        fontsize=13
    )

    plt.title(
        f"{fingerprint_name} - Random Forest",
        fontsize=14
    )

    plt.tight_layout()

    plt.savefig(
        f"/content/{fingerprint_name}_RF_prediction.pdf",
        bbox_inches="tight"
    )

    plt.show()


    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    feature_importance = pd.DataFrame({

        "Feature": selected_features,

        "Importance": model.feature_importances_

    })

    feature_importance = (
        feature_importance
        .sort_values(
            "Importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    feature_importance.to_csv(
        f"/content/{fingerprint_name}_RF_feature_importance.csv",
        index=False
    )

    print("\nTop 20 features:")
    display(
        feature_importance.head(20)
    )


    # --------------------------------------------------------
    # Save complete model package as .pkl
    # --------------------------------------------------------

    model_package = {

        "model": model,

        "selected_features": selected_features,

        "fingerprint": fingerprint_name,

        "target": "pIC50",

        "target_chembl_id": "CHEMBL202",

        "n_features": len(selected_features),

        "random_state": RANDOM_STATE,

        "test_size": TEST_SIZE,

        "n_estimators": n_estimators,

        "max_features": max_features,

        "max_depth": max_depth

    }


    with open(
        model_file,
        "wb"
    ) as f:

        pickle.dump(
            model_package,
            f
        )


    print(
        "\nModel saved:",
        model_file
    )


    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return {

        "Fingerprint": fingerprint_name,

        "N_features": len(selected_features),

        "N_train": len(X_train),

        "N_test": len(X_test),

        "R2_train": r2_train,

        "R2_test": r2_test,

        "RMSE_train": rmse_train,

        "RMSE_test": rmse_test,

        "MAE_train": mae_train,

        "MAE_test": mae_test

    }


# ============================================================
# 4. PUBCHEM MODEL
# ============================================================

pubchem_results = build_model(

    input_file="/content/hDHFR_PubChem_final.csv",

    fingerprint_name="PubChem",

    model_file="/content/hDHFR_PubChem_RF.pkl",

    n_estimators=500,

    max_features="sqrt",

    max_depth=10

)


# ============================================================
# 5. SUBSTRUCTURE MODEL
# ============================================================

substructure_results = build_model(

    input_file="/content/hDHFR_Substructure_final.csv",

    fingerprint_name="Substructure",

    model_file="/content/hDHFR_Substructure_RF.pkl",

    n_estimators=500,

    max_features="log2",

    max_depth=20

)


# ============================================================
# 6. MACCS MODEL
# ============================================================

maccs_results = build_model(

    input_file="/content/hDHFR_MACCS_final.csv",

    fingerprint_name="MACCS",

    model_file="/content/hDHFR_MACCS_RF.pkl",

    n_estimators=1000,

    max_features="log2",

    max_depth=50

)


# ============================================================
# 7. COMPARE THE THREE MODELS
# ============================================================

results_df = pd.DataFrame([

    pubchem_results,

    substructure_results,

    maccs_results

])

print("\n")
print("=" * 70)
print("FINAL COMPARISON OF THE THREE hDHFR RANDOM FOREST MODELS")
print("=" * 70)

display(results_df)


# ============================================================
# 8. SAVE PERFORMANCE TABLE
# ============================================================

results_df.to_csv(
    "/content/hDHFR_RF_model_performance.csv",
    index=False
)


# ============================================================
# 9. FINAL FILE CHECK
# ============================================================

print("\nGenerated model files:")

for file in [
    "/content/hDHFR_PubChem_RF.pkl",
    "/content/hDHFR_Substructure_RF.pkl",
    "/content/hDHFR_MACCS_RF.pkl"
]:

    if os.path.exists(file):

        print(
            "✓",
            file
        )

    else:

        print(
            "✗ Missing:",
            file
        )

print("\nCompleted.")
