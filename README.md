# hDHFR-QSAR

## Overview

This repository contains the Python code and curated datasets developed for
machine learning-based quantitative structure–activity relationship (QSAR)
modeling of human dihydrofolate reductase (hDHFR) inhibitors.

The workflow integrates molecular fingerprint descriptors, Random Forest
regression, and SHAP-based explainable artificial intelligence (XAI) to
predict and interpret hDHFR inhibitor bioactivity.

## Target

**Human dihydrofolate reductase (hDHFR)**

**ChEMBL target:** CHEMBL202

## Workflow

The computational workflow includes:

1. Data collection from ChEMBL (CHEMBL202)
2. Data curation and preprocessing
3. Bioactivity classification and pIC50 calculation
4. Molecular property calculation
5. Molecular fingerprint generation
6. QSAR model development using Random Forest regression
7. Model evaluation and validation
8. Local SHAP analysis of selected molecules

## Molecular fingerprints

Three fingerprint types are investigated:

- PubChem fingerprints
- Substructure fingerprints
- MACCS fingerprints

Feature selection is performed prior to model development.

## Machine Learning

Random Forest regression models are developed independently for
the three fingerprint representations to predict hDHFR inhibitor
bioactivity expressed as pIC50.

## Explainable AI

SHAP (SHapley Additive exPlanations) is used to investigate the contribution
of molecular fingerprint features to individual molecular predictions.

Local SHAP analysis is applied to selected hDHFR inhibitor molecules to
identify fingerprint features associated with increased or decreased
predicted bioactivity.

## Repository structure

```text
hDHFR-QSAR-AI/
│
├── README.md
│
├── code/
│   ├── 01_data_collection_CHEMBL202.py
│   ├── 02_preprocessing_EDA.py
│   ├── 03_fingerprint_generation.py
│   ├── 04_dataset_integration.py
│   ├── 05_RF_QSAR_models.py
│   └── 06_local_SHAP.py
│
└── data/
    ├── hDHFR_dataset_3classes.csv
    └── hDHFR_dataset_2classes.csv
