# QSAR-Based DILI Prediction Using Graph-based Approaches

This repository contains the code and results for a QSAR study that predicts Drug-Induced Liver Injury (DILI) using classical machine learning and graph-based approaches trained on in vitro proxy assay data.

## Overview

DILI is a leading cause of drug withdrawal and clinical trial failure. This work investigates whether predictive models trained on high-throughput in vitro proxy assays (caspase activation, mitochondrial toxicity, CHO-K1 cell viability) can generalize to clinical DILI outcomes on an independent held-out test set.

Models are evaluated in a two-stage training pipeline: trained on proxy assay data alone (Stage 1), then fine-tuned with additional DILI-labeled compounds (Stage 2). We compare classical models (Random Forest, XGBoost), graph-based approaches (Graphormer, GraphGPS), and ensemble consensus models.

## Repository Structure

```
├── qsar_prediction/        # QSAR prediction models and results
│   ├── liver/              # Main DILI prediction notebook and results
│   ├── caspase/            # Caspase assay models
│   ├── mito/               # Mitochondrial toxicity models
│   ├── qsar_cho-k1/        # CHO-K1 cell viability models
│   ├── combined/           # Multi-assay and combined model analyses
│   ├── dili_invivo/        # In vivo DILI external validation
│   └── cleaned_datasets/   # Preprocessed PubChem AID datasets
│
└── substructure_analysis/  # Substructure and fragment-level analysis
```

## Requirements

```
python >= 3.8
torch
torch-geometric
rdkit
scikit-learn
xgboost
numpy
pandas
matplotlib
```
