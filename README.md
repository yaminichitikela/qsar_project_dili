# QSAR-Based DILI Prediction Using Graph Neural Networks

This repository contains the code and results for a QSAR (Quantitative Structure-Activity Relationship) study that predicts **Drug-Induced Liver Injury (DILI)** using classical machine learning and graph neural network (GNN) models trained on in vitro proxy assay data.

---

## Overview

DILI is a leading cause of drug withdrawal and clinical trial failure. This work investigates whether predictive models trained on high-throughput in vitro proxy assays (e.g., caspase activation, mitochondrial toxicity, cell viability) can generalize to clinical DILI outcomes on an independent held-out test set of 222 compounds.

We evaluate two graph transformer architectures — **Graphormer** and **GraphGPS** — alongside classical models (Random Forest, XGBoost) and ensemble consensus models in a two-stage training pipeline:

- **Stage 1 (PROXY only):** trained on ~12,900 proxy assay compounds
- **Stage 2 (PROXY + DILI):** fine-tuned with an additional 874 DILI-labeled compounds (NCV-888 set)

---

## Repository Structure

```
├── liver/                          # DILI prediction (main analysis)
│   ├── proxy_to_dili_consistent copy 4.ipynb   # Main notebook: RF, XGB, Graphormer, GraphGPS
│   ├── table6_results.csv          # Final 14-model results table
│   ├── roc_pr_proxy_dili_updated.png           # ROC & PR curves (Proxy+DILI)
│   └── ...
│
├── caspase/                        # Caspase assay QSAR models
│   └── QSAR_Caspase_GraphTransformers.ipynb
│
├── mito/                           # Mitochondrial toxicity QSAR models
│   └── QSAR_Mito_GraphTransformers_v2.ipynb
│
├── qsar_cho-k1/                    # CHO-K1 cell viability QSAR models
│   └── QSAR_CHOK1_GraphTransformers.ipynb
│
├── combined/                       # Multi-assay and combined model analyses
├── dili_invivo/                    # In vivo DILI external validation
└── cleaned_datasets/               # Preprocessed PubChem AID datasets
```

---

## Models

| Model | Type | Description |
|-------|------|-------------|
| RF | Classical | Random Forest with Morgan fingerprints |
| XGB | Classical | XGBoost with Morgan fingerprints |
| Graphormer | Graph Transformer | TransformerConv with degree embeddings |
| GraphGPS | Graph Transformer | GINEConv + multi-head attention + Laplacian PE |
| Consensus-Classic | Ensemble | Mean of RF + XGB probabilities |
| Consensus-Graph | Ensemble | Mean of Graphormer + GraphGPS probabilities |
| Consensus-Full | Ensemble | Mean of all four model probabilities |

---



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

---

## Data Sources

- **DILI Gold Standard (1,111):** DILIst curated benchmark
- **Proxy Assay Training (~12,900):** PubChem AID datasets (caspase, mitochondrial, CHO-K1 cell viability)
- **NCV-888:** Internal DILI-labeled compounds used for Stage-2 fine-tuning
- **Held-out Test (222):** Independent DILIst compounds, DILI+ = 155, DILI− = 67
