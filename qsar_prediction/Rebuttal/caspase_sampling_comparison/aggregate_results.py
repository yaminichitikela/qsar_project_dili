"""Pulls together every model x variant result, for both caspase datasets,
into one comparison table -- once submit_sampling.sh (run from this same
folder) has finished writing results_sampling/<dataset>/<variant>/*_results.csv.

V0 (random 1:1, the current baseline) is read read-only from the original
wahab_jobs_s2/wahab_jobs_s3 reference copies -- already trained, not rerun,
and never written to by anything in this folder.

Usage: python aggregate_results.py
Output: ./results/sampling_comparison_all.csv
"""
import os
import pandas as pd

from common import CASPASE_DATASETS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'results')
os.makedirs(OUT_DIR, exist_ok=True)

# read-only reference to the ORIGINAL, already-trained V0 (random 1:1)
# results -- adjust these two paths if your local V0 baseline results live
# somewhere other than the standard Rebuttal layout.
V0_BASELINE_DIRS = {
    'hepg2': CASPASE_DATASETS['hepg2']['wj'] + '/results',
    'cho_k1': CASPASE_DATASETS['cho_k1']['wj'] + '/results',
}

CLASSICAL_MODELS = ['rf', 'svm', 'xgb']
GRAPH_MODELS = ['gcn', 'gat', 'gin', 'sage', 'graphormer', 'graphgps', 'mvgnn']
ALL_MODELS = CLASSICAL_MODELS + GRAPH_MODELS

VARIANTS = ['v0_random_1to1', 'v1_ratio_1to5', 'v2_similarity', 'v3_diversity', 'v4_smote']

rows = []
for ds_name, cfg in CASPASE_DATASETS.items():
    for variant in VARIANTS:
        models = CLASSICAL_MODELS if variant == 'v4_smote' else ALL_MODELS
        for m in models:
            if variant == 'v0_random_1to1':
                csv_path = os.path.join(V0_BASELINE_DIRS[ds_name], f'{m}_results.csv')
            else:
                csv_path = os.path.join(HERE, 'results_sampling', ds_name, variant, f'{m}_results.csv')
            if not os.path.exists(csv_path):
                print(f'  [missing] {csv_path}')
                continue
            df = pd.read_csv(csv_path)
            df = df[df['Split'] == 'Test'].copy()
            df['Dataset'] = cfg['label']
            df['Variant'] = variant
            rows.append(df)

if not rows:
    print('No results found yet -- run the Wahab jobs first.')
else:
    all_df = pd.concat(rows, ignore_index=True)
    cols = ['Dataset', 'Variant', 'Model', 'CPT', 'Sensitivity', 'Specificity', 'CCR',
            'ROC-AUC', 'PR-AUC', 'MCC', 'Bal-Acc', 'F1']
    all_df = all_df[cols].sort_values(['Dataset', 'Model', 'CPT', 'Variant'])
    out_path = os.path.join(OUT_DIR, 'sampling_comparison_all.csv')
    all_df.to_csv(out_path, index=False)
    print(f'{len(all_df)} rows -> {out_path}')

    print('\n=== ROC-AUC (CPT=0.5) by Dataset x Model x Variant ===')
    pivot = all_df[all_df['CPT'] == 0.5].pivot_table(
        index=['Dataset', 'Model'], columns='Variant', values='ROC-AUC')
    print(pivot.to_string())

    print('\n=== F1 (CPT=0.5) by Dataset x Model x Variant -- the metric Reviewer 1 #13 flagged as low ===')
    pivot_f1 = all_df[all_df['CPT'] == 0.5].pivot_table(
        index=['Dataset', 'Model'], columns='Variant', values='F1')
    print(pivot_f1.to_string())
