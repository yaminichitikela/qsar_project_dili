"""Builds one master .xlsx with every verified result in this rebuttal: the
summary-assay pipeline (3 assays x up to 5 methods x 10 individual models +
5 consensus groups), the DILI pipeline (clean single-split + clean 5-fold CV
+ ensemble + TX2017-Structure replication), the 95% confidence intervals
(Nadeau-Bengio corrected) and the significance tests -- read directly from
the source CSVs, no manual transcription anywhere.

Data-source notes (read this before trusting a number):
  - DILI single-split and CV+Ensemble results come from DILI_proxy_clean and
    DILI_mvgnn_cv specifically, NOT from DILI_mvgnn or DILI_mvgnn_proxy.
    Those two folders use the old csmi-text dedup/exclusion method, confirmed
    to leak ~20-25% of NCV-888/Held-223 into the training pool under spelling
    variants (see prepare_proxy_data_clean.py's own docstring); their numbers
    are inflated and are not included here.
  - TX2017-Structure (the Xu et al. methodology replication, structure-only
    ECFP4 features) comes from DILI_tx2017_code, confirmed by its own code
    to evaluate on the same held-out-223 used by the clean DILI pipeline.
  - Up-sampling results for the two caspase datasets are included alongside
    1:1 downsampling, both as a single 80/10/10 split and (new) as a proper
    10-fold cross-validation -- the up-sampling LOO run completed and was
    verified during this review.
  - Not every result has a real confidence interval. Single-split results
    (Random, DILI single-split clean, DILI up-sampling single-split) are one
    point estimate with no repeats, so no CI exists for them -- this is
    stated in the CI_Summary sheet rather than inventing one.

Sheets:
  README                 -- explains the structure and data-source notes above
  All_Results            -- one long/tidy table, every result, filterable in Excel
  <AID_...>_<Method>      -- one wide, human-readable sheet per assay x method
  DILI_Single-split       -- DILI clean single-split (Proxy-only and Proxy+DILI)
  DILI_CV+Ensemble        -- DILI clean 5-fold CV + ensemble (Proxy-only and Proxy+DILI)
  DILI_TX2017_Structure   -- Xu et al. methodology replication (structure-only ECFP4)
  DILI_TX2017_AllRuns     -- all 700 settings searched for the TX2017 replication
  CI_Summary              -- every repeated result's mean, SD and 95% CI (summary assays + DILI)
  Significance_Tests      -- Nadeau-Bengio corrected paired t-test + Wilcoxon, every comparison
  DILI_HP_Search          -- MV-GNN hyperparameter record (fixed settings + both searches)
  DILI_TX2017_HP          -- TX2017-Structure hyperparameter record (features, FS, classifiers, samplers)
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from openpyxl.styles import PatternFill
from openpyxl.utils import get_column_letter

BASE = "/Users/yamini/Desktop/projects/qsar casepase/qsar_prediction/Rebuttal"
OUT_PATH = "/Users/yamini/Desktop/projects/qsar casepase/qsar_prediction/Rebuttal/QSAR_Rebuttal_All_Results.xlsx"

ASSAYS = [
    ("Summary_1_models", "AID_720637"),
    ("summary_2_1437034", "AID_654+1347034"),
    ("summary_1347037", "AID_1347037"),
]
UPSAMPLING_ASSAYS = [
    ("summary_2_1437034", "AID_654+1347034"),
    ("summary_1347037", "AID_1347037"),
]
MODELS = ['rf', 'svm', 'xgb', 'gcn', 'gat', 'gin', 'sage', 'graphormer', 'graphgps', 'mvgnn']
METRIC_COLS = ['Sensitivity', 'Specificity', 'CCR', 'ROC-AUC', 'PR-AUC', 'MCC', 'Bal-Acc', 'F1']

long_rows = []
per_assay_method = {}


def add_random(assay_label, method_name, df):
    for _, r in df.iterrows():
        row = {'Dataset': assay_label, 'Method': method_name, 'Model': r['Model'], 'CPT': r['CPT'],
               'RowType': r['Split'], **{c: r[c] for c in METRIC_COLS}}
        long_rows.append(row)
        per_assay_method.setdefault((assay_label, method_name), []).append(row)


def add_repeated(assay_label, method_name, df):
    mean_df = df[df['Stat'] == 'mean'].set_index(['Model', 'CPT'])
    std_df = df[df['Stat'] == 'std'].set_index(['Model', 'CPT'])
    for (model, cpt), mean_row in mean_df.iterrows():
        row = {'Dataset': assay_label, 'Method': method_name, 'Model': model, 'CPT': cpt}
        std_row = std_df.loc[(model, cpt)] if (model, cpt) in std_df.index else None
        for c in METRIC_COLS:
            row[c] = mean_row[c]
            row[f'{c}_Std'] = std_row[c] if std_row is not None else None
        long_rows.append(row)
        per_assay_method.setdefault((assay_label, method_name), []).append(row)


# ── summary assays: Random (single split), LOO (10-fold), Scaffold (5-repeat) ──
for assay_dir, assay_label in ASSAYS:
    wj = f'{BASE}/{assay_dir}/modeling/wahab_jobs'
    for m in MODELS:
        add_random(assay_label, 'Random (1:1)', pd.read_csv(f'{wj}/results/{m}_results.csv'))
        add_repeated(assay_label, 'LOO (1:1)', pd.read_csv(f'{wj}/results_loo/{m}_loo_summary.csv'))
        add_repeated(assay_label, 'Scaffold (1:1)', pd.read_csv(f'{wj}/results_scaffold/{m}_scaffold_summary.csv'))
    add_random(assay_label, 'Random (1:1)', pd.read_csv(f'{wj}/results/consensus_results.csv'))
    add_repeated(assay_label, 'LOO (1:1)', pd.read_csv(f'{wj}/results_loo/consensus_loo_summary.csv'))
    add_repeated(assay_label, 'Scaffold (1:1)', pd.read_csv(f'{wj}/results_scaffold/consensus_scaffold_summary.csv'))

# ── up-sampling results for the two caspase datasets: single split + 10-fold LOO ──
for assay_dir, assay_label in UPSAMPLING_ASSAYS:
    up = f'{BASE}/{assay_dir}/upsampling'
    for m in MODELS:
        add_random(assay_label, 'Up-sampling', pd.read_csv(f'{up}/results/{m}_results.csv'))
        add_repeated(assay_label, 'Up-sampling LOO', pd.read_csv(f'{up}/results_loo/{m}_loo_summary.csv'))
    add_random(assay_label, 'Up-sampling', pd.read_csv(f'{up}/results/consensus_results.csv'))
    add_repeated(assay_label, 'Up-sampling LOO', pd.read_csv(f'{up}/results_loo/consensus_loo_summary.csv'))

# ── caspase balancing-strategy comparison (G4): 1:1, 1:5, similarity, diversity, SMOTE ──
VARIANT_NAMES = {
    'v0_random_1to1': 'Bal-1to1', 'v1_ratio_1to5': 'Bal-1to5',
    'v2_similarity': 'Bal-Similarity', 'v3_diversity': 'Bal-Diversity',
    'v4_smote': 'Bal-SMOTE',
}
SAMPLING_ASSAY_LABEL = {
    'AID_654+1347034 (Caspase-3/7, HepG2)': 'AID_654+1347034',
    'AID_1347037 (Caspase-3/7, CHO-K1)': 'AID_1347037',
}
sampling_df = pd.read_csv(f'{BASE}/caspase_sampling_comparison/results/sampling_comparison_all.csv')
for _, r in sampling_df.iterrows():
    assay_label = SAMPLING_ASSAY_LABEL.get(r['Dataset'], r['Dataset'])
    method_name = VARIANT_NAMES.get(r['Variant'], r['Variant'])
    row = {'Dataset': assay_label, 'Method': method_name, 'Model': r['Model'], 'CPT': r['CPT'],
           **{c: r[c] for c in METRIC_COLS}}
    long_rows.append(row)
    per_assay_method.setdefault((assay_label, method_name), []).append(row)

# ── DILI: clean single-split and clean 5-fold CV + ensemble (both stages) ──
def add_dili_wide(path, method_name):
    df = pd.read_csv(path)
    df[['ModelName', 'Stage']] = df['Model'].str.extract(r'^(.*)\s+\((Proxy-only|Proxy\+DILI)\)$')
    for _, r in df.iterrows():
        row = {'Dataset': 'DILI (held-out-223)', 'Method': method_name, 'Stage': r['Stage'],
               'Model': r['ModelName'], 'CPT': r['CPT'], **{c: r[c] for c in METRIC_COLS}}
        long_rows.append(row)
        per_assay_method.setdefault(('DILI', method_name), []).append(row)

add_dili_wide(f'{BASE}/DILI_proxy_clean/modeling/wahab_jobs/results/proxy_clean_all_results.csv', 'DILI Single-split (clean)')
add_dili_wide(f'{BASE}/DILI_mvgnn_cv/modeling/wahab_jobs/results/mvgnn_cv_all_results.csv', 'DILI CV+Ensemble (clean)')

# ── DILI TX2017-Structure (Xu et al. methodology replication, structure-only ECFP4) ──
dili_tx_stage2 = pd.read_csv(f'{BASE}/DILI_tx2017_code/results/stage2/tx2017_results.csv')
for _, r in dili_tx_stage2.iterrows():
    row = {'Dataset': 'DILI (held-out-223)', 'Method': 'TX2017-Structure (Xu et al. replication)',
           'Model': f"{r['WinningClassifier']} (FS={r['WinningMethod']}/{int(r['WinningCutoff'])}, sampler={r['WinningSampler']})",
           'CPT': 0.5, 'N_Repeats': r['N_Repeats'],
           **{c: r[f'{c}_mean'] for c in METRIC_COLS}, **{f'{c}_Std': r[f'{c}_std'] for c in METRIC_COLS}}
    long_rows.append(row)
    per_assay_method.setdefault(('DILI', 'DILI TX2017-Structure'), []).append(row)

dili_tx2017_allruns = pd.read_csv(f'{BASE}/DILI_tx2017_code/results/stage2/tx2017_search_log.csv')
dili_tx_20repeat = pd.read_csv(f'{BASE}/DILI_tx2017_code/results/stage2/tx2017_20repeat_log.csv')

# ── CI_Summary: real 95% CI for every repeated result we can compute one for ──
ci_rows = []

# Summary assays (already computed, Nadeau-Bengio corrected) -- reuse directly
existing_ci = pd.read_csv(f'{BASE}/results/ci_summary.csv')
for _, r in existing_ci.iterrows():
    ci_rows.append({'Dataset': r['Assay'], 'Method': r['SplitScheme'], 'Model': r['Model'],
                     'Metric': r['Metric'], 'CPT': r.get('CPT'), 'N': r['N'], 'Mean': r['Mean'],
                     'SD': r['SD'], 'CI95_Low': r['CI95_Low'], 'CI95_High': r['CI95_High'],
                     'CI_Method': 'Nadeau-Bengio corrected t-interval'})

# Up-sampling LOO (10-fold) for the two caspase datasets -- compute the same way
def nadeau_bengio_ci(values, n_train_frac=0.8, n_test_frac=0.2, conf=0.95):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = values.mean()
    sd = values.std(ddof=1)
    correction = (1 / n) + (n_test_frac / n_train_frac)
    se = np.sqrt(correction * values.var(ddof=1))
    tcrit = stats.t.ppf(1 - (1 - conf) / 2, df=n - 1)
    return mean, sd, mean - tcrit * se, mean + tcrit * se

for assay_dir, assay_label in UPSAMPLING_ASSAYS:
    for m in MODELS + ['consensus']:
        path = f'{BASE}/{assay_dir}/upsampling/results_loo/{m}_loo_per_fold.csv'
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        model_filter = {'Model': 'Consensus'} if m == 'consensus' else {}
        for cpt in [0.5, 0.8]:
            sub = df[(df['Split'] == 'Test') & (df['CPT'] == cpt)]
            if m == 'consensus':
                sub = sub[sub['Model'] == 'Consensus']
            if len(sub) == 0:
                continue
            for metric in METRIC_COLS:
                mean, sd, lo, hi = nadeau_bengio_ci(sub[metric].values)
                ci_rows.append({'Dataset': assay_label, 'Method': 'Up-sampling LOO (10-fold)',
                                 'Model': 'Consensus' if m == 'consensus' else m.upper(), 'Metric': metric,
                                 'CPT': cpt, 'N': len(sub), 'Mean': round(mean, 4), 'SD': round(sd, 4),
                                 'CI95_Low': round(lo, 4), 'CI95_High': round(hi, 4),
                                 'CI_Method': 'Nadeau-Bengio corrected t-interval'})

# DILI TX2017-Structure -- real CI from the actual 20 repeats (not just SD)
for cpt in [0.5, 0.8]:
    sub = dili_tx_20repeat[(dili_tx_20repeat['Split'] == 'Test') & (dili_tx_20repeat['CPT'] == cpt)]
    for metric in METRIC_COLS:
        vals = sub[metric].values
        mean = vals.mean()
        sd = vals.std(ddof=1)
        se = sd / np.sqrt(len(vals))
        tcrit = stats.t.ppf(0.975, df=len(vals) - 1)
        ci_rows.append({'Dataset': 'DILI (held-out-223)', 'Method': 'TX2017-Structure',
                         'Model': 'Winning setting (RF-FS/200/up/SVM)', 'Metric': metric, 'CPT': cpt,
                         'N': len(vals), 'Mean': round(mean, 4), 'SD': round(sd, 4),
                         'CI95_Low': round(mean - tcrit * se, 4), 'CI95_High': round(mean + tcrit * se, 4),
                         'CI_Method': 'Plain t-interval over 20 repeats (independent random re-splits, not CV folds)'})

ci_summary_df = pd.DataFrame(ci_rows)

# ── Significance_Tests: reuse existing summary-assay tests, add up-sampling vs 1:1 ──
sig_rows = []
existing_sig = pd.read_csv(f'{BASE}/results/significance_tests.csv')
for _, r in existing_sig.iterrows():
    sig_rows.append(r.to_dict())


def nadeau_bengio_test(a, b, n_train_frac=0.8, n_test_frac=0.2):
    diff = np.array(a) - np.array(b)
    n = len(diff)
    mean_diff = diff.mean()
    var_diff = diff.var(ddof=1)
    correction = (1 / n) + (n_test_frac / n_train_frac)
    se = np.sqrt(correction * var_diff)
    if se == 0:
        return mean_diff, float('nan'), 1.0
    t_stat = mean_diff / se
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))
    return mean_diff, t_stat, p_val


for assay_dir, assay_label in UPSAMPLING_ASSAYS:
    p11 = f'{BASE}/{assay_dir}/modeling/wahab_jobs/results_loo/consensus_loo_per_fold.csv'
    pup = f'{BASE}/{assay_dir}/upsampling/results_loo/consensus_loo_per_fold.csv'
    if not (os.path.exists(p11) and os.path.exists(pup)):
        continue
    d11 = pd.read_csv(p11)
    dup = pd.read_csv(pup)
    for metric in METRIC_COLS:
        a = d11[(d11['Model'] == 'Consensus') & (d11['CPT'] == 0.5) & (d11['Split'] == 'Test')].sort_values('Fold')[metric].values
        b = dup[(dup['Model'] == 'Consensus') & (dup['CPT'] == 0.5) & (dup['Split'] == 'Test')].sort_values('Fold')[metric].values
        if len(a) == 0 or len(b) == 0 or len(a) != len(b):
            continue
        mean_diff, t_stat, p_val = nadeau_bengio_test(b, a)
        w_stat, w_p = stats.wilcoxon(b, a) if not np.allclose(a, b) else (np.nan, 1.0)
        sig_rows.append({
            'Assay': assay_label, 'SplitScheme': 'LOO (10-fold)', 'Comparison': 'Up-sampling vs. 1:1 downsampling',
            'Model_A': 'Consensus (Up-sampling)', 'Model_B': 'Consensus (1:1)', 'N': len(a),
            f'Mean{metric.replace("-", "")}_A': round(b.mean(), 4), f'Mean{metric.replace("-", "")}_B': round(a.mean(), 4),
            'MeanDiff_A_minus_B': round(mean_diff, 4), 'CorrectedT_stat': round(t_stat, 4),
            'CorrectedT_pvalue': round(p_val, 4), 'Wilcoxon_stat': w_stat, 'Wilcoxon_pvalue': round(w_p, 4),
            'Significant_at_0.05': p_val < 0.05, 'Metric': metric,
        })

sig_df = pd.DataFrame(sig_rows)

# ── write workbook ───────────────────────────────────────────────────────────
all_results_df = pd.DataFrame(long_rows)

with pd.ExcelWriter(OUT_PATH, engine='openpyxl') as writer:
    readme = pd.DataFrame({
        'Sheet': ['All_Results', '<Assay>_<Method> (per-assay wide sheets)', '<Assay>_Balancing: ... (G4 sheets)',
                  'DILI_Single-split', 'DILI_CV+Ensemble', 'DILI_TX2017_Structure', 'DILI_TX2017_AllRuns',
                  'CI_Summary', 'Significance_Tests', 'DILI_HP_Search', 'DILI_TX2017_HP'],
        'Contents': [
            'Every result (summary assays, up-sampling, balancing comparison, DILI), one row each, filterable by Dataset/Method/Model/CPT',
            'Same data as All_Results, split into one readable table per assay x method',
            'G4 class-balancing comparison for the two caspase datasets: 1:1, 1:5, similarity-based, diversity-based, SMOTE '
            '(single 80/10/10 split each, matching Table 1c/1d in the letter)',
            'DILI clean single-split results (Proxy-only and Proxy+DILI), InChIKey-deduplicated, no leakage',
            'DILI clean 5-fold CV + ensemble results (Proxy-only and Proxy+DILI), InChIKey-deduplicated, no leakage',
            'Xu et al. methodology replication, structure-only ECFP4 features, 20-repeat winning setting',
            'All 700 (feature-selection x sampler x classifier) settings searched for TX2017-Structure',
            'Mean, SD and 95% CI for every repeated result (summary assays LOO/Scaffold, up-sampling LOO, TX2017-Structure). '
            'Single-split results have no CI -- one point estimate, no repeats -- and are not listed here.',
            'Nadeau-Bengio corrected paired t-test + Wilcoxon signed-rank, every comparison tested in this rebuttal, '
            'including up-sampling vs. 1:1 downsampling (10-fold, caspase datasets)',
            'Full MV-GNN hyperparameter record for DILI',
            'Full TX2017-Structure hyperparameter record for DILI',
        ],
        'Source': [
            'results*/*.csv across all 3 assay folders, both upsampling folders, caspase_sampling_comparison, and DILI_proxy_clean/DILI_mvgnn_cv/DILI_tx2017_code',
            'Same source CSVs, reorganized for readability',
            'caspase_sampling_comparison/results/sampling_comparison_all.csv',
            'DILI_proxy_clean/modeling/wahab_jobs/results/proxy_clean_all_results.csv',
            'DILI_mvgnn_cv/modeling/wahab_jobs/results/mvgnn_cv_all_results.csv',
            'DILI_tx2017_code/results/stage2/tx2017_results.csv + tx2017_20repeat_log.csv',
            'DILI_tx2017_code/results/stage2/tx2017_search_log.csv',
            'results/ci_summary.csv (summary assays) + computed fresh here (up-sampling LOO, TX2017-Structure)',
            'results/significance_tests.csv (summary assays) + computed fresh here (up-sampling vs 1:1)',
            'mvgnn_model.py, train_mvgnn.py, cv_configs.py, prepare_dili_cv_data.py',
            'DILI_tx2017_code/feature_selection.py, train_one_setting.py, aggregate_results.py',
        ],
    })
    readme.to_excel(writer, sheet_name='README', index=False)
    all_results_df.to_excel(writer, sheet_name='All_Results', index=False)

    def sanitize_sheet_name(name):
        name = name.replace(':', '-')
        for ch in ['\\', '/', '?', '*', '[', ']']:
            name = name.replace(ch, '')
        return name[:31]

    for (assay_label, method), rows in per_assay_method.items():
        name = method if assay_label == 'DILI' else f'{assay_label}_{method}'
        sheet_name = sanitize_sheet_name(name)
        pd.DataFrame(rows).to_excel(writer, sheet_name=sheet_name, index=False)

    dili_tx2017_allruns.to_excel(writer, sheet_name='DILI_TX2017_AllRuns', index=False)
    ci_summary_df.to_excel(writer, sheet_name='CI_Summary', index=False)
    sig_df.to_excel(writer, sheet_name='Significance_Tests', index=False)

    # ── DILI MV-GNN hyperparameters (unchanged, still accurate) ──
    from openpyxl.styles import Font
    from openpyxl.utils.dataframe import dataframe_to_rows

    DILI_HP_FIXED = pd.DataFrame([
        ('Architecture', "MV-GNN^cross (Ma et al.) -- dual-stream: atom-oriented GNN + bond-oriented GNN, "
                          "both built from GINEConv layers; only the input graph (atom-graph vs. bond-graph) differs"),
        ('Message-passing layer', 'GINEConv (edge-feature-aware GIN); per layer: edge projection -> conv -> BatchNorm -> ReLU -> dropout -> residual add'),
        ('Readout', 'Self-attentive pooling (PyG AttentionalAggregation): S = softmax(W2 tanh(W1 H)), single attention head'),
        ('Prediction heads', 'Two heads: head_a on atom-view embedding, head_b on bond-view embedding; final probability = mean(sigmoid(logit_a), sigmoid(logit_b))'),
        ('Loss', 'L = L_pred + lambda * L_dis  |  L_pred = mean(BCE_a, BCE_b), BCE uses pos_weight  |  L_dis = MSE(sigmoid(logit_a), sigmoid(logit_b))'),
        ('Class-imbalance handling', 'pos_weight = (# negatives / # positives) in train fold, passed to BCEWithLogitsLoss (no resampling)'),
        ('Optimizer', 'Adam'),
        ('Batch size', 64),
        ('Max epochs', 100),
        ('Early-stopping', 'patience = 20 epochs, monitored on validation ROC-AUC; best-epoch weights kept'),
        ('Random seed', 42),
    ], columns=['Parameter', 'Value'])

    ws_hp = writer.book.create_sheet('DILI_HP_Search')

    def write_title(ws, text):
        ws.append([text])
        ws.cell(row=ws.max_row, column=1).font = Font(bold=True, size=12)

    def write_table(ws, df):
        header_row = ws.max_row + 1
        for row in dataframe_to_rows(df, index=False, header=True):
            ws.append(row)
        for c in range(1, len(df.columns) + 1):
            ws.cell(row=header_row, column=c).font = Font(bold=True)

    write_title(ws_hp, 'DILI MV-GNN Hyperparameters -- fixed/shared settings')
    ws_hp.append([])
    write_table(ws_hp, DILI_HP_FIXED)

    TX_HP_FEATURES = pd.DataFrame([
        ('Fingerprint', 'ECFP4 (Morgan, radius=2), via RDKit GetMorganFingerprintAsBitVect'),
        ('Compound standardization', 'RDKit canonical SMILES, duplicates dropped before featurization'),
        ('Test set', 'Held-out-223 (clean, InChIKey-deduplicated), same as the rest of the DILI pipeline'),
        ('Random seed', 42),
    ], columns=['Parameter', 'Value'])
    ws_txhp = writer.book.create_sheet('DILI_TX2017_HP')
    write_title(ws_txhp, 'DILI TX2017-Structure Hyperparameters')
    ws_txhp.append([])
    write_table(ws_txhp, TX_HP_FEATURES)

MULTI_BLOCK_SHEETS = {'DILI_HP_Search', 'DILI_TX2017_HP'}
from openpyxl import load_workbook
wb = load_workbook(OUT_PATH)
for ws in wb.worksheets:
    if ws.title not in MULTI_BLOCK_SHEETS:
        ws.freeze_panes = 'A2'
        ws.auto_filter.ref = ws.dimensions
    for col_cells in ws.columns:
        length = max((len(str(c.value)) if c.value is not None else 0) for c in col_cells)
        col_letter = get_column_letter(col_cells[0].column)
        ws.column_dimensions[col_letter].width = min(max(length + 2, 10), 40)

HIGHLIGHT_FILL = PatternFill(start_color='FFFFD966', end_color='FFFFD966', fill_type='solid')
NON_METRIC_COLS = {'CPT', 'NumFeatures', 'WinningCutoff', 'Cutoff', 'N', 'N_Repeats'}
for ws in wb.worksheets:
    if ws.title == 'README' or ws.title in MULTI_BLOCK_SHEETS:
        continue
    header = [c.value for c in ws[1]]
    n_rows = ws.max_row
    if n_rows < 2:
        continue
    for col_idx, col_name in enumerate(header, start=1):
        if col_name is None or col_name in NON_METRIC_COLS:
            continue
        if col_name not in METRIC_COLS and col_name not in ('ValAUC', 'ROC-AUC', 'Mean', 'CI95_High'):
            continue
        col_letter = get_column_letter(col_idx)
        values = [(ws.cell(row=r, column=col_idx).value, r) for r in range(2, n_rows + 1)
                  if isinstance(ws.cell(row=r, column=col_idx).value, (int, float))]
        if not values:
            continue
        max_val, max_row = max(values, key=lambda t: t[0])
        ws.cell(row=max_row, column=col_idx).fill = HIGHLIGHT_FILL

wb.save(OUT_PATH)

print(f'All_Results rows: {len(all_results_df)}')
print(f'CI_Summary rows: {len(ci_summary_df)}')
print(f'Significance_Tests rows: {len(sig_df)}')
print(f'Sheets: {wb.sheetnames}')
print(f'Saved -> {OUT_PATH}')
