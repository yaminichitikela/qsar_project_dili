"""Combines all per-setting result files for ONE stage, finds the single
best (method, cutoff, sampler, classifier) combination by val ROC-AUC (this
is where SVM, if it wins, gets its full probability=True calibration --
only once, not hundreds of times), then repeats that ONE winning combination
20 times with freshly reshuffled train/val splits (seeds 0-19), each time
evaluated on the SAME fixed held-out-223 -- reporting mean +/- SD, matching
the paper's own model-stability-assessment protocol ("repeated 20 times with
different random seeds") without re-running the full ~700-combination search
20 times (which would be 14,000 fits; this way it's ~700 + 20 = 720 per
stage).

Held-out-223 is untouched by everything except the final evaluation line --
never used to pick the winning combination, never used to pick hyperparameters.

Also pulls out the RF-top20 and XGBoost-top20 settings specifically from the
search log, so that particular cutoff's performance is visible on its own,
not just buried in the full ~700-row log.

Requires all of that stage's train_one_setting.py runs to have finished
first (reads from ./results/stage<N>/settings/setting_<i>.csv).

Usage: python aggregate_results.py <1|2>
"""
import os
import sys
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from settings_list import load_settings
from eval_utils import evaluate, METRIC_COLS

SEED = 42
N_REPEATS = 20
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

XGB_PARAMS = dict(
    objective='binary:logistic', eval_metric='logloss',
    max_depth=3, learning_rate=0.01, colsample_bytree=0.5,
    gamma=1, min_child_weight=1, n_estimators=100,
    n_jobs=-1, random_state=SEED, verbosity=0,
)

CLASSIFIERS = {
    'NB': lambda: BernoulliNB(alpha=1.0),  # alpha=1.0 = Laplace smoothing, matching the paper
    'SVM': lambda: SVC(kernel='rbf', probability=True, random_state=SEED),  # full calibration here only
    'RF': lambda: RandomForestClassifier(random_state=SEED, n_jobs=-1),
    'NNET': lambda: MLPClassifier(random_state=SEED, max_iter=500),
    'XGBoost': lambda: XGBClassifier(**XGB_PARAMS),
}

SAMPLERS = {
    'original': None,
    'down': lambda: RandomUnderSampler(random_state=SEED),
    'up': lambda: RandomOverSampler(random_state=SEED),
    'rose': lambda: SMOTE(random_state=SEED),
}


def fit_and_eval(X_tr, y_tr, X_va, y_va, X_te, y_te, sampler_name, clf_name, stage_label, seed):
    sampler_fn = SAMPLERS[sampler_name]
    if sampler_fn is None:
        Xs, ys = X_tr, y_tr
    else:
        Xs, ys = sampler_fn().fit_resample(X_tr, y_tr)

    clf = CLASSIFIERS[clf_name]()
    clf.fit(Xs, ys)
    prob_val = clf.predict_proba(X_va)[:, 1]
    prob_test = clf.predict_proba(X_te)[:, 1]

    rows = []
    for cpt in [0.5, 0.8]:
        rows.append({**evaluate(y_va, prob_val, f'TX2017-Structure ({stage_label})', cpt),
                     'Split': 'Val', 'Seed': seed})
        rows.append({**evaluate(y_te, prob_test, f'TX2017-Structure ({stage_label})', cpt),
                     'Split': 'Test', 'Seed': seed})
    return rows, prob_test


def main(stage):
    stage_label = 'Proxy-only' if stage == 1 else 'Proxy+DILI'
    stage_dir = os.path.join(RESULTS_DIR, f'stage{stage}')
    settings_dir = os.path.join(stage_dir, 'settings')

    settings = load_settings(stage)
    all_dfs = []
    missing = []
    for i in range(len(settings)):
        path = os.path.join(settings_dir, f'setting_{i}.csv')
        if not os.path.exists(path):
            missing.append(i)
            continue
        all_dfs.append(pd.read_csv(path))
    if missing:
        raise FileNotFoundError(f'Missing {len(missing)} setting results: {missing} -- '
                                 f'run train_one_setting.py {stage} for these indices first.')

    search_df = pd.concat(all_dfs, ignore_index=True).sort_values('ValAUC', ascending=False)
    search_df.to_csv(os.path.join(stage_dir, 'tx2017_search_log.csv'), index=False)
    print(f'Stage {stage} ({stage_label}): combined {len(search_df)} runs from {len(settings)} settings.')
    print('\nTop 10 by val AUC:')
    print(search_df.head(10).to_string(index=False))

    # ── surface the top-20-features settings specifically -- compare Cutoff
    #    numerically, not as a string: it round-trips through CSV as a float
    #    (e.g. 20.0), so a string comparison against '20' would silently
    #    match nothing (same bug class caught earlier in the winner lookup
    #    below). ────────────────────────────────────────────────────────────
    top20 = search_df[(search_df['Method'].isin(['rf', 'xgb']))
                       & (search_df['Cutoff'].astype(float).sub(20).abs() < 1e-9)]
    top20 = top20.sort_values('ValAUC', ascending=False)
    top20.to_csv(os.path.join(stage_dir, 'tx2017_top20_settings.csv'), index=False)
    print(f'\n=== top-20-features settings (RF/XGB, cutoff=20) -- {len(top20)} rows ===')
    print(top20.to_string(index=False))

    winner = search_df.iloc[0]
    method, cutoff, sampler_name, clf_name = (
        winner['Method'], winner['Cutoff'], winner['Sampler'], winner['Classifier'])
    print(f"\n>> WINNER: method={method} cutoff={cutoff} sampler={sampler_name} "
          f"classifier={clf_name} val_auc={winner['ValAUC']:.4f}")

    target_cutoff = float(cutoff)
    feat_idx = None
    for m, c, fi in settings:
        if m == method and abs(float(c) - target_cutoff) < 1e-9:
            feat_idx = fi
            break
    if feat_idx is None:
        raise ValueError(f'Could not find feature indices for method={method} cutoff={cutoff}')

    # ── full clinical pool (clin_train + clin_val stacked back together) --
    #    this is what gets reshuffled fresh each of the 20 repeats. NCV-888
    #    (Stage-2 only) and held-out-223 are NEVER reshuffled or touched by
    #    the search -- only the clinical train/val boundary moves. ─────────
    X_train0 = np.load(os.path.join(DATA_DIR, 'X_train.npy'))[:, feat_idx]
    y_train0 = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    X_val0 = np.load(os.path.join(DATA_DIR, 'X_val.npy'))[:, feat_idx]
    y_val0 = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
    X_pool = np.vstack([X_train0, X_val0])
    y_pool = np.concatenate([y_train0, y_val0])

    if stage == 2:
        X_ncv = np.load(os.path.join(DATA_DIR, 'X_ncv.npy'))[:, feat_idx]
        y_ncv = np.load(os.path.join(DATA_DIR, 'y_ncv.npy'))
    X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))[:, feat_idx]
    y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))

    print(f'\n=== Repeating winning combination {N_REPEATS} times '
          f'(fresh train/val reshuffle each time, held-out-223 always fixed) ===', flush=True)
    repeat_log_path = os.path.join(stage_dir, 'tx2017_20repeat_log.csv')
    repeat_rows = []
    test_probs_all = []
    for seed in range(N_REPEATS):
        idx = np.arange(len(y_pool))
        idx_tr, idx_va = train_test_split(idx, test_size=0.10, stratify=y_pool, random_state=seed)
        X_tr, y_tr = X_pool[idx_tr], y_pool[idx_tr]
        X_va, y_va = X_pool[idx_va], y_pool[idx_va]
        if stage == 2:
            X_tr = np.vstack([X_tr, X_ncv])
            y_tr = np.concatenate([y_tr, y_ncv])

        rows, prob_test = fit_and_eval(X_tr, y_tr, X_va, y_va, X_test, y_test,
                                        sampler_name, clf_name, stage_label, seed)
        repeat_rows.extend(rows)
        test_probs_all.append(prob_test)
        test_row = [r for r in rows if r['Split'] == 'Test' and r['CPT'] == 0.5][0]
        print(f'  seed {seed:2d}: Test ROC-AUC={test_row["ROC-AUC"]:.4f}  MCC={test_row["MCC"]:.4f}', flush=True)

        # checkpoint after every seed -- a killed/timed-out job still leaves
        # partial progress on disk instead of losing all N_REPEATS fits
        pd.DataFrame(repeat_rows)[['Seed', 'Split'] + METRIC_COLS].to_csv(repeat_log_path, index=False)

    repeat_df = pd.DataFrame(repeat_rows)[['Seed', 'Split'] + METRIC_COLS]

    test05 = repeat_df[(repeat_df['Split'] == 'Test') & (repeat_df['CPT'] == 0.5)]
    summary = {
        'Stage': stage_label, 'WinningMethod': method, 'WinningCutoff': cutoff,
        'WinningSampler': sampler_name, 'WinningClassifier': clf_name, 'NumFeatures': len(feat_idx),
        'N_Repeats': N_REPEATS,
    }
    for c in ['Sensitivity', 'Specificity', 'CCR', 'ROC-AUC', 'PR-AUC', 'MCC', 'Bal-Acc', 'F1']:
        summary[f'{c}_mean'] = round(test05[c].mean(), 4)
        summary[f'{c}_std'] = round(test05[c].std(), 4)
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(stage_dir, 'tx2017_results.csv'), index=False)

    # average test-set probability across the 20 repeats, for use in any
    # downstream consensus computation
    np.save(os.path.join(stage_dir, 'tx2017_test_probs.npy'), np.mean(test_probs_all, axis=0))

    print(f'\n== Stage {stage} ({stage_label}) final: {N_REPEATS}-repeat mean +/- SD on held-out-223 ==')
    print(summary_df.T.to_string())
    print(f'\nSaved -> {stage_dir}/tx2017_search_log.csv, tx2017_top20_settings.csv, '
          f'tx2017_20repeat_log.csv, tx2017_results.csv')


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ('1', '2'):
        print(f'Usage: python {sys.argv[0]} <1|2>')
        sys.exit(1)
    main(int(sys.argv[1]))
