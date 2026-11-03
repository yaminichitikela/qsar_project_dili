"""
Trains RF, SVM, XGBoost on each of the 5 repeated random splits, for each of
the three summary assays -- same grid search / fixed-hyperparameter recipe as
the main pipeline's train_rf.py / train_svm.py / train_xgb.py (grid search on
val ROC-AUC, evaluate on test), just looped over 5 repeats per assay.

Usage: python train_classical.py
Outputs: ./results/<assay_key>/{rf,svm,xgb}_repeat_per_repeat.csv,
         ./results/<assay_key>/{rf,svm,xgb}_repeat_summary.csv
"""
import warnings; warnings.filterwarnings('ignore')
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, f1_score, confusion_matrix
from sklearn.model_selection import ParameterGrid

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'data')
OUT_DIR = os.path.join(HERE, 'results')
N_REPEATS = 5
SEED = 42

ASSAYS = ['MMP_720637', 'Caspase_HepG2_654_1347034', 'Caspase_CHOK1_1347037']


def evaluate(y_true, y_prob, name, cpt=0.5):
    y_pred = (y_prob >= cpt).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    return {
        'Model': name, 'CPT': cpt,
        'Sensitivity': round(sens * 100, 1), 'Specificity': round(spec * 100, 1),
        'CCR': round((sens + spec) / 2 * 100, 1),
        'ROC-AUC': round(roc_auc_score(y_true, y_prob), 4),
        'PR-AUC': round(average_precision_score(y_true, y_prob), 4),
        'MCC': round(matthews_corrcoef(y_true, y_pred), 4),
        'F1': round(f1_score(y_true, y_pred), 4),
    }


RF_GRID = ParameterGrid({'n_estimators': [300, 500], 'max_features': ['sqrt', 0.3],
                          'min_samples_leaf': [1, 3], 'max_depth': [None, 20]})
SVM_GRID = ParameterGrid({'C': [1, 10, 50], 'gamma': ['scale', 0.01, 0.001]})


def xgb_grid(y_train):
    sw = (y_train == 0).sum() / (y_train == 1).sum()
    grid = ParameterGrid({'n_estimators': [300, 500], 'max_depth': [4, 6, 8],
                           'learning_rate': [0.05, 0.1], 'subsample': [0.8], 'colsample_bytree': [0.8]})
    return grid, sw


def fit_best(model_fn, grid, X_train, y_train, X_val, y_val):
    best_auc, best_clf = -1, None
    for p in grid:
        clf = model_fn(p)
        clf.fit(X_train, y_train)
        auc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
        if auc > best_auc:
            best_auc, best_clf = auc, clf
    return best_clf, best_auc


def main():
    for assay in ASSAYS:
        print(f'\n=== {assay} ===')
        out_dir = os.path.join(OUT_DIR, assay)
        os.makedirs(out_dir, exist_ok=True)
        per_model_rows = {'Random Forest': [], 'SVM': [], 'XGBoost': []}

        for r in range(N_REPEATS):
            rep_dir = os.path.join(DATA_DIR, assay, f'repeat_{r}')
            X_train = np.load(os.path.join(rep_dir, 'X_train.npy'))
            y_train = np.load(os.path.join(rep_dir, 'y_train.npy'))
            X_val = np.load(os.path.join(rep_dir, 'X_val.npy'))
            y_val = np.load(os.path.join(rep_dir, 'y_val.npy'))
            X_test = np.load(os.path.join(rep_dir, 'X_test.npy'))
            y_test = np.load(os.path.join(rep_dir, 'y_test.npy'))

            rf_clf, _ = fit_best(
                lambda p: RandomForestClassifier(**p, class_weight='balanced', n_jobs=-1, random_state=SEED),
                RF_GRID, X_train, y_train, X_val, y_val)
            prob_test = rf_clf.predict_proba(X_test)[:, 1]
            row = evaluate(y_test, prob_test, 'Random Forest')
            row['Repeat'] = r
            per_model_rows['Random Forest'].append(row)

            svm_clf, _ = fit_best(
                lambda p: SVC(**p, kernel='rbf', class_weight='balanced', probability=True, random_state=SEED),
                SVM_GRID, X_train, y_train, X_val, y_val)
            prob_test = svm_clf.predict_proba(X_test)[:, 1]
            row = evaluate(y_test, prob_test, 'SVM')
            row['Repeat'] = r
            per_model_rows['SVM'].append(row)

            grid, sw = xgb_grid(y_train)
            xgb_clf, _ = fit_best(
                lambda p: XGBClassifier(**p, scale_pos_weight=sw, eval_metric='logloss',
                                         n_jobs=-1, random_state=SEED, verbosity=0),
                grid, X_train, y_train, X_val, y_val)
            prob_test = xgb_clf.predict_proba(X_test)[:, 1]
            row = evaluate(y_test, prob_test, 'XGBoost')
            row['Repeat'] = r
            per_model_rows['XGBoost'].append(row)

            print(f'  repeat {r}: RF={per_model_rows["Random Forest"][-1]["ROC-AUC"]:.4f}  '
                  f'SVM={per_model_rows["SVM"][-1]["ROC-AUC"]:.4f}  '
                  f'XGB={per_model_rows["XGBoost"][-1]["ROC-AUC"]:.4f}')

        for model_key, fname in [('Random Forest', 'rf'), ('SVM', 'svm'), ('XGBoost', 'xgb')]:
            df = pd.DataFrame(per_model_rows[model_key])
            df.to_csv(os.path.join(out_dir, f'{fname}_repeat_per_repeat.csv'), index=False)
            summary = df.drop(columns=['Repeat', 'Model', 'CPT']).agg(['mean', 'std']).reset_index()
            summary.insert(0, 'Model', model_key)
            summary.to_csv(os.path.join(out_dir, f'{fname}_repeat_summary.csv'), index=False)

    print('\nDone.')


if __name__ == '__main__':
    main()
