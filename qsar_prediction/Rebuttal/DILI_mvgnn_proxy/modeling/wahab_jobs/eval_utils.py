"""Shared evaluation helper -- same metrics/format as every other pipeline
in this project (Rebuttal/*/modeling/wahab_jobs/eval_utils.py)."""
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, matthews_corrcoef,
    balanced_accuracy_score, f1_score, confusion_matrix,
)

METRIC_COLS = ['Model', 'CPT', 'Sensitivity', 'Specificity', 'CCR',
               'ROC-AUC', 'PR-AUC', 'MCC', 'Bal-Acc', 'F1']


def evaluate(y_true, y_prob, name, cpt=0.5):
    y_pred = (y_prob >= cpt).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    return {
        'Model': name,
        'CPT': cpt,
        'Sensitivity': round(sens * 100, 1),
        'Specificity': round(spec * 100, 1),
        'CCR': round((sens + spec) / 2 * 100, 1),
        'ROC-AUC': round(roc_auc_score(y_true, y_prob), 4),
        'PR-AUC': round(average_precision_score(y_true, y_prob), 4),
        'MCC': round(matthews_corrcoef(y_true, y_pred), 4),
        'Bal-Acc': round(balanced_accuracy_score(y_true, y_pred) * 100, 1),
        'F1': round(f1_score(y_true, y_pred), 4),
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
    }
