"""Shared classifier grids for RF/SVM/XGB -- RF/XGB grids match
proxy_to_dili_consistent copy 2.ipynb cell 13 exactly (8 configs each); SVM grid
added for parity (6 configs) using the same probability=False + decision_function()
trick as DILI_tx2017/train_one_setting.py so the HP search doesn't pay for
SVC(probability=True)'s expensive internal Platt-scaling calibration on every
config -- only the final winning combination gets calibrated, in each stage
script, exactly once."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC
from xgboost import XGBClassifier

SEED = 42

RF_GRID = list(ParameterGrid({
    'n_estimators': [300, 500], 'max_features': ['sqrt', 0.3], 'min_samples_leaf': [1, 3],
}))
XGB_GRID = list(ParameterGrid({
    'n_estimators': [300, 500], 'max_depth': [4, 6], 'learning_rate': [0.05, 0.1],
}))
SVM_GRID = list(ParameterGrid({
    'C': [0.1, 1.0, 10.0], 'gamma': ['scale', 'auto'],
}))


def make_rf(params):
    return RandomForestClassifier(**params, class_weight='balanced', n_jobs=-1, random_state=SEED)


def make_xgb(params, y_train):
    sw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    return XGBClassifier(**params, scale_pos_weight=sw, eval_metric='logloss',
                          n_jobs=-1, random_state=SEED, verbosity=0)


def make_svm(params, calibrated=False):
    return SVC(**params, kernel='rbf', probability=calibrated, random_state=SEED)
