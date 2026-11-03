"""Shared settings for the TX2017-Structure split-protocol experiment.

WHAT THIS FOLDER DOES
---------------------
Runs the TX2017-Structure pipeline (Xu et al. 2017, as replicated in
../DILI_tx2017) under the three rebuttal split protocols, with the training set
balanced by UP-sampling:

    random_repeat   5 repeated random 80/10/10 splits of the clinical pool
    cv             10-fold cross-validation over the clinical pool
    scaffold        5 repeated Murcko-scaffold 8:1:1 splits (Yang et al.)

Structure-only: ECFP4 fingerprints and nothing else -- no physicochemical
descriptors, no pharmacokinetic parameters, no in vitro features.

WHAT IS KEPT FROM TX2017
------------------------
  * ECFP4 at 1024 bits, the bit length Xu et al. state (the graph-model folder
    uses 2048, so the fingerprints are rebuilt here rather than reused)
  * the five classifiers: BernoulliNB, SVM (RBF), RandomForest, MLP, XGBoost,
    with the same constructor arguments as ../DILI_tx2017/train_one_setting.py
  * imblearn's RandomOverSampler as the 'up' sampler, which is the sampler the
    original search selected for both stages
  * the winning feature-selection setting from that search: Random-Forest Gini
    importance, top 180 bits, for Stage 1 and Stage 2 alike
  * the two-stage protocol: Stage 1 ("Proxy-only") trains on the clinical
    training split; Stage 2 ("Proxy+DILI") adds all of NCV-888, unresampled

WHAT IS DIFFERENT, AND WHY
--------------------------
  * The 20 random train/val reshuffles of aggregate_results.py are replaced by
    the three split protocols' units, so TX2017 is evaluated on exactly the
    same splits as every other DILI model in the rebuttal.
  * The feature indices are re-selected inside every unit, from that unit's own
    training rows. The original run selected once on one split and reused the
    indices across its 20 repeats; re-selecting per unit keeps validation and
    the internal test set out of feature selection, which matters more once the
    splits genuinely differ (scaffold especially).
  * Feature selection runs on the RAW training rows, before up-sampling, as in
    ../DILI_tx2017 (feature_selection.py reads the unresampled training set).
    Selecting on duplicated rows would inflate the Fisher/importance counts.

COMPOUND POOL
-------------
The clinical pool, NCV-888 and Held-223 are the same arrays the other DILI
split-protocol runs use (12,145 / 874 / 222 compounds), and the split indices in
splits/ are copied from ../DILI_split_protocols. That is what makes the numbers
comparable row-for-row with the graph and classical tables. It is a slightly
smaller clinical pool than ../DILI_tx2017 built on its own, because the shared
pool keeps only compounds that could be featurized for every model family.

UP-SAMPLING
-----------
Applied to the training rows only. Validation and the internal test split keep
their natural class ratio, and Held-223 is never resampled. Each unit's raw
training rows are recovered as (clinical pool) minus (val) minus (internal
test), which is exact for all three protocols because each one partitions the
whole pool.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

SEED = 42
N_BITS = 1024                      # Xu et al.'s stated ECFP4 bit length
FS_METHOD = 'rf'                   # winning feature-selection method
FS_TOPN = 180                      # winning cutoff (top-N by RF Gini importance)
WINNER_CLF = 'SVM'                 # winning classifier for both stages
PROTOCOLS = {'random_repeat': 5, 'cv': 10, 'scaffold': 5}
STAGES = (1, 2)
CPTS = (0.5, 0.8)
CLASSIFIERS = ['NB', 'SVM', 'RF', 'NNET', 'XGBoost']

DATA = os.path.join(HERE, 'data_tx')
SPLITS = os.path.join(HERE, 'splits')

# Which class-balancing scheme the run uses, set by $TX_SAMPLING:
#   'up'    random over-sampling of the training rows (the TX2017 default and
#           the setting of the published tables)   -> results/
#   'down'  the 1:1 DOWNsampled training rows that ../DILI_split_protocols
#           already wrote to splits/<protocol>/unit_<u>/idx_train.npy, used
#           unchanged so TX2017 trains on exactly the compounds the graph and
#           classical models saw in the downsampled tables   -> results_down/
SAMPLING = os.environ.get('TX_SAMPLING', 'up')
if SAMPLING not in ('up', 'down'):
    raise SystemExit(f"TX_SAMPLING must be 'up' or 'down', got {SAMPLING!r}")
SAMPLING_TAG = {'up': 'upsampled', 'down': 'downsampled'}[SAMPLING]
RESULTS = os.path.join(HERE, 'results' if SAMPLING == 'up' else 'results_down')

XGB_PARAMS = dict(
    objective='binary:logistic', eval_metric='logloss',
    max_depth=3, learning_rate=0.01, colsample_bytree=0.5,
    gamma=1, min_child_weight=1, n_estimators=100,
    n_jobs=-1, random_state=SEED, verbosity=0,
)


def unit_dir(protocol, unit):
    return os.path.join(SPLITS, protocol, f'unit_{unit}')


def read_lines(path):
    return [ln.strip() for ln in open(path) if ln.strip()]


def make_classifier(name, calibrate=True):
    """Same constructors as ../DILI_tx2017/train_one_setting.py.

    calibrate=False makes the SVM rank with decision_function() instead of
    fitting Platt scaling; ROC-AUC is unchanged by that, but the cutoff-based
    metrics need real probabilities, so the reported fits use calibrate=True.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from xgboost import XGBClassifier
    if name == 'NB':
        return BernoulliNB(alpha=1.0)
    if name == 'SVM':
        return SVC(kernel='rbf', probability=calibrate, random_state=SEED)
    if name == 'RF':
        return RandomForestClassifier(random_state=SEED, n_jobs=-1)
    if name == 'NNET':
        return MLPClassifier(random_state=SEED, max_iter=500)
    if name == 'XGBoost':
        return XGBClassifier(**XGB_PARAMS)
    raise ValueError(name)
