"""Shared settings for the DILI split-protocol experiment.

WHAT THIS FOLDER DOES
---------------------
Applies the three split protocols used for the in vitro assays in the rebuttal
(MMP, caspase HepG2, caspase CHO-K1) to the DILI data, so DILI gets the same
robustness analysis:

    random_repeat         5 repeated random 80/10/10 splits, 1:1 downsampled train
    random_repeat_nodown  the SAME 5 splits, train left at its natural ratio
                          (control: separates the downsampling effect from the
                          split effect)
    cv                    10-fold CV: fold i = internal test, fold i+1 = val,
                          remaining 8 folds = train, 1:1 downsampled
    scaffold              5 repeated Murcko-scaffold 8:1:1 splits (Yang et al.
                          bin rule), 1:1 downsampled train
    random_repeat_up      the same three split families, but with the training
    cv_up                 set balanced by UP-sampling (DILI-positive compounds
    scaffold_up           duplicated with replacement up to the negative count)
                          instead of downsampling. Validation and internal test
                          keep their natural class ratio, and Held-223 is
                          untouched, so only the training set differs.

The split logic is copied from the assay scripts in
summary_1347037/modeling/wahab_jobs (prepare_random_repeat_splits.py,
prepare_loo_folds.py, prepare_scaffold_splits.py): same seeds (42 + unit), same
fractions, same downsampling rule (inactives sampled down to the active count,
train only; val and test keep their natural ratio).

WHAT IS SPLIT, AND WHAT IS NEVER SPLIT
--------------------------------------
Only the clinical pool (Alloriginal minus Gold-1111, Held-223 and NCV-888) is
split. Held-223 is never split, never used for early stopping or model
selection, and is scored once per trained model -- it is the primary test set,
exactly as in the published DILI tables. The "internal test" carved out of the
clinical pool by each protocol is reported as a secondary result only; its
labels are the weak clinical labels.

Stage 1 ("Proxy-only") trains on the protocol's training split.
Stage 2 ("Proxy+DILI") trains on the same split plus all of NCV-888; graph
families warm-start from that unit's Stage-1 weights, as in the published
pipeline. NCV-888 is never downsampled.

WHAT IS HELD FIXED
------------------
Model code (trainer.py and the five model files) is a byte-identical copy of
DILI_assay_link, which already uses the corrected Graphormer and GraphGPS.
Hyperparameters are the winning configurations from the DILI_assay_link
Structure-only arm (configs/), so the only thing that changes against the
current DILI table is the split protocol.

DATA
----
No new featurization. The featurized clinical pool, NCV-888 and Held-223 arrays
are copied unchanged from DILI_assay_link/data_dili_link (the same arrays the
current DILI table was built from) into ./data, so this folder can be uploaded
to Wahab on its own. Resolution order:
    1. $DILI_LINK_DATA
    2. <this folder>/data                               (bundled copy)
    3. <repo>/Rebuttal/DILI_assay_link/data_dili_link   (laptop original)
The clinical-pool InChIKey order is checked against baseline/ikeys_clin.txt
before any training, so the split indices cannot silently point at the wrong
compounds.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REB = os.path.dirname(HERE)

SEED = 42
PROTOCOLS = {                      # name -> number of units (repeats or folds)
    'random_repeat': 5,
    'random_repeat_nodown': 5,
    'cv': 10,
    'scaffold': 5,
    # Up-sampled versions of the three splits above. Same seeds, so the val and
    # internal-test sets are identical; only the training set differs: every
    # DILI-negative compound is kept and DILI-positive compounds are duplicated
    # with replacement until the classes match, as in the assay up-sampling runs
    # (summary_1347037/upsampling and upsampling_protocols).
    'random_repeat_up': 5,
    'cv_up': 10,
    'scaffold_up': 5,
}
# each up-sampled protocol and the downsampled protocol it shares splits with
UP_PAIRS = {'random_repeat_up': 'random_repeat', 'cv_up': 'cv', 'scaffold_up': 'scaffold'}
CLASSICAL = ['RF', 'SVM', 'XGB']
GNN = ['GCN', 'GAT', 'GIN', 'SAGE']
GT = ['Graphormer', 'GraphGPS']
MVGNN = 'MVGNN'
ALL_FAMILIES = CLASSICAL + GNN + GT + [MVGNN]
STAGES = (1, 2)
CPTS = (0.5, 0.8)
# Stage 1 trains from scratch; stage 2 is a shorter fine-tune -- the same
# schedules as DILI_assay_link/train_dili_link.py
EPOCHS = {1: (80, 15), 2: (60, 12)}

SPLITS = os.path.join(HERE, 'splits')
CONFIGS = os.path.join(HERE, 'configs')
BASELINE = os.path.join(HERE, 'baseline')
RESULTS = os.path.join(HERE, 'results')


def data_dir():
    cands = []
    if os.environ.get('DILI_LINK_DATA'):
        cands.append(os.environ['DILI_LINK_DATA'])
    cands.append(os.path.join(HERE, 'data'))
    cands.append(os.path.join(REB, 'DILI_assay_link', 'data_dili_link'))
    for c in cands:
        if os.path.exists(os.path.join(c, 'X_clin.npy')):
            return c
    raise FileNotFoundError(
        'DILI data not found. Looked in:\n  ' + '\n  '.join(cands) +
        '\nCopy DILI_assay_link/data_dili_link to Wahab or set DILI_LINK_DATA.')


def read_lines(path):
    return [ln.strip() for ln in open(path) if ln.strip()]


def check_clin_order(ddir):
    """The split indices address the clinical pool by position, so the data
    directory must hold the clinical pool in exactly the order the splits were
    built from."""
    have = read_lines(os.path.join(ddir, 'ikeys_clin.txt'))
    want = read_lines(os.path.join(BASELINE, 'ikeys_clin.txt'))
    if have != want:
        raise RuntimeError(f'{ddir}/ikeys_clin.txt does not match '
                           f'baseline/ikeys_clin.txt -- wrong or rebuilt data')


def unit_dir(protocol, unit):
    return os.path.join(SPLITS, protocol, f'unit_{unit}')
