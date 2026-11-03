"""Single source of truth for where the published splits and predictions live.

Every path the retrain touches is declared here, so a layout difference between
this laptop and Wahab is fixed in one place rather than scattered through the
training scripts.

THE RULE THIS ENCODES
---------------------
The corrected graph transformers must be trained on the EXACT splits the
published results used. If the splits drifted, the new Graphormer/GraphGPS rows
could not sit in the same table as the existing GCN/GIN/SAGE/classical rows,
because they would describe different test compounds. So nothing here creates a
split -- it only locates the ones that already exist.

The same applies to the consensus: Consensus-GT, Consensus (Full) and
Consensus-MVGNN change only because their graph-transformer members changed, so
the other eight families' saved per-unit probabilities are read as-is.

PROTOCOLS (what the manuscript reports per assay)
-------------------------------------------------
    random_repeat   5 repeated random 80/10/10 splits, 1:1 downsampled train
    loo            10-fold cross-validation, 1:1 downsampled train
    scaffold        5 repeated scaffold splits, 1:1 downsampled train
    up_single       single 80/10/10 split, random up-sampled train
    up_loo         10-fold cross-validation, random up-sampled train

MMP reports the first three; HepG2 and CHO-K1 report all five.

NOTE: random_repeat has no MV-GNN predictions (45 prob files = 9 models x 5
repeats), so Consensus-MVGNN does not exist for that protocol. The model lists
below record that, rather than letting the consensus step fail on a missing file.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REB = os.path.dirname(HERE)

# lowercase basenames, matching the published prob filenames
GT_MODELS = ['graphormer', 'graphgps']
OTHER_MODELS = ['rf', 'svm', 'xgb', 'gcn', 'gat', 'gin', 'sage']
MVGNN = 'mvgnn'

#  TWO LAYOUTS
#  -----------
#  The same runs live under different names in the two places this code runs:
#
#    on Wahab   flat folders in $HOME, named by job batch --
#               wahab_jobs / wahab_jobs_s2 / wahab_jobs_s3 for the downsampled
#               runs, qsar_upsampling_* for the up-sampled ones
#    locally    reorganised into the repository by assay --
#               Summary_1_models/modeling/wahab_jobs, .../upsampling, etc.
#
#  Both are declared; the first root that exists on disk wins. This is why the
#  first Wahab check reported 0/82 while MMP/LOO passed locally: only the
#  repository layout was declared, so every Wahab path was wrong.
LAYOUTS = [
    ('wahab', os.path.expanduser('~'),
     {'mmp': 'wahab_jobs', 'hepg2': 'wahab_jobs_s2', 'chok1': 'wahab_jobs_s3'},
     {'mmp': 'qsar_upsampling_mmp', 'hepg2': 'qsar_upsampling_hepg2',
      'chok1': 'qsar_upsampling_chok1'}),
    ('repo', REB,
     {'mmp': 'Summary_1_models/modeling/wahab_jobs',
      'hepg2': 'summary_2_1437034/modeling/wahab_jobs',
      'chok1': 'summary_1347037/modeling/wahab_jobs'},
     {'mmp': 'Summary_1_models/upsampling',
      'hepg2': 'summary_2_1437034/upsampling',
      'chok1': 'summary_1347037/upsampling'}),
]

# kept for reference / error messages only
ASSAY_ROOT = {
    'mmp': 'Summary_1_models',
    'hepg2': 'summary_2_1437034',
    'chok1': 'summary_1347037',
}


def root(assay, kind):
    """Root folder for one assay's downsampled ('down') or up-sampled ('up') runs.

    Returns the first candidate that exists; if none do, returns the Wahab
    candidate so the error message names the machine where the runs were made.
    """
    cands = []
    for _, base, down, up in LAYOUTS:
        cands.append(os.path.join(base, (down if kind == 'down' else up)[assay]))
    for c in cands:
        if os.path.isdir(c):
            return c
    return cands[0]


def layout_name():
    """Which layout is in use here, for logging."""
    for name, base, down, _ in LAYOUTS:
        if os.path.isdir(os.path.join(base, down['mmp'])):
            return name
    return 'none-found'
ASSAY_LABEL = {
    'mmp': 'MMP (AID 720637, HepG2)',
    'hepg2': 'Caspase-3/7 (AID 654+1347034, HepG2)',
    'chok1': 'Caspase-3/7 (AID 1347037, CHO-K1)',
}

PROTOCOLS = {
    'random_repeat': dict(
        n_units=5, kind='down',
        data='data_random_repeat/repeat_{u}',
        probs='results_random_repeat/probs',
        prob_file='{model}_repeat{u}_test_probs.npy',
        has_mvgnn=False,
        caption='repeated 80/10/10 split (5 repeats), 1:1 downsampled train'),
    'loo': dict(
        n_units=10, kind='down',
        data='data_loo/fold_{u}',
        probs='results_loo/probs',
        prob_file='{model}_fold{u}_test_probs.npy',
        has_mvgnn=True,
        caption='10-fold cross-validation, 1:1 downsampled train'),
    'scaffold': dict(
        n_units=5, kind='down',
        data='data_scaffold/repeat_{u}',
        probs='results_scaffold/probs',
        prob_file='{model}_repeat{u}_test_probs.npy',
        has_mvgnn=True,
        caption='scaffold split (5 repeats), 1:1 downsampled train'),
    'up_single': dict(
        n_units=1, kind='up',
        data='data',
        probs='results',
        prob_file='{model}_test_probs.npy',
        has_mvgnn=True,
        caption='single 80/10/10 split, random up-sampled train'),
    'up_loo': dict(
        n_units=10, kind='up',
        data='data_loo/fold_{u}',
        probs='results_loo/probs',
        prob_file='{model}_fold{u}_test_probs.npy',
        has_mvgnn=True,
        caption='10-fold cross-validation, random up-sampled train'),
}

ASSAY_PROTOCOLS = {
    'mmp': ['random_repeat', 'loo', 'scaffold'],
    'hepg2': ['random_repeat', 'loo', 'scaffold', 'up_single', 'up_loo'],
    'chok1': ['random_repeat', 'loo', 'scaffold', 'up_single', 'up_loo'],
}

RANDOM_REPEAT_ROOT = {k: f'wahab_random_repeat/{k}/probs' for k in ASSAY_ROOT}


def units(assay, protocol):
    return list(range(PROTOCOLS[protocol]['n_units']))


def data_dir(assay, protocol, unit):
    p = PROTOCOLS[protocol]
    return os.path.join(root(assay, p['kind']), p['data'].format(u=unit))


def probs_dir(assay, protocol):
    """Where that protocol's published per-unit predictions live.

    One extra wrinkle for random_repeat: on Wahab those predictions sit in the
    job folder as results_random_repeat/probs, but the local checkout keeps them
    under wahab_random_repeat/<assay>/probs -- the directory they were rsynced
    into. Both are tried.
    """
    p = PROTOCOLS[protocol]
    primary = os.path.join(root(assay, p['kind']), p['probs'])
    if os.path.isdir(primary):
        return primary
    if protocol == 'random_repeat':
        local = os.path.join(REB, RANDOM_REPEAT_ROOT[assay])
        if os.path.isdir(local):
            return local
    return primary


def prob_path(assay, protocol, unit, model):
    p = PROTOCOLS[protocol]
    return os.path.join(probs_dir(assay, protocol),
                        p['prob_file'].format(model=model, u=unit))


def models_for(protocol):
    """The non-GT families whose predictions the consensus reuses."""
    m = list(OTHER_MODELS)
    if PROTOCOLS[protocol]['has_mvgnn']:
        m.append(MVGNN)
    return m


def out_dir(assay, protocol, unit=None):
    """Where the retrain writes. Deliberately a separate tree: the published
    results_loo/, results_scaffold/ etc. are never modified, so old and new can
    be compared and a rollback is simply not using this folder."""
    d = os.path.join(HERE, 'results', assay, protocol)
    if unit is not None:
        d = os.path.join(d, f'unit_{unit}')
    return d


def all_jobs():
    """Every (assay, protocol, unit) the retrain covers."""
    for assay, protos in ASSAY_PROTOCOLS.items():
        for protocol in protos:
            for unit in units(assay, protocol):
                yield assay, protocol, unit


def summary():
    tot = 0
    for assay, protos in ASSAY_PROTOCOLS.items():
        n = sum(PROTOCOLS[p]['n_units'] for p in protos)
        tot += n
        print(f'  {assay:6s} {ASSAY_LABEL[assay]:40s} '
              f'{len(protos)} protocols, {n:3d} split units')
    print(f'  total {tot} split units x 2 architectures = {tot * 2} training runs')
    return tot


if __name__ == '__main__':
    summary()
