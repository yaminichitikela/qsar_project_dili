"""1:5 downsampled Caspase-3/7 CHO-K1 (AID 1347037), 5 repeated scaffold splits.

Every train_*.py / prepare_*.py / compute_consensus.py script takes the protocol
name ("scaffold") as its first argument and reads its data dir, results dir and
unit naming from here.

    scaffold       5 repeated Yang et al. Murcko-scaffold 8:1:1 splits,
                   training set downsampled to 1 active : 5 inactives

Prob files are named <model>_repeat<i>_test_probs.npy, the same naming the
published runs use.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

PROTOCOLS = {
    'scaffold':      dict(data='data_scaffold', results='results_scaffold',
                          unit='repeat', n=5, prep='prepare_scaffold_1to5.py'),
}


def from_argv(usage_rest=''):
    """Pops the protocol name off sys.argv[1] and returns its settings."""
    if len(sys.argv) < 2 or sys.argv[1] not in PROTOCOLS:
        print(f'Usage: python {os.path.basename(sys.argv[0])} '
              f'<{"|".join(PROTOCOLS)}> {usage_rest}'.rstrip())
        sys.exit(1)
    name = sys.argv.pop(1)
    p = dict(PROTOCOLS[name], name=name)
    p['data_dir'] = os.path.join(HERE, p['data'])
    p['out_dir'] = os.path.join(HERE, p['results'])
    p['unit_col'] = p['unit'].capitalize()
    return p


def unit_dir(p, i):
    return os.path.join(p['data_dir'], f'{p["unit"]}_{i}')


def prob_name(p, model, i):
    return f'{model}_{p["unit"]}{i}_test_probs.npy'
