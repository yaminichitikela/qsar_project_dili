"""Verifies every input the retrain needs, BEFORE any GPU time is spent.

The retrain reuses published splits and published predictions, so it depends on a
lot of pre-existing state scattered across three assay folders and five
protocols. This script checks all of it and exits non-zero on any gap, so
submit_corrected.sh can gate the GPU jobs on it.

Per (assay, protocol, unit) it checks:
  1. the split directory exists
  2. graphs_train/val/test.pt and y_train/val/test.npy are present
  3. the GraphGPS positional-encoding graphs (graphs_*_gps.pt) are present
  4. each non-GT family has a saved test-probability file
  5. every prob file has the same length as y_test -- a length mismatch means
     the probs and the split describe different compounds, which would silently
     corrupt the recomputed consensus
  6. the published Graphormer/GraphGPS probs exist too, since the comparison
     "old architecture vs corrected architecture" needs both

Usage:
    python check_inputs.py                 # everything
    python check_inputs.py mmp             # one assay
    python check_inputs.py mmp loo         # one assay + protocol
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data_map import (ASSAY_LABEL, ASSAY_PROTOCOLS, GT_MODELS, PROTOCOLS,
                      data_dir, layout_name, models_for, prob_path, probs_dir,
                      root, units)

GRAPH_FILES = ['graphs_train.pt', 'graphs_val.pt', 'graphs_test.pt']
Y_FILES = ['y_train.npy', 'y_val.npy', 'y_test.npy']
GPS_FILES = ['graphs_train_gps.pt', 'graphs_val_gps.pt', 'graphs_test_gps.pt']


def check(assays, protocols):
    problems, ok_units, checked = [], 0, 0
    for assay in assays:
        for protocol in [p for p in ASSAY_PROTOCOLS[assay] if p in protocols]:
            print(f'\n{assay} / {protocol}  ({PROTOCOLS[protocol]["caption"]})')
            pdir = probs_dir(assay, protocol)
            if not os.path.isdir(pdir):
                print(f'  MISSING probs dir: {pdir}')
                problems.append(f'{assay}/{protocol}: probs dir missing')
            for unit in units(assay, protocol):
                checked += 1
                issues = []
                d = data_dir(assay, protocol, unit)
                if not os.path.isdir(d):
                    issues.append(f'split dir missing ({d})')
                else:
                    for f in GRAPH_FILES + Y_FILES:
                        if not os.path.exists(os.path.join(d, f)):
                            issues.append(f'missing {f}')
                    missing_gps = [f for f in GPS_FILES
                                   if not os.path.exists(os.path.join(d, f))]
                    if missing_gps:
                        issues.append(f'missing GraphGPS PE graphs ({len(missing_gps)}/3)')

                n_test = None
                yt = os.path.join(d, 'y_test.npy')
                if os.path.exists(yt):
                    n_test = len(np.load(yt))

                for model in models_for(protocol) + GT_MODELS:
                    p = prob_path(assay, protocol, unit, model)
                    if not os.path.exists(p):
                        issues.append(f'no probs: {model}')
                    elif n_test is not None:
                        n = len(np.load(p))
                        if n != n_test:
                            issues.append(f'{model} probs {n} != y_test {n_test}')

                if issues:
                    print(f'  unit {unit:2d}  FAIL  ' + '; '.join(issues[:4])
                          + (f' (+{len(issues) - 4} more)' if len(issues) > 4 else ''))
                    problems.append(f'{assay}/{protocol}/unit{unit}')
                else:
                    ok_units += 1
                    print(f'  unit {unit:2d}  ok    n_test={n_test}, '
                          f'{len(models_for(protocol))} reusable families + 2 published GTs')
    return problems, ok_units, checked


if __name__ == '__main__':
    args = sys.argv[1:]
    assays = [a for a in args if a in ASSAY_PROTOCOLS] or list(ASSAY_PROTOCOLS)
    protocols = [p for p in args if p in PROTOCOLS] or list(PROTOCOLS)
    print('Checking published splits and predictions the retrain depends on')
    print(f'assays: {assays}\nprotocols: {protocols}')
    lay = layout_name()
    print(f'layout detected: {lay}')
    if lay == 'none-found':
        print('  NOTHING FOUND -- neither the Wahab nor the repository layout is\n'
              '  present here, so every check below will fail for that reason\n'
              '  alone. Fix LAYOUTS in data_map.py before reading further.')
    for a in assays:
        print(f'  {a:6s} down={root(a, "down")}')
        print(f'  {a:6s} up  ={root(a, "up")}')

    problems, ok_units, checked = check(assays, protocols)

    print('\n' + '=' * 72)
    print(f'{ok_units} / {checked} split units ready')
    if problems:
        print(f'{len(problems)} PROBLEM(S) -- do not submit GPU jobs for these:')
        for p in problems:
            print(f'  - {p}')
        print('\nIf these are missing only locally, run this on Wahab: the splits'
              '\nand probability files were generated there.')
        sys.exit(1)
    print('ALL INPUTS PRESENT -- safe to submit')
