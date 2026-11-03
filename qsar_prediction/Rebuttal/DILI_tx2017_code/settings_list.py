"""Shared list of all valid feature-selection settings for one stage, built
from results/stage<N>/feature_selection.json, indexed 0..len(settings)-1 so
each can run as its own parallel job. Kept in its own module (not inside
train_one_setting.py) so aggregate_results.py can import load_settings
without triggering any training code as an import side effect -- same lesson
learned from the DILI_mvgnn_cv build.
"""
import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


def load_settings(stage):
    stage_dir = os.path.join(RESULTS_DIR, f'stage{stage}')
    with open(os.path.join(stage_dir, 'feature_selection.json')) as f:
        feature_selections = json.load(f)
    settings = []
    for method, cutoffs in feature_selections.items():
        for cutoff, feat_idx in cutoffs.items():
            if len(feat_idx) == 0:
                continue
            settings.append((method, cutoff, feat_idx))
    return settings
