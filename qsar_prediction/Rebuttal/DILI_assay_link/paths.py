"""Input-file resolution, so this folder can be rsynced to Wahab on its own.

The first Wahab run failed here: the scripts resolved their inputs relative to
the repository (../Summary_1_models/cleaning/...), but only DILI_assay_link had
been copied over, so prep looked for /home/ychit001/Summary_1_models/... and died
with FileNotFoundError.

Resolution order for each logical input name:
  1. $QSAR_LINK_INPUTS/<name>        -- explicit override
  2. <this folder>/inputs/<name>     -- the bundled copy (what Wahab uses)
  3. <repo>/<original path>          -- the in-repo original (what a local run uses)

inputs/ is populated by bundle_inputs.py and is ~23 MB, so the whole folder stays
small enough to copy in one go. Keeping the bundled copies byte-identical to the
originals matters: these are the same cleaned datasets every published result in
the paper was built from, and silently substituting a re-derived file would make
the in vitro features incomparable to those results.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REB = os.path.dirname(HERE)
INPUTS = os.path.join(HERE, 'inputs')

# logical name -> path inside the repository, relative to Rebuttal/
ORIGINALS = {
    'AID_720637_cleaned.csv': 'Summary_1_models/cleaning/AID_720637_cleaned.csv',
    'AID_654_1347034_cleaned.csv':
        'summary_2_1437034/modeling/wahab_jobs/cleaned_data/AID_654_1347034_cleaned.csv',
    'Alloriginal_deduped.csv': 'results/Alloriginal_deduped.csv',
    'Gold-1111_deduped.csv': 'results/Gold-1111_deduped.csv',
    'NCV-888_deduped.csv': 'results/NCV-888_deduped.csv',
    'Held-223_deduped.csv': 'results/Held-223_deduped.csv',
}


def resolve(name):
    """Returns an existing path for a logical input name, or raises."""
    if name not in ORIGINALS:
        raise KeyError(f'unknown input: {name}')
    env = os.environ.get('QSAR_LINK_INPUTS')
    candidates = []
    if env:
        candidates.append(os.path.join(env, name))
    candidates.append(os.path.join(INPUTS, name))
    candidates.append(os.path.join(REB, ORIGINALS[name]))
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f'input "{name}" not found. Looked in:\n  ' + '\n  '.join(candidates) +
        '\nRun bundle_inputs.py from a full checkout, or set QSAR_LINK_INPUTS.')


def describe():
    """Prints where each input resolved to -- cheap provenance for the logs."""
    for name in ORIGINALS:
        try:
            p = resolve(name)
            where = 'bundled' if p.startswith(INPUTS) else 'repo'
            print(f'  {name:32s} [{where}] {p}')
        except FileNotFoundError:
            print(f'  {name:32s} [MISSING]')
