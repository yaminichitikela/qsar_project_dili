"""Rebuilds tables_dili_6parts_short.tex -- the six SI DILI tables, short captions.

Takes the bodies straight out of the tables DILI_split_protocols generates, so
the numbers can never drift from the generator. Differences from those tables:
short captions, no footnote block, and the $^\\dagger$/$^\\ddagger$ markers
dropped from the model names (there is no footnote left for them to point at).

Every metric now carries its 95% interval, so the captions name the interval
type without listing columns and note that the percentage columns' intervals
are in percentage points.

Run after make_assay_style_tables.py.
Usage: python build_dili_6parts_short.py
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'DILI_split_protocols', 'results')
DOWN = os.path.join(SRC, 'tables_dili_split_protocols_assay_style_cpt50.tex')
UP = os.path.join(SRC, 'tables_dili_split_protocols_assay_style_upsampled_cpt50.tex')

SD = 'Mean $\\pm$ SD over 5 repeats'
NB5 = 'Mean over 5 repeats with Nadeau--Bengio corrected 95\\% intervals'
NB10 = ('Ten models, each trained on eight folds and validated on one; mean of their '
        'Held-223 scores with Nadeau--Bengio corrected 95\\% intervals')
TAIL = ('Intervals are given on every metric; on the percentage columns they are in '
        'percentage points. Percentages are given to two decimals and all other values '
        'to three. Cutoff 0.5; best value per column within each stage in bold, '
        'second best underlined.')

# (source file, source label, part number, split description, interval sentence, new label)
PARTS = [
    (DOWN, 'tab:dili_random_assay_style', 1,
     'repeated 80/10/10 split, 1:1 downsampled train', SD, 'tab:dili_random_assay_style'),
    (UP, 'tab:dili_random_up_assay_style', 2,
     'repeated 80/10/10 split, 1:1 up-sampled train', SD, 'tab:dili_random_up_assay_style'),
    (DOWN, 'tab:dili_cv_assay_style', 3,
     '10-fold cross-validation, 1:1 downsampled train', NB10, 'tab:dili_cv_assay_style'),
    (UP, 'tab:dili_cv_up_assay_style', 4,
     '10-fold cross-validation, 1:1 up-sampled train', NB10, 'tab:dili_cv_up_assay_style'),
    (DOWN, 'tab:dili_scaffold_assay_style', 5,
     'scaffold split (8:1:1), 1:1 downsampled train', NB5, 'tab:dili_scaffold_assay_style'),
    (UP, 'tab:dili_scaffold_up_assay_style', 6,
     'scaffold split (8:1:1), 1:1 up-sampled train', NB5, 'tab:dili_scaffold_up_assay_style'),
]


def body(path, label):
    """The \\begin{tabular}...\\end{tabular} block of the table carrying `label`."""
    for block in re.split(r'\\begin\{table\*\}', open(path).read())[1:]:
        if f'\\label{{{label}}}' not in block:
            continue
        m = re.search(r'(\\begin\{tabular\}.*?\\end\{tabular\})', block, re.S)
        tab = m.group(1)
        # the footnote block is dropped, so the markers that referenced it go too
        return tab.replace('$^{\\dagger}$', '').replace('$^{\\ddagger}$', '')
    raise SystemExit(f'{label} not found in {path}')


def main():
    out = ['% DILI Table S7, Parts 1-6: downsampled and up-sampled training under three '
           'split protocols.',
           '% Built by build_dili_6parts_short.py from the tables that',
           '% DILI_split_protocols/make_assay_style_tables.py writes (cutoff 0.5).',
           '% 95% intervals on all eight metrics. Do not hand-edit: re-run the builder.',
           '']
    for path, label, part, desc, interval, newlabel in PARTS:
        out += ['\\begin{table}[htbp]', '\\centering',
                f'\\caption{{\\textit{{(Part {part} of 6)}} Performance on the DILI held-out '
                f'test set (Held-223) -- {desc}. {interval}. {TAIL}}}',
                f'\\label{{{newlabel}}}',
                '\\resizebox{\\textwidth}{!}{%',
                body(path, label) + '%', '}', '\\end{table}', '']
    path = os.path.join(HERE, 'tables_dili_6parts_short.tex')
    with open(path, 'w') as fh:
        fh.write('\n'.join(out))
    print(f'-> {path}')


if __name__ == '__main__':
    main()
