"""Check Active:Inactive ratio in each cleaned summary-assay CSV.
Run from the Rebuttal/ directory: python check_pos_neg_ratio.py
"""
import pandas as pd

FILES = {
    'AID_720637 (Summary_1_models)': 'Summary_1_models/cleaning/AID_720637_cleaned.csv',
    'AID_654+AID_1347034 (summary_2_1437034)': 'summary_2_1437034/cleaning/AID_654_1347034_cleaned.csv',
    'AID_1347037 (summary_1347037)': 'summary_1347037/cleaning/AID_1347037_cleaned.csv',
}

for label, path in FILES.items():
    df = pd.read_csv(path, usecols=['PUBCHEM_ACTIVITY_OUTCOME'])
    counts = df['PUBCHEM_ACTIVITY_OUTCOME'].value_counts()
    n_active = counts.get('Active', 0)
    n_inactive = counts.get('Inactive', 0)
    n_inconclusive = counts.get('Inconclusive', 0)
    total = len(df)
    ratio = n_inactive / n_active if n_active else float('nan')

    print(f'\n{label}')
    print(f'  File: {path}')
    print(f'  Total rows: {total:,}')
    print(f'  Active: {n_active:,}  Inactive: {n_inactive:,}  Inconclusive: {n_inconclusive:,}')
    print(f'  Active %: {100*n_active/total:.2f}%   Inactive %: {100*n_inactive/total:.2f}%   Inconclusive %: {100*n_inconclusive/total:.2f}%')
    print(f'  Active:Inactive ratio -> 1:{ratio:.2f}')
