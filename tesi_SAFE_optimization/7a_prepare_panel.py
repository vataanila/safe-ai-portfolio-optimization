"""
Builds the train/test panels the rest of Step 7 runs on: merges the ML
panel with each model's mu predictions and sector metadata, then splits
on 2023-01-01 into train_panel.csv (in-sample) and test_panel.csv (OOS).

Prediction files are wide (dates x tickers) with first-trading-day dates,
so alignment to the panel uses year-month period matching rather than
exact date matching, to handle business-day offsets.

Reads data/results/step4/ml_panel.csv, the three ml_mu_*.csv prediction
files under data/results/step5/, and meta_clean.csv. Run this before
7b_rga.py.

Author: Anila Vata
"""

import os

import pandas as pd

os.makedirs('data/results/step7', exist_ok=True)

FEATURE_COLS = [
    'ret_1w', 'ret_1m', 'ret_3m', 'ret_6m', 'ret_12m',
    'vol_1m', 'vol_3m', 'vol_ratio', 'amihud', 'log_mktcap',
]

SPLIT_DATE = pd.Timestamp('2023-01-01')

print("\n[1] Loading ml_panel.csv ...")

_PANEL_PATH = 'data/results/step4/ml_panel.csv'

# Read header first to detect optional target_end_date column
_header_cols = pd.read_csv(_PANEL_PATH, nrows=0).columns.tolist()
_HAS_TARGET_END_DATE = 'target_end_date' in _header_cols

_base_cols    = ['date', 'ticker', 'target_raw', 'target_rank'] + FEATURE_COLS
_use_cols     = _base_cols + (['target_end_date'] if _HAS_TARGET_END_DATE else [])
_parse_dates  = ['date'] + (['target_end_date'] if _HAS_TARGET_END_DATE else [])

panel = pd.read_csv(
    _PANEL_PATH,
    parse_dates=_parse_dates,
    usecols=_use_cols,
)
print(f"  Loaded : {panel.shape}")
print(f"  target_end_date present: {_HAS_TARGET_END_DATE}")
print(f"  Dates  : {panel['date'].min().date()} to {panel['date'].max().date()}")
print(f"  Tickers: {panel['ticker'].nunique()}")

panel = panel.dropna()
print(f"  After dropna: {panel.shape}")

print(f"\n[2] Splitting train / test on {SPLIT_DATE.date()} ...")

train = panel[panel['date'] <  SPLIT_DATE].copy()
test  = panel[panel['date'] >= SPLIT_DATE].copy()

print(f"  Train: {train.shape}  |  {train['date'].min().date()} – {train['date'].max().date()}  |  {train['ticker'].nunique()} tickers")
print(f"  Test : {test.shape}   |  {test['date'].min().date()} – {test['date'].max().date()}   |  {test['ticker'].nunique()} tickers")

print("\n[3] Loading and melting prediction files ...")

def load_and_melt(model_name: str) -> pd.DataFrame:
    """Read a wide prediction CSV and return a long DataFrame with columns
    [period, ticker, mu_{model_name}].  Period is Year-Month so that
    first-trading-day dates align with first-of-month panel dates."""
    path = f'data/results/step5/{model_name}/predictions/ml_mu_{model_name}.csv'
    wide = pd.read_csv(path, index_col=0, parse_dates=True)
    wide.index.name = 'date'
    long = (
        wide.reset_index()
        .melt(id_vars='date', var_name='ticker', value_name=f'mu_{model_name}')
    )
    long['period'] = long['date'].dt.to_period('M')

    dups = long[long.duplicated(subset=['period', 'ticker'], keep=False)]
    if not dups.empty:
        dup_pairs = (
            dups[['period', 'ticker']]
            .drop_duplicates()
            .to_dict('records')
        )
        raise ValueError(
            f"Duplicate period-ticker predictions in {model_name}: {dup_pairs}"
        )

    print(f"  {model_name:8s}: {long.shape}  |  {long['date'].min().date()} – {long['date'].max().date()}")
    return long[['period', 'ticker', f'mu_{model_name}']]

pred_ridge   = load_and_melt('ridge')
pred_xgboost = load_and_melt('xgboost')
pred_mlp     = load_and_melt('mlp')

print("\n[4] Merging predictions onto test panel ...")

test['period'] = test['date'].dt.to_period('M')

test = test.merge(pred_ridge,   on=['period', 'ticker'], how='left')
test = test.merge(pred_xgboost, on=['period', 'ticker'], how='left')
test = test.merge(pred_mlp,     on=['period', 'ticker'], how='left')

test = test.drop(columns='period')

print(f"  After merge : {test.shape}")
print(f"  NaN mu_ridge   : {test['mu_ridge'].isna().sum()}")
print(f"  NaN mu_xgboost : {test['mu_xgboost'].isna().sum()}")
print(f"  NaN mu_mlp     : {test['mu_mlp'].isna().sum()}")

print("\n[5] Loading meta_clean.csv and merging sector ...")

meta = pd.read_csv('data/clean/meta_clean.csv', usecols=['ticker', 'sector'])
print(f"  Metadata : {meta.shape}  |  {meta['ticker'].nunique()} tickers")

train = train.merge(meta, on='ticker', how='left')
test  = test.merge(meta,  on='ticker', how='left')

print(f"  Train NaN sector : {train['sector'].isna().sum()}")
print(f"  Test  NaN sector : {test['sector'].isna().sum()}")

print("\n[6] Dropping NaN rows ...")

train = train.dropna(subset=['sector'])
test  = test.dropna(subset=['mu_ridge', 'mu_xgboost', 'mu_mlp', 'sector'])

print(f"  Train final: {train.shape}  |  {train['date'].min().date()} – {train['date'].max().date()}  |  {train['ticker'].nunique()} tickers")
print(f"  Test  final: {test.shape}   |  {test['date'].min().date()} – {test['date'].max().date()}   |  {test['ticker'].nunique()} tickers")

print("\n[7] Saving ...")

train.to_csv('data/results/step7/train_panel.csv', index=False)
test.to_csv( 'data/results/step7/test_panel.csv',  index=False)

print(f"  Saved: data/results/step7/train_panel.csv  ({train.shape[0]:,} rows  x  {train.shape[1]} cols)")
print(f"  Saved: data/results/step7/test_panel.csv   ({test.shape[0]:,} rows   x  {test.shape[1]} cols)")
print(f"\n  Columns in test_panel : {list(test.columns)}")
print("\n  Done. Run 7b_rga.py next.")
