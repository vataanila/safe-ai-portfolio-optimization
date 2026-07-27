"""
This script summarizes the four SAFE dimensions separately.
It intentionally does not compute a composite Compliance Score because the SAFE
paper/package define RGA, RGR, RGE, and RGA-parity as separate diagnostic dimensions.
"""

import os
import sys

import numpy as np
import pandas as pd

os.makedirs('data/results/step7', exist_ok=True)

# ── Input file paths ──────────────────────────────────────────────────────────
FILES = {
    'rga': 'data/results/step7/rga_summary.csv',
    'rgr': 'data/results/step7/rgr_group_summary.csv',
    'rge': 'data/results/step7/rge_summary.csv',
    'rgf': 'data/results/step7/rgf_summary.csv',
}

# ── Validation: confirm all input files exist ─────────────────────────────────
for key, path in FILES.items():
    if not os.path.exists(path):
        sys.exit(f"ERROR: required input file not found: {path}")

# ── Load inputs ───────────────────────────────────────────────────────────────
rga_df       = pd.read_csv(FILES['rga'])
rgr_group_df = pd.read_csv(FILES['rgr'])
rge_df       = pd.read_csv(FILES['rge'])
rgf_df       = pd.read_csv(FILES['rgf'])

# ── Validate required columns ─────────────────────────────────────────────────
REQUIRED = {
    'rga': ['model', 'mean_RGA', 'n_dates'],
    'rgr': ['model', 'group', 'mean_RGR', 'n_dates'],
    'rge': ['model', 'variable', 'mean_RGE', 'n_dates'],
    'rgf': ['model', 'mean_RGA_gap', 'n_dates'],
}
dfs = {'rga': rga_df, 'rgr': rgr_group_df, 'rge': rge_df, 'rgf': rgf_df}

for key, cols in REQUIRED.items():
    missing = [c for c in cols if c not in dfs[key].columns]
    if missing:
        sys.exit(f"ERROR: {FILES[key]} is missing columns: {missing}")

# ── RGA: mean_RGA per model ───────────────────────────────────────────────────
rga = (
    rga_df[['model', 'mean_RGA', 'n_dates']]
    .rename(columns={'mean_RGA': 'RGA_accuracy', 'n_dates': 'RGA_n_dates'})
)

# ── RGR: group == "all_features" ──────────────────────────────────────────────
rgr_all = rgr_group_df[rgr_group_df['group'] == 'all_features'].copy()
if rgr_all.empty:
    sys.exit(
        "ERROR: rgr_group_summary.csv contains no rows where group == 'all_features'. "
        "Check step7b output."
    )
rgr = (
    rgr_all[['model', 'mean_RGR', 'n_dates']]
    .rename(columns={'mean_RGR': 'RGR_group_robustness', 'n_dates': 'RGR_n_dates'})
)

# ── RGE: variable with highest mean_RGE per model ────────────────────────────
idx_max = rge_df.groupby('model', sort=False)['mean_RGE'].idxmax()
rge_top = rge_df.loc[idx_max, ['model', 'variable', 'mean_RGE', 'n_dates']].copy()
rge_top = rge_top.rename(columns={
    'mean_RGE':  'max_RGE_explainability',
    'variable':  'max_RGE_variable',
    'n_dates':   'RGE_n_dates_for_max_variable',
})

# ── RGF: mean_RGA_gap and derived sector_parity ──────────────────────────────
rgf = (
    rgf_df[['model', 'mean_RGA_gap', 'n_dates']]
    .rename(columns={'mean_RGA_gap': 'sector_RGA_gap', 'n_dates': 'RGF_n_dates'})
)
rgf['sector_parity'] = 1.0 - rgf['sector_RGA_gap']

# ── Build all-model index (union) to catch missing models gracefully ──────────
all_models = (
    pd.concat([
        rga_df[['model']],
        rgr_all[['model']],
        rge_df[['model']],
        rgf_df[['model']],
    ])
    .drop_duplicates()
    .reset_index(drop=True)
)

# ── Merge with outer joins so missing models get NaN, not dropped ─────────────
summary = all_models.copy()
for part, key in [(rga, 'RGA'), (rgr, 'RGR'), (rge_top, 'RGE'), (rgf, 'RGF')]:
    before = set(summary['model'])
    summary = summary.merge(part, on='model', how='left')
    after   = set(summary['model'])
    if before != after:
        print(f"WARNING: merge on {key} changed model set unexpectedly.")

# Warn about any models with NaN in core metrics
core_cols = ['RGA_accuracy', 'RGR_group_robustness', 'max_RGE_explainability', 'sector_RGA_gap']
for _, row in summary.iterrows():
    missing_dims = [c for c in core_cols if pd.isna(row[c])]
    if missing_dims:
        print(f"WARNING: model '{row['model']}' has NaN for: {missing_dims}")

# ── Final column order ────────────────────────────────────────────────────────
main_cols = [
    'model',
    'RGA_accuracy',
    'RGR_group_robustness',
    'max_RGE_explainability',
    'max_RGE_variable',
    'sector_RGA_gap',
    'sector_parity',
    'RGA_n_dates',
    'RGR_n_dates',
    'RGE_n_dates_for_max_variable',
    'RGF_n_dates',
]
summary = summary[main_cols]

# ── Console summary ───────────────────────────────────────────────────────────
print('\nSAFE Dimension Summary')
print('-' * 72)
for _, row in summary.iterrows():
    rga_val  = f"{row['RGA_accuracy']:.6f}"  if pd.notna(row['RGA_accuracy'])          else 'NaN'
    rgr_val  = f"{row['RGR_group_robustness']:.6f}" if pd.notna(row['RGR_group_robustness'])  else 'NaN'
    rge_val  = f"{row['max_RGE_explainability']:.6f}" if pd.notna(row['max_RGE_explainability']) else 'NaN'
    rge_var  = str(row['max_RGE_variable']) if pd.notna(row['max_RGE_variable']) else 'NaN'
    gap_val  = f"{row['sector_RGA_gap']:.6f}" if pd.notna(row['sector_RGA_gap'])        else 'NaN'
    par_val  = f"{row['sector_parity']:.6f}"  if pd.notna(row['sector_parity'])          else 'NaN'
    print(
        f"Model {row['model']:<10}: "
        f"RGA={rga_val}, RGR={rgr_val}, "
        f"max RGE={rge_val} ({rge_var}), "
        f"sector gap={gap_val}, sector parity={par_val}"
    )
print()

# ── Save wide-format CSV ──────────────────────────────────────────────────────
out_wide = 'data/results/step7/safe_dimension_summary.csv'
summary.to_csv(out_wide, index=False)
print(f'Saved: {out_wide}')

# ── Build long-format version ─────────────────────────────────────────────────
long_rows = []
for _, row in summary.iterrows():
    m = row['model']
    long_rows += [
        (m, 'Accuracy',     'RGA_accuracy',          row['RGA_accuracy'],          'mean RGA across rebalancing dates'),
        (m, 'Robustness',   'RGR_group_robustness',  row['RGR_group_robustness'],  'group RGR for all_features'),
        (m, 'Explainability','max_RGE_explainability',row['max_RGE_explainability'],str(row['max_RGE_variable'])),
        (m, 'Group parity', 'sector_RGA_gap',        row['sector_RGA_gap'],        'max sector RGA minus min sector RGA'),
        (m, 'Group parity', 'sector_parity',         row['sector_parity'],         '1 - sector_RGA_gap'),
    ]

long_df = pd.DataFrame(long_rows, columns=['model', 'dimension', 'metric', 'value', 'detail'])

out_long = 'data/results/step7/safe_dimension_summary_long.csv'
long_df.to_csv(out_long, index=False)
print(f'Saved: {out_long}')

# ── Grouped bar chart: four SAFE dimensions by model ─────────────────────────
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size']   = 11

os.makedirs('figures/step7', exist_ok=True)

MODELS  = ['ridge', 'xgboost', 'mlp']
COLORS  = {'ridge': '#2166AC', 'xgboost': '#B2182B', 'mlp': '#4D9221'}
LABELS  = ['Accuracy\n(RGA)', 'Robustness\n(RGR)', 'Explainability\n(max RGE)', 'Group Parity\n(1 − sector gap)']
DIM_COLS = ['RGA_accuracy', 'RGR_group_robustness', 'max_RGE_explainability', 'sector_parity']

summary_idx = summary.set_index('model')

n_dims   = len(DIM_COLS)
n_models = len(MODELS)
bar_w    = 0.22
x        = np.arange(n_dims)

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

for i, model in enumerate(MODELS):
    if model not in summary_idx.index:
        print(f"WARNING: model '{model}' not found in summary; skipping bar.")
        continue
    values = [summary_idx.loc[model, col] for col in DIM_COLS]
    offset = (i - (n_models - 1) / 2) * bar_w
    bars = ax.bar(x + offset, values, width=bar_w, color=COLORS[model],
                  label=model.capitalize() if model != 'xgboost' else 'XGBoost',
                  zorder=3)
    for bar, val in zip(bars, values):
        if pd.notna(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7, zorder=4)

ax.set_xticks(x)
ax.set_xticklabels(LABELS)
ax.set_ylabel('Score')
ax.yaxis.grid(True, linestyle=':', alpha=0.6, zorder=0)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(frameon=False)

fig.tight_layout()
out_fig = 'figures/step7/safe_dimensions_grouped_bar.png'
fig.savefig(out_fig, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'Saved: {out_fig}')
