"""
SAFE group parity is computed using the RGA-parity logic from the official safeaipackage.
For each rebalancing date and sector, RGA compares realized target ranks with
model-implied expected-return scores.
The parity gap is max sector RGA minus min sector RGA.
Because sectors are market groups rather than protected demographic attributes, this script
reports group parity rather than ethical fairness.

Reference: Babaei and Giudici, "A statistical package for safe artificial intelligence".
"""

import os

import numpy as np
import pandas as pd
from safeaipackage import core

OUT_DIR    = "data/results/step7"
PANEL_PATH = os.path.join(OUT_DIR, "test_panel.csv")

MODELS               = ["ridge", "xgboost", "mlp"]
GROUP_COL            = "sector"
MIN_ASSETS_PER_GROUP = 10
MIN_GROUPS_PER_DATE  = 2

# target_rank from step4 uses cs_rank (pct=True, ascending), so rank=1.0 is the best asset.
# Higher target_rank = better realized future return → no inversion needed.
RANK_HIGHER_IS_BETTER = True  # documented in 4a_features.py

os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(PANEL_PATH):
    raise FileNotFoundError(f"Input panel not found: {PANEL_PATH}")

test = pd.read_csv(PANEL_PATH)
test["date"] = pd.to_datetime(test["date"])

required_cols = ["date", "target_rank", GROUP_COL] + [f"mu_{m}" for m in MODELS]
for col in required_cols:
    if col not in test.columns:
        raise ValueError(f"Required column missing from test_panel.csv: '{col}'")

test["target_rank"] = pd.to_numeric(test["target_rank"], errors="coerce")
for m in MODELS:
    test[f"mu_{m}"] = pd.to_numeric(test[f"mu_{m}"], errors="coerce")

dates = sorted(test["date"].dropna().unique())

group_detail_rows = []
gap_rows          = []
skipped_rows      = []

print("\nSAFE Group Parity / RGA Gap")
print("-" * 50)

for model in MODELS:
    col_pred    = f"mu_{model}"
    gap_values  = []

    for d in dates:
        date_str = pd.Timestamp(d).date().isoformat()

        sub = test[test["date"] == d].copy()
        sub = sub.dropna(subset=["target_rank", col_pred, GROUP_COL])

        group_rga_values  = []
        group_n_assets    = []
        valid_group_rows  = []

        for group in sub[GROUP_COL].unique():
            sub_group = sub[sub[GROUP_COL] == group]

            if len(sub_group) < MIN_ASSETS_PER_GROUP:
                continue

            if RANK_HIGHER_IS_BETTER:
                ytest = sub_group["target_rank"].astype(float).values
            else:
                ytest = -sub_group["target_rank"].astype(float).values

            yhat = sub_group[col_pred].astype(float).values

            if np.nanstd(ytest) < 1e-12:
                continue
            if np.nanstd(yhat) < 1e-12:
                continue

            try:
                rga_val = float(core.rga(ytest.tolist(), yhat.tolist()))
            except Exception as exc:
                skipped_rows.append({
                    "date": date_str, "model": model,
                    "reason": "rga_error",
                    "details": f"sector={group}: {exc}",
                })
                continue

            group_rga_values.append(rga_val)
            group_n_assets.append(len(sub_group))
            valid_group_rows.append({
                "date":           date_str,
                "model":          model,
                "group_variable": GROUP_COL,
                "group":          group,
                "RGA":            round(rga_val, 6),
                "n_assets":       len(sub_group),
            })

        if len(group_rga_values) < MIN_GROUPS_PER_DATE:
            skipped_rows.append({
                "date": date_str, "model": model,
                "reason": "too_few_valid_groups",
                "details": f"valid groups = {len(group_rga_values)}, need {MIN_GROUPS_PER_DATE}",
            })
            continue

        group_detail_rows.extend(valid_group_rows)

        rga_gap = float(max(group_rga_values) - min(group_rga_values))
        gap_values.append(rga_gap)

        gap_rows.append({
            "date":                  date_str,
            "model":                 model,
            "group_variable":        GROUP_COL,
            "RGA_gap":               round(rga_gap, 6),
            "min_group_RGA":         round(float(min(group_rga_values)), 6),
            "max_group_RGA":         round(float(max(group_rga_values)), 6),
            "n_valid_groups":        len(group_rga_values),
            "mean_n_assets_per_group": round(float(np.mean(group_n_assets)), 2),
        })

    n_dates  = len(gap_values)
    mean_gap = float(np.mean(gap_values)) if n_dates > 0 else float("nan")
    print(f"Model {model:<8}: mean gap = {mean_gap:.6f}, dates = {n_dates}")

group_detail_df = pd.DataFrame(
    group_detail_rows,
    columns=["date", "model", "group_variable", "group", "RGA", "n_assets"],
)
gap_df = pd.DataFrame(
    gap_rows,
    columns=["date", "model", "group_variable", "RGA_gap",
             "min_group_RGA", "max_group_RGA", "n_valid_groups", "mean_n_assets_per_group"],
)
skipped_df = pd.DataFrame(
    skipped_rows,
    columns=["date", "model", "reason", "details"],
)

summary_rows = []
for model in MODELS:
    sub_gap = gap_df[gap_df["model"] == model]
    vals    = sub_gap["RGA_gap"].dropna().values
    n       = len(vals)
    summary_rows.append({
        "model":              model,
        "group_variable":     GROUP_COL,
        "mean_RGA_gap":       round(float(np.mean(vals)),        6) if n > 0 else np.nan,
        "std_RGA_gap":        round(float(np.std(vals, ddof=1)), 6) if n > 1 else np.nan,
        "min_RGA_gap":        round(float(np.min(vals)),         6) if n > 0 else np.nan,
        "max_RGA_gap":        round(float(np.max(vals)),         6) if n > 0 else np.nan,
        "n_dates":            n,
        "mean_valid_groups":  round(float(sub_gap["n_valid_groups"].mean()), 2) if n > 0 else np.nan,
    })
summary_df = pd.DataFrame(
    summary_rows,
    columns=["model", "group_variable", "mean_RGA_gap", "std_RGA_gap",
             "min_RGA_gap", "max_RGA_gap", "n_dates", "mean_valid_groups"],
)

group_detail_path = os.path.join(OUT_DIR, "rgf_group_rga_detail_by_date.csv")
gap_path          = os.path.join(OUT_DIR, "rgf_gap_by_date.csv")
summary_path      = os.path.join(OUT_DIR, "rgf_summary.csv")
skipped_path      = os.path.join(OUT_DIR, "rgf_skipped_dates.csv")

group_detail_df.to_csv(group_detail_path, index=False)
gap_df.to_csv(gap_path,                   index=False)
summary_df.to_csv(summary_path,           index=False)
skipped_df.to_csv(skipped_path,           index=False)

print(f"Saved: {group_detail_path}")
print(f"Saved: {gap_path}")
print(f"Saved: {summary_path}")
print(f"Saved: {skipped_path}")
