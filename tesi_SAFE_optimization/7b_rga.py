"""
SAFE Accuracy is computed using Rank Graduation Accuracy from the official safeaipackage.
For each rebalancing date, ytest is the realized cross-sectional target rank and yhat is
the model-implied expected return score. The resulting RGA measures concordance between
realized asset ranking and predicted asset ranking.

Reference: Babaei and Giudici, "A statistical package for safe artificial intelligence".
"""

import os

import numpy as np
import pandas as pd
from safeaipackage import core

OUT_DIR    = "data/results/step7"
PANEL_PATH = os.path.join(OUT_DIR, "test_panel.csv")

MODELS     = ["ridge", "xgboost", "mlp"]
MIN_ASSETS = 10

# target_rank from step4 uses cs_rank (pct=True, ascending), so rank=1.0 means
# the asset had the highest realized return in that cross-section.
# Higher target_rank = better realized future return → no inversion needed.
RANK_HIGHER_IS_BETTER = True  # documented in 4a_features.py: target_rank = cs_rank(target_raw)

if not os.path.exists(PANEL_PATH):
    raise FileNotFoundError(f"Input panel not found: {PANEL_PATH}")

panel = pd.read_csv(PANEL_PATH)

required_cols = ["date", "target_rank"] + [f"mu_{m}" for m in MODELS]
for col in required_cols:
    if col not in panel.columns:
        raise ValueError(f"Required column missing from test_panel.csv: '{col}'")

panel["date"] = pd.to_datetime(panel["date"])
panel["target_rank"] = pd.to_numeric(panel["target_rank"], errors="coerce")
for m in MODELS:
    panel[f"mu_{m}"] = pd.to_numeric(panel[f"mu_{m}"], errors="coerce")

dates = sorted(panel["date"].dropna().unique())

detail_rows  = []
summary_rows = []
skipped_rows = []

print("\nSAFE Accuracy / RGA")
print("-" * 50)

for model in MODELS:
    col_pred   = f"mu_{model}"
    rga_values = []
    n_assets_per_date = []

    for d in dates:
        date_str = pd.Timestamp(d).date().isoformat()
        sub = panel.loc[panel["date"] == d].copy()
        sub = sub.dropna(subset=["target_rank", col_pred])
        n_assets = len(sub)

        if n_assets < MIN_ASSETS:
            skipped_rows.append({
                "date": date_str, "model": model,
                "reason": "too_few_assets", "n_assets": n_assets, "details": "",
            })
            continue

        if np.nanstd(sub["target_rank"].values) < 1e-12:
            skipped_rows.append({
                "date": date_str, "model": model,
                "reason": "constant_target_rank", "n_assets": n_assets, "details": "",
            })
            continue

        if RANK_HIGHER_IS_BETTER:
            ytest = sub["target_rank"].astype(float).tolist()
        else:
            # invert so that higher value still means better asset
            ytest = (-sub["target_rank"].astype(float).values).tolist()

        yhat = sub[col_pred].astype(float).tolist()

        if np.nanstd(sub[col_pred].values) < 1e-12:
            skipped_rows.append({
                "date": date_str, "model": model,
                "reason": "constant_yhat", "n_assets": n_assets, "details": "",
            })
            continue

        try:
            rga_d = float(core.rga(ytest, yhat))
        except Exception as exc:
            skipped_rows.append({
                "date": date_str, "model": model,
                "reason": "rga_error", "n_assets": n_assets, "details": str(exc),
            })
            continue

        rga_values.append(rga_d)
        n_assets_per_date.append(n_assets)

        detail_rows.append({
            "date":     date_str,
            "model":    model,
            "RGA":      round(rga_d, 6),
            "n_assets": n_assets,
        })

    n_dates = len(rga_values)

    if n_dates == 0:
        mean_rga       = np.nan
        std_rga        = np.nan
        min_rga        = np.nan
        max_rga        = np.nan
        mean_n_assets  = np.nan
    else:
        mean_rga      = float(np.mean(rga_values))
        std_rga       = float(np.std(rga_values, ddof=1)) if n_dates > 1 else np.nan
        min_rga       = float(np.min(rga_values))
        max_rga       = float(np.max(rga_values))
        mean_n_assets = float(np.mean(n_assets_per_date))

    print(f"Model {model:<8}: mean RGA = {mean_rga:.6f}, dates = {n_dates}")

    summary_rows.append({
        "model":        model,
        "mean_RGA":     round(mean_rga,      6) if not np.isnan(mean_rga)      else np.nan,
        "std_RGA":      round(std_rga,       6) if not np.isnan(std_rga)       else np.nan,
        "min_RGA":      round(min_rga,       6) if not np.isnan(min_rga)       else np.nan,
        "max_RGA":      round(max_rga,       6) if not np.isnan(max_rga)       else np.nan,
        "n_dates":      n_dates,
        "mean_n_assets": round(mean_n_assets, 2) if not np.isnan(mean_n_assets) else np.nan,
    })

os.makedirs(OUT_DIR, exist_ok=True)

detail_df  = pd.DataFrame(detail_rows,  columns=["date", "model", "RGA", "n_assets"])
summary_df = pd.DataFrame(summary_rows, columns=["model", "mean_RGA", "std_RGA",
                                                  "min_RGA", "max_RGA", "n_dates",
                                                  "mean_n_assets"])

skipped_df = pd.DataFrame(
    skipped_rows,
    columns=["date", "model", "reason", "n_assets", "details"],
)

detail_path  = os.path.join(OUT_DIR, "rga_detail_by_date.csv")
summary_path = os.path.join(OUT_DIR, "rga_summary.csv")
skipped_path = os.path.join(OUT_DIR, "rga_skipped_dates.csv")

detail_df.to_csv(detail_path,   index=False)
summary_df.to_csv(summary_path, index=False)
skipped_df.to_csv(skipped_path, index=False)

print(f"Saved: {detail_path}")
print(f"Saved: {summary_path}")
print(f"Saved: {skipped_path}")
