"""
RGR compliance vector for the integrated SAFE Compliance Score (same
Giudici/Kolesnikov integration as 8a_rga_vector.py).

For each model and rebalancing date, group RGR (all features perturbed
together) is evaluated across a grid of perturbation intensities: 1.0 at
p=0 by definition, safeaipackage's group=True call for each p>0. This
does not replace 7d_rgr.py's feature-wise RGR at one fixed
perturbation -- it's the multi-intensity curve the compliance score
needs.

Needs the trained model/scaler files from Step 5 in addition to
test_panel.csv, since perturbation is applied through the actual
trained predict function. Writes rgr_vector_* under data/results/step8/.

Reference: Giudici and Kolesnikov, "SAFE AI metrics: An integrated
approach"; Babaei and Giudici, "A statistical package for safe
artificial intelligence".
Author: Anila Vata
"""

import glob as _glob
import json
import os
import re
import warnings

import joblib
import numpy as np
import pandas as pd
from safeaipackage import check_robustness
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

OUT_DIR    = "data/results/step8"
PANEL_PATH = "data/results/step7/test_panel.csv"

MODEL_DIR = {
    "ridge":   "data/results/step5/ridge/models",
    "xgboost": "data/results/step5/xgboost/models",
    "mlp":     "data/results/step5/mlp/models",
}

MODELS     = ["ridge", "xgboost", "mlp"]
MIN_ASSETS = 10

PERTURBATION_GRID = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25,
                     0.30, 0.35, 0.40, 0.45, 0.50]

# Column names for the wide pivot (rgr_vector_by_model.csv)
_P_COL = {
    0.00: "RGR_p00", 0.05: "RGR_p05", 0.10: "RGR_p10",
    0.15: "RGR_p15", 0.20: "RGR_p20", 0.25: "RGR_p25",
    0.30: "RGR_p30", 0.35: "RGR_p35", 0.40: "RGR_p40",
    0.45: "RGR_p45", 0.50: "RGR_p50",
}

os.makedirs(OUT_DIR, exist_ok=True)

class ScaledRegressor(RegressorMixin, BaseEstimator):
    """Wraps model + training-time StandardScaler: scales features inside predict()."""
    def __init__(self, scaler, model, feature_cols):
        self.scaler       = scaler
        self.model        = model
        self.feature_cols = feature_cols

    def predict(self, X):
        X = pd.DataFrame(X, columns=self.feature_cols)
        return self.model.predict(self.scaler.transform(X[self.feature_cols]))

class ColumnOrderRegressor(RegressorMixin, BaseEstimator):
    """Enforces column order before calling model.predict()."""
    def __init__(self, model, feature_cols):
        self.model        = model
        self.feature_cols = feature_cols

    def predict(self, X):
        X = pd.DataFrame(X, columns=self.feature_cols)
        return self.model.predict(X[self.feature_cols])

class _MissingTrainingScaler(Exception):
    """Raised when Ridge has no training-time scaler available."""

def load_feature_cols(model_name):
    """Load the authoritative feature list saved by Step 5 for this model."""
    path = os.path.join(MODEL_DIR[model_name], f"{model_name}_feature_cols.json")
    with open(path) as f:
        return json.load(f)

def build_model_date_map(model_name):
    """Scan the model directory for {model_name}_model_YYYY-MM-DD.pkl files and
    return a dict mapping each monthly Period to the date string in the filename.
    Raises ValueError if two model files fall in the same monthly period."""
    pattern  = os.path.join(MODEL_DIR[model_name], f"{model_name}_model_*.pkl")
    files    = _glob.glob(pattern)
    date_map = {}
    for fpath in files:
        fname = os.path.basename(fpath)
        m = re.search(r'(\d{4}-\d{2}-\d{2})\.pkl$', fname)
        if not m:
            continue
        date_str = m.group(1)
        period   = pd.Period(date_str[:7], freq='M')
        if period in date_map:
            raise ValueError(
                f"Two {model_name} model files share period {period}: "
                f"{date_map[period]} and {date_str}"
            )
        date_map[period] = date_str
    return date_map

def load_proxy(model_name, date_str, feat_cols):
    """
    Load the trained model for the given model and date and return a wrapper
    whose .predict(X) reproduces the actual trained prediction function,
    including training-time preprocessing.  No scaler is ever fitted here.

    Ridge lookup order:
      1. ridge_model_*.pkl, dict with model+scaler
      2. ridge_model_*.pkl - sklearn Pipeline
      3. ridge_model_*.pkl (plain) + ridge_scaler_*.pkl
      4. ridge_bundle_*.pkl, dict with model+scaler
    Falls back to _MissingTrainingScaler if a model is found but no scaler,
    or FileNotFoundError if neither model nor bundle is available.
    """
    if model_name == "ridge":
        model_path  = os.path.join(MODEL_DIR["ridge"], f"ridge_model_{date_str}.pkl")
        bundle_path = os.path.join(MODEL_DIR["ridge"], f"ridge_bundle_{date_str}.pkl")

        if os.path.exists(model_path):
            obj = joblib.load(model_path)

            # Case 1: dict with embedded model + scaler
            if isinstance(obj, dict):
                if "model" in obj and "scaler" in obj:
                    return ScaledRegressor(obj["scaler"], obj["model"], feat_cols)
                # dict without scaler - fall through to bundle

            # Case 2: sklearn Pipeline (scaler baked in)
            elif isinstance(obj, Pipeline):
                return ColumnOrderRegressor(obj, feat_cols)

            else:
                # Case 3: plain model, try separate scaler file
                scaler_path = os.path.join(MODEL_DIR["ridge"], f"ridge_scaler_{date_str}.pkl")
                if os.path.exists(scaler_path):
                    sc = joblib.load(scaler_path)
                    return ScaledRegressor(sc, obj, feat_cols)
                # No scaler file -- fall through to bundle

            # Case 4: fallback to bundle file
            if os.path.exists(bundle_path):
                bundle = joblib.load(bundle_path)
                if isinstance(bundle, dict) and "model" in bundle and "scaler" in bundle:
                    return ScaledRegressor(bundle["scaler"], bundle["model"], feat_cols)
                raise _MissingTrainingScaler(
                    f"Ridge bundle exists ({bundle_path}) but contains no training-time scaler."
                )

            # Model found but no scaler available anywhere
            raise _MissingTrainingScaler(
                f"Ridge model loaded ({model_path}) but no training-time scaler found "
                "(no scaler key in dict, no Pipeline, no ridge_scaler file, no usable bundle)."
            )

        # model_path does not exist, try bundle directly
        if os.path.exists(bundle_path):
            bundle = joblib.load(bundle_path)
            if isinstance(bundle, dict) and "model" in bundle and "scaler" in bundle:
                return ScaledRegressor(bundle["scaler"], bundle["model"], feat_cols)
            raise _MissingTrainingScaler(
                f"Ridge bundle exists ({bundle_path}) but contains no training-time scaler."
            )

        raise FileNotFoundError(
            f"No Ridge model or bundle found for date {date_str} "
            f"(checked: {model_path}, {bundle_path})"
        )

    if model_name == "xgboost":
        model_path = os.path.join(MODEL_DIR["xgboost"], f"xgboost_model_{date_str}.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"XGBoost model not found: {model_path}")
        obj = joblib.load(model_path)
        if isinstance(obj, dict):
            mdl = obj.get("model", obj)
            sc  = obj.get("scaler", None)
            if sc is not None:
                return ScaledRegressor(sc, mdl, feat_cols)
            return ColumnOrderRegressor(mdl, feat_cols)
        return ColumnOrderRegressor(obj, feat_cols)

    if model_name == "mlp":
        model_path  = os.path.join(MODEL_DIR["mlp"], f"mlp_model_{date_str}.pkl")
        scaler_path = os.path.join(MODEL_DIR["mlp"], f"mlp_scaler_{date_str}.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MLP model not found: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"MLP scaler not found: {scaler_path}")
        mdl = joblib.load(model_path)
        sc  = joblib.load(scaler_path)
        return ScaledRegressor(sc, mdl, feat_cols)

    raise ValueError(f"Unknown model: {model_name}")

if not os.path.exists(PANEL_PATH):
    raise FileNotFoundError(f"Input panel not found: {PANEL_PATH}")

test = pd.read_csv(PANEL_PATH)
test["date"] = pd.to_datetime(test["date"])

rebal_dates = sorted(test["date"].dropna().unique())

detail_rows  = []
skipped_rows = []

print("\nRGR Compliance Vector  (group=True, perturbation grid)")
print("-" * 60)

for model_name in MODELS:
    feat_cols = load_feature_cols(model_name)

    missing_feat = [c for c in feat_cols if c not in test.columns]
    if missing_feat:
        raise ValueError(
            f"Feature columns for {model_name} missing from test_panel.csv: {missing_feat}"
        )

    model_date_map = build_model_date_map(model_name)

    valid_dates   = 0
    skipped_dates = 0

    for d in rebal_dates:
        period   = pd.Timestamp(d).to_period("M")
        date_str = model_date_map.get(period)
        raw_date = pd.Timestamp(d).date().isoformat()

        if date_str is None:
            skipped_rows.append({
                "date": raw_date, "model": model_name,
                "reason": "missing_model_file", "n_assets": 0,
                "details": f"no model file for period {period}",
                "model_date": "",
            })
            skipped_dates += 1
            continue

        sub = test[test["date"] == d].copy()
        sub = sub.dropna(subset=feat_cols)
        n_assets = len(sub)

        if n_assets < MIN_ASSETS:
            skipped_rows.append({
                "date": raw_date, "model": model_name,
                "reason": "too_few_assets", "n_assets": n_assets,
                "details": "", "model_date": date_str,
            })
            skipped_dates += 1
            continue

        xtest_d = sub[feat_cols].reset_index(drop=True)

        # Load trained model wrapper
        try:
            proxy = load_proxy(model_name, date_str, feat_cols)
        except _MissingTrainingScaler as exc:
            skipped_rows.append({
                "date": raw_date, "model": model_name,
                "reason": "missing_training_scaler", "n_assets": n_assets,
                "details": str(exc), "model_date": date_str,
            })
            skipped_dates += 1
            continue
        except FileNotFoundError as exc:
            reason = "missing_scaler" if "scaler" in str(exc).lower() else "missing_model_file"
            skipped_rows.append({
                "date": raw_date, "model": model_name,
                "reason": reason, "n_assets": n_assets,
                "details": str(exc), "model_date": date_str,
            })
            skipped_dates += 1
            continue
        except Exception as exc:
            skipped_rows.append({
                "date": raw_date, "model": model_name,
                "reason": "model_load_error", "n_assets": n_assets,
                "details": str(exc), "model_date": date_str,
            })
            skipped_dates += 1
            continue

        # Original predictions - same proxy passed to safeaipackage
        try:
            yhat = proxy.predict(xtest_d).tolist()
        except Exception as exc:
            skipped_rows.append({
                "date": raw_date, "model": model_name,
                "reason": "prediction_error", "n_assets": n_assets,
                "details": str(exc), "model_date": date_str,
            })
            skipped_dates += 1
            continue

        if np.std(yhat) < 1e-12:
            skipped_rows.append({
                "date": raw_date, "model": model_name,
                "reason": "constant_yhat", "n_assets": n_assets,
                "details": "", "model_date": date_str,
            })
            skipped_dates += 1
            continue

        # Perturbation grid—collect into a temporary list first so that
        # partial results are never written to detail_rows if any level fails.
        date_detail_rows = []
        date_ok = True
        for p in PERTURBATION_GRID:
            if p == 0.00:
                # No perturbation: original_yhat == perturbed_yhat → RGR = 1.0 by definition
                rgr_val = 1.0
            else:
                try:
                    rgr_result = check_robustness.compute_rgr_values(
                        xtest_d,
                        yhat,
                        proxy,
                        feat_cols,
                        perturbation_percentage=p,
                        group=True,
                    )
                    rgr_val = round(float(rgr_result["RGR"].iloc[0]), 6)
                except Exception as exc:
                    skipped_rows.append({
                        "date": raw_date, "model": model_name,
                        "reason": "rgr_error", "n_assets": n_assets,
                        "details": f"p={p}: {exc}", "model_date": date_str,
                    })
                    date_ok = False
                    break

            date_detail_rows.append({
                "date":                   raw_date,
                "model":                  model_name,
                "group":                  "all_features",
                "perturbation_percentage": p,
                "RGR":                    rgr_val,
                "n_assets":               n_assets,
                "model_date":             date_str,
            })

        if date_ok:
            detail_rows.extend(date_detail_rows)
            valid_dates += 1
        else:
            skipped_dates += 1

    print(f"  Model {model_name:<8}: valid dates = {valid_dates}, skipped = {skipped_dates}")

detail_df = pd.DataFrame(
    detail_rows,
    columns=["date", "model", "group", "perturbation_percentage",
             "RGR", "n_assets", "model_date"],
)

skipped_df = pd.DataFrame(
    skipped_rows,
    columns=["date", "model", "reason", "n_assets", "details", "model_date"],
)

# ── Summary: mean/std/min/max RGR by model × perturbation level ──────────────
summary_rows = []
for (model_name, p), grp in detail_df.groupby(["model", "perturbation_percentage"]):
    vals = grp["RGR"].dropna().values
    n    = len(vals)
    summary_rows.append({
        "model":                  model_name,
        "perturbation_percentage": p,
        "mean_RGR":               round(float(np.mean(vals)),        6) if n > 0 else np.nan,
        "std_RGR":                round(float(np.std(vals, ddof=1)), 6) if n > 1 else np.nan,
        "min_RGR":                round(float(np.min(vals)),         6) if n > 0 else np.nan,
        "max_RGR":                round(float(np.max(vals)),         6) if n > 0 else np.nan,
        "n_dates":                n,
        "mean_n_assets":          round(float(grp["n_assets"].mean()), 2) if n > 0 else np.nan,
    })

summary_df = pd.DataFrame(
    summary_rows,
    columns=["model", "perturbation_percentage", "mean_RGR", "std_RGR",
             "min_RGR", "max_RGR", "n_dates", "mean_n_assets"],
)

# ── Wide pivot: one row per model, one column per perturbation level ──────────
# Values are mean_RGR from summary_df (the RGR compliance vector per model).
pivot = summary_df.pivot(
    index="model", columns="perturbation_percentage", values="mean_RGR"
)
pivot.columns = [_P_COL[c] for c in pivot.columns]
pivot = pivot.reset_index()

# Ensure canonical model order and all expected columns are present
pivot = pivot.set_index("model").reindex(MODELS).reset_index()
for col in _P_COL.values():
    if col not in pivot.columns:
        pivot[col] = np.nan

col_order = ["model"] + list(_P_COL.values())
pivot = pivot[col_order]

detail_path  = os.path.join(OUT_DIR, "rgr_vector_detail_by_date.csv")
summary_path = os.path.join(OUT_DIR, "rgr_vector_summary.csv")
vector_path  = os.path.join(OUT_DIR, "rgr_vector_by_model.csv")
skipped_path = os.path.join(OUT_DIR, "rgr_vector_skipped_dates.csv")

detail_df.to_csv(detail_path,  index=False)
summary_df.to_csv(summary_path, index=False)
pivot.to_csv(vector_path,       index=False)
skipped_df.to_csv(skipped_path, index=False)

print()
print(f"Saved: {detail_path}")
print(f"Saved: {summary_path}")
print(f"Saved: {vector_path}")
print(f"Saved: {skipped_path}")

print("\nRGR compliance vector (mean_RGR per perturbation level):")
print(f"  {'model':<10}", end="")
for p in PERTURBATION_GRID:
    print(f"  p={p:.2f}", end="")
print()

for _, row in pivot.iterrows():
    print(f"  {row['model']:<10}", end="")
    for col in _P_COL.values():
        val = row[col]
        print(f"  {val:.4f}" if pd.notna(val) else "     NaN", end="")
    print()

print(f"\n  detail rows : {len(detail_df)}")
print(f"  skipped rows: {len(skipped_df)}")

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["font.size"]   = 11

FIG_DIR         = "figures/step8/rgr"
RANDOM_BASELINE = 0.50
MAX_PERT        = max(PERTURBATION_GRID)   # 0.50

os.makedirs(FIG_DIR, exist_ok=True)

MODEL_DISPLAY = {
    "ridge":   ("Ridge",   "#2166AC"),
    "xgboost": ("XGBoost", "#B2182B"),
    "mlp":     ("MLP",     "#4D9221"),
}

def _model_curve(model_name):
    """Return (x_norm, y_values) for model_name, sorted by normalized perturbation.

    x_norm = perturbation_percentage / MAX_PERT  →  maps [0.00, 0.50] to [0.0, 1.0].
    """
    sub = (
        summary_df[summary_df["model"] == model_name]
        .sort_values("perturbation_percentage")
        .dropna(subset=["mean_RGR"])
    )
    x_norm = (sub["perturbation_percentage"] / MAX_PERT).tolist()
    y_vals = sub["mean_RGR"].tolist()
    return x_norm, y_vals

def _aurgr(x_vals, y_vals):
    """Trapezoidal AURGR from the plotted mean curve (x already normalized to [0,1])."""
    if len(x_vals) < 2:
        return np.nan
    return round(float(np.trapz(y_vals, x_vals)), 4)

def _apply_style(ax, title, y_lower, y_upper):
    ax.set_xlim(0.00, 1.00)
    ax.set_ylim(y_lower, y_upper)
    ax.set_xlabel("Normalized Perturbation")
    ax.set_ylabel("RGR")
    ax.grid(True, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def _save(fig, base_path):
    fig.tight_layout()
    fig.savefig(base_path + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {base_path}.png")

# ── Dynamic y-axis bounds (computed from actual data, not hardcoded) ──────────
global_min_rgr = summary_df["mean_RGR"].min()
y_lower = max(0.0, float(np.floor(global_min_rgr * 20) / 20) - 0.05)
y_upper = 1.05

print(f"\n  global_min_rgr = {global_min_rgr:.6f}")
print(f"  y_lower        = {y_lower:.4f}")
print(f"  y_upper        = {y_upper:.4f}")
print(f"  No clipping    : {global_min_rgr >= y_lower}")

# ── Individual model figures ──────────────────────────────────────────────────
print()
for model_name, (display_name, color) in MODEL_DISPLAY.items():
    x_vals, y_vals = _model_curve(model_name)

    if len(x_vals) < 2:
        print(f"  Warning: {model_name} has fewer than 2 valid RGR points; skipping figure.")
        continue

    aurgr = _aurgr(x_vals, y_vals)
    curve_label = (f"RGR Curve (AURGR = {aurgr:.4f})"
                   if not np.isnan(aurgr) else "RGR Curve")

    fig, ax = plt.subplots(figsize=(6, 5), facecolor="white")
    ax.set_facecolor("white")

    ax.plot(x_vals, y_vals, color=color, marker="o", linewidth=1.8,
            markersize=5, label=curve_label, zorder=3)
    ax.axhline(RANDOM_BASELINE, color="red", linestyle="--", linewidth=1.2,
               label=f"Random Baseline (RGR = {RANDOM_BASELINE:.2f})", zorder=2)

    _apply_style(ax, f"{display_name} RGR Curve", y_lower, y_upper)
    ax.legend(loc="upper right", frameon=True, framealpha=0.8)

    _save(fig, os.path.join(FIG_DIR, f"rgr_curve_{model_name}"))

# ── Combined figure: all models ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
ax.set_facecolor("white")

for model_name, (display_name, color) in MODEL_DISPLAY.items():
    x_vals, y_vals = _model_curve(model_name)
    if len(x_vals) < 2:
        continue
    ax.plot(x_vals, y_vals, color=color, marker="o", linewidth=1.8,
            markersize=5, label=display_name, zorder=3)

ax.axhline(RANDOM_BASELINE, color="red", linestyle="--", linewidth=1.2,
           label=f"Random Baseline (RGR = {RANDOM_BASELINE:.2f})", zorder=2)

_apply_style(ax, "RGR Curve by Model", y_lower, y_upper)
ax.legend(loc="upper right", frameon=True, framealpha=0.8)

_save(fig, os.path.join(FIG_DIR, "rgr_curve_by_model"))
