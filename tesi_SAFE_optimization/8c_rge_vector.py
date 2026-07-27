"""
Step 8c, RGE* compliance vector (third leg of the same Giudici/Kolesnikov
integration as 8a/8b).

The feature-wise RGE values already computed in Step 7c only tell us
individual feature contribution; here they're used just to fix a
date-specific removal order (ascending, least explainable first). Then,
for each removal step k, the k selected features are replaced together
with their training-time means (or modes for categoricals), the proxy
model re-predicts, and RGE_star = core.rga(original_yhat, modified_yhat)
is recorded. k=0 -> RGE_star = 1.0 by definition, giving the same
11-point shape as the RGA/RGR vectors.

Needs train_panel.csv (for the training-time means/modes),
test_panel.csv, Step 7c's rge_detail_by_date.csv for the removal order,
and the trained model/scaler files from Step 5. Writes rge_vector_*
under data/results/step8/.

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
from safeaipackage import core as safe_core
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

OUT_DIR         = "data/results/step8"
TRAIN_PATH      = "data/results/step7/train_panel.csv"
TEST_PATH       = "data/results/step7/test_panel.csv"
RGE_DETAIL_PATH = "data/results/step7/rge_detail_by_date.csv"

MODEL_DIR = {
    "ridge":   "data/results/step5/ridge/models",
    "xgboost": "data/results/step5/xgboost/models",
    "mlp":     "data/results/step5/mlp/models",
}

MODELS     = ["ridge", "xgboost", "mlp"]
MIN_ASSETS = 10

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
      1. ridge_model_*.pkl - dict with model+scaler
      2. ridge_model_*.pkl, sklearn Pipeline
      3. ridge_model_*.pkl (plain) + ridge_scaler_*.pkl
      4. ridge_bundle_*.pkl -- dict with model+scaler
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
                # Case 3: plain model -- try separate scaler file
                scaler_path = os.path.join(MODEL_DIR["ridge"], f"ridge_scaler_{date_str}.pkl")
                if os.path.exists(scaler_path):
                    sc = joblib.load(scaler_path)
                    return ScaledRegressor(sc, obj, feat_cols)
                # No scaler file, fall through to bundle

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

        # model_path does not exist - try bundle directly
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

def neutralize_features(xtest_d, xtrain_d, features_to_remove):
    """Replace the specified features in xtest_d with training-time statistics.

    Numerical features are replaced with the training mean.
    Categorical features are replaced with the training mode.
    All replacements are applied simultaneously on a copy of xtest_d.
    No preprocessing is fitted; only training-time summaries are used.
    """
    xmod = xtest_d.copy()
    for col in features_to_remove:
        if pd.api.types.is_numeric_dtype(xtrain_d[col]):
            fill_val = float(xtrain_d[col].mean())
        else:
            mode_vals = xtrain_d[col].mode()
            fill_val  = mode_vals.iloc[0] if len(mode_vals) > 0 else np.nan
        xmod[col] = fill_val
    return xmod

for _path in [TRAIN_PATH, TEST_PATH, RGE_DETAIL_PATH]:
    if not os.path.exists(_path):
        raise FileNotFoundError(f"Input file not found: {_path}")

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

train["date"] = pd.to_datetime(train["date"])
test["date"]  = pd.to_datetime(test["date"])

# Use target_end_date for no-leakage filter if available; fall back to date < d
_TRAIN_HAS_TARGET_END_DATE = "target_end_date" in train.columns
if _TRAIN_HAS_TARGET_END_DATE:
    train["target_end_date"] = pd.to_datetime(train["target_end_date"], errors="coerce")

# Load Step 7C feature-wise RGE results, used only to determine removal order
rge_detail = pd.read_csv(RGE_DETAIL_PATH)
rge_detail["date"] = rge_detail["date"].astype(str)

rebal_dates = sorted(test["date"].dropna().unique())

detail_rows       = []
order_rows        = []
skipped_rows      = []

print("\nRGE* Compliance Vector  (progressive feature neutralization)")
print("-" * 62)

for model_name in MODELS:
    feat_cols = load_feature_cols(model_name)
    n_feats   = len(feat_cols)

    # Validate feature columns exist in both panels
    for col in feat_cols:
        if col not in train.columns:
            raise ValueError(
                f"Feature column '{col}' for {model_name} missing from train_panel.csv"
            )
        if col not in test.columns:
            raise ValueError(
                f"Feature column '{col}' for {model_name} missing from test_panel.csv"
            )

    model_date_map = build_model_date_map(model_name)

    # Pre-filter Step 7C RGE rows for this model
    rge_model = rge_detail[rge_detail["model"] == model_name].copy()

    valid_dates   = 0
    skipped_dates = 0

    for d in rebal_dates:
        period   = pd.Timestamp(d).to_period("M")
        raw_date = pd.Timestamp(d).date().isoformat()
        date_str = model_date_map.get(period)

        if date_str is None:
            skipped_rows.append({
                "date": raw_date, "model": model_name,
                "reason": "missing_model_file", "n_assets": 0,
                "details": f"no model file for period {period}",
                "model_date": "",
            })
            skipped_dates += 1
            continue

        # ── Date-specific xtrain (no-leakage) ────────────────────────────────
        # Use observations whose realized target period ends no later than the
        # test date (target_end_date <= d). Fall back to date < d when absent.
        if _TRAIN_HAS_TARGET_END_DATE:
            train_d = train.loc[
                train["target_end_date"].notna()
                & (train["target_end_date"] <= pd.Timestamp(d))
            ].copy()
        else:
            train_d = train.loc[train["date"] < pd.Timestamp(d)].copy()

        xtrain_d = train_d[feat_cols].dropna().reset_index(drop=True)

        if len(xtrain_d) < MIN_ASSETS:
            skipped_rows.append({
                "date": raw_date, "model": model_name,
                "reason": "too_few_train_rows", "n_assets": 0,
                "details": "", "model_date": date_str,
            })
            skipped_dates += 1
            continue

        # ── xtest_d ───────────────────────────────────────────────────────────
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

        # ── Load trained model wrapper ────────────────────────────────────────
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

        # ── Original predictions ──────────────────────────────────────────────
        try:
            original_yhat = proxy.predict(xtest_d).tolist()
        except Exception as exc:
            skipped_rows.append({
                "date": raw_date, "model": model_name,
                "reason": "prediction_error", "n_assets": n_assets,
                "details": str(exc), "model_date": date_str,
            })
            skipped_dates += 1
            continue

        if np.std(original_yhat) < 1e-12:
            skipped_rows.append({
                "date": raw_date, "model": model_name,
                "reason": "constant_yhat", "n_assets": n_assets,
                "details": "", "model_date": date_str,
            })
            skipped_dates += 1
            continue

        # ── Date-specific feature removal order from Step 7C ─────────────────
        rge_date = rge_model[rge_model["date"] == raw_date][["variable", "RGE"]].copy()

        # Ensure all feature columns appear exactly once in the order
        present   = set(rge_date["variable"].tolist())
        expected  = set(feat_cols)
        if not expected.issubset(present):
            missing_vars = sorted(expected - present)
            skipped_rows.append({
                "date": raw_date, "model": model_name,
                "reason": "missing_rge_order", "n_assets": n_assets,
                "details": f"missing RGE for variables: {missing_vars}",
                "model_date": date_str,
            })
            skipped_dates += 1
            continue

        # Sort ascending: least explainable (lowest RGE) first
        rge_date = (
            rge_date[rge_date["variable"].isin(feat_cols)]
            .sort_values("RGE", ascending=True)
            .drop_duplicates(subset="variable")
            .reset_index(drop=True)
        )
        ordered_features = rge_date["variable"].tolist()

        # ── Progressive removal - accumulate into a temporary list ────────────
        date_detail_rows = []
        date_order_rows  = []
        date_ok          = True

        # Record feature order for this date
        for rank_idx, row in rge_date.iterrows():
            date_order_rows.append({
                "date":       raw_date,
                "model":      model_name,
                "model_date": date_str,
                "rank":       int(rank_idx) + 1,   # 1 = least explainable
                "variable":   row["variable"],
                "RGE":        round(float(row["RGE"]), 6),
                "order_type": "date_specific_ascending_RGE",
            })

        for k in range(n_feats + 1):
            if k == 0:
                # No features removed -- RGE_star = 1.0 by definition
                rge_star        = 1.0
                removed_list    = []
                removed_str     = ""
            else:
                removed_list = ordered_features[:k]
                removed_str  = ";".join(removed_list)

                try:
                    xtest_mod    = neutralize_features(xtest_d, xtrain_d, removed_list)
                    modified_yhat = proxy.predict(xtest_mod).tolist()
                    # RGE_star = rga(original, modified): measures remaining rank
                    # concordance after neutralization.
                    # Note: the package's RGE ≈ 1 - rga(original, modified), so
                    # RGE_star is the complementary similarity measure.
                    rge_star = round(float(safe_core.rga(original_yhat, modified_yhat)), 6)
                except Exception as exc:
                    skipped_rows.append({
                        "date": raw_date, "model": model_name,
                        "reason": "rge_vector_error", "n_assets": n_assets,
                        "details": f"k={k}: {exc}", "model_date": date_str,
                    })
                    date_ok = False
                    break

            date_detail_rows.append({
                "date":                     raw_date,
                "model":                    model_name,
                "model_date":               date_str,
                "k_removed":                k,
                "n_removed_features":       k,
                "fraction_features_removed": round(k / n_feats, 6),
                "removed_features":         removed_str,
                "RGE_star":                 rge_star,
                "n_assets":                 n_assets,
            })

        if date_ok:
            detail_rows.extend(date_detail_rows)
            order_rows.extend(date_order_rows)
            valid_dates += 1
        else:
            skipped_dates += 1

    print(f"  Model {model_name:<8}: valid dates = {valid_dates}, skipped = {skipped_dates}")

detail_df = pd.DataFrame(
    detail_rows,
    columns=["date", "model", "model_date", "k_removed", "n_removed_features",
             "fraction_features_removed", "removed_features", "RGE_star", "n_assets"],
)

order_df = pd.DataFrame(
    order_rows,
    columns=["date", "model", "model_date", "rank", "variable", "RGE", "order_type"],
)

skipped_df = pd.DataFrame(
    skipped_rows,
    columns=["date", "model", "reason", "n_assets", "details", "model_date"],
)

# ── Summary: aggregate by model × k_removed ──────────────────────────────────
summary_rows = []
for (model_name, k), grp in detail_df.groupby(["model", "k_removed"]):
    vals   = grp["RGE_star"].dropna().values
    n      = len(vals)
    frac   = grp["fraction_features_removed"].iloc[0] if n > 0 else np.nan
    summary_rows.append({
        "model":                    model_name,
        "k_removed":                int(k),
        "fraction_features_removed": round(float(frac), 6),
        "mean_RGE_star":            round(float(np.mean(vals)),        6) if n > 0 else np.nan,
        "std_RGE_star":             round(float(np.std(vals, ddof=1)), 6) if n > 1 else np.nan,
        "min_RGE_star":             round(float(np.min(vals)),         6) if n > 0 else np.nan,
        "max_RGE_star":             round(float(np.max(vals)),         6) if n > 0 else np.nan,
        "n_dates":                  n,
        "mean_n_assets":            round(float(grp["n_assets"].mean()), 2) if n > 0 else np.nan,
    })

summary_df = pd.DataFrame(
    summary_rows,
    columns=["model", "k_removed", "fraction_features_removed",
             "mean_RGE_star", "std_RGE_star", "min_RGE_star", "max_RGE_star",
             "n_dates", "mean_n_assets"],
)

# ── Wide vector: one row per model, columns RGE_k00 … RGE_k{n_feats} ─────────
# Determine the maximum k observed (should be n_features for all models)
all_k    = sorted(detail_df["k_removed"].unique()) if not detail_df.empty else []
k_cols   = {k: f"RGE_k{k:02d}" for k in all_k}

pivot = summary_df.pivot(index="model", columns="k_removed", values="mean_RGE_star")
pivot.columns = [k_cols[c] for c in pivot.columns]
pivot = pivot.reset_index()

pivot = pivot.set_index("model").reindex(MODELS).reset_index()
for col in k_cols.values():
    if col not in pivot.columns:
        pivot[col] = np.nan

col_order = ["model"] + [k_cols[k] for k in sorted(k_cols)]
pivot = pivot[col_order]

detail_path  = os.path.join(OUT_DIR, "rge_vector_detail_by_date.csv")
summary_path = os.path.join(OUT_DIR, "rge_vector_summary.csv")
vector_path  = os.path.join(OUT_DIR, "rge_vector_by_model.csv")
order_path   = os.path.join(OUT_DIR, "rge_vector_feature_order_by_date.csv")
skipped_path = os.path.join(OUT_DIR, "rge_vector_skipped_dates.csv")

detail_df.to_csv(detail_path,  index=False)
summary_df.to_csv(summary_path, index=False)
pivot.to_csv(vector_path,       index=False)
order_df.to_csv(order_path,     index=False)
skipped_df.to_csv(skipped_path, index=False)

print()
print(f"Saved: {detail_path}")
print(f"Saved: {summary_path}")
print(f"Saved: {vector_path}")
print(f"Saved: {order_path}")
print(f"Saved: {skipped_path}")

print("\nRGE* compliance vector (mean_RGE_star per k):")
k_col_labels = [k_cols[k] for k in sorted(k_cols)]
header = f"  {'model':<10}" + "".join(f"  {c}" for c in k_col_labels)
print(header)

for _, row in pivot.iterrows():
    line = f"  {row['model']:<10}"
    for col in k_col_labels:
        val = row[col]
        line += f"  {val:.4f}" if pd.notna(val) else "      NaN"
    print(line)

print(f"\n  detail rows : {len(detail_df)}")
print(f"  order rows  : {len(order_df)}")
print(f"  skipped rows: {len(skipped_df)}")

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["font.size"]   = 11

FIG_DIR = "figures/step8/rge"
os.makedirs(FIG_DIR, exist_ok=True)

RANDOM_BASELINE = 0.50

MODEL_DISPLAY = {
    "ridge":   ("Ridge",   "#2166AC"),
    "xgboost": ("XGBoost", "#B2182B"),
    "mlp":     ("MLP",     "#4D9221"),
}

def _model_curve(model_name):
    """Return (x_values, y_values) for model_name from summary_df, sorted by x."""
    sub = (
        summary_df[summary_df["model"] == model_name]
        .sort_values("fraction_features_removed")
        .dropna(subset=["mean_RGE_star"])
    )
    return sub["fraction_features_removed"].tolist(), sub["mean_RGE_star"].tolist()

def _aurge(x_vals, y_vals):
    """Trapezoidal AURGE* from the plotted mean curve."""
    if len(x_vals) < 2:
        return np.nan
    return round(float(np.trapz(y_vals, x_vals)), 4)

def _apply_style(ax, title):
    ax.set_xlim(0.00, 1.00)
    ax.set_ylim(0.45, 1.05)
    ax.set_xlabel("Fraction of Variables Removed")
    ax.set_ylabel("RGE*")
    ax.grid(True, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def _save(fig, base_path):
    fig.tight_layout()
    fig.savefig(base_path + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {base_path}.png")

# ── Individual model figures ──────────────────────────────────────────────────
print()
for model_name, (display_name, color) in MODEL_DISPLAY.items():
    x_vals, y_vals = _model_curve(model_name)

    if not x_vals:
        print(f"  Skipping {model_name} figure: no summary data.")
        continue

    aurge = _aurge(x_vals, y_vals)
    curve_label = f"RGE* Curve (AURGE* = {aurge:.4f})" if not np.isnan(aurge) else "RGE* Curve"

    fig, ax = plt.subplots(figsize=(6, 5), facecolor="white")
    ax.set_facecolor("white")

    ax.plot(x_vals, y_vals, color=color, marker="o", linewidth=1.8,
            markersize=5, label=curve_label, zorder=3)
    ax.axhline(RANDOM_BASELINE, color="red", linestyle="--", linewidth=1.2,
               label=f"Random Baseline (RGE* = {RANDOM_BASELINE:.2f})", zorder=2)

    _apply_style(ax, f"{display_name} RGE* Curve")
    ax.legend(loc="lower left", frameon=True, framealpha=0.8)

    _save(fig, os.path.join(FIG_DIR, f"rge_curve_{model_name}"))

# ── Combined figure: all models ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
ax.set_facecolor("white")

for model_name, (display_name, color) in MODEL_DISPLAY.items():
    x_vals, y_vals = _model_curve(model_name)
    if not x_vals:
        continue
    aurge = _aurge(x_vals, y_vals)
    label = (f"{display_name} (AURGE* = {aurge:.4f})"
             if not np.isnan(aurge) else display_name)
    ax.plot(x_vals, y_vals, color=color, marker="o", linewidth=1.8,
            markersize=5, label=label, zorder=3)

ax.axhline(RANDOM_BASELINE, color="red", linestyle="--", linewidth=1.2,
           label=f"Random Baseline (RGE* = {RANDOM_BASELINE:.2f})", zorder=2)

_apply_style(ax, "RGE* Curve by Model")
ax.legend(loc="lower left", frameon=True, framealpha=0.8)

_save(fig, os.path.join(FIG_DIR, "rge_curve_by_model"))
