"""
SAFE Robustness is computed using Rank Graduation Robustness from the official safeaipackage.
For each rebalancing date, the original model predictions are compared with predictions obtained
after lower/upper tail-swapping perturbations of selected input features.
The official safeaipackage computes RGR as core.rga(original_yhat, perturbed_yhat).
The model wrapper must reproduce the actual trained prediction function, including
training-time preprocessing.

Reference: Babaei and Giudici, "A statistical package for safe artificial intelligence".
"""

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from safeaipackage import check_robustness
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

OUT_DIR    = "data/results/step7"
PANEL_PATH = os.path.join(OUT_DIR, "test_panel.csv")

MODEL_DIR = {
    "ridge":   "data/results/step5/ridge/models",
    "xgboost": "data/results/step5/xgboost/models",
    "mlp":     "data/results/step5/mlp/models",
}

MODELS = ["ridge", "xgboost", "mlp"]
MIN_ASSETS = 10
PERTURBATION_PERCENTAGE = 0.05

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

def load_feature_cols(model_name):
    """Load the authoritative feature list saved by Step 5 for this model."""
    path = os.path.join(MODEL_DIR[model_name], f"{model_name}_feature_cols.json")
    with open(path) as f:
        return json.load(f)

def build_model_date_map(model_name):
    """Scan the model directory for {model_name}_model_YYYY-MM-DD.pkl files and
    return a dict mapping each monthly Period to the date string in the filename.
    Raises ValueError if two model files fall in the same monthly period."""
    import glob as _glob
    import re
    pattern = os.path.join(MODEL_DIR[model_name], f"{model_name}_model_*.pkl")
    files   = _glob.glob(pattern)
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

class _MissingTrainingScaler(Exception):
    """Raised when Ridge has no training-time scaler available."""

def load_proxy(model_name, date_str, feat_cols):
    """
    Load the trained model for the given model and date and return a wrapper
    whose .predict(X) reproduces the actual trained prediction function,
    including training-time preprocessing.

    Ridge: 5a_ridge.py uses a StandardScaler during training.
    The training-time scaler may be stored as ridge_scaler_YYYY-MM-DD.pkl,
    embedded inside ridge_model_YYYY-MM-DD.pkl (dict bundle), inside a
    sklearn Pipeline, or inside ridge_bundle_YYYY-MM-DD.pkl.
    All four locations are tried in order. No scaler is ever fitted in Step 7.
    If no valid model+scaler combination is found, Ridge is skipped for that date.
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
                # dict without scaler, fall through to bundle

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
        # Dict bundle -- extract model (and scaler if present)
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

detail_rows       = []
group_detail_rows = []
skipped_rows      = []

print("\nSAFE Robustness / RGR")
print("-" * 50)

for model_name in MODELS:
    # Load and validate feature columns for this model
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
        if date_str is None:
            skipped_rows.append({
                "date": pd.Timestamp(d).date().isoformat(), "model": model_name,
                "reason": "missing_model_file", "n_assets": 0, "details": f"no model file for period {period}",
            })
            skipped_dates += 1
            continue

        sub = test[test["date"] == d].copy()
        sub = sub.dropna(subset=feat_cols)
        n_assets = len(sub)

        if n_assets < MIN_ASSETS:
            skipped_rows.append({
                "date": date_str, "model": model_name,
                "reason": "too_few_assets", "n_assets": n_assets, "details": ""
            })
            skipped_dates += 1
            continue

        xtest_d = sub[feat_cols].reset_index(drop=True)

        # Load the actual trained model wrapper
        try:
            proxy = load_proxy(model_name, date_str, feat_cols)
        except _MissingTrainingScaler as exc:
            skipped_rows.append({
                "date": date_str, "model": model_name,
                "reason": "missing_training_scaler", "n_assets": n_assets, "details": str(exc)
            })
            skipped_dates += 1
            continue
        except FileNotFoundError as exc:
            reason = "missing_scaler" if "scaler" in str(exc).lower() else "missing_model_file"
            skipped_rows.append({
                "date": date_str, "model": model_name,
                "reason": reason, "n_assets": n_assets, "details": str(exc)
            })
            skipped_dates += 1
            continue
        except Exception as exc:
            skipped_rows.append({
                "date": date_str, "model": model_name,
                "reason": "model_load_error", "n_assets": n_assets, "details": str(exc)
            })
            skipped_dates += 1
            continue

        # Original predictions - same model object used inside the official package
        try:
            yhat = proxy.predict(xtest_d).tolist()
        except Exception as exc:
            skipped_rows.append({
                "date": date_str, "model": model_name,
                "reason": "prediction_error", "n_assets": n_assets, "details": str(exc)
            })
            skipped_dates += 1
            continue

        if np.std(yhat) < 1e-12:
            skipped_rows.append({
                "date": date_str, "model": model_name,
                "reason": "constant_yhat", "n_assets": n_assets, "details": ""
            })
            skipped_dates += 1
            continue

        # Per-variable RGR
        try:
            rgr_df = check_robustness.compute_rgr_values(
                xtest_d, yhat, proxy, feat_cols,
                perturbation_percentage=PERTURBATION_PERCENTAGE,
                group=False,
            )
        except Exception as exc:
            skipped_rows.append({
                "date": date_str, "model": model_name,
                "reason": "rgr_error", "n_assets": n_assets, "details": str(exc)
            })
            skipped_dates += 1
            continue

        for var, row in rgr_df.iterrows():
            detail_rows.append({
                "date":                    date_str,
                "model":                   model_name,
                "variable":                var,
                "RGR":                     round(float(row["RGR"]), 6),
                "n_assets":                n_assets,
                "perturbation_percentage": PERTURBATION_PERCENTAGE,
            })

        # Group RGR (all features perturbed together)
        try:
            rgr_group_df = check_robustness.compute_rgr_values(
                xtest_d, yhat, proxy, feat_cols,
                perturbation_percentage=PERTURBATION_PERCENTAGE,
                group=True,
            )
        except Exception as exc:
            # per-variable succeeded; group failure is logged but does not invalidate the date
            skipped_rows.append({
                "date": date_str, "model": model_name,
                "reason": "rgr_error", "n_assets": n_assets,
                "details": f"group RGR failed: {exc}",
            })
            rgr_group_df = None

        if rgr_group_df is not None:
            # group=True returns a 1-row DataFrame; index is str(variables) -- normalise label
            group_detail_rows.append({
                "date":                    date_str,
                "model":                   model_name,
                "group":                   "all_features",
                "RGR":                     round(float(rgr_group_df["RGR"].iloc[0]), 6),
                "n_assets":                n_assets,
                "perturbation_percentage": PERTURBATION_PERCENTAGE,
            })

        valid_dates += 1

    print(f"Model {model_name:<8}: valid dates = {valid_dates}, skipped = {skipped_dates}")

detail_df = pd.DataFrame(
    detail_rows,
    columns=["date", "model", "variable", "RGR", "n_assets", "perturbation_percentage"],
)
group_detail_df = pd.DataFrame(
    group_detail_rows,
    columns=["date", "model", "group", "RGR", "n_assets", "perturbation_percentage"],
)
skipped_df = pd.DataFrame(
    skipped_rows,
    columns=["date", "model", "reason", "n_assets", "details"],
)

def build_summary(df, key_col):
    """Aggregate detail DataFrame by (model, key_col) → summary statistics."""
    col_order = ["model", key_col, "mean_RGR", "std_RGR",
                 "min_RGR", "max_RGR", "n_dates", "mean_n_assets"]
    if df.empty:
        return pd.DataFrame(columns=col_order)
    rows = []
    for (model, key), grp in df.groupby(["model", key_col]):
        vals = grp["RGR"].dropna().values
        n    = len(vals)
        rows.append({
            "model":         model,
            key_col:         key,
            "mean_RGR":      round(float(np.mean(vals)),        6) if n > 0 else np.nan,
            "std_RGR":       round(float(np.std(vals, ddof=1)), 6) if n > 1 else np.nan,
            "min_RGR":       round(float(np.min(vals)),         6) if n > 0 else np.nan,
            "max_RGR":       round(float(np.max(vals)),         6) if n > 0 else np.nan,
            "n_dates":       n,
            "mean_n_assets": round(float(grp["n_assets"].mean()), 2) if n > 0 else np.nan,
        })
    return pd.DataFrame(rows, columns=col_order)

summary_df       = build_summary(detail_df,       "variable")
group_summary_df = build_summary(group_detail_df, "group")

detail_path        = os.path.join(OUT_DIR, "rgr_detail_by_date.csv")
group_detail_path  = os.path.join(OUT_DIR, "rgr_group_detail_by_date.csv")
summary_path       = os.path.join(OUT_DIR, "rgr_summary.csv")
group_summary_path = os.path.join(OUT_DIR, "rgr_group_summary.csv")
skipped_path       = os.path.join(OUT_DIR, "rgr_skipped_dates.csv")

detail_df.to_csv(detail_path,               index=False)
group_detail_df.to_csv(group_detail_path,   index=False)
summary_df.to_csv(summary_path,             index=False)
group_summary_df.to_csv(group_summary_path, index=False)
skipped_df.to_csv(skipped_path,             index=False)

print(f"Saved: {detail_path}")
print(f"Saved: {group_detail_path}")
print(f"Saved: {summary_path}")
print(f"Saved: {group_summary_path}")
print(f"Saved: {skipped_path}")
