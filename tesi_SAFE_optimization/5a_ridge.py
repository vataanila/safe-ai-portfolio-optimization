"""
Ridge regression mu_hat estimation (Step 5a).

Expanding-window OOS training over the rebalancing dates from
baseline_weights.csv: at each date, alpha is grid-searched over
[0.01, 0.1, 1.0, 10.0, 100.0] on a validation split, then refit on
train+val. Output is a cross-sectional return ranking, rescaled to the
baseline historical-mean mu distribution so it plugs into the same MIQP
as step3. No portfolio optimization happens here, just mu_hat.

Reads: data/results/step4/ml_panel.csv, data/clean/returns.csv,
       data/results/step3/baseline_weights.csv
Writes: data/results/step5/ridge/predictions/ml_mu_ridge.csv
        + per-date model/scaler pickles and IC diagnostics under
          data/results/step5/ridge/

Author: Anila Vata
"""

import json
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from config import BASE_DIR, CLEAN_DIR, ESTIM_WINDOW, RANDOM_STATE, TRADING_DAYS

warnings.filterwarnings("ignore")

N_VAL_DATES  = 6

RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")
STEP3_DIR   = os.path.join(RESULTS_DIR, "step3")
STEP4_DIR   = os.path.join(RESULTS_DIR, "step4")
STEP5_DIR   = os.path.join(RESULTS_DIR, "step5", "ridge")
PRED_DIR    = os.path.join(STEP5_DIR, "predictions")
DIAG_DIR    = os.path.join(STEP5_DIR, "diagnostics")
MODEL_DIR   = os.path.join(STEP5_DIR, "models")

def load_inputs():
    """Load ml_panel, returns, and baseline_weights; validate ticker alignment."""
    print("\n[1] Loading inputs ...")

    panel = pd.read_csv(
        os.path.join(STEP4_DIR, "ml_panel.csv"),
        parse_dates=["date", "target_end_date"],
    )

    returns = pd.read_csv(
        os.path.join(CLEAN_DIR, "returns.csv"),
        index_col=0, parse_dates=True,
    ).sort_index()

    baseline_weights = pd.read_csv(
        os.path.join(STEP3_DIR, "baseline_weights.csv"),
        index_col=0, parse_dates=True,
    )

    oos_dates = baseline_weights.index
    tickers   = baseline_weights.columns.astype(str).tolist()

    panel_dates = pd.DatetimeIndex(panel["date"].drop_duplicates()).sort_values()
    missing_oos_dates = [d for d in oos_dates if d not in panel_dates]
    if missing_oos_dates:
        raise ValueError(
            f"{len(missing_oos_dates)} OOS rebalancing dates from baseline_weights.csv "
            f"are missing from ml_panel.csv: "
            f"{[d.date().isoformat() for d in missing_oos_dates[:5]]}"
            f"{'...' if len(missing_oos_dates) > 5 else ''}"
        )

    missing = [t for t in tickers if t not in returns.columns]
    if missing:
        raise ValueError(
            f"returns.csv is missing {len(missing)} tickers required by "
            f"baseline_weights.csv: {missing[:5]}"
            f"{'...' if len(missing) > 5 else ''}"
        )
    returns = returns[tickers]

    print(f"  panel    shape : {panel.shape}")
    print(f"           dates : {panel['date'].min().date()} to {panel['date'].max().date()}")
    print(f"  returns  shape : {returns.shape}")
    print(f"           range : {returns.index[0].date()} to {returns.index[-1].date()}")
    print(f"  OOS dates      : {len(oos_dates)}")
    print(f"           range : {oos_dates[0].date()} to {oos_dates[-1].date()}")

    non_feature_cols = {"date", "ticker", "target_end_date", "target_raw", "target_rank"}
    feature_cols = [c for c in panel.columns if c not in non_feature_cols]
    print(f"  Features ({len(feature_cols)}): {feature_cols}")

    if not feature_cols:
        raise ValueError("No feature columns found in ml_panel.csv.")

    for c in feature_cols:
        if not pd.api.types.is_numeric_dtype(panel[c]):
            raise ValueError(f"Feature column '{c}' is not numeric.")

    return panel, returns, baseline_weights, oos_dates, tickers, feature_cols

def compute_baseline_mu(rebal_date: pd.Timestamp,
                        returns: pd.DataFrame) -> pd.Series:
    """
    Trailing 252-day annualised mean, winsorized cross-sectionally at p1/p99.
    Uses only data strictly before rebal_date.
    """
    hist = returns.loc[returns.index < rebal_date].iloc[-ESTIM_WINDOW:]
    raw  = hist.mean() * TRADING_DAYS
    lo   = np.nanpercentile(raw.values, 1)
    hi   = np.nanpercentile(raw.values, 99)
    return raw.clip(lo, hi)

def scores_to_mu_hat(scores: pd.Series,
                     baseline_mu: pd.Series) -> pd.Series:
    """
    Convert ML ranked scores to annualized mu_hat matching the
    cross-sectional scale of baseline_mu.

    Raises ValueError if scores are constant.
    """
    common     = scores.index.intersection(baseline_mu.index)
    s          = scores[common].dropna()
    bmu        = baseline_mu[common].dropna()
    both       = s.index.intersection(bmu.index)
    s, bmu     = s[both], bmu[both]

    s_std = s.std(ddof=0)
    if s_std < 1e-10:
        raise ValueError(
            "Constant prediction scores: cannot rescale to mu_hat. "
            "All predicted scores are identical, check the model or training data."
        )

    s_scaled = (s - s.mean()) / s_std
    mu_hat   = s_scaled * bmu.std(ddof=0) + bmu.mean()

    lo = np.nanpercentile(mu_hat.values, 1)
    hi = np.nanpercentile(mu_hat.values, 99)
    return mu_hat.clip(lo, hi)

def tune_ridge(X_tr, y_tr, X_vl, y_vl):
    """
    Grid-search alpha for Ridge on the validation set (R²).
    Refit on train+val with the best alpha.  Returns (model, scaler, alpha).
    """
    best_alpha, best_r2 = None, -np.inf
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        sc    = StandardScaler().fit(X_tr)
        preds = Ridge(alpha=alpha, random_state=RANDOM_STATE).fit(
            sc.transform(X_tr), y_tr
        ).predict(sc.transform(X_vl))
        ss_res = np.sum((y_vl - preds) ** 2)
        ss_tot = np.sum((y_vl - y_vl.mean()) ** 2)
        r2     = 1 - ss_res / ss_tot if ss_tot > 0 else -np.inf
        if r2 > best_r2:
            best_r2, best_alpha = r2, alpha

    X_tv = np.vstack([X_tr, X_vl])
    y_tv = np.concatenate([y_tr, y_vl])
    sc   = StandardScaler().fit(X_tv)
    mdl  = Ridge(alpha=best_alpha, random_state=RANDOM_STATE).fit(
        sc.transform(X_tv), y_tv
    )
    return mdl, sc, best_alpha, best_r2

def run_oos(panel, oos_dates, returns, tickers, feature_cols):
    """
    Expanding-window OOS loop for Ridge Regression.

    Split logic (no leakage):
      - eligible obs : target_end_date <= t
      - val_dates    : last N_VAL_DATES in eligible
      - train_dates  : all eligible dates before val_dates
      - test         : panel rows with date == t
    """
    print(f"\n[2] OOS expanding-window loop (Ridge) over {len(oos_dates)} dates ...")

    mu_hat_dict  = {}
    diag_rows    = []
    hyper_rows   = []

    for i, t in enumerate(oos_dates):
        t_str = t.date().isoformat()
        print(f"\n  [{i+1:02d}/{len(oos_dates)}] {t_str}")

        baseline_mu = compute_baseline_mu(t, returns)

        eligible = panel[
            panel["target_end_date"].notna()
            & (panel["target_end_date"] <= t)
            & panel["target_rank"].notna()
        ]
        elig_dates = pd.DatetimeIndex(eligible["date"].drop_duplicates()).sort_values()

        if len(elig_dates) < N_VAL_DATES + 1:
            print(f"    SKIP: only {len(elig_dates)} eligible dates "
                  f"(need {N_VAL_DATES + 1})")
            continue

        val_dates   = elig_dates[-N_VAL_DATES:]
        train_dates = elig_dates[:-N_VAL_DATES]

        train = eligible[eligible["date"].isin(train_dates)]
        val   = eligible[eligible["date"].isin(val_dates)]
        test  = panel[panel["date"] == t].dropna(subset=feature_cols)

        if train.empty or val.empty or test.empty:
            print(f"    SKIP: empty split "
                  f"(train={len(train)}, val={len(val)}, test={len(test)})")
            continue

        X_tr = train[feature_cols].values.astype(float)
        y_tr = train["target_rank"].values.astype(float)
        X_vl = val[feature_cols].values.astype(float)
        y_vl = val["target_rank"].values.astype(float)
        X_te = test[feature_cols].values.astype(float)
        te_tickers = test["ticker"].values

        print(f"    train  : {len(train):5d} obs  ({len(train_dates)} dates, "
              f"up to {max(train_dates).date()})")
        print(f"    val    : {len(val):5d} obs  ({len(val_dates)} dates, "
              f"{min(val_dates).date()} to {max(val_dates).date()})")
        print(f"    test   : {len(test):5d} stocks")

        t0 = time.perf_counter()
        mdl, sc, alpha, val_r2 = tune_ridge(X_tr, y_tr, X_vl, y_vl)
        preds  = mdl.predict(sc.transform(X_te))
        scores = pd.Series(preds, index=te_tickers)
        mu_hat = scores_to_mu_hat(scores, baseline_mu)
        mu_hat_dict[t] = mu_hat

        model_path  = os.path.join(MODEL_DIR, f"ridge_model_{t_str}.pkl")
        scaler_path = os.path.join(MODEL_DIR, f"ridge_scaler_{t_str}.pkl")
        bundle_path = os.path.join(MODEL_DIR, f"ridge_bundle_{t_str}.pkl")

        joblib.dump(mdl, model_path)
        joblib.dump(sc,  scaler_path)
        # The scaler is saved because Step 7 SAFE Robustness / RGR must evaluate
        # the actual trained Ridge prediction function. Re-fitting a scaler on the
        # test slice would not reproduce the trained model.
        joblib.dump(
            {
                "model":        mdl,
                "scaler":       sc,
                "feature_cols": feature_cols,
                "alpha":        alpha,
                "date":         t_str,
            },
            bundle_path,
        )

        print(f"    saved model : {model_path}")
        print(f"    saved scaler: {scaler_path}")
        print(f"    saved bundle: {bundle_path}")

        print(f"    Ridge   alpha={alpha:<6}  val_r2={val_r2:.4f}  "
              f"t={time.perf_counter()-t0:.1f}s  "
              f"pred_std={scores.std():.4f}")

        hyper_rows.append({
            "date":        t_str,
            "alpha":       alpha,
            "val_r2":      round(val_r2, 6) if val_r2 is not None and not np.isnan(val_r2) else np.nan,
            "model_path":  model_path,
            "scaler_path": scaler_path,
            "bundle_path": bundle_path,
        })

        # Per-date diagnostics
        test_indexed = test.set_index("ticker")
        t_rank = test_indexed["target_rank"]
        common = scores.index.intersection(t_rank.dropna().index)
        if len(common) >= 10:
            s_c  = scores[common]
            tr_c = t_rank[common]
            sp, _ = spearmanr(s_c.values, tr_c.values)
            pe, _ = pearsonr(s_c.values,  tr_c.values)
            var_tr  = float(np.var(tr_c.values))
            r2_rank = (
                1 - float(np.mean((tr_c.values - s_c.values) ** 2)) / var_tr
                if var_tr > 0 else np.nan
            )
            diag_rows.append({
                "date"       : t_str,
                "year"       : t.year,
                "spearman_ic": round(sp,      4),
                "pearson_ic" : round(pe,      4),
                "r2_rank"    : round(r2_rank, 4) if not np.isnan(r2_rank) else np.nan,
                "n_stocks"   : len(common),
            })

    return mu_hat_dict, diag_rows, hyper_rows

def assemble_mu_matrix(mu_hat_dict, oos_dates, tickers):
    """Convert {date -> pd.Series} to (n_dates × n_tickers) DataFrame."""
    rows = []
    for t in oos_dates:
        if t in mu_hat_dict:
            s = mu_hat_dict[t].reindex(tickers)
        else:
            s = pd.Series(np.nan, index=tickers)
        s.name = t
        rows.append(s)
    df = pd.DataFrame(rows)
    df.index.name = "date"
    return df

def fill_nan_with_baseline(mu_mat, returns, tickers):
    """
    For any ticker/date cell still NaN, substitute the baseline mu.
    Raises ValueError if more than 1% of cells require filling.
    """
    total_cells = mu_mat.size
    nan_count   = int(mu_mat.isna().sum().sum())
    nan_pct     = nan_count / total_cells * 100

    if nan_pct > 1.0:
        raise ValueError(
            f"Too many NaN cells in mu_hat before baseline fill: "
            f"{nan_count}/{total_cells} ({nan_pct:.2f}% > 1% threshold). "
            "Check feature computation for systematic missing data."
        )

    print(f"  WARNING: filling {nan_count} NaN cells ({nan_pct:.3f}% of matrix) "
          f"with baseline mu.")

    for t in mu_mat.index:
        nan_cols = mu_mat.columns[mu_mat.loc[t].isna()].tolist()
        if not nan_cols:
            continue
        bmu = compute_baseline_mu(t, returns)
        mu_mat.loc[t, nan_cols] = bmu.reindex(nan_cols).values
    return mu_mat

def save_diagnostics(diag_rows, hyper_rows, mu_mat, oos_dates, returns):
    """Compute and save all Ridge diagnostics to DIAG_DIR."""
    os.makedirs(DIAG_DIR, exist_ok=True)
    print(f"\n[4] Saving diagnostics to {DIAG_DIR} ...")

    diag_df = pd.DataFrame(diag_rows)

    if diag_df.empty:
        raise ValueError(
            "No diagnostic rows were produced. "
            "Check the expanding-window split and ml_panel.csv."
        )

    # -- 1. IC detail ----------------------------------------------------------
    diag_df.to_csv(os.path.join(DIAG_DIR, "ridge_ic_detail.csv"), index=False)
    print("  Saved: ridge_ic_detail.csv")

    # -- 2. IC summary by year -------------------------------------------------
    ic_rows = []
    for year in sorted(diag_df["year"].unique()):
        ydf = diag_df[diag_df["year"] == year]
        ic_rows.append({
            "year"         : year,
            "mean_IC"      : round(ydf["spearman_ic"].mean(), 4),
            "hit_rate"     : round((ydf["spearman_ic"] > 0).mean(), 4),
            "mean_pearson" : round(ydf["pearson_ic"].mean(), 4),
            "mean_r2_rank" : round(ydf["r2_rank"].mean(), 4)
                             if ydf["r2_rank"].notna().any() else np.nan,
            "n_dates"      : len(ydf),
        })
    ic_rows.append({
        "year"         : "all",
        "mean_IC"      : round(diag_df["spearman_ic"].mean(), 4),
        "hit_rate"     : round((diag_df["spearman_ic"] > 0).mean(), 4),
        "mean_pearson" : round(diag_df["pearson_ic"].mean(), 4),
        "mean_r2_rank" : round(diag_df["r2_rank"].mean(), 4)
                         if diag_df["r2_rank"].notna().any() else np.nan,
        "n_dates"      : len(diag_df),
    })
    ic_df = pd.DataFrame(ic_rows)
    ic_df.to_csv(os.path.join(DIAG_DIR, "ridge_ic_summary.csv"), index=False)
    print("  Saved: ridge_ic_summary.csv")
    print(ic_df.to_string(index=False))

    # -- 3. Mu-baseline correlation --------------------------------------------
    corr_rows = []
    for t in oos_dates:
        if t not in mu_mat.index:
            continue
        bmu    = compute_baseline_mu(t, returns)
        ml_row = mu_mat.loc[t].dropna()
        common = ml_row.index.intersection(bmu.dropna().index)
        if len(common) < 10:
            continue
        c = float(ml_row[common].corr(bmu[common]))
        corr_rows.append({
            "date"               : t.date().isoformat(),
            "corr_with_baseline" : round(c, 4),
        })
    pd.DataFrame(corr_rows).to_csv(
        os.path.join(DIAG_DIR, "ridge_mu_baseline_correlation.csv"), index=False
    )
    print(f"  Saved: ridge_mu_baseline_correlation.csv ({len(corr_rows)} rows)")

    # -- 4. Prediction summary -------------------------------------------------
    summ_rows = []
    for t in oos_dates:
        if t not in mu_mat.index:
            continue
        s = mu_mat.loc[t].dropna()
        if s.empty:
            continue
        summ_rows.append({
            "date": t.date().isoformat(),
            "mean": round(float(s.mean()),         6),
            "std" : round(float(s.std()),          6),
            "min" : round(float(s.min()),          6),
            "max" : round(float(s.max()),          6),
            "p1"  : round(float(s.quantile(0.01)), 6),
            "p99" : round(float(s.quantile(0.99)), 6),
        })
    pd.DataFrame(summ_rows).to_csv(
        os.path.join(DIAG_DIR, "ridge_prediction_summary.csv"), index=False
    )
    print(f"  Saved: ridge_prediction_summary.csv ({len(summ_rows)} rows)")

    # -- 5. Hyperparameters (alpha per date) -----------------------------------
    pd.DataFrame(hyper_rows).to_csv(
        os.path.join(DIAG_DIR, "ridge_hyperparameters.csv"), index=False
    )
    print(f"  Saved: ridge_hyperparameters.csv ({len(hyper_rows)} rows)")

def quality_checks(mu_mat, oos_dates, tickers, baseline_weights):
    """Verify shape, ticker alignment, date coverage, no NaN, no constant rows."""
    print("\n[5] Quality checks ...")

    expected_shape   = (len(oos_dates), len(tickers))
    expected_tickers = list(baseline_weights.columns)
    expected_dates   = pd.DatetimeIndex(baseline_weights.index)

    print(f"  shape          : {mu_mat.shape}  expected {expected_shape}")

    if list(mu_mat.columns) != expected_tickers:
        raise ValueError(
            "Column tickers do not match baseline_weights.columns. "
            f"Expected {len(expected_tickers)}, got {len(mu_mat.columns)}."
        )
    print("  Column tickers match baseline_weights.columns: OK")

    if not expected_dates.equals(pd.DatetimeIndex(mu_mat.index)):
        raise ValueError(
            "Row dates do not match baseline_weights.index.\n"
            f"  Missing : {sorted(set(expected_dates) - set(mu_mat.index))}\n"
            f"  Extra   : {sorted(set(mu_mat.index) - set(expected_dates))}"
        )
    print(f"  Row dates match baseline_weights.index ({len(expected_dates)} dates): OK")

    nan_n = int(mu_mat.isna().sum().sum())
    if nan_n > 0:
        raise ValueError(f"mu_hat contains {nan_n} NaN values!")
    print("  NaN count      : 0  OK")

    const_d = mu_mat.index[mu_mat.std(axis=1) < 1e-10].tolist()
    if const_d:
        raise ValueError(f"Constant predictions on: {[d.date() for d in const_d]}")
    print("  Constant dates : 0  OK")

    if mu_mat.shape != baseline_weights.shape:
        raise ValueError(
            f"Shape mismatch: mu_mat={mu_mat.shape}, "
            f"baseline_weights={baseline_weights.shape}"
        )
    print(f"  Shape matches baseline_weights.csv {baseline_weights.shape}: OK")

    global_min = float(mu_mat.min().min())
    global_max = float(mu_mat.max().max())
    print(f"  mu_hat range   : [{global_min:.4f}, {global_max:.4f}]")
    print("\n  All quality checks passed.")

def main():
    for d in [STEP5_DIR, PRED_DIR, DIAG_DIR, MODEL_DIR]:
        os.makedirs(d, exist_ok=True)

    print("=" * 70)
    print("  STEP 5A -- RIDGE REGRESSION EXPECTED RETURN ESTIMATION (MU_HAT)")
    print("=" * 70)

    panel, returns, baseline_weights, oos_dates, tickers, feature_cols = load_inputs()

    feature_cols_path = os.path.join(MODEL_DIR, "ridge_feature_cols.json")
    with open(feature_cols_path, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"  Saved feature columns: {feature_cols_path}")

    mu_hat_dict, diag_rows, hyper_rows = run_oos(
        panel, oos_dates, returns, tickers, feature_cols
    )

    manifest_path = os.path.join(MODEL_DIR, "ridge_saved_models_manifest.csv")
    pd.DataFrame(hyper_rows).to_csv(manifest_path, index=False)
    print(f"  Saved model manifest: {manifest_path}")

    print("\n[3] Assembling and saving mu_hat matrix ...")
    mu_mat = assemble_mu_matrix(mu_hat_dict, oos_dates, tickers)

    nan_count = int(mu_mat.isna().sum().sum())
    if nan_count > 0:
        mu_mat = fill_nan_with_baseline(mu_mat, returns, tickers)

    out_path = os.path.join(PRED_DIR, "ml_mu_ridge.csv")
    mu_mat.to_csv(out_path)
    print(f"  Saved: ml_mu_ridge.csv  shape={mu_mat.shape}  "
          f"NaN remaining={int(mu_mat.isna().sum().sum())}")

    save_diagnostics(diag_rows, hyper_rows, mu_mat, oos_dates, returns)

    quality_checks(mu_mat, oos_dates, tickers, baseline_weights)

    print("\n" + "=" * 70)
    print("  STEP 5A, COMPLETE")
    print("=" * 70)
    print(f"""
  Outputs saved to: {PRED_DIR}

  mu_hat matrix ({len(oos_dates)} dates × {len(tickers)} tickers):
    ml_mu_ridge.csv

  Models (data/results/step5/ridge/models/):
    ridge_model_YYYY-MM-DD.pkl
    ridge_scaler_YYYY-MM-DD.pkl   (training-time scaler for Step 7 RGR)
    ridge_bundle_YYYY-MM-DD.pkl   (dict: model + scaler + feature_cols + alpha + date)
    ridge_feature_cols.json
    ridge_saved_models_manifest.csv

  Diagnostics ({DIAG_DIR}):
    ridge_ic_detail.csv
    ridge_ic_summary.csv
    ridge_mu_baseline_correlation.csv
    ridge_prediction_summary.csv
    ridge_hyperparameters.csv

  Next step: run 5b_xgboost.py or 5c_mlp.py.
""")

if __name__ == "__main__":
    main()
