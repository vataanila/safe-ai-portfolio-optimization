"""
step5b_train_xgboost.py
=================
PURPOSE : XGBoost-based expected return estimation (mu_hat).

          Trains XGBRegressor in an expanding-window OOS framework over the
          rebalancing dates defined by baseline_weights.csv.

          Predicts cross-sectional return rankings; predictions are then
          scaled to the same distribution as the baseline historical-mean mu.
          No portfolio optimization is performed here.

INPUTS  :
  data/results/step4/ml_panel.csv
  data/clean/returns.csv
  data/results/step3/baseline_weights.csv

OUTPUTS :
  data/results/step5/ml_mu_xgboost.csv
  data/results/step5/diagnostics/xgboost/
    xgboost_ic_detail.csv
    xgboost_ic_summary.csv
    xgboost_mu_baseline_correlation.csv
    xgboost_prediction_summary.csv
    xgboost_feature_importance.csv
    xgboost_hyperparameters.csv

Author  : Anila Vata
"""

# =============================================================================
# 0. IMPORTS AND CONSTANTS
# =============================================================================
import os
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import xgboost as xgb

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TRADING_DAYS = 252
ESTIM_WINDOW = 252
N_VAL_DATES  = 6

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DIR   = os.path.join(BASE_DIR, "data", "clean")
RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")
STEP3_DIR   = os.path.join(RESULTS_DIR, "step3")
STEP4_DIR   = os.path.join(RESULTS_DIR, "step4")
STEP5_DIR   = os.path.join(RESULTS_DIR, "step5")
DIAG_DIR    = os.path.join(STEP5_DIR, "diagnostics", "xgboost")


# =============================================================================
# 1. LOAD INPUTS
# =============================================================================

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


# =============================================================================
# 2. BASELINE MU (identical to Step 3)
# =============================================================================

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


# =============================================================================
# 3. SCORE -> MU_HAT
# =============================================================================

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
            "All predicted scores are identical - check the model or training data."
        )

    s_scaled = (s - s.mean()) / s_std
    mu_hat   = s_scaled * bmu.std(ddof=0) + bmu.mean()

    lo = np.nanpercentile(mu_hat.values, 1)
    hi = np.nanpercentile(mu_hat.values, 99)
    return mu_hat.clip(lo, hi)


# =============================================================================
# 4. XGBOOST TRAINING
# =============================================================================

def tune_xgb(X_tr, y_tr, X_vl, y_vl):
    """
    Grid-search n_estimators, max_depth, learning_rate for XGBoost,
    using early stopping on the validation set (MSE).
    Refit on train+val with best params.  Returns (model, best_params).
    """
    best_params, best_mse = None, np.inf
    best_n_est = None

    for n_est in [100, 300, 500]:
        for depth in [3, 4, 6]:
            for lr in [0.01, 0.05]:
                mdl = xgb.XGBRegressor(
                    n_estimators          = n_est,
                    max_depth             = depth,
                    learning_rate         = lr,
                    subsample             = 0.8,
                    colsample_bytree      = 0.8,
                    objective             = "reg:squarederror",
                    early_stopping_rounds = 20,
                    random_state          = RANDOM_STATE,
                    verbosity             = 0,
                    n_jobs                = 1,
                )
                mdl.fit(X_tr, y_tr,
                        eval_set=[(X_vl, y_vl)],
                        verbose=False)
                preds    = mdl.predict(X_vl)
                mse      = float(np.mean((preds - y_vl) ** 2))
                actual_n = mdl.best_iteration + 1 if hasattr(mdl, "best_iteration") else n_est
                if mse < best_mse:
                    best_mse    = mse
                    best_params = {"n_estimators": n_est,
                                   "max_depth":    depth,
                                   "learning_rate": lr}
                    best_n_est  = actual_n

    X_tv = np.vstack([X_tr, X_vl])
    y_tv = np.concatenate([y_tr, y_vl])
    final_params = dict(best_params)
    final_params["n_estimators"] = best_n_est or best_params["n_estimators"]
    mdl = xgb.XGBRegressor(
        **final_params,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        objective        = "reg:squarederror",
        random_state     = RANDOM_STATE,
        verbosity        = 0,
        n_jobs           = 1,
    )
    mdl.fit(X_tv, y_tv)
    return mdl, final_params


# =============================================================================
# 5. EXPANDING-WINDOW OOS LOOP
# =============================================================================

def run_oos(panel, oos_dates, returns, tickers, feature_cols):
    """
    Expanding-window OOS loop for XGBoost.

    Split logic (no leakage):
      - eligible obs : target_end_date <= t
      - val_dates    : last N_VAL_DATES in eligible
      - train_dates  : all eligible dates before val_dates
      - test         : panel rows with date == t
    """
    print(f"\n[2] OOS expanding-window loop (XGBoost) over {len(oos_dates)} dates ...")

    mu_hat_dict  = {}
    diag_rows    = []
    hyper_rows   = []
    fi_list      = []

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
        mdl, params = tune_xgb(X_tr, y_tr, X_vl, y_vl)
        preds  = mdl.predict(X_te)
        scores = pd.Series(preds, index=te_tickers)
        mu_hat = scores_to_mu_hat(scores, baseline_mu)
        mu_hat_dict[t] = mu_hat

        fi = pd.Series(mdl.feature_importances_, index=feature_cols)
        fi_list.append(fi)

        print(f"    XGBoost n={params['n_estimators']:<4} "
              f"d={params['max_depth']} "
              f"lr={params['learning_rate']}  "
              f"t={time.perf_counter()-t0:.1f}s  "
              f"pred_std={scores.std():.4f}")

        hyper_rows.append({
            "date"          : t_str,
            "n_estimators"  : params["n_estimators"],
            "max_depth"     : params["max_depth"],
            "learning_rate" : params["learning_rate"],
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

    return mu_hat_dict, diag_rows, hyper_rows, fi_list


# =============================================================================
# 6. ASSEMBLE MU_HAT MATRIX
# =============================================================================

def assemble_mu_matrix(mu_hat_dict, oos_dates, tickers):
    """Convert {date -> pd.Series} to (n_dates x n_tickers) DataFrame."""
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


# =============================================================================
# 7. FILL REMAINING NaN WITH BASELINE MU
# =============================================================================

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


# =============================================================================
# 8. DIAGNOSTICS
# =============================================================================

def save_diagnostics(diag_rows, hyper_rows, fi_list, mu_mat, oos_dates, returns, feature_cols):
    """Compute and save all XGBoost diagnostics to DIAG_DIR."""
    os.makedirs(DIAG_DIR, exist_ok=True)
    print(f"\n[4] Saving diagnostics to {DIAG_DIR} ...")

    diag_df = pd.DataFrame(diag_rows)

    if diag_df.empty:
        raise ValueError(
            "No diagnostic rows were produced. "
            "Check the expanding-window split and ml_panel.csv."
        )

    # -- 1. IC detail ----------------------------------------------------------
    diag_df.to_csv(os.path.join(DIAG_DIR, "xgboost_ic_detail.csv"), index=False)
    print("  Saved: xgboost_ic_detail.csv")

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
            "n_dates"      : int(len(ydf)),
        })
    ic_rows.append({
        "year"         : "all",
        "mean_IC"      : round(diag_df["spearman_ic"].mean(), 4),
        "hit_rate"     : round((diag_df["spearman_ic"] > 0).mean(), 4),
        "mean_pearson" : round(diag_df["pearson_ic"].mean(), 4),
        "mean_r2_rank" : round(diag_df["r2_rank"].mean(), 4)
                         if diag_df["r2_rank"].notna().any() else np.nan,
        "n_dates"      : int(len(diag_df)),
    })
    ic_df = pd.DataFrame(ic_rows)
    ic_df.to_csv(os.path.join(DIAG_DIR, "xgboost_ic_summary.csv"), index=False)
    print("  Saved: xgboost_ic_summary.csv")
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
        os.path.join(DIAG_DIR, "xgboost_mu_baseline_correlation.csv"), index=False
    )
    print(f"  Saved: xgboost_mu_baseline_correlation.csv ({len(corr_rows)} rows)")

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
        os.path.join(DIAG_DIR, "xgboost_prediction_summary.csv"), index=False
    )
    print(f"  Saved: xgboost_prediction_summary.csv ({len(summ_rows)} rows)")

    # -- 5. Feature importance (average across all dates) ----------------------
    if fi_list:
        fi_df = (
            pd.DataFrame(fi_list)
            .mean()
            .sort_values(ascending=False)
            .rename("importance")
        )
        fi_df.to_csv(
            os.path.join(DIAG_DIR, "xgboost_feature_importance.csv"), header=True
        )
        print(f"\n  XGBoost average feature importance (across {len(fi_list)} dates):")
        print(fi_df.to_string())

    # -- 6. Hyperparameters per date -------------------------------------------
    pd.DataFrame(hyper_rows).to_csv(
        os.path.join(DIAG_DIR, "xgboost_hyperparameters.csv"), index=False
    )
    print(f"  Saved: xgboost_hyperparameters.csv ({len(hyper_rows)} rows)")


# =============================================================================
# 9. QUALITY CHECKS
# =============================================================================

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


# =============================================================================
# 10. MAIN
# =============================================================================

def main():
    os.makedirs(STEP5_DIR, exist_ok=True)
    os.makedirs(DIAG_DIR,  exist_ok=True)

    print("=" * 70)
    print("  STEP 5B - XGBOOST EXPECTED RETURN ESTIMATION (MU_HAT)")
    print("=" * 70)

    panel, returns, baseline_weights, oos_dates, tickers, feature_cols = load_inputs()

    mu_hat_dict, diag_rows, hyper_rows, fi_list = run_oos(
        panel, oos_dates, returns, tickers, feature_cols
    )

    print("\n[3] Assembling and saving mu_hat matrix ...")
    mu_mat = assemble_mu_matrix(mu_hat_dict, oos_dates, tickers)

    nan_count = int(mu_mat.isna().sum().sum())
    if nan_count > 0:
        mu_mat = fill_nan_with_baseline(mu_mat, returns, tickers)

    out_path = os.path.join(STEP5_DIR, "ml_mu_xgboost.csv")
    mu_mat.to_csv(out_path)
    print(f"  Saved: ml_mu_xgboost.csv  shape={mu_mat.shape}  "
          f"NaN remaining={int(mu_mat.isna().sum().sum())}")

    save_diagnostics(diag_rows, hyper_rows, fi_list, mu_mat, oos_dates, returns, feature_cols)

    quality_checks(mu_mat, oos_dates, tickers, baseline_weights)

    print("\n" + "=" * 70)
    print("  STEP 5B - COMPLETE")
    print("=" * 70)
    print(f"""
  Outputs saved to: {STEP5_DIR}

  mu_hat matrix ({len(oos_dates)} dates x {len(tickers)} tickers):
    ml_mu_xgboost.csv

  Diagnostics ({DIAG_DIR}):
    xgboost_ic_detail.csv
    xgboost_ic_summary.csv
    xgboost_mu_baseline_correlation.csv
    xgboost_prediction_summary.csv
    xgboost_feature_importance.csv
    xgboost_hyperparameters.csv

  Next step: run step5a_ridge.py or step5c_mlp.py.
""")


if __name__ == "__main__":
    main()
