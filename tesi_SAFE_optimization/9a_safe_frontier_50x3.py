"""
9a_safe_frontier_50x3.py
==============================
Expands the SAFE–performance frontier to 50 configurations per model family
(Ridge, XGBoost, MLP).  Lambda is fixed at 1.0 throughout; the only varying
dimension is the ML model configuration.

Grids
-----
Ridge   : alpha = np.logspace(-4, 4, 50)
XGBoost : 50 reproducible configurations via random search (seed 42)
MLP     : 50 reproducible configurations via random search (seed 42)

Pipeline (identical to Step 10 for each configuration)
-------------------------------------------------------
1. Expanding-window OOS prediction
2. mu_hat generation (score rescaling to baseline distribution)
3. MIQP portfolio optimisation at lambda = 1.0
4. Portfolio backtest (buy-and-hold between rebalancing dates)
5. Performance metrics (Sharpe, Sortino, Calmar, MaxDD, Turnover)
6. SAFE compliance score - all three aggregations (arithmetic, geometric, RMS)

Output
------
data/results/step10/safe_performance_frontier_50x3.csv
  One row per configuration, columns:
    model_family, configuration_id, configuration_parameters,
    compliance_score_arithmetic, compliance_score_geometric, compliance_score_rms,
    accuracy_score, robustness_score, fairness_score, explainability_score,
    sharpe, max_drawdown, avg_turnover, ann_return, ann_vol, sortino, calmar

Checkpointing: completed rows are written immediately.  Restart the script to
resume—already-completed configuration_ids are skipped automatically.

Author: Anila Vata
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from config import (
    BASE_DIR,
    CLEAN_DIR,
    ESTIM_WINDOW,
    RANDOM_STATE,
    TRADING_DAYS,
    W_MAX,
    W_MIN,
)

warnings.filterwarnings("ignore")

try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False

try:
    from safeaipackage import check_robustness
    from safeaipackage import core as safe_core
    HAS_SAFE = True
except ImportError:
    HAS_SAFE = False

N_VAL_DATES   = 6
LAMBDA        = 1.0
K_PORT        = 10
SECTOR_CAP    = 0.30
TIME_LIMIT    = 60
MIP_GAP       = 0.01
TEST_START    = "2023-01-01"
TEST_END      = "2025-12-31"

N_SEGMENTS        = 10
PERTURBATION_GRID = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25,
                     0.30, 0.35, 0.40, 0.45, 0.50]
N_SAFE_POINTS     = 11
X_TARGET          = np.linspace(0.0, 1.0, N_SAFE_POINTS)
MIN_ASSETS        = 10
MIN_GROUP_ASSETS  = 5
MIN_VALID_SECTORS = 2

RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")
STEP3_DIR   = os.path.join(RESULTS_DIR, "step3")
STEP4_DIR   = os.path.join(RESULTS_DIR, "step4")
STEP10_DIR  = os.path.join(RESULTS_DIR, "step10")
os.makedirs(STEP10_DIR, exist_ok=True)

OUT_PATH = os.path.join(STEP10_DIR, "safe_performance_frontier_50x3.csv")

OUT_COLS = [
    "model_family", "configuration_id", "configuration_parameters",
    "compliance_score_arithmetic", "compliance_score_geometric",
    "compliance_score_rms",
    "accuracy_score", "robustness_score", "fairness_score", "explainability_score",
    "sharpe", "max_drawdown", "avg_turnover", "ann_return", "ann_vol",
    "sortino", "calmar",
]

def _make_ridge_configs():
    return [{"alpha": float(a)} for a in np.logspace(-4, 4, 50)]

def _make_xgb_configs(n: int = 50) -> list:
    rng = np.random.default_rng(42)
    md_opts = [2, 3, 4, 5, 6, 8]
    lr_opts = [0.005, 0.01, 0.03, 0.05, 0.10, 0.15]
    ne_opts = [50, 100, 150, 200, 300, 500]
    ss_opts = [0.6, 0.8, 1.0]
    cb_opts = [0.6, 0.8, 1.0]
    ra_opts = [0.0, 0.01, 0.1, 1.0]
    rl_opts = [0.1, 1.0, 5.0, 10.0]

    configs, seen = [], set()
    for _ in range(100_000):
        cfg = {
            "max_depth":        int(md_opts[int(rng.integers(len(md_opts)))]),
            "learning_rate":    float(lr_opts[int(rng.integers(len(lr_opts)))]),
            "n_estimators":     int(ne_opts[int(rng.integers(len(ne_opts)))]),
            "subsample":        float(ss_opts[int(rng.integers(len(ss_opts)))]),
            "colsample_bytree": float(cb_opts[int(rng.integers(len(cb_opts)))]),
            "reg_alpha":        float(ra_opts[int(rng.integers(len(ra_opts)))]),
            "reg_lambda":       float(rl_opts[int(rng.integers(len(rl_opts)))]),
        }
        key = tuple(cfg[k] for k in sorted(cfg))
        if key not in seen:
            seen.add(key)
            configs.append(cfg)
        if len(configs) == n:
            break
    return configs

def _make_mlp_configs(n: int = 50) -> list:
    rng = np.random.default_rng(42)
    hls_opts = [
        (32,), (64,), (128,), (256,),
        (64, 32), (128, 64), (256, 128),
        (128, 64, 32),
    ]
    alpha_opts  = [0.0001, 0.001, 0.01, 0.1]
    lr_opts     = [0.0001, 0.0005, 0.001, 0.005]
    bs_opts     = [64, 128, 256]

    configs, seen = [], set()
    for _ in range(100_000):
        hls = hls_opts[int(rng.integers(len(hls_opts)))]
        cfg = {
            "hidden_layer_sizes": hls,
            "alpha":              float(alpha_opts[int(rng.integers(len(alpha_opts)))]),
            "learning_rate_init": float(lr_opts[int(rng.integers(len(lr_opts)))]),
            "batch_size":         int(bs_opts[int(rng.integers(len(bs_opts)))]),
        }
        key = (cfg["hidden_layer_sizes"], cfg["alpha"],
               cfg["learning_rate_init"], cfg["batch_size"])
        if key not in seen:
            seen.add(key)
            configs.append(cfg)
        if len(configs) == n:
            break
    return configs

RIDGE_CONFIGS = _make_ridge_configs()
XGB_CONFIGS   = _make_xgb_configs(50)
MLP_CONFIGS   = _make_mlp_configs(50)

class ScaledRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, scaler, model, feature_cols):
        self.scaler       = scaler
        self.model        = model
        self.feature_cols = feature_cols

    def predict(self, X):
        X = pd.DataFrame(X, columns=self.feature_cols)
        return self.model.predict(self.scaler.transform(X[self.feature_cols]))

class DirectRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, model, feature_cols):
        self.model        = model
        self.feature_cols = feature_cols

    def predict(self, X):
        X = pd.DataFrame(X, columns=self.feature_cols)
        return self.model.predict(X[self.feature_cols].values)

def compute_baseline_mu(rebal_date, returns):
    hist = returns.loc[returns.index < rebal_date].iloc[-ESTIM_WINDOW:]
    raw  = hist.mean() * TRADING_DAYS
    lo   = np.nanpercentile(raw.values, 1)
    hi   = np.nanpercentile(raw.values, 99)
    return raw.clip(lo, hi)

def scores_to_mu_hat(scores, baseline_mu):
    common = scores.index.intersection(baseline_mu.index)
    s      = scores[common].dropna()
    bmu    = baseline_mu[common].dropna()
    both   = s.index.intersection(bmu.index)
    s, bmu = s[both], bmu[both]
    s_std  = s.std(ddof=0)
    if s_std < 1e-10:
        raise ValueError("Constant prediction scores.")
    s_scaled = (s - s.mean()) / s_std
    mu_hat   = s_scaled * bmu.std(ddof=0) + bmu.mean()
    lo = np.nanpercentile(mu_hat.values, 1)
    hi = np.nanpercentile(mu_hat.values, 99)
    return mu_hat.clip(lo, hi)

def compute_sigma(rebal_date, returns_df):
    hist = returns_df.loc[returns_df.index < rebal_date].iloc[-ESTIM_WINDOW:]
    lw   = LedoitWolf().fit(hist.values)
    return lw.covariance_ * TRADING_DAYS

def neutralize_features(X_te_df, X_tv_df, feat_list):
    xmod = X_te_df.copy()
    for col in feat_list:
        if col not in X_tv_df.columns:
            continue
        if pd.api.types.is_numeric_dtype(X_tv_df[col]):
            xmod[col] = float(X_tv_df[col].mean())
        else:
            mode_vals = X_tv_df[col].mode()
            xmod[col] = mode_vals.iloc[0] if len(mode_vals) > 0 else 0.0
    return xmod

def get_permutation_importance(proxy, feat_cols, X_te_df):
    rng = np.random.default_rng(RANDOM_STATE)
    try:
        base_preds = np.array(proxy.predict(X_te_df), dtype=float)
    except Exception:
        return np.ones(len(feat_cols))
    importance = np.zeros(len(feat_cols))
    for j, feat in enumerate(feat_cols):
        X_perm = X_te_df.copy()
        X_perm[feat] = rng.permutation(X_perm[feat].values)
        try:
            perm_preds    = np.array(proxy.predict(X_perm), dtype=float)
            importance[j] = float(np.mean((base_preds - perm_preds) ** 2))
        except Exception:
            importance[j] = 0.0
    return importance

def load_inputs():
    print("[LOAD] Loading static inputs ...")

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
    meta = pd.read_csv(os.path.join(CLEAN_DIR, "meta_clean.csv"))
    meta.columns = meta.columns.str.strip().str.lower()
    if "ticker" not in meta.columns:
        meta = meta.rename(columns={meta.columns[0]: "ticker"})
    meta["ticker"] = meta["ticker"].astype(str).str.strip().str.upper()

    meta_idx = meta.set_index("ticker")
    _SECTOR_CANDIDATES = ["sector", "gics_sector", "gics sector"]
    sector_col = next(
        (c for c in _SECTOR_CANDIDATES if c in meta_idx.columns), None
    )
    if sector_col is None:
        raise ValueError(
            f"No sector column in meta_clean.csv. Available: {list(meta_idx.columns)}"
        )

    oos_dates = baseline_weights.index
    tickers   = baseline_weights.columns.astype(str).tolist()
    returns   = returns[tickers]

    non_feature_cols = {"date", "ticker", "target_end_date", "target_raw", "target_rank"}
    feature_cols = [c for c in panel.columns if c not in non_feature_cols]
    if not feature_cols:
        raise ValueError("No feature columns in ml_panel.csv.")

    universe = tickers
    N        = len(universe)

    sector_map = {
        t: str(meta_idx.loc[t, sector_col]).strip()
        for t in universe if t in meta_idx.index
    }
    sectors = sorted(set(sector_map.values()))
    sector_indices = {
        s: [i for i, t in enumerate(universe) if sector_map.get(t) == s]
        for s in sectors
    }
    ticker_sector = pd.Series({t: sector_map.get(t, "Unknown") for t in tickers})

    returns_df = returns[universe]
    test_idx   = returns_df.index[
        (returns_df.index >= TEST_START) & (returns_df.index <= TEST_END)
    ]

    print(f"  panel      : {panel.shape}")
    print(f"  returns    : {returns_df.shape}")
    print(f"  OOS dates  : {len(oos_dates)}")
    print(f"  Tickers    : {N}")
    print(f"  Features   : {len(feature_cols)}: {feature_cols[:5]}...")
    print(f"  Sectors    : {len(sectors)}")

    return (panel, returns, baseline_weights, oos_dates, tickers, feature_cols,
            returns_df, universe, N, sector_indices, sectors, ticker_sector, test_idx)

def solve_miqp(mu_row, Sigma, N, sector_indices, label=""):
    m = gp.Model(f"miqp_{label}")
    m.Params.OutputFlag   = 0
    m.Params.TimeLimit    = TIME_LIMIT
    m.Params.MIPGap       = MIP_GAP
    m.Params.LogToConsole = 0

    w = m.addMVar(N, lb=0.0, ub=1.0,   name="w")
    z = m.addMVar(N, vtype=GRB.BINARY,  name="z")

    m.setObjective(w @ Sigma @ w - LAMBDA * (mu_row @ w), GRB.MINIMIZE)
    m.addConstr(w.sum() == 1.0,           name="budget")
    m.addConstr(z.sum() == float(K_PORT), name="cardinality")
    m.addConstr(w >= W_MIN * z,           name="lb")
    m.addConstr(w <= W_MAX * z,           name="ub")
    for s, idx_list in sector_indices.items():
        if idx_list:
            m.addConstr(w[idx_list].sum() <= SECTOR_CAP, name=f"sec_{s[:12]}")
    m.optimize()

    status    = m.Status
    STATUS_MAP = {
        GRB.OPTIMAL: "OPTIMAL", GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT", GRB.INFEASIBLE: "INFEASIBLE",
    }
    status_str = STATUS_MAP.get(status, f"STATUS_{status}")

    if status in (GRB.OPTIMAL, GRB.SUBOPTIMAL) or \
       (status == GRB.TIME_LIMIT and m.SolCount > 0):
        weights = np.maximum(w.X, 0.0)
        weights = weights / weights.sum() if weights.sum() > 1e-8 else None
        obj_val = float(m.ObjVal)
    else:
        weights = None
        obj_val = np.nan

    m.dispose()
    return weights, status_str, obj_val

def run_backtest(mu_mat, oos_dates, tickers, returns_df, N, sector_indices, test_idx):
    weights_all  = {}
    prev_weights = None

    for rebal_date in oos_dates:
        hist_ok = len(returns_df.loc[returns_df.index < rebal_date]) >= ESTIM_WINDOW
        if not hist_ok:
            if prev_weights is not None:
                weights_all[rebal_date] = prev_weights.copy()
            continue

        mu_row  = mu_mat.loc[rebal_date].values.astype(float)
        Sigma_t = compute_sigma(rebal_date, returns_df)
        eig_min = float(np.linalg.eigvalsh(Sigma_t).min())
        if eig_min < -1e-6:
            Sigma_t += (abs(eig_min) + 1e-6) * np.eye(N)

        weights, status_str, _ = solve_miqp(
            mu_row, Sigma_t, N, sector_indices,
            label=rebal_date.strftime("%Y%m%d"),
        )
        if weights is None:
            if prev_weights is None:
                raise RuntimeError(
                    f"Optimisation FAILED on first rebal date {rebal_date.date()} "
                    f"({status_str}), no fallback weights available."
                )
            weights = prev_weights.copy()

        weights_all[rebal_date] = weights
        prev_weights = weights.copy()

    rebal_list = sorted(weights_all.keys())

    daily_records            = []
    drifted_weights_by_rebal = {}

    for i, rebal_date in enumerate(rebal_list):
        next_rebal = (rebal_list[i + 1]
                      if i + 1 < len(rebal_list)
                      else test_idx[-1] + pd.Timedelta(days=1))
        mask = (
            (returns_df.index > rebal_date)
            & (returns_df.index < next_rebal)
            & (returns_df.index >= TEST_START)
            & (returns_df.index <= TEST_END)
        )
        period_ret = returns_df.loc[mask]
        if period_ret.empty:
            drifted_weights_by_rebal[rebal_date] = weights_all[rebal_date].copy()
            continue

        w_drift = weights_all[rebal_date].copy()
        for date, row in period_ret.iterrows():
            asset_log_ret = row.values
            gross_ret     = float(w_drift @ np.exp(asset_log_ret))
            daily_records.append({
                "date":             date,
                "portfolio_return": float(np.log(gross_ret)),
            })
            w_drift = w_drift * np.exp(asset_log_ret) / gross_ret
        drifted_weights_by_rebal[rebal_date] = w_drift.copy()

    if daily_records:
        daily_ret_df = pd.DataFrame(daily_records).set_index("date").sort_index()
    else:
        daily_ret_df = pd.DataFrame(columns=["portfolio_return"])

    turnover_records = []
    for i in range(1, len(rebal_list)):
        prev_date   = rebal_list[i - 1]
        curr_date   = rebal_list[i]
        w_pre_trade = drifted_weights_by_rebal[prev_date]
        w_target    = weights_all[curr_date]
        to          = float(np.abs(w_target - w_pre_trade).sum()) / 2.0
        turnover_records.append({"rebal_date": curr_date.date().isoformat(), "turnover": to})
    turnover_df = pd.DataFrame(turnover_records)

    return daily_ret_df, turnover_df

def compute_performance(daily_ret_df, turnover_df):
    nan_result = dict(
        ann_return=np.nan, ann_vol=np.nan, sharpe=np.nan,
        max_drawdown=np.nan, avg_turnover=np.nan,
        sortino=np.nan, calmar=np.nan,
    )
    if daily_ret_df.empty or "portfolio_return" not in daily_ret_df.columns:
        return nan_result
    r = daily_ret_df["portfolio_return"].dropna().values
    if len(r) < 2:
        return nan_result

    ann_ret = float(r.mean() * TRADING_DAYS)
    ann_vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan

    wealth  = np.concatenate([[1.0], np.exp(np.cumsum(r))])
    run_max = np.maximum.accumulate(wealth)
    max_dd  = float(abs((wealth / run_max - 1).min()))

    # Sortino: downside deviation uses only days with negative log-return
    neg_r = r[r < 0]
    if len(neg_r) >= 2:
        down_std = float(neg_r.std(ddof=1) * np.sqrt(TRADING_DAYS))
        sortino  = ann_ret / down_std if down_std > 0 else np.nan
    else:
        sortino = np.nan

    calmar  = ann_ret / max_dd if max_dd > 0 else np.nan
    avg_to  = float(turnover_df["turnover"].mean()) if len(turnover_df) > 0 else np.nan

    return dict(
        ann_return=ann_ret, ann_vol=ann_vol, sharpe=sharpe,
        max_drawdown=max_dd, avg_turnover=avg_to,
        sortino=sortino, calmar=calmar,
    )

def compute_rga_date(target_rank_s, mu_hat_s):
    if not HAS_SAFE:
        return None
    common = target_rank_s.index.intersection(mu_hat_s.index)
    sub    = pd.DataFrame(
        {"target_rank": target_rank_s[common], "mu_hat": mu_hat_s[common]}
    ).dropna()
    n_full = len(sub)
    if n_full < MIN_ASSETS:
        return None
    if np.nanstd(sub["target_rank"].values) < 1e-12:
        return None
    if np.nanstd(sub["mu_hat"].values) < 1e-12:
        return None

    sub_sorted  = sub.sort_values("mu_hat", ascending=False).reset_index(drop=True)
    sorted_idx  = np.arange(n_full)
    segments    = np.array_split(sorted_idx, N_SEGMENTS)
    rga_vals    = []

    for k in range(N_SEGMENTS + 1):
        if k == N_SEGMENTS:
            rga_vals.append(0.0)
            continue
        if k == 0:
            remaining_idx = sorted_idx.copy()
        else:
            removed_set   = set(np.concatenate(segments[:k]).tolist())
            remaining_idx = np.array([i for i in sorted_idx if i not in removed_set])

        if len(remaining_idx) < MIN_ASSETS:
            return None

        target_rem = sub_sorted["target_rank"].iloc[remaining_idx].tolist()
        yhat_rem   = sub_sorted["mu_hat"].iloc[remaining_idx].tolist()
        if np.nanstd(target_rem) < 1e-12 or np.nanstd(yhat_rem) < 1e-12:
            return None

        try:
            rga_vals.append(round(float(safe_core.rga(target_rem, yhat_rem)), 6))
        except Exception:
            return None

    return np.array(rga_vals)

def compute_rgr_date(X_te_df, yhat_list, proxy, feat_cols):
    if not HAS_SAFE:
        return None
    if np.std(yhat_list) < 1e-12:
        return None

    rgr_vals = []
    for p in PERTURBATION_GRID:
        if p == 0.00:
            rgr_vals.append(1.0)
            continue
        try:
            result = check_robustness.compute_rgr_values(
                X_te_df, yhat_list, proxy, feat_cols,
                perturbation_percentage=p, group=True,
            )
            rgr_vals.append(round(float(result["RGR"].iloc[0]), 6))
        except Exception:
            return None

    return np.array(rgr_vals)

def compute_rge_date(yhat_list, X_te_df, X_tv_df, proxy, feat_cols):
    if not HAS_SAFE:
        return None
    if np.std(yhat_list) < 1e-12:
        return None

    importance    = get_permutation_importance(proxy, feat_cols, X_te_df)
    removal_order = np.argsort(-importance)
    n_feats       = len(feat_cols)
    rge_vals      = [1.0]

    for k in range(1, n_feats + 1):
        feat_list = [feat_cols[removal_order[j]] for j in range(k)]
        try:
            X_mod    = neutralize_features(X_te_df, X_tv_df, feat_list)
            mod_yhat = proxy.predict(X_mod).tolist()
            if np.std(mod_yhat) < 1e-12:
                rge_k = 0.0
            else:
                rge_k = round(float(safe_core.rga(yhat_list, mod_yhat)), 6)
        except Exception:
            return None
        rge_vals.append(rge_k)

    x_raw  = np.linspace(0.0, 1.0, n_feats + 1)
    rge_11 = np.interp(X_TARGET, x_raw, np.array(rge_vals))
    return np.clip(rge_11, 0.0, 1.0)

def compute_rgf_date(target_rank_s, mu_hat_s, sector_s):
    if not HAS_SAFE:
        return None

    common = (target_rank_s.index
              .intersection(mu_hat_s.index)
              .intersection(sector_s.index))
    tr_s = target_rank_s[common].dropna()
    mh_s = mu_hat_s[common].dropna()
    valid = tr_s.index.intersection(mh_s.index)
    if len(valid) < MIN_ASSETS:
        return None

    tr_v = tr_s[valid]
    mh_v = mh_s[valid]
    se_v = sector_s[valid]

    sector_rgas = {}
    for s in se_v.unique():
        if s == "Unknown":
            continue
        mask = se_v == s
        if mask.sum() < MIN_GROUP_ASSETS:
            continue
        tgt = tr_v[mask].tolist()
        mh  = mh_v[mask].tolist()
        if np.nanstd(tgt) < 1e-12 or np.nanstd(mh) < 1e-12:
            continue
        try:
            sector_rgas[s] = round(float(safe_core.rga(tgt, mh)), 6)
        except Exception:
            continue

    if len(sector_rgas) < MIN_VALID_SECTORS:
        return None

    sorted_sectors = sorted(sector_rgas.keys(), key=lambda s: -sector_rgas[s])
    best_rga       = sector_rgas[sorted_sectors[0]]
    G              = len(sorted_sectors)
    x_raw          = np.linspace(0.0, 1.0, G)
    rgf_raw        = np.array([1.0 - (best_rga - sector_rgas[s]) for s in sorted_sectors])

    rgf_11 = np.interp(X_TARGET, x_raw, rgf_raw)
    return np.clip(rgf_11, 0.0, 1.0)

def compute_cs4(A, B, C, D):
    """Return (cs4_arith, cs4_geo, cs4_rms) for four 11-point compliance vectors."""
    A, B, C, D = [np.clip(v, 0.0, 1.0) for v in [A, B, C, D]]
    A4 = A[:, None, None, None]
    B4 = B[None, :, None, None]
    C4 = C[None, None, :, None]
    D4 = D[None, None, None, :]
    cs4_arith = float(np.mean((A4 + B4 + C4 + D4) / 4.0))
    prod      = A4 * B4 * C4 * D4
    cs4_geo   = float(np.mean(np.where(prod > 0, prod ** 0.25, 0.0)))
    cs4_rms   = float(np.mean(np.sqrt((A4**2 + B4**2 + C4**2 + D4**2) / 4.0)))
    return cs4_arith, cs4_geo, cs4_rms

def train_ridge_date(config, X_tv, y_tv, X_te, te_tickers, baseline_mu, feat_cols):
    sc     = StandardScaler().fit(X_tv)
    mdl    = Ridge(alpha=config["alpha"],
                   random_state=RANDOM_STATE).fit(sc.transform(X_tv), y_tv)
    preds  = mdl.predict(sc.transform(X_te))
    scores = pd.Series(preds, index=te_tickers)
    mu_hat = scores_to_mu_hat(scores, baseline_mu)
    proxy  = ScaledRegressor(sc, mdl, feat_cols)
    return proxy, scores, mu_hat

def train_xgb_date(config, X_tv, y_tv, X_te, te_tickers, baseline_mu, feat_cols):
    mdl = xgb.XGBRegressor(
        max_depth        = config["max_depth"],
        learning_rate    = config["learning_rate"],
        n_estimators     = config["n_estimators"],
        subsample        = config["subsample"],
        colsample_bytree = config["colsample_bytree"],
        reg_alpha        = config["reg_alpha"],
        reg_lambda       = config["reg_lambda"],
        objective        = "reg:squarederror",
        random_state     = RANDOM_STATE,
        verbosity        = 0,
        n_jobs           = 1,
    )
    mdl.fit(X_tv, y_tv)
    preds  = mdl.predict(X_te)
    scores = pd.Series(preds, index=te_tickers)
    mu_hat = scores_to_mu_hat(scores, baseline_mu)
    proxy  = DirectRegressor(mdl, feat_cols)
    return proxy, scores, mu_hat

def train_mlp_date(config, X_tv, y_tv, X_te, te_tickers, baseline_mu, feat_cols):
    sc  = StandardScaler().fit(X_tv)
    mdl = MLPRegressor(
        hidden_layer_sizes = config["hidden_layer_sizes"],
        alpha              = config["alpha"],
        learning_rate_init = config["learning_rate_init"],
        batch_size         = config["batch_size"],
        activation         = "relu",
        solver             = "adam",
        early_stopping     = True,
        max_iter           = 500,
        random_state       = RANDOM_STATE,
    ).fit(sc.transform(X_tv), y_tv)
    preds  = mdl.predict(sc.transform(X_te))
    scores = pd.Series(preds, index=te_tickers)
    mu_hat = scores_to_mu_hat(scores, baseline_mu)
    proxy  = ScaledRegressor(sc, mdl, feat_cols)
    return proxy, scores, mu_hat

TRAIN_FUNS = {
    "ridge":   train_ridge_date,
    "xgboost": train_xgb_date,
    "mlp":     train_mlp_date,
}

def run_one_config(model_family, config_id, config_params, config_str,
                   panel, returns, baseline_weights, oos_dates, tickers,
                   feature_cols, returns_df, universe, N, sector_indices,
                   ticker_sector, test_idx):
    print(f"\n{'='*62}")
    print(f"  [{config_id}]  {model_family.upper()}  |  {config_str}")
    print(f"{'='*62}")
    t_config = time.perf_counter()

    mu_hat_dict = {}
    rga_list, rgr_list, rge_list, rgf_list = [], [], [], []
    train_fn = TRAIN_FUNS[model_family]

    for t in oos_dates:
        t_str = t.date().isoformat()

        baseline_mu = compute_baseline_mu(t, returns)

        eligible = panel[
            panel["target_end_date"].notna()
            & (panel["target_end_date"] <= t)
            & panel["target_rank"].notna()
        ]
        elig_dates = pd.DatetimeIndex(eligible["date"].drop_duplicates()).sort_values()

        if len(elig_dates) < N_VAL_DATES + 1:
            continue

        val_dates   = elig_dates[-N_VAL_DATES:]
        train_dates = elig_dates[:-N_VAL_DATES]

        train_slice = eligible[eligible["date"].isin(train_dates)]
        val_slice   = eligible[eligible["date"].isin(val_dates)]
        test_slice  = panel[panel["date"] == t].dropna(subset=feature_cols)

        if train_slice.empty or val_slice.empty or test_slice.empty:
            continue

        X_tr = train_slice[feature_cols].values.astype(float)
        y_tr = train_slice["target_rank"].values.astype(float)
        X_vl = val_slice[feature_cols].values.astype(float)
        y_vl = val_slice["target_rank"].values.astype(float)
        X_tv = np.vstack([X_tr, X_vl])
        y_tv = np.concatenate([y_tr, y_vl])
        X_te = test_slice[feature_cols].values.astype(float)
        te_tickers_arr = test_slice["ticker"].values

        try:
            proxy, scores, mu_hat = train_fn(
                config_params, X_tv, y_tv, X_te, te_tickers_arr,
                baseline_mu, feature_cols,
            )
        except Exception as exc:
            print(f"    [{t_str}] SKIP (train): {exc}")
            continue

        mu_hat_dict[t] = mu_hat

        if t < pd.Timestamp(TEST_START):
            continue

        test_indexed  = test_slice.set_index("ticker")
        target_rank_s = test_indexed["target_rank"].dropna()
        sector_s      = ticker_sector.reindex(test_indexed.index).fillna("Unknown")

        X_te_df   = pd.DataFrame(X_te, columns=feature_cols, index=te_tickers_arr)
        X_tv_df   = pd.DataFrame(X_tv, columns=feature_cols)
        yhat_list = scores.reindex(te_tickers_arr).fillna(0.0).tolist()

        v = compute_rga_date(target_rank_s, mu_hat)
        if v is not None:
            rga_list.append(v)

        v = compute_rgr_date(X_te_df, yhat_list, proxy, feature_cols)
        if v is not None:
            rgr_list.append(v)

        v = compute_rge_date(yhat_list, X_te_df, X_tv_df, proxy, feature_cols)
        if v is not None:
            rge_list.append(v)

        v = compute_rgf_date(target_rank_s, mu_hat, sector_s)
        if v is not None:
            rgf_list.append(v)

    if not mu_hat_dict:
        raise RuntimeError(f"[{config_id}] No valid OOS dates produced mu_hat.")

    print(f"  mu_hat dates : {len(mu_hat_dict)} / {len(oos_dates)}")
    print(f"  SAFE valid   : RGA={len(rga_list)}  RGR={len(rgr_list)}  "
          f"RGE={len(rge_list)}  RGF={len(rgf_list)}")

    # Build mu_mat, filling missing dates with baseline
    rows = []
    for t in oos_dates:
        if t in mu_hat_dict:
            s = mu_hat_dict[t].reindex(tickers)
            missing = s.index[s.isna()].tolist()
            if missing:
                bmu = compute_baseline_mu(t, returns)
                s[missing] = bmu.reindex(missing).values
        else:
            bmu = compute_baseline_mu(t, returns)
            s   = bmu.reindex(tickers)
        s.name = t
        rows.append(s)
    mu_mat = pd.DataFrame(rows)
    mu_mat.index.name = "date"
    mu_mat.fillna(0.0, inplace=True)

    print("  Running MIQP backtest ...")
    t0 = time.perf_counter()
    daily_ret_df, turnover_df = run_backtest(
        mu_mat, oos_dates, tickers, returns_df, N, sector_indices, test_idx,
    )
    perf = compute_performance(daily_ret_df, turnover_df)
    print(f"  Backtest: {time.perf_counter()-t0:.1f}s  "
          f"Sharpe={perf['sharpe']:.4f}  Sortino={perf['sortino']:.4f}  "
          f"Calmar={perf['calmar']:.4f}  MaxDD={perf['max_drawdown']:.4f}")

    # SAFE compliance scores
    def _mean_vec(lst):
        if not lst:
            return np.full(N_SAFE_POINTS, np.nan)
        return np.mean(np.vstack(lst), axis=0)

    A = _mean_vec(rga_list)
    B = _mean_vec(rgr_list)
    C = _mean_vec(rge_list)
    D = _mean_vec(rgf_list)

    cs_arith = cs_geo = cs_rms = np.nan
    accuracy_score = robustness_score = fairness_score = explainability_score = np.nan

    if HAS_SAFE and all(not np.isnan(v).all() for v in [A, B, C, D]):
        A_c = np.nan_to_num(A, nan=0.0)
        B_c = np.nan_to_num(B, nan=0.0)
        C_c = np.nan_to_num(C, nan=0.0)
        D_c = np.nan_to_num(D, nan=0.0)
        cs_arith, cs_geo, cs_rms = compute_cs4(A_c, B_c, C_c, D_c)
        accuracy_score       = float(np.nanmean(A))
        robustness_score     = float(np.nanmean(B))
        explainability_score = float(np.nanmean(C))
        fairness_score       = float(np.nanmean(D))

    print(f"  CS4: arith={cs_arith:.4f}  geo={cs_geo:.4f}  rms={cs_rms:.4f}")
    print(f"  Total: {time.perf_counter()-t_config:.1f}s")

    def _r(v):
        return round(float(v), 6) if np.isfinite(v) else np.nan

    return {
        "model_family":               model_family,
        "configuration_id":           config_id,
        "configuration_parameters":   config_str,
        "compliance_score_arithmetic": _r(cs_arith),
        "compliance_score_geometric":  _r(cs_geo),
        "compliance_score_rms":        _r(cs_rms),
        "accuracy_score":              _r(accuracy_score),
        "robustness_score":            _r(robustness_score),
        "fairness_score":              _r(fairness_score),
        "explainability_score":        _r(explainability_score),
        "sharpe":                      _r(perf["sharpe"]),
        "max_drawdown":                _r(perf["max_drawdown"]),
        "avg_turnover":                _r(perf["avg_turnover"]),
        "ann_return":                  _r(perf["ann_return"]),
        "ann_vol":                     _r(perf["ann_vol"]),
        "sortino":                     _r(perf["sortino"]),
        "calmar":                      _r(perf["calmar"]),
    }

def _config_str_ridge(cfg):
    return f"alpha={cfg['alpha']:.6g}"

def _config_str_xgb(cfg):
    return (f"md={cfg['max_depth']},lr={cfg['learning_rate']:.4g},"
            f"ne={cfg['n_estimators']},ss={cfg['subsample']},"
            f"cb={cfg['colsample_bytree']},ra={cfg['reg_alpha']},"
            f"rl={cfg['reg_lambda']}")

def _config_str_mlp(cfg):
    return (f"layers={cfg['hidden_layer_sizes']},alpha={cfg['alpha']:.4g},"
            f"lr_init={cfg['learning_rate_init']:.4g},bs={cfg['batch_size']}")

def main():
    if not HAS_GUROBI:
        raise ImportError("gurobipy not available -- install Gurobi with a valid licence.")
    if not HAS_SAFE:
        raise ImportError("safeaipackage not available, install it first.")

    print("=" * 70)
    print("  STEP 9A - SAFE PERFORMANCE FRONTIER  (50 × 3 configurations)")
    print(f"  lambda={LAMBDA}  K={K_PORT}  W_MIN={W_MIN}  W_MAX={W_MAX}  "
          f"SECTOR_CAP={SECTOR_CAP}")
    print("=" * 70)
    print(f"  Ridge   : {len(RIDGE_CONFIGS)} configs")
    print(f"  XGBoost : {len(XGB_CONFIGS)} configs")
    print(f"  MLP     : {len(MLP_CONFIGS)} configs")

    # Resume: skip already-completed configuration_ids
    if os.path.exists(OUT_PATH):
        existing     = pd.read_csv(OUT_PATH)
        completed    = set(existing["configuration_id"].tolist())
        n_existing   = len(existing)
        print(f"\n  Resuming -- {n_existing} rows already in {os.path.basename(OUT_PATH)}")
    else:
        completed  = set()
        n_existing = 0

    (panel, returns, baseline_weights, oos_dates, tickers, feature_cols,
     returns_df, universe, N, sector_indices, sectors, ticker_sector, test_idx) = load_inputs()

    all_configs = (
        [("ridge",   f"ridge_{i+1:02d}",   cfg, _config_str_ridge(cfg))
         for i, cfg in enumerate(RIDGE_CONFIGS)]
        + [("xgboost", f"xgboost_{i+1:02d}", cfg, _config_str_xgb(cfg))
           for i, cfg in enumerate(XGB_CONFIGS)]
        + [("mlp",    f"mlp_{i+1:02d}",    cfg, _config_str_mlp(cfg))
           for i, cfg in enumerate(MLP_CONFIGS)]
    )

    n_done = n_existing
    n_total = len(all_configs)

    for family, cid, cfg, cstr in all_configs:
        if cid in completed:
            print(f"  [SKIP, already done]  {cid}")
            continue

        try:
            row = run_one_config(
                family, cid, cfg, cstr,
                panel, returns, baseline_weights, oos_dates, tickers,
                feature_cols, returns_df, universe, N, sector_indices,
                ticker_sector, test_idx,
            )
        except Exception as exc:
            print(f"\n  [{cid}] FAILED: {exc}")
            continue

        row_df = pd.DataFrame([row], columns=OUT_COLS)
        write_header = not os.path.exists(OUT_PATH)
        row_df.to_csv(OUT_PATH, mode="a", header=write_header, index=False)
        n_done += 1
        print(f"\n  Progress: {n_done}/{n_total} written to {os.path.basename(OUT_PATH)}")

    print("\n" + "=" * 70)
    print("  STEP 9A - COMPLETE")
    print(f"  Output: {OUT_PATH}")
    print("=" * 70)

    final = pd.read_csv(OUT_PATH)
    disp  = ["model_family", "configuration_id",
             "compliance_score_arithmetic", "sharpe", "sortino", "calmar"]
    print(final[disp].to_string(index=False))

if __name__ == "__main__":
    main()
