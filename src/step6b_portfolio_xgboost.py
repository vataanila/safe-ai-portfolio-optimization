"""
step6b_portfolio_xgboost.py
===========================
PURPOSE : Step 6b - XGBoost ML-enhanced Markowitz MIQP portfolio construction
          and backtest.

          This script loads precomputed XGBoost mu_hat predictions from Step 5b
          and runs the identical MIQP backtest used in step3_baseline_portfolio.py.
          It does NOT recompute ML predictions.

          The following components are held exactly identical to step3_baseline_portfolio.py
          to isolate the contribution of the ML mu signal:
            - Covariance estimator : rolling Ledoit-Wolf, trailing 252 daily
                                     returns strictly before each rebal date
            - MIQP optimizer       : Gurobi, identical formulation and parameters
            - MIQP constraints     : K=10, W_MIN=1%, W_MAX=20%, SECTOR_CAP=30%
            - Daily return calc    : buy-and-hold drift convention (log returns)
            - Turnover convention  : 0.5 * sum|w_target - w_pre_trade|,
                                     first rebalance excluded
            - Performance metrics  : ann_return, ann_vol, Sharpe, Sortino,
                                     Calmar, max_drawdown, avg_turnover

          The ONLY difference vs. step3_baseline_portfolio.py is the mu estimator:
            - Baseline (Step 3) : trailing historical mean return
            - XGBoost (Step 6b) : XGBoost ML predictions from Step 5b

          Note: Step 2 produces full-window Ledoit-Wolf covariance files for
          diagnostic purposes only.  This script estimates covariance rolling
          inside the backtest loop; those Step 2 files are not used here.

PIPELINE:
  Step 3  -> step3_baseline_portfolio.py              (Markowitz baseline, historical mu)
  Step 5b -> step5b_xgboost_mu.py           (XGBoost mu predictions)
  Step 6a -> step6a_portfolio_ridge.py      (Ridge variant)
  Step 6b -> this script                    (XGBoost MIQP portfolio, backtested)
  Step 6c -> step6c_portfolio_mlp.py        (MLP variant)
  Step 6d -> step6d_compare_portfolios.py   (cross-model comparison)

INPUTS  :
  data/clean/returns.csv
  data/clean/meta_clean.csv
  data/results/baseline_weights.csv          -- universe order and rebal dates
  data/results/step5/ml_mu_xgboost.csv       -- XGBoost mu_hat (rebal dates x tickers)

OUTPUTS (data/results/step6/xgboost/) :
  xgboost_weights.csv
  xgboost_returns.csv
  xgboost_turnover.csv
  xgboost_summary.csv
  xgboost_wealth_drawdown.csv
  xgboost_net_cost_summary.csv
  xgboost_portfolio_log.txt

METHODOLOGY:
  mu  : XGBoost ML predictions loaded from step5b - NOT recomputed here.
  Sigma   : rolling Ledoit-Wolf, re-estimated at each rebalancing date using
        trailing ESTIM_WINDOW-day returns strictly before rebal_date.
  MIQP:
    minimize  w'Sigmaw - lambda*mu'w          (lambda = 1)
    s.t.      Sigma w_i  = 1            (fully invested)
              z_i in {0,1}, Sigma z_i = K = 10   (cardinality)
              0.01*z_i <= w_i <= 0.20*z_i      (weight bounds, selected only)
              Sigma_{iins} w_i <= 0.30              (sector concentration cap)
  Solver: Gurobi via gurobipy (MVar API). TimeLimit=60s, MIPGap=0.01.

Author  : Anila Vata
Project : MSc Thesis - ML-Enhanced Portfolio Optimization with SAFE AI Evaluation
          University of Pavia * Supervisor: Prof. Paolo Giudici
"""

# =============================================================================
# 0. IMPORTS AND PATHS
# =============================================================================
import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False

warnings.filterwarnings("ignore")

# ---------- model identity ---------------------------------------------------
MODEL_NAME = "xgboost"
MU_FILE    = "ml_mu_xgboost.csv"

# ---------- paths ------------------------------------------------------------
BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DIR        = os.path.join(BASE_DIR, "data", "clean")
RESULTS_DIR      = os.path.join(BASE_DIR, "data", "results")
STEP3_DIR        = os.path.join(RESULTS_DIR, "step3")
STEP5_DIR        = os.path.join(RESULTS_DIR, "step5")
STEP6_DIR        = os.path.join(RESULTS_DIR, "step6")
MODEL_STEP6_DIR  = os.path.join(STEP6_DIR, MODEL_NAME)
os.makedirs(MODEL_STEP6_DIR, exist_ok=True)

# ---------- optimization parameters (identical to step3_baseline_portfolio.py) ----------
LAMBDA       = 1.0
K            = 10
W_MIN        = 0.01
W_MAX        = 0.20
SECTOR_CAP   = 0.30
TRADING_DAYS = 252
ESTIM_WINDOW = 252
TIME_LIMIT   = 60
MIP_GAP      = 0.01

TEST_START   = "2023-01-01"
TEST_END     = "2025-12-31"

# ---------- logging ----------------------------------------------------------
log_lines = []

def log(msg=""):
    print(msg)
    log_lines.append(str(msg))

log("=" * 70)
log(f"  STEP 6b - XGBOOST ML-ENHANCED MARKOWITZ MIQP PORTFOLIO")
log("=" * 70)

if not HAS_GUROBI:
    raise ImportError(
        "gurobipy is not installed or no valid licence found.\n"
        "Install with: pip install gurobipy   (requires Gurobi licence)")

log(f"  Gurobi version : {gp.gurobi.version()}")

# =============================================================================
# 1. LOAD STATIC INPUTS
# =============================================================================
log("\n-- 1. LOAD STATIC INPUTS --------------------------------------------")

for fpath in [
    os.path.join(CLEAN_DIR, "returns.csv"),
    os.path.join(CLEAN_DIR, "meta_clean.csv"),
    os.path.join(STEP3_DIR, "baseline_weights.csv"),
]:
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Required file not found: {fpath}")

mu_path = os.path.join(STEP5_DIR, MU_FILE)
if not os.path.exists(mu_path):
    raise FileNotFoundError(f"ML mu file not found: {mu_path}")

returns_raw = pd.read_csv(
    os.path.join(CLEAN_DIR, "returns.csv"),
    index_col=0, parse_dates=True
).sort_index()

meta = pd.read_csv(os.path.join(CLEAN_DIR, "meta_clean.csv"))
meta.columns = meta.columns.str.strip().str.lower()
if "ticker" not in meta.columns:
    meta = meta.rename(columns={meta.columns[0]: "ticker"})
meta["ticker"] = meta["ticker"].astype(str).str.strip().str.upper()

baseline_weights = pd.read_csv(
    os.path.join(STEP3_DIR, "baseline_weights.csv"),
    index_col=0, parse_dates=True
)

log(f"  returns_raw      : {returns_raw.shape[0]} days x {returns_raw.shape[1]} stocks")
log(f"  meta             : {meta.shape}")
log(f"  Return range     : {returns_raw.index[0].date()} to {returns_raw.index[-1].date()}")
log(f"  baseline_weights : {baseline_weights.shape[0]} rebal dates x "
    f"{baseline_weights.shape[1]} tickers")
log(f"  ML mu file       : {MU_FILE}")

# =============================================================================
# 2. UNIVERSE ALIGNMENT
#    Universe is taken directly from baseline_weights.columns - no resorting.
#    This guarantees identical ticker order to the baseline.
# =============================================================================
log("\n-- 2. UNIVERSE ALIGNMENT --------------------------------------------")

baseline_tickers = list(baseline_weights.columns)
N = len(baseline_tickers)

log(f"  Baseline tickers  : {N} (order preserved from baseline_weights.columns)")

# Every baseline ticker must exist in returns.csv
missing_in_returns = [t for t in baseline_tickers if t not in returns_raw.columns]
if missing_in_returns:
    raise ValueError(
        f"{len(missing_in_returns)} baseline tickers missing from returns.csv:\n"
        f"  {missing_in_returns[:10]}"
    )
log(f"  returns coverage  : PASS - all {N} baseline tickers present in returns.csv")

# Every baseline ticker must exist in meta_clean.csv
meta_tickers = set(meta["ticker"])
missing_in_meta = [t for t in baseline_tickers if t not in meta_tickers]
if missing_in_meta:
    raise ValueError(
        f"{len(missing_in_meta)} baseline tickers missing from meta_clean.csv:\n"
        f"  {missing_in_meta[:10]}"
    )
log(f"  meta coverage     : PASS - all {N} baseline tickers present in meta_clean.csv")

if N < K:
    raise ValueError(
        f"Universe has only {N} stocks but cardinality K={K} requires at least K.")

universe   = baseline_tickers
returns_df = returns_raw[universe]

# Sector mapping - require an explicitly recognised column name
meta_idx          = meta.set_index("ticker")
_SECTOR_CANDIDATES = ["sector", "gics_sector", "gics sector"]
sector_col = next(
    (c for c in _SECTOR_CANDIDATES if c in meta_idx.columns),
    None
)
if sector_col is None:
    raise ValueError(
        f"No sector column found in meta_clean.csv.\n"
        f"  Expected one of {_SECTOR_CANDIDATES}.\n"
        f"  Available columns: {list(meta_idx.columns)}"
    )
log(f"  Sector column     : '{sector_col}'")

sector_map = {
    t: str(meta_idx.loc[t, sector_col]).strip()
    for t in universe
}
sectors        = sorted(set(sector_map.values()))
sector_indices = {
    s: [i for i, t in enumerate(universe) if sector_map[t] == s]
    for s in sectors
}

log(f"\n  Sectors ({len(sectors)}):")
for s in sectors:
    log(f"    {s:<42} {len(sector_indices[s]):>3} stocks")

# =============================================================================
# 3. REBALANCING DATES - taken directly from baseline_weights.csv
# =============================================================================
log("\n-- 3. REBALANCING DATES (from baseline_weights.csv) -----------------")

rebal_dates = baseline_weights.index.sort_values()

log(f"  Test window    : {TEST_START} to {TEST_END}")
log(f"  Rebalancing    : first trading day of each calendar month (from baseline)")
log(f"  Rebal periods  : {len(rebal_dates)}"
    f"  ({rebal_dates[0].date()} to {rebal_dates[-1].date()})")

test_idx = returns_df.index[
    (returns_df.index >= TEST_START) & (returns_df.index <= TEST_END)
]

if test_idx.empty:
    raise ValueError(
        f"No trading days found in returns_df between {TEST_START} and {TEST_END}."
    )

# =============================================================================
# 4. QUALITY CHECKS ON ML MU MATRIX
# =============================================================================
log("\n-- 4. QUALITY CHECKS ON ML MU MATRIX --------------------------------")

mu_df = pd.read_csv(mu_path, index_col=0, parse_dates=True).sort_index()
log(f"  [{MODEL_NAME}]  {mu_df.shape[0]} dates x {mu_df.shape[1]} tickers")

# QC 1 - dates must exactly match baseline_weights.index
if not mu_df.index.equals(rebal_dates):
    missing_in_mu = sorted(set(rebal_dates) - set(mu_df.index))
    extra_in_mu   = sorted(set(mu_df.index)  - set(rebal_dates))
    raise ValueError(
        f"[{MODEL_NAME}] Date mismatch with baseline_weights.\n"
        f"  Missing in ML mu : {missing_in_mu[:5]}\n"
        f"  Extra in ML mu   : {extra_in_mu[:5]}"
    )
log(f"  QC1 dates      : PASS - {len(rebal_dates)} dates match baseline")

# QC 2 - columns must exactly match baseline_weights.columns (order included)
expected_cols = list(baseline_weights.columns)
if list(mu_df.columns) != expected_cols:
    first_diff = next(
        (i for i, (a, b) in enumerate(zip(mu_df.columns, expected_cols)) if a != b),
        None
    )
    raise ValueError(
        f"[{MODEL_NAME}] Column mismatch with baseline_weights.\n"
        f"  Expected first 5 : {expected_cols[:5]}\n"
        f"  Got first 5      : {list(mu_df.columns)[:5]}\n"
        f"  First diff at idx: {first_diff}"
    )
log(f"  QC2 columns    : PASS - {len(expected_cols)} tickers, correct order")

# QC 3 - no NaN values
n_nan = int(mu_df.isna().sum().sum())
if n_nan > 0:
    raise ValueError(f"[{MODEL_NAME}] {n_nan} NaN values found in ml_mu matrix.")
log(f"  QC3 no-NaN     : PASS")

# QC 4 - no constant rows
n_constant = int((mu_df.nunique(axis=1) == 1).sum())
if n_constant > 0:
    bad_dates = mu_df.index[mu_df.nunique(axis=1) == 1].tolist()
    raise ValueError(
        f"[{MODEL_NAME}] {n_constant} constant row(s) found (all stocks same mu).\n"
        f"  Dates: {[str(d.date()) for d in bad_dates[:5]]}"
    )
log(f"  QC4 non-const  : PASS")

log(f"\n  ML mu matrix passed all quality checks.")

# =============================================================================
# 5. HELPER FUNCTIONS
# =============================================================================

def compute_sigma(rebal_date: pd.Timestamp) -> np.ndarray:
    """Trailing ESTIM_WINDOW-day Ledoit-Wolf covariance, annualised.
    Uses only returns strictly before rebal_date - identical to step3_baseline_portfolio.py."""
    hist = returns_df.loc[returns_df.index < rebal_date].iloc[-ESTIM_WINDOW:]
    lw   = LedoitWolf().fit(hist.values)
    return lw.covariance_ * TRADING_DAYS


def solve_miqp(mu: np.ndarray, Sigma: np.ndarray) -> tuple:
    """
    MIQP via gurobipy MVar API - identical formulation to step3_baseline_portfolio.py.

    minimize  w @ Sigma @ w  -  LAMBDA * mu @ w
    s.t.      sum(w) = 1
              sum(z) = K
              W_MIN * z <= w <= W_MAX * z
              sector aggregate <= SECTOR_CAP  (per sector)

    Returns (weights | None, status_str, obj_val, solve_sec)
    """
    t0 = time.perf_counter()

    m = gp.Model(f"{MODEL_NAME}_miqp")
    m.Params.OutputFlag   = 0
    m.Params.TimeLimit    = TIME_LIMIT
    m.Params.MIPGap       = MIP_GAP
    m.Params.LogToConsole = 0

    w = m.addMVar(N, lb=0.0, ub=1.0,   name="w")
    z = m.addMVar(N, vtype=GRB.BINARY,  name="z")

    m.setObjective(w @ Sigma @ w - LAMBDA * (mu @ w), GRB.MINIMIZE)

    m.addConstr(w.sum() == 1.0,       name="budget")
    m.addConstr(z.sum() == float(K),  name="cardinality")
    m.addConstr(w >= W_MIN * z,       name="lb")
    m.addConstr(w <= W_MAX * z,       name="ub")

    for s, idx_list in sector_indices.items():
        if idx_list:
            m.addConstr(
                w[idx_list].sum() <= SECTOR_CAP,
                name=f"sector_{s[:20]}"
            )

    m.optimize()

    elapsed = time.perf_counter() - t0
    status  = m.Status

    STATUS_MAP = {
        GRB.OPTIMAL     : "OPTIMAL",
        GRB.SUBOPTIMAL  : "SUBOPTIMAL",
        GRB.TIME_LIMIT  : "TIME_LIMIT",
        GRB.INFEASIBLE  : "INFEASIBLE",
        GRB.INF_OR_UNBD : "INF_OR_UNBD",
    }
    status_str = STATUS_MAP.get(status, f"STATUS_{status}")

    if status in (GRB.OPTIMAL, GRB.SUBOPTIMAL) or \
       (status == GRB.TIME_LIMIT and m.SolCount > 0):
        weights = np.maximum(w.X, 0.0)
        if weights.sum() > 1e-8:
            weights /= weights.sum()
        else:
            weights = None
        obj_val = float(m.ObjVal)
    else:
        weights = None
        obj_val = np.nan

    m.dispose()
    return weights, status_str, obj_val, elapsed


def compute_metrics(daily_ret_df: pd.DataFrame, turnover_df: pd.DataFrame) -> dict:
    """Performance metrics - identical convention to step3_baseline_portfolio.py."""
    r = daily_ret_df["portfolio_return"].dropna().values

    ann_ret = float(r.mean() * TRADING_DAYS)
    ann_vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan

    neg_r    = r[r < 0]
    down_vol = (float(neg_r.std(ddof=1) * np.sqrt(TRADING_DAYS))
                if len(neg_r) > 1 else np.nan)
    sortino  = ann_ret / down_vol if (down_vol and down_vol > 0) else np.nan

    wealth_full      = np.concatenate([[1.0], np.exp(np.cumsum(r))])
    running_max_full = np.maximum.accumulate(wealth_full)
    drawdown_full    = wealth_full / running_max_full - 1
    max_dd           = float(abs(drawdown_full.min()))
    calmar           = ann_ret / max_dd if max_dd > 0 else np.nan

    avg_turnover = float(turnover_df["turnover"].mean()) if len(turnover_df) else np.nan

    return dict(
        r           = r,
        ann_ret     = ann_ret,
        ann_vol     = ann_vol,
        sharpe      = sharpe,
        sortino     = sortino,
        calmar      = calmar,
        max_dd      = max_dd,
        avg_turnover= avg_turnover,
        wealth      = wealth_full[1:],
        running_max = running_max_full[1:],
        drawdown    = drawdown_full[1:],
    )


def compute_net_cost(daily_ret_df: pd.DataFrame,
                     turnover_df: pd.DataFrame) -> dict:
    """Transaction-cost sensitivity at 10, 20, 30 bps - identical to baseline."""
    cost_results = {}
    for cost_bps in [10, 20, 30]:
        r_net = daily_ret_df["portfolio_return"].copy()
        for _, trow in turnover_df.iterrows():
            rebal_ts       = pd.Timestamp(trow["rebal_date"])
            first_day_mask = r_net.index > rebal_ts
            if first_day_mask.any():
                first_day = r_net.index[first_day_mask][0]
                r_net.loc[first_day] -= float(trow["turnover"]) * cost_bps / 10_000
        rn           = r_net.dropna().values
        net_ann_ret  = float(rn.mean() * TRADING_DAYS)
        net_ann_vol  = float(rn.std(ddof=1) * np.sqrt(TRADING_DAYS))
        net_sharpe   = net_ann_ret / net_ann_vol if net_ann_vol > 0 else np.nan
        net_neg_r    = rn[rn < 0]
        net_down_vol = (float(net_neg_r.std(ddof=1) * np.sqrt(TRADING_DAYS))
                        if len(net_neg_r) > 1 else np.nan)
        net_sortino  = (net_ann_ret / net_down_vol
                        if (net_down_vol and net_down_vol > 0) else np.nan)
        net_wf       = np.concatenate([[1.0], np.exp(np.cumsum(rn))])
        net_max_dd   = float(abs((net_wf / np.maximum.accumulate(net_wf) - 1).min()))
        net_calmar   = net_ann_ret / net_max_dd if net_max_dd > 0 else np.nan
        cost_results[cost_bps] = {
            "cost_bps"    : cost_bps,
            "ann_return"  : round(net_ann_ret,  4),
            "ann_vol"     : round(net_ann_vol,  4),
            "sharpe"      : round(net_sharpe,   4),
            "sortino"     : round(net_sortino,  4),
            "calmar"      : round(net_calmar,   4),
            "max_drawdown": round(net_max_dd,   4),
        }
    return cost_results


# =============================================================================
# 6. OPTIMIZATION LOOP
# =============================================================================
log("\n-- 6. OPTIMIZATION LOOP ---------------------------------------------")
log(f"  Objective  : minimize w'Sigmaw - lambda*mu'w  (lambda={LAMBDA})")
log(f"  K={K}, win[{W_MIN:.0%},{W_MAX:.0%}], sector cap={SECTOR_CAP:.0%}")
log(f"  mu source   : {MODEL_NAME} ML predictions from step5 (not recomputed here)")
log(f"  Sigma source   : rolling Ledoit-Wolf, trailing {ESTIM_WINDOW} daily returns "
    f"strictly before each rebal date")
log(f"  Gurobi     : TimeLimit={TIME_LIMIT}s, MIPGap={MIP_GAP}")
log("-" * 70)

weights_all   = {}
prev_weights  = None
total_solve_t = 0.0

for rebal_date in rebal_dates:
    hist_available = returns_df.loc[returns_df.index < rebal_date]
    if len(hist_available) < ESTIM_WINDOW:
        log(f"  {rebal_date.date()}  SKIP - {len(hist_available)} history rows "
            f"< {ESTIM_WINDOW} required for Ledoit-Wolf")
        if prev_weights is not None:
            weights_all[rebal_date] = prev_weights.copy()
        continue

    # mu: row from ML matrix (column order guaranteed identical to universe by QC2)
    mu_row  = mu_df.loc[rebal_date].values.astype(float)
    Sigma_t = compute_sigma(rebal_date)

    # PSD correction - identical to step3_baseline_portfolio.py
    eig_min = float(np.linalg.eigvalsh(Sigma_t).min())
    if eig_min < -1e-6:
        Sigma_t += (abs(eig_min) + 1e-6) * np.eye(N)

    weights, status_str, obj_val, elapsed = solve_miqp(mu_row, Sigma_t)
    total_solve_t += elapsed

    if weights is None:
        if prev_weights is None:
            raise RuntimeError(
                f"[{MODEL_NAME}] Optimization FAILED on the first rebalancing date "
                f"({rebal_date.date()}) with status '{status_str}' and no previous "
                f"portfolio to carry forward.  Check solver licence and data."
            )
        log(f"  {rebal_date.date()}  FAILED ({status_str}, {elapsed:.1f}s) "
            "- carrying forward previous weights")
        weights = prev_weights.copy()

    weights_all[rebal_date] = weights
    prev_weights = weights.copy()

    # Per-rebalancing log
    selected = sorted(
        [(universe[i], weights[i]) for i in range(N) if weights[i] > 1e-4],
        key=lambda x: -x[1]
    )
    sector_alloc = {}
    for s, idx_list in sector_indices.items():
        sw = float(sum(weights[i] for i in idx_list))
        if sw > 1e-4:
            sector_alloc[s] = sw
    top4_sec = sorted(sector_alloc.items(), key=lambda x: -x[1])[:4]
    sec_str  = "  ".join(f"{s[:14]}:{v:.1%}" for s, v in top4_sec)

    log(f"\n  {rebal_date.date()}  [{status_str:<12}] "
        f"n={len(selected)}  obj={obj_val:.6f}  t={elapsed:.1f}s")
    log(f"  Sectors : {sec_str}")
    log(f"  {'Ticker':<30} {'Weight':>8}")
    log("  " + "-" * 40)
    for ticker, wt in selected:
        log(f"  {ticker:<30} {wt:>8.4f}")

log("\n" + "-" * 70)
log(f"  Rebalancing periods solved : {len(weights_all)}")
log(f"  Total Gurobi solve time    : {total_solve_t:.1f}s  "
    f"({total_solve_t / max(len(weights_all), 1):.1f}s avg)")

rebal_list = sorted(weights_all.keys())

# =============================================================================
# 7. QUALITY CHECKS ON OPTIMIZED WEIGHTS
# =============================================================================
log("\n-- 7. QUALITY CHECKS ON OPTIMIZED WEIGHTS ---------------------------")

# QC 5 - row count must match baseline_weights
expected_n_rebal = baseline_weights.shape[0]
if len(rebal_list) != expected_n_rebal:
    raise ValueError(
        f"[{MODEL_NAME}] QC5 FAIL: expected {expected_n_rebal} rebalancing rows, "
        f"got {len(rebal_list)}."
    )
log(f"  QC5 shape          : PASS - {len(rebal_list)} rows x {N} cols")

# QC 6 - per-date constraints
violations = []
for rd in rebal_list:
    w = weights_all[rd]
    wsum = float(w.sum())
    if abs(wsum - 1.0) > 1e-4:
        violations.append(f"{rd.date()} | sum={wsum:.6f} (expected 1.0)")
    n_sel = int((w > 1e-4).sum())
    if n_sel != K:
        violations.append(f"{rd.date()} | selected K={n_sel} (expected {K})")
    w_sel = w[w > 1e-4]
    if len(w_sel) > 0:
        if w_sel.min() < W_MIN - 1e-4:
            violations.append(f"{rd.date()} | w_min={w_sel.min():.6f} < {W_MIN}")
        if w_sel.max() > W_MAX + 1e-4:
            violations.append(f"{rd.date()} | w_max={w_sel.max():.6f} > {W_MAX}")
    for s, idx_list in sector_indices.items():
        sw = float(sum(w[i] for i in idx_list))
        if sw > SECTOR_CAP + 1e-4:
            violations.append(
                f"{rd.date()} | sector '{s}' weight={sw:.6f} > {SECTOR_CAP}")

if violations:
    msg = "\n  ".join(violations[:20])
    raise ValueError(
        f"[{MODEL_NAME}] QC6 FAIL - {len(violations)} constraint violation(s):\n"
        f"  {msg}"
    )
log(f"  QC6 constraints    : PASS - all {len(rebal_list)} dates satisfy "
    f"sum=1, K={K}, W_MIN/W_MAX, sector cap")

# QC 7 - column order
if universe != list(baseline_weights.columns):
    raise ValueError(
        f"[{MODEL_NAME}] QC7 FAIL: output universe column order does not match "
        f"baseline_weights.columns.\n"
        f"  First 5 expected : {list(baseline_weights.columns)[:5]}\n"
        f"  First 5 actual   : {universe[:5]}"
    )
log(f"  QC7 column order   : PASS - matches baseline_weights.columns exactly")

# =============================================================================
# 8. DAILY PORTFOLIO RETURNS (buy-and-hold drift - identical to baseline)
# =============================================================================
log("\n-- 8. DAILY PORTFOLIO RETURNS (buy-and-hold drift) ------------------")
log("  Weights set to optimised target at each rebal date.")
log("  Rebal-date return excluded; weights drift with asset prices inside period.")

daily_records            = []
drifted_weights_by_rebal = {}

for i, rebal_date in enumerate(rebal_list):
    next_rebal = (rebal_list[i + 1]
                  if i + 1 < len(rebal_list)
                  else test_idx[-1] + pd.Timedelta(days=1))

    mask = (
        (returns_df.index >  rebal_date) &
        (returns_df.index <  next_rebal)  &
        (returns_df.index >= TEST_START)  &
        (returns_df.index <= TEST_END)
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
            "date"             : date,
            "portfolio_return" : float(np.log(gross_ret)),
        })
        w_drift = w_drift * np.exp(asset_log_ret) / gross_ret

    drifted_weights_by_rebal[rebal_date] = w_drift.copy()

daily_ret_df = (
    pd.DataFrame(daily_records)
    .set_index("date")
    .sort_index()
)

log(f"  Daily observations : {len(daily_ret_df)}")
log(f"  Date range         : {daily_ret_df.index[0].date()} "
    f"to {daily_ret_df.index[-1].date()}")
cum = float(daily_ret_df["portfolio_return"].cumsum().iloc[-1])
log(f"  Cumulative log-ret : {cum:.4f}  ({np.expm1(cum):.2%} simple return)")

# =============================================================================
# 9. TURNOVER - 0.5 * sum|w_target - w_pre_trade|  (first rebal excluded)
# =============================================================================
log("\n-- 9. TURNOVER ------------------------------------------------------")
log("  Turnover = 0.5 * sum|w_target_t - w_pre_trade_t|")
log("  w_pre_trade_t = end-of-period drifted weight from previous holding period.")
log("  First rebalance excluded (no prior portfolio).")

turnover_records = []
for i in range(1, len(rebal_list)):
    prev_date   = rebal_list[i - 1]
    curr_date   = rebal_list[i]
    w_pre_trade = drifted_weights_by_rebal[prev_date]
    w_target    = weights_all[curr_date]
    to          = float(np.abs(w_target - w_pre_trade).sum()) / 2
    turnover_records.append({
        "rebal_date"    : curr_date.date().isoformat(),
        "turnover"      : round(to, 6),
        "n_stocks_prev" : int((w_pre_trade > 1e-4).sum()),
        "n_stocks_curr" : int((w_target    > 1e-4).sum()),
    })

turnover_df  = pd.DataFrame(turnover_records)
avg_turnover = float(turnover_df["turnover"].mean()) if len(turnover_df) else np.nan

log(f"  Average monthly turnover : {avg_turnover:.4f}  ({avg_turnover:.2%})")
log(f"  Min / Max                : {turnover_df['turnover'].min():.4f} / "
    f"{turnover_df['turnover'].max():.4f}")

# =============================================================================
# 10. PERFORMANCE METRICS
# =============================================================================
log("\n-- 10. PERFORMANCE METRICS ------------------------------------------")

met = compute_metrics(daily_ret_df, turnover_df)
ann_ret     = met["ann_ret"]
ann_vol     = met["ann_vol"]
sharpe      = met["sharpe"]
sortino     = met["sortino"]
calmar      = met["calmar"]
max_dd      = met["max_dd"]
wealth      = met["wealth"]
running_max = met["running_max"]
drawdown    = met["drawdown"]
r           = met["r"]

log(f"\n  Annualised Return    : {ann_ret:.4f}  ({ann_ret:.2%})")
log(f"  Annualised Vol       : {ann_vol:.4f}  ({ann_vol:.2%})")
log(f"  Sharpe Ratio         : {sharpe:.4f}")
log(f"  Sortino Ratio        : {sortino:.4f}")
log(f"  Calmar Ratio         : {calmar:.4f}")
log(f"  Maximum Drawdown     : {max_dd:.4f}  ({max_dd:.2%})")
log(f"  Avg Monthly Turnover : {avg_turnover:.4f}  ({avg_turnover:.2%})")

log(f"\n  Per-year breakdown:")
log(f"  {'Year':<6} {'Ann Ret':>10} {'Ann Vol':>10} {'Sharpe':>10} "
    f"{'Max DD':>10} {'Days':>6}")
log("  " + "-" * 55)
for yr in sorted(set(daily_ret_df.index.year)):
    r_yr = daily_ret_df.loc[
        daily_ret_df.index.year == yr, "portfolio_return"
    ].values
    if len(r_yr) < 2:
        continue
    ar  = r_yr.mean() * TRADING_DAYS
    av  = r_yr.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sh  = ar / av if av > 0 else np.nan
    wyf = np.concatenate([[1.0], np.exp(np.cumsum(r_yr))])
    dd  = float(abs((wyf / np.maximum.accumulate(wyf) - 1).min()))
    log(f"  {yr:<6} {ar:>9.2%}  {av:>9.2%}  {sh:>9.4f}  {dd:>9.2%}  {len(r_yr):>5}")

log(f"\n  Average sector weights (test period):")
log(f"  {'Sector':<42} {'Avg Weight':>12}")
log("  " + "-" * 56)
for s in sectors:
    idx_s = sector_indices[s]
    if not idx_s:
        continue
    avg_w = float(np.mean([weights_all[d][idx_s].sum() for d in rebal_list]))
    if avg_w > 0.001:
        log(f"  {s:<42} {avg_w:>11.2%}")

# Wealth / drawdown peak-trough diagnostics
r_index    = daily_ret_df["portfolio_return"].dropna().index
trough_idx = int(drawdown.argmin())
peak_idx   = int(np.argmax(wealth[:trough_idx + 1]))
log(f"\n  Final wealth index : {float(wealth[-1]):.6f}")
log(f"  Peak date          : {r_index[peak_idx].date()}  "
    f"(wealth = {float(wealth[peak_idx]):.6f})")
log(f"  Trough date        : {r_index[trough_idx].date()}  "
    f"(wealth = {float(wealth[trough_idx]):.6f})")

# =============================================================================
# 11. TRANSACTION-COST SENSITIVITY
# =============================================================================
log("\n-- 11. TRANSACTION-COST SENSITIVITY (gross -> net) -------------------")
log("  One-way cost applied to traded turnover on the first investable day")
log("  after each rebalancing date, starting from the 2nd rebalance.")

cost_results = compute_net_cost(daily_ret_df, turnover_df)

log(f"\n  {'Scenario':<22} {'Ann Ret':>9} {'Ann Vol':>9} {'Sharpe':>9} {'Max DD':>9}")
log("  " + "-" * 63)
log(f"  {'Gross (0 bps)':<22} "
    f"{ann_ret:>8.2%}  {ann_vol:>8.2%}  {sharpe:>8.4f}  {max_dd:>8.2%}")
for bps, res in cost_results.items():
    log(f"  {f'Net ({bps} bps)':<22} "
        f"{res['ann_return']:>8.2%}  {res['ann_vol']:>8.2%}  "
        f"{res['sharpe']:>8.4f}  {res['max_drawdown']:>8.2%}")

# =============================================================================
# 12. SAVE OUTPUTS
# =============================================================================
log("\n-- 12. SAVING OUTPUTS -----------------------------------------------")

# {MODEL_NAME}_weights.csv - column order matches baseline_weights.csv exactly
rows_w = []
for date, w in weights_all.items():
    row = {"date": date.date().isoformat()}
    row.update({universe[i]: round(float(w[i]), 6) for i in range(N)})
    rows_w.append(row)

weights_out = pd.DataFrame(rows_w).set_index("date")
weights_out.to_csv(os.path.join(MODEL_STEP6_DIR, f"{MODEL_NAME}_weights.csv"))
log(f"  Saved: {MODEL_NAME}_weights.csv  "
    f"({weights_out.shape[0]} dates x {weights_out.shape[1]} tickers)")

# {MODEL_NAME}_returns.csv
daily_ret_df.to_csv(os.path.join(MODEL_STEP6_DIR, f"{MODEL_NAME}_returns.csv"))
log(f"  Saved: {MODEL_NAME}_returns.csv  ({len(daily_ret_df)} rows)")

# {MODEL_NAME}_turnover.csv
turnover_df.to_csv(
    os.path.join(MODEL_STEP6_DIR, f"{MODEL_NAME}_turnover.csv"), index=False
)
log(f"  Saved: {MODEL_NAME}_turnover.csv  ({len(turnover_df)} rows)")

# {MODEL_NAME}_summary.csv
summary_row = {
    "model"            : MODEL_NAME,
    "mu_estimator"     : f"ml_{MODEL_NAME}",
    "sigma_estimator"  : f"rolling_ledoit_wolf_{ESTIM_WINDOW}d",
    "estimation_window": ESTIM_WINDOW,
    "K"                : K,
    "lambda"           : LAMBDA,
    "w_min"            : W_MIN,
    "w_max"            : W_MAX,
    "sector_cap"       : SECTOR_CAP,
    "test_start"       : TEST_START,
    "test_end"         : TEST_END,
    "n_rebal"          : len(rebal_list),
    "n_days"           : len(daily_ret_df),
    "ann_return"       : round(ann_ret,      4),
    "ann_vol"          : round(ann_vol,      4),
    "sharpe"           : round(sharpe,       4),
    "sortino"          : round(sortino,      4),
    "calmar"           : round(calmar,       4),
    "max_drawdown"     : round(max_dd,       4),
    "avg_turnover"     : round(avg_turnover, 4),
}
pd.DataFrame([summary_row]).to_csv(
    os.path.join(MODEL_STEP6_DIR, f"{MODEL_NAME}_summary.csv"), index=False
)
log(f"  Saved: {MODEL_NAME}_summary.csv")

# {MODEL_NAME}_wealth_drawdown.csv
wd_df = pd.DataFrame({
    "portfolio_return" : r,
    "wealth_index"     : wealth,
    "running_max"      : running_max,
    "drawdown"         : drawdown,
}, index=r_index)
wd_df.index.name = "date"
wd_df.to_csv(os.path.join(MODEL_STEP6_DIR, f"{MODEL_NAME}_wealth_drawdown.csv"))
log(f"  Saved: {MODEL_NAME}_wealth_drawdown.csv  ({len(wd_df)} rows)")

# {MODEL_NAME}_net_cost_summary.csv
net_cost_df = pd.DataFrame(list(cost_results.values()))
net_cost_df.insert(0, "model", MODEL_NAME)
net_cost_df["test_start"] = TEST_START
net_cost_df["test_end"]   = TEST_END
net_cost_df.to_csv(
    os.path.join(MODEL_STEP6_DIR, f"{MODEL_NAME}_net_cost_summary.csv"), index=False
)
log(f"  Saved: {MODEL_NAME}_net_cost_summary.csv  ({len(net_cost_df)} cost scenarios)")

# =============================================================================
# 13. SAVE LOG
# =============================================================================
log_path = os.path.join(MODEL_STEP6_DIR, f"{MODEL_NAME}_portfolio_log.txt")
with open(log_path, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
print(f"  Log saved: {MODEL_NAME}_portfolio_log.txt")

# =============================================================================
# 14. FINAL SUMMARY TABLE
# =============================================================================
def _fmt(v: float, pct: bool = True) -> str:
    if not np.isfinite(v):
        return f"{'N/A':>8}"
    return f"{v:>8.2%}" if pct else f"{v:>8.4f}"

print("\n" + "=" * 70)
print(f"  STEP 6b - {MODEL_NAME.upper()} PORTFOLIO COMPLETE")
print("=" * 70)
print(f"  {'Model':<22} {'Ann Ret':>9} {'Ann Vol':>9} {'Sharpe':>8} "
      f"{'Sortino':>9} {'Calmar':>8} {'MaxDD':>8} {'AvgTO':>8}")
print("  " + "-" * 87)
print(
    f"  {MODEL_NAME:<22}"
    f" {_fmt(ann_ret)}"
    f"  {_fmt(ann_vol)}"
    f"  {_fmt(sharpe,   pct=False)}"
    f"  {_fmt(sortino,  pct=False)}"
    f"  {_fmt(calmar,   pct=False)}"
    f"  {_fmt(max_dd)}"
    f"  {_fmt(avg_turnover)}"
)
print("=" * 70)
print(f"  Output directory : data/results/step6/{MODEL_NAME}/")
print("=" * 70)
