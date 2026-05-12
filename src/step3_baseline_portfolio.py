"""
step3_baseline_portfolio.py
=================
PURPOSE : Classical Markowitz MIQP baseline portfolio - trailing historical
          mean as mu, rolling Ledoit-Wolf covariance, monthly rebalancing.

          This is the benchmark against which all ML models are compared.
          Every design choice (solver, constraints, Sigma estimator) is held
          identical in the ML step so that the only moving part is mu.

          Note: Step 2 produces full-window Ledoit-Wolf covariance files for
          diagnostic purposes only. Step 3 estimates covariance rolling inside
          the backtest loop; those Step 2 files are not loaded or used here.

INPUTS  :
  data/clean/returns.csv      -- daily log-return matrix (full 2010-2025)
  data/clean/meta_clean.csv   -- ticker metadata including GICS sector

OUTPUTS :
  data/results/step3/baseline_weights.csv          -- rebalancing dates x tickers (sparse)
  data/results/step3/baseline_returns.csv          -- daily portfolio returns over test period
  data/results/step3/baseline_turnover.csv         -- per-rebalancing-date turnover
  data/results/step3/baseline_summary.csv          -- Sharpe, Sortino, Calmar, max drawdown,
                                                      annualised return/vol, avg turnover
  data/results/step3/baseline_wealth_drawdown.csv  -- daily wealth index, running max, drawdown
  data/results/step3/equal_weight_returns.csv      -- equal-weight benchmark daily returns
  data/results/step3/equal_weight_summary.csv      -- equal-weight benchmark performance metrics
  data/results/step3/baseline_net_cost_summary.csv -- net-of-cost performance at 10/20/30 bps

DIAGNOSTICS (Sections 8b-8d):
  Implementability diagnostics used in the Sustainability dimension of the
  SAFE AI framework (Giudici 2024). They do not alter portfolio construction.
    8b - wealth/drawdown curve with peak and trough dates
    8c - equal-weight benchmark (full universe, monthly rebalanced)
    8d - transaction-cost sensitivity: net Sharpe at 10, 20, 30 bps per unit of traded turnover

METHODOLOGY:
  mu    : trailing 252-day historical mean x 252 (annualised), winsorized p1/p99
  Sigma : rolling Ledoit-Wolf, re-estimated at each rebalancing date using
          trailing ESTIM_WINDOW-day returns strictly before rebal_date.
          Both mu and Sigma use only information available before each rebalancing
          date. Step 2 full-window covariance matrices are diagnostic only
          and are not used in the OOS optimizer.
  MIQP:
    minimize  w'Sw - lambda * mu'w        (lambda = 1)
    s.t.      sum(w_i) = 1               (fully invested)
              z_i in {0,1}, sum(z_i) = K = 10   (cardinality)
              0.01*z_i <= w_i <= 0.20*z_i        (weight bounds, selected only)
              sum_{i in s} w_i <= 0.30            (sector concentration cap)
  Solver: Gurobi via gurobipy (MVar API). TimeLimit=60s, MIPGap=0.01.
  Rebalancing: monthly (first trading day of each calendar month in test window)
  Test window: 2023-01-01 to 2025-12-31

Author  : Anila Vata
Project : MSc Thesis - ML-Enhanced Portfolio Optimization with SAFE AI Evaluation
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

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DIR   = os.path.join(BASE_DIR, "data", "clean")
RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")
STEP3_DIR   = os.path.join(RESULTS_DIR, "step3")
os.makedirs(STEP3_DIR, exist_ok=True)

# ---------- optimization parameters ------------------------------------------
LAMBDA       = 1.0     # risk-return tradeoff: minimize w'Sw - lambda*mu'w
K            = 10      # cardinality: exactly K stocks selected
W_MIN        = 0.01    # minimum weight per selected stock
W_MAX        = 0.20    # maximum weight per selected stock
SECTOR_CAP   = 0.30    # maximum aggregate weight per GICS sector
TRADING_DAYS = 252     # annualisation factor
ESTIM_WINDOW = 252     # trailing days for mu and Sigma estimation
TIME_LIMIT   = 60      # Gurobi time limit per solve (seconds)
MIP_GAP      = 0.01    # Gurobi MIPGap

TEST_START   = "2023-01-01"
TEST_END     = "2025-12-31"

# Logging accumulator - written to file at the end
log_lines = []
def log(msg=""):
    print(msg)
    log_lines.append(str(msg))

log("=" * 70)
log("  STEP 3 - MARKOWITZ MIQP BASELINE PORTFOLIO")
log("=" * 70)

if not HAS_GUROBI:
    raise ImportError(
        "gurobipy is not installed or no valid licence found.\n"
        "Install with: pip install gurobipy   (requires Gurobi licence)")

log(f"  Gurobi version : {gp.gurobi.version()}")

# =============================================================================
# 1. LOAD INPUTS
# =============================================================================
log("\n-- 1. LOAD INPUTS ---------------------------------------------------")

for fname in ["returns.csv", "meta_clean.csv"]:
    path = os.path.join(CLEAN_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Run step1_load.py and step2_preprocess.py first.")

returns_df = pd.read_csv(
    os.path.join(CLEAN_DIR, "returns.csv"),
    index_col=0, parse_dates=True
).sort_index()

meta = pd.read_csv(os.path.join(CLEAN_DIR, "meta_clean.csv"))
meta.columns = meta.columns.str.strip().str.lower()
if "ticker" not in meta.columns:
    meta = meta.rename(columns={meta.columns[0]: "ticker"})
meta["ticker"] = meta["ticker"].astype(str).str.strip().str.upper()

log(f"  returns      : {returns_df.shape[0]} days x {returns_df.shape[1]} stocks")
log(f"  meta         : {meta.shape}")
log(f"  Return range : {returns_df.index[0].date()} to {returns_df.index[-1].date()}")

# =============================================================================
# 2. UNIVERSE ALIGNMENT
# =============================================================================
log("\n-- 2. UNIVERSE ALIGNMENT --------------------------------------------")

universe = sorted(
    set(returns_df.columns) & set(meta["ticker"])
)
N = len(universe)

log(f"  returns tickers  : {len(returns_df.columns)}")
log(f"  meta tickers     : {len(meta['ticker'])}")
log(f"  Intersection     : {N} stocks  (this is the optimization universe)")

if N < K:
    raise ValueError(
        f"Universe has only {N} stocks but cardinality K={K} requires at least K.")

returns_df = returns_df[universe]

# Sector mapping
meta_idx   = meta.set_index("ticker")
sector_col = next(
    (c for c in ["sector", "gics_sector", "gics sector"] if c in meta_idx.columns),
    meta_idx.columns[0]
)
sector_map     = {t: str(meta_idx.loc[t, sector_col]).strip()
                  if t in meta_idx.index else "Unknown"
                  for t in universe}
sectors        = sorted(set(sector_map.values()))
sector_indices = {s: [i for i, t in enumerate(universe) if sector_map[t] == s]
                  for s in sectors}

log(f"\n  Sectors ({len(sectors)}):")
for s in sectors:
    log(f"    {s:<42} {len(sector_indices[s]):>3} stocks")

# =============================================================================
# 3. REBALANCING DATES
# =============================================================================
log("\n-- 3. REBALANCING DATES ---------------------------------------------")

test_idx = returns_df.index[
    (returns_df.index >= TEST_START) & (returns_df.index <= TEST_END)
]
rebal_dates = (
    pd.Series(test_idx)
    .groupby([test_idx.year, test_idx.month])
    .first()
    .values
)
rebal_dates = pd.DatetimeIndex(rebal_dates)

log(f"  Test window    : {TEST_START} to {TEST_END}")
log(f"  Rebalancing    : first trading day of each calendar month")
log(f"  Rebal periods  : {len(rebal_dates)}"
    f"  ({rebal_dates[0].date()} to {rebal_dates[-1].date()})")

# =============================================================================
# 4. HELPER FUNCTIONS
# =============================================================================

def compute_mu(rebal_date: pd.Timestamp) -> np.ndarray:
    """
    Trailing ESTIM_WINDOW-day annualised mean, winsorized cross-sectionally
    at p1 / p99.  Only uses data strictly before rebal_date.
    """
    hist = returns_df.loc[returns_df.index < rebal_date].iloc[-ESTIM_WINDOW:]
    raw  = hist.mean().values * TRADING_DAYS
    lo   = np.nanpercentile(raw, 1)
    hi   = np.nanpercentile(raw, 99)
    return np.clip(raw, lo, hi)


def compute_sigma(rebal_date: pd.Timestamp) -> np.ndarray:
    """
    Trailing ESTIM_WINDOW-day Ledoit-Wolf covariance, annualised.
    Only uses returns strictly before rebal_date.
    """
    hist = returns_df.loc[returns_df.index < rebal_date].iloc[-ESTIM_WINDOW:]
    lw = LedoitWolf().fit(hist.values)
    return lw.covariance_ * TRADING_DAYS


def solve_miqp(mu: np.ndarray, Sigma: np.ndarray) -> tuple:
    """
    Solve the MIQP via gurobipy MVar API.

    minimize  w @ Sigma @ w  -  LAMBDA * mu @ w
    s.t.      sum(w) = 1
              sum(z) = K
              W_MIN * z <= w <= W_MAX * z   (element-wise)
              sector aggregate <= SECTOR_CAP  (for each sector)

    Returns (weights: np.ndarray | None, status: str, obj_val: float, solve_sec: float)
    """
    t0 = time.perf_counter()

    m = gp.Model("baseline_miqp")
    m.Params.OutputFlag  = 0           # suppress Gurobi console output
    m.Params.TimeLimit   = TIME_LIMIT
    m.Params.MIPGap      = MIP_GAP
    m.Params.LogToConsole = 0

    # Decision variables
    w = m.addMVar(N, lb=0.0, ub=1.0,         name="w")
    z = m.addMVar(N, vtype=GRB.BINARY,        name="z")

    # Objective: minimize w'Sw - lambda*mu'w
    m.setObjective(w @ Sigma @ w - LAMBDA * (mu @ w), GRB.MINIMIZE)

    # Constraints
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
            weights /= weights.sum()   # renormalize to exactly 1
        else:
            weights = None
        obj_val = float(m.ObjVal)
    else:
        weights = None
        obj_val = np.nan

    m.dispose()
    return weights, status_str, obj_val, elapsed


# =============================================================================
# 5. OPTIMIZATION LOOP
# =============================================================================
log("\n-- 5. OPTIMIZATION LOOP ---------------------------------------------")
log(f"  Objective  : minimize w'Sw - lambda*mu'w  (lambda={LAMBDA})")
log(f"  K={K}, w in [{W_MIN:.0%},{W_MAX:.0%}], sector cap={SECTOR_CAP:.0%}")
log(f"  mu estimator: trailing {ESTIM_WINDOW}-day mean x {TRADING_DAYS}, winsorized p1/p99")
log(f"  Sigma estimator: rolling Ledoit-Wolf, re-estimated at each rebalancing date"
    f" using trailing {ESTIM_WINDOW} returns (same window as mu)")
log(f"  Gurobi     : TimeLimit={TIME_LIMIT}s, MIPGap={MIP_GAP}")
log(f"  Out-of-sample design: both mu and Sigma are re-estimated at each rebalance"
    f" date using only past returns, following the rolling-window evaluation logic"
    f" of DeMiguel, Garlappi and Uppal (2009).")
log("-" * 70)

weights_all   = {}   # rebal_date -> np.ndarray (N,)
prev_weights  = None
total_solve_t = 0.0

for rebal_date in rebal_dates:
    hist_available = returns_df.loc[returns_df.index < rebal_date]
    if len(hist_available) < ESTIM_WINDOW:
        log(f"  {rebal_date.date()}  SKIP - {len(hist_available)} history days "
            f"< {ESTIM_WINDOW} required")
        if prev_weights is not None:
            weights_all[rebal_date] = prev_weights.copy()
        continue

    mu_vec  = compute_mu(rebal_date)
    Sigma_t = compute_sigma(rebal_date)

    # PSD correction per-period if needed
    eig_min = float(np.linalg.eigvalsh(Sigma_t).min())
    if eig_min < -1e-6:
        Sigma_t += (abs(eig_min) + 1e-6) * np.eye(N)

    weights, status_str, obj_val, elapsed = solve_miqp(mu_vec, Sigma_t)
    total_solve_t += elapsed

    # Carry forward previous weights on solver failure
    if weights is None:
        log(f"  {rebal_date.date()}  FAILED ({status_str}, {elapsed:.1f}s) "
            "- carrying forward previous weights")
        weights = prev_weights.copy() if prev_weights is not None else np.ones(N) / N

    weights_all[rebal_date] = weights
    prev_weights = weights.copy()

    # --- Per-rebalancing log --------------------------------------------------
    selected   = [(universe[i], weights[i]) for i in range(N) if weights[i] > 1e-4]
    selected   = sorted(selected, key=lambda x: -x[1])
    n_selected = len(selected)

    # Sector allocation
    sector_alloc = {}
    for s, idx_list in sector_indices.items():
        sw = float(sum(weights[i] for i in idx_list))
        if sw > 1e-4:
            sector_alloc[s] = sw

    top4_sec = sorted(sector_alloc.items(), key=lambda x: -x[1])[:4]
    sec_str  = "  ".join(f"{s[:14]}:{v:.1%}" for s, v in top4_sec)

    log(f"\n  {rebal_date.date()}  [{status_str:<12}] "
        f"n={n_selected}  obj={obj_val:.6f}  t={elapsed:.1f}s")
    log(f"  Sectors : {sec_str}")
    log(f"  {'Ticker':<30} {'Weight':>8}")
    log("  " + "-" * 40)
    for ticker, wt in selected:
        log(f"  {ticker:<30} {wt:>8.4f}")

log("\n" + "-" * 70)
log(f"\n  Rebalancing periods solved : {len(weights_all)}")
log(f"  Total Gurobi solve time    : {total_solve_t:.1f}s  "
    f"({total_solve_t / max(len(weights_all), 1):.1f}s avg)")

# =============================================================================
# 6. COMPUTE DAILY PORTFOLIO RETURNS (monthly rebalanced, buy-and-hold drift)
# =============================================================================
log("\n-- 6. DAILY PORTFOLIO RETURNS (monthly rebalanced, buy-and-hold drift) --")
log("  At each rebalance date, weights are set to the optimised target.")
log("  Inside the holding period, weights drift with asset returns (no daily")
log("  rebalancing). Portfolio log-return = log(w_drift @ exp(asset_log_returns)).")

rebal_list              = sorted(weights_all.keys())
daily_records           = []
drifted_weights_by_rebal = {}   # rebal_date -> end-of-period drifted weights

for i, rebal_date in enumerate(rebal_list):
    next_rebal = (rebal_list[i + 1]
                  if i + 1 < len(rebal_list)
                  else test_idx[-1] + pd.Timedelta(days=1))

    mask = (
        (returns_df.index > rebal_date) &   # returns are close-to-close; rebalance-date return excluded
        (returns_df.index <  next_rebal) &
        (returns_df.index >= TEST_START) &
        (returns_df.index <= TEST_END)
    )
    period_ret = returns_df.loc[mask]
    if period_ret.empty:
        drifted_weights_by_rebal[rebal_date] = weights_all[rebal_date].copy()
        continue

    # Initialise drifted weights to target weights at the rebalance date
    w_drift = weights_all[rebal_date].copy()

    for date, row in period_ret.iterrows():
        asset_log_ret = row.values
        gross_ret     = float(w_drift @ np.exp(asset_log_ret))
        daily_records.append({
            "date"             : date,
            "portfolio_return" : float(np.log(gross_ret)),
        })
        # Update drifted weights: each stock's weight grows proportionally to its gross return
        w_drift = w_drift * np.exp(asset_log_ret) / gross_ret

    drifted_weights_by_rebal[rebal_date] = w_drift.copy()

daily_ret_df = (
    pd.DataFrame(daily_records)
    .set_index("date")
    .sort_index()
)

log(f"  Daily observations : {len(daily_ret_df)}")
log(f"  Date range         : {daily_ret_df.index[0].date()} to {daily_ret_df.index[-1].date()}")
cum = float(daily_ret_df["portfolio_return"].cumsum().iloc[-1])
log(f"  Cumulative log-ret : {cum:.4f}  ({np.expm1(cum):.2%} simple return)")

# =============================================================================
# 7. TURNOVER (pre-trade drifted weights -> new target weights)
# =============================================================================
log("\n-- 7. TURNOVER (pre-trade drifted weights -> new target weights) -----")
log("  Turnover = 0.5 * sum|w_target_t - w_pre_trade_t|")
log("  w_pre_trade_t is the end-of-period drifted weight from the previous")
log("  holding period, immediately before the current rebalance trade.")
log("  First rebalance is excluded (no previous portfolio).")

turnover_records = []
for i in range(1, len(rebal_list)):
    prev_date   = rebal_list[i - 1]
    curr_date   = rebal_list[i]
    w_pre_trade = drifted_weights_by_rebal[prev_date]   # drifted from previous period
    w_target    = weights_all[curr_date]                 # new optimised target
    to          = float(np.abs(w_target - w_pre_trade).sum()) / 2
    turnover_records.append({
        "rebal_date"    : curr_date.date().isoformat(),
        "turnover"      : round(to, 6),
        "n_stocks_prev" : int((w_pre_trade > 1e-4).sum()),
        "n_stocks_curr" : int((w_target    > 1e-4).sum()),
    })

turnover_df = pd.DataFrame(turnover_records)
avg_turnover = float(turnover_df["turnover"].mean()) if len(turnover_df) else np.nan

log(f"  Average monthly turnover : {avg_turnover:.4f}  ({avg_turnover:.2%})")
log(f"  Min / Max                : {turnover_df['turnover'].min():.4f} / "
    f"{turnover_df['turnover'].max():.4f}")

# =============================================================================
# 8. PERFORMANCE METRICS
# =============================================================================
log("\n-- 8. PERFORMANCE METRICS ------------------------------------------")

r = daily_ret_df["portfolio_return"].dropna().values

ann_ret = float(r.mean() * TRADING_DAYS)
ann_vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan

neg_r    = r[r < 0]
down_vol = float(neg_r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(neg_r) > 1 else np.nan
sortino  = ann_ret / down_vol if (down_vol and down_vol > 0) else np.nan

# Wealth series starting from 1.0 so drawdown from the very first day is captured
wealth_full      = np.concatenate([[1.0], np.exp(np.cumsum(r))])
running_max_full = np.maximum.accumulate(wealth_full)
drawdown_full    = wealth_full / running_max_full - 1   # <= 0 by construction
max_dd           = float(abs(drawdown_full.min()))       # positive loss percentage
calmar           = ann_ret / max_dd if max_dd > 0 else np.nan

# Slice to real-date-aligned series (saved to CSV and used in diagnostics)
wealth      = wealth_full[1:]
running_max = running_max_full[1:]
drawdown    = drawdown_full[1:]

log(f"\n  Annualised Return    : {ann_ret:.4f}  ({ann_ret:.2%})")
log(f"  Annualised Vol       : {ann_vol:.4f}  ({ann_vol:.2%})")
log(f"  Sharpe Ratio         : {sharpe:.4f}")
log(f"  Sortino Ratio        : {sortino:.4f}")
log(f"  Calmar Ratio         : {calmar:.4f}")
log(f"  Maximum Drawdown     : {max_dd:.4f}  ({max_dd:.2%})")
log(f"  Avg Monthly Turnover : {avg_turnover:.4f}  ({avg_turnover:.2%})")

log(f"\n  Per-year breakdown:")
log(f"  {'Year':<6} {'Ann Ret':>10} {'Ann Vol':>10} {'Sharpe':>10} {'Max DD':>10} {'Days':>6}")
log("  " + "-" * 55)
for yr in sorted(set(daily_ret_df.index.year)):
    r_yr = daily_ret_df.loc[daily_ret_df.index.year == yr, "portfolio_return"].values
    if len(r_yr) < 2:
        continue
    ar   = r_yr.mean() * TRADING_DAYS
    av   = r_yr.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sh   = ar / av if av > 0 else np.nan
    wealth_yr_full = np.concatenate([[1.0], np.exp(np.cumsum(r_yr))])
    dd_yr          = float(abs((wealth_yr_full / np.maximum.accumulate(wealth_yr_full) - 1).min()))
    log(f"  {yr:<6} {ar:>9.2%}  {av:>9.2%}  {sh:>9.4f}  {dd_yr:>9.2%}  {len(r_yr):>5}")

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

# =============================================================================
# 8b. WEALTH / DRAWDOWN DIAGNOSTICS
# =============================================================================
log("\n-- 8b. WEALTH / DRAWDOWN DIAGNOSTICS --------------------------------")

r_index      = daily_ret_df["portfolio_return"].dropna().index
trough_idx   = int(drawdown.argmin())
trough_date  = r_index[trough_idx]
peak_idx     = int(np.argmax(wealth[:trough_idx + 1]))
peak_date    = r_index[peak_idx]
final_wealth      = float(wealth[-1])
wealth_at_peak    = float(wealth[peak_idx])
wealth_at_trough  = float(wealth[trough_idx])

log(f"  Final wealth index       : {final_wealth:.6f}")
log(f"  Maximum Drawdown         : {max_dd:.4f}  ({max_dd:.2%})")
log(f"  Previous peak date       : {peak_date.date()}  "
    f"(wealth = {wealth_at_peak:.6f})")
log(f"  Trough date              : {trough_date.date()}  "
    f"(wealth = {wealth_at_trough:.6f})")

# =============================================================================
# 8c. EQUAL-WEIGHT BENCHMARK (monthly rebalanced, buy-and-hold drift)
# =============================================================================
log("\n-- 8c. EQUAL-WEIGHT BENCHMARK (monthly rebalanced, buy-and-hold drift) -")
log(f"  Universe   : {N} stocks, weight = 1/{N} per stock at each rebalance date")
log(f"  Rebalancing: monthly (same dates as baseline); weights drift between")
log(f"  rebalancing dates - identical convention to the baseline portfolio.")

ew_records = []

for i, rebal_date in enumerate(rebal_list):
    next_rebal = (rebal_list[i + 1]
                  if i + 1 < len(rebal_list)
                  else test_idx[-1] + pd.Timedelta(days=1))
    mask_ew = (
        (returns_df.index > rebal_date) &
        (returns_df.index <  next_rebal) &
        (returns_df.index >= TEST_START) &
        (returns_df.index <= TEST_END)
    )
    period_ret_ew = returns_df.loc[mask_ew]
    if period_ret_ew.empty:
        continue

    # Reset to equal weight at each rebalance; let drift inside the month
    ew_w_drift = np.ones(N) / N

    for date, row in period_ret_ew.iterrows():
        asset_log_ret = row.values
        ew_gross      = float(ew_w_drift @ np.exp(asset_log_ret))
        ew_records.append({
            "date"             : date,
            "portfolio_return" : float(np.log(ew_gross)),
        })
        ew_w_drift = ew_w_drift * np.exp(asset_log_ret) / ew_gross

ew_ret_df = (
    pd.DataFrame(ew_records)
    .set_index("date")
    .sort_index()
)

r_ew        = ew_ret_df["portfolio_return"].dropna().values
ew_ann_ret  = float(r_ew.mean() * TRADING_DAYS)
ew_ann_vol  = float(r_ew.std(ddof=1) * np.sqrt(TRADING_DAYS))
ew_sharpe   = ew_ann_ret / ew_ann_vol if ew_ann_vol > 0 else np.nan
ew_neg_r    = r_ew[r_ew < 0]
ew_down_vol = float(ew_neg_r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(ew_neg_r) > 1 else np.nan
ew_sortino  = ew_ann_ret / ew_down_vol if (ew_down_vol and ew_down_vol > 0) else np.nan
ew_wealth_full = np.concatenate([[1.0], np.exp(np.cumsum(r_ew))])
ew_max_dd      = float(abs((ew_wealth_full / np.maximum.accumulate(ew_wealth_full) - 1).min()))
ew_calmar      = ew_ann_ret / ew_max_dd if ew_max_dd > 0 else np.nan

log(f"  Daily obs         : {len(ew_ret_df)}")
log(f"  Annualised Return : {ew_ann_ret:.2%}")
log(f"  Annualised Vol    : {ew_ann_vol:.2%}")
log(f"  Sharpe Ratio      : {ew_sharpe:.4f}")
log(f"  Sortino Ratio     : {ew_sortino:.4f}")
log(f"  Calmar Ratio      : {ew_calmar:.4f}")
log(f"  Maximum Drawdown  : {ew_max_dd:.2%}")

# =============================================================================
# 8d. TRANSACTION-COST SENSITIVITY
# =============================================================================
log("\n-- 8d. TRANSACTION-COST SENSITIVITY (gross -> net) ------------------")
log("  Cost applied as one-way transaction cost on traded turnover")
log("  on the first investable day after each rebalancing date")
log("  starting from the 2nd rebalance.")

cost_results = {}
for cost_bps in [10, 20, 30]:
    r_net = daily_ret_df["portfolio_return"].copy()
    for _, trow in turnover_df.iterrows():
        rebal_ts        = pd.Timestamp(trow["rebal_date"])
        first_day_mask  = r_net.index > rebal_ts
        if first_day_mask.any():
            first_day = r_net.index[first_day_mask][0]
            r_net.loc[first_day] -= float(trow["turnover"]) * cost_bps / 10_000
    rn           = r_net.dropna().values
    net_ann_ret  = float(rn.mean() * TRADING_DAYS)
    net_ann_vol  = float(rn.std(ddof=1) * np.sqrt(TRADING_DAYS))
    net_sharpe   = net_ann_ret / net_ann_vol if net_ann_vol > 0 else np.nan
    net_neg_r    = rn[rn < 0]
    net_down_vol = float(net_neg_r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(net_neg_r) > 1 else np.nan
    net_sortino  = net_ann_ret / net_down_vol if (net_down_vol and net_down_vol > 0) else np.nan
    net_wealth_full = np.concatenate([[1.0], np.exp(np.cumsum(rn))])
    net_max_dd   = float(abs((net_wealth_full / np.maximum.accumulate(net_wealth_full) - 1).min()))
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

log(f"\n  {'Scenario':<22} {'Ann Ret':>9} {'Ann Vol':>9} {'Sharpe':>9} {'Max DD':>9}")
log("  " + "-" * 63)
log(f"  {'Gross (0 bps)':<22} {ann_ret:>8.2%}  {ann_vol:>8.2%}  {sharpe:>8.4f}  {max_dd:>8.2%}")
for bps, res in cost_results.items():
    log(f"  {f'Net ({bps} bps)':<22} "
        f"{res['ann_return']:>8.2%}  {res['ann_vol']:>8.2%}  "
        f"{res['sharpe']:>8.4f}  {res['max_drawdown']:>8.2%}")
log(f"  {'Equal Weight':<22} {ew_ann_ret:>8.2%}  {ew_ann_vol:>8.2%}  "
    f"{ew_sharpe:>8.4f}  {ew_max_dd:>8.2%}")

# =============================================================================
# 9. SAVE OUTPUTS
# =============================================================================
log("\n-- 9. SAVING OUTPUTS -----------------------------------------------")

# baseline_weights.csv - wide format, one row per rebalancing date
rows_w = []
for date, w in weights_all.items():
    row = {"date": date.date().isoformat()}
    row.update({universe[i]: round(float(w[i]), 6) for i in range(N)})
    rows_w.append(row)

weights_df = pd.DataFrame(rows_w).set_index("date")
weights_df.to_csv(os.path.join(STEP3_DIR, "baseline_weights.csv"))
log(f"  Saved: baseline_weights.csv  "
    f"({weights_df.shape[0]} dates x {weights_df.shape[1]} tickers)")

# baseline_returns.csv - daily portfolio returns
daily_ret_df.to_csv(os.path.join(STEP3_DIR, "baseline_returns.csv"))
log(f"  Saved: baseline_returns.csv  ({len(daily_ret_df)} rows)")

# baseline_turnover.csv
turnover_df.to_csv(
    os.path.join(STEP3_DIR, "baseline_turnover.csv"), index=False
)
log(f"  Saved: baseline_turnover.csv  ({len(turnover_df)} rows)")

# baseline_summary.csv
summary_df = pd.DataFrame([{
    "model"           : "baseline_markowitz",
    "mu_estimator"    : f"trailing_{ESTIM_WINDOW}d_mean_winsorized",
    "sigma_estimator" : f"rolling_ledoit_wolf_{ESTIM_WINDOW}d",
    "estimation_window": ESTIM_WINDOW,
    "K"               : K,
    "lambda"          : LAMBDA,
    "w_min"           : W_MIN,
    "w_max"           : W_MAX,
    "sector_cap"      : SECTOR_CAP,
    "test_start"      : TEST_START,
    "test_end"        : TEST_END,
    "n_rebal"         : len(rebal_list),
    "n_days"          : len(daily_ret_df),
    "ann_return"      : round(ann_ret,     4),
    "ann_vol"         : round(ann_vol,     4),
    "sharpe"          : round(sharpe,      4),
    "sortino"         : round(sortino,     4),
    "calmar"          : round(calmar,      4),
    "max_drawdown"    : round(max_dd,      4),
    "avg_turnover"    : round(avg_turnover, 4),
}])
summary_df.to_csv(os.path.join(STEP3_DIR, "baseline_summary.csv"), index=False)
log("  Saved: baseline_summary.csv")

# baseline_wealth_drawdown.csv - daily wealth curve and drawdown series
wd_df = pd.DataFrame({
    "portfolio_return" : r,
    "wealth_index"     : wealth,
    "running_max"      : running_max,
    "drawdown"         : drawdown,
}, index=r_index)
wd_df.index.name = "date"
wd_df.to_csv(os.path.join(STEP3_DIR, "baseline_wealth_drawdown.csv"))
log(f"  Saved: baseline_wealth_drawdown.csv  ({len(wd_df)} rows)")

# equal_weight_returns.csv
ew_ret_df.to_csv(os.path.join(STEP3_DIR, "equal_weight_returns.csv"))
log(f"  Saved: equal_weight_returns.csv  ({len(ew_ret_df)} rows)")

# equal_weight_summary.csv
ew_summary_df = pd.DataFrame([{
    "model"        : "equal_weight",
    "n_stocks"     : N,
    "test_start"   : TEST_START,
    "test_end"     : TEST_END,
    "n_rebal"      : len(rebal_list),
    "n_days"       : len(ew_ret_df),
    "ann_return"   : round(ew_ann_ret, 4),
    "ann_vol"      : round(ew_ann_vol, 4),
    "sharpe"       : round(ew_sharpe,  4),
    "sortino"      : round(ew_sortino, 4),
    "calmar"       : round(ew_calmar,  4),
    "max_drawdown" : round(ew_max_dd,  4),
}])
ew_summary_df.to_csv(os.path.join(STEP3_DIR, "equal_weight_summary.csv"), index=False)
log("  Saved: equal_weight_summary.csv")

# baseline_net_cost_summary.csv
net_cost_df = pd.DataFrame(list(cost_results.values()))
net_cost_df.insert(0, "model", "baseline_markowitz")
net_cost_df["test_start"] = TEST_START
net_cost_df["test_end"]   = TEST_END
net_cost_df.to_csv(os.path.join(STEP3_DIR, "baseline_net_cost_summary.csv"), index=False)
log(f"  Saved: baseline_net_cost_summary.csv  ({len(net_cost_df)} cost scenarios)")

# =============================================================================
# 10. SAVE LOG
# =============================================================================
log_path = os.path.join(STEP3_DIR, "baseline_log.txt")
with open(log_path, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
print(f"  Log saved: baseline_log.txt")

# =============================================================================
# 11. FINAL SUMMARY
# =============================================================================
log("\n" + "=" * 70)
log("  STEP 3 - BASELINE PORTFOLIO COMPLETE")
log("=" * 70)
log(f"""
  Model            : Markowitz MIQP (gurobipy), historical-mean mu
  Universe         : {N} stocks
  Test window      : {TEST_START} to {TEST_END}
  Rebalancing      : monthly ({len(rebal_list)} periods)

  Portfolio parameters:
    K (cardinality): {K}
    Weight bounds  : [{W_MIN:.0%}, {W_MAX:.0%}]
    Sector cap     : {SECTOR_CAP:.0%}
    Lambda (lambda)     : {LAMBDA}

  Performance (gross):
    Annualised Return  : {ann_ret:.2%}
    Annualised Vol     : {ann_vol:.2%}
    Sharpe Ratio       : {sharpe:.4f}
    Sortino Ratio      : {sortino:.4f}
    Calmar Ratio       : {calmar:.4f}
    Maximum Drawdown   : {max_dd:.2%}
    Avg Turnover/month : {avg_turnover:.2%}

  SAFE AI - Sustainability diagnostics:
    Net Sharpe @ 10 bps  : {cost_results[10]['sharpe']:.4f}
    Net Sharpe @ 20 bps  : {cost_results[20]['sharpe']:.4f}
    Net Sharpe @ 30 bps  : {cost_results[30]['sharpe']:.4f}
    Equal-weight Sharpe  : {ew_sharpe:.4f}  (MDD: {ew_max_dd:.2%})

  Outputs (data/results/):
    baseline_weights.csv
    baseline_returns.csv
    baseline_turnover.csv
    baseline_summary.csv
    baseline_wealth_drawdown.csv
    equal_weight_returns.csv
    equal_weight_summary.csv
    baseline_net_cost_summary.csv
    baseline_log.txt
""")
log("=" * 70)
log("  Next: run step4_ml.py to build XGBoost mu predictions.")
log("=" * 70)
