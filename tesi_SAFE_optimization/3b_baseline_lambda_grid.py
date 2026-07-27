"""
Same MIQP as 3a_baseline.py, but reruns it across the full lambda grid
{0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0} instead of just lambda=1.0 -- needed
to plot the baseline against the ML models' lambda-sensitivity curves in
6f. Everything else (universe, rebal dates, mu/Sigma estimators, turnover
convention) is identical to 3a and to 6a/6b/6c.

Per-lambda outputs go to baseline_*_lambda_{tag}.csv (tag = "0p1", "1p0",
etc.) plus an aggregated baseline_frontier_summary.csv. For lambda=1.0 it
also overwrites baseline_summary.csv for backward compatibility with
6d_compare_portfolios.py -- but NOT baseline_weights.csv, since 6a/6b/6c
depend on that file for the universe and rebalancing dates and it must
stay whatever 3a_baseline.py originally wrote.

Reads: data/clean/returns.csv, data/clean/meta_clean.csv,
       data/results/step3/baseline_weights.csv

Author: Anila Vata
Project: MSc Thesis -- ML-Enhanced Portfolio Optimization with SAFE AI Evaluation
         University of Pavia, Supervisor: Prof. Paolo Giudici
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from config import (
    BASE_DIR,
    CLEAN_DIR,
    ESTIM_WINDOW,
    LAMBDA_GRID,
    TRADING_DAYS,
    W_MAX,
    W_MIN,
)

try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False

warnings.filterwarnings("ignore")

MODEL_NAME = "baseline_markowitz"

RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")
STEP3_DIR   = os.path.join(RESULTS_DIR, "step3")
LOG_DIR     = os.path.join(STEP3_DIR, "logs")

for _d in [STEP3_DIR, LOG_DIR]:
    os.makedirs(_d, exist_ok=True)

# ---------- optimization parameters ------------------------------------------
K            = 10
SECTOR_CAP   = 0.30
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
log("  STEP 3b -- BASELINE MARKOWITZ MIQP LAMBDA FRONTIER")
log("=" * 70)

if not HAS_GUROBI:
    raise ImportError(
        "gurobipy is not installed or no valid licence found.\n"
        "Install with: pip install gurobipy   (requires Gurobi licence)")

log(f"  Gurobi version : {gp.gurobi.version()}")
log(f"  Lambda grid    : {LAMBDA_GRID}")

log("\n-- 1. LOAD STATIC INPUTS --------------------------------------------")

for fpath in [
    os.path.join(CLEAN_DIR, "returns.csv"),
    os.path.join(CLEAN_DIR, "meta_clean.csv"),
    os.path.join(STEP3_DIR, "baseline_weights.csv"),
]:
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Required file not found: {fpath}")

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

log("\n-- 2. UNIVERSE ALIGNMENT --------------------------------------------")

# Pin universe to baseline_weights.columns, identical order to step6b/c
universe = list(baseline_weights.columns)
N = len(universe)

log(f"  Universe source   : baseline_weights.columns ({N} tickers)")

missing_in_returns = [t for t in universe if t not in returns_raw.columns]
if missing_in_returns:
    raise ValueError(
        f"{len(missing_in_returns)} baseline tickers missing from returns.csv:\n"
        f"  {missing_in_returns[:10]}"
    )
log(f"  returns coverage  : PASS -- all {N} tickers present in returns.csv")

meta_tickers = set(meta["ticker"])
missing_in_meta = [t for t in universe if t not in meta_tickers]
if missing_in_meta:
    raise ValueError(
        f"{len(missing_in_meta)} baseline tickers missing from meta_clean.csv:\n"
        f"  {missing_in_meta[:10]}"
    )
log(f"  meta coverage     : PASS - all {N} tickers present in meta_clean.csv")

if N < K:
    raise ValueError(
        f"Universe has only {N} stocks but cardinality K={K} requires at least K.")

returns_df = returns_raw[universe]

meta_idx           = meta.set_index("ticker")
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

log("\n-- 3. REBALANCING DATES (from baseline_weights.csv) -----------------")

rebal_dates = baseline_weights.index.sort_values()

log(f"  Test window    : {TEST_START} to {TEST_END}")
log("  Rebalancing    : first trading day of each calendar month (from baseline)")
log(f"  Rebal periods  : {len(rebal_dates)}"
    f"  ({rebal_dates[0].date()} to {rebal_dates[-1].date()})")

test_idx = returns_df.index[
    (returns_df.index >= TEST_START) & (returns_df.index <= TEST_END)
]

if test_idx.empty:
    raise ValueError(
        f"No trading days found in returns_df between {TEST_START} and {TEST_END}.")

def compute_mu(rebal_date: pd.Timestamp) -> np.ndarray:
    """Trailing ESTIM_WINDOW-day annualised mean, winsorized p1/p99."""
    hist = returns_df.loc[returns_df.index < rebal_date].iloc[-ESTIM_WINDOW:]
    raw  = hist.mean().values * TRADING_DAYS
    lo   = np.nanpercentile(raw, 1)
    hi   = np.nanpercentile(raw, 99)
    return np.clip(raw, lo, hi)

def compute_sigma(rebal_date: pd.Timestamp) -> np.ndarray:
    """Trailing ESTIM_WINDOW-day Ledoit-Wolf covariance, annualised."""
    hist = returns_df.loc[returns_df.index < rebal_date].iloc[-ESTIM_WINDOW:]
    lw   = LedoitWolf().fit(hist.values)
    return lw.covariance_ * TRADING_DAYS

def solve_miqp(mu: np.ndarray, Sigma: np.ndarray, lam: float) -> tuple:
    """
    MIQP via gurobipy MVar API.

    minimize  w @ Sigma @ w  -  lam * mu @ w
    s.t.      sum(w) = 1,  sum(z) = K,
              W_MIN * z <= w <= W_MAX * z,
              sector aggregate <= SECTOR_CAP

    Returns (weights | None, status_str, obj_val, solve_sec)
    """
    t0 = time.perf_counter()

    m = gp.Model("baseline_miqp")
    m.Params.OutputFlag   = 0
    m.Params.TimeLimit    = TIME_LIMIT
    m.Params.MIPGap       = MIP_GAP
    m.Params.LogToConsole = 0

    w = m.addMVar(N, lb=0.0, ub=1.0,   name="w")
    z = m.addMVar(N, vtype=GRB.BINARY,  name="z")

    m.setObjective(w @ Sigma @ w - lam * (mu @ w), GRB.MINIMIZE)

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
    """Performance metrics -- identical convention to step6a/b/c."""
    r = daily_ret_df["portfolio_return"].dropna().values

    ann_ret  = float(r.mean() * TRADING_DAYS)
    ann_vol  = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe   = ann_ret / ann_vol if ann_vol > 0 else np.nan

    down_vol = float(np.sqrt(np.mean(np.minimum(r, 0) ** 2)) * np.sqrt(TRADING_DAYS))
    sortino  = ann_ret / down_vol if down_vol > 0 else np.nan

    wealth_full      = np.concatenate([[1.0], np.exp(np.cumsum(r))])
    running_max_full = np.maximum.accumulate(wealth_full)
    drawdown_full    = wealth_full / running_max_full - 1
    max_dd           = float(abs(drawdown_full.min()))
    calmar           = ann_ret / max_dd if max_dd > 0 else np.nan

    avg_turnover = float(turnover_df["turnover"].mean()) if len(turnover_df) else np.nan

    return dict(
        r            = r,
        ann_ret      = ann_ret,
        ann_vol      = ann_vol,
        sharpe       = sharpe,
        sortino      = sortino,
        calmar       = calmar,
        max_dd       = max_dd,
        avg_turnover = avg_turnover,
        wealth       = wealth_full[1:],
        running_max  = running_max_full[1:],
        drawdown     = drawdown_full[1:],
    )

def compute_net_cost(daily_ret_df: pd.DataFrame,
                     turnover_df: pd.DataFrame) -> dict:
    """Transaction-cost sensitivity at 10, 20, 30 bps, identical to step6b."""
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
        net_down_vol = float(np.sqrt(np.mean(np.minimum(rn, 0) ** 2)) * np.sqrt(TRADING_DAYS))
        net_sortino  = net_ann_ret / net_down_vol if net_down_vol > 0 else np.nan
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

def _fmt(v: float, pct: bool = True) -> str:
    if not np.isfinite(v):
        return f"{'N/A':>8}"
    return f"{v:>8.2%}" if pct else f"{v:>8.4f}"

pre_loop_log  = list(log_lines)
frontier_rows = []

for lambda_value in LAMBDA_GRID:
    lam_tag = str(lambda_value).replace(".", "p")

    log_lines.clear()
    log_lines.extend(pre_loop_log)

    log("\n" + "=" * 70)
    log(f"  LAMBDA = {lambda_value}  (tag: {lam_tag})")
    log("=" * 70)
    log("\n-- 5. OPTIMIZATION LOOP ---------------------------------------------")
    log(f"  Objective  : minimize w'Sw - L*m'w  (L={lambda_value})")
    log(f"  K={K}, w in [{W_MIN:.0%},{W_MAX:.0%}], sector cap={SECTOR_CAP:.0%}")
    log(f"  mu source  : trailing {ESTIM_WINDOW}-day mean x {TRADING_DAYS}, "
        f"winsorized p1/p99 (recomputed at each rebal date)")
    log(f"  Sigma src  : rolling Ledoit-Wolf, trailing {ESTIM_WINDOW} daily returns "
        f"strictly before each rebal date")
    log(f"  Gurobi     : TimeLimit={TIME_LIMIT}s, MIPGap={MIP_GAP}")
    log("-" * 70)

    weights_all   = {}
    prev_weights  = None
    total_solve_t = 0.0

    for rebal_date in rebal_dates:
        hist_available = returns_df.loc[returns_df.index < rebal_date]
        if len(hist_available) < ESTIM_WINDOW:
            log(f"  {rebal_date.date()}  SKIP -- {len(hist_available)} history rows "
                f"< {ESTIM_WINDOW} required for estimation")
            if prev_weights is not None:
                weights_all[rebal_date] = prev_weights.copy()
            continue

        mu_vec  = compute_mu(rebal_date)
        Sigma_t = compute_sigma(rebal_date)

        eig_min = float(np.linalg.eigvalsh(Sigma_t).min())
        if eig_min < -1e-6:
            Sigma_t += (abs(eig_min) + 1e-6) * np.eye(N)

        weights, status_str, obj_val, elapsed = solve_miqp(mu_vec, Sigma_t, lambda_value)
        total_solve_t += elapsed

        if weights is None:
            if prev_weights is None:
                raise RuntimeError(
                    f"[{MODEL_NAME}] Optimization FAILED on the first rebalancing date "
                    f"({rebal_date.date()}) with status '{status_str}' and no previous "
                    f"portfolio to carry forward.  Check solver licence and data."
                )
            log(f"  {rebal_date.date()}  FAILED ({status_str}, {elapsed:.1f}s) "
                ", carrying forward previous weights")
            weights = prev_weights.copy()

        weights_all[rebal_date] = weights
        prev_weights = weights.copy()

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

    log("\n-- 6. QUALITY CHECKS ON OPTIMIZED WEIGHTS ---------------------------")

    expected_n_rebal = baseline_weights.shape[0]
    if len(rebal_list) != expected_n_rebal:
        raise ValueError(
            f"[{MODEL_NAME}] QC5 FAIL: expected {expected_n_rebal} rebalancing rows, "
            f"got {len(rebal_list)}."
        )
    log(f"  QC5 shape          : PASS - {len(rebal_list)} rows x {N} cols")

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
            f"[{MODEL_NAME}] QC6 FAIL -- {len(violations)} constraint violation(s):\n"
            f"  {msg}"
        )
    log(f"  QC6 constraints    : PASS - all {len(rebal_list)} dates satisfy "
        f"sum=1, K={K}, W_MIN/W_MAX, sector cap")

    if universe != list(baseline_weights.columns):
        raise ValueError(
            f"[{MODEL_NAME}] QC7 FAIL: output universe column order does not match "
            f"baseline_weights.columns."
        )
    log("  QC7 column order   : PASS -- matches baseline_weights.columns exactly")

    log("\n-- 7. DAILY PORTFOLIO RETURNS (buy-and-hold drift) ------------------")
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

    log("\n-- 8. TURNOVER ------------------------------------------------------")
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

    log("\n-- 9. PERFORMANCE METRICS ------------------------------------------")

    met         = compute_metrics(daily_ret_df, turnover_df)
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

    log("\n  Per-year breakdown:")
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

    r_index    = daily_ret_df["portfolio_return"].dropna().index
    trough_idx = int(drawdown.argmin())
    peak_idx   = int(np.argmax(wealth[:trough_idx + 1]))
    log(f"\n  Final wealth index : {float(wealth[-1]):.6f}")
    log(f"  Peak date          : {r_index[peak_idx].date()}  "
        f"(wealth = {float(wealth[peak_idx]):.6f})")
    log(f"  Trough date        : {r_index[trough_idx].date()}  "
        f"(wealth = {float(wealth[trough_idx]):.6f})")

    log("\n-- 10. TRANSACTION-COST SENSITIVITY (gross -> net) ------------------")

    cost_results = compute_net_cost(daily_ret_df, turnover_df)

    log(f"\n  {'Scenario':<22} {'Ann Ret':>9} {'Ann Vol':>9} {'Sharpe':>9} {'Max DD':>9}")
    log("  " + "-" * 63)
    log(f"  {'Gross (0 bps)':<22} "
        f"{ann_ret:>8.2%}  {ann_vol:>8.2%}  {sharpe:>8.4f}  {max_dd:>8.2%}")
    for bps, res in cost_results.items():
        log(f"  {f'Net ({bps} bps)':<22} "
            f"{res['ann_return']:>8.2%}  {res['ann_vol']:>8.2%}  "
            f"{res['sharpe']:>8.4f}  {res['max_drawdown']:>8.2%}")

    log("\n-- 11. SAVING OUTPUTS -----------------------------------------------")

    rows_w = []
    for date, w in weights_all.items():
        row = {"date": date.date().isoformat()}
        row.update({universe[i]: round(float(w[i]), 6) for i in range(N)})
        rows_w.append(row)

    weights_out = pd.DataFrame(rows_w).set_index("date")
    wfname = f"baseline_weights_lambda_{lam_tag}.csv"
    weights_out.to_csv(os.path.join(STEP3_DIR, wfname))
    log(f"  Saved: {wfname}  ({weights_out.shape[0]} dates x {weights_out.shape[1]} tickers)")

    rfname = f"baseline_returns_lambda_{lam_tag}.csv"
    daily_ret_df.to_csv(os.path.join(STEP3_DIR, rfname))
    log(f"  Saved: {rfname}  ({len(daily_ret_df)} rows)")

    tfname = f"baseline_turnover_lambda_{lam_tag}.csv"
    turnover_df.to_csv(os.path.join(STEP3_DIR, tfname), index=False)
    log(f"  Saved: {tfname}  ({len(turnover_df)} rows)")

    summary_row = {
        "model"            : MODEL_NAME,
        "mu_estimator"     : f"trailing_{ESTIM_WINDOW}d_mean_winsorized",
        "sigma_estimator"  : f"rolling_ledoit_wolf_{ESTIM_WINDOW}d",
        "estimation_window": ESTIM_WINDOW,
        "K"                : K,
        "lambda"           : lambda_value,
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
    sfname = f"baseline_summary_lambda_{lam_tag}.csv"
    pd.DataFrame([summary_row]).to_csv(os.path.join(STEP3_DIR, sfname), index=False)
    log(f"  Saved: {sfname}")

    wdfname = f"baseline_wealth_drawdown_lambda_{lam_tag}.csv"
    wd_df = pd.DataFrame({
        "portfolio_return" : r,
        "wealth_index"     : wealth,
        "running_max"      : running_max,
        "drawdown"         : drawdown,
    }, index=r_index)
    wd_df.index.name = "date"
    wd_df.to_csv(os.path.join(STEP3_DIR, wdfname))
    log(f"  Saved: {wdfname}  ({len(wd_df)} rows)")

    ncfname = f"baseline_net_cost_summary_lambda_{lam_tag}.csv"
    net_cost_df = pd.DataFrame(list(cost_results.values()))
    net_cost_df.insert(0, "model", MODEL_NAME)
    net_cost_df.insert(1, "lambda", lambda_value)
    net_cost_df["test_start"] = TEST_START
    net_cost_df["test_end"]   = TEST_END
    net_cost_df.to_csv(os.path.join(STEP3_DIR, ncfname), index=False)
    log(f"  Saved: {ncfname}  ({len(net_cost_df)} cost scenarios)")

    if lambda_value == 1.0:
        # Overwrite baseline_summary.csv for backward compatibility with step6d.
        # baseline_weights.csv is intentionally NOT overwritten, step6b/c depend on it.
        pd.DataFrame([summary_row]).to_csv(
            os.path.join(STEP3_DIR, "baseline_summary.csv"), index=False
        )
        log("  Also saved baseline_summary.csv (lambda=1.0, backward compat for step6d)")

    log_path = os.path.join(LOG_DIR, f"baseline_portfolio_log_lambda_{lam_tag}.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"  Log saved: baseline_portfolio_log_lambda_{lam_tag}.txt")

    frontier_rows.append({
        "model"        : MODEL_NAME,
        "lambda"       : lambda_value,
        "ann_return"   : round(ann_ret,      4),
        "ann_vol"      : round(ann_vol,      4),
        "sharpe"       : round(sharpe,       4),
        "sortino"      : round(sortino,      4),
        "calmar"       : round(calmar,       4),
        "max_drawdown" : round(max_dd,       4),
        "avg_turnover" : round(avg_turnover, 4),
    })

    print(f"\n  lambda={lambda_value:<5}  "
          f"Ret={_fmt(ann_ret)}  Vol={_fmt(ann_vol)}  "
          f"SR={_fmt(sharpe, pct=False)}  MaxDD={_fmt(max_dd)}")

frontier_df   = pd.DataFrame(frontier_rows)
frontier_path = os.path.join(STEP3_DIR, "baseline_frontier_summary.csv")
frontier_df.to_csv(frontier_path, index=False)
print(f"\n  Saved: baseline_frontier_summary.csv  ({len(frontier_df)} rows)")

print("\n" + "=" * 70)
print("  STEP 3b -- BASELINE LAMBDA FRONTIER COMPLETE")
print("=" * 70)
print(f"  {'Lambda':<8} {'Ann Ret':>9} {'Ann Vol':>9} {'Sharpe':>8} "
      f"{'Sortino':>9} {'Calmar':>8} {'MaxDD':>8} {'AvgTO':>8}")
print("  " + "-" * 73)
for _, frow in frontier_df.iterrows():
    print(
        f"  {frow['lambda']:<8}"
        f" {_fmt(frow['ann_return'])}"
        f"  {_fmt(frow['ann_vol'])}"
        f"  {_fmt(frow['sharpe'],      pct=False)}"
        f"  {_fmt(frow['sortino'],     pct=False)}"
        f"  {_fmt(frow['calmar'],      pct=False)}"
        f"  {_fmt(frow['max_drawdown'])}"
        f"  {_fmt(frow['avg_turnover'])}"
    )
print("=" * 70)
print(f"  Output directory : {STEP3_DIR}")
print("=" * 70)
