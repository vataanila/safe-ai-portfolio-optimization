"""
step6d_compare_portfolios.py
============================
PURPOSE : Step 6d - Baseline vs ML portfolio comparison.
          Reads pre-computed summary CSVs from Step 3 (baseline) and
          Steps 6a-6c (Ridge, XGBoost, MLP). Aggregates results into a
          unified comparison table with rankings.

          This script does NOT rerun any optimization, backtesting,
          ML predictions, or baseline results. It only reads already-
          computed summary files and produces comparison and ranking outputs.
          Optionally aggregates net-cost summaries if those files exist.

PIPELINE:
  Step 3  -> step3_baseline_portfolio.py              (Markowitz baseline, historical mu)
  Step 6a -> step6a_portfolio_ridge.py      (Ridge MIQP portfolio)
  Step 6b -> step6b_portfolio_xgboost.py    (XGBoost MIQP portfolio)
  Step 6c -> step6c_portfolio_mlp.py        (MLP MIQP portfolio)
  Step 6d -> this script                    (aggregate comparison, rankings)

INPUTS (required):
  data/results/baseline_summary.csv
  data/results/step6/ridge/ridge_summary.csv
  data/results/step6/xgboost/xgboost_summary.csv
  data/results/step6/mlp/mlp_summary.csv

INPUTS (optional - net cost aggregation):
  data/results/baseline_net_cost_summary.csv
  data/results/step6/ridge/ridge_net_cost_summary.csv
  data/results/step6/xgboost/xgboost_net_cost_summary.csv
  data/results/step6/mlp/mlp_net_cost_summary.csv

OUTPUTS (data/results/step6/):
  ml_portfolio_comparison_summary.csv        -- raw merged summary (numeric)
  ml_portfolio_comparison_summary_pretty.csv -- human-readable formatted version
  ml_net_cost_comparison_summary.csv         -- net cost table (if files exist)
  step6d_comparison_log.txt                  -- full console log

RANKING METHODOLOGY:
  rank_sharpe         descending (higher Sharpe is better)
  rank_sortino        descending (higher Sortino is better)
  rank_calmar         descending (higher Calmar is better)
  rank_max_drawdown   ascending  (max_drawdown stored as positive magnitude;
                                  lower value = smaller drawdown = better)
  rank_avg_turnover   ascending  (lower turnover is better)
  overall_rank_score  = average of the five individual rank columns
  overall_rank        = rank of overall_rank_score ascending (lower is better)

ROW ORDER (all output tables):
  1. baseline_markowitz
  2. ridge
  3. xgboost
  4. mlp

Author  : Anila Vata
Project : MSc Thesis - ML-Enhanced Portfolio Optimization with SAFE AI Evaluation
          University of Pavia * Supervisor: Prof. Paolo Giudici
"""

import os
import datetime
import numpy as np
import pandas as pd

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")
STEP3_DIR   = os.path.join(RESULTS_DIR, "step3")
STEP6_DIR   = os.path.join(RESULTS_DIR, "step6")

BASELINE_SUMMARY  = os.path.join(STEP3_DIR,            "baseline_summary.csv")
RIDGE_SUMMARY     = os.path.join(STEP6_DIR, "ridge",   "ridge_summary.csv")
XGBOOST_SUMMARY   = os.path.join(STEP6_DIR, "xgboost", "xgboost_summary.csv")
MLP_SUMMARY       = os.path.join(STEP6_DIR, "mlp",     "mlp_summary.csv")

BASELINE_NET_COST = os.path.join(STEP3_DIR,            "baseline_net_cost_summary.csv")
RIDGE_NET_COST    = os.path.join(STEP6_DIR, "ridge",   "ridge_net_cost_summary.csv")
XGBOOST_NET_COST  = os.path.join(STEP6_DIR, "xgboost", "xgboost_net_cost_summary.csv")
MLP_NET_COST      = os.path.join(STEP6_DIR, "mlp",     "mlp_net_cost_summary.csv")

OUT_SUMMARY        = os.path.join(STEP6_DIR, "ml_portfolio_comparison_summary.csv")
OUT_SUMMARY_PRETTY = os.path.join(STEP6_DIR, "ml_portfolio_comparison_summary_pretty.csv")
OUT_NET_COST       = os.path.join(STEP6_DIR, "ml_net_cost_comparison_summary.csv")
OUT_LOG            = os.path.join(STEP6_DIR, "step6d_comparison_log.txt")

os.makedirs(STEP6_DIR, exist_ok=True)

# =============================================================================
# LOGGING
# =============================================================================
_log_lines: list[str] = []

def log(msg: str = "") -> None:
    print(msg)
    _log_lines.append(msg)

# Fixed canonical order for all output tables
MODEL_ORDER = ["baseline_markowitz", "ridge", "xgboost", "mlp"]

# =============================================================================
# 1. LOAD SUMMARIES
# =============================================================================
log("=" * 70)
log("STEP 6d - BASELINE vs ML PORTFOLIO COMPARISON")
log(f"Run: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log("=" * 70)
log("\n-- 1. LOADING SUMMARIES ------------------------------------------------")

_sources = {
    "baseline_markowitz" : BASELINE_SUMMARY,
    "ridge"              : RIDGE_SUMMARY,
    "xgboost"            : XGBOOST_SUMMARY,
    "mlp"                : MLP_SUMMARY,
}

_step_map = {
    "baseline_markowitz" : "step3_baseline_portfolio.py",
    "ridge"              : "step6a_portfolio_ridge.py",
    "xgboost"            : "step6b_portfolio_xgboost.py",
    "mlp"                : "step6c_portfolio_mlp.py",
}

frames: list[pd.DataFrame] = []
for label, path in _sources.items():
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[step6d] Required summary file not found: {path}\n"
            f"  Run the corresponding step first: {_step_map[label]}"
        )
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"[step6d] Summary file is empty: {path}")
    if len(df) != 1:
        raise ValueError(
            f"[step6d] Expected exactly 1 row in {path}, got {len(df)}."
        )
    df["model"] = label  # normalise label (baseline -> baseline_markowitz)
    frames.append(df)
    log(f"  Loaded: {os.path.relpath(path, BASE_DIR)}")

combined = pd.concat(frames, ignore_index=True)
log(f"\n  Models loaded: {list(combined['model'])}")

# =============================================================================
# 2. VALIDATE REQUIRED COLUMNS
# =============================================================================
log("\n-- 2. VALIDATING COLUMNS -----------------------------------------------")

REQUIRED_METRICS = [
    "model", "mu_estimator",
    "ann_return", "ann_vol", "sharpe", "sortino",
    "calmar", "max_drawdown", "avg_turnover",
    "n_rebal", "n_days",
]
missing = [c for c in REQUIRED_METRICS if c not in combined.columns]
if missing:
    raise ValueError(
        f"[step6d] Missing required metric columns: {missing}\n"
        f"  Check that step3 and step6a-c outputs contain all expected fields."
    )
log(f"  All required metric columns present: {REQUIRED_METRICS}")

# =============================================================================
# 3. ENFORCE ROW ORDER
# =============================================================================
log("\n-- 3. ENFORCING ROW ORDER ----------------------------------------------")

combined["_order"] = combined["model"].map(
    {m: i for i, m in enumerate(MODEL_ORDER)}
)
missing_models = combined[combined["_order"].isna()]["model"].tolist()
if missing_models:
    raise ValueError(
        f"[step6d] Unexpected model labels not in MODEL_ORDER: {missing_models}\n"
        f"  Expected: {MODEL_ORDER}"
    )
combined = (
    combined.sort_values("_order")
    .drop(columns=["_order"])
    .reset_index(drop=True)
)
log(f"  Row order enforced: {list(combined['model'])}")

# =============================================================================
# 4. COMPUTE RANKINGS
# =============================================================================
log("\n-- 4. COMPUTING RANKINGS -----------------------------------------------")

ranked = combined.copy()

ranked["rank_sharpe"]       = ranked["sharpe"].rank(      ascending=False, method="min").astype(int)
ranked["rank_sortino"]      = ranked["sortino"].rank(     ascending=False, method="min").astype(int)
ranked["rank_calmar"]       = ranked["calmar"].rank(      ascending=False, method="min").astype(int)
ranked["rank_max_drawdown"] = ranked["max_drawdown"].rank(ascending=True,  method="min").astype(int)
ranked["rank_avg_turnover"] = ranked["avg_turnover"].rank(ascending=True,  method="min").astype(int)

RANK_COLS = [
    "rank_sharpe", "rank_sortino", "rank_calmar",
    "rank_max_drawdown", "rank_avg_turnover",
]
ranked["overall_rank_score"] = ranked[RANK_COLS].mean(axis=1).round(4)
ranked["overall_rank"]       = ranked["overall_rank_score"].rank(
    ascending=True, method="min"
).astype(int)

log("  Individual ranking directions:")
log("    rank_sharpe       - descending (higher Sharpe is better)")
log("    rank_sortino      - descending (higher Sortino is better)")
log("    rank_calmar       - descending (higher Calmar is better)")
log("    rank_max_drawdown - ascending  (positive magnitude; smaller drawdown is better)")
log("    rank_avg_turnover - ascending  (lower turnover is better)")
log("  overall_rank_score = average of the five individual rank columns")
log("  overall_rank       = rank of overall_rank_score ascending (lower is better)")

# =============================================================================
# 5. SAVE RAW COMPARISON WITH RANKINGS (numeric)
# =============================================================================
log("\n-- 5. SAVING RAW COMPARISON --------------------------------------------")

ranked.to_csv(OUT_SUMMARY, index=False)
log(f"  Saved: ml_portfolio_comparison_summary.csv  "
    f"({ranked.shape[0]} models x {ranked.shape[1]} columns)")

# =============================================================================
# 6. SAVE PRETTY COMPARISON (human-readable)
# =============================================================================
log("\n-- 6. SAVING PRETTY COMPARISON -----------------------------------------")

pretty = ranked.copy()

for col in ["ann_return", "ann_vol", "max_drawdown", "avg_turnover"]:
    pretty[col] = pretty[col].map(
        lambda x: f"{x:.2%}" if pd.notna(x) and np.isfinite(float(x)) else "N/A"
    )
for col in ["sharpe", "sortino", "calmar"]:
    pretty[col] = pretty[col].map(
        lambda x: f"{x:.4f}" if pd.notna(x) and np.isfinite(float(x)) else "N/A"
    )
pretty["overall_rank_score"] = pretty["overall_rank_score"].map(
    lambda x: f"{x:.4f}" if pd.notna(x) and np.isfinite(float(x)) else "N/A"
)
for col in RANK_COLS + ["overall_rank"]:
    pretty[col] = pretty[col].map(
        lambda x: str(int(x)) if pd.notna(x) else "N/A"
    )

pretty.to_csv(OUT_SUMMARY_PRETTY, index=False)
log(f"  Saved: ml_portfolio_comparison_summary_pretty.csv")

# =============================================================================
# 7. OPTIONAL NET COST AGGREGATION
# =============================================================================
log("\n-- 7. OPTIONAL NET COST AGGREGATION ------------------------------------")

_net_cost_sources = {
    "baseline_markowitz" : BASELINE_NET_COST,
    "ridge"              : RIDGE_NET_COST,
    "xgboost"            : XGBOOST_NET_COST,
    "mlp"                : MLP_NET_COST,
}

net_frames: list[pd.DataFrame] = []
for label, path in _net_cost_sources.items():
    if not os.path.exists(path):
        if label == "baseline_markowitz":
            log(f"  WARNING: baseline net cost file not found - skipping: "
                f"{os.path.relpath(path, BASE_DIR)}")
        else:
            log(f"  WARNING: {label} net cost file not found - "
                f"step6a/6b/6c may not have been run yet: "
                f"{os.path.relpath(path, BASE_DIR)}")
        continue
    df = pd.read_csv(path)
    df["model"] = label
    net_frames.append(df)
    log(f"  Loaded net cost: {os.path.relpath(path, BASE_DIR)}")

if net_frames:
    net_combined = pd.concat(net_frames, ignore_index=True)
    net_combined["_order"] = net_combined["model"].map(
        {m: i for i, m in enumerate(MODEL_ORDER)}
    )
    net_combined = (
        net_combined.sort_values(["_order", "cost_bps"])
        .drop(columns=["_order"])
        .reset_index(drop=True)
    )
    net_combined.to_csv(OUT_NET_COST, index=False)
    log(f"  Saved: ml_net_cost_comparison_summary.csv  ({len(net_combined)} rows)")
else:
    log("  No net cost files found - ml_net_cost_comparison_summary.csv not written.")

# =============================================================================
# 8. CONSOLE PERFORMANCE TABLE (sorted by overall_rank)
# =============================================================================
log("\n-- 8. PERFORMANCE TABLE (sorted by overall rank) -----------------------")

display = ranked.sort_values("overall_rank")


def _fmt(v, pct: bool = True) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return f"{'N/A':>8}"
    if not np.isfinite(f):
        return f"{'N/A':>8}"
    return f"{f:>8.2%}" if pct else f"{f:>8.4f}"


_sep = "-" * 96
log(_sep)
log(
    f"  {'Model':<22}"
    f"{'Ann Ret':>9}"
    f"{'Ann Vol':>9}"
    f"{'Sharpe':>9}"
    f"{'Sortino':>9}"
    f"{'Calmar':>9}"
    f"{'MaxDD':>9}"
    f"{'AvgTO':>9}"
    f"{'Rank':>6}"
)
log(_sep)
for _, row in display.iterrows():
    log(
        f"  {str(row['model']):<22}"
        f"{_fmt(row['ann_return'])}"
        f" {_fmt(row['ann_vol'])}"
        f" {_fmt(row['sharpe'],      pct=False)}"
        f" {_fmt(row['sortino'],     pct=False)}"
        f" {_fmt(row['calmar'],      pct=False)}"
        f" {_fmt(row['max_drawdown'])}"
        f" {_fmt(row['avg_turnover'])}"
        f" {int(row['overall_rank']):>5}"
    )
log(_sep)

# =============================================================================
# 9. INDIVIDUAL RANKING BREAKDOWN (sorted by overall_rank)
# =============================================================================
log("\n-- 9. INDIVIDUAL RANKING BREAKDOWN (sorted by overall rank) ------------")

_sep2 = "-" * 80
log(_sep2)
log(
    f"  {'Model':<22}"
    f"{'Sharpe':>8}"
    f"{'Sortino':>8}"
    f"{'Calmar':>8}"
    f"{'MaxDD':>8}"
    f"{'AvgTO':>8}"
    f"{'Score':>8}"
    f"{'Rank':>8}"
)
log(_sep2)
for _, row in display.iterrows():
    log(
        f"  {str(row['model']):<22}"
        f"{int(row['rank_sharpe']):>8}"
        f"{int(row['rank_sortino']):>8}"
        f"{int(row['rank_calmar']):>8}"
        f"{int(row['rank_max_drawdown']):>8}"
        f"{int(row['rank_avg_turnover']):>8}"
        f"{float(row['overall_rank_score']):>8.4f}"
        f"{int(row['overall_rank']):>8}"
    )
log(_sep2)

# =============================================================================
# 10. SAVE LOG
# =============================================================================
with open(OUT_LOG, "w", encoding="utf-8") as fh:
    fh.write("\n".join(_log_lines) + "\n")
log(f"\n  Log saved: step6d_comparison_log.txt")

log("\n" + "=" * 70)
log("STEP 6d COMPLETE")
log("=" * 70)
