"""
Pulls together the baseline (3a), Ridge (6a), XGBoost (6b) and MLP (6c)
summary CSVs into one comparison table with rankings -- doesn't rerun
anything, just reads what's already there. Net-cost summaries get merged
in too if those files happen to exist.

Ranking: Sharpe/Sortino/Calmar ranked descending (higher better),
max_drawdown and avg_turnover ranked ascending (lower better, drawdown is
stored as a positive magnitude). overall_rank_score is the mean of those
five ranks, overall_rank is its rank ascending. Rows always come out in a
fixed order: baseline_markowitz, ridge, xgboost, mlp.

Writes ml_portfolio_comparison_summary.csv (+ a formatted "_pretty"
version), ml_net_cost_comparison_summary.csv if net-cost files exist, and
a full console log, all under data/results/step6/comparison/.

Author: Anila Vata
Project: MSc Thesis -- ML-Enhanced Portfolio Optimization with SAFE AI Evaluation
         University of Pavia, Supervisor: Prof. Paolo Giudici
"""

import datetime
import os

import numpy as np
import pandas as pd

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")
STEP3_DIR   = os.path.join(RESULTS_DIR, "step3")
STEP6_DIR   = os.path.join(RESULTS_DIR, "step6")
STEP6D_DIR  = os.path.join(STEP6_DIR, "comparison")

BASELINE_SUMMARY  = os.path.join(STEP3_DIR,            "baseline_summary.csv")
RIDGE_SUMMARY     = os.path.join(STEP6_DIR, "ridge",   "performance", "ridge_summary.csv")
XGBOOST_SUMMARY   = os.path.join(STEP6_DIR, "xgboost", "performance", "xgboost_summary.csv")
MLP_SUMMARY       = os.path.join(STEP6_DIR, "mlp",     "performance", "mlp_summary.csv")

BASELINE_NET_COST = os.path.join(STEP3_DIR,            "baseline_net_cost_summary.csv")
RIDGE_NET_COST    = os.path.join(STEP6_DIR, "ridge",   "performance", "ridge_net_cost_summary.csv")
XGBOOST_NET_COST  = os.path.join(STEP6_DIR, "xgboost", "performance", "xgboost_net_cost_summary.csv")
MLP_NET_COST      = os.path.join(STEP6_DIR, "mlp",     "performance", "mlp_net_cost_summary.csv")

OUT_SUMMARY        = os.path.join(STEP6D_DIR, "ml_portfolio_comparison_summary.csv")
OUT_SUMMARY_PRETTY = os.path.join(STEP6D_DIR, "ml_portfolio_comparison_summary_pretty.csv")
OUT_NET_COST       = os.path.join(STEP6D_DIR, "ml_net_cost_comparison_summary.csv")
OUT_LOG            = os.path.join(STEP6D_DIR, "step6d_comparison_log.txt")

os.makedirs(STEP6D_DIR, exist_ok=True)

_log_lines: list[str] = []

def log(msg: str = "") -> None:
    print(msg)
    _log_lines.append(msg)

# Fixed canonical order for all output tables
MODEL_ORDER = ["baseline_markowitz", "ridge", "xgboost", "mlp"]

log("=" * 70)
log("STEP 6d, BASELINE vs ML PORTFOLIO COMPARISON")
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
    "baseline_markowitz" : "3a_baseline.py",
    "ridge"              : "6a_ridge_portfolio.py",
    "xgboost"            : "6b_xgboost_portfolio.py",
    "mlp"                : "6c_mlp_portfolio.py",
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
    df["model"] = label  # normalise label (baseline → baseline_markowitz)
    frames.append(df)
    log(f"  Loaded: {os.path.relpath(path, BASE_DIR)}")

combined = pd.concat(frames, ignore_index=True)
log(f"\n  Models loaded: {list(combined['model'])}")

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
        f"  Check that step3 and step6a–c outputs contain all expected fields."
    )
log(f"  All required metric columns present: {REQUIRED_METRICS}")

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
log("    rank_sharpe - descending (higher Sharpe is better)")
log("    rank_sortino, descending (higher Sortino is better)")
log("    rank_calmar - descending (higher Calmar is better)")
log("    rank_max_drawdown -- ascending  (positive magnitude; smaller drawdown is better)")
log("    rank_avg_turnover - ascending  (lower turnover is better)")
log("  overall_rank_score = average of the five individual rank columns")
log("  overall_rank       = rank of overall_rank_score ascending (lower is better)")

log("\n-- 5. SAVING RAW COMPARISON --------------------------------------------")

ranked.to_csv(OUT_SUMMARY, index=False)
log(f"  Saved: ml_portfolio_comparison_summary.csv  "
    f"({ranked.shape[0]} models × {ranked.shape[1]} columns)")

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
log("  Saved: ml_portfolio_comparison_summary_pretty.csv")

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
            log(f"  WARNING: baseline net cost file not found, skipping: "
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
    log("  No net cost files found -- ml_net_cost_comparison_summary.csv not written.")

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
        f"  {row['model']!s:<22}"
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
        f"  {row['model']!s:<22}"
        f"{int(row['rank_sharpe']):>8}"
        f"{int(row['rank_sortino']):>8}"
        f"{int(row['rank_calmar']):>8}"
        f"{int(row['rank_max_drawdown']):>8}"
        f"{int(row['rank_avg_turnover']):>8}"
        f"{float(row['overall_rank_score']):>8.4f}"
        f"{int(row['overall_rank']):>8}"
    )
log(_sep2)

with open(OUT_LOG, "w", encoding="utf-8") as fh:
    fh.write("\n".join(_log_lines) + "\n")
log("\n  Log saved: step6d_comparison_log.txt")

log("\n" + "=" * 70)
log("STEP 6d COMPLETE")
log("=" * 70)
