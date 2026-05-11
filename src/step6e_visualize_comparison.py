"""
step6e_visualize_comparison.py
=========================================
PURPOSE:
    Generate out-of-sample comparison plots for the baseline Markowitz
    portfolio and the three ML-enhanced portfolios (Ridge, XGBoost, MLP).

    This script does NOT rerun portfolio optimisation and does NOT recompute
    ML predictions.  It reads only the realised returns, weights, turnover,
    and summary statistics already produced by Step 3 and Steps 6aâ€“6d.

PIPELINE:
    Step 3  â†’ step3_baseline_portfolio.py              (Markowitz baseline)
    Step 6a â†’ step6a_portfolio_ridge.py      (Ridge MIQP)
    Step 6b â†’ step6b_portfolio_xgboost.py    (XGBoost MIQP)
    Step 6c â†’ step6c_portfolio_mlp.py        (MLP MIQP)
    Step 6d â†’ step6d_compare_portfolios.py   (numerical comparison)
    Step 6e â†’ this script                    (visual comparison)

INPUTS (required):
    data/results/baseline_returns.csv
    data/results/baseline_weights.csv
    data/results/baseline_turnover.csv
    data/results/baseline_summary.csv
    data/results/step6/ridge/ridge_returns.csv
    data/results/step6/ridge/ridge_weights.csv
    data/results/step6/ridge/ridge_turnover.csv
    data/results/step6/ridge/ridge_summary.csv
    data/results/step6/xgboost/xgboost_returns.csv
    data/results/step6/xgboost/xgboost_weights.csv
    data/results/step6/xgboost/xgboost_turnover.csv
    data/results/step6/xgboost/xgboost_summary.csv
    data/results/step6/mlp/mlp_returns.csv
    data/results/step6/mlp/mlp_weights.csv
    data/results/step6/mlp/mlp_turnover.csv
    data/results/step6/mlp/mlp_summary.csv
    data/clean/meta_clean.csv

INPUTS (optional):
    data/results/step6/ml_portfolio_comparison_summary.csv
    data/results/step6/ml_net_cost_comparison_summary.csv

OUTPUTS (data/figures/step6/):
    portfolio_cumulative_wealth_comparison.png
    portfolio_drawdown_comparison.png
    portfolio_rolling_sharpe_comparison.png
    portfolio_turnover_comparison.png
    portfolio_average_turnover_barplot.png
    portfolio_average_sector_allocation_comparison.png
    baseline_sector_heatmap.png
    xgboost_sector_heatmap.png
    portfolio_sharpe_turnover_scatter.png

LOG:
    data/results/step6/step6e_portfolio_visuals_log.txt

METHODOLOGY NOTES:
    - Daily portfolio returns are log-returns produced by the portfolio steps.
    - Cumulative wealth = exp(cumsum(log_returns)), indexed to 1.0 at start.
    - Drawdown         = wealth / running_max_wealth âˆ’ 1  (plotted negative).
    - Rolling Sharpe   = (roll_mean Ã— 252) / (roll_std Ã— âˆš252), 126-day window.
    - Turnover values are taken directly from the step-output files.

Author  : Anila Vata
Project : MSc Thesis â€” ML-Enhanced Portfolio Optimization with SAFE AI
          Evaluation, University of Pavia Â· Supervisor: Prof. Paolo Giudici
"""

# =============================================================================
# 0.  IMPORTS AND PATHS
# =============================================================================
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR  = os.path.join(BASE_DIR, "data", "results")
STEP3_DIR    = os.path.join(RESULTS_DIR, "step3")
STEP6_DIR    = os.path.join(RESULTS_DIR, "step6")
CLEAN_DIR    = os.path.join(BASE_DIR, "data", "clean")
FIGURES_DIR  = os.path.join(BASE_DIR, "data", "figures", "step6")
LOG_PATH     = os.path.join(STEP6_DIR, "step6e_portfolio_visuals_log.txt")

os.makedirs(FIGURES_DIR, exist_ok=True)

ROLL_WIN = 126    # rolling Sharpe window (trading days)
WEIGHT_TOL = 1e-6 # minimum weight treated as held


# =============================================================================
# LOGGING
# =============================================================================

class _Tee:
    """Write every print() to both stdout and a log file."""
    def __init__(self, stream, filepath):
        self._file   = open(filepath, "w", encoding="utf-8")
        self._stream = stream

    def write(self, msg):
        self._stream.write(msg)
        self._file.write(msg)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def close(self):
        self._file.close()


# =============================================================================
# MODEL REGISTRY
# =============================================================================

MODELS = {
    "baseline_markowitz": {
        "label"   : "Baseline",
        "color"   : "#1f3c6b",
        "returns" : os.path.join(STEP3_DIR, "baseline_returns.csv"),
        "weights" : os.path.join(STEP3_DIR, "baseline_weights.csv"),
        "turnover": os.path.join(STEP3_DIR, "baseline_turnover.csv"),
        "summary" : os.path.join(STEP3_DIR, "baseline_summary.csv"),
    },
    "ridge": {
        "label"   : "Ridge",
        "color"   : "#e07b39",
        "returns" : os.path.join(STEP6_DIR, "ridge", "ridge_returns.csv"),
        "weights" : os.path.join(STEP6_DIR, "ridge", "ridge_weights.csv"),
        "turnover": os.path.join(STEP6_DIR, "ridge", "ridge_turnover.csv"),
        "summary" : os.path.join(STEP6_DIR, "ridge", "ridge_summary.csv"),
    },
    "xgboost": {
        "label"   : "XGBoost",
        "color"   : "#2ca02c",
        "returns" : os.path.join(STEP6_DIR, "xgboost", "xgboost_returns.csv"),
        "weights" : os.path.join(STEP6_DIR, "xgboost", "xgboost_weights.csv"),
        "turnover": os.path.join(STEP6_DIR, "xgboost", "xgboost_turnover.csv"),
        "summary" : os.path.join(STEP6_DIR, "xgboost", "xgboost_summary.csv"),
    },
    "mlp": {
        "label"   : "MLP",
        "color"   : "#9467bd",
        "returns" : os.path.join(STEP6_DIR, "mlp", "mlp_returns.csv"),
        "weights" : os.path.join(STEP6_DIR, "mlp", "mlp_weights.csv"),
        "turnover": os.path.join(STEP6_DIR, "mlp", "mlp_turnover.csv"),
        "summary" : os.path.join(STEP6_DIR, "mlp", "mlp_summary.csv"),
    },
}

MODEL_ORDER = ["baseline_markowitz", "ridge", "xgboost", "mlp"]

COMPARISON_SUMMARY = os.path.join(STEP6_DIR, "ml_portfolio_comparison_summary.csv")
NET_COST_SUMMARY   = os.path.join(STEP6_DIR, "ml_net_cost_comparison_summary.csv")


# =============================================================================
# SHARED HELPERS
# =============================================================================

def _require(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")


def setup_style() -> None:
    plt.rcParams.update({
        "font.family"       : "serif",
        "font.size"         : 11,
        "axes.spines.top"   : False,
        "axes.spines.right" : False,
        "axes.grid"         : True,
        "grid.alpha"        : 0.25,
        "grid.linestyle"    : "--",
        "grid.linewidth"    : 0.6,
        "figure.dpi"        : 100,
        "savefig.dpi"       : 300,
    })


def savefig(fig: plt.Figure, fname: str) -> str:
    path = os.path.join(FIGURES_DIR, fname)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")
    return path


def source_note(fig: plt.Figure) -> None:
    fig.text(
        0.99, 0.01, "Source: Author's elaboration.",
        ha="right", va="bottom", fontsize=8,
        color="grey", style="italic",
        transform=fig.transFigure,
    )


def fmt_xdates(ax: plt.Axes, interval: int = 3) -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")


def drawdown_series(wealth: pd.Series) -> pd.Series:
    """Drawdown from peak: always non-negative."""
    return 1.0 - wealth / wealth.cummax()


def _resolve_metric(row: pd.Series, *candidates) -> float:
    """Return the first candidate key that exists in the Series, as float."""
    for c in candidates:
        if c in row.index:
            try:
                return float(row[c])
            except (TypeError, ValueError):
                continue
    return np.nan


def _load_returns(path: str, label: str) -> pd.Series:
    """Load a returns CSV robustly: handles index=date or column='date'."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    # Find the numeric return column
    num_cols = [c for c in df.columns
                if any(kw in c.lower() for kw in ("return", "ret", "pnl"))]
    if num_cols:
        col = num_cols[0]
    else:
        # Fallback: pick the first numeric column
        num_cols_all = df.select_dtypes(include=[np.number]).columns
        if len(num_cols_all) == 0:
            raise ValueError(f"No numeric column in {path}")
        col = num_cols_all[0]
    return df[col].dropna().rename(label)


def _load_turnover(path: str) -> pd.DataFrame:
    """Load turnover CSV with either 'rebal_date' as column or index."""
    df = pd.read_csv(path)
    date_col = next(
        (c for c in df.columns
         if any(kw in c.lower() for kw in ("date", "rebal"))),
        None,
    )
    if date_col is None:
        # date is the index
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index.name = "rebal_date"
        df = df.reset_index()
    else:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.rename(columns={date_col: "rebal_date"})

    to_col = next(
        (c for c in df.columns if "turnover" in c.lower()),
        None,
    )
    if to_col is None:
        raise ValueError(f"No turnover column found in {path}")
    if to_col != "turnover":
        df = df.rename(columns={to_col: "turnover"})

    return df.sort_values("rebal_date").reset_index(drop=True)


def _load_summary(path: str) -> pd.Series:
    df = pd.read_csv(path)
    return df.iloc[0]


def _get_metric(summ: pd.Series, metric: str) -> float:
    aliases = {
        "sharpe"      : ["sharpe", "sharpe_ratio", "Sharpe"],
        "avg_turnover": ["avg_turnover", "AvgTO", "average_turnover", "mean_turnover"],
        "ann_return"  : ["ann_return", "ann_ret", "AnnReturn"],
        "ann_vol"     : ["ann_vol", "ann_volatility", "AnnVol"],
        "sortino"     : ["sortino", "sortino_ratio", "Sortino"],
        "calmar"      : ["calmar", "calmar_ratio", "Calmar"],
        "max_drawdown": ["max_drawdown", "MaxDD", "max_dd", "maxdd"],
    }
    candidates = aliases.get(metric, [metric])
    return _resolve_metric(summ, *candidates)


# =============================================================================
# 1.  LOAD DATA
# =============================================================================

def load_data():
    print("=" * 72)
    print("  STEP 6e â€” ML PORTFOLIO COMPARISON VISUALISATIONS")
    print("=" * 72)
    print()
    print("[1] Loading and validating inputs â€¦")
    print()

    # Validate all required files
    for key, cfg in MODELS.items():
        for field in ("returns", "weights", "turnover", "summary"):
            _require(cfg[field])
    _require(os.path.join(CLEAN_DIR, "meta_clean.csv"))

    # Returns
    returns = {}
    for key, cfg in MODELS.items():
        returns[key] = _load_returns(cfg["returns"], cfg["label"])
        s = returns[key]
        print(f"  returns [{cfg['label']:8s}]: {len(s)} days  "
              f"({s.index[0].date()} â€“ {s.index[-1].date()})")

    # Turnover
    turnovers = {}
    for key, cfg in MODELS.items():
        turnovers[key] = _load_turnover(cfg["turnover"])
        df = turnovers[key]
        print(f"  turnover[{cfg['label']:8s}]: {len(df)} rebal dates  "
              f"avg = {df['turnover'].mean():.2%}")

    # Summaries
    summaries = {}
    for key, cfg in MODELS.items():
        summaries[key] = _load_summary(cfg["summary"])

    # Comparison summary (from step6d)
    if os.path.exists(COMPARISON_SUMMARY):
        comp_df = pd.read_csv(COMPARISON_SUMMARY)
        if "model" in comp_df.columns:
            comp_df = comp_df.set_index("model").reindex(MODEL_ORDER)
        print(f"  comparison summary : loaded from {COMPARISON_SUMMARY}")
    else:
        rows = []
        for key in MODEL_ORDER:
            row = summaries[key].copy()
            row["model"] = key
            rows.append(row)
        comp_df = pd.DataFrame(rows).set_index("model")
        print("  comparison summary : rebuilt from individual summary files")

    # Net-cost summary (optional)
    net_cost_df = None
    if os.path.exists(NET_COST_SUMMARY):
        net_cost_df = pd.read_csv(NET_COST_SUMMARY)
        print(f"  net-cost summary   : loaded ({len(net_cost_df)} rows)")

    # Weights
    weights = {}
    for key, cfg in MODELS.items():
        df = pd.read_csv(cfg["weights"], index_col=0, parse_dates=True).sort_index()
        weights[key] = df
        print(f"  weights [{cfg['label']:8s}]: {df.shape[0]} dates Ã— {df.shape[1]} tickers")

    # Metadata
    meta = pd.read_csv(os.path.join(CLEAN_DIR, "meta_clean.csv"))
    meta.columns = meta.columns.str.strip().str.lower()
    if "ticker" not in meta.columns:
        meta = meta.rename(columns={meta.columns[0]: "ticker"})
    meta["ticker"] = meta["ticker"].astype(str).str.strip().str.upper()
    sector_col = next(
        (c for c in ["sector", "gics_sector", "gics sector"] if c in meta.columns),
        None,
    )
    if sector_col is None:
        raise ValueError(f"No sector column in meta_clean.csv. Columns: {list(meta.columns)}")
    ticker_sector = meta.set_index("ticker")[sector_col].to_dict()
    print(f"  meta               : {len(meta)} tickers Â· sector col = '{sector_col}'")

    # Align returns on common index
    ret_frame = pd.DataFrame({cfg["label"]: returns[key]
                               for key, cfg in MODELS.items()}).sort_index()
    wealth_frame = ret_frame.cumsum().apply(np.exp)   # cumulative wealth (base 1)

    print()
    return (returns, turnovers, summaries, comp_df, net_cost_df,
            weights, ticker_sector, ret_frame, wealth_frame)


# =============================================================================
# FIGURE 1 â€” CUMULATIVE WEALTH
# =============================================================================

def fig_cumulative_wealth(wealth_frame):
    print("[Fig 1] Cumulative Wealth Comparison â€¦")
    try:
        setup_style()
        fig, ax = plt.subplots(figsize=(12, 5))

        for key in MODEL_ORDER:
            cfg    = MODELS[key]
            series = wealth_frame[cfg["label"]].dropna()
            ax.plot(series.index, series.values,
                    color=cfg["color"], linewidth=1.8,
                    label=cfg["label"], zorder=4)

        ax.axhline(1.0, color="black", linewidth=0.6, linestyle=":", zorder=2)
        ax.set_title(
            "Out-of-Sample Cumulative Wealth â€” Baseline vs ML Portfolios",
            fontsize=13, fontweight="bold", pad=10,
        )
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Portfolio Value (base = 1.0)", fontsize=11)
        fmt_xdates(ax)
        ax.legend(fontsize=10, framealpha=0.85, edgecolor="#cccccc")
        source_note(fig)
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        savefig(fig, "portfolio_cumulative_wealth_comparison.png")

        for key in MODEL_ORDER:
            cfg = MODELS[key]
            s   = wealth_frame[cfg["label"]].dropna()
            print(f"      {cfg['label']:8s}: terminal wealth = {s.iloc[-1]:.4f}  "
                  f"(+{s.iloc[-1] - 1:.2%})")
    except Exception as exc:
        print(f"    ERROR: {exc}")


# =============================================================================
# FIGURE 2 â€” DRAWDOWN
# =============================================================================

def fig_drawdown(wealth_frame):
    print("\n[Fig 2] Drawdown Comparison â€¦")
    try:
        setup_style()
        fig, ax = plt.subplots(figsize=(12, 5))

        for key in MODEL_ORDER:
            cfg  = MODELS[key]
            widx = wealth_frame[cfg["label"]].dropna()
            dd   = -drawdown_series(widx)
            ax.plot(dd.index, dd.values,
                    color=cfg["color"], linewidth=1.4,
                    label=cfg["label"], zorder=4)
            ax.fill_between(dd.index, dd.values, 0,
                            color=cfg["color"], alpha=0.07, zorder=2)

        ax.axhline(0, color="black", linewidth=0.6, linestyle=":", zorder=2)
        ax.set_title(
            "Out-of-Sample Drawdown â€” Baseline vs ML Portfolios",
            fontsize=13, fontweight="bold", pad=10,
        )
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Drawdown", fontsize=11)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        fmt_xdates(ax)
        ax.legend(fontsize=10, framealpha=0.85, edgecolor="#cccccc")
        source_note(fig)
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        savefig(fig, "portfolio_drawdown_comparison.png")

        for key in MODEL_ORDER:
            cfg  = MODELS[key]
            widx = wealth_frame[cfg["label"]].dropna()
            dd   = drawdown_series(widx)
            print(f"      {cfg['label']:8s}: max drawdown = {dd.max():.2%}")
    except Exception as exc:
        print(f"    ERROR: {exc}")


# =============================================================================
# FIGURE 3 â€” ROLLING SHARPE (126-day)
# =============================================================================

def fig_rolling_sharpe(returns):
    print(f"\n[Fig 3] Rolling Sharpe Ratio ({ROLL_WIN}-day window) â€¦")
    try:
        setup_style()
        fig, ax = plt.subplots(figsize=(12, 5))

        ax.axhline(0, color="black",   linewidth=0.7, linestyle=":",  zorder=2)
        ax.axhline(1, color="#555555", linewidth=0.9, linestyle="--",
                   alpha=0.6, zorder=2, label="Sharpe = 1")

        for key in MODEL_ORDER:
            cfg         = MODELS[key]
            series      = returns[key]
            roll_mean   = series.rolling(ROLL_WIN).mean() * 252
            roll_std    = series.rolling(ROLL_WIN).std(ddof=1) * np.sqrt(252)
            roll_sharpe = (roll_mean / roll_std).dropna()
            ax.plot(roll_sharpe.index, roll_sharpe.values,
                    color=cfg["color"], linewidth=1.4,
                    label=cfg["label"], zorder=4)

        ax.set_title(
            f"Rolling Sharpe Ratio ({ROLL_WIN}-Day Window) â€” Baseline vs ML Portfolios",
            fontsize=13, fontweight="bold", pad=10,
        )
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Sharpe Ratio (annualised)", fontsize=11)
        fmt_xdates(ax)
        ax.legend(fontsize=10, framealpha=0.85, edgecolor="#cccccc")
        source_note(fig)
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        savefig(fig, "portfolio_rolling_sharpe_comparison.png")

        for key in MODEL_ORDER:
            cfg   = MODELS[key]
            s     = returns[key]
            rm    = s.rolling(ROLL_WIN).mean() * 252
            rstd  = s.rolling(ROLL_WIN).std(ddof=1) * np.sqrt(252)
            rsh   = (rm / rstd).dropna()
            print(f"      {cfg['label']:8s}: avg rolling Sharpe = {rsh.mean():.4f}  "
                  f"pct > 0 = {(rsh > 0).mean():.1%}")
    except Exception as exc:
        print(f"    ERROR: {exc}")


# =============================================================================
# FIGURE 4 â€” TURNOVER THROUGH TIME (time series)
# =============================================================================

def fig_turnover_timeseries(turnovers):
    print("\n[Fig 4] Monthly Turnover Through Time â€¦")
    try:
        setup_style()
        fig, ax = plt.subplots(figsize=(12, 4.5))

        for key in MODEL_ORDER:
            cfg  = MODELS[key]
            df   = turnovers[key]
            ax.plot(
                df["rebal_date"], df["turnover"],
                color=cfg["color"], linewidth=1.4, alpha=0.9,
                marker="o", markersize=3.5,
                label=cfg["label"], zorder=4,
            )

        ax.set_title(
            "Monthly Portfolio Turnover Through Time â€” Baseline vs ML Portfolios",
            fontsize=13, fontweight="bold", pad=10,
        )
        ax.set_xlabel("Rebalancing Date", fontsize=11)
        ax.set_ylabel("One-Way Turnover", fontsize=11)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        fmt_xdates(ax, interval=3)
        ax.legend(fontsize=10, framealpha=0.85, edgecolor="#cccccc")
        source_note(fig)
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        savefig(fig, "portfolio_turnover_comparison.png")
    except Exception as exc:
        print(f"    ERROR: {exc}")


# =============================================================================
# FIGURE 5 â€” AVERAGE TURNOVER BAR CHART
# =============================================================================

def fig_average_turnover_bar(turnovers):
    print("\n[Fig 5] Average Turnover Bar Chart â€¦")
    try:
        setup_style()
        fig, ax = plt.subplots(figsize=(8, 5))

        labels = [MODELS[k]["label"] for k in MODEL_ORDER]
        colors = [MODELS[k]["color"] for k in MODEL_ORDER]
        means  = [float(turnovers[k]["turnover"].mean()) for k in MODEL_ORDER]

        x    = np.arange(len(labels))
        bars = ax.bar(x, means, color=colors, width=0.55, alpha=0.85,
                      edgecolor="white", linewidth=0.8, zorder=3)

        for bar, val in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.005,
                f"{val:.1%}",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
                color="#333333",
            )

        ax.set_title(
            "Average Monthly Turnover by Model",
            fontsize=13, fontweight="bold", pad=10,
        )
        ax.set_xlabel("Model", fontsize=11)
        ax.set_ylabel("Average One-Way Turnover", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.set_ylim(0, max(means) * 1.25)
        source_note(fig)
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        savefig(fig, "portfolio_average_turnover_barplot.png")

        for key, m in zip(MODEL_ORDER, means):
            print(f"      {MODELS[key]['label']:8s}: avg turnover = {m:.2%}")
    except Exception as exc:
        print(f"    ERROR: {exc}")


# =============================================================================
# SECTOR AGGREGATION HELPER
# =============================================================================

def _build_sector_matrix(key, weights, ticker_sector):
    """Average sector weights for a single model across all rebalancing dates."""
    wdf         = weights[key]
    all_sectors = sorted(set(ticker_sector.values()))
    rows = []
    for _, wrow in wdf.iterrows():
        sec_wts = {s: 0.0 for s in all_sectors}
        for ticker, wt in wrow.items():
            try:
                wt_f = float(wt)
            except (TypeError, ValueError):
                continue
            if wt_f > WEIGHT_TOL:
                sec = ticker_sector.get(str(ticker).strip().upper(), "Unknown")
                sec_wts[sec] = sec_wts.get(sec, 0.0) + wt_f
        rows.append(sec_wts)
    return pd.DataFrame(rows, columns=all_sectors).mean()


def _build_sector_time_matrix(key, weights, ticker_sector):
    """Sector weights at each rebalancing date (sectors Ã— dates)."""
    wdf         = weights[key]
    all_sectors = sorted(set(ticker_sector.values()))
    rows = []
    for date, wrow in wdf.iterrows():
        sec_wts = {s: 0.0 for s in all_sectors}
        for ticker, wt in wrow.items():
            try:
                wt_f = float(wt)
            except (TypeError, ValueError):
                continue
            if wt_f > WEIGHT_TOL:
                sec = ticker_sector.get(str(ticker).strip().upper(), "Unknown")
                sec_wts[sec] = sec_wts.get(sec, 0.0) + wt_f
        sec_wts["_date"] = date
        rows.append(sec_wts)
    df = pd.DataFrame(rows).set_index("_date")
    df.index = pd.to_datetime(df.index)
    df = df[all_sectors]
    # Drop sectors never allocated to
    return df.loc[:, (df > 1e-4).any(axis=0)]


# =============================================================================
# FIGURE 6 â€” AVERAGE SECTOR ALLOCATION COMPARISON (heatmap)
# =============================================================================

def fig_sector_allocation_comparison(weights, ticker_sector):
    print("\n[Fig 6] Average Sector Allocation Comparison â€¦")
    try:
        setup_style()

        # Build average sector weight per model
        avgs = {key: _build_sector_matrix(key, weights, ticker_sector)
                for key in MODEL_ORDER}

        all_sectors = sorted(set(ticker_sector.values()))
        labels      = [MODELS[k]["label"] for k in MODEL_ORDER]
        matrix      = pd.DataFrame(
            {MODELS[k]["label"]: avgs[k].reindex(all_sectors, fill_value=0.0)
             for k in MODEL_ORDER},
            index=all_sectors,
        )
        matrix = matrix.loc[(matrix > 1e-4).any(axis=1)]
        active = list(matrix.index)

        fig_h  = max(6, len(active) * 0.65)
        fig, ax = plt.subplots(figsize=(10, fig_h))

        if HAS_SEABORN:
            annot = matrix.apply(
                lambda col: col.map(lambda v: f"{v:.1%}" if v >= 0.005 else "")
            )
            sns.heatmap(
                matrix, ax=ax, cmap="Blues",
                vmin=0.0, vmax=matrix.values.max() * 1.05,
                annot=annot, fmt="",
                linewidths=0.5, linecolor="#dddddd",
                cbar_kws={"label": "Average Portfolio Weight", "shrink": 0.7},
            )
        else:
            im = ax.imshow(
                matrix.values, aspect="auto", cmap="Blues",
                vmin=0.0, vmax=matrix.values.max() * 1.05,
                interpolation="nearest",
            )
            plt.colorbar(im, ax=ax, label="Average Portfolio Weight", shrink=0.7)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=10)
            ax.set_yticks(range(len(active)))
            ax.set_yticklabels(active, fontsize=9)
            for i, sec in enumerate(active):
                for j, lbl in enumerate(labels):
                    v = matrix.loc[sec, lbl]
                    if v >= 0.005:
                        ax.text(j, i, f"{v:.1%}", ha="center", va="center",
                                fontsize=8,
                                color="white" if v > matrix.values.max() * 0.65
                                else "black")

        ax.set_title("Average Sector Allocation by Model",
                     fontsize=13, fontweight="bold", pad=14)
        ax.set_xlabel("Model", fontsize=11)
        ax.set_ylabel("GICS Sector", fontsize=11)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, fontsize=10)
        ax.tick_params(axis="y", labelsize=9)
        source_note(fig)
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        savefig(fig, "portfolio_average_sector_allocation_comparison.png")

        for key in MODEL_ORDER:
            lbl   = MODELS[key]["label"]
            top_s = matrix[lbl].idxmax()
            top_w = matrix[lbl].max()
            print(f"      {lbl:8s}: dominant sector = {top_s}  ({top_w:.1%})")
    except Exception as exc:
        print(f"    ERROR: {exc}")


# =============================================================================
# FIGURE 7 â€” SECTOR HEATMAPS OVER TIME (Baseline and XGBoost)
# =============================================================================

def _plot_sector_heatmap(key, weights, ticker_sector, fname):
    """Heatmap of sector weights over rebalancing dates for one model."""
    cfg      = MODELS[key]
    time_mat = _build_sector_time_matrix(key, weights, ticker_sector)

    if time_mat.empty:
        print(f"      No sector data for {cfg['label']} â€” skipping heatmap.")
        return

    sectors = list(time_mat.columns)
    dates   = [d.strftime("%b %Y") for d in time_mat.index]

    fig_w = max(12, len(dates) * 0.4)
    fig_h = max(5,  len(sectors) * 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    mat_vals = time_mat.values.T   # sectors Ã— dates

    if HAS_SEABORN:
        annot_mat = pd.DataFrame(
            [[f"{v:.0%}" if v >= 0.01 else "" for v in row]
             for row in mat_vals],
            index=sectors,
            columns=dates,
        )
        sns.heatmap(
            pd.DataFrame(mat_vals, index=sectors, columns=dates),
            ax=ax, cmap="Blues",
            vmin=0.0, vmax=min(mat_vals.max() * 1.1, 0.50),
            annot=annot_mat, fmt="",
            linewidths=0.4, linecolor="#eeeeee",
            cbar_kws={"label": "Portfolio Weight", "shrink": 0.6},
        )
    else:
        im = ax.imshow(mat_vals, aspect="auto", cmap="Blues",
                       vmin=0.0, vmax=min(mat_vals.max() * 1.1, 0.50),
                       interpolation="nearest")
        plt.colorbar(im, ax=ax, label="Portfolio Weight", shrink=0.6)
        ax.set_xticks(range(len(dates)))
        ax.set_xticklabels(dates, fontsize=7, rotation=60, ha="right")
        ax.set_yticks(range(len(sectors)))
        ax.set_yticklabels(sectors, fontsize=8)

    ax.set_title(
        f"{cfg['label']} â€” Sector Allocation Over Time",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Rebalancing Date", fontsize=10)
    ax.set_ylabel("GICS Sector", fontsize=10)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    source_note(fig)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    savefig(fig, fname)


def fig_sector_heatmaps(weights, ticker_sector):
    print("\n[Fig 7] Sector Heatmaps (Baseline and XGBoost) â€¦")
    try:
        setup_style()
        _plot_sector_heatmap("baseline_markowitz", weights, ticker_sector,
                             "baseline_sector_heatmap.png")
        _plot_sector_heatmap("xgboost",            weights, ticker_sector,
                             "xgboost_sector_heatmap.png")
    except Exception as exc:
        print(f"    ERROR: {exc}")


# =============================================================================
# FIGURE 8 â€” SHARPE vs AVERAGE TURNOVER SCATTER (SAFE trade-off)
# =============================================================================

def fig_sharpe_turnover_scatter(summaries, turnovers):
    print("\n[Fig 8] Sharpe vs Average Turnover Scatter (SAFE Trade-off) â€¦")
    try:
        setup_style()
        fig, ax = plt.subplots(figsize=(8, 6))

        for key in MODEL_ORDER:
            cfg    = MODELS[key]
            sharpe = _get_metric(summaries[key], "sharpe")
            avgto  = float(turnovers[key]["turnover"].mean())

            ax.scatter(avgto, sharpe,
                       color=cfg["color"], s=160, zorder=5,
                       edgecolors="white", linewidths=1.2)
            ax.annotate(
                cfg["label"],
                xy=(avgto, sharpe),
                xytext=(8, 4),
                textcoords="offset points",
                fontsize=10, color=cfg["color"],
                fontweight="bold",
            )

        # Reference lines
        mean_to = np.mean([turnovers[k]["turnover"].mean() for k in MODEL_ORDER])
        ax.axvline(mean_to, color="#bbbbbb", linewidth=0.8,
                   linestyle="--", zorder=2)

        ax.set_title(
            "SAFE AI Trade-off: Accuracy vs Sustainability\n"
            "Sharpe Ratio (Accuracy) vs Average Turnover (Sustainability)",
            fontsize=12, fontweight="bold", pad=12,
        )
        ax.set_xlabel("Average Monthly Turnover  â†  Lower = more Sustainable",
                      fontsize=10)
        ax.set_ylabel("Annualised Sharpe Ratio  â†‘  Higher = more Accurate",
                      fontsize=10)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))

        # Quadrant labels
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.text(xlim[0] + 0.01 * (xlim[1] - xlim[0]),
                ylim[0] + 0.92 * (ylim[1] - ylim[0]),
                "High Accuracy\nHigh Sustainability",
                fontsize=8, color="#2ca02c", style="italic", alpha=0.7)
        ax.text(xlim[0] + 0.65 * (xlim[1] - xlim[0]),
                ylim[0] + 0.92 * (ylim[1] - ylim[0]),
                "High Accuracy\nLow Sustainability",
                fontsize=8, color="#e07b39", style="italic", alpha=0.7)
        ax.text(xlim[0] + 0.01 * (xlim[1] - xlim[0]),
                ylim[0] + 0.04 * (ylim[1] - ylim[0]),
                "Low Accuracy\nHigh Sustainability",
                fontsize=8, color="#555555", style="italic", alpha=0.7)
        ax.text(xlim[0] + 0.65 * (xlim[1] - xlim[0]),
                ylim[0] + 0.04 * (ylim[1] - ylim[0]),
                "Low Accuracy\nLow Sustainability",
                fontsize=8, color="#9467bd", style="italic", alpha=0.7)

        source_note(fig)
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        savefig(fig, "portfolio_sharpe_turnover_scatter.png")

        print("      SAFE AI summary:")
        for key in MODEL_ORDER:
            cfg    = MODELS[key]
            sharpe = _get_metric(summaries[key], "sharpe")
            avgto  = float(turnovers[key]["turnover"].mean())
            print(f"        {cfg['label']:8s}: Sharpe = {sharpe:.4f}  AvgTO = {avgto:.2%}")
    except Exception as exc:
        print(f"    ERROR: {exc}")


# =============================================================================
# FINAL SUMMARY
# =============================================================================

def print_final_summary():
    print()
    print("=" * 72)
    print("  STEP 6e â€” PORTFOLIO COMPARISON VISUALISATIONS COMPLETE")
    print("=" * 72)
    print(f"  Figures directory : {FIGURES_DIR}")
    print(f"  Log file          : {LOG_PATH}")
    print()

    EXPECTED = [
        "portfolio_cumulative_wealth_comparison.png",
        "portfolio_drawdown_comparison.png",
        "portfolio_rolling_sharpe_comparison.png",
        "portfolio_turnover_comparison.png",
        "portfolio_average_turnover_barplot.png",
        "portfolio_average_sector_allocation_comparison.png",
        "baseline_sector_heatmap.png",
        "xgboost_sector_heatmap.png",
        "portfolio_sharpe_turnover_scatter.png",
    ]
    print("  Figure status:")
    for fname in EXPECTED:
        full   = os.path.join(FIGURES_DIR, fname)
        status = "OK     " if os.path.exists(full) else "MISSING"
        print(f"    [{status}]  {fname}")
    print("=" * 72)


# =============================================================================
# MAIN
# =============================================================================

def main():
    tee        = _Tee(sys.stdout, LOG_PATH)
    sys.stdout = tee

    try:
        (returns, turnovers, summaries, comp_df, net_cost_df,
         weights, ticker_sector, ret_frame, wealth_frame) = load_data()

        fig_cumulative_wealth(wealth_frame)
        fig_drawdown(wealth_frame)
        fig_rolling_sharpe(returns)
        fig_turnover_timeseries(turnovers)
        fig_average_turnover_bar(turnovers)
        fig_sector_allocation_comparison(weights, ticker_sector)
        fig_sector_heatmaps(weights, ticker_sector)
        fig_sharpe_turnover_scatter(summaries, turnovers)

        print_final_summary()

    except Exception as exc:
        print(f"\nFATAL ERROR: {exc}")
        raise

    finally:
        sys.stdout = tee._stream
        tee.close()


if __name__ == "__main__":
    main()
