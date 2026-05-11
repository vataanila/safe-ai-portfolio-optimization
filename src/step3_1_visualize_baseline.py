"""
step3_1_visualize_baseline.py
====================
PURPOSE : Diagnostic visualisations for the Markowitz MIQP baseline
          portfolio produced by step3_baseline.py.

INPUTS  :
  data/results/baseline_returns.csv -- daily log-return series (date, portfolio_return)
  data/results/baseline_weights.csv -- rebal_date Ã— tickers (sparse weights)
  data/clean/meta_clean.csv         -- ticker metadata with GICS sector column
  ^GSPC (yfinance, optional)        -- S&P 500 benchmark daily prices

OUTPUTS (data/figures/):
  baseline_cumulative_returns.png  -- cumulative log returns vs S&P 500 benchmark
  baseline_rolling_sharpe.png      -- 126-day rolling Sharpe ratio
  baseline_drawdown.png            -- drawdown as negative percentage, annotated
  baseline_turnover.png            -- monthly turnover bar chart
  baseline_sector_heatmap.png      -- GICS sector allocation heatmap over time

Author  : Anila Vata 
"""

# =============================================================================
# 0. IMPORTS AND SHARED SETUP
# =============================================================================
import os
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
    print("  NOTE: seaborn not found â€” heatmap will use matplotlib imshow.")

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print("  NOTE: yfinance not found â€” S&P 500 benchmark line will be omitted.")

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")
CLEAN_DIR   = os.path.join(BASE_DIR, "data", "clean")
FIGURES_DIR = os.path.join(BASE_DIR, "data", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def setup_style():
    """Apply shared academic rcParams to every figure."""
    plt.rcParams.update({
        "font.family"        : "serif",
        "font.size"          : 11,
        "axes.spines.top"    : False,
        "axes.spines.right"  : False,
        "axes.grid"          : True,
        "grid.alpha"         : 0.25,
        "grid.linestyle"     : "--",
        "grid.linewidth"     : 0.6,
        "figure.dpi"         : 100,
        "savefig.dpi"        : 300,
    })


def savefig(fig, fname: str):
    path = os.path.join(FIGURES_DIR, fname)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def source_note(ax):
    ax.text(
        0.99, -0.12, "Source: Author's elaboration.",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8, color="#aaaaaa",
    )


print("=" * 70)
print("  STEP VIZ â€” BASELINE PORTFOLIO DIAGNOSTICS")
print("=" * 70)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n[0] Loading data â€¦")

def _require(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Run step3_baseline.py first.")

_require(os.path.join(RESULTS_DIR, "baseline_returns.csv"))
_require(os.path.join(RESULTS_DIR, "baseline_weights.csv"))
_require(os.path.join(CLEAN_DIR,   "meta_clean.csv"))

# Daily returns
ret_df = pd.read_csv(
    os.path.join(RESULTS_DIR, "baseline_returns.csv"),
    index_col=0, parse_dates=True
).sort_index()
ret_series = ret_df["portfolio_return"].dropna()

# Weights
weights_df = pd.read_csv(
    os.path.join(RESULTS_DIR, "baseline_weights.csv"),
    index_col=0, parse_dates=True
).sort_index()
weights_df.index = pd.DatetimeIndex(weights_df.index)

# Meta
meta = pd.read_csv(os.path.join(CLEAN_DIR, "meta_clean.csv"))
meta.columns = meta.columns.str.strip().str.lower()
if "ticker" not in meta.columns:
    meta = meta.rename(columns={meta.columns[0]: "ticker"})
meta["ticker"] = meta["ticker"].astype(str).str.strip().str.upper()

sector_col = next(
    (c for c in ["sector", "gics_sector", "gics sector"] if c in meta.columns),
    None
)
if sector_col is None:
    raise ValueError("No sector column found in meta_clean.csv.")

ticker_sector = meta.set_index("ticker")[sector_col].to_dict()

DATE_START = ret_series.index[0]
DATE_END   = ret_series.index[-1]

print(f"  baseline_returns : {len(ret_series)} days "
      f"({DATE_START.date()} to {DATE_END.date()})")
print(f"  baseline_weights : {weights_df.shape[0]} rebal dates Ã— "
      f"{weights_df.shape[1]} tickers")
print(f"  meta             : {len(meta)} stocks, sector col='{sector_col}'")

# =============================================================================
# 2. DERIVED SERIES (shared across figures)
# =============================================================================

# Cumulative log-return path
cum_ret = ret_series.cumsum()

# Price index: base = 1
price_idx = np.exp(cum_ret)

# Drawdown (as proportion, positive number = loss)
rolling_max = price_idx.cummax()
drawdown    = (rolling_max - price_idx) / rolling_max   # [0, 1], 0 = no drawdown

max_dd_val  = float(drawdown.max())
max_dd_date = drawdown.idxmax()

# Benchmark (S&P 500 via yfinance)
bench_cum = None
if HAS_YFINANCE:
    try:
        raw = yf.download(
            "^GSPC",
            start=DATE_START.strftime("%Y-%m-%d"),
            end=(DATE_END + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )
        if not raw.empty:
            # yfinance may return MultiIndex columns
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            close = raw["Close"].dropna()
            bench_log = np.log(close / close.shift(1)).dropna()
            bench_log.index = pd.DatetimeIndex(bench_log.index)
            bench_log = bench_log.reindex(ret_series.index).dropna()
            bench_cum = bench_log.cumsum().reindex(ret_series.index)
            print(f"  S&P 500 benchmark: {len(bench_log)} days downloaded")
        else:
            print("  yfinance returned empty data â€” benchmark omitted.")
    except Exception as e:
        print(f"  yfinance error ({e}) â€” benchmark omitted.")

# Turnover data
turnover_path = os.path.join(RESULTS_DIR, "baseline_turnover.csv")
if os.path.exists(turnover_path):
    turnover_df = pd.read_csv(turnover_path, parse_dates=["rebal_date"])
else:
    # Recompute from weights
    w_vals = weights_df.values
    rebal_list = weights_df.index.tolist()
    to_vals = [
        float(np.abs(w_vals[i] - w_vals[i - 1]).sum()) / 2
        for i in range(1, len(rebal_list))
    ]
    turnover_df = pd.DataFrame({
        "rebal_date": rebal_list[1:],
        "turnover"  : to_vals,
    })

turnover_df["rebal_date"] = pd.to_datetime(turnover_df["rebal_date"])
turnover_df = turnover_df.sort_values("rebal_date").reset_index(drop=True)
mean_turnover = float(turnover_df["turnover"].mean())

# =============================================================================
# FIGURE 1 â€” CUMULATIVE RETURNS
# =============================================================================
print("\n[1] Figure 1: Cumulative Returns â€¦")
try:
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 5))

    # Shade only the single maximum drawdown episode
    # Peak = last date where cum_ret reached its high before max DD trough
    peak_date = cum_ret.loc[:max_dd_date].idxmax()
    # Recovery = first date after trough where cum_ret exceeds the peak level
    peak_level   = float(cum_ret.loc[peak_date])
    post_trough  = cum_ret.loc[max_dd_date:]
    recovery_mask = post_trough > peak_level
    recovery_date = post_trough[recovery_mask].index[0] if recovery_mask.any() \
                    else cum_ret.index[-1]
    ax.axvspan(peak_date, recovery_date, color="red", alpha=0.10, zorder=1)

    # Max drawdown vertical line
    ax.axvline(max_dd_date, color="#C00000", linewidth=1.0,
               linestyle="--", alpha=0.7, zorder=3)
    ax.annotate(
        f"Max DD\n{max_dd_val:.1%}",
        xy=(max_dd_date, float(cum_ret.loc[max_dd_date])),
        xytext=(12, -30), textcoords="offset points",
        fontsize=8.5, color="#C00000",
        arrowprops=dict(arrowstyle="->", color="#C00000", lw=0.8),
    )

    # Benchmark
    if bench_cum is not None:
        bench_aligned = bench_cum.reindex(ret_series.index).ffill()
        ax.plot(bench_aligned.index, bench_aligned.values,
                color="#aaaaaa", linewidth=1.2, linestyle="--",
                label="S&P 500", zorder=4)

    # Baseline
    ax.plot(cum_ret.index, cum_ret.values,
            color="#1f3c6b", linewidth=1.8,
            label=r"Baseline (Historical Mean $\mu$)", zorder=5)

    ax.axhline(0, color="black", linewidth=0.6, linestyle=":", zorder=2)

    ax.set_title("Baseline Portfolio â€” Cumulative Returns (2023â€“2025)",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Cumulative Log Return", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.legend(fontsize=10, framealpha=0.85, edgecolor="#cccccc")
    fig.text(0.99, 0.01, "Source: Author's elaboration.",
             ha='right', va='bottom', fontsize=8,
             color='grey', style='italic',
             transform=fig.transFigure)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = savefig(fig, "baseline_cumulative_returns.png")
    final_ret = float(cum_ret.iloc[-1])
    print(f"  Saved: {path}")
    print(f"  Cumulative log return : {final_ret:.4f}  ({np.expm1(final_ret):.2%} simple)")
    print(f"  Max drawdown          : {max_dd_val:.2%}  on {max_dd_date.date()}")

except Exception as e:
    print(f"  ERROR in Figure 1: {e}")

# =============================================================================
# FIGURE 2 â€” ROLLING SHARPE
# =============================================================================
print("\n[2] Figure 2: Rolling Sharpe Ratio â€¦")
try:
    ROLL_WIN = 126
    setup_style()

    roll_mean  = ret_series.rolling(ROLL_WIN).mean() * 252
    roll_std   = ret_series.rolling(ROLL_WIN).std(ddof=1) * np.sqrt(252)
    roll_sharpe = (roll_mean / roll_std).dropna()

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.axhline(0,  color="black",  linewidth=0.7, linestyle=":",  zorder=2)
    ax.axhline(1,  color="#2ca02c", linewidth=1.0, linestyle="--",
               alpha=0.7, zorder=2, label="Sharpe = 1")
    ax.axhline(-1, color="#d62728", linewidth=0.8, linestyle="--",
               alpha=0.5, zorder=2)

    # Fill above/below zero
    ax.fill_between(
        roll_sharpe.index, roll_sharpe.values, 0,
        where=(roll_sharpe.values >= 0),
        color="#2ca02c", alpha=0.25, zorder=3, label="_nolegend_",
    )
    ax.fill_between(
        roll_sharpe.index, roll_sharpe.values, 0,
        where=(roll_sharpe.values < 0),
        color="#d62728", alpha=0.25, zorder=3, label="_nolegend_",
    )

    ax.plot(roll_sharpe.index, roll_sharpe.values,
            color="#1f3c6b", linewidth=1.4, zorder=4,
            label=f"Rolling {ROLL_WIN}-day Sharpe")

    ax.set_title(f"Baseline Portfolio â€” Rolling Sharpe Ratio ({ROLL_WIN}-day window)",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Sharpe Ratio", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.legend(fontsize=10, framealpha=0.85, edgecolor="#cccccc")
    fig.text(0.99, 0.01, "Source: Author's elaboration.",
             ha='right', va='bottom', fontsize=8,
             color='grey', style='italic',
             transform=fig.transFigure)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = savefig(fig, "baseline_rolling_sharpe.png")
    avg_rs = float(roll_sharpe.mean())
    pct_pos = float((roll_sharpe > 0).mean())
    print(f"  Saved: {path}")
    print(f"  Avg rolling Sharpe   : {avg_rs:.4f}")
    print(f"  % time Sharpe > 0    : {pct_pos:.1%}")
    print(f"  % time Sharpe > 1    : {(roll_sharpe > 1).mean():.1%}")

except Exception as e:
    print(f"  ERROR in Figure 2: {e}")

# =============================================================================
# FIGURE 3 â€” DRAWDOWN
# =============================================================================
print("\n[3] Figure 3: Drawdown â€¦")
try:
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 4))

    dd_neg = -drawdown   # plot as negative (drawdown below zero)

    ax.fill_between(dd_neg.index, dd_neg.values, 0,
                    color="#d62728", alpha=0.40, zorder=2)
    ax.plot(dd_neg.index, dd_neg.values,
            color="#C00000", linewidth=1.2, zorder=3)
    ax.axhline(0, color="black", linewidth=0.6, linestyle=":", zorder=2)

    # Annotate max drawdown
    ax.annotate(
        f"Max DD: {max_dd_val:.1%}\n{max_dd_date.strftime('%d %b %Y')}",
        xy=(max_dd_date, -max_dd_val),
        xytext=(20, -25), textcoords="offset points",
        fontsize=9, color="#C00000",
        arrowprops=dict(arrowstyle="->", color="#C00000", lw=0.8),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#C00000",
                  alpha=0.85),
    )

    ax.set_title("Baseline Portfolio â€” Drawdown (2023â€“2025)",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Drawdown", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    fig.text(0.99, 0.01, "Source: Author's elaboration.",
             ha='right', va='bottom', fontsize=8,
             color='grey', style='italic',
             transform=fig.transFigure)

    # Secondary stats
    dd_duration = int((drawdown > 0.01).sum())   # trading days in drawdown > 1%
    ax.text(
        0.01, 0.05,
        f"Days in drawdown >1%: {dd_duration}",
        transform=ax.transAxes, fontsize=8.5, color="#666666",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = savefig(fig, "baseline_drawdown.png")
    print(f"  Saved: {path}")
    print(f"  Max drawdown         : {max_dd_val:.2%}  on {max_dd_date.date()}")
    print(f"  Days in DD > 1%      : {dd_duration}")

except Exception as e:
    print(f"  ERROR in Figure 3: {e}")

# =============================================================================
# FIGURE 4 â€” TURNOVER
# =============================================================================
print("\n[4] Figure 4: Monthly Turnover â€¦")
try:
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 4))

    # Ensure datetime dtype (guard against string dates from CSV)
    dates  = pd.to_datetime(turnover_df["rebal_date"])
    to_val = turnover_df["turnover"].values
    colors = ["#2ca02c" if v < mean_turnover else "#FF6B6B" for v in to_val]

    plt.bar(dates, to_val, color=colors, width=15, align="center",
            alpha=0.70, zorder=3)

    # Value labels on top of each bar
    for d, v in zip(dates, to_val):
        ax.text(d, v + 0.003, f"{v:.0%}",
                ha="center", va="bottom", fontsize=7,
                rotation=90, color="#333333")

    ax.axhline(mean_turnover, color="#C00000", linewidth=1.2,
               linestyle="--", zorder=4)
    ax.text(
        dates.iloc[-1], mean_turnover + 0.005,
        f"Mean: {mean_turnover:.1%}",
        ha="right", va="bottom", fontsize=9, color="#C00000",
    )

    # Pad x-axis so first/last bars are not clipped
    ax.set_xlim(
        dates.iloc[0]  - pd.Timedelta(days=20),
        dates.iloc[-1] + pd.Timedelta(days=20),
    )

    ax.set_title("Baseline Portfolio â€” Monthly Turnover",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Rebalancing Date", fontsize=11)
    ax.set_ylabel("Turnover (one-way)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    fig.text(0.99, 0.01, "Source: Author's elaboration.",
             ha='right', va='bottom', fontsize=8,
             color='grey', style='italic',
             transform=fig.transFigure)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = savefig(fig, "baseline_turnover.png")
    print(f"  Saved: {path}")
    print(f"  Mean turnover        : {mean_turnover:.2%}")
    print(f"  Min / Max turnover   : {to_val.min():.2%} / {to_val.max():.2%}")

except Exception as e:
    print(f"  ERROR in Figure 4: {e}")

# =============================================================================
# FIGURE 5 â€” SECTOR HEATMAP
# =============================================================================
print("\n[5] Figure 5: Sector Allocation Heatmap â€¦")
try:
    setup_style()

    # Build sector weight matrix: rows = rebal dates, cols = sectors
    all_sectors = sorted(set(ticker_sector.values()))

    sector_rows = []
    for rebal_date, row in weights_df.iterrows():
        sec_wts = {s: 0.0 for s in all_sectors}
        for ticker, wt in row.items():
            ticker = str(ticker).strip().upper()
            if wt > 1e-6:
                s = ticker_sector.get(ticker, "Unknown")
                sec_wts[s] = sec_wts.get(s, 0.0) + float(wt)
        sec_wts["date"] = rebal_date
        sector_rows.append(sec_wts)

    sec_df = (
        pd.DataFrame(sector_rows)
        .set_index("date")
        .sort_index()
        [all_sectors]
    )

    # Row labels: "Jan 2023" style
    row_labels = [pd.Timestamp(d).strftime("%b %Y") for d in sec_df.index]

    fig_h  = max(8, len(sec_df) * 0.55)
    fig, ax = plt.subplots(figsize=(14, fig_h))

    if HAS_SEABORN:
        # Build annotation matrix: "18%" or "" for < 1%
        annot = sec_df.applymap(
            lambda v: f"{v:.0%}" if v >= 0.01 else ""
        )
        sns.heatmap(
            sec_df,
            ax=ax,
            cmap="YlOrRd",
            vmin=0.0,
            vmax=0.30,
            annot=annot,
            fmt="",
            linewidths=0.4,
            linecolor="#dddddd",
            cbar_kws={"label": "Portfolio Weight", "shrink": 0.6},
            xticklabels=all_sectors,
            yticklabels=row_labels,
        )
    else:
        im = ax.imshow(
            sec_df.values,
            aspect="auto",
            cmap="YlOrRd",
            vmin=0.0,
            vmax=0.30,
            interpolation="nearest",
        )
        plt.colorbar(im, ax=ax, label="Portfolio Weight", shrink=0.6)
        ax.set_xticks(range(len(all_sectors)))
        ax.set_xticklabels(all_sectors, rotation=35, ha="right", fontsize=9)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=9)
        for i in range(len(sec_df)):
            for j in range(len(all_sectors)):
                v = sec_df.values[i, j]
                if v >= 0.01:
                    ax.text(j, i, f"{v:.0%}",
                            ha="center", va="center",
                            fontsize=7.5,
                            color="black" if v < 0.20 else "white")

    ax.set_title("Baseline Portfolio â€” Sector Allocation Over Time",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("GICS Sector", fontsize=11)
    ax.set_ylabel("Rebalancing Date", fontsize=11)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=9)
    ax.tick_params(axis="y", labelsize=9)

    fig.text(0.99, 0.01, "Source: Author's elaboration.",
             ha='right', va='bottom', fontsize=8,
             color='grey', style='italic',
             transform=fig.transFigure)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = savefig(fig, "baseline_sector_heatmap.png")
    avg_concentration = float(sec_df.max(axis=1).mean())
    top_sector = sec_df.mean().idxmax()
    print(f"  Saved: {path}")
    print(f"  Avg top-sector weight per period : {avg_concentration:.2%}")
    print(f"  Most-allocated sector (avg)      : {top_sector}")
    print(f"  Periods hitting sector cap (30%) : "
          f"{int((sec_df >= 0.299).any(axis=1).sum())}")

except Exception as e:
    print(f"  ERROR in Figure 5: {e}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("  STEP VIZ â€” BASELINE DIAGNOSTICS COMPLETE")
print("=" * 70)
print("  Figures written to: data/figures/")
for fname in [
    "baseline_cumulative_returns.png",
    "baseline_rolling_sharpe.png",
    "baseline_drawdown.png",
    "baseline_turnover.png",
    "baseline_sector_heatmap.png",
]:
    full = os.path.join(FIGURES_DIR, fname)
    status = "OK " if os.path.exists(full) else "MISSING"
    print(f"    [{status}] {fname}")
print("=" * 70)
