"""
Three-panel figure (Sharpe / Max Drawdown / Avg Turnover vs lambda) built
from the four frontier_summary.csv files produced by 3b, 6a, 6b, 6c across
the {0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0} lambda grid. Main comparison is
Baseline vs XGBoost; Ridge and MLP are plotted as secondary lines for a
robustness check.

Writes figures/step6/lambda_sensitivity_baseline_vs_xgboost.png.

Author: Anila Vata
Project: MSc Thesis -- ML-Enhanced Portfolio Optimization with SAFE AI Evaluation
         University of Pavia, Supervisor: Prof. Paolo Giudici
"""

import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures", "step6")
os.makedirs(FIGURES_DIR, exist_ok=True)

PATHS = {
    "baseline": os.path.join(RESULTS_DIR, "step3",
                             "baseline_frontier_summary.csv"),
    "xgboost":  os.path.join(RESULTS_DIR, "step6", "xgboost", "performance",
                             "xgboost_frontier_summary.csv"),
    "ridge":    os.path.join(RESULTS_DIR, "step6", "ridge", "performance",
                             "ridge_frontier_summary.csv"),
    "mlp":      os.path.join(RESULTS_DIR, "step6", "mlp", "performance",
                             "mlp_frontier_summary.csv"),
}
OUT_PATH = os.path.join(FIGURES_DIR, "lambda_sensitivity_baseline_vs_xgboost.png")

expected_lambdas = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

dfs = {}
for name, fpath in PATHS.items():
    if not os.path.exists(fpath):
        raise FileNotFoundError(
            f"Required frontier summary not found: {fpath}\n"
            f"Run the corresponding step script first."
        )
    df = pd.read_csv(fpath).sort_values("lambda").reset_index(drop=True)
    if list(df["lambda"]) != expected_lambdas:
        raise ValueError(
            f"[{name}] Lambda values do not match expected grid.\n"
            f"  Expected : {expected_lambdas}\n"
            f"  Got      : {list(df['lambda'])}"
        )
    dfs[name] = df
    print(f"  Loaded {name:<10}: {len(df)} rows")

lam = dfs["baseline"]["lambda"].values
print(f"  Lambda grid : {lam.tolist()}")

plt.rcParams.update({
    "font.family"       : "serif",
    "font.serif"        : ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size"         : 11,
    "axes.titlesize"    : 12,
    "axes.labelsize"    : 11,
    "xtick.labelsize"   : 10,
    "ytick.labelsize"   : 10,
    "legend.fontsize"   : 9,
    "figure.facecolor"  : "white",
    "axes.facecolor"    : "white",
    "axes.grid"         : False,
})

# Main comparison -- prominent
STYLE_MAIN = {
    "baseline": dict(color="#1f77b4", linestyle="--", linewidth=1.8,
                     marker="o", markersize=6,  zorder=4),
    "xgboost":  dict(color="#d62728", linestyle="-",  linewidth=1.8,
                     marker="s", markersize=6,  zorder=4),
}
# Robustness check, subordinate
STYLE_ROB = {
    "ridge":    dict(color="#2ca02c", linestyle=":",  linewidth=1.1,
                     marker=None, alpha=0.75, zorder=3),
    "mlp":      dict(color="#ff7f0e", linestyle=":",  linewidth=1.1,
                     marker=None, alpha=0.75, zorder=3),
}

GREY = "#888888"

panels = [
    ("sharpe",       "Sharpe Ratio",                False),
    ("max_drawdown", "Maximum Drawdown (%)",         True),
    ("avg_turnover", "Average Monthly Turnover (%)", True),
]

fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
fig.subplots_adjust(hspace=0.38)

for ax, (col, ylabel, as_pct) in zip(axes, panels):

    # --- robustness lines first (behind main) ---
    for name, style in STYLE_ROB.items():
        vals = dfs[name][col].values
        if as_pct:
            vals = vals * 100
        ax.plot(lam, vals, **style)

    # --- main comparison lines on top ---
    for name, style in STYLE_MAIN.items():
        vals = dfs[name][col].values
        if as_pct:
            vals = vals * 100
        ax.plot(lam, vals, **style)

    ax.axvline(x=1.0, color=GREY, linestyle="--", linewidth=0.9, zorder=1)

    ax.set_xscale("log")
    ax.set_ylabel(ylabel)
    ax.set_xticks(lam)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[2].set_xlabel(r"Risk-aversion parameter $\lambda$")
axes[0].set_title(
    r"Lambda Sensitivity: Baseline vs XGBoost (Ridge, MLP as robustness check)",
    pad=10, fontsize=11
)

# λ=1 annotation on the top panel, placed after all y-data is drawn
ylo, yhi = axes[0].get_ylim()
axes[0].text(
    1.0, ylo + 0.01 * (yhi - ylo),
    r"$\lambda=1$",
    color=GREY, fontsize=8.5, ha="center", va="bottom"
)

legend_handles = [
    # main comparison group
    Line2D([0], [0], color="#1f77b4", linestyle="--", linewidth=1.8,
           marker="o", markersize=5, label="Historical Mean Baseline"),
    Line2D([0], [0], color="#d62728", linestyle="-",  linewidth=1.8,
           marker="s", markersize=5, label="XGBoost"),
    # spacer + robustness group
    Line2D([0], [0], color="none", label=""),
    Line2D([0], [0], color="#2ca02c", linestyle=":", linewidth=1.1,
           alpha=0.85, label="Ridge (robustness)"),
    Line2D([0], [0], color="#ff7f0e", linestyle=":", linewidth=1.1,
           alpha=0.85, label="MLP (robustness)"),
]
axes[0].legend(
    handles=legend_handles,
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(1.01, 1.0),
    borderaxespad=0,
    fontsize=9,
)

fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"\n  Saved: {OUT_PATH}")

CSV_OUT = os.path.join(RESULTS_DIR, "step6", "lambda_sensitivity_results.csv")

rows = []
for model_name, df in dfs.items():
    for _, row in df.iterrows():
        rows.append({
            "lambda":        row["lambda"],
            "model":         model_name,
            "sharpe":        row["sharpe"],
            "max_drawdown":  row["max_drawdown"],
            "avg_turnover":  row["avg_turnover"],
        })

sensitivity_df = pd.DataFrame(rows, columns=["lambda", "model", "sharpe",
                                               "max_drawdown", "avg_turnover"])
sensitivity_df = sensitivity_df.sort_values(["lambda", "model"]).reset_index(drop=True)
sensitivity_df.to_csv(CSV_OUT, index=False)

print(f"  Saved CSV: {CSV_OUT}  ({len(sensitivity_df)} rows)")
print("  Done.")
