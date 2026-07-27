"""
10a_appendix.py
Generates all supplementary outputs for the thesis appendix.
Reads from existing result files only - no source files are modified.
"""

import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "outputs", "step11")
os.makedirs(OUT_DIR, exist_ok=True)

LAMBDA_CSV = os.path.join(BASE, "data", "results", "step6", "lambda_sensitivity_results.csv")
FRONTIER_CSV = os.path.join(BASE, "data", "results", "step10", "safe_performance_frontier_50x3.csv")
SUMMARY_CSV = os.path.join(BASE, "data", "results", "step10", "safe_performance_summary.csv")

print("=" * 60)
print("STEP 0 -- Inspection phase")
print("=" * 60)

df_lambda = pd.read_csv(LAMBDA_CSV)
print(f"\n[lambda_sensitivity_results.csv]  path: {LAMBDA_CSV}")
print("Columns:", df_lambda.columns.tolist())
print(df_lambda.head())

df_frontier = pd.read_csv(FRONTIER_CSV)
print(f"\n[safe_performance_frontier_50x3.csv]  path: {FRONTIER_CSV}")
print("Columns:", df_frontier.columns.tolist())
print(df_frontier.head())

df_summary = pd.read_csv(SUMMARY_CSV)
print(f"\n[safe_performance_summary.csv]  path: {SUMMARY_CSV}")
print("Columns:", df_summary.columns.tolist())
print(df_summary.head())

print("\n" + "=" * 60)
print("OUTPUT 1 - Lambda sensitivity tables")
print("=" * 60)

LAMBDAS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
MODEL_ORDER = ["baseline", "ridge", "mlp", "xgboost"]
COL_NAMES = {"baseline": "Baseline", "ridge": "Ridge", "mlp": "MLP", "xgboost": "XGBoost"}

def build_pivot(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pivot lambda × model for a single metric."""
    piv = df.pivot(index="lambda", columns="model", values=metric)
    piv = piv.loc[LAMBDAS, MODEL_ORDER]
    piv.columns = [COL_NAMES[c] for c in piv.columns]
    piv.index.name = "Lambda"
    return piv

def mark_best(piv: pd.DataFrame, higher_is_better: bool) -> pd.DataFrame:
    """Return a copy of piv with '*' appended to the best value in each row."""
    result = piv.copy().astype(object)
    for idx in piv.index:
        row = piv.loc[idx]
        best_col = row.idxmax() if higher_is_better else row.idxmin()
        for col in piv.columns:
            val = piv.loc[idx, col]
            formatted = f"{val:.4f}"
            result.loc[idx, col] = formatted + ("*" if col == best_col else "")
    return result

# --- Sharpe (higher is better) ---
piv_sharpe = build_pivot(df_lambda, "sharpe")
tbl_sharpe = mark_best(piv_sharpe, higher_is_better=True)
tbl_sharpe.to_csv(os.path.join(OUT_DIR, "appendix_table_A1_sharpe.csv"))
print("  Saved appendix_table_A1_sharpe.csv")

# --- Max Drawdown (lower is better) ---
piv_maxdd = build_pivot(df_lambda, "max_drawdown")
tbl_maxdd = mark_best(piv_maxdd, higher_is_better=False)
tbl_maxdd.to_csv(os.path.join(OUT_DIR, "appendix_table_A2_maxdd.csv"))
print("  Saved appendix_table_A2_maxdd.csv")

# --- Avg Turnover (lower is better) ---
piv_turnover = build_pivot(df_lambda, "avg_turnover")
tbl_turnover = mark_best(piv_turnover, higher_is_better=False)
tbl_turnover.to_csv(os.path.join(OUT_DIR, "appendix_table_A3_turnover.csv"))
print("  Saved appendix_table_A3_turnover.csv")

# --- Combined Excel ---
xlsx_path = os.path.join(OUT_DIR, "appendix_lambda_tables.xlsx")
with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
    tbl_sharpe.to_excel(writer, sheet_name="Sharpe")
    tbl_maxdd.to_excel(writer, sheet_name="MaxDrawdown")
    tbl_turnover.to_excel(writer, sheet_name="AvgTurnover")
print("  Saved appendix_lambda_tables.xlsx")

print("\n" + "=" * 60)
print("OUTPUT 2 -- Pipeline flowchart")
print("=" * 60)

STEPS = [
    "Bloomberg Raw Data\n(2010–2025)",
    "Data Cleaning & Universe Filters\n(407 stocks)",
    "Feature Engineering\n(10 features)",
    "Return Estimation Models\n(Baseline / Ridge / MLP / XGBoost)",
    "MIQP Portfolio Optimization\n(Gurobi 13)",
    "Out-of-Sample Portfolio Evaluation\n(Jan 2023 – Dec 2025)",
    "SAFE AI Model-Level Evaluation\n(RGS, RGA, RGF, RGE)",
    "SAFE-Performance Frontier\n(150 configurations)",
]

NAVY = "#1F3864"
ARROW_COLOR = "#404040"
BOX_W = 0.62
BOX_H = 0.07
X_CENTER = 0.5
GAP = 0.025  # gap between bottom of one box and top of next

fig, ax = plt.subplots(figsize=(6, 10), facecolor="white")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

n = len(STEPS)
# total height used by boxes and gaps
total_content = n * BOX_H + (n - 1) * GAP
top_margin = 0.04
bottom_margin = 0.04
available = 1.0 - top_margin - bottom_margin

# scale so everything fits
scale = min(1.0, available / total_content)
box_h = BOX_H * scale
gap = GAP * scale

# top of first box
y_top = 1.0 - top_margin

for i, label in enumerate(STEPS):
    y_box_top = y_top - i * (box_h + gap)
    y_box_bot = y_box_top - box_h
    y_center = (y_box_top + y_box_bot) / 2

    rect = mpatches.FancyBboxPatch(
        (X_CENTER - BOX_W / 2, y_box_bot),
        BOX_W,
        box_h,
        boxstyle="round,pad=0.012",
        facecolor=NAVY,
        edgecolor="white",
        linewidth=0.8,
        zorder=2,
    )
    ax.add_patch(rect)
    ax.text(
        X_CENTER,
        y_center,
        label,
        ha="center",
        va="center",
        color="white",
        fontsize=8.5,
        fontfamily="DejaVu Sans",
        zorder=3,
        linespacing=1.35,
    )

    # Arrow from bottom of this box to top of next box
    if i < n - 1:
        arrow_y_start = y_box_bot - 0.003
        arrow_y_end = y_box_bot - gap + 0.003
        ax.annotate(
            "",
            xy=(X_CENTER, arrow_y_end),
            xytext=(X_CENTER, arrow_y_start),
            arrowprops=dict(
                arrowstyle="-|>",
                color=ARROW_COLOR,
                lw=1.4,
                mutation_scale=12,
            ),
            zorder=1,
        )

flowchart_path = os.path.join(OUT_DIR, "figure_A1_pipeline_flowchart.png")
plt.savefig(flowchart_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved figure_A1_pipeline_flowchart.png")

caption_A1 = (
    "Figure A1. Empirical pipeline from raw financial data to the SAFE-Performance Frontier."
)
with open(os.path.join(OUT_DIR, "figure_A1_caption.txt"), "w", encoding="utf-8") as f:
    f.write(caption_A1)
print("  Saved figure_A1_caption.txt")

print("\n" + "=" * 60)
print("OUTPUT 3 - SAFE-Performance scatter")
print("=" * 60)

# Extract Spearman rho for all_models × Arithmetic × sharpe
mask = (
    (df_summary["figure_scope"] == "all_models")
    & (df_summary["compliance_score_type"] == "Arithmetic")
    & (df_summary["performance_metric"] == "sharpe")
)
spearman_rho = float(df_summary.loc[mask, "spearman_rho"].values[0])
spearman_p = float(df_summary.loc[mask, "spearman_p_value"].values[0])
print(f"  Spearman rho (Arithmetic x Sharpe, all models): {spearman_rho:.4f}  p={spearman_p:.4e}")

# Colour map
COLOR_MAP = {
    "baseline": "#888888",
    "ridge": "#1f77b4",
    "mlp": "#ff7f0e",
    "xgboost": "#2ca02c",
}

fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")

for family, grp in df_frontier.groupby("model_family"):
    color = COLOR_MAP.get(family, "black")
    ax.scatter(
        grp["sharpe"],
        grp["compliance_score_arithmetic"],
        c=color,
        s=45,
        alpha=0.75,
        edgecolors="none",
        label=family.capitalize(),
        zorder=3,
    )

# Pareto-dominant point
pareto_row = df_frontier[df_frontier["configuration_id"] == "xgboost_47"]
if not pareto_row.empty:
    px = float(pareto_row["sharpe"].values[0])
    py = float(pareto_row["compliance_score_arithmetic"].values[0])
    ax.scatter(px, py, marker="*", s=280, c="red", zorder=5, label="Pareto-dominant (xgboost_47)")
    ax.annotate(
        "xgboost_47",
        xy=(px, py),
        xytext=(px + 0.04, py - 0.018),
        fontsize=8,
        color="red",
        arrowprops=dict(arrowstyle="-", color="red", lw=0.8),
        zorder=6,
    )

# LOWESS smoothing line
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess

    x_all = df_frontier["sharpe"].values
    y_all = df_frontier["compliance_score_arithmetic"].values
    order = np.argsort(x_all)
    smoothed = lowess(y_all[order], x_all[order], frac=0.55, return_sorted=True)
    ax.plot(
        smoothed[:, 0],
        smoothed[:, 1],
        color="#404040",
        lw=1.5,
        linestyle="--",
        label="LOWESS (frac=0.55)",
        zorder=4,
    )
except ImportError:
    print("  [WARNING] statsmodels not available—LOWESS line skipped")

# Spearman annotation
ax.text(
    0.03,
    0.97,
    f"Spearman ρ = {spearman_rho:.4f}",
    transform=ax.transAxes,
    fontsize=9,
    va="top",
    ha="left",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.85),
)

ax.set_xlabel("Sharpe Ratio", fontsize=11)
ax.set_ylabel("CS4 Compliance Score (Arithmetic)", fontsize=11)
ax.set_title("SAFE-Performance Frontier (150 configurations)", fontsize=12, pad=10)
ax.legend(fontsize=8.5, loc="lower right", framealpha=0.85)
ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
ax.set_facecolor("white")

fig.tight_layout()
scatter_path = os.path.join(OUT_DIR, "figure_A2_safe_performance_frontier.png")
plt.savefig(scatter_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved figure_A2_safe_performance_frontier.png")

caption_A2 = (
    "Figure A2. SAFE-Performance Frontier across 150 model configurations. "
    "Each point represents one configuration. The dashed line shows the LOWESS smoothing curve. "
    "The red star marks the Pareto-dominant configuration (XGBoost, config 47)."
)
with open(os.path.join(OUT_DIR, "figure_A2_caption.txt"), "w", encoding="utf-8") as f:
    f.write(caption_A2)
print("  Saved figure_A2_caption.txt")

print("\n" + "=" * 60)
print("OUTPUT 4 -- Software table")
print("=" * 60)

software_rows = [
    ("Python 3.x", "Data processing, modelling, and visualization"),
    ("pandas / NumPy", "Data manipulation and numerical computation"),
    ("scikit-learn", "Ridge Regression and Neural Network"),
    ("XGBoost", "Gradient boosting return prediction"),
    ("Gurobi 13", "Constrained MIQP portfolio optimization"),
    ("safeaipackage", "SAFE AI framework evaluation"),
    ("Bloomberg Terminal", "Primary financial data source (Jan 2010 – Dec 2025)"),
]
df_software = pd.DataFrame(software_rows, columns=["Tool", "Use"])
software_path = os.path.join(OUT_DIR, "appendix_table_D1_software.csv")
df_software.to_csv(software_path, index=False)
print("  Saved appendix_table_D1_software.csv")

print("""
10a_appendix.py completed.
Outputs saved to outputs/step11/:
  - appendix_table_A1_sharpe.csv
  - appendix_table_A2_maxdd.csv
  - appendix_table_A3_turnover.csv
  - appendix_lambda_tables.xlsx
  - figure_A1_pipeline_flowchart.png
  - figure_A2_safe_performance_frontier.png
  - appendix_table_D1_software.csv
""")
