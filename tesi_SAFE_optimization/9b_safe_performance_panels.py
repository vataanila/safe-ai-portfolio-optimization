"""
9b_safe_performance_panels.py
===================================
Generates compact three-panel SAFE-Performance figures for the thesis.
Reads only the existing CSV produced by 9a_safe_frontier_50x3.py. No retraining. No MIQP.

For each performance metric (sharpe, max_drawdown, avg_turnover) creates
four figures, each with three aligned panels (arithmetic / geometric / RMS):

  A.  all_models  --  Ridge + XGBoost + MLP together
  B.  ridge       --  Ridge only
  C.  xgboost     --  XGBoost only
  D.  mlp         --  MLP only

Outputs
-------
  figures/step10h/sharpe/all_models.png
  figures/step10h/sharpe/ridge.png
  ...
  figures/step10h/max_drawdown/all_models.png
  ...
  figures/step10h/avg_turnover/all_models.png
  ...
  data/results/step10h/safe_performance_summary.csv

Author: Anila Vata
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

matplotlib.use("Agg")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CSV_PATH  = os.path.join(BASE_DIR, "data", "results", "step10",
                         "safe_performance_frontier_50x3.csv")
FIG_BASE  = os.path.join(BASE_DIR, "figures", "step10h")
TABLE_DIR = os.path.join(BASE_DIR, "data", "results", "step10h")

# ── Analysis grid ──────────────────────────────────────────────────────────────
SCORE_COLS = [
    "compliance_score_arithmetic",
    "compliance_score_geometric",
    "compliance_score_rms",
]
SCORE_SHORT = {
    "compliance_score_arithmetic": "Arithmetic",
    "compliance_score_geometric":  "Geometric",
    "compliance_score_rms":        "RMS",
}
SCORE_LABEL = {
    "compliance_score_arithmetic": "Arithmetic Compliance Score",
    "compliance_score_geometric":  "Geometric Compliance Score",
    "compliance_score_rms":        "RMS Compliance Score",
}

PERF_METRICS = {
    "sharpe":       "higher",
    "max_drawdown": "lower",
    "avg_turnover": "lower",
}
PERF_LABEL = {
    "sharpe":       "Annualized Sharpe Ratio",
    "max_drawdown": "Maximum Drawdown",
    "avg_turnover": "Average Turnover",
}

# ── Style ──────────────────────────────────────────────────────────────────────
FAMILY_ORDER  = ["ridge", "xgboost", "mlp"]
FAMILY_LABELS = {"ridge": "Ridge", "xgboost": "XGBoost", "mlp": "MLP"}
FAMILY_COLORS = {"ridge": "#2166AC", "xgboost": "#B2182B", "mlp": "#4DAF4A"}

COLOR_BEST_SAFE = "gold"
COLOR_BEST_PERF = "#FF7F00"
COLOR_BOTH      = "gold"
COLOR_TREND     = "#555555"

MARKER_BEST_SAFE = "D"
MARKER_BEST_PERF = "*"
MARKER_BOTH      = "*"

DPI = 300

def load_results() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    for col in SCORE_COLS + list(PERF_METRICS.keys()):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=SCORE_COLS + list(PERF_METRICS.keys()))
    print(f"Loaded {len(df)} rows  |  families: {sorted(df['model_family'].unique())}")
    return df

def get_best_rows(sub: pd.DataFrame, score_col: str,
                  metric_col: str, direction: str) -> tuple:
    """
    Return (row_best_safe, row_best_perf, same_flag).
    best_safe  = highest value of score_col within sub.
    best_perf  = max or min of metric_col depending on direction.
    """
    idx_safe = sub[score_col].idxmax()
    idx_perf = (sub[metric_col].idxmax() if direction == "higher"
                else sub[metric_col].idxmin())
    same = (idx_safe == idx_perf)
    return sub.loc[idx_safe], sub.loc[idx_perf], same

def add_model_curve(ax, x: np.ndarray, y: np.ndarray, color: str,
                    frac: float = 0.6):
    """
    Draw a LOWESS curve (sm_lowess, frac=0.6, it=3) for all families.
    Falls back to a linear fit only if statsmodels is unavailable or raises.
    """
    if len(x) < 4:
        return

    from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
    try:
        smoothed = sm_lowess(y, x, frac=frac, it=3, return_sorted=True)
        ax.plot(smoothed[:, 0], smoothed[:, 1],
                color=color, linewidth=2.5, linestyle="-",
                alpha=1.0, zorder=2)
    except Exception:
        try:
            coeffs = np.polyfit(x, y, deg=1)
            xfit   = np.linspace(x.min(), x.max(), 200)
            ax.plot(xfit, np.polyval(coeffs, xfit),
                    color=color, linewidth=2.0, linestyle="-",
                    alpha=1.0, zorder=2)
        except Exception:
            pass

def _annotate(ax, x: float, y: float, label: str, xytext: tuple):
    ax.annotate(
        label,
        xy=(x, y),
        xytext=xytext,
        textcoords="offset points",
        fontsize=10,
        color="#222222",
        va="center",
        arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6),
    )

def _spearman_text(x: np.ndarray, y: np.ndarray) -> str:
    rho, p = stats.spearmanr(x, y)
    star   = "*" if p < 0.05 else ""
    return f"$\\rho_s$={rho:+.3f}{star}\n$p$={p:.3f}"

def plot_panel(ax, sub: pd.DataFrame, score_col: str,
               metric_col: str, direction: str,
               panel_title: str, scope: str,
               trend_type: str = "lowess", lowess_frac: float = 0.45):
    """
    Draw one panel (scatter + trend + highlights + Spearman).

    scope: 'all_models' or a family key ('ridge', 'xgboost', 'mlp').
    """
    x = sub[score_col].values
    y = sub[metric_col].values

    # ── Scatter ──
    if scope == "all_models":
        for fam in FAMILY_ORDER:
            fsub = sub[sub["model_family"] == fam]
            if fsub.empty:
                continue
            ax.scatter(fsub[score_col], fsub[metric_col],
                       color=FAMILY_COLORS[fam], s=20, alpha=0.15,
                       linewidths=0, zorder=3)
    else:
        ax.scatter(sub[score_col], sub[metric_col],
                   color=FAMILY_COLORS[scope], s=20, alpha=0.15,
                   linewidths=0, zorder=3)

    # ── Trend curves ──
    if scope == "all_models":
        for fam in FAMILY_ORDER:
            fsub = sub[sub["model_family"] == fam]
            if len(fsub) < 4:
                continue
            add_model_curve(ax,
                            fsub[score_col].values,
                            fsub[metric_col].values,
                            color=FAMILY_COLORS[fam],
                            frac=lowess_frac)
    else:
        add_model_curve(ax, x, y,
                        color=FAMILY_COLORS[scope],
                        frac=lowess_frac)

    # ── Best points ──
    row_safe, row_perf, same = get_best_rows(sub, score_col, metric_col, direction)
    cfg_safe = row_safe["configuration_id"]
    cfg_perf = row_perf["configuration_id"]
    xs, ys   = float(row_safe[score_col]), float(row_safe[metric_col])
    xp, yp   = float(row_perf[score_col]), float(row_perf[metric_col])

    if same:
        ax.scatter(xs, ys, marker=MARKER_BOTH, s=120, color=COLOR_BOTH,
                   edgecolors="black", linewidths=0.8, zorder=7)
        _annotate(ax, xs, ys, f"{cfg_safe}\n(Best SAFE & Perf.)", (6, 9))
    else:
        ax.scatter(xs, ys, marker=MARKER_BEST_SAFE, s=120, color=COLOR_BEST_SAFE,
                   edgecolors="black", linewidths=0.7, zorder=7)
        _annotate(ax, xs, ys, f"{cfg_safe}\n(Best SAFE)", (6, 9))

        ax.scatter(xp, yp, marker=MARKER_BEST_PERF, s=120, color=COLOR_BEST_PERF,
                   edgecolors="black", linewidths=0.8, zorder=7)
        _annotate(ax, xp, yp, f"{cfg_perf}\n(Best Perf.)", (6, -12))

    # ── Spearman annotation ──
    ax.text(0.04, 0.97, _spearman_text(x, y),
            transform=ax.transAxes, fontsize=14, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#CCCCCC", alpha=0.85))

    # ── Axes style ──
    ax.set_xlabel("")  # shared xlabel placed at figure level
    ax.set_title(f"{SCORE_SHORT[score_col]} CS", fontsize=16, pad=5)
    ax.grid(True, alpha=0.22, linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=14)

def _legend_handles(scope: str, same: bool) -> list:
    """Build legend handle list appropriate for this scope."""
    handles = []

    if scope == "all_models":
        for fam in FAMILY_ORDER:
            handles.append(Line2D([0], [0], marker="o", color="w",
                                   markerfacecolor=FAMILY_COLORS[fam],
                                   markersize=12, markeredgewidth=0,
                                   label=FAMILY_LABELS[fam]))
        for fam in FAMILY_ORDER:
            handles.append(Line2D([0], [0], linestyle="-",
                                   color=FAMILY_COLORS[fam],
                                   linewidth=2.0, alpha=1.0,
                                   label=f"{FAMILY_LABELS[fam]} trend"))
    else:
        handles.append(Line2D([0], [0], marker="o", color="w",
                               markerfacecolor=FAMILY_COLORS[scope],
                               markersize=12, markeredgewidth=0,
                               label=FAMILY_LABELS[scope]))
        handles.append(Line2D([0], [0], linestyle="-",
                               color=FAMILY_COLORS[scope],
                               linewidth=2.5, alpha=1.0,
                               label=f"{FAMILY_LABELS[scope]} trend"))

    if same:
        handles.append(Line2D([0], [0], marker=MARKER_BOTH, color="w",
                               markerfacecolor=COLOR_BOTH,
                               markeredgecolor="black", markeredgewidth=0.8,
                               markersize=12, label="Best SAFE & Best Perf."))
    else:
        handles.append(Line2D([0], [0], marker=MARKER_BEST_SAFE, color="w",
                               markerfacecolor=COLOR_BEST_SAFE,
                               markeredgecolor="black", markeredgewidth=0.7,
                               markersize=12, label="Best SAFE"))
        handles.append(Line2D([0], [0], marker=MARKER_BEST_PERF, color="w",
                               markerfacecolor=COLOR_BEST_PERF,
                               markeredgecolor="black", markeredgewidth=0.8,
                               markersize=12, label="Best Perf."))
    return handles

def plot_three_panel_figure(df: pd.DataFrame, metric_col: str,
                             direction: str, scope: str, out_path: str):
    """
    Three-panel figure: one panel per compliance score aggregation.
    scope: 'all_models' | 'ridge' | 'xgboost' | 'mlp'
    """
    sub = df if scope == "all_models" else df[df["model_family"] == scope].copy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    for ax in axes:
        ax.set_box_aspect(1)

    # Determine same-flag globally (use arithmetic as reference for legend)
    _, _, same_ref = get_best_rows(sub, "compliance_score_arithmetic",
                                    metric_col, direction)

    lowess_frac = 0.6
    for ax, score_col in zip(axes, SCORE_COLS):
        plot_panel(ax, sub, score_col, metric_col, direction,
                   SCORE_SHORT[score_col], scope,
                   trend_type="lowess", lowess_frac=lowess_frac)

    # y-label only on leftmost panel (sharey=True)
    axes[0].set_ylabel(PERF_LABEL[metric_col], fontsize=16)

    # Shared xlabel centered below all panels
    fig.text(0.5, 0.01, "CS4 Integrated Compliance Score",
             ha="center", va="bottom", fontsize=16)

    # Shared legend above the xlabel
    handles = _legend_handles(scope, same_ref)
    fig.legend(handles=handles, loc="lower center",
               ncol=len(handles), fontsize=14,
               bbox_to_anchor=(0.5, 0.07), framealpha=0.85)

    plt.tight_layout(rect=[0, 0.17, 1, 1], w_pad=2.0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {os.path.relpath(out_path, BASE_DIR)}")

def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    scopes = ["all_models"] + FAMILY_ORDER
    rows   = []

    for scope in scopes:
        sub = df if scope == "all_models" else df[df["model_family"] == scope]

        for score_col in SCORE_COLS:
            for metric_col, direction in PERF_METRICS.items():
                clean = sub.dropna(subset=[score_col, metric_col])
                if clean.empty:
                    continue

                rho, pval     = stats.spearmanr(clean[score_col], clean[metric_col])
                row_s, row_p, same = get_best_rows(clean, score_col,
                                                    metric_col, direction)
                rows.append({
                    "figure_scope":                  scope,
                    "compliance_score_type":         SCORE_SHORT[score_col],
                    "performance_metric":            metric_col,
                    "n_configurations":              len(clean),
                    "spearman_rho":                  round(float(rho), 4),
                    "spearman_p_value":              round(float(pval), 4),
                    "best_safe_configuration_id":    row_s["configuration_id"],
                    "best_safe_model_family":        row_s["model_family"],
                    "best_safe_compliance_score":    round(float(row_s[score_col]), 6),
                    "best_safe_performance_value":   round(float(row_s[metric_col]), 4),
                    "best_perf_configuration_id":    row_p["configuration_id"],
                    "best_perf_model_family":        row_p["model_family"],
                    "best_perf_compliance_score":    round(float(row_p[score_col]), 6),
                    "best_performance_value":        round(float(row_p[metric_col]), 4),
                    "same_configuration_flag":       same,
                })

    return pd.DataFrame(rows)

def print_console_summary(summary: pd.DataFrame):
    sep = "-" * 68
    print(f"\n{'=' * 68}")
    print("STEP 9B -- SAFE-Performance Panels: Console Summary")
    print(f"{'=' * 68}")

    for metric_col in PERF_METRICS:
        print(f"\n{sep}")
        print(f"Metric: {PERF_LABEL[metric_col]}")
        print(sep)

        sub = summary[summary["performance_metric"] == metric_col]

        # Global (all models, arithmetic CS as reference)
        row_g = sub[(sub["figure_scope"] == "all_models") &
                    (sub["compliance_score_type"] == "Arithmetic")]
        if not row_g.empty:
            r = row_g.iloc[0]
            flag = "YES -- same configuration" if r["same_configuration_flag"] \
                   else "NO  -- different configurations"
            print(f"  [All models, Arithmetic]  Best SAFE == Best Perf? {flag}")
            print(f"    Best SAFE : {r['best_safe_configuration_id']} "
                  f"({r['best_safe_model_family']})  "
                  f"compliance={r['best_safe_compliance_score']:.4f}  "
                  f"{metric_col}={r['best_safe_performance_value']:.4f}")
            print(f"    Best Perf : {r['best_perf_configuration_id']} "
                  f"({r['best_perf_model_family']})  "
                  f"compliance={r['best_perf_compliance_score']:.4f}  "
                  f"{metric_col}={r['best_performance_value']:.4f}")

        # Per family (arithmetic CS)
        for fam in FAMILY_ORDER:
            row_f = sub[(sub["figure_scope"] == fam) &
                        (sub["compliance_score_type"] == "Arithmetic")]
            if row_f.empty:
                continue
            r    = row_f.iloc[0]
            flag = "YES" if r["same_configuration_flag"] else "NO"
            note = ""
            if fam == "xgboost" and metric_col == "sharpe":
                if r["best_safe_configuration_id"] == "xgboost_47":
                    note = "  <-- xgboost_47 dominates both axes"
            print(f"  [{FAMILY_LABELS[fam]:<8}]  "
                  f"Best SAFE == Best Perf? {flag:<3}  "
                  f"SAFE={r['best_safe_configuration_id']}  "
                  f"Perf={r['best_perf_configuration_id']}{note}")

    print(f"\n{'=' * 68}\n")

def main():
    print("=" * 68)
    print("9b  --  Compact SAFE-Performance Three-Panel Figures")
    print("=" * 68)

    df = load_results()

    scopes = ["all_models"] + FAMILY_ORDER
    total  = len(PERF_METRICS) * len(scopes)
    done   = 0

    for metric_col, direction in PERF_METRICS.items():
        metric_dir = os.path.join(FIG_BASE, metric_col)
        print(f"\n[Metric: {metric_col}]")

        for scope in scopes:
            done += 1
            out_path = os.path.join(metric_dir, f"{scope}.png")
            plot_three_panel_figure(df, metric_col, direction, scope, out_path)

    # Summary table
    os.makedirs(TABLE_DIR, exist_ok=True)
    summary = build_summary_table(df)
    table_path = os.path.join(TABLE_DIR, "safe_performance_summary.csv")
    summary.to_csv(table_path, index=False)
    print("\nSaved: data/results/step10h/safe_performance_summary.csv")
    print(f"  {len(summary)} rows  ({total} figures x {len(SCORE_COLS)} score types)")

    print_console_summary(summary)

    print(f"Figures  -> {os.path.relpath(FIG_BASE,  BASE_DIR)}/")
    print(f"Table    -> {os.path.relpath(table_path, BASE_DIR)}")

if __name__ == "__main__":
    main()
