"""
Step 8d - the fourth vector for the extended integrated Compliance Score:
RGF*, sector-level group parity.

Sector-level RGA gaps are lower-is-better, so to line this dimension up
with RGA/RGR/RGE* (all higher-is-better) it's flipped into
RGF* = 1 - gap. For each model/date, restrict to sectors with at least
MIN_GROUP_ASSETS names, compute sector RGA = core.rga(target_rank,
mu_model), sort sectors by RGA descending, then build the progressive
parity curve:

    RGF*_0 = 1 - (best_RGA - best_RGA) = 1.0
    RGF*_j = 1 - (best_RGA - sector_RGA_j),   j = 1, ..., G-1

interpolated to N_POINTS = 11 so it shares the same grid as the other
three Step 8 vectors. This does not replace 7e_rgf.py's scalar
sector RGA gap. Sectors here are market groups, not protected
demographic attributes, hence "group parity" rather than fairness in
the ethical/legal sense.

Reads data/results/step7/test_panel.csv, writes rgf_vector_* /
rgf_sector_rga_by_date.csv under data/results/step8/.

Reference: Giudici and Kolesnikov, "SAFE AI metrics: An integrated
approach"; Babaei and Giudici, "A statistical package for safe
artificial intelligence".
Author: Anila Vata
"""

import os
import warnings

import numpy as np
import pandas as pd
from safeaipackage import core as safe_core

warnings.filterwarnings("ignore")

OUT_DIR    = "data/results/step8"
PANEL_PATH = "data/results/step7/test_panel.csv"

MODELS             = ["ridge", "xgboost", "mlp"]
N_POINTS           = 11
MIN_ASSETS         = 10
MIN_GROUP_ASSETS   = 5
MIN_VALID_SECTORS  = 2

# Target x-grid shared by all dates after interpolation
X_TARGET = np.linspace(0.0, 1.0, N_POINTS)   # [0.0, 0.1, …, 1.0]

# Column names for wide pivot
_K_COL = {k: f"RGF_k{k:02d}" for k in range(N_POINTS)}

os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(PANEL_PATH):
    raise FileNotFoundError(f"Input panel not found: {PANEL_PATH}")

test = pd.read_csv(PANEL_PATH)
test["date"] = pd.to_datetime(test["date"])

required_cols = ["date", "ticker", "sector", "target_rank"] + [f"mu_{m}" for m in MODELS]
for col in required_cols:
    if col not in test.columns:
        raise ValueError(f"Required column missing from test_panel.csv: '{col}'")

test["target_rank"] = pd.to_numeric(test["target_rank"], errors="coerce")
for m in MODELS:
    test[f"mu_{m}"] = pd.to_numeric(test[f"mu_{m}"], errors="coerce")

rebal_dates = sorted(test["date"].dropna().unique())

def _find_raw_match(x_k, x_raw, tol=1e-9):
    """Return the index j in x_raw closest to x_k if within tolerance, else None."""
    for j, xr in enumerate(x_raw):
        if abs(x_k - xr) < tol:
            return j
    return None

def _build_rgf_curve(sorted_sectors, sector_rga, sector_assets):
    """
    Given sectors sorted by RGA descending, build the raw G-point RGF* curve.

    Returns:
      x_raw    : np.ndarray shape (G,), linspace(0, 1, G)
      rgf_raw  : np.ndarray shape (G,), RGF* values
      gaps_raw : np.ndarray shape (G,), corresponding sector_gap values
    """
    G        = len(sorted_sectors)
    best_rga = sector_rga[0]
    gaps_raw = np.array([best_rga - rga for rga in sector_rga])
    rgf_raw  = 1.0 - gaps_raw
    # Numerical clamp: values should be in [0,1] by construction but
    # floating-point subtraction may produce tiny violations.
    rgf_raw  = np.clip(rgf_raw, 0.0, 1.0)
    x_raw    = np.linspace(0.0, 1.0, G)
    return x_raw, rgf_raw, gaps_raw

detail_rows     = []
sector_rga_rows = []
skipped_rows    = []

print("\nRGF* Sector-Parity Vector  (progressive cross-sector RGA gap)")
print("-" * 64)

for model_name in MODELS:
    col_pred = f"mu_{model_name}"

    valid_dates   = 0
    skipped_dates = 0

    for d in rebal_dates:
        raw_date = pd.Timestamp(d).date().isoformat()

        sub = test.loc[test["date"] == d].copy()
        sub = sub.dropna(subset=["sector", "target_rank", col_pred])
        n_assets_total = len(sub)

        if n_assets_total < MIN_ASSETS:
            skipped_rows.append({
                "date": raw_date, "model": model_name,
                "reason": "too_few_assets", "n_assets": n_assets_total, "details": "",
            })
            skipped_dates += 1
            continue

        # ── Compute sector-level RGA ──────────────────────────────────────────
        sector_names    = []
        sector_rga_vals = []
        sector_asset_ct = []
        date_sector_rows = []
        rga_error_logged = False

        for sector in sub["sector"].unique():
            s_sub = sub.loc[sub["sector"] == sector]
            if len(s_sub) < MIN_GROUP_ASSETS:
                continue

            ytest = s_sub["target_rank"].astype(float).tolist()
            yhat  = s_sub[col_pred].astype(float).tolist()

            if np.nanstd(ytest) < 1e-12 or np.nanstd(yhat) < 1e-12:
                continue

            try:
                rga_val = float(safe_core.rga(ytest, yhat))
            except Exception as exc:
                skipped_rows.append({
                    "date": raw_date, "model": model_name,
                    "reason": "rga_error", "n_assets": n_assets_total,
                    "details": f"sector={sector}: {exc}",
                })
                rga_error_logged = True
                continue

            sector_names.append(sector)
            sector_rga_vals.append(rga_val)
            sector_asset_ct.append(len(s_sub))

            date_sector_rows.append({
                "date":             raw_date,
                "model":            model_name,
                "sector":           sector,
                "sector_RGA":       round(rga_val, 6),
                "sector_rank_by_RGA": None,    # filled after sorting
                "n_assets_sector":  len(s_sub),
            })

        n_valid = len(sector_names)
        if n_valid < MIN_VALID_SECTORS:
            if not rga_error_logged:
                skipped_rows.append({
                    "date": raw_date, "model": model_name,
                    "reason": "too_few_valid_sectors", "n_assets": n_assets_total,
                    "details": f"valid sectors = {n_valid}, need {MIN_VALID_SECTORS}",
                })
            skipped_dates += 1
            continue

        # ── Sort sectors by sector RGA descending ─────────────────────────────
        order      = np.argsort(sector_rga_vals)[::-1]
        s_names    = [sector_names[i]    for i in order]
        s_rga      = [sector_rga_vals[i] for i in order]
        s_assets   = [sector_asset_ct[i] for i in order]

        # Assign sector ranks and stage sector-RGA rows
        for rank_idx, orig_sector in enumerate(s_names):
            for row in date_sector_rows:
                if row["sector"] == orig_sector and row["model"] == model_name:
                    row["sector_rank_by_RGA"] = rank_idx + 1   # 1 = highest RGA

        # ── Build raw RGF* curve ──────────────────────────────────────────────
        x_raw, rgf_raw, gaps_raw = _build_rgf_curve(s_names, s_rga, s_assets)

        # ── Interpolate to N_POINTS ───────────────────────────────────────────
        rgf_interp = np.interp(X_TARGET, x_raw, rgf_raw)
        # Ensure boundary values are exact
        rgf_interp[0]  = 1.0
        rgf_interp[-1] = float(rgf_raw[-1])

        # ── Build per-k detail rows ───────────────────────────────────────────
        best_sector        = s_names[0]
        best_sector_rga    = s_rga[0]
        best_sector_assets = s_assets[0]

        date_detail_rows = []
        date_ok          = True

        for k_idx, (xk, rgf_k) in enumerate(zip(X_TARGET, rgf_interp)):
            gap_k = 1.0 - float(rgf_k)

            # Check whether this grid point coincides with a raw sector point
            raw_j = _find_raw_match(xk, x_raw)
            if raw_j is not None:
                comp_sector       = s_names[raw_j]
                comp_sector_rga   = round(s_rga[raw_j], 6)
                comp_assets       = s_assets[raw_j]
            else:
                comp_sector       = "interpolated"
                comp_sector_rga   = np.nan
                comp_assets       = np.nan

            date_detail_rows.append({
                "date":                       raw_date,
                "model":                      model_name,
                "k":                          k_idx,
                "fraction_sector_gap_path":   round(float(xk), 6),
                "RGF_star":                   round(float(rgf_k), 6),
                "sector_gap":                 round(gap_k, 6),
                "best_sector":                best_sector,
                "comparison_sector":          comp_sector,
                "best_sector_RGA":            round(best_sector_rga, 6),
                "comparison_sector_RGA":      comp_sector_rga,
                "n_valid_sectors":            n_valid,
                "n_assets":                   n_assets_total,
                "n_assets_best_sector":       best_sector_assets,
                "n_assets_comparison_sector": comp_assets,
            })

        if date_ok:
            detail_rows.extend(date_detail_rows)
            sector_rga_rows.extend(date_sector_rows)
            valid_dates += 1
        else:
            skipped_dates += 1

    print(f"  Model {model_name:<8}: valid dates = {valid_dates}, skipped = {skipped_dates}")

detail_df = pd.DataFrame(
    detail_rows,
    columns=["date", "model", "k", "fraction_sector_gap_path", "RGF_star",
             "sector_gap", "best_sector", "comparison_sector",
             "best_sector_RGA", "comparison_sector_RGA",
             "n_valid_sectors", "n_assets",
             "n_assets_best_sector", "n_assets_comparison_sector"],
)

sector_rga_df = pd.DataFrame(
    sector_rga_rows,
    columns=["date", "model", "sector", "sector_RGA",
             "sector_rank_by_RGA", "n_assets_sector"],
)

skipped_df = pd.DataFrame(
    skipped_rows,
    columns=["date", "model", "reason", "n_assets", "details"],
)

# ── Summary: aggregate by model × k ──────────────────────────────────────────
summary_rows = []
for (model_name, k), grp in detail_df.groupby(["model", "k"]):
    rgf_vals  = grp["RGF_star"].dropna().values
    gap_vals  = grp["sector_gap"].dropna().values
    n         = len(rgf_vals)
    frac      = grp["fraction_sector_gap_path"].iloc[0] if n > 0 else np.nan

    def _agg(vals):
        m = len(vals)
        return (
            round(float(np.mean(vals)),        6) if m > 0 else np.nan,
            round(float(np.std(vals, ddof=1)), 6) if m > 1 else np.nan,
            round(float(np.min(vals)),         6) if m > 0 else np.nan,
            round(float(np.max(vals)),         6) if m > 0 else np.nan,
        )

    mean_r, std_r, min_r, max_r = _agg(rgf_vals)
    mean_g, std_g, min_g, max_g = _agg(gap_vals)

    summary_rows.append({
        "model":                   model_name,
        "k":                       int(k),
        "fraction_sector_gap_path": round(float(frac), 6),
        "mean_RGF_star":            mean_r,
        "std_RGF_star":             std_r,
        "min_RGF_star":             min_r,
        "max_RGF_star":             max_r,
        "mean_sector_gap":          mean_g,
        "std_sector_gap":           std_g,
        "min_sector_gap":           min_g,
        "max_sector_gap":           max_g,
        "n_dates":                  n,
        "mean_n_valid_sectors":     round(float(grp["n_valid_sectors"].mean()), 2) if n > 0 else np.nan,
        "mean_n_assets":            round(float(grp["n_assets"].mean()),        2) if n > 0 else np.nan,
    })

summary_df = pd.DataFrame(
    summary_rows,
    columns=["model", "k", "fraction_sector_gap_path",
             "mean_RGF_star", "std_RGF_star", "min_RGF_star", "max_RGF_star",
             "mean_sector_gap", "std_sector_gap", "min_sector_gap", "max_sector_gap",
             "n_dates", "mean_n_valid_sectors", "mean_n_assets"],
)

# ── Wide pivot: RGF* vector ───────────────────────────────────────────────────
pivot = summary_df.pivot(index="model", columns="k", values="mean_RGF_star")
pivot.columns = [_K_COL[c] for c in pivot.columns]
pivot = pivot.reset_index().set_index("model").reindex(MODELS).reset_index()
for col in _K_COL.values():
    if col not in pivot.columns:
        pivot[col] = np.nan
pivot = pivot[["model"] + list(_K_COL.values())]

detail_path     = os.path.join(OUT_DIR, "rgf_vector_detail_by_date.csv")
sector_rga_path = os.path.join(OUT_DIR, "rgf_sector_rga_by_date.csv")
summary_path    = os.path.join(OUT_DIR, "rgf_vector_summary.csv")
vector_path     = os.path.join(OUT_DIR, "rgf_vector_by_model.csv")
skipped_path    = os.path.join(OUT_DIR, "rgf_vector_skipped_dates.csv")

detail_df.to_csv(detail_path,         index=False)
sector_rga_df.to_csv(sector_rga_path, index=False)
summary_df.to_csv(summary_path,       index=False)
pivot.to_csv(vector_path,             index=False)
skipped_df.to_csv(skipped_path,       index=False)

print()
print(f"Saved: {detail_path}")
print(f"Saved: {sector_rga_path}")
print(f"Saved: {summary_path}")
print(f"Saved: {vector_path}")
print(f"Saved: {skipped_path}")

print("\nRGF* compliance vector (mean_RGF_star per k):")
k_labels = list(_K_COL.values())
print(f"  {'model':<10}" + "".join(f"  {c}" for c in k_labels))
for _, row in pivot.iterrows():
    line = f"  {row['model']:<10}"
    for col in k_labels:
        val = row[col]
        line += f"  {val:.4f}" if pd.notna(val) else "      NaN"
    print(line)

print(f"\n  detail rows     : {len(detail_df)}")
print(f"  sector-RGA rows : {len(sector_rga_df)}")
print(f"  skipped rows    : {len(skipped_df)}")

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["font.size"]   = 11

FIG_DIR = "figures/step8/rgf"
os.makedirs(FIG_DIR, exist_ok=True)

MODEL_DISPLAY = {
    "ridge":   ("Ridge",   "#2166AC"),
    "xgboost": ("XGBoost", "#B2182B"),
    "mlp":     ("MLP",     "#4D9221"),
}

def _model_curve(model_name):
    """Return (x, y) for model_name from summary_df, sorted by fraction_sector_gap_path."""
    sub = (
        summary_df[summary_df["model"] == model_name]
        .sort_values("fraction_sector_gap_path")
        .dropna(subset=["mean_RGF_star"])
    )
    return sub["fraction_sector_gap_path"].tolist(), sub["mean_RGF_star"].tolist()

def _apply_style(ax, title):
    # Unlike RGA, RGR and RGE*, the 0.50 reference in RGF* is not a random baseline.
    # It is an intermediate parity threshold: RGF* = 0.50 corresponds to a sector
    # RGA gap of 0.50.
    ax.axhline(1.00, color="green", linestyle="--", linewidth=1.0,
               label="Perfect Sector Parity (RGF* = 1.00)", zorder=2)
    ax.axhline(0.50, color="gray", linestyle="--", linewidth=1.0,
               label="Intermediate Parity Threshold (RGF* = 0.50)", zorder=2)
    ax.set_xlim(0.00, 1.00)
    ax.set_ylim(0.00, 1.05)
    ax.set_xlabel("Fraction of Sector-Gap Path")
    ax.set_ylabel("RGF*")
    ax.grid(True, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def _save(fig, base_path):
    fig.tight_layout()
    fig.savefig(base_path + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {base_path}.png")

# ── Individual model figures ──────────────────────────────────────────────────
print()
for model_name, (display_name, color) in MODEL_DISPLAY.items():
    x_vals, y_vals = _model_curve(model_name)

    if len(x_vals) < 2:
        print(f"  Warning: {model_name} has fewer than 2 valid RGF* points; skipping figure.")
        continue

    fig, ax = plt.subplots(figsize=(6, 5), facecolor="white")
    ax.set_facecolor("white")

    ax.plot(x_vals, y_vals, color=color, marker="o", linewidth=1.8,
            markersize=5, label="RGF* Curve", zorder=3)

    _apply_style(ax, f"{display_name} RGF* Sector-Parity Curve")
    ax.legend(loc="lower left", frameon=True, framealpha=0.8)

    _save(fig, os.path.join(FIG_DIR, f"rgf_curve_{model_name}"))

# ── Combined figure: all models ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
ax.set_facecolor("white")

for model_name, (display_name, color) in MODEL_DISPLAY.items():
    x_vals, y_vals = _model_curve(model_name)
    if len(x_vals) < 2:
        continue
    ax.plot(x_vals, y_vals, color=color, marker="o", linewidth=1.8,
            markersize=5, label=display_name, zorder=3)

_apply_style(ax, "RGF* Sector-Parity Curve by Model")
ax.legend(loc="lower left", frameon=True, framealpha=0.8)
_save(fig, os.path.join(FIG_DIR, "rgf_curve_by_model"))

if not sector_rga_df.empty:
    print("\nBuilding sector RGA heatmap ...")

    heatmap_data = (
        sector_rga_df
        .groupby(["sector", "model"])["sector_RGA"]
        .mean()
        .unstack(level="model")
        .reindex(columns=MODELS)
    )

    sectors = heatmap_data.index.tolist()
    mat     = heatmap_data.values   # shape: (n_sectors, n_models)

    # Replace NaN with 0 for display (masked imshow would need extra setup)
    mat_display = np.where(np.isnan(mat), 0.0, mat)

    fig, ax = plt.subplots(figsize=(max(5, len(MODELS) * 1.8),
                                    max(4, len(sectors) * 0.5 + 1)),
                           facecolor="white")
    ax.set_facecolor("white")

    im = ax.imshow(mat_display, aspect="auto", cmap="RdYlGn",
                   vmin=0.0, vmax=1.0, origin="upper")

    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([MODEL_DISPLAY[m][0] for m in MODELS])
    ax.set_yticks(range(len(sectors)))
    ax.set_yticklabels(sectors, fontsize=9)

    # Annotate each cell with the value
    for i in range(len(sectors)):
        for j in range(len(MODELS)):
            val = mat[i, j]
            if not np.isnan(val):
                text_color = "white" if val < 0.35 or val > 0.85 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, color=text_color)

    plt.colorbar(im, ax=ax, label="Mean Sector RGA")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    heatmap_base = os.path.join(FIG_DIR, "sector_rga_heatmap_by_model")
    fig.tight_layout()
    fig.savefig(heatmap_base + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {heatmap_base}.png")
else:
    print("\nNo sector-RGA data available; skipping heatmap.")
