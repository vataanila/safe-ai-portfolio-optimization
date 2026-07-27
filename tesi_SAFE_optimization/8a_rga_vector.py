"""
Step 8a builds the RGA compliance vector for the integrated SAFE AI
Compliance Score, following the data-removal logic in Giudici and
Kolesnikov ("SAFE AI metrics: An integrated approach").

For each model and rebalancing date, the cross-section is sorted by
mu_model descending and split into N_SEGMENTS equal segments. RGA is
recomputed on progressively reduced cross-sections as segments are
removed one by one from the top (target_rank is always ytest; mu_model
is both the removal-order driver and yhat) -- this is the paper's
"segments of the ranked predictions" idea. A reproducible random-
prediction baseline is computed alongside each model curve so the
degradation relative to chance can be assessed.

Note this does NOT replace 7b_rga.py, which computes scalar RGA per
rebalancing date -- this script builds the 11-point removal curve used
in the compliance score integration.

Reads data/results/step7/test_panel.csv (and, only for the combined
diagnostic figure, the RGR/RGE vector summaries from 8b/8c if
they already exist). Writes the rga_vector_* files under
data/results/step8/.

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

MODELS     = ["ridge", "xgboost", "mlp"]
N_SEGMENTS = 10
MIN_ASSETS = 10
MIN_REMAINING_ASSETS = 10
RANDOM_SEED = 42

# Column names for wide-pivot outputs
_K_COL     = {k: f"RGA_k{k:02d}"        for k in range(N_SEGMENTS + 1)}
_K_RND_COL = {k: f"Random_RGA_k{k:02d}" for k in range(N_SEGMENTS + 1)}

os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(PANEL_PATH):
    raise FileNotFoundError(f"Input panel not found: {PANEL_PATH}")

test = pd.read_csv(PANEL_PATH)
test["date"] = pd.to_datetime(test["date"])

required_cols = ["date", "target_rank"] + [f"mu_{m}" for m in MODELS]
for col in required_cols:
    if col not in test.columns:
        raise ValueError(f"Required column missing from test_panel.csv: '{col}'")

test["target_rank"] = pd.to_numeric(test["target_rank"], errors="coerce")
for m in MODELS:
    test[f"mu_{m}"] = pd.to_numeric(test[f"mu_{m}"], errors="coerce")

rebal_dates = sorted(test["date"].dropna().unique())

detail_rows  = []
skipped_rows = []

print("\nRGA Compliance Vector  (data-removal along ranked predictions)")
print("-" * 64)

for model_name in MODELS:
    col_pred = f"mu_{model_name}"
    model_idx = MODELS.index(model_name)   # used for per-model random seed

    valid_dates   = 0
    skipped_dates = 0

    for d in rebal_dates:
        raw_date = pd.Timestamp(d).date().isoformat()

        sub = test.loc[test["date"] == d].copy()
        sub = sub.dropna(subset=["target_rank", col_pred])
        n_assets_full = len(sub)

        if n_assets_full < MIN_ASSETS:
            skipped_rows.append({
                "date": raw_date, "model": model_name,
                "reason": "too_few_assets", "n_assets": n_assets_full, "details": "",
            })
            skipped_dates += 1
            continue

        # ── Initial constant checks (k = 0) ──────────────────────────────────
        if np.nanstd(sub["target_rank"].values) < 1e-12:
            skipped_rows.append({
                "date": raw_date, "model": model_name,
                "reason": "constant_target_rank", "n_assets": n_assets_full, "details": "",
            })
            skipped_dates += 1
            continue

        if np.nanstd(sub[col_pred].values) < 1e-12:
            skipped_rows.append({
                "date": raw_date, "model": model_name,
                "reason": "constant_yhat", "n_assets": n_assets_full, "details": "",
            })
            skipped_dates += 1
            continue

        # ── Sort by mu_model descending: highest predicted rank removed first ─
        sub_sorted = sub.sort_values(col_pred, ascending=False).reset_index(drop=True)

        # ── Reproducible random baseline: one per model/date ─────────────────
        # Seed is deterministic: combines global seed, model index, and a stable
        # hash of the date string so that each model/date pair is independently
        # reproducible regardless of loop order.
        date_hash = int(abs(hash(raw_date)) % (2 ** 31))
        rng = np.random.default_rng([RANDOM_SEED, model_idx, date_hash])
        random_yhat_full = rng.normal(size=n_assets_full)

        # ── Split sorted indices into N_SEGMENTS equal segments ───────────────
        sorted_indices = np.arange(n_assets_full)
        segments = np.array_split(sorted_indices, N_SEGMENTS)

        # ── Progressive removal loop ──────────────────────────────────────────
        # Rows are staged here; committed to detail_rows only on full success.
        date_detail_rows = []
        date_ok          = True

        for k in range(N_SEGMENTS + 1):   # k = 0 … 10
            fraction_removed = k / N_SEGMENTS

            if k == N_SEGMENTS:
                # k = 10: all data removed, RGA = 0.0 by definition
                date_detail_rows.append({
                    "date":                  raw_date,
                    "model":                 model_name,
                    "k_removed":             k,
                    "n_removed_segments":    k,
                    "fraction_data_removed": fraction_removed,
                    "RGA":                   0.0,
                    "random_RGA":            0.0,
                    "n_assets_full":         n_assets_full,
                    "n_assets_remaining":    0,
                })
                continue

            # Build remaining-observation mask
            if k == 0:
                remaining_idx = sorted_indices.copy()
            else:
                removed_idx   = np.concatenate(segments[:k])
                removed_set   = set(removed_idx.tolist())
                remaining_idx = np.array([i for i in sorted_indices if i not in removed_set])

            n_remaining = len(remaining_idx)

            if n_remaining < MIN_REMAINING_ASSETS:
                skipped_rows.append({
                    "date": raw_date, "model": model_name,
                    "reason": "too_few_remaining_assets", "n_assets": n_assets_full,
                    "details": f"k={k}, n_remaining={n_remaining}",
                })
                date_ok = False
                break

            target_rem = sub_sorted["target_rank"].iloc[remaining_idx].tolist()
            yhat_rem   = sub_sorted[col_pred].iloc[remaining_idx].tolist()
            rand_rem   = random_yhat_full[remaining_idx].tolist()

            if np.nanstd(target_rem) < 1e-12:
                skipped_rows.append({
                    "date": raw_date, "model": model_name,
                    "reason": "constant_target_rank", "n_assets": n_assets_full,
                    "details": f"k={k}",
                })
                date_ok = False
                break

            if np.nanstd(yhat_rem) < 1e-12:
                skipped_rows.append({
                    "date": raw_date, "model": model_name,
                    "reason": "constant_yhat", "n_assets": n_assets_full,
                    "details": f"k={k}",
                })
                date_ok = False
                break

            try:
                rga_val      = round(float(safe_core.rga(target_rem, yhat_rem)),   6)
                random_rga_val = round(float(safe_core.rga(target_rem, rand_rem)), 6)
            except Exception as exc:
                skipped_rows.append({
                    "date": raw_date, "model": model_name,
                    "reason": "rga_error", "n_assets": n_assets_full,
                    "details": f"k={k}: {exc}",
                })
                date_ok = False
                break

            date_detail_rows.append({
                "date":                  raw_date,
                "model":                 model_name,
                "k_removed":             k,
                "n_removed_segments":    k,
                "fraction_data_removed": fraction_removed,
                "RGA":                   rga_val,
                "random_RGA":            random_rga_val,
                "n_assets_full":         n_assets_full,
                "n_assets_remaining":    n_remaining,
            })

        if date_ok:
            detail_rows.extend(date_detail_rows)
            valid_dates += 1
        else:
            skipped_dates += 1

    print(f"  Model {model_name:<8}: valid dates = {valid_dates}, skipped = {skipped_dates}")

detail_df = pd.DataFrame(
    detail_rows,
    columns=["date", "model", "k_removed", "n_removed_segments",
             "fraction_data_removed", "RGA", "random_RGA",
             "n_assets_full", "n_assets_remaining"],
)

skipped_df = pd.DataFrame(
    skipped_rows,
    columns=["date", "model", "reason", "n_assets", "details"],
)

# ── Summary: aggregate by model × k_removed ──────────────────────────────────
summary_rows = []
for (model_name, k), grp in detail_df.groupby(["model", "k_removed"]):
    rga_vals  = grp["RGA"].dropna().values
    rand_vals = grp["random_RGA"].dropna().values
    n         = len(rga_vals)
    frac      = grp["fraction_data_removed"].iloc[0] if n > 0 else np.nan

    def _agg(vals):
        m = len(vals)
        return (
            round(float(np.mean(vals)),        6) if m > 0 else np.nan,
            round(float(np.std(vals, ddof=1)), 6) if m > 1 else np.nan,
            round(float(np.min(vals)),         6) if m > 0 else np.nan,
            round(float(np.max(vals)),         6) if m > 0 else np.nan,
        )

    mean_r, std_r, min_r, max_r       = _agg(rga_vals)
    mean_rnd, std_rnd, min_rnd, max_rnd = _agg(rand_vals)

    summary_rows.append({
        "model":                   model_name,
        "k_removed":               int(k),
        "fraction_data_removed":   round(float(frac), 6),
        "mean_RGA":                mean_r,
        "std_RGA":                 std_r,
        "min_RGA":                 min_r,
        "max_RGA":                 max_r,
        "mean_random_RGA":         mean_rnd,
        "std_random_RGA":          std_rnd,
        "min_random_RGA":          min_rnd,
        "max_random_RGA":          max_rnd,
        "n_dates":                 n,
        "mean_n_assets_full":      round(float(grp["n_assets_full"].mean()),      2) if n > 0 else np.nan,
        "mean_n_assets_remaining": round(float(grp["n_assets_remaining"].mean()), 2) if n > 0 else np.nan,
    })

summary_df = pd.DataFrame(
    summary_rows,
    columns=["model", "k_removed", "fraction_data_removed",
             "mean_RGA", "std_RGA", "min_RGA", "max_RGA",
             "mean_random_RGA", "std_random_RGA", "min_random_RGA", "max_random_RGA",
             "n_dates", "mean_n_assets_full", "mean_n_assets_remaining"],
)

# ── Wide pivot: model RGA vector ──────────────────────────────────────────────
pivot_rga = summary_df.pivot(index="model", columns="k_removed", values="mean_RGA")
pivot_rga.columns = [_K_COL[c] for c in pivot_rga.columns]
pivot_rga = pivot_rga.reset_index().set_index("model").reindex(MODELS).reset_index()
for col in _K_COL.values():
    if col not in pivot_rga.columns:
        pivot_rga[col] = np.nan
pivot_rga = pivot_rga[["model"] + list(_K_COL.values())]

# ── Wide pivot: random baseline vector ───────────────────────────────────────
pivot_rnd = summary_df.pivot(index="model", columns="k_removed", values="mean_random_RGA")
pivot_rnd.columns = [_K_RND_COL[c] for c in pivot_rnd.columns]
pivot_rnd = pivot_rnd.reset_index().set_index("model").reindex(MODELS).reset_index()
for col in _K_RND_COL.values():
    if col not in pivot_rnd.columns:
        pivot_rnd[col] = np.nan
pivot_rnd = pivot_rnd[["model"] + list(_K_RND_COL.values())]

detail_path  = os.path.join(OUT_DIR, "rga_vector_detail_by_date.csv")
summary_path = os.path.join(OUT_DIR, "rga_vector_summary.csv")
vector_path  = os.path.join(OUT_DIR, "rga_vector_by_model.csv")
random_path  = os.path.join(OUT_DIR, "rga_random_baseline_by_model.csv")
skipped_path = os.path.join(OUT_DIR, "rga_vector_skipped_dates.csv")

detail_df.to_csv(detail_path,  index=False)
summary_df.to_csv(summary_path, index=False)
pivot_rga.to_csv(vector_path,   index=False)
pivot_rnd.to_csv(random_path,   index=False)
skipped_df.to_csv(skipped_path, index=False)

print()
print(f"Saved: {detail_path}")
print(f"Saved: {summary_path}")
print(f"Saved: {vector_path}")
print(f"Saved: {random_path}")
print(f"Saved: {skipped_path}")

print("\nRGA compliance vector (mean_RGA per k):")
k_labels = list(_K_COL.values())
print(f"  {'model':<10}" + "".join(f"  {c}" for c in k_labels))
for _, row in pivot_rga.iterrows():
    line = f"  {row['model']:<10}"
    for col in k_labels:
        val = row[col]
        line += f"  {val:.4f}" if pd.notna(val) else "      NaN"
    print(line)

print(f"\n  detail rows : {len(detail_df)}")
print(f"  skipped rows: {len(skipped_df)}")

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["font.size"]   = 11

FIG_DIR = "figures/step8/rga"
os.makedirs(FIG_DIR, exist_ok=True)

MODEL_DISPLAY = {
    "ridge":   ("Ridge",   "#2166AC"),
    "xgboost": ("XGBoost", "#B2182B"),
    "mlp":     ("MLP",     "#4D9221"),
}

def _model_curves(model_name):
    """Return (x, y_model, y_random) sorted by fraction_data_removed."""
    sub = (
        summary_df[summary_df["model"] == model_name]
        .sort_values("fraction_data_removed")
        .dropna(subset=["mean_RGA", "mean_random_RGA"])
    )
    x      = sub["fraction_data_removed"].tolist()
    y_rga  = sub["mean_RGA"].tolist()
    y_rand = sub["mean_random_RGA"].tolist()
    return x, y_rga, y_rand

def _aurga(x_vals, y_vals):
    """Trapezoidal AURGA from the plotted mean curve."""
    if len(x_vals) < 2:
        return np.nan
    return round(float(np.trapz(y_vals, x_vals)), 4)

def _apply_style(ax, title):
    ax.set_xlim(0.00, 1.00)
    ax.set_ylim(0.00, 1.05)
    ax.set_xlabel("Fraction of Data Removed")
    ax.set_ylabel("RGA")
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
    x_vals, y_rga, y_rand = _model_curves(model_name)

    if len(x_vals) < 2:
        print(f"  Warning: {model_name} has fewer than 2 valid RGA points; skipping figure.")
        continue

    aurga_model = _aurga(x_vals, y_rga)
    aurga_rand  = _aurga(x_vals, y_rand)

    rga_label  = (f"RGA Curve (AURGA = {aurga_model:.4f})"
                  if not np.isnan(aurga_model) else "RGA Curve")
    rand_label = (f"Random Baseline (AURGA = {aurga_rand:.4f})"
                  if not np.isnan(aurga_rand) else "Random Baseline")

    fig, ax = plt.subplots(figsize=(6, 5), facecolor="white")
    ax.set_facecolor("white")

    ax.plot(x_vals, y_rga,  color=color,    marker="o", linewidth=1.8,
            markersize=5, label=rga_label,  zorder=3)
    ax.plot(x_vals, y_rand, color="orange", marker="o", linewidth=1.4,
            markersize=4, linestyle="--", label=rand_label, zorder=2)

    _apply_style(ax, f"{display_name} RGA Curve")
    ax.legend(loc="upper right", frameon=True, framealpha=0.8)

    _save(fig, os.path.join(FIG_DIR, f"rga_curve_{model_name}"))

# ── Combined figure: all models + averaged random baseline ────────────────────
fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
ax.set_facecolor("white")

rand_curves = []

for model_name, (display_name, color) in MODEL_DISPLAY.items():
    x_vals, y_rga, y_rand = _model_curves(model_name)
    if len(x_vals) < 2:
        continue
    ax.plot(x_vals, y_rga, color=color, marker="o", linewidth=1.8,
            markersize=5, label=display_name, zorder=3)
    rand_curves.append((x_vals, y_rand))

# Average random baseline across models (curves share the same x grid)
if rand_curves:
    x_ref = rand_curves[0][0]
    y_avg = np.mean([np.interp(x_ref, xc, yc) for xc, yc in rand_curves], axis=0).tolist()
    ax.plot(x_ref, y_avg, color="orange", linestyle="--", linewidth=1.4,
            marker="o", markersize=4, label="Random Baseline", zorder=2)

_apply_style(ax, "RGA Curve by Model")
ax.legend(loc="upper right", frameon=True, framealpha=0.8)
_save(fig, os.path.join(FIG_DIR, "rga_curve_by_model"))

RGR_SUMMARY_PATH = os.path.join(OUT_DIR, "rgr_vector_summary.csv")
RGE_SUMMARY_PATH = os.path.join(OUT_DIR, "rge_vector_summary.csv")

if os.path.exists(RGR_SUMMARY_PATH) and os.path.exists(RGE_SUMMARY_PATH):
    print("\nBuilding combined SAFE curves diagnostic figure ...")

    rgr_sum = pd.read_csv(RGR_SUMMARY_PATH)
    rge_sum = pd.read_csv(RGE_SUMMARY_PATH)

    # Normalise RGR x-axis: perturbation_percentage / max value
    max_pert = rgr_sum["perturbation_percentage"].max()
    if max_pert > 0:
        rgr_sum["x_norm"] = rgr_sum["perturbation_percentage"] / max_pert
    else:
        rgr_sum["x_norm"] = rgr_sum["perturbation_percentage"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="white")
    fig.suptitle("SAFE AI Compliance Curves by Model", fontsize=13)

    CURVE_COLORS = {"RGA": "#2166AC", "RGR": "#B2182B", "RGE*": "#4D9221"}

    for ax, model_name in zip(axes, MODELS):
        display_name = MODEL_DISPLAY[model_name][0]

        # RGA curve
        rga_sub = (
            summary_df[summary_df["model"] == model_name]
            .sort_values("fraction_data_removed")
            .dropna(subset=["mean_RGA"])
        )
        if not rga_sub.empty:
            ax.plot(rga_sub["fraction_data_removed"], rga_sub["mean_RGA"],
                    color=CURVE_COLORS["RGA"], marker="o", markersize=3,
                    linewidth=1.6, label=f"RGA Curve {display_name}", zorder=3)

        # RGR curve
        rgr_sub = (
            rgr_sum[rgr_sum["model"] == model_name]
            .sort_values("x_norm")
            .dropna(subset=["mean_RGR"])
        )
        if not rgr_sub.empty:
            ax.plot(rgr_sub["x_norm"], rgr_sub["mean_RGR"],
                    color=CURVE_COLORS["RGR"], marker="s", markersize=3,
                    linewidth=1.6, label=f"RGR Curve {display_name}", zorder=3)

        # RGE* curve
        rge_sub = (
            rge_sum[rge_sum["model"] == model_name]
            .sort_values("fraction_features_removed")
            .dropna(subset=["mean_RGE_star"])
        )
        if not rge_sub.empty:
            ax.plot(rge_sub["fraction_features_removed"], rge_sub["mean_RGE_star"],
                    color=CURVE_COLORS["RGE*"], marker="^", markersize=3,
                    linewidth=1.6, label=f"RGE* Curve {display_name}", zorder=3)

        ax.set_xlim(0.00, 1.00)
        ax.set_ylim(0.00, 1.05)
        ax.set_xlabel("Steps")
        ax.set_ylabel("Values")
        ax.grid(True, alpha=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.8)

    diag_base = os.path.join(FIG_DIR, "safe_curves_by_model")
    fig.tight_layout()
    fig.savefig(diag_base + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {diag_base}.png")

else:
    missing = [p for p in [RGR_SUMMARY_PATH, RGE_SUMMARY_PATH] if not os.path.exists(p)]
    print(f"\nSkipping combined SAFE diagnostic figure: missing file(s): {missing}")
    print("  Run 8b_rgr_vector.py and 8c_rge_vector.py first.")
