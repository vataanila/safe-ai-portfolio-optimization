"""
Step 8e pulls together the four Step 8 vectors (RGA, RGR, RGE*, RGF* --
all positively oriented: higher = more accurate/robust/explainable/fair)
into the Extended 4D SAFE Compliance Score. The original Giudici/
Kolesnikov paper integrates RGA+RGR+RGE only; RGF* is this project's
addition as a fourth dimension.

The score is the mean, over every (i,j,k,l) index combination across the
four 11-point vectors, of an aggregation function applied to
(A[i], B[j], C[k], D[l]) -- computed three ways: arithmetic mean,
geometric mean, and RMS. Nothing is recomputed here, this just reads the
four *_vector_by_model.csv files from Step 8 and integrates them.

Reference: Giudici and Kolesnikov, "SAFE AI metrics: An integrated
approach"; Babaei and Giudici, "A statistical package for safe
artificial intelligence".
Author: Anila Vata
"""

import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["font.size"]   = 11

OUT_DIR = "data/results/step8"
FIG_DIR = "figures/step8/compliance"

MODELS = ["ridge", "xgboost", "mlp"]

RGA_PATH = os.path.join(OUT_DIR, "rga_vector_by_model.csv")
RGR_PATH = os.path.join(OUT_DIR, "rgr_vector_by_model.csv")
RGE_PATH = os.path.join(OUT_DIR, "rge_vector_by_model.csv")
RGF_PATH = os.path.join(OUT_DIR, "rgf_vector_by_model.csv")

RGA_COLS = [f"RGA_k{k:02d}" for k in range(11)]
RGR_COLS = [f"RGR_p{p:02d}" for p in range(0, 55, 5)]   # p00,p05,...,p50
RGE_COLS = [f"RGE_k{k:02d}" for k in range(11)]
RGF_COLS = [f"RGF_k{k:02d}" for k in range(11)]

EXPECTED_N = 11
FP_TOL     = 1e-9

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

MODEL_DISPLAY = {
    "ridge":   ("Ridge",   "#2166AC"),
    "xgboost": ("XGBoost", "#B2182B"),
    "mlp":     ("MLP",     "#4D9221"),
}

for path in [RGA_PATH, RGR_PATH, RGE_PATH, RGF_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input vector file not found: {path}")

rga_df = pd.read_csv(RGA_PATH).set_index("model")
rgr_df = pd.read_csv(RGR_PATH).set_index("model")
rge_df = pd.read_csv(RGE_PATH).set_index("model")
rgf_df = pd.read_csv(RGF_PATH).set_index("model")

results_rows = []
skipped_rows = []

print("\nExtended 4D SAFE AI Compliance Score")
print("-" * 64)

for model_name in MODELS:

    # ── 1. Check model presence in all four vector files ──────────────────────
    missing_files = [
        name
        for name, df in [("RGA", rga_df), ("RGR", rgr_df),
                          ("RGE", rge_df), ("RGF", rgf_df)]
        if model_name not in df.index
    ]
    if missing_files:
        skipped_rows.append({
            "model":   model_name,
            "reason":  "missing_model_in_input",
            "details": f"Not found in: {', '.join(missing_files)}",
        })
        print(f"  Model {model_name:<8}: SKIPPED - missing_model_in_input "
              f"({', '.join(missing_files)})")
        continue

    # ── 2. Extract raw vector rows ────────────────────────────────────────────
    try:
        A_raw = rga_df.loc[model_name, RGA_COLS]
        B_raw = rgr_df.loc[model_name, RGR_COLS]
        C_raw = rge_df.loc[model_name, RGE_COLS]
        D_raw = rgf_df.loc[model_name, RGF_COLS]
    except KeyError as exc:
        skipped_rows.append({
            "model":   model_name,
            "reason":  "missing_vector_values",
            "details": str(exc),
        })
        print(f"  Model {model_name:<8}: SKIPPED -- missing_vector_values: {exc}")
        continue

    # ── 3. Check for NaN values ───────────────────────────────────────────────
    nan_counts = {
        "RGA": int(pd.Series(A_raw).isna().sum()),
        "RGR": int(pd.Series(B_raw).isna().sum()),
        "RGE": int(pd.Series(C_raw).isna().sum()),
        "RGF": int(pd.Series(D_raw).isna().sum()),
    }
    if any(v > 0 for v in nan_counts.values()):
        details = "; ".join(
            f"{k}={v} NaN" for k, v in nan_counts.items() if v > 0
        )
        skipped_rows.append({
            "model":   model_name,
            "reason":  "missing_vector_values",
            "details": details,
        })
        print(f"  Model {model_name:<8}: SKIPPED - missing_vector_values: {details}")
        continue

    A = A_raw.astype(float).values
    B = B_raw.astype(float).values
    C = C_raw.astype(float).values
    D = D_raw.astype(float).values

    # ── 4 & 5. Validate vector lengths ───────────────────────────────────────
    lengths = {"RGA": len(A), "RGR": len(B), "RGE": len(C), "RGF": len(D)}
    unique  = set(lengths.values())
    if len(unique) != 1 or next(iter(unique)) != EXPECTED_N:
        details = ", ".join(f"{k}={v}" for k, v in lengths.items())
        raise ValueError(
            f"Vector length error for model '{model_name}': {details}. "
            f"All four vectors must have equal length of {EXPECTED_N}."
        )

    n = len(A)

    # ── 6 & 7 & 8. Validate and clip value ranges ─────────────────────────────
    # Raises ValueError for substantial out-of-range values; clips only
    # tiny floating-point deviations within FP_TOL of [0, 1].
    for vec_name, vec in [("RGA", A), ("RGR", B), ("RGE", C), ("RGF", D)]:
        n_below = int(np.sum(vec < -FP_TOL))
        n_above = int(np.sum(vec > 1.0 + FP_TOL))
        if n_below > 0 or n_above > 0:
            raise ValueError(
                f"Substantial out-of-range values in {vec_name} for model "
                f"'{model_name}': {n_below} values below 0, "
                f"{n_above} values above 1. Check vector computation."
            )

    try:
        A = np.clip(A, 0.0, 1.0)
        B = np.clip(B, 0.0, 1.0)
        C = np.clip(C, 0.0, 1.0)
        D = np.clip(D, 0.0, 1.0)

        # ── 9. 4D compliance scores via broadcasting ──────────────────────────
        # Shapes: A4 (n,1,1,1), B4 (1,n,1,1), C4 (1,1,n,1), D4 (1,1,1,n)
        A4 = A[:, None, None, None]
        B4 = B[None, :, None, None]
        C4 = C[None, None, :, None]
        D4 = D[None, None, None, :]

        arith_tensor = (A4 + B4 + C4 + D4) / 4.0
        geo_tensor   = (A4 * B4 * C4 * D4) ** 0.25
        rms_tensor   = np.sqrt((A4**2 + B4**2 + C4**2 + D4**2) / 4.0)

        cs4_arith = float(np.mean(arith_tensor))
        cs4_geo   = float(np.mean(geo_tensor))
        cs4_rms   = float(np.mean(rms_tensor))

        # ── 10. 3D compliance scores (RGA, RGR, RGE* only) ───────────────────
        A3 = A[:, None, None]
        B3 = B[None, :, None]
        C3 = C[None, None, :]

        arith3 = (A3 + B3 + C3) / 3.0
        geo3   = (A3 * B3 * C3) ** (1.0 / 3.0)
        rms3   = np.sqrt((A3**2 + B3**2 + C3**2) / 3.0)

        cs3_arith = float(np.mean(arith3))
        cs3_geo   = float(np.mean(geo3))
        cs3_rms   = float(np.mean(rms3))

    except Exception as exc:
        skipped_rows.append({
            "model":   model_name,
            "reason":  "compliance_error",
            "details": str(exc),
        })
        print(f"  Model {model_name:<8}: SKIPPED, compliance_error: {exc}")
        continue

    delta_arith = cs4_arith - cs3_arith
    delta_geo   = cs4_geo   - cs3_geo
    delta_rms   = cs4_rms   - cs3_rms

    results_rows.append({
        "model":            model_name,
        "CS4_arithmetic":   round(cs4_arith,   6),
        "CS4_geometric":    round(cs4_geo,     6),
        "CS4_rms":          round(cs4_rms,     6),
        "CS3_arithmetic":   round(cs3_arith,   6),
        "CS3_geometric":    round(cs3_geo,     6),
        "CS3_rms":          round(cs3_rms,     6),
        "delta_arithmetic": round(delta_arith, 6),
        "delta_geometric":  round(delta_geo,   6),
        "delta_rms":        round(delta_rms,   6),
        "n_vector_points":  n,
        "mean_RGA":         round(float(np.mean(A)), 6),
        "mean_RGR":         round(float(np.mean(B)), 6),
        "mean_RGE_star":    round(float(np.mean(C)), 6),
        "mean_RGF_star":    round(float(np.mean(D)), 6),
        "min_RGA":          round(float(np.min(A)),  6),
        "min_RGR":          round(float(np.min(B)),  6),
        "min_RGE_star":     round(float(np.min(C)),  6),
        "min_RGF_star":     round(float(np.min(D)),  6),
        "max_RGA":          round(float(np.max(A)),  6),
        "max_RGR":          round(float(np.max(B)),  6),
        "max_RGE_star":     round(float(np.max(C)),  6),
        "max_RGF_star":     round(float(np.max(D)),  6),
    })

    print(
        f"  Model {model_name:<8}: "
        f"CS4_arith={cs4_arith:.4f}  "
        f"CS4_geo={cs4_geo:.4f}  "
        f"CS4_rms={cs4_rms:.4f}"
    )

RESULT_COLS = [
    "model",
    "CS4_arithmetic", "CS4_geometric", "CS4_rms",
    "CS3_arithmetic", "CS3_geometric", "CS3_rms",
    "delta_arithmetic", "delta_geometric", "delta_rms",
    "n_vector_points",
    "mean_RGA",     "mean_RGR",     "mean_RGE_star",  "mean_RGF_star",
    "min_RGA",      "min_RGR",      "min_RGE_star",   "min_RGF_star",
    "max_RGA",      "max_RGR",      "max_RGE_star",   "max_RGF_star",
]

results_df = pd.DataFrame(results_rows, columns=RESULT_COLS)

skipped_df = pd.DataFrame(
    skipped_rows,
    columns=["model", "reason", "details"],
)

# ── Long format ───────────────────────────────────────────────────────────────
SCORE_TYPES = [
    "CS4_arithmetic", "CS4_geometric", "CS4_rms",
    "CS3_arithmetic", "CS3_geometric", "CS3_rms",
    "delta_arithmetic", "delta_geometric", "delta_rms",
]
long_rows = [
    {"model": row["model"], "score_type": st, "value": row[st]}
    for _, row in results_df.iterrows()
    for st in SCORE_TYPES
]
long_df = pd.DataFrame(long_rows, columns=["model", "score_type", "value"])

# ── Dimension summary ─────────────────────────────────────────────────────────
DIM_COLS = [
    "model",
    "mean_RGA",    "mean_RGR",    "mean_RGE_star",  "mean_RGF_star",
    "min_RGA",     "min_RGR",     "min_RGE_star",   "min_RGF_star",
    "max_RGA",     "max_RGR",     "max_RGE_star",   "max_RGF_star",
]
dim_df = (
    results_df[DIM_COLS].copy()
    if not results_df.empty
    else pd.DataFrame(columns=DIM_COLS)
)

cs4_path     = os.path.join(OUT_DIR, "compliance_score_4d.csv")
long_path    = os.path.join(OUT_DIR, "compliance_score_4d_long.csv")
skipped_path = os.path.join(OUT_DIR, "compliance_score_4d_skipped_models.csv")
dim_path     = os.path.join(OUT_DIR, "compliance_dimension_means.csv")

results_df.to_csv(cs4_path,     index=False)
long_df.to_csv(long_path,       index=False)
skipped_df.to_csv(skipped_path, index=False)
dim_df.to_csv(dim_path,         index=False)

print("\nExtended 4D SAFE AI Compliance Score -- final table:")
print("-" * 96)
if not results_df.empty:
    header = (
        f"  {'model':<10}"
        f"  {'CS4_arith':>10}"
        f"  {'CS4_geo':>10}"
        f"  {'CS4_rms':>10}"
        f"  {'CS3_arith':>10}"
        f"  {'CS3_geo':>10}"
        f"  {'CS3_rms':>10}"
        f"  {'d_arith':>10}"
        f"  {'d_geo':>10}"
        f"  {'d_rms':>10}"
    )
    print(header)
    for _, r in results_df.iterrows():
        print(
            f"  {r['model']:<10}"
            f"  {r['CS4_arithmetic']:>10.6f}"
            f"  {r['CS4_geometric']:>10.6f}"
            f"  {r['CS4_rms']:>10.6f}"
            f"  {r['CS3_arithmetic']:>10.6f}"
            f"  {r['CS3_geometric']:>10.6f}"
            f"  {r['CS3_rms']:>10.6f}"
            f"  {r['delta_arithmetic']:>10.6f}"
            f"  {r['delta_geometric']:>10.6f}"
            f"  {r['delta_rms']:>10.6f}"
        )
else:
    print("  No valid models.")

if not skipped_df.empty:
    print("\nSkipped models:")
    for _, r in skipped_df.iterrows():
        print(f"  {r['model']}: {r['reason']} - {r['details']}")

print()
print(f"Saved: {cs4_path}")
print(f"Saved: {long_path}")
print(f"Saved: {skipped_path}")
print(f"Saved: {dim_path}")

if results_df.empty:
    print("\nNo valid models, skipping all figures.")
    raise SystemExit(0)

def _save(fig, base_path):
    fig.tight_layout()
    fig.savefig(base_path + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {base_path}.png")

models_valid = results_df["model"].tolist()
x            = np.arange(len(models_valid))
x_labels     = [MODEL_DISPLAY[m][0] for m in models_valid]

print()

# ── Figure 1: Extended 4D Compliance Score by Model ──────────────────────────
fig1_base  = os.path.join(FIG_DIR, "compliance_score_4d_by_model")
bar_cols1  = ["CS4_arithmetic", "CS4_geometric", "CS4_rms"]
bar_clrs1  = ["#2166AC",        "#B2182B",        "#4D9221"]
bar_w1     = 0.25

fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
ax.set_facecolor("white")

for i, (col, color) in enumerate(zip(bar_cols1, bar_clrs1)):
    vals  = results_df[col].tolist()
    rects = ax.bar(
        x + (i - 1) * bar_w1, vals, bar_w1,
        label=col, color=color, alpha=0.85, edgecolor="white", linewidth=0.6,
    )
    for rect, v in zip(rects, vals):
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            rect.get_height() + 0.008,
            f"{v:.3f}",
            ha="center", va="bottom", fontsize=8,
        )

ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.set_xlabel("Model")
ax.set_ylabel("Compliance Score")
ax.set_ylim(0, 1.05)
ax.legend(loc="upper right", frameon=True, framealpha=0.8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.4)

_save(fig, fig1_base)

# ── Figure 2: 3D vs 4D Compliance Score (arithmetic) ─────────────────────────
fig2_base  = os.path.join(FIG_DIR, "compliance_score_3d_vs_4d")
bar_cols2  = ["CS3_arithmetic", "CS4_arithmetic"]
bar_clrs2  = ["#92C5DE",        "#2166AC"]
bar_w2     = 0.35

fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
ax.set_facecolor("white")

for i, (col, color) in enumerate(zip(bar_cols2, bar_clrs2)):
    vals  = results_df[col].tolist()
    rects = ax.bar(
        x + (i - 0.5) * bar_w2, vals, bar_w2,
        label=col, color=color, alpha=0.85, edgecolor="white", linewidth=0.6,
    )
    for rect, v in zip(rects, vals):
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            rect.get_height() + 0.008,
            f"{v:.3f}",
            ha="center", va="bottom", fontsize=8,
        )

ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.set_xlabel("Model")
ax.set_ylabel("Compliance Score")
ax.set_ylim(0, 1.05)
ax.legend(loc="upper right", frameon=True, framealpha=0.8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.4)

_save(fig, fig2_base)

# ── Figure 3: SAFE Vector Mean Dimensions by Model ───────────────────────────
fig3_base   = os.path.join(FIG_DIR, "compliance_dimension_means_by_model")
dim_cols    = ["mean_RGA", "mean_RGR", "mean_RGE_star", "mean_RGF_star"]
dim_display = ["mean RGA", "mean RGR", "mean RGE*",     "mean RGF*"]
dim_clrs    = ["#2166AC",  "#B2182B",  "#4D9221",        "#D6604D"]
bar_w3      = 0.18

fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
ax.set_facecolor("white")

for i, (col, label, color) in enumerate(zip(dim_cols, dim_display, dim_clrs)):
    vals   = results_df[col].tolist()
    offset = (i - 1.5) * bar_w3
    rects  = ax.bar(
        x + offset, vals, bar_w3,
        label=label, color=color, alpha=0.85, edgecolor="white", linewidth=0.6,
    )
    for rect, v in zip(rects, vals):
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            rect.get_height() + 0.008,
            f"{v:.3f}",
            ha="center", va="bottom", fontsize=7.5,
        )

ax.axhline(
    0.5, color="gray", linestyle="--", linewidth=1.2, alpha=0.7,
    label="Intermediate Reference (0.50)", zorder=2,
)

ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.set_xlabel("Model")
ax.set_ylabel("Mean Vector Value")
ax.set_ylim(0, 1.05)
ax.legend(loc="upper right", frameon=True, framealpha=0.8, fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.4)

_save(fig, fig3_base)
