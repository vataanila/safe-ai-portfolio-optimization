"""
step3_2_visualize_frontier.py
====================
PURPOSE : Efficient frontier visualisation for the thesis chapter on
          portfolio construction methodology.

INPUTS  :
  data/clean/returns.csv          -- daily log-return matrix (full 2010-2025)
  data/results/baseline_summary.csv -- Markowitz MIQP baseline performance stats

OUTPUT  :
  data/figures/efficient_frontier.png  (300 dpi, 10Ã—7 in)

METHODOLOGY:
  All statistics are estimated on the MODEL WINDOW only: 2016-01-01 to
  2022-12-31 (in-sample period, pre-test).

  mu    : sample mean Ã— 252 (annualised)
  Sigma : Ledoit-Wolf shrinkage covariance Ã— 252 (annualised)

  Random portfolios : 500 sparse Dirichlet samples (K~U[5,30] tickers each)
  MVP  : minimise w'Î£w  s.t. sum(w)=1, wâ‰¥0  (SLSQP)
  MSP  : maximise Sharpe = (w'Î¼) / sqrt(w'Î£w)  s.t. same  (SLSQP)
  EW   : 1/N weights

Author  : Anila Vata 
"""

# =============================================================================
# 0. IMPORTS
# =============================================================================
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize

try:
    from adjustText import adjust_text
    HAS_ADJUSTTEXT = True
except ImportError:
    HAS_ADJUSTTEXT = False

warnings.filterwarnings("ignore")

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DIR   = os.path.join(BASE_DIR, "data", "clean")
RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")
FIGURES_DIR = os.path.join(BASE_DIR, "data", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

MODEL_START  = "2016-01-01"
MODEL_END    = "2022-12-31"
TRADING_DAYS = 252
N_RANDOM     = 500
RNG_SEED     = 42

print("=" * 70)
print("  EFFICIENT FRONTIER VISUALISATION")
print("=" * 70)

# =============================================================================
# 1. LOAD AND SLICE TO MODEL WINDOW
# =============================================================================
print("\n[1] Loading returns â€¦")

ret_path = os.path.join(CLEAN_DIR, "returns.csv")
if not os.path.exists(ret_path):
    raise FileNotFoundError(f"Not found: {ret_path}\nRun step2_preprocess.py first.")

returns_full = pd.read_csv(ret_path, index_col=0, parse_dates=True).sort_index()
returns_df   = returns_full.loc[MODEL_START:MODEL_END].dropna(axis=1, how="any")

N       = returns_df.shape[1]
tickers = returns_df.columns.tolist()

print(f"  Full series  : {returns_full.shape[0]} days Ã— {returns_full.shape[1]} stocks")
print(f"  Model window : {MODEL_START} to {MODEL_END}")
print(f"  After dropna : {returns_df.shape[0]} days Ã— {N} stocks")

# =============================================================================
# 2. ESTIMATE mu AND Sigma
# =============================================================================
print("\n[2] Estimating mu and Sigma (Ledoit-Wolf) â€¦")

R     = returns_df.values          # (T, N)
mu    = R.mean(axis=0) * TRADING_DAYS
lw    = LedoitWolf(assume_centered=False).fit(R)
Sigma = lw.covariance_ * TRADING_DAYS

print(f"  mu    : mean={mu.mean():.4f}  std={mu.std():.4f}  "
      f"range=[{mu.min():.4f}, {mu.max():.4f}]")
print(f"  Sigma : LW shrinkage={lw.shrinkage_:.4f}  "
      f"min_eig={np.linalg.eigvalsh(Sigma).min():.6f}")

# =============================================================================
# 3. INDIVIDUAL ASSET STATS
# =============================================================================
print("\n[3] Individual asset statistics â€¦")

asset_vol    = np.sqrt(np.diag(Sigma))
asset_ret    = mu
asset_sharpe = np.where(asset_vol > 0, asset_ret / asset_vol, np.nan)

print(f"  Asset vol    : mean={asset_vol.mean():.4f}  "
      f"range=[{asset_vol.min():.4f}, {asset_vol.max():.4f}]")
print(f"  Asset return : mean={asset_ret.mean():.4f}  "
      f"range=[{asset_ret.min():.4f}, {asset_ret.max():.4f}]")

# =============================================================================
# 4. MONTE CARLO RANDOM PORTFOLIOS
# =============================================================================
print(f"\n[4] Sampling {N_RANDOM} random portfolios (sparse Dirichlet, K~U[5,30]) â€¦")

rng       = np.random.default_rng(RNG_SEED)
rand_vols = np.zeros(N_RANDOM)
rand_rets = np.zeros(N_RANDOM)

for k in range(N_RANDOM):
    K_k     = int(rng.integers(5, 31))            # K ~ Uniform{5, ..., 30}
    idx     = rng.choice(N, size=K_k, replace=False)
    w_sub   = rng.dirichlet(np.ones(K_k))
    w       = np.zeros(N)
    w[idx]  = w_sub
    rand_rets[k] = float(w @ mu)
    rand_vols[k] = float(np.sqrt(w @ Sigma @ w))

rand_sharpe = np.where(rand_vols > 0, rand_rets / rand_vols, np.nan)
print(f"  Return range : [{rand_rets.min():.4f}, {rand_rets.max():.4f}]")
print(f"  Vol range    : [{rand_vols.min():.4f}, {rand_vols.max():.4f}]")
print(f"  Sharpe range : [{np.nanmin(rand_sharpe):.4f}, {np.nanmax(rand_sharpe):.4f}]")
print(f"  DEBUG rand_rets[:10]   : {rand_rets[:10]}")
print(f"  DEBUG rand_sharpe[:10] : {rand_sharpe[:10]}")
valid = rand_rets > 0
rand_vols   = rand_vols[valid]
rand_rets   = rand_rets[valid]
rand_sharpe = rand_sharpe[valid]
print(f"  Valid portfolios (ret>0): {valid.sum()} / {N_RANDOM}")

# =============================================================================
# 5. MINIMUM VARIANCE PORTFOLIO (MVP)
# =============================================================================
print("\n[5] Computing Minimum Variance Portfolio (MVP) â€¦")

def portfolio_variance(w):
    return float(w @ Sigma @ w)

def portfolio_variance_grad(w):
    return 2.0 * Sigma @ w

n_assets   = N
x0         = np.ones(n_assets) / n_assets
constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0,
                "jac": lambda w: np.ones(n_assets)}]
bounds      = [(0.0, 1.0)] * n_assets

res_mvp = minimize(
    portfolio_variance,
    x0,
    jac=portfolio_variance_grad,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"ftol": 1e-12, "maxiter": 1000},
)

if not res_mvp.success:
    print(f"  WARNING: MVP solver did not converge: {res_mvp.message}")

w_mvp     = np.maximum(res_mvp.x, 0.0)
w_mvp    /= w_mvp.sum()
ret_mvp   = float(w_mvp @ mu)
vol_mvp   = float(np.sqrt(w_mvp @ Sigma @ w_mvp))
sharpe_mvp = ret_mvp / vol_mvp if vol_mvp > 0 else np.nan
n_mvp      = int((w_mvp > 1e-4).sum())

print(f"  MVP  Return : {ret_mvp:.4f}  ({ret_mvp:.2%})")
print(f"  MVP  Vol    : {vol_mvp:.4f}  ({vol_mvp:.2%})")
print(f"  MVP  Sharpe : {sharpe_mvp:.4f}")
print(f"  MVP  n_hold : {n_mvp}")

# =============================================================================
# 6. MAXIMUM SHARPE PORTFOLIO (MSP)
# =============================================================================
print("\n[6] Computing Maximum Sharpe Portfolio (MSP) â€¦")

def neg_sharpe(w):
    r = float(w @ mu)
    v = float(np.sqrt(w @ Sigma @ w))
    return -(r / v) if v > 1e-12 else 0.0

def neg_sharpe_grad(w):
    r   = float(w @ mu)
    v   = float(np.sqrt(w @ Sigma @ w))
    if v < 1e-12:
        return np.zeros_like(w)
    dr  = mu
    dv  = (Sigma @ w) / v
    return -(dr * v - r * dv) / (v * v)

res_msp = minimize(
    neg_sharpe,
    x0,
    jac=neg_sharpe_grad,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"ftol": 1e-12, "maxiter": 1000},
)

if not res_msp.success:
    print(f"  WARNING: MSP solver did not converge: {res_msp.message}")

w_msp      = np.maximum(res_msp.x, 0.0)
w_msp     /= w_msp.sum()
ret_msp    = float(w_msp @ mu)
vol_msp    = float(np.sqrt(w_msp @ Sigma @ w_msp))
sharpe_msp = ret_msp / vol_msp if vol_msp > 0 else np.nan
n_msp      = int((w_msp > 1e-4).sum())

print(f"  MSP  Return : {ret_msp:.4f}  ({ret_msp:.2%})")
print(f"  MSP  Vol    : {vol_msp:.4f}  ({vol_msp:.2%})")
print(f"  MSP  Sharpe : {sharpe_msp:.4f}")
print(f"  MSP  n_hold : {n_msp}")

# =============================================================================
# 7. EQUALLY WEIGHTED PORTFOLIO (EW)
# =============================================================================
print("\n[7] Equally Weighted Portfolio (EW) â€¦")

w_ew      = np.ones(N) / N
ret_ew    = float(w_ew @ mu)
vol_ew    = float(np.sqrt(w_ew @ Sigma @ w_ew))
sharpe_ew = ret_ew / vol_ew if vol_ew > 0 else np.nan

print(f"  EW   Return : {ret_ew:.4f}  ({ret_ew:.2%})")
print(f"  EW   Vol    : {vol_ew:.4f}  ({vol_ew:.2%})")
print(f"  EW   Sharpe : {sharpe_ew:.4f}")

# =============================================================================
# 7b. EFFICIENT FRONTIER CURVE
# =============================================================================
print("\n[7b] Tracing efficient frontier (100 target-return points) â€¦")

N_FRONTIER  = 100
ret_max     = float(mu.max())          # upper bound: 100% in best-return stock
target_rets = np.linspace(ret_mvp, ret_max, N_FRONTIER)

frontier_vols = []
frontier_rets = []

for target in target_rets:
    ef_constraints = [
        {"type": "eq", "fun": lambda w: w.sum() - 1.0,
         "jac": lambda w: np.ones(N)},
        {"type": "eq", "fun": lambda w, t=target: float(w @ mu) - t,
         "jac": lambda w, _t=None: mu},
    ]
    res_ef = minimize(
        portfolio_variance,
        x0,
        jac=portfolio_variance_grad,
        method="SLSQP",
        bounds=bounds,
        constraints=ef_constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if res_ef.success and res_ef.x is not None:
        w_ef = np.maximum(res_ef.x, 0.0)
        if w_ef.sum() > 1e-8:
            w_ef /= w_ef.sum()
            frontier_vols.append(float(np.sqrt(w_ef @ Sigma @ w_ef)))
            frontier_rets.append(float(w_ef @ mu))

frontier_vols = np.array(frontier_vols)
frontier_rets = np.array(frontier_rets)

# Keep only the upper branch (efficient part: return >= MVP return)
mask_eff = frontier_rets >= ret_mvp - 1e-6
frontier_vols = frontier_vols[mask_eff]
frontier_rets = frontier_rets[mask_eff]

# Sort by vol for a clean line
order = np.argsort(frontier_vols)
frontier_vols = frontier_vols[order]
frontier_rets = frontier_rets[order]

print(f"  Points traced : {len(frontier_vols)}")
print(f"  Vol range     : [{frontier_vols.min():.4f}, {frontier_vols.max():.4f}]")
print(f"  Return range  : [{frontier_rets.min():.4f}, {frontier_rets.max():.4f}]")

# =============================================================================
# 8. BASELINE MIQP PORTFOLIO
# =============================================================================
print("\n[8] Baseline MIQP portfolio (from baseline_summary.csv) â€¦")

baseline_path = os.path.join(RESULTS_DIR, "baseline_summary.csv")
HAS_BASELINE  = os.path.exists(baseline_path)

if HAS_BASELINE:
    bl_summary  = pd.read_csv(baseline_path)
    ret_bl      = float(bl_summary["ann_return"].iloc[0])
    vol_bl      = float(bl_summary["ann_vol"].iloc[0])
    sharpe_bl   = float(bl_summary["sharpe"].iloc[0])
    print(f"  BL   Return : {ret_bl:.4f}  ({ret_bl:.2%})")
    print(f"  BL   Vol    : {vol_bl:.4f}  ({vol_bl:.2%})")
    print(f"  BL   Sharpe : {sharpe_bl:.4f}")
else:
    print("  baseline_summary.csv not found â€” baseline point will be omitted.")
    ret_bl = vol_bl = sharpe_bl = None

# =============================================================================
# 9. LABELS FOR INDIVIDUAL ASSETS
# =============================================================================
print("\n[9] Selecting asset labels â€¦")

def clean_ticker(t: str) -> str:
    for suffix in (" UN EQUITY", " UW EQUITY", " US EQUITY", " UP EQUITY",
                   " UA EQUITY", " UF EQUITY"):
        if t.upper().endswith(suffix):
            return t[: -len(suffix)].strip()
    # generic: strip last word if it is EQUITY
    parts = t.strip().split()
    if parts and parts[-1].upper() == "EQUITY":
        return " ".join(parts[:-1]).strip()
    return t.strip()

# 1 highest return asset
idx_hr = int(np.argmax(asset_ret))

# 1 lowest vol asset (excluding MVP stocks with weight > 1e-4)
non_mvp = ~(w_mvp > 1e-4)
if non_mvp.any():
    idx_lv = int(np.argmin(np.where(non_mvp, asset_vol, np.inf)))
else:
    idx_lv = int(np.argmin(asset_vol))

# 1 highest Sharpe asset
idx_hs = int(np.nanargmax(asset_sharpe))

# 1 asset nearest 75th percentile in both vol and return (upper-right)
vol_75 = np.percentile(asset_vol, 75)
ret_75 = np.percentile(asset_ret, 75)
idx_ur = int(np.argmin((asset_vol - vol_75)**2 + (asset_ret - ret_75)**2))

# 1 asset nearest 25th percentile in both vol and return (lower-left)
vol_25 = np.percentile(asset_vol, 25)
ret_25 = np.percentile(asset_ret, 25)
idx_ll = int(np.argmin((asset_vol - vol_25)**2 + (asset_ret - ret_25)**2))

label_specs = [
    (idx_hr, (+15, +5),  "highest_return"),
    (idx_lv, (-50, -15), "lowest_vol"),
    (idx_hs, (+10, +10), "highest_sharpe"),
    (idx_ur, (+10, -15), "upper_right"),
    (idx_ll, (-40, +10), "lower_left"),
]

print("  Labelled assets:")
for idx, offset, role in label_specs:
    print(f"    [{idx:>4}] {clean_ticker(tickers[idx]):<20}  "
          f"ret={asset_ret[idx]:.4f}  vol={asset_vol[idx]:.4f}  "
          f"sharpe={asset_sharpe[idx]:.4f}  role={role}")

# =============================================================================
# 10. FIGURE
# =============================================================================
print("\n[10] Rendering figure â€¦")

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

fig, ax = plt.subplots(figsize=(10, 7))

# -- (a) Individual stocks (grey, small) --------------------------------------
ax.scatter(
    asset_vol, asset_ret,
    c="lightgrey", s=8, alpha=0.25, zorder=2,
    label="_nolegend_",
)

# -- (b) Random portfolios (viridis by Sharpe) --------------------------------
sharpe_p5  = float(np.nanpercentile(rand_sharpe, 5))
sharpe_p95 = float(np.nanpercentile(rand_sharpe, 95))

sc = ax.scatter(
    rand_vols, rand_rets,
    c=rand_sharpe,
    cmap="viridis",
    vmin=sharpe_p5,
    vmax=sharpe_p95,
    s=22,
    alpha=0.75,
    zorder=3,
    label="_nolegend_",
)
cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.03)
cbar.set_label("Sharpe Ratio", fontsize=10)
cbar.ax.tick_params(labelsize=9)

# -- (b2) Efficient frontier curve --------------------------------------------
ax.plot(
    frontier_vols, frontier_rets,
    color="black", linewidth=2, zorder=5,
    label="Efficient Frontier",
)

# -- (b3) Labels for spread assets --------------------------------------------
for idx, (xt, yt), _role in label_specs:
    ax.annotate(
        clean_ticker(tickers[idx]),
        xy=(asset_vol[idx], asset_ret[idx]),
        xytext=(xt, yt),
        textcoords="offset points",
        fontsize=8,
        color="#333333",
        zorder=8,
        arrowprops=dict(arrowstyle="-", color="#888888",
                        lw=0.6, shrinkA=0, shrinkB=3),
    )

# -- (c) Special portfolios ---------------------------------------------------
STAR_SIZE    = 280
DIAMOND_SIZE = 200
TRI_SIZE     = 200

ax.scatter(
    vol_mvp, ret_mvp,
    marker="*", s=STAR_SIZE, color="#1f3c6b", zorder=7,
    label=f"Min Variance  (Sharpe {sharpe_mvp:.2f})",
)
ax.scatter(
    vol_msp, ret_msp,
    marker="*", s=STAR_SIZE, color="gold", edgecolors="#888800",
    linewidths=0.6, zorder=7,
    label=f"Max Sharpe    (Sharpe {sharpe_msp:.2f})",
)
ax.scatter(
    vol_ew, ret_ew,
    marker="D", s=DIAMOND_SIZE, color="#d62728", zorder=7,
    label=f"Equal Weight  (Sharpe {sharpe_ew:.2f})",
)

if HAS_BASELINE:
    ax.scatter(
        vol_bl, ret_bl,
        marker="^", s=TRI_SIZE, color="#2ca02c", zorder=7,
        label=f"Baseline MIQP (Sharpe {sharpe_bl:.2f})",
    )

# -- (d) Axes formatting ------------------------------------------------------
ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
ax.set_xlabel("Annualised Volatility", fontsize=12)
ax.set_ylabel("Annualised Return", fontsize=12)

ax.set_title(
    "Efficient Frontier â€” S&P 500 Universe (2016â€“2022)",
    fontsize=13, fontweight="bold", pad=10,
)
ax.text(
    0.5, 1.015,
    f"{N} stocks Â· Ledoit-Wolf covariance Â· Monte Carlo simulation (n={N_RANDOM})",
    transform=ax.transAxes,
    ha="center", va="bottom",
    fontsize=9, color="#777777",
)

ax.legend(
    loc="upper left",
    fontsize=9.5,
    framealpha=0.85,
    edgecolor="#cccccc",
    handletextpad=0.5,
    # Efficient Frontier line entry appears first, then the 4 markers
)

fig.text(0.99, 0.01, "Source: Author's elaboration.",
         ha='right', va='bottom', fontsize=8,
         color='grey', style='italic',
         transform=fig.transFigure)

plt.tight_layout(rect=[0, 0.04, 1, 1])

out_path = os.path.join(FIGURES_DIR, "efficient_frontier.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"  Saved: {out_path}")

# =============================================================================
# 11. SUMMARY PRINT
# =============================================================================
print("\n" + "=" * 70)
print("  EFFICIENT FRONTIER â€” PORTFOLIO SUMMARY")
print("=" * 70)
print(f"  {'Portfolio':<24} {'Ann Return':>12} {'Ann Vol':>10} {'Sharpe':>10}")
print("  " + "-" * 58)
for label, r, v, s in [
    ("Min Variance (MVP)",  ret_mvp, vol_mvp, sharpe_mvp),
    ("Max Sharpe (MSP)",    ret_msp, vol_msp, sharpe_msp),
    ("Equal Weight (EW)",   ret_ew,  vol_ew,  sharpe_ew),
]:
    print(f"  {label:<24} {r:>11.2%}  {v:>9.2%}  {s:>9.4f}")
if HAS_BASELINE:
    print(f"  {'Baseline MIQP':<24} {ret_bl:>11.2%}  {vol_bl:>9.2%}  {sharpe_bl:>9.4f}")
print("=" * 70)
print("  VISUALISATION COMPLETE")
print("=" * 70)
