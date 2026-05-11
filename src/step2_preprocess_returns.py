"""
step2_preprocess_returns.py
===================
PURPOSE : Compute log returns and estimate covariance matrices for the
          clean universe produced by Step 1.

          Step 1 answered: which stocks survive structural cleaning?
          Step 2 answers:  what does the return distribution look like,
                           and what full-window diagnostic covariance estimates
                           are obtained for the cleaned universe?

          NOTE: mu (expected returns) is NOT computed here.
                - Baseline Markowitz: rolling historical mean computed
                  inside the optimization step.
                - ML model: mu supplied as predictions from step4_feature_engineering.py.

INPUTS  : data/clean/meta_step1_clean.csv      (from step1, structural-cleaning snapshot)
          data/clean/prices_step1_clean.csv    (from step1)
          data/clean/tri_step1_clean.csv       (from step1)
          data/clean/mkt_cap_step1_clean.csv   (from step1)
          data/clean/volume_step1_clean.csv    (from step1)

OUTPUTS : data/clean/returns.csv            -- daily log-return matrix (full 2010-2025)
          data/clean/Sigma_sample.csv        -- sample covariance, annualised, full model window
                                               (diagnostic only; not used directly in OOS optimizer)
          data/clean/Sigma_lw.csv            -- Ledoit-Wolf shrinkage covariance, annualised,
                                               full model window (diagnostic only)
          data/clean/Sigma_oas.csv           -- OAS shrinkage covariance, annualised,
                                               full model window (diagnostic only)
          data/clean/stock_diagnostics.csv   -- per-stock: ann_return, ann_vol, sharpe,
                                               skewness, kurtosis, pct_zero_ret, n_spikes
          data/clean/corr_summary.csv        -- correlation structure metrics
          data/clean/preproc_summary.csv     -- one-row dataset overview
          data/clean/preproc_log.txt         -- full methodology log
          data/figures/vol_distribution.png
          data/figures/return_distribution.png
          data/figures/corr_distribution.png
          data/figures/sector_breakdown.png

METHODOLOGICAL CHOICES (recorded here and in the log):
  1. Returns source   : log returns computed from TRI (total return index, gross
                        dividends) to capture dividend income.
                        Prices retained for Amihud ratio only.
  2. Time window      : full sample 2010-01-01 to 2025-12-31 (inherited from Step 1).
                        Model/estimation window: 2016-01-01 to 2025-12-31.
                        Warm-up period 2010-2015 excluded from parameter estimation.
  3. Missing data     : validation only; structural cleaning completed in Step 1.
                        NaN in warm-up is expected (late IPO/entry dates).
                        NaN in model window (2016-2025) triggers a WARNING.
  4. Covariance       : annualised sample + Ledoit-Wolf + OAS shrinkage estimators,
                        estimated on the full model window (2016-2025) for descriptive
                        diagnostics only. In the out-of-sample optimization, covariance
                        matrices are re-estimated at each rebalance date using only past
                        returns available at the time of portfolio formation.
                        Following the out-of-sample evaluation logic in DeMiguel, Garlappi
                        and Uppal (2009), full-window covariance estimates are not used as
                        portfolio inputs in the OOS backtest.

Author  : Anila Vata 
"""

# =============================================================================
# 0. IMPORTS AND PATHS
# =============================================================================
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf, OAS

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("future.no_silent_downcasting", True)

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "data", "clean")
FIGURES_DIR = os.path.join(BASE_DIR, "data", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

TRADING_DAYS = 252
SAMPLE_START = "2010-01-01"
SAMPLE_END   = "2025-12-31"
MODEL_START  = "2016-01-01"
MODEL_END    = "2025-12-31"
WARMUP_END   = "2015-12-31"

# Logging accumulator -- written to file at the end
log_lines = []
def log(msg=""):
    print(msg)
    log_lines.append(str(msg))

log("=" * 70)
log("  STEP 2 -- RETURNS AND COVARIANCE PREPROCESSING")
log("=" * 70)

# =============================================================================
# 1. LOAD STEP 1 CLEAN OUTPUTS
# =============================================================================
log("\n-- 1. LOAD STEP 1 CLEAN OUTPUTS -------------------------------------")

meta_path    = os.path.join(RESULTS_DIR, "meta_step1_clean.csv")
prices_path  = os.path.join(RESULTS_DIR, "prices_step1_clean.csv")
tri_path     = os.path.join(RESULTS_DIR, "tri_step1_clean.csv")
mktcap_path  = os.path.join(RESULTS_DIR, "mkt_cap_step1_clean.csv")
volume_path  = os.path.join(RESULTS_DIR, "volume_step1_clean.csv")

for path in [meta_path, prices_path, tri_path, mktcap_path, volume_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Run step1_load.py first.")

meta    = pd.read_csv(meta_path)
prices  = pd.read_csv(prices_path, index_col=0, parse_dates=True)
tri     = pd.read_csv(tri_path,    index_col=0, parse_dates=True)
mkt_cap = pd.read_csv(mktcap_path, index_col=0, parse_dates=True)
volume  = pd.read_csv(volume_path, index_col=0, parse_dates=True)

# Normalise column names
meta.columns    = meta.columns.str.strip().str.lower()
prices.columns  = prices.columns.str.strip().str.upper()   # tickers stay UPPER
tri.columns     = tri.columns.str.strip().str.upper()
mkt_cap.columns = mkt_cap.columns.str.strip().str.upper()
volume.columns  = volume.columns.str.strip().str.upper()
prices  = prices.sort_index()
tri     = tri.sort_index()
mkt_cap = mkt_cap.sort_index()
volume  = volume.sort_index()

if "ticker" not in meta.columns and "index" in meta.columns:
    meta = meta.rename(columns={"index": "ticker"})
meta["ticker"] = meta["ticker"].astype(str).str.strip().str.upper()

n_stocks = len(meta)
n_days   = len(prices)

log(f"  meta    : {meta.shape[0]} rows x {meta.shape[1]} columns")
log(f"  prices  : {prices.shape[0]} days x {prices.shape[1]} stocks")
log(f"  tri     : {tri.shape[0]} days x {tri.shape[1]} stocks")
log(f"  mkt_cap : {mkt_cap.shape[0]} days x {mkt_cap.shape[1]} stocks")
log(f"  volume  : {volume.shape[0]} days x {volume.shape[1]} stocks")

# Alignment check: meta vs prices
meta_tickers  = set(meta["ticker"])
price_tickers = set(prices.columns)
if meta_tickers != price_tickers:
    only_meta  = meta_tickers - price_tickers
    only_price = price_tickers - meta_tickers
    raise ValueError(
        f"Ticker mismatch between meta and prices.\n"
        f"  Only in meta  : {only_meta}\n"
        f"  Only in prices: {only_price}\n"
        "Re-run step1.")
log(f"  Ticker alignment (meta/prices) : OK ({n_stocks} stocks in both files)")

# Alignment check: tri vs prices
tri_tickers = set(tri.columns)
if tri_tickers != price_tickers:
    only_tri   = tri_tickers - price_tickers
    only_price = price_tickers - tri_tickers
    raise ValueError(
        f"Ticker mismatch between tri and prices.\n"
        f"  Only in tri   : {only_tri}\n"
        f"  Only in prices: {only_price}\n"
        "Re-run step1.")
log(f"  Ticker alignment (tri/prices)  : OK")

# Alignment check: mkt_cap vs prices
mktcap_tickers = set(mkt_cap.columns)
if mktcap_tickers != price_tickers:
    only_mktcap = mktcap_tickers - price_tickers
    only_price  = price_tickers - mktcap_tickers
    raise ValueError(
        f"Ticker mismatch between mkt_cap and prices.\n"
        f"  Only in mkt_cap: {only_mktcap}\n"
        f"  Only in prices : {only_price}\n"
        "Re-run step1.")
log(f"  Ticker alignment (mkt_cap/prices): OK")

# Date range check
actual_start = prices.index[0].date().isoformat()
actual_end   = prices.index[-1].date().isoformat()
log(f"  Date range       : {actual_start} to {actual_end}")
if actual_start != SAMPLE_START or actual_end != SAMPLE_END:
    log(f"  NOTE: expected {SAMPLE_START} to {SAMPLE_END} -- actual range may "
        f"differ slightly due to weekend/holiday alignment.")
else:
    log(f"  Date range       : matches expected window  OK")

# =============================================================================
# 2. COMPUTE DAILY LOG RETURNS FROM TRI
# =============================================================================
log("\n-- 2. COMPUTE DAILY LOG RETURNS -------------------------------------")

# returns computed from TRI to capture dividend income --
# prices retained for Amihud ratio only
returns = np.log(tri / tri.shift(1)).iloc[1:]   # drop first NaN row

n_ret_days   = len(returns)
n_ret_stocks = returns.shape[1]

log(f"  Return matrix shape : {n_ret_days} days x {n_ret_stocks} stocks")
log(f"  Return source       : TRI (total return index, gross dividends)")
log(f"  Full series         : {returns.index[0].date()} to "
    f"{returns.index[-1].date()}")

# NaN check: report separately for warm-up and model windows
returns_warmup = returns.loc[SAMPLE_START:WARMUP_END]
returns_model  = returns.loc[MODEL_START:MODEL_END]
n_model_days   = len(returns_model)

nan_warmup = int(returns_warmup.isna().sum().sum())
nan_model  = int(returns_model.isna().sum().sum())

log(f"\n  NaN in warm-up window (2010-2015)  : {nan_warmup} cells"
    "  -- expected (stocks with late IPO/entry dates)")
if nan_model > 0:
    log(f"  NaN in model window  (2016-2025)   : {nan_model} cells"
        "  WARNING -- investigate before running optimizer")
else:
    log(f"  NaN in model window  (2016-2025)   : {nan_model} cells  OK")

log(f"  Model window        : {MODEL_START} to {MODEL_END}  ({n_model_days} days)")

# =============================================================================
# 2b. ZERO-RETURN FILTER (stale Bloomberg TRI data)
# =============================================================================
log("\n-- 2b. ZERO-RETURN FILTER -------------------------------------------")
log(f"  Threshold : >5% zero returns in model window ({MODEL_START} to {MODEL_END})")
log("  Rationale : Bloomberg TRI feeds sometimes stop updating for specific")
log("              stocks while forward-filling flat prices. This produces")
log("              long runs of exact-zero returns that are NOT genuine.")
log("              Such stocks corrupt features, covariance, and portfolio")
log("              returns and must be removed before estimation.")

ZERO_THRESHOLD = 0.05
zero_pct_model = (returns_model == 0).mean()
stale_tickers  = zero_pct_model[zero_pct_model > ZERO_THRESHOLD].index.tolist()

log(f"\n  Stocks exceeding threshold : {len(stale_tickers)}")
if stale_tickers:
    log(f"\n  {'Ticker':<25} {'Zero% (model window)':>22}")
    log("  " + "-" * 50)
    for tkr in sorted(stale_tickers, key=lambda t: -zero_pct_model[t]):
        log(f"  {tkr:<25} {zero_pct_model[tkr]:>21.1%}")

    # Drop from all DataFrames
    returns       = returns.drop(columns=stale_tickers)
    prices        = prices.drop(columns=stale_tickers)
    tri           = tri.drop(columns=stale_tickers)
    mkt_cap       = mkt_cap.drop(columns=stale_tickers)
    volume        = volume.drop(columns=stale_tickers)
    meta          = meta[~meta["ticker"].isin(stale_tickers)].reset_index(drop=True)

    log(f"\n  Universe after zero-return filter : {len(meta)} stocks"
        f"  ({len(stale_tickers)} removed)")
else:
    log("  No stocks removed.")

# =============================================================================
# 2c. HARDCODED EXCLUSION LIST (manually verified data errors)
# =============================================================================
log("\n-- 2c. HARDCODED EXCLUSIONS -----------------------------------------")

HARDCODED_EXCLUSIONS = {
    "SATS UW EQUITY": (
        "Unverifiable return spike of +53.21% on 2025-08-26 "
        "followed by +14.42% the next day; inconsistent with "
        "any documented corporate event; suspected TRI feed error."
    ),
}

to_exclude = [t for t in HARDCODED_EXCLUSIONS.keys() if t in returns.columns]
if to_exclude:
    returns = returns.drop(columns=to_exclude)
    prices  = prices.drop(columns=to_exclude)
    tri     = tri.drop(columns=to_exclude)
    mkt_cap = mkt_cap.drop(columns=to_exclude)
    volume  = volume.drop(columns=to_exclude)
    meta    = meta[~meta["ticker"].isin(to_exclude)].reset_index(drop=True)

    for t in to_exclude:
        log(f"  Removed : {t}")
        log(f"    Reason : {HARDCODED_EXCLUSIONS[t]}")
    log(f"  Universe after hardcoded exclusions : {len(meta)} stocks"
        f"  ({len(to_exclude)} removed)")
else:
    log("  No hardcoded exclusions applied (tickers not present in universe).")

log(f"\n  Final return matrix : {returns.shape[0]} days x {returns.shape[1]} stocks")

# =============================================================================
# 2d. FINAL UNIVERSE FILTER: DUPLICATE SHARE CLASS REMOVAL (GOOGL â†’ keep GOOG)
# =============================================================================
log("\n-- 2d. FINAL UNIVERSE FILTER: DUPLICATE SHARE CLASS REMOVAL ----------")

GOOGL_TICKER = "GOOGL UW EQUITY"
GOOG_TICKER  = "GOOG UW EQUITY"

if GOOGL_TICKER in returns.columns and GOOG_TICKER in returns.columns:
    _googl_corr = round(
        returns.loc[MODEL_START:MODEL_END, [GOOG_TICKER, GOOGL_TICKER]]
        .corr().iloc[0, 1], 4
    )
    returns = returns.drop(columns=[GOOGL_TICKER])
    prices  = prices.drop(columns=[GOOGL_TICKER])
    tri     = tri.drop(columns=[GOOGL_TICKER])
    mkt_cap = mkt_cap.drop(columns=[GOOGL_TICKER])
    volume  = volume.drop(columns=[GOOGL_TICKER])
    meta    = meta[meta["ticker"] != GOOGL_TICKER].reset_index(drop=True)

    log(f"  Removed duplicate share class: {GOOGL_TICKER} "
        f"(correlation {_googl_corr:.4f} with {GOOG_TICKER})")
    log(f"  Universe after share class filter: {len(meta)} stocks")
else:
    _googl_corr = None
    if GOOGL_TICKER not in returns.columns:
        log(f"  NOTE: {GOOGL_TICKER} not found in universe -- skipping removal.")
    if GOOG_TICKER not in returns.columns:
        log(f"  NOTE: {GOOG_TICKER} not found in universe -- cannot determine pair.")

# THESIS NOTE (Chapter 3 â€” Data and Methodology):
# The S&P 500 universe includes two share classes of Alphabet Inc.:
# GOOG (Class C, no voting rights) and GOOGL (Class A, voting rights).
# These two series have a sample correlation of 0.9952 over the model window,
# making them near-perfect substitutes from a portfolio optimization standpoint.
# Following standard practice in the empirical asset pricing literature
# (see Gu, Kelly & Xiu, 2020, Review of Financial Studies), we retain only
# the more liquid share class (GOOG) and remove GOOGL from the universe.
# This prevents the optimizer from artificially doubling its Alphabet exposure
# by splitting weight across two near-identical assets.

# -- Recompute all derived quantities after all universe filters --
returns_model  = returns.loc[MODEL_START:MODEL_END]
returns_warmup = returns.loc[SAMPLE_START:WARMUP_END]
tickers        = prices.columns.tolist()
n_stocks       = len(meta)
n_model_days   = len(returns_model)
R              = returns_model.values

log("\n  NOTE: Covariance matrices saved in Step 2 are full-model-window")
log("  diagnostics only. In the out-of-sample optimization, covariance")
log("  matrices are re-estimated at each rebalancing date using only")
log("  past returns available at the time of portfolio formation.")

# =============================================================================
log("\n-- 3. ESTIMATE COVARIANCE MATRICES ----------------------------------")
log(f"  All matrices annualised by multiplying by {TRADING_DAYS}.")
log(f"  Estimated on model window ({MODEL_START} to {MODEL_END}) only.")

Sigma_sample = np.cov(R, rowvar=False) * TRADING_DAYS
eig_sample   = np.linalg.eigvalsh(Sigma_sample)
log(f"\n  Sample covariance:")
log(f"    Shape            : {Sigma_sample.shape}")
log(f"    Min eigenvalue   : {eig_sample.min():.6f}")
log(f"    Condition number : {np.linalg.cond(Sigma_sample):.2e}")

lw_model  = LedoitWolf().fit(R)
Sigma_lw  = lw_model.covariance_ * TRADING_DAYS
lw_shrink = lw_model.shrinkage_
eig_lw    = np.linalg.eigvalsh(Sigma_lw)
log(f"\n  Ledoit-Wolf covariance (LW 2004 JMVA, identity target):")
log(f"    Shrinkage coeff  : {lw_shrink:.4f}")
log(f"    Min eigenvalue   : {eig_lw.min():.6f}")
log(f"    Condition number : {np.linalg.cond(Sigma_lw):.2e}")

oas_model  = OAS().fit(R)
Sigma_oas  = oas_model.covariance_ * TRADING_DAYS
oas_shrink = oas_model.shrinkage_
eig_oas    = np.linalg.eigvalsh(Sigma_oas)
log(f"\n  OAS covariance:")
log(f"    Shrinkage coeff  : {oas_shrink:.4f}")
log(f"    Min eigenvalue   : {eig_oas.min():.6f}")
log(f"    Condition number : {np.linalg.cond(Sigma_oas):.2e}")

# =============================================================================
# 3b. HIGH-CORRELATION PAIRS DIAGNOSTIC
# =============================================================================
log("\n-- 3b. HIGH-CORRELATION PAIRS DIAGNOSTIC ----------------------------")
log("  Threshold : pairwise correlation > 0.95 (sample estimator, model window)")

_vol_s = np.sqrt(np.diag(Sigma_sample))
_D_s   = np.diag(1.0 / _vol_s)
_C_s   = _D_s @ Sigma_sample @ _D_s          # sample correlation matrix

n_tickers       = len(tickers)
high_corr_pairs = []
for _i in range(n_tickers):
    for _j in range(_i + 1, n_tickers):
        if _C_s[_i, _j] > 0.95:
            high_corr_pairs.append({
                "ticker_i"    : tickers[_i],
                "ticker_j"    : tickers[_j],
                "correlation" : round(_C_s[_i, _j], 6),
            })

log(f"\n  Pairs with correlation > 0.95 : {len(high_corr_pairs)}")
if high_corr_pairs:
    log(f"\n  {'Ticker i':<12} {'Ticker j':<12} {'Correlation':>12}")
    log("  " + "-" * 38)
    for _p in sorted(high_corr_pairs, key=lambda x: -x["correlation"]):
        log(f"  {_p['ticker_i']:<12} {_p['ticker_j']:<12} {_p['correlation']:>12.6f}")
else:
    log("  No pairs exceed the threshold.")

high_corr_df = (pd.DataFrame(high_corr_pairs) if high_corr_pairs
                else pd.DataFrame(columns=["ticker_i", "ticker_j", "correlation"]))
_hc_path = os.path.join(RESULTS_DIR, "high_corr_pairs.csv")
high_corr_df.to_csv(_hc_path, index=False)
log(f"\n  Saved: high_corr_pairs.csv  ({len(high_corr_df)} rows)")

# =============================================================================
# 3c. UNKNOWN-SECTOR CHECK
# =============================================================================
log("\n-- 3c. UNKNOWN-SECTOR CHECK -----------------------------------------")
if "sector" in meta.columns:
    _unknown_mask    = meta["sector"].astype(str).str.strip().str.lower() == "unknown"
    _unknown_tickers = meta.loc[_unknown_mask, "ticker"].tolist()
    if _unknown_tickers:
        log(f"  Tickers with Sector = 'Unknown' ({len(_unknown_tickers)}):")
        for _tkr in _unknown_tickers:
            log(f"    {_tkr}")
    else:
        log("  No tickers with Sector = 'Unknown'.")
else:
    log("  'sector' column not found in meta_clean -- skipping check.")

# =============================================================================
# 4. RETURN QUALITY DIAGNOSTICS (model window only)
# =============================================================================
log("\n-- 4. RETURN QUALITY DIAGNOSTICS ------------------------------------")
log(f"  Checks performed on model window ({MODEL_START} to {MODEL_END}).")
log("  Diagnostic only -- no stocks removed here.")

missing_by_stock    = returns_model.isna().sum()
stocks_with_missing = (missing_by_stock > 0).sum()
log(f"\n  Stocks with any missing returns : {stocks_with_missing}")

spike_mask   = returns_model.abs() > 0.50
spike_count  = spike_mask.sum().sum()
spike_stocks = (spike_mask.sum(axis=0) > 0).sum()
log(f"  Single-day |return| > 50%       : {spike_count} observations "
    f"in {spike_stocks} stocks")

if spike_count > 0:
    _ticker_to_sector = (meta.set_index("ticker")["sector"].to_dict()
                         if "sector" in meta.columns else {})
    _name_col = next((c for c in meta.columns
                      if c in ("security_name", "name", "company_name")), None)
    _ticker_to_name = (meta.set_index("ticker")[_name_col].to_dict()
                       if _name_col else {})

    _spike_rows = []
    for _col in returns_model.columns[spike_mask.any(axis=0)]:
        for _dt in returns_model.index[spike_mask[_col]]:
            _ret = returns_model.at[_dt, _col]
            _spike_rows.append({
                "date"          : _dt.date().isoformat(),
                "ticker"        : _col,
                "return"        : round(_ret, 6),
                "abs_return"    : round(abs(_ret), 6),
                "sector"        : _ticker_to_sector.get(_col, "n/a"),
                "security_name" : _ticker_to_name.get(_col, "n/a"),
            })

    spike_diag = (pd.DataFrame(_spike_rows)
                  .sort_values("abs_return", ascending=False)
                  .reset_index(drop=True))

    _col_w = {"date": 12, "ticker": 26, "return": 10, "abs_return": 11,
              "sector": 32, "security_name": 40}
    _hdr = (f"  {'date':<{_col_w['date']}} {'ticker':<{_col_w['ticker']}}"
            f" {'return':>{_col_w['return']}} {'abs_return':>{_col_w['abs_return']}}"
            f" {'sector':<{_col_w['sector']}} {'security_name':<{_col_w['security_name']}}")
    log(f"\n  Spike observations (|return| > 50%, model window) -- {len(spike_diag)} rows:")
    log(_hdr)
    log("  " + "-" * (sum(_col_w.values()) + len(_col_w) - 1))
    for _, _r in spike_diag.iterrows():
        log(f"  {_r['date']:<{_col_w['date']}} {_r['ticker']:<{_col_w['ticker']}}"
            f" {_r['return']:>{_col_w['return']}.4f}"
            f" {_r['abs_return']:>{_col_w['abs_return']}.4f}"
            f" {_r['sector']:<{_col_w['sector']}}"
            f" {_r['security_name']:<{_col_w['security_name']}}")

    _spike_path = os.path.join(RESULTS_DIR, "spike_diagnostics.csv")
    spike_diag.to_csv(_spike_path, index=False)
    log(f"\n  Saved: spike_diagnostics.csv  ({len(spike_diag)} rows)")
    log("  Diagnostic only -- no stocks removed here.")

zero_mask         = (returns_model == 0)
zero_pct_by_stock = zero_mask.mean()
high_zero         = (zero_pct_by_stock > 0.10).sum()
log(f"  Stocks with >10% zero returns   : {high_zero}  "
    f"(potential stale-price / low-liquidity flag)")

if high_zero > 0:
    log("\n  High zero-return stocks (ticker | sector | pct_zero_ret):")
    high_zero_tickers = zero_pct_by_stock[zero_pct_by_stock > 0.10].index.tolist()
    ticker_to_sector  = meta.set_index("ticker")["sector"].to_dict() \
                        if "sector" in meta.columns else {}
    for tkr in high_zero_tickers:
        pct    = zero_pct_by_stock[tkr] * 100
        sector = ticker_to_sector.get(tkr, "n/a")
        log(f"    {tkr:<10}  sector={sector:<30}  pct_zero_ret={pct:.1f}%")

if spike_count > 0 or high_zero > 0:
    log("  NOTE: flagged stocks retained -- review before final thesis submission.")

# =============================================================================
# 5. BASIC DESCRIPTIVE STATISTICS (model window only)
# =============================================================================
log("\n-- 5. BASIC DESCRIPTIVE STATISTICS ----------------------------------")
log(f"  All statistics computed on model window ({MODEL_START} to {MODEL_END}).")

ann_vol   = returns_model.std()  * np.sqrt(TRADING_DAYS)
ann_ret   = returns_model.mean() * TRADING_DAYS
skew_vals = returns_model.skew()
kurt_vals = returns_model.kurtosis()   # excess kurtosis (Fisher definition)

log(f"\n  Annualised volatility (cross-stock):")
log(f"    Min    : {ann_vol.min():.4f}")
log(f"    Median : {ann_vol.median():.4f}")
log(f"    Mean   : {ann_vol.mean():.4f}")
log(f"    Max    : {ann_vol.max():.4f}")

log(f"\n  Annualised mean return (cross-stock):")
log(f"    Min    : {ann_ret.min():.4f}")
log(f"    Median : {ann_ret.median():.4f}")
log(f"    Mean   : {ann_ret.mean():.4f}")
log(f"    Max    : {ann_ret.max():.4f}")

log(f"\n  Skewness (cross-stock mean):")
log(f"    Mean   : {skew_vals.mean():.3f}  "
    f"Std : {skew_vals.std():.3f}  "
    f"Range : [{skew_vals.min():.2f}, {skew_vals.max():.2f}]")

log(f"\n  Excess kurtosis (cross-stock mean):")
log(f"    Mean   : {kurt_vals.mean():.3f}  "
    f"(heavy tails expected in daily equity returns)")

# Correlation structure summary for all three estimators
log(f"\n  Correlation structure (all three estimators):")

def corr_stats(Sigma, label):
    vol = np.sqrt(np.diag(Sigma))
    D   = np.diag(1.0 / vol)
    C   = D @ Sigma @ D
    n   = C.shape[0]
    idx = np.triu_indices(n, k=1)
    off = C[idx]
    log(f"\n    {label}:")
    log(f"      Avg off-diagonal correlation : {off.mean():.4f}")
    log(f"      Std of correlations          : {off.std():.4f}")
    log(f"      Min / Max correlation        : [{off.min():.4f}, {off.max():.4f}]")
    log(f"      Pct of pairs > 0.50          : {(off > 0.50).mean()*100:.1f}%")
    log(f"      Pct of pairs < 0             : {(off < 0).mean()*100:.1f}%")
    return {"estimator"    : label,
            "avg_corr"     : round(off.mean(), 4),
            "std_corr"     : round(off.std(), 4),
            "min_corr"     : round(off.min(), 4),
            "max_corr"     : round(off.max(), 4),
            "pct_gt_050"   : round((off > 0.50).mean()*100, 2),
            "pct_negative" : round((off < 0).mean()*100, 2)}, off

corr_rows = []
off_diags = {}
for Sig, lbl in [(Sigma_sample, "Sample"),
                 (Sigma_lw,     "Ledoit-Wolf"),
                 (Sigma_oas,    "OAS")]:
    row, off = corr_stats(Sig, lbl)
    corr_rows.append(row)
    off_diags[lbl] = off

corr_summary = pd.DataFrame(corr_rows)

# Per-stock diagnostics table
diag = pd.DataFrame({
    "ticker"       : returns_model.columns,
    "ann_return"   : ann_ret.values,
    "ann_vol"      : ann_vol.values,
    "sharpe"       : (ann_ret / ann_vol).values,
    "skewness"     : skew_vals.values,
    "kurtosis"     : kurt_vals.values,
    "pct_zero_ret" : zero_pct_by_stock.values,
    "n_spikes"     : spike_mask.sum(axis=0).values,
})

# =============================================================================
# 6. SECTOR AND UNIVERSE SUMMARY
# =============================================================================
log("\n-- 6. SECTOR AND UNIVERSE SUMMARY -----------------------------------")

log(f"\n  Total stocks        : {n_stocks}")
log(f"  Total trading days  : {n_days}")
log(f"  Model window days   : {n_model_days}")

if "sector" in meta.columns:
    sector_counts = meta["sector"].value_counts()
    log(f"\n  Sector breakdown ({len(sector_counts)} sectors):")
    for sector, count in sector_counts.items():
        log(f"    {sector:<40} {count:>4} stocks")

if "country" in meta.columns:
    country_counts = meta["country"].value_counts()
    top10 = country_counts.head(10)
    log(f"\n  Country breakdown (top 10):")
    for country, count in top10.items():
        log(f"    {country:<40} {count:>4} stocks")

# =============================================================================
# 7. SAVE ALL OUTPUTS
# =============================================================================
log("\n-- 7. SAVING OUTPUTS ------------------------------------------------")

def save_csv(df_or_arr, fname, index=True, tickers_idx=None, tickers_col=None):
    path = os.path.join(RESULTS_DIR, fname)
    if isinstance(df_or_arr, np.ndarray):
        df = pd.DataFrame(df_or_arr,
                          index=tickers_idx if tickers_idx is not None else None,
                          columns=tickers_col if tickers_col is not None else None)
    else:
        df = df_or_arr
    df.to_csv(path, index=index)
    log(f"  Saved: {fname}  ({df.shape})")

save_csv(returns,      "returns.csv")
save_csv(Sigma_sample, "Sigma_sample.csv",
         tickers_idx=tickers, tickers_col=tickers)
save_csv(Sigma_lw,     "Sigma_lw.csv",
         tickers_idx=tickers, tickers_col=tickers)
save_csv(Sigma_oas,    "Sigma_oas.csv",
         tickers_idx=tickers, tickers_col=tickers)
save_csv(diag,         "stock_diagnostics.csv",  index=False)
save_csv(corr_summary, "corr_summary.csv",        index=False)

log("  NOTE: saving final modelling universe under standard downstream filenames. "
    "Step-1 structural-cleaning files are preserved separately as *_step1_clean.csv.")
# All downstream steps (3-6) load from these paths, so they must stay
# in sync with the filtered universe.
save_csv(meta,    "meta_clean.csv",     index=False)
save_csv(prices,  "prices_clean.csv")
save_csv(tri,     "tri_clean.csv")
save_csv(mkt_cap, "mkt_cap_clean.csv")
save_csv(volume,  "volume_clean.csv")

preproc_summary = pd.DataFrame([{
    "n_stocks"          : n_stocks,
    "n_return_days"     : n_ret_days,
    "n_model_days"      : n_model_days,
    "sample_start"      : SAMPLE_START,
    "sample_end"        : SAMPLE_END,
    "model_start"       : MODEL_START,
    "model_end"         : MODEL_END,
    "vol_mean"          : round(ann_vol.mean(), 4),
    "vol_median"        : round(ann_vol.median(), 4),
    "ann_ret_mean"      : round(ann_ret.mean(), 4),
    "ann_ret_median"    : round(ann_ret.median(), 4),
    "skew_mean"         : round(skew_vals.mean(), 4),
    "kurt_mean"         : round(kurt_vals.mean(), 4),
    "lw_shrinkage"      : round(lw_shrink, 4),
    "oas_shrinkage"     : round(oas_shrink, 4),
    "avg_corr_sample"   : corr_rows[0]["avg_corr"],
    "avg_corr_lw"       : corr_rows[1]["avg_corr"],
    "avg_corr_oas"      : corr_rows[2]["avg_corr"],
    "stocks_with_missing": int(stocks_with_missing),
    "n_spikes_total"    : int(spike_count),
    "stocks_high_zero"  : int(high_zero),
}])
save_csv(preproc_summary, "preproc_summary.csv", index=False)

# =============================================================================
# 8. FIGURES
# =============================================================================
log("\n-- 8. FIGURES -------------------------------------------------------")

plt.rcParams.update({"font.family": "serif", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

def savefig(fname):
    path = os.path.join(FIGURES_DIR, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Saved: {fname}")

# Fig 1: Annualised volatility distribution
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(ann_vol, bins=35, color="#70AD47", edgecolor="white", linewidth=0.4)
ax.axvline(ann_vol.median(), color="#C00000", linewidth=1.2, linestyle="--",
           label=f"Median = {ann_vol.median():.3f}")
ax.set_xlabel("Annualised Volatility")
ax.set_ylabel("Number of stocks")
ax.set_title("Annualised Volatility Distribution â€” Clean Universe (model window)")
ax.legend(frameon=False)
ax.text(0.98, 0.92, "Source: Author's elaboration.",
        transform=ax.transAxes, ha="right", fontsize=8, color="gray")
savefig("vol_distribution.png")

# Fig 2: Annualised mean return distribution
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(ann_ret, bins=35, color="#ED7D31", edgecolor="white", linewidth=0.4)
ax.axvline(ann_ret.median(), color="#C00000", linewidth=1.2, linestyle="--",
           label=f"Median = {ann_ret.median():.3f}")
ax.axvline(0, color="black", linewidth=0.8, linestyle=":")
ax.set_xlabel("Annualised Mean Return")
ax.set_ylabel("Number of stocks")
ax.set_title("Annualised Mean Return Distribution â€” Clean Universe (model window)")
ax.legend(frameon=False)
ax.text(0.98, 0.92, "Source: Author's elaboration.",
        transform=ax.transAxes, ha="right", fontsize=8, color="gray")
savefig("return_distribution.png")

# Fig 3: Off-diagonal correlation distribution (sample estimator)
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(off_diags["Sample"], bins=60, color="#7030A0",
        edgecolor="white", linewidth=0.3, alpha=0.8)
ax.axvline(off_diags["Sample"].mean(), color="#C00000", linewidth=1.2,
           linestyle="--", label=f"Mean = {off_diags['Sample'].mean():.3f}")
ax.set_xlabel("Pairwise Correlation")
ax.set_ylabel("Number of stock pairs")
ax.set_title("Pairwise Correlation Distribution â€” Sample Estimator (model window)")
ax.legend(frameon=False)
ax.text(0.98, 0.92, "Source: Author's elaboration.",
        transform=ax.transAxes, ha="right", fontsize=8, color="gray")
savefig("corr_distribution.png")

# Fig 4: Sector breakdown
if "sector" in meta.columns:
    sector_counts = meta["sector"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4))
    sector_counts.sort_values().plot(kind="barh", ax=ax, color="#4472C4",
                                     edgecolor="white")
    ax.set_xlabel("Number of stocks")
    ax.set_title(f"Sector Breakdown â€” Clean Universe (n={n_stocks})")
    ax.text(0.98, 0.02, "Source: Author's elaboration.",
            transform=ax.transAxes, ha="right", fontsize=8, color="gray")
    savefig("sector_breakdown.png")

# =============================================================================
# 9. SAVE PREPROCESSING LOG
# =============================================================================
log_path = os.path.join(RESULTS_DIR, "preproc_log.txt")
with open(log_path, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
print(f"  Log saved: preproc_log.txt")

# =============================================================================
# 10. FINAL SUMMARY
# =============================================================================
log("\n-- 10. FINAL SUMMARY ------------------------------------------------")
log(f"""
  Dataset             : Clean universe (step1_load.py)
  Final universe      : {n_stocks} stocks
  {"Duplicate share class removed: GOOGL UW EQUITY (kept GOOG UW EQUITY)" if _googl_corr is not None else "Duplicate share class filter: see log above."}
  Return observations : {n_ret_days} trading days per stock (full series)
  Model window        : {n_model_days} trading days ({MODEL_START} to {MODEL_END})
  Sample window       : {SAMPLE_START} to {SAMPLE_END}

  Annualised volatility (model window):
    mean              : {ann_vol.mean():.4f}
    median            : {ann_vol.median():.4f}
    range             : [{ann_vol.min():.4f}, {ann_vol.max():.4f}]

  Annualised mean return (model window):
    mean              : {ann_ret.mean():.4f}
    median            : {ann_ret.median():.4f}
    range             : [{ann_ret.min():.4f}, {ann_ret.max():.4f}]

  Skewness (cross-stock mean)  : {skew_vals.mean():.3f}
  Excess kurtosis (cross-stock mean) : {kurt_vals.mean():.3f}

  Covariance estimators:
    Ledoit-Wolf shrinkage : {lw_shrink:.4f}
    OAS shrinkage         : {oas_shrink:.4f}

  Correlation structure (sample estimator):
    avg off-diagonal      : {corr_rows[0]['avg_corr']:.4f}
    std off-diagonal      : {corr_rows[0]['std_corr']:.4f}
""")

log("=" * 70)
log("  STEP 2 COMPLETE -- Run step3_baseline.py next.")
log("=" * 70)
