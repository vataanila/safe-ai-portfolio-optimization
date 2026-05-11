"""
step1_load_data.py
=============
PURPOSE : Load, inspect, and clean the raw dataset.
          Mirrors the structure of the Bloomberg pipeline:
          load -> inspect -> diagnose duplicates -> clean step by step
          -> final clean objects + before/after summary.

INPUTS  : data/raw/METADATA.xlsx              -- ticker, sector, industry, country, ESG scores
          data/raw/PRICES.xlsx                -- daily closing prices
          data/raw/MKT_CAP.xlsx               -- daily market capitalisation
          data/raw/VOLUME.xlsx                -- daily trading volume
          data/raw/TOT_RETURN_INDEX_GROSS_DVDS.xlsx  -- total return index (gross dividends)

OUTPUTS : data/clean/meta_clean.csv           (standard downstream filenames)
          data/clean/prices_clean.csv
          data/clean/mkt_cap_clean.csv
          data/clean/volume_clean.csv
          data/clean/tri_clean.csv
          data/clean/meta_step1_clean.csv      (Step-1 structural-cleaning snapshot,
          data/clean/prices_step1_clean.csv     read by step2_preprocess.py)
          data/clean/mkt_cap_step1_clean.csv
          data/clean/volume_step1_clean.csv
          data/clean/tri_step1_clean.csv

DATA FORMAT NOTES:
  - All files are .xlsx (Excel)
  - METADATA.xlsx has column names in the first row; last 3 columns are unnamed
    and correspond to ENVIRONMENTAL_SCORE, SOCIAL_SCORE, GOVERNANCE_SCORE
  - PRICES.xlsx, MKT_CAP.xlsx, VOLUME.xlsx, TOT_RETURN_INDEX_GROSS_DVDS.xlsx
    share a common 6-row header: rows 0-1 are date range metadata, row 2 is
    blank, row 3 contains ticker names, rows 4-5 are field labels, data
    begins at row 6 with a "Dates" column in position 0.

MANUAL CORRECTIONS:
  - One stock was found with a missing GICS sector in the raw metadata.
    The sector was identified manually from Bloomberg and corrected in
    METADATA.xlsx prior to any cleaning step. Document this correction
    in the thesis data cleaning chapter.

Author  : Anila Vata 
"""

# =============================================================================
# 0. IMPORTS AND PATHS
# =============================================================================
import os
import numpy as np
import pandas as pd

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR     = os.path.join(BASE_DIR, "data", "raw")
RESULTS_DIR = os.path.join(BASE_DIR, "data", "clean")
os.makedirs(RESULTS_DIR, exist_ok=True)

METADATA_XLSX = os.path.join(RAW_DIR, "METADATA.xlsx")
PRICES_XLSX   = os.path.join(RAW_DIR, "PRICES.xlsx")
MKT_CAP_XLSX  = os.path.join(RAW_DIR, "MKT_CAP.xlsx")
VOLUME_XLSX   = os.path.join(RAW_DIR, "VOLUME.xlsx")
TRI_XLSX      = os.path.join(RAW_DIR, "TOT_RETURN_INDEX_GROSS_DVDS.xlsx")

# ------------------------------------------------------------------
# Analysis window: fixed for the thesis.
# ------------------------------------------------------------------
SAMPLE_START = "2010-01-01"
SAMPLE_END   = "2025-12-31"

# Sub-windows: warm-up period used for feature/momentum computation;
# model window is the live period for returns, covariance, and optimisation.
MODEL_START = "2016-01-01"
MODEL_END   = "2025-12-31"
WARM_UP_END = "2015-12-31"

# Stock-level coverage thresholds.
# EARLY_START_CUTOFF : stock must have its first valid price by this date
#   so that 12-month momentum features are available from MODEL_START.
# LATE_END_CUTOFF    : stock must still have data after this date.
# MAX_MISSING_MODEL_PCT : max fraction of model-window trading days missing;
#   stricter than a full-sample filter because the model window (2016-2025)
#   is what drives return and covariance estimates.
EARLY_START_CUTOFF    = "2015-01-01"  # must have data by this date (12-month lead-in for momentum)
LATE_END_CUTOFF       = "2024-12-31"  # must still have data after this date
MAX_MISSING_MODEL_PCT = 0.05          # drop if >5% of model-window days missing
MAX_FFILL_DAYS        = 5             # forward-fill gaps up to 5 consecutive days

print("=" * 70)
print("  STEP 1 -- DATA LOAD + CLEAN PIPELINE")
print("=" * 70)
print(f"\n  Analysis window : {SAMPLE_START} to {SAMPLE_END}")


# =============================================================================
# HELPER: load a Bloomberg-format Excel file (shared 6-row header)
# =============================================================================
def load_bloomberg_excel(path):
    """
    Load an Excel file that uses the Bloomberg multi-row header convention:
      row 0 : 'Start Date' / value
      row 1 : 'End Date'   / value
      row 2 : blank
      row 3 : ticker names (from column 1 onward)
      row 4 : field description
      row 5 : field code / 'Dates' label in column 0
      row 6+: data

    Returns a DataFrame with a DatetimeIndex and ticker names as columns.
    All values are numeric; missing entries become NaN.
    """
    raw = pd.read_excel(path, header=None)

    tickers = (raw.iloc[3, 1:]
               .astype(str).str.strip().str.upper().tolist())
    tickers = [t for t in tickers if t not in ("NAN", "")]

    data = raw.iloc[6:, :len(tickers) + 1].copy()
    data.columns = ["Date"] + tickers

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"]).set_index("Date").sort_index()
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.dropna(how="all")

    return data


# =============================================================================
# 1. LOAD RAW METADATA (METADATA.xlsx)
# =============================================================================
print("\n-- 1. LOAD RAW METADATA ---------------------------------------------")

meta_raw = pd.read_excel(METADATA_XLSX)
meta_raw.columns = meta_raw.columns.str.strip()

if "Ticker" not in meta_raw.columns:
    raise ValueError("Column 'Ticker' not found in METADATA.xlsx -- check column names.")

# Standardise ticker format: strip whitespace, consistent uppercase
meta_raw["Ticker"] = meta_raw["Ticker"].astype(str).str.strip().str.upper()

# The last three columns are unnamed and correspond to ESG sub-scores;
# rename them so they are preserved with readable names in meta_clean.csv.
unnamed_map = {
    "Unnamed: 7" : "ENVIRONMENTAL_SCORE",
    "Unnamed: 8" : "SOCIAL_SCORE",
    "Unnamed: 9" : "GOVERNANCE_SCORE",
}
meta_raw = meta_raw.rename(columns={k: v for k, v in unnamed_map.items()
                                     if k in meta_raw.columns})

# Rename columns to consistent casing used downstream
meta_raw = meta_raw.rename(columns={
    "sector"        : "Sector",
    "industry_group": "INDUSTRY_GROUP",
    "industry"      : "INDUSTRY",
    "country"       : "COUNTRY",
    "security_name" : "SECURITY_NAME",
    "esg_score"     : "ESG_SCORE",
})

for col in ["ESG_SCORE", "ENVIRONMENTAL_SCORE", "SOCIAL_SCORE", "GOVERNANCE_SCORE"]:
    if col in meta_raw.columns:
        meta_raw[col] = pd.to_numeric(meta_raw[col], errors="coerce")

n_meta_raw = len(meta_raw)
print(f"Loaded: {n_meta_raw} rows x {meta_raw.shape[1]} columns")
print(f"Columns: {meta_raw.columns.tolist()}")

# =============================================================================
# 2. LOAD RAW PRICES (PRICES.xlsx)
# =============================================================================
print("\n-- 2. LOAD RAW PRICES -----------------------------------------------")

# Read start/end dates from the header metadata rows for the diagnostic log
_header = pd.read_excel(PRICES_XLSX, header=None, nrows=2)
reported_start = _header.iloc[0, 1]
reported_end   = _header.iloc[1, 1]
print(f"Reported range in file: {reported_start} to {reported_end}")

prices_raw = load_bloomberg_excel(PRICES_XLSX)
tickers_from_prices = prices_raw.columns.tolist()

n_prices_raw = prices_raw.shape[1]
n_days_raw   = prices_raw.shape[0]
print(f"Reconstructed: {n_days_raw} days x {n_prices_raw} stocks")
print(f"Actual range : {prices_raw.index[0].date()} to {prices_raw.index[-1].date()}")

# =============================================================================
# 3. LOAD SUPPLEMENTARY DATA FILES
# =============================================================================
print("\n-- 3. LOAD SUPPLEMENTARY DATA FILES ---------------------------------")

mkt_cap_raw = load_bloomberg_excel(MKT_CAP_XLSX)
volume_raw  = load_bloomberg_excel(VOLUME_XLSX)
tri_raw     = load_bloomberg_excel(TRI_XLSX)

print(f"  MKT_CAP  : {mkt_cap_raw.shape[0]} days x {mkt_cap_raw.shape[1]} stocks")
print(f"  VOLUME   : {volume_raw.shape[0]} days x {volume_raw.shape[1]} stocks")
print(f"  TRI      : {tri_raw.shape[0]} days x {tri_raw.shape[1]} stocks")

# =============================================================================
# 4. APPLY SAMPLE WINDOW (first operation on prices)
# =============================================================================
# Applied before stock-level cleaning so that all downstream missingness
# calculations and day counts refer to the same fixed period.
print("\n-- 4. APPLY SAMPLE WINDOW -------------------------------------------")

n_days_before_trim = len(prices_raw)
prices_data = prices_raw.loc[SAMPLE_START:SAMPLE_END]
mkt_cap_data = mkt_cap_raw.loc[SAMPLE_START:SAMPLE_END]
volume_data  = volume_raw.loc[SAMPLE_START:SAMPLE_END]
tri_data     = tri_raw.loc[SAMPLE_START:SAMPLE_END]
n_days_after_trim = len(prices_data)

print(f"  Days before trim : {n_days_before_trim}")
print(f"  Days after trim  : {n_days_after_trim}  ({SAMPLE_START} to {SAMPLE_END})")
print(f"  Days removed     : {n_days_before_trim - n_days_after_trim}  "
      f"(outside analysis window)")
print(f"  Stocks in file   : {n_prices_raw}  (unchanged -- trimming is time-only)")

# =============================================================================
# 5. RAW DIAGNOSTICS -- METADATA
# =============================================================================
print("\n-- 5. RAW METADATA DIAGNOSTICS --------------------------------------")

print("\nMissing values per column:")
print(meta_raw.isnull().sum().to_string())

print("\nSector distribution:")
print(meta_raw["Sector"].value_counts().to_string())
print(f"  Missing sector: {meta_raw['Sector'].isna().sum()}")
print(f"  Tickers with missing sector: {meta_raw.loc[meta_raw['Sector'].isna(), 'Ticker'].tolist()}")

print("\nCountry distribution (top 10):")
print(meta_raw["COUNTRY"].value_counts().head(10).to_string())
print(f"  Missing country: {meta_raw['COUNTRY'].isna().sum()}")

# =============================================================================
# 6. RAW DIAGNOSTICS -- PRICES (within sample window)
# =============================================================================
print("\n-- 6. RAW PRICE DIAGNOSTICS (within sample window) -----------------")

valid_vals = prices_data.values[~np.isnan(prices_data.values)]
print(f"  Negative prices: {(valid_vals < 0).sum()} | Zero: {(valid_vals == 0).sum()} | "
      f"Min: {valid_vals.min():.2f} | Max: {valid_vals.max():.2f}")

nan_pct       = prices_data.isna().mean()
nan_pct_model = prices_data.loc[MODEL_START:MODEL_END].isna().mean()
print(f"\n  Missing price structure (full sample window {SAMPLE_START}â€“{SAMPLE_END}):")
print(f"    0% missing    : {(nan_pct == 0).sum()} stocks")
print(f"    0-5% missing  : {((nan_pct > 0) & (nan_pct < 0.05)).sum()} stocks")
print(f"    5-20% missing : {((nan_pct >= 0.05) & (nan_pct <= 0.20)).sum()} stocks")
print(f"    >20% missing  : {(nan_pct > 0.20).sum()} stocks")
print(f"\n  Missing price structure (model window {MODEL_START}â€“{MODEL_END}):")
print(f"    0% missing    : {(nan_pct_model == 0).sum()} stocks")
print(f"    0-5% missing  : {((nan_pct_model > 0) & (nan_pct_model <= 0.05)).sum()} stocks  (at or below removal threshold)")
print(f"    >5% missing   : {(nan_pct_model > 0.05).sum()} stocks  (above removal threshold, filter E)")

# =============================================================================
# 7. DUPLICATE DIAGNOSTICS (raw, before any cleaning)
# =============================================================================
print("\n-- 7. DUPLICATE DIAGNOSTICS (raw) -----------------------------------")

dup_meta = meta_raw[meta_raw["Ticker"].duplicated(keep=False)]["Ticker"].unique()
print(f"  Duplicate tickers in metadata : {len(dup_meta)}"
      + (f"  -> {list(dup_meta)}" if len(dup_meta) > 0 else "  (none)"))

price_col_counts = pd.Series(tickers_from_prices).value_counts()
dup_price = price_col_counts[price_col_counts > 1].index.tolist()
print(f"  Duplicate tickers in prices   : {len(dup_price)}"
      + (f"  -> {dup_price}" if len(dup_price) > 0 else "  (none)"))

# =============================================================================
# 8. TICKER ALIGNMENT CHECK
# =============================================================================
print("\n-- 8. TICKER ALIGNMENT CHECK ----------------------------------------")

meta_tickers  = set(meta_raw["Ticker"])
price_tickers = set(tickers_from_prices)
in_both    = meta_tickers & price_tickers
only_meta  = meta_tickers - price_tickers
only_price = price_tickers - meta_tickers

print(f"  In metadata only : {len(only_meta)}"
      + (f"  -> {sorted(only_meta)}" if only_meta else ""))
print(f"  In prices only   : {len(only_price)}"
      + (f"  -> {sorted(only_price)}" if only_price else ""))
print(f"  In both          : {len(in_both)}"
      + ("  (perfect alignment)" if not only_meta and not only_price else ""))

# =============================================================================
# 9. CLEANING PIPELINE
# =============================================================================
print("\n-- 9. CLEANING PIPELINE ---------------------------------------------")
print(f"\n  Starting universe: {n_meta_raw} stocks\n")

# After each step, re-align meta rows and price columns so that one
# consistent universe count flows through all steps. Both the stock SET
# and stock ORDER are enforced: meta rows are reindexed to follow price
# column order, so the two objects are always in sync for any downstream
# matrix operation.
def realign(meta_in, prices_in, *extra_dfs):
    """Intersect stocks across meta and all price DataFrames; keep price-column order."""
    common = [t for t in prices_in.columns if t in set(meta_in["Ticker"])]
    p = prices_in[common].copy()
    m = (meta_in.set_index("Ticker")
                .reindex(common)
                .reset_index())
    if len(m) != p.shape[1]:
        print(f"  WARNING: meta rows ({len(m)}) != price columns ({p.shape[1]}) "
              f"after realign -- investigate before continuing.")
    extras = []
    for df in extra_dfs:
        shared = [t for t in common if t in df.columns]
        extras.append(df[shared].copy() if shared else df[[c for c in common if c in df.columns]].copy())
    return (m, p, *extras)

removal_log = []  # (label, n_removed, n_remaining)

# -- A: Keep only tickers present in BOTH metadata and prices ----------------
meta   = meta_raw[meta_raw["Ticker"].isin(in_both)].copy()
prices = prices_data[[t for t in prices_data.columns if t in in_both]].copy()
mkt_cap, volume, tri = (
    mkt_cap_data[[t for t in mkt_cap_data.columns if t in in_both]],
    volume_data[[t for t in volume_data.columns if t in in_both]],
    tri_data[[t for t in tri_data.columns if t in in_both]],
)
meta, prices, mkt_cap, volume, tri = realign(meta, prices, mkt_cap, volume, tri)
removed = n_meta_raw - len(meta)
removal_log.append(("A: not in both files", removed, len(meta)))
print(f"  [A] removed {removed:3d}  ->  {len(meta)} stocks remain")

# -- B: Remove duplicate tickers in metadata ---------------------------------
before = len(meta)
meta   = meta.drop_duplicates(subset="Ticker", keep="first")
meta, prices, mkt_cap, volume, tri = realign(meta, prices, mkt_cap, volume, tri)
removed = before - len(meta)
removal_log.append(("B: duplicate tickers (meta)", removed, len(meta)))
print(f"  [B] removed {removed:3d}  ->  {len(meta)} stocks remain")

# -- C: Remove duplicate columns in prices -----------------------------------
before = len(meta)
prices = prices.loc[:, ~prices.columns.duplicated(keep="first")]
meta, prices, mkt_cap, volume, tri = realign(meta, prices, mkt_cap, volume, tri)
removed = before - len(meta)
removal_log.append(("C: duplicate tickers (prices)", removed, len(meta)))
print(f"  [C] removed {removed:3d}  ->  {len(meta)} stocks remain")

# -- D: Handle missing Sector / Country --------------------------------------
# Sector is an operational constraint in the MIQP: aggregate portfolio weight
# per GICS sector is capped at 30% (Kee & Wyatt 2014). With K=10 stocks and
# 11 GICS sectors, a tighter cap would make many rebalancing periods infeasible
# (e.g. with K=10 and a 10% cap, at most one stock per sector would be allowed,
# making many rebalancing periods infeasible).
# Stocks with missing sector are retained but assigned "Unknown", which forms
# its own bucket in the sector-cap constraint. Country is used for descriptive
# reporting only and does not enter the optimiser.
n_miss_sector  = meta["Sector"].isna().sum()
n_miss_country = meta["COUNTRY"].isna().sum()
meta["Sector"]  = meta["Sector"].fillna("Unknown")
meta["COUNTRY"] = meta["COUNTRY"].fillna("Unknown")
removal_log.append(("D: missing sector/country (filled Unknown)", 0, len(meta)))
print(f"  [D] removed   0  ->  {len(meta)} stocks remain  "
      f"(filled {n_miss_sector} sector, {n_miss_country} country with 'Unknown')")

# -- E: Remove stocks with >5% missing prices or TRI in the model window -----
# Missingness is evaluated only in the model window (MODEL_STARTâ€“MODEL_END)
# because that is where returns, covariance, and factor features are computed.
# A stock may have gaps in the warm-up period (2010â€“2015) without penalty,
# as long as its model-window coverage meets the 95% threshold.
# Both prices and TRI must meet the threshold: prices are used for liquidity proxies such as the Amihud ratio; TRI is the
# primary series for log-return computation, covariance estimation, realized
# portfolio returns, and the forward-return target in step2.
before             = len(meta)
prices_model_win   = prices.loc[MODEL_START:MODEL_END]
tri_model_win      = tri.loc[MODEL_START:MODEL_END]
nan_pct_prices     = prices_model_win.isna().mean()
nan_pct_tri        = tri_model_win.isna().mean()
fail_prices        = set(nan_pct_prices[nan_pct_prices >  MAX_MISSING_MODEL_PCT].index)
fail_tri           = set(nan_pct_tri[nan_pct_tri       >  MAX_MISSING_MODEL_PCT].index)
fail_both          = fail_prices & fail_tri
fail_prices_only   = fail_prices - fail_tri
fail_tri_only      = fail_tri - fail_prices
removed_E          = list(fail_prices | fail_tri)
keep               = [t for t in prices.columns if t not in removed_E]
prices             = prices[keep]
meta, prices, mkt_cap, volume, tri = realign(meta, prices, mkt_cap, volume, tri)
removed            = before - len(meta)
removal_log.append(("E: >5% missing prices or TRI in model window (2016-2025)",
                    removed, len(meta)))
print(f"  [E] removed {removed:3d}  ->  {len(meta)} stocks remain"
      f"  (prices only: {len(fail_prices_only)}, "
      f"TRI only: {len(fail_tri_only)}, "
      f"both: {len(fail_both)})")

# -- F: Remove stocks with late start or early end ---------------------------
# Edge-coverage filter: a stock must have its first valid price by
# EARLY_START_CUTOFF (2015-01-01) so that 12-month momentum features are
# computable from MODEL_START (2016-01-01). It must also have valid prices
# after LATE_END_CUTOFF to avoid right-censoring in the model window.
# Both prices and TRI are checked: a stock is removed if either series fails
# the start or end coverage criterion.
before      = len(meta)
first_valid_prices = prices.apply(lambda col: col.first_valid_index())
last_valid_prices  = prices.apply(lambda col: col.last_valid_index())
first_valid_tri    = tri.apply(lambda col: col.first_valid_index())
last_valid_tri     = tri.apply(lambda col: col.last_valid_index())

too_late_prices  = set(first_valid_prices[first_valid_prices > pd.Timestamp(EARLY_START_CUTOFF)].index)
too_early_prices = set(last_valid_prices[last_valid_prices   < pd.Timestamp(LATE_END_CUTOFF)].index)
too_late_tri     = set(first_valid_tri[first_valid_tri       > pd.Timestamp(EARLY_START_CUTOFF)].index)
too_early_tri    = set(last_valid_tri[last_valid_tri         < pd.Timestamp(LATE_END_CUTOFF)].index)

fail_F_prices = too_late_prices | too_early_prices
fail_F_tri    = too_late_tri    | too_early_tri
fail_F_both   = fail_F_prices & fail_F_tri
fail_F_prices_only = fail_F_prices - fail_F_tri
fail_F_tri_only    = fail_F_tri - fail_F_prices

incomplete  = pd.Index(sorted(fail_F_prices | fail_F_tri))
removed_F   = incomplete.tolist()

if len(incomplete) > 0:
    n_too_late  = len(too_late_prices  | too_late_tri)
    n_too_early = len(too_early_prices | too_early_tri)
    print(f"         {n_too_late} stocks have no data before {EARLY_START_CUTOFF} (prices or TRI)")
    print(f"         {n_too_early} stocks have no data after  {LATE_END_CUTOFF} (prices or TRI)")
    prices = prices.drop(columns=[c for c in incomplete if c in prices.columns])
    meta, prices, mkt_cap, volume, tri = realign(meta, prices, mkt_cap, volume, tri)

removed = before - len(meta)
removal_log.append(("F: late start / early end", removed, len(meta)))
print(f"  [F] removed {removed:3d}  ->  {len(meta)} stocks remain"
      f"  (prices only: {len(fail_F_prices_only)}, "
      f"TRI only: {len(fail_F_tri_only)}, "
      f"both: {len(fail_F_both)})")

# -- G: Forward-fill short gaps only (no back-fill) --------------------------
# Carries the last known price forward over short gaps (e.g. non-synchronous
# trading calendars across exchanges). The 5-day limit is calibrated to cover
# at most one standard working week of cross-market holidays; it is not
# intended to fill genuine trading suspensions or halts.
# Forward-filling TRI produces artificial zero log-returns on filled days
# (log(TRI[t] / TRI[t-1]) = 0), which suppresses daily volatility and
# compresses momentum signals. This is acceptable for micro-gaps but would
# systematically bias features if applied to longer interruptions â€” hence the
# 5-day cap. Back-fill is excluded because it uses future index levels to
# impute earlier missing observations, introducing look-ahead bias
# (Lo & MacKinlay 1990).
# Stocks are dropped only if they still have NaN in the MODEL window after
# forward-fill; NaN values confined to the warm-up period (2010-2015) are
# acceptable and do not trigger removal.
before_nan = prices.isna().sum().sum()
prices  = prices.ffill(limit=MAX_FFILL_DAYS)
mkt_cap = mkt_cap.ffill(limit=MAX_FFILL_DAYS)
# Volume is intentionally excluded: ffill repeats yesterday's traded shares,
# distorting illiquidity proxies (e.g. Amihud ratio) that use volume as denominator.
tri     = tri.ffill(limit=MAX_FFILL_DAYS)
after_nan = prices.isna().sum().sum()
print(f"\n  [G: forward-fill only, max {MAX_FFILL_DAYS} days -- prices/mkt_cap/tri only, volume excluded] "
      f"NaN cells in prices (full window): {before_nan} -> {after_nan}")

# Check for remaining NaN only within the model window for each series
def _model_nan_cols(df):
    mw = df.loc[MODEL_START:MODEL_END]
    return set(mw.columns[mw.isna().any()])

still_nan_set = (_model_nan_cols(prices)
                 | _model_nan_cols(mkt_cap)
                 | _model_nan_cols(tri))
still_nan = sorted(still_nan_set)

if still_nan:
    prices  = prices.drop(columns=[c for c in still_nan if c in prices.columns])
    mkt_cap = mkt_cap.drop(columns=[c for c in still_nan if c in mkt_cap.columns])
    volume  = volume.drop(columns=[c for c in still_nan if c in volume.columns])
    tri     = tri.drop(columns=[c for c in still_nan if c in tri.columns])
    meta, prices, mkt_cap, volume, tri = realign(meta, prices, mkt_cap, volume, tri)
    print(f"         Dropped {len(still_nan)} stock(s) with unfillable gaps "
          f"in model window (2016-2025).")
    removal_log.append(("G: unfillable gaps in model window after ffill",
                        len(still_nan), len(meta)))
else:
    print(f"         No unfillable gaps in model window -- all stocks retained.")
    removal_log.append(("G: unfillable gaps in model window after ffill", 0, len(meta)))

n_final      = len(meta)
n_days_final = len(prices)

# -- G (diagnostic): remaining NaN location and gap-length distribution ------
_prices_warmup = prices.loc[SAMPLE_START:WARM_UP_END]
_prices_model  = prices.loc[MODEL_START:MODEL_END]

_nan_warmup = int(_prices_warmup.isna().sum().sum())
_nan_model  = int(_prices_model.isna().sum().sum())
_cells_warmup = _prices_warmup.size
_cells_model  = _prices_model.size

print(f"\n  [G diagnostic] Remaining NaN in prices after ffill:")
print(f"    Warm-up period ({SAMPLE_START}--{WARM_UP_END}): "
      f"{_nan_warmup:>7,d} NaN  "
      f"({100 * _nan_warmup / _cells_warmup:.2f}% of {_cells_warmup:,d} cells)")
print(f"    Model window  ({MODEL_START}--{MODEL_END}): "
      f"{_nan_model:>7,d} NaN  "
      f"({100 * _nan_model / _cells_model:.2f}% of {_cells_model:,d} cells)")

# Gap-length distribution in the warm-up period only.
# Model-window NaN already triggered stock removal above, so only warm-up gaps remain.
_run_counts = {1: 0, "2-5": 0, "6-20": 0, ">20": 0}
for col in _prices_warmup.columns:
    _s = _prices_warmup[col].isna()
    if not _s.any():
        continue
    # identify consecutive NaN runs via groupby on (value, cumulative non-nan count)
    _run_id = (~_s).cumsum()
    for _, _grp in _s.groupby(_run_id):
        if not _grp.iloc[0]:   # group is a run of False (non-NaN)
            continue
        _length = len(_grp)
        if _length == 1:
            _run_counts[1] += 1
        elif _length <= 5:
            _run_counts["2-5"] += 1
        elif _length <= 20:
            _run_counts["6-20"] += 1
        else:
            _run_counts[">20"] += 1

_total_runs = sum(_run_counts.values())
print(f"\n  [G diagnostic] NaN run-length distribution in warm-up period "
      f"({_total_runs} runs total):")
for _label, _count in _run_counts.items():
    _pct = 100 * _count / _total_runs if _total_runs > 0 else 0.0
    print(f"    {str(_label)+' day(s)':<12}: {_count:>6,d} runs  ({_pct:.1f}%)")

del _prices_warmup, _prices_model, _nan_warmup, _nan_model
del _cells_warmup, _cells_model, _run_counts, _total_runs

# =============================================================================
# 10. FINAL CLEAN DATASET SUMMARY
# =============================================================================
print("\n-- 10. FINAL CLEAN DATASET SUMMARY ----------------------------------")

print(f"\n  {'Step':<48} {'Removed':>8} {'Remaining':>10}")
print("  " + "-" * 68)
print(f"  {'Raw metadata':<48} {'':>8} {n_meta_raw:>10}")
for label, removed, remaining in removal_log:
    print(f"  {label:<48} {removed:>8} {remaining:>10}")
print("  " + "-" * 68)
print(f"  {'FINAL CLEAN UNIVERSE':<48} {'':>8} {n_final:>10} stocks")
print(f"  {'FINAL TRADING DAYS':<48} {'':>8} {n_days_final:>10}")
print(f"\n  NOTE: this is a balanced investable panel conditional on data")
print(f"  availability over the model window ({MODEL_START}â€“{MODEL_END}),")
print(f"  not the full historical S&P 500 constituent universe. Stocks")
print(f"  that entered or exited the index within the sample period are")
print(f"  excluded if they fail coverage thresholds, introducing a degree of survivorship/data-availability bias documented in section 12.")

pct_retained = n_final / n_meta_raw * 100
print(f"\n  Cross-section (stocks):")
print(f"    {n_meta_raw} raw  ->  {n_final} clean  "
      f"({n_meta_raw - n_final} removed by stock-level filters, {pct_retained:.1f}% retained)")
print(f"\n  Time dimension (trading days):")
print(f"    {n_days_raw} in raw file")
print(f"    {n_days_after_trim} after sample-window trim  "
      f"({SAMPLE_START} to {SAMPLE_END})")
print(f"    {n_days_final} final  (days with >=1 valid price after stock cleaning)")

print("\n  Sector breakdown (clean universe):")
print(meta["Sector"].value_counts().to_string())

print("\n  Country breakdown (clean universe, top 10):")
print(meta["COUNTRY"].value_counts().head(10).to_string())

# =============================================================================
# 11. SAVE CLEAN OUTPUTS
# =============================================================================
print("\n-- 11. SAVING OUTPUTS -----------------------------------------------")

# Align meta row order to price column order before saving
meta = meta.set_index("Ticker").reindex(prices.columns).reset_index()

meta.to_csv(os.path.join(RESULTS_DIR, "meta_clean.csv"), index=False)
prices.to_csv(os.path.join(RESULTS_DIR, "prices_clean.csv"))
mkt_cap.to_csv(os.path.join(RESULTS_DIR, "mkt_cap_clean.csv"))
volume.to_csv(os.path.join(RESULTS_DIR, "volume_clean.csv"))
tri.to_csv(os.path.join(RESULTS_DIR, "tri_clean.csv"))

print(f"  meta_clean.csv    -- {meta.shape[0]} stocks x {meta.shape[1]} cols")
print(f"  prices_clean.csv  -- {prices.shape[0]} days x {prices.shape[1]} stocks")
print(f"  mkt_cap_clean.csv -- {mkt_cap.shape[0]} days x {mkt_cap.shape[1]} stocks")
print(f"  volume_clean.csv  -- {volume.shape[0]} days x {volume.shape[1]} stocks")
print(f"  tri_clean.csv     -- {tri.shape[0]} days x {tri.shape[1]} stocks")

# Also save Step-1-specific structural-cleaning files.
# These preserve the universe after structural cleaning (filters A-G) before
# Step 2 applies modelling filters (zero-return, hardcoded exclusions, share class).
# Step 2 reads from these paths so that standard downstream *_clean.csv files
# always reflect the final modelling universe without overwriting this snapshot.
meta.to_csv(os.path.join(RESULTS_DIR, "meta_step1_clean.csv"), index=False)
prices.to_csv(os.path.join(RESULTS_DIR, "prices_step1_clean.csv"))
mkt_cap.to_csv(os.path.join(RESULTS_DIR, "mkt_cap_step1_clean.csv"))
volume.to_csv(os.path.join(RESULTS_DIR, "volume_step1_clean.csv"))
tri.to_csv(os.path.join(RESULTS_DIR, "tri_step1_clean.csv"))

print(f"\n  Step-1 structural-cleaning files saved (preserve pre-Step-2 universe):")
print(f"  meta_step1_clean.csv    -- {meta.shape[0]} stocks x {meta.shape[1]} cols")
print(f"  prices_step1_clean.csv  -- {prices.shape[0]} days x {prices.shape[1]} stocks")
print(f"  mkt_cap_step1_clean.csv -- {mkt_cap.shape[0]} days x {mkt_cap.shape[1]} stocks")
print(f"  volume_step1_clean.csv  -- {volume.shape[0]} days x {volume.shape[1]} stocks")
print(f"  tri_step1_clean.csv     -- {tri.shape[0]} days x {tri.shape[1]} stocks")
print(f"  NOTE: *_step1_clean.csv files preserve the structural-cleaning universe "
      f"before Step 2 modelling filters.")

# =============================================================================
# 12. SURVIVORSHIP BIAS DIAGNOSTICS
# =============================================================================
print("\n-- 12. SURVIVORSHIP BIAS DIAGNOSTICS --------------------------------")

# Combine tickers removed by filters E and F; use prices_data (full sample
# window, before any stock-level cleaning) to compute coverage statistics.
# missing_pct reflects the model window (2016-2025) -- the criterion used by
# filter E. First/last valid dates are computed over the full sample window.
all_removed = list(dict.fromkeys(removed_E + removed_F))   # preserve insertion order, no dupes

if len(all_removed) == 0:
    print("  No stocks were removed by filters E or F.")
else:
    rows = []
    for t in all_removed:
        if t not in prices_data.columns:
            continue
        col_full      = prices_data[t]
        col_model     = prices_data.loc[MODEL_START:MODEL_END, t]
        col_tri_full  = tri_data[t] if t in tri_data.columns else None
        col_tri_model = tri_data.loc[MODEL_START:MODEL_END, t] if t in tri_data.columns else None
        fv            = col_full.first_valid_index()
        lv            = col_full.last_valid_index()
        tri_fv        = col_tri_full.first_valid_index() if col_tri_full is not None else None
        tri_lv        = col_tri_full.last_valid_index()  if col_tri_full is not None else None
        source        = ("E" if t in removed_E else "") + ("F" if t in removed_F else "")
        rows.append({
            "ticker"               : t,
            "filter"               : source,
            "first_valid_date"     : fv.date()     if fv     is not None else None,
            "last_valid_date"      : lv.date()     if lv     is not None else None,
            "tri_first_valid"      : tri_fv.date() if tri_fv is not None else None,
            "tri_last_valid"       : tri_lv.date() if tri_lv is not None else None,
            "missing_pct_model"    : round(col_model.isna().mean() * 100, 2),
            "missing_pct_tri_model": round(col_tri_model.isna().mean() * 100, 2)
                                     if col_tri_model is not None else float("nan"),
        })

    removed_df = pd.DataFrame(rows).sort_values("first_valid_date", na_position="last")

    print(f"\n  {'Ticker':<14} {'Filt':<5} {'Px first':<12} {'Px last':<12} "
          f"{'TRI first':<12} {'TRI last':<12} {'Miss%(px)':>10} {'Miss%(TRI)':>11}")
    print("  " + "-" * 100)
    for _, r in removed_df.iterrows():
        print(f"  {r['ticker']:<14} {r['filter']:<5} "
              f"{str(r['first_valid_date']):<12} "
              f"{str(r['last_valid_date']):<12} "
              f"{str(r['tri_first_valid']):<12} "
              f"{str(r['tri_last_valid']):<12} "
              f"{r['missing_pct_model']:>9.2f}% "
              f"{r['missing_pct_tri_model']:>10.2f}%")

    # -- Summary by first-valid-date thresholds --
    fv_dates = removed_df["first_valid_date"].dropna()
    n_total  = len(removed_df)
    print(f"\n  First-valid-date distribution among removed stocks ({n_total} total):")
    for cutoff in ["2013-01-01", "2015-01-01", "2018-01-01"]:
        n_late = (fv_dates > pd.Timestamp(cutoff).date()).sum()
        print(f"    first valid date > {cutoff} : {n_late} stocks")

    # -- Conclusion line --
    late_entry_cutoff = "2013-01-01"
    n_late_entry = (fv_dates > pd.Timestamp(late_entry_cutoff).date()).sum()
    print(f"\n  Stocks removed due to late entry into index "
          f"(first valid date > {late_entry_cutoff}): "
          f"{n_late_entry} out of {n_total} total removed")

# =============================================================================
# 13. FINAL CLEAN OBJECTS (passed to step2_preprocess.py)
# =============================================================================
meta_clean   = meta.copy()     # DataFrame: n_final rows, Ticker as first column
prices_clean = prices.copy()   # DataFrame: n_days_final rows (date index), n_final cols
mkt_cap_clean = mkt_cap.copy()
volume_clean  = volume.copy()
tri_clean     = tri.copy()

print(f"""
-- FINAL CLEAN OBJECTS ---------------------------------------------------
  meta_clean    : {meta_clean.shape[0]} stocks x {meta_clean.shape[1]} columns
  prices_clean  : {prices_clean.shape[0]} trading days x {prices_clean.shape[1]} stocks
  mkt_cap_clean : {mkt_cap_clean.shape[0]} trading days x {mkt_cap_clean.shape[1]} stocks
  volume_clean  : {volume_clean.shape[0]} trading days x {volume_clean.shape[1]} stocks
  tri_clean     : {tri_clean.shape[0]} trading days x {tri_clean.shape[1]} stocks
  Date range    : {prices_clean.index[0].date()} to {prices_clean.index[-1].date()}
-------------------------------------------------------------------------
""")

print("=" * 70)
print("  STEP 1 COMPLETE -- Run step2_preprocess.py next.")
print("=" * 70)
