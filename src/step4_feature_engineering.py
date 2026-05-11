"""
step4_feature_engineering.py
=================
PURPOSE : Build the leakage-free monthly ML panel with cross-sectionally
          ranked features and holding-period targets.

INPUTS  :
  data/clean/returns.csv
  data/clean/prices_clean.csv
  data/clean/volume_clean.csv
  data/clean/mkt_cap_clean.csv     (optional)
  data/results/step3/baseline_weights.csv

OUTPUTS :
  data/results/step4/ml_panel.csv
  data/results/step4/feature_diagnostics.csv

Author  : Anila Vata
"""

# =============================================================================
# 0. IMPORTS AND CONSTANTS
# =============================================================================
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

TRADING_DAYS = 252
ESTIM_WINDOW = 252

MODEL_START = "2016-01-01"   # first date included in ml_panel.csv
MODEL_END   = "2025-12-31"   # last  date included in ml_panel.csv

# "log_mktcap" is added to the local feature list in build_panel() if mkt_cap is available.
BASE_FEATURE_COLS = [
    "ret_1w", "ret_1m", "ret_3m", "ret_6m", "ret_12m",
    "vol_1m", "vol_3m", "vol_ratio", "amihud",
]

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DIR   = os.path.join(BASE_DIR, "data", "clean")
RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")
STEP3_DIR   = os.path.join(RESULTS_DIR, "step3")
STEP4_DIR   = os.path.join(RESULTS_DIR, "step4")


# =============================================================================
# 1. LOAD DATA
# =============================================================================

def load_data():
    """Load returns, prices, volume, optional market cap, and baseline_weights."""
    print("\n[1] Loading input data ...")

    returns = pd.read_csv(
        os.path.join(CLEAN_DIR, "returns.csv"),
        index_col=0, parse_dates=True,
    ).sort_index()

    prices = pd.read_csv(
        os.path.join(CLEAN_DIR, "prices_clean.csv"),
        index_col=0, parse_dates=True,
    ).sort_index()

    volume = pd.read_csv(
        os.path.join(CLEAN_DIR, "volume_clean.csv"),
        index_col=0, parse_dates=True,
    ).sort_index()

    baseline_weights = pd.read_csv(
        os.path.join(STEP3_DIR, "baseline_weights.csv"),
        index_col=0, parse_dates=True,
    )

    # Ticker order is defined by baseline_weights.csv â€” all other files must conform.
    baseline_tickers = baseline_weights.columns.astype(str).tolist()

    for file_name, df in [("returns", returns), ("prices", prices), ("volume", volume)]:
        missing = [t for t in baseline_tickers if t not in df.columns]
        if missing:
            raise ValueError(
                f"{file_name}.csv is missing {len(missing)} tickers required by "
                f"baseline_weights.csv: {missing[:5]}"
                f"{'...' if len(missing) > 5 else ''}"
            )

    returns = returns[baseline_tickers]
    prices  = prices[baseline_tickers]
    volume  = volume[baseline_tickers]

    print(f"  returns  shape : {returns.shape}")
    print(f"           range : {returns.index[0].date()} to {returns.index[-1].date()}")
    print(f"  prices   shape : {prices.shape}")
    print(f"           range : {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"  volume   shape : {volume.shape}")
    print(f"           range : {volume.index[0].date()} to {volume.index[-1].date()}")
    print(f"  Ticker universe: {len(baseline_tickers)} (order from baseline_weights.csv)")
    print(f"  First 3        : {baseline_tickers[:3]}")
    print(f"  Last 3         : {baseline_tickers[-3:]}")

    # Market cap â€” optional; adds log_mktcap feature if the file exists.
    mkt_cap_path = os.path.join(CLEAN_DIR, "mkt_cap_clean.csv")
    if os.path.exists(mkt_cap_path):
        mkt_cap = pd.read_csv(mkt_cap_path, index_col=0, parse_dates=True).sort_index()
        missing_mc = [t for t in baseline_tickers if t not in mkt_cap.columns]
        if missing_mc:
            print(f"  WARNING: mkt_cap_clean.csv missing {len(missing_mc)} tickers; "
                  f"log_mktcap will be NaN for those stocks.")
        mkt_cap = mkt_cap.reindex(columns=baseline_tickers)
        print(f"  mkt_cap  shape : {mkt_cap.shape}")
        print(f"           range : {mkt_cap.index[0].date()} to {mkt_cap.index[-1].date()}")
    else:
        print("  WARNING: mkt_cap_clean.csv not found â€” log_mktcap feature will not be used.")
        mkt_cap = None

    return returns, prices, volume, mkt_cap, baseline_tickers, baseline_weights


# =============================================================================
# 2. PANEL DATES â€” ALL MONTHLY FIRST TRADING DAYS
# =============================================================================

def get_panel_dates(returns_index: pd.DatetimeIndex,
                    min_history: int = ESTIM_WINDOW) -> pd.DatetimeIndex:
    """
    Return all first trading days of each calendar month in the full
    returns history, keeping only those with at least min_history days
    of prior data (needed for feature computation).
    """
    s             = pd.Series(returns_index, index=returns_index)
    monthly_first = s.groupby(returns_index.to_period("M")).first()
    all_dates     = pd.DatetimeIndex(monthly_first.values)
    valid = [d for d in all_dates if (returns_index < d).sum() >= min_history]
    return pd.DatetimeIndex(valid)


# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================

def cs_rank(col: pd.Series) -> pd.Series:
    """Cross-sectional rank normalized to [0, 1] (average method for ties)."""
    return col.rank(method="average", pct=True)


def compute_features_at(t: pd.Timestamp,
                        returns: pd.DataFrame,
                        prices:  pd.DataFrame,
                        volume:  pd.DataFrame,
                        mkt_cap: pd.DataFrame | None = None) -> pd.DataFrame | None:
    """
    Compute raw features for all stocks using only data strictly before t.
    Returns DataFrame of shape (n_tickers, n_features) or None if insufficient history.
    """
    ret_hist = returns.loc[returns.index < t]
    if len(ret_hist) < ESTIM_WINDOW:
        return None

    # Momentum: cumulative log-returns.
    # ret_1w (5-day) also captures short-term reversal; the model learns the sign.
    ret_1w  = ret_hist.iloc[-5:].sum()
    ret_1m  = ret_hist.iloc[-21:].sum()
    ret_3m  = ret_hist.iloc[-63:].sum()
    ret_6m  = ret_hist.iloc[-126:].sum()
    ret_12m = ret_hist.iloc[-252:].sum()

    # Volatility: annualized realized vol.
    vol_1m = ret_hist.iloc[-21:].std(ddof=1) * np.sqrt(TRADING_DAYS)
    vol_3m = ret_hist.iloc[-63:].std(ddof=1) * np.sqrt(TRADING_DAYS)

    # Volatility ratio.
    vol_ratio = vol_1m / vol_3m.replace(0.0, np.nan)

    # Amihud illiquidity over last 21 days.
    # dollar_volume = price Ã— volume on each trading day.
    px_aligned  = prices.loc[prices.index < t].reindex(ret_hist.index)
    vol_aligned = volume.loc[volume.index < t].reindex(ret_hist.index)
    dollar_vol  = (px_aligned * vol_aligned).replace(0.0, np.nan)
    illiq       = ret_hist.abs() / dollar_vol
    amihud      = illiq.iloc[-21:].mean()

    feat = pd.DataFrame({
        "ret_1w"   : ret_1w,
        "ret_1m"   : ret_1m,
        "ret_3m"   : ret_3m,
        "ret_6m"   : ret_6m,
        "ret_12m"  : ret_12m,
        "vol_1m"   : vol_1m,
        "vol_3m"   : vol_3m,
        "vol_ratio": vol_ratio,
        "amihud"   : amihud,
    })

    # Market cap: log of the most recent daily value strictly before t.
    if mkt_cap is not None:
        mc_hist = mkt_cap.loc[mkt_cap.index < t]
        log_mktcap = (
            np.log(mc_hist.iloc[-1].replace(0.0, np.nan))
            if not mc_hist.empty
            else pd.Series(np.nan, index=returns.columns)
        )
        feat["log_mktcap"] = log_mktcap

    # Replace infinite values before ranking/dropping.
    feat = feat.replace([np.inf, -np.inf], np.nan)
    return feat


# =============================================================================
# 4. BUILD PANEL
# =============================================================================

def build_panel(returns: pd.DataFrame,
                prices:  pd.DataFrame,
                volume:  pd.DataFrame,
                panel_dates: pd.DatetimeIndex,
                mkt_cap: pd.DataFrame | None = None):
    """
    Build long-format panel: (date, ticker) Ã— (features + targets).

    For each panel date t:
      - features are computed using data strictly before t
      - features are cross-sectionally ranked in [0, 1]
      - target_raw  = cumulative log-return over the holding period (t, t_next):
                      the return on date t is excluded, matching Step 3 where
                      portfolio returns start strictly after the rebalancing date
      - target_rank = cross-sectional rank of target_raw in [0, 1]
      - target_end_date = t_next  (used to prevent leakage in OOS splits)

    Returns (panel, feature_cols, n_rows_before_drop, n_rows_after_drop).
    """
    feature_cols = list(BASE_FEATURE_COLS)
    if mkt_cap is not None:
        feature_cols = feature_cols + ["log_mktcap"]

    print(f"\n[2] Building panel over {len(panel_dates)} monthly dates ...")
    print(f"  Features ({len(feature_cols)}): {feature_cols}")
    rows = []

    for i, t in enumerate(panel_dates):
        t_next = panel_dates[i + 1] if i + 1 < len(panel_dates) else None

        # Features (strictly before t).
        feat = compute_features_at(t, returns, prices, volume, mkt_cap)
        if feat is None:
            continue

        # Cross-sectional rank in [0, 1].
        feat_ranked = feat.apply(cs_rank, axis=0)

        # Target.
        # Holding period is (t, t_next): strictly after rebalancing date t and
        # strictly before the next rebalancing date. This matches Step 3, where
        # returns_df.index > rebal_date (the rebalancing-date return is excluded
        # because it is not investable once weights are set at the close of t).
        if t_next is not None:
            mask       = (returns.index > t) & (returns.index < t_next)
            period_ret = returns.loc[mask]
            target_raw = (period_ret.sum() if not period_ret.empty
                          else pd.Series(np.nan, index=returns.columns))
            target_end_date = t_next
        else:
            target_raw      = pd.Series(np.nan, index=returns.columns)
            target_end_date = pd.NaT

        target_rank = cs_rank(target_raw)

        # Stack into long format.
        frame = feat_ranked.copy()
        frame["target_raw"]      = target_raw
        frame["target_rank"]     = target_rank
        frame["date"]            = t
        frame["target_end_date"] = target_end_date
        frame.index.name = "ticker"
        frame = frame.reset_index()
        rows.append(frame)

        if (i + 1) % 24 == 0 or (i + 1) == len(panel_dates):
            print(f"  ... {i+1}/{len(panel_dates)} dates  ({t.date()})")

    panel = pd.concat(rows, ignore_index=True)

    # Column order as specified.
    cols  = ["date", "ticker", "target_end_date", "target_raw", "target_rank"] + feature_cols
    panel = panel[cols]

    # Replace any infinite values before counting/dropping NaN rows.
    panel = panel.replace([np.inf, -np.inf], np.nan)

    n_before = len(panel)

    panel = panel.dropna(subset=feature_cols, how="all")
    panel = panel.dropna(subset=feature_cols, how="any")

    # Restrict to the model window; 2010â€“2015 rows served as warm-up only.
    panel = panel[
        (panel["date"] >= MODEL_START) &
        (panel["date"] <= MODEL_END)
    ].copy()

    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)

    n_after = len(panel)

    print(f"\n  Panel shape    : {panel.shape}")
    print(f"  Date range     : {panel['date'].min().date()} to {panel['date'].max().date()}")
    print(f"  Unique dates   : {panel['date'].nunique()}")
    print(f"  Unique tickers : {panel['ticker'].nunique()}")
    print(f"  Rows dropped   : {n_before - n_after}")
    print(f"  Missing values:")
    for col in ["target_raw", "target_rank"] + feature_cols:
        pct = panel[col].isna().mean() * 100
        print(f"    {col:<14} : {pct:.1f}%")

    return panel, feature_cols, n_before, n_after


# =============================================================================
# 5. FEATURE DIAGNOSTICS
# =============================================================================

def save_feature_diagnostics(panel: pd.DataFrame,
                              feature_cols: list,
                              n_before: int,
                              n_after: int,
                              mkt_cap) -> None:
    """Save feature_diagnostics.csv to STEP4_DIR."""
    n_dropped = n_before - n_after

    diag = {
        "n_rows_before_drop"     : n_before,
        "n_rows_after_drop"      : n_after,
        "n_rows_dropped"         : n_dropped,
        "n_unique_dates"         : panel["date"].nunique(),
        "n_unique_tickers"       : panel["ticker"].nunique(),
        "min_date"               : panel["date"].min().date().isoformat(),
        "max_date"               : panel["date"].max().date().isoformat(),
        "pct_missing_target_rank": round(panel["target_rank"].isna().mean() * 100, 4),
        "market_cap_included"    : mkt_cap is not None,
        "final_feature_list"     : "|".join(feature_cols),
    }

    for col in ["target_raw", "target_rank"] + feature_cols:
        if col in panel.columns:
            diag[f"pct_missing_{col}"] = round(panel[col].isna().mean() * 100, 4)

    diag_df = pd.DataFrame(list(diag.items()), columns=["metric", "value"])
    path = os.path.join(STEP4_DIR, "feature_diagnostics.csv")
    diag_df.to_csv(path, index=False)
    print(f"  Saved: feature_diagnostics.csv  ({len(diag_df)} rows)")


# =============================================================================
# 6. MAIN
# =============================================================================

def main():
    os.makedirs(STEP4_DIR, exist_ok=True)

    print("=" * 70)
    print("  STEP 4 â€” ML PANEL FEATURE ENGINEERING")
    print("=" * 70)

    # 1. Load data.
    returns, prices, volume, mkt_cap, tickers, baseline_weights = load_data()

    # 2. All monthly first trading days across the full returns history.
    #    Pre-2016 dates are used only to compute rolling features; the
    #    final panel is filtered to [MODEL_START, MODEL_END] in build_panel().
    panel_dates = get_panel_dates(returns.index, min_history=ESTIM_WINDOW)
    print(f"\n  Monthly panel dates: {len(panel_dates)}")
    print(f"    {panel_dates[0].date()} to {panel_dates[-1].date()}")
    print(f"  Model window       : {MODEL_START} to {MODEL_END}")

    # 3. Build panel.
    panel, feature_cols, n_before, n_after = build_panel(
        returns, prices, volume, panel_dates, mkt_cap=mkt_cap
    )

    # 4. Quality checks.
    if panel.empty:
        raise ValueError("ml_panel.csv would be empty â€” check feature computation.")

    for col in feature_cols:
        if col not in panel.columns:
            raise ValueError(f"Feature column '{col}' missing from panel.")

    inf_count = np.isinf(panel[feature_cols].values).sum()
    if inf_count > 0:
        raise ValueError(f"Panel contains {inf_count} infinite values after cleaning.")

    panel_tickers = set(panel["ticker"].unique())
    missing_order = [t for t in tickers if t not in panel_tickers]
    if missing_order:
        raise ValueError(
            f"{len(missing_order)} tickers from baseline_weights have no panel rows: "
            f"{missing_order[:5]}{'...' if len(missing_order) > 5 else ''}"
        )

    # 5. Save panel.
    panel_path = os.path.join(STEP4_DIR, "ml_panel.csv")
    panel.to_csv(panel_path, index=False)
    print(f"\n  Saved: ml_panel.csv  {panel.shape}")

    # 6. Save diagnostics.
    save_feature_diagnostics(panel, feature_cols, n_before, n_after, mkt_cap)

    print("\n" + "=" * 70)
    print("  STEP 4 â€” COMPLETE")
    print("=" * 70)
    print(f"""
  Outputs saved to: {STEP4_DIR}

  Panel:
    ml_panel.csv                ({panel.shape[0]:,} rows Ã— {panel.shape[1]} cols)
    feature_diagnostics.csv

  Features used: {feature_cols}
  Date range   : {panel['date'].min().date()} to {panel['date'].max().date()}

  Next step: run step5_ml.py to train models and produce mu_hat.
""")


if __name__ == "__main__":
    main()
