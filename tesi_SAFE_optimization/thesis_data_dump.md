# Thesis Data Dump — Numerical Results by Step
*All values read directly from pipeline output files. No values inferred or assumed.*

---

## Step 1 — Data Loading & Structural Cleaning

**Raw inputs (Bloomberg):**
- PRICES.xlsx, TOT_RETURN_INDEX_GROSS_DVDS.xlsx, MKT_CAP.xlsx, VOLUME.xlsx, METADATA.xlsx
- Period: 2010-01-01 to 2025-12-31
- Initial universe: 426 stocks

**Step 1 output (step1_clean files):**
- 426 stocks × 4,174 trading days
- Warm-up period: 2010–2015; model window: 2016-01-01 to 2025-12-31
- One stock had missing GICS sector; corrected manually in METADATA.xlsx before any cleaning

---

## Step 2 — Preprocessing, Returns, Covariance

**Universe reduction (applied in step2_preprocess.py):**
- Input: 426 stocks × 4,174 days
- Zero-return filter (>5% zeros in model window): 17 stocks removed
  - Worst offenders: UAL 29.7%, FISV 27.6%, SMCI 18.0%, IBKR 13.8%
  - After filter: 409 stocks
- Hardcoded exclusion: SATS UW EQUITY (suspected TRI feed error: +53.21% on 2025-08-26 followed by +14.42%)
  - After exclusion: 408 stocks
- Duplicate share class removal: GOOGL UW EQUITY (corr = 0.9952 with GOOG UW EQUITY)
  - **Final universe: 407 stocks**

**Return matrix:**
- Shape: 4,173 days × 407 stocks
- NaN in warm-up (2010-2015): 28,412 cells (expected — late IPO entries)
- NaN in model window (2016-2025): 0 cells
- Model window: 2,609 days

**Cross-sectional return statistics (model window 2016–2025):**
- Annualised vol: mean 30.72%, median 29.07%, min 17.80%, max 58.05%
- Annualised mean return: mean 11.68%, median 11.44%, min −5.21%, max 52.61%
- Skewness (cross-stock mean): −0.668, std 1.045, range [−14.53, 1.57]
- Excess kurtosis (cross-stock mean): 17.877 (heavy tails)
- Return spikes (|r| > 50%, model window): 7 observations across 7 stocks (GL, TRGP, OXY, EPAM, FANG, DXCM, CNC) — retained, diagnostic only

**Covariance matrices (full model-window, diagnostic; rolling re-estimation used in optimizer):**
- Sample: min eigenvalue 0.002660, condition number 5,560
- Ledoit-Wolf (identity target): shrinkage coeff = 0.0159, min eigenvalue 0.004202, condition number 3,460
- OAS: shrinkage coeff = 0.0032, min eigenvalue 0.002967, condition number 4,970

**Correlation structure (model window):**
- Sample: avg off-diagonal 0.3530, std 0.1218, range [−0.073, 0.919], % pairs > 0.50: 11.3%
- Ledoit-Wolf: avg 0.3461, std 0.1195, range [−0.072, 0.896], % pairs > 0.50: 9.8%
- High-correlation pairs (> 0.95): **0 pairs**

**Sector breakdown (407 stocks, 11 GICS sectors):**
- Financials: 65; Industrials: 65; IT: 56; Health Care: 54; Consumer Discretionary: 41
- Consumer Staples: 28; Real Estate: 26; Materials: 21; Utilities: 20; Energy: 18; Communication Services: 13

**Country breakdown:** US 392, Ireland 8, Bermuda 2, Britain 2, others 5

**Figures produced:** vol_distribution.png, return_distribution.png, corr_distribution.png, sector_breakdown.png

---

## Step 3 — Baseline MIQP Portfolio (Markowitz + trailing mean μ)

**MIQP formulation:**
- Objective: minimize w′Σw − λ·μ′w, λ = 1.0
- μ: trailing 252-day annualised mean, winsorized p1/p99
- Σ: rolling Ledoit-Wolf, re-estimated at each rebalance using trailing 252 returns strictly before rebalance date
- K = 10 (cardinality), w ∈ [1%, 20%] per stock, sector cap 30%
- Solver: Gurobi 13.0.1, TimeLimit = 60 s, MIPGap = 0.01
- Rebalancing: first trading day of each calendar month, 2023-01-02 to 2025-12-01
- Test window: 2023-01-01 to 2025-12-31 → **36 rebalancing dates, 747 daily observations**
- Avg Gurobi solve time: 0.3 s; all 36 periods solved to OPTIMAL

**Out-of-sample performance (gross):**
- Annualised return: **20.93%**
- Annualised vol: **25.39%**
- Sharpe ratio: **0.8243**
- Sortino ratio: **1.1062**
- Calmar ratio: **0.8156**
- Maximum drawdown: **−25.66%** (peak 2024-12-05 wealth = 1.9054; trough 2025-04-08 wealth = 1.4164)
- Average monthly turnover: **46.46%** (min 21.63%, max 78.21%)
- Final wealth index: **1.8597** (85.97% simple return)
- Cumulative log-return: 0.6204

**Per-year breakdown:**
- 2023: ret 23.79%, vol 20.77%, Sharpe 1.1452, MaxDD −10.32%
- 2024: ret 29.00%, vol 25.46%, Sharpe 1.1389, MaxDD −15.59%
- 2025: ret  9.98%, vol 29.29%, Sharpe  0.3408, MaxDD −24.74%

**Average sector weights (test period):**
- IT 21.41%, Industrials 19.97%, Consumer Discretionary 18.95%, Communication Services 11.39%
- Health Care 7.22%, Energy 7.39%, Utilities 4.98%, Financials 3.22%, Consumer Staples 2.28%
- Materials 2.12%, Real Estate 1.08%

**Net-of-cost sensitivity (baseline):**
- 10 bps: Sharpe 0.8025, Ann Ret 20.38%
- 20 bps: Sharpe 0.7807, Ann Ret 19.83%
- 30 bps: Sharpe 0.7588, Ann Ret 19.28%

**Equal-weight benchmark (407 stocks, monthly rebalanced, same dates):**
- Ann ret 15.80%, vol 14.14%, Sharpe 1.1170, Sortino 1.5392, Calmar 0.9780, MaxDD −16.15%

**Figures produced:** baseline_cumulative_returns.png, baseline_drawdown.png, baseline_rolling_sharpe.png, baseline_sector_heatmap.png, baseline_turnover.png, efficient_frontier.png

---

## Step 4 — Feature Engineering & ML Panel

**Panel construction:**
- Universe: 407 tickers × rebalancing months 2016-01-01 to 2025-12-01 (120 dates)
- Rows before drop: 73,260; after dropping NaN targets: **48,840 rows**
- Missing target_rank: 0.8333% (expected — first months lack a full 21-day forward window)

**10 features (cross-sectionally ranked within each date):**
| Feature | Economic meaning |
|---|---|
| ret_1w | 1-week return (short-term reversal) |
| ret_1m | 1-month return (short-term reversal) |
| ret_3m | 3-month momentum |
| ret_6m | 6-month momentum |
| ret_12m | 12-month momentum |
| vol_1m | 1-month realised volatility |
| vol_3m | 3-month realised volatility |
| vol_ratio | vol_1m / vol_3m (vol regime change) |
| amihud | Amihud illiquidity ratio |
| log_mktcap | Log market capitalisation (size) |

**Target:** 21-trading-day forward return, converted to cross-sectional rank (percentile)

---

## Step 5A — Ridge Expected-Return Prediction

**IC diagnostics (35 rebalancing dates, 2023–2025):**
- Overall mean IC (Spearman): **−0.0217**
- Hit rate (% dates with positive IC): **37.14%**
- Mean Pearson IC: **−0.021**
- Mean R²_rank: **−0.0056**
- By year:
  - 2023: IC = +0.0214, hit rate 58.33%
  - 2024: IC = −0.0471, hit rate 25.00%
  - 2025: IC = −0.0411, hit rate 27.27%

**Sustainability metrics:**
- avg_rank_spearman_consecutive: 0.7335 (most stable across three models)
- avg_top10_overlap: 0.306
- avg_noise_rank_corr: 0.9986
- avg_winsor_rank_corr: 1.0000

---

## Step 5B — XGBoost Expected-Return Prediction

**Training protocol:**
- Expanding-window OOS: for each of 36 rebalance dates, train on all data strictly before that date
- Validation set: last 6 months of available training data
- Grid search over n_estimators ∈ {100,300,500}, max_depth ∈ {3,4,6}, lr ∈ {0.01,0.05}; early stopping 20 rounds
- Final model refitted on train+val with best params
- All 36 monthly models serialised as JSON

**Typical hyperparameters (sample across 36 dates):**
- n_estimators: range 1–69 (very shallow models dominate, especially late in sample)
- max_depth: mostly 3 or 6
- learning_rate: mostly 0.05

**μ scaling:** XGBoost raw score z-standardised and rescaled to match the cross-sectional mean/std of the baseline trailing-mean μ at each date, then winsorized p1/p99

**IC diagnostics (35 rebalancing dates, 2023–2025):**
- Overall mean IC (Spearman): **+0.0049**
- Hit rate: **48.57%**
- Mean Pearson IC: **+0.012**
- Mean R²_rank: **−0.0006**
- By year:
  - 2023: IC = +0.0334, hit rate 50.00%
  - 2024: IC = −0.0276, hit rate 41.67%
  - 2025: IC = +0.0094, hit rate 54.55%

**Sustainability metrics:**
- avg_rank_spearman_consecutive: 0.4643
- avg_top10_overlap: 0.320
- avg_noise_rank_corr: 0.9885
- avg_winsor_rank_corr: 1.0000

---

## Step 5C — MLP Expected-Return Prediction

**IC diagnostics (35 rebalancing dates, 2023–2025):**
- Overall mean IC (Spearman): **−0.0008**
- Hit rate: **45.71%**
- Mean Pearson IC: **+0.0009**
- Mean R²_rank: **−0.0214**
- By year:
  - 2023: IC = +0.0259, hit rate 50.00%
  - 2024: IC = −0.0383, hit rate 33.33%
  - 2025: IC = +0.0111, hit rate 54.55%

**Sustainability metrics:**
- avg_rank_spearman_consecutive: 0.1650 (most unstable)
- avg_top10_overlap: 0.086 (very low — high stock replacement each month)
- avg_noise_rank_corr: 0.9984
- avg_winsor_rank_corr: 1.0000

---

## Step 6 — ML Portfolio Construction (same MIQP, μ only changes)

**Design:** all three ML portfolios use identical MIQP formulation to Step 3 (K=10, bounds, sector cap, rolling LW Σ); only μ differs.

**Portfolio performance summary (2023–2025, gross):**

| Portfolio | Ann Ret | Ann Vol | Sharpe | Sortino | Calmar | MaxDD | AvgTurnover | Final Wealth | Overall Rank |
|---|---|---|---|---|---|---|---|---|---|
| XGBoost   | 21.60%  | 23.98%  | 0.9007 | 1.3764  | 0.9225 | −23.41% | 78.15% | 1.8969 | 1 |
| Baseline  | 20.93%  | 25.39%  | 0.8243 | 1.1062  | 0.8156 | −25.66% | 46.46% | 1.8597 | 2 |
| Ridge     | 13.39%  | 21.76%  | 0.6151 | 0.7909  | 0.5927 | −22.58% | 79.86% | 1.4870 | 3 |
| MLP       |  5.45%  | 18.04%  | 0.3021 | 0.4004  | 0.2131 | −25.57% | 93.15% | 1.1753 | 4 |

**Net-of-cost at 30 bps:**
- XGBoost: Sharpe 0.7846, Ann Ret 18.83%
- Baseline: Sharpe 0.7588, Ann Ret 19.28%
- Ridge: Sharpe 0.4846, Ann Ret 10.56%
- MLP: Sharpe 0.1190, Ann Ret  2.15%

**Individual criterion winners:**
- Best Sharpe: XGBoost (0.9007)
- Best Sortino: XGBoost (1.3764)
- Best Calmar: XGBoost (0.9225)
- Best Final Wealth: XGBoost (1.8969)
- Smallest MaxDD: Ridge (22.58%)
- Lowest Turnover: Baseline (46.46%)

**Figures produced:** portfolio_cumulative_wealth_comparison.png, portfolio_drawdown_comparison.png, portfolio_rolling_sharpe_comparison.png, portfolio_sharpe_turnover_scatter.png, portfolio_average_sector_allocation_comparison.png, portfolio_average_turnover_barplot.png, portfolio_turnover_comparison.png, baseline_sector_heatmap.png, xgboost_sector_heatmap.png

---

## Step 7A — Model-Level SAFE: Sustainability

*Object of evaluation: the ML return-forecasting models (Ridge, XGBoost, MLP). Not the portfolios.*

**Metrics computed:** rolling rank stability (consecutive Spearman IC), top-K overlap, μ dispersion (p99–p01 range), winsorisation robustness, noise robustness (5% Gaussian perturbation to features)

| Model | RGR (Reliability) | **Sust. Rank** |
|---|---|---|
| Ridge   | 0.7234 | **1** |
| XGBoost | 0.6077 | **2** |
| MLP     | 0.5864 | **3** |

*Note: Ridge ranks best on reliability (RGR = 0.7234); XGBoost is second (0.6077); MLP ranks third (0.5864). Consecutive Spearman and top-K overlap detail is reported in Steps 5A–5C.*

**Figures:** model_sustainability_rank_stability.png, model_sustainability_noise_robustness.png, model_sustainability_topk_overlap.png, model_sustainability_mu_dispersion.png, model_sustainability_winsorization_robustness.png, model_sustainability_rank_summary.png

---

## Step 7B — Model-Level SAFE: Accuracy

*Metric: RGA (Rank Graduation Accuracy — how well predicted ranks match realised return ranks). Baseline random = 0.500.*

| Model | Avg RGA | Avg Spearman IC | % Pos IC | Avg Pearson IC | Top–Bottom Spread | **Acc. Rank** |
|---|---|---|---|---|---|---|
| MLP     | 0.4847 | −0.0007 | 45.71% | +0.0067 | +0.0028 | **1** |
| XGBoost | 0.4815 | +0.0049 | 48.57% | +0.0170 | +0.0046 | **2** |
| Ridge   | 0.4757 | −0.0217 | 37.14% | −0.0176 | −0.0029 | **3** |

*All three models' RGA < 0.500 (random baseline): MLP 0.4847, XGBoost 0.4815, Ridge 0.4757 — all slightly below random. Rankings reflect relative performance among the three models. This is consistent with near-zero IC values and the difficulty of cross-sectional return prediction.*

*Avg top-10 realised returns per month: XGBoost +1.71%, MLP +0.81%, Ridge +1.26%*

**Figures:** model_accuracy_ic_timeseries.png, model_accuracy_ic_boxplot.png, model_accuracy_decile_returns.png, model_accuracy_rga_timeseries.png, model_accuracy_top_bottom_spread.png, model_accuracy_rank_summary.png

---

## Step 7C — Model-Level SAFE: Fairness

*Fairness = whether the model's predictions are equitable across GICS sector groups and market-cap size groups. Metrics: RGA parity gap across groups, absolute rank error per group, top-10 representation deviation.*

| Model | RGF Imparity (↓ better) | Sector RGA gap | Sector rank error | Size RGA gap | Size rank error | **Fair. Rank** |
|---|---|---|---|---|---|---|
| MLP     | 0.0773 | 0.0934 | 0.3333 | 0.0511 | 0.3324 | **1** |
| XGBoost | 0.1101 | 0.0921 | 0.3273 | 0.0596 | 0.3294 | **2** |
| Ridge   | 0.1158 | 0.0999 | 0.3374 | 0.0550 | 0.3368 | **3** |

*All three models show broadly similar fairness profiles. Differences are small (< 1 pp in RGA parity gap). MLP ranks best by fairness primarily because of slightly lower size-level concentration and lower top-10 sector deviation.*

**Figures:** model_fairness_rga_parity_sector.png, model_fairness_rga_parity_size.png, model_fairness_rank_error_sector.png, model_fairness_rank_error_size.png, model_fairness_top10_sector_deviation.png, model_fairness_top10_size_deviation.png, model_fairness_rank_summary.png

---

## Step 7D — Model-Level SAFE: Explainability (RGE, all models)

*RGE (Rank Gradient Explainability) computed for all three models via feature-level rank gradients. Cross-model RGE ranking: XGBoost = 1, Ridge = 2, MLP = 3. SHAP (TreeExplainer) deep-dive retained for XGBoost only (narrative below).*

**Cross-model RGE summary (from rge_results.csv, top feature per model):**

| Model | log_mktcap RGE | ret_12m RGE | ret_1m RGE | **RGE Rank** |
|---|---|---|---|---|
| XGBoost | 0.6387 | 0.2809 | 0.2404 | **1** |
| Ridge   | 0.5886 | 0.1338 | 0.0811 | **2** |
| MLP     | 0.5373 | 0.3591 | 0.3372 | **3** |

*XGBoost has the highest RGE across most features; log_mktcap is the dominant explainability driver for all three models.*

**Global SHAP feature importance (mean |SHAP|, normalised):**

| Rank | Feature | SHAP share | Economic group |
|---|---|---|---|
| 1 | log_mktcap | 31.82% | Size |
| 2 | ret_1m     | 13.21% | Momentum |
| 3 | ret_3m     |  8.25% | Momentum |
| 4 | vol_3m     |  8.02% | Volatility |
| 5 | ret_6m     |  7.71% | Momentum |
| 6 | ret_12m    |  6.88% | Momentum |
| 7 | ret_1w     |  6.68% | Short-term Reversal |
| 8 | amihud     |  6.47% | Liquidity |
| 9 | vol_ratio  |  6.15% | Volatility |
| 10 | vol_1m    |  4.82% | Volatility |

**Economic group breakdown (SHAP-weighted):**
- Momentum: 36.04%
- Size: 31.81%
- Volatility: 18.99%
- Short-term Reversal: 6.68%
- Liquidity: 6.47%

**Directional effects (feature–SHAP correlations):**
- Negative (higher value → lower predicted return): log_mktcap (−0.817), ret_3m (−0.589), vol_ratio (−0.552), ret_1m (−0.551), amihud (−0.337), ret_1w (−0.289), ret_12m (−0.088)
- Positive (higher value → higher predicted return): ret_6m (+0.744), vol_3m (+0.666), vol_1m (+0.187)

**Concentration metrics:**
- HHI (feature importance Herfindahl): **0.1573** (moderate — no single feature dominates)
- Effective number of features: **6.36** (out of 10)
- Top-1 share: 31.8%; top-3 share: 53.3%; top-5 share: 69.0%
- **SHAP HHI-based Explainability score (1 − HHI): 0.8427** (near 1 = well-distributed importance; XGBoost only)

*Note: RGE (Rank Gradient Explainability) is the cross-model explainability metric included in the SAFE composite (Step 7E). The SHAP analysis above provides interpretive detail for XGBoost.*

**Figures:** xgboost_shap_summary.png, xgboost_shap_global_importance_bar.png, xgboost_shap_group_importance_bar.png, shap_dependence_log_mktcap.png, shap_dependence_ret_1m.png, shap_dependence_ret_3m.png, shap_dependence_ret_6m.png, shap_dependence_vol_3m.png

---

## Step 7E — Model-Level SAFE Composite Summary

*Composite = mean(RGR_rank, RGA_rank, RGE_rank, Fairness_rank) — four dimensions. Source: safe_ranking.csv.*

| Model | RGR Rank | RGA Rank | RGE Rank | Fair Rank | SAFE Composite Score | SAFE Composite Rank |
|---|---|---|---|---|---|---|
| XGBoost | 2 | 2 | 1 | 2 | 1.75 | **1** |
| MLP     | 3 | 1 | 3 | 1 | 2.00 | **2** |
| Ridge   | 1 | 3 | 2 | 3 | 2.25 | **3** |

**Best by dimension:**
- Reliability (RGR): Ridge
- Accuracy (RGA): MLP
- Explainability (RGE): XGBoost
- Fairness: MLP

*XGBoost is the primary ML model of interest. It ranks first on explainability (RGE) and second on all other dimensions (reliability, accuracy, fairness) — making it the consistently well-rounded estimator under the SAFE framework.*

**Figures:** model_safe_dimension_ranks.png, model_safe_composite_score.png, xgboost_safe_profile.png

---

## Step 8A — Performance

*Object of evaluation: four MIQP portfolios (Baseline, Ridge, XGBoost, MLP). EqualWeight included as passive reference benchmark.*
*No composite score. Sharpe is the primary portfolio quality metric.*

| Portfolio | Ann Ret | Ann Vol | Sharpe | Sortino | Calmar | MaxDD (%) | Final Wealth |
|---|---|---|---|---|---|---|---|
| XGBoost   | 21.60%  | 23.98%  | 0.9007 | 1.3578 | 0.9225 | −23.41% | 1.8969 |
| Baseline  | 20.93%  | 25.39%  | 0.8243 | 1.1532 | 0.8156 | −25.66% | 1.8597 |
| Ridge     | 13.39%  | 21.76%  | 0.6151 | 0.8759 | 0.5927 | −22.58% | 1.4870 |
| MLP       |  5.45%  | 18.04%  | 0.3021 | 0.4178 | 0.2131 | −25.57% | 1.1753 |
| EqualWt†  | 15.80%  | 14.14%  | 1.1170 | 1.5392 | 0.9780 | −16.15% | —      |

*† EqualWeight: holds all 407 stocks; MaxDD simple-return basis. Not ranked (different cardinality).*

**Net-of-cost Sharpe:**

| Portfolio | Gross Sharpe | Net 10bps | Net 20bps | Net 30bps | Sharpe Decay 30bps |
|---|---|---|---|---|---|
| XGBoost  | 0.9007 | 0.8621 | 0.8234 | 0.7846 | 0.1161 |
| Baseline | 0.8243 | 0.8025 | 0.7807 | 0.7588 | 0.0655 |
| Ridge    | 0.6151 | 0.5717 | 0.5281 | 0.4846 | 0.1305 |
| MLP      | 0.3021 | 0.2411 | 0.1800 | 0.1190 | 0.1831 |

**Performance rank (Sharpe primary):** XGBoost (1) > Baseline (2) > Ridge (3) > MLP (4)

**Individual criterion winners (MIQP portfolios only):**
- Best Sharpe / Sortino / Calmar / Final Wealth / Net Sharpe 30bps: **XGBoost**
- Smallest MaxDD: **Ridge** (−22.58%)
- Lowest turnover (see Step 8B): **Baseline** (46.46%)

**Figures:** portfolio_cumulative_wealth.png, portfolio_drawdowns.png, portfolio_risk_return_scatter.png, portfolio_performance_bars.png

---

## Step 8B — Implementability

*Assesses practical deployability: turnover, cost sensitivity, weight stability, holding persistence, drawdown recovery.*

| Portfolio | Avg Turnover | Max Turnover | Sharpe Decay @30bps | Avg L1 ΔWeight | Avg Holding Overlap | MaxDD Duration (days) | Recovery (days) |
|---|---|---|---|---|---|---|---|
| Baseline | 46.46% | 78.21% | 0.0655 | 0.9347 | 64.29% | 251 | 246 |
| XGBoost  | 78.15% | 100.00% | 0.1161 | 1.5572 | 30.86% | 140 | 164 |
| Ridge    | 79.86% | 100.00% | 0.1305 | 1.5824 | 28.29% | 215 | 265 |
| MLP      | 93.15% | 100.00% | 0.1831 | 1.8632 |  8.00% | 316 | 141 |

**Implementability rank:** Baseline (1) > XGBoost (2) > Ridge (3) > MLP (4)

**Key observations:**
- Baseline: lowest turnover, most stable weights, most persistent holdings
- XGBoost: shortest MaxDD recovery (140 days); MLP longest (316 days)
- At 30 bps, XGBoost Sharpe decays by 0.1161 (0.9007 → 0.7846, −12.9%); Baseline by 0.0655 (−7.9%)
- Even after 30 bps, XGBoost net Sharpe (0.7846) exceeds Baseline net Sharpe (0.7588)
- Break-even transaction cost (XGBoost = Baseline on net Sharpe): approximately 40–45 bps per leg

**Figures:** portfolio_turnover_timeseries.png, holding_overlap_ratio.png, net_sharpe_by_cost.png, sharpe_decay_30bps.png

---

## Step 8C — Diversification

*Assesses portfolio concentration. Near-identical profiles expected under K=10 constraint — validates experimental isolation.*
*(Key finding: diversification differences do not explain performance differences.)*

| Portfolio | Avg Wt HHI | Avg Eff. Holdings | Avg Top-3 Share | Avg Top-5 Share | Max Stock Wt | Avg Sector HHI | Avg Active Sector Dev |
|---|---|---|---|---|---|---|---|
| Baseline | 0.1605 | 6.2558 | 58.23% | 85.01% | 19.97% | 0.2320 | 0.5194 |
| Ridge    | 0.1520 | 6.6260 | 56.66% | 81.29% | 19.93% | 0.2292 | 0.4841 |
| XGBoost  | 0.1629 | 6.1688 | 58.26% | 85.75% | 20.00% | 0.2209 | 0.4586 |
| MLP      | 0.1620 | 6.1981 | 58.31% | 85.47% | 20.00% | 0.2124 | 0.4668 |

**Key observations:**
- Effective holdings range: 6.17–6.63 (spread < 0.5 holdings — negligible)
- Top-3 weight share range: 56.66%–58.31% (spread 1.65 pp — negligible)
- Sector HHI range: 0.2124–0.2320 (spread 0.0196 — negligible)
- Max stock weight binding at 20% for XGBoost and MLP (weight bound binding)
- Near-identical profiles across all four portfolios: no diversification confound on performance ranking

**Figures:** weight_hhi_by_model.png, effective_holdings_by_model.png, top3_weight_share_by_model.png, sector_hhi_by_model.png, active_sector_deviation_by_model.png

---

## Step 8D — Comparative Summary and Research Question

### Full comparative table (all five strategies)

| Portfolio | Ann Ret | Vol | Sharpe | Sortino | Calmar | MaxDD | Turnover | Eff. Hold. | Net SR 30bps | Wt HHI | Active Sec Dev |
|---|---|---|---|---|---|---|---|---|---|---|---|
| XGBoost   | 21.60% | 23.98% | 0.9007 | 1.3578 | 0.9225 | −23.41% | 78.15% | 6.17 | 0.7846 | 0.1629 | 0.4586 |
| Baseline  | 20.93% | 25.39% | 0.8243 | 1.1532 | 0.8156 | −25.66% | 46.46% | 6.26 | 0.7588 | 0.1605 | 0.5194 |
| Ridge     | 13.39% | 21.76% | 0.6151 | 0.8759 | 0.5927 | −22.58% | 79.86% | 6.63 | 0.4846 | 0.1520 | 0.4841 |
| MLP       |  5.45% | 18.04% | 0.3021 | 0.4178 | 0.2131 | −25.57% | 93.15% | 6.20 | 0.1190 | 0.1620 | 0.4668 |
| EqualWt†  | 15.80% | 14.14% | 1.1170 | 1.5392 | 0.9780 | −16.15% | — | 407 | — | — | — |

*† EqualWeight: all 407 stocks; turnover and HHI not comparable (different cardinality).*

### SAFE rank concordance table

*ML models only (Baseline and EqualWeight excluded; not ML models evaluated by SAFE). SAFE composite = mean of 4 dimension ranks (RGR, RGA, RGE, Fairness); source: safe_ranking.csv.*

| Model | SAFE Composite Rank (Step 7E) | Portfolio Sharpe Rank | Concordant? |
|---|---|---|---|
| XGBoost | 1 | 1 | **Yes** |
| MLP | 2 | 3 | No |
| Ridge | 3 | 2 | No |

- Spearman ρ (SAFE rank vs Portfolio Sharpe rank): **0.50**
- p-value: 0.667 (n = 3; statistical inference not feasible)
- Concordant pairs: 1 / 3
- Interpretation: SAFE correctly identifies top-performing ML portfolio (XGBoost); MLP/Ridge ordering reversed

### Ledoit-Wolf (2008) pairwise Sharpe significance tests

*Method: HAC-robust, Newey–West automatic bandwidth (≈11 lags at T=750). Null: N(0,1). Bonferroni threshold: 0.05 / 6 = 0.0083.*

| Portfolio A | Portfolio B | SR_A | SR_B | Diff | Z-stat | p-value | Sig. 5%? | Bonf. thresh. | Sig. Bonf.? |
|---|---|---|---|---|---|---|---|---|---|
| XGBoost | Baseline | 0.9007 | 0.8243 | +0.0764 | 0.1398 | 0.8888 | No | 0.0083 | No |
| XGBoost | Ridge    | 0.9007 | 0.6151 | +0.2856 | 0.6027 | 0.5467 | No | 0.0083 | No |
| XGBoost | MLP      | 0.9007 | 0.3021 | +0.5986 | 1.3211 | 0.1865 | No | 0.0083 | No |
| Baseline | Ridge    | 0.8243 | 0.6151 | +0.2092 | 0.3553 | 0.7223 | No | 0.0083 | No |
| Baseline | MLP      | 0.8243 | 0.3021 | +0.5222 | 0.9900 | 0.3222 | No | 0.0083 | No |
| Ridge    | MLP      | 0.6151 | 0.3021 | +0.3130 | 0.6677 | 0.5043 | No | 0.0083 | No |

*All pairs non-significant at 5% (unadjusted) and 0.83% (Bonferroni). Largest Z = 1.32 (XGBoost vs MLP, p = 0.187).*
*Non-significance reflects power limitation: Lo (2002) requires ~5,800 observations for 80% power at SR diff = 0.60; T ≈ 750 here.*
*Results interpreted directionally, not inferentially.*

### Key thesis questions answered

1. **Does the SAFE-preferred ML model (XGBoost, SAFE rank 1) produce the best portfolio?**
   Yes. XGBoost portfolio Sharpe rank = 1. Direct concordance at the top of the ranking.

2. **Does XGBoost beat the Baseline in portfolio performance?**
   Yes. Sharpe 0.9007 vs 0.8243 (+0.076); Sortino 1.3578 vs 1.1532; Calmar 0.9225 vs 0.8156; MaxDD −23.41% vs −25.66%; final wealth 1.8969 vs 1.8597. Advantage preserved net of 30bps (0.7846 vs 0.7588).

3. **Is the SAFE ranking perfectly concordant with portfolio ranking?**
   Partial only. SAFE rank 1 (XGBoost) = Portfolio rank 1. Positions 2 and 3 inverted: MLP (SAFE 2) is portfolio rank 3; Ridge (SAFE 3) is portfolio rank 2. Spearman ρ = 0.50.

4. **Are the performance differences statistically significant?**
   No pair significant at 5% or Bonferroni 0.83%. Underpowered by design (T ≈ 750). Results directional only.

**Figures:** portfolio_comparative_summary.png, safe_rank_concordance.png, sharpe_significance_heatmap.png

---

## Step 10 — SAFE-Performance Frontier Analysis

**Design:**
- 150 portfolio configurations total: 50 per model family (Ridge, XGBoost, MLP) × 3 families
- Each configuration varies XGBoost-specific hyperparameters (max_depth, learning_rate, n_estimators, subsample, colsample_bytree, reg_alpha, reg_lambda) or equivalent for Ridge/MLP
- SAFE compliance score computed in three aggregation variants: Arithmetic (mean), Geometric (geometric mean), RMS (root mean square) of four SAFE dimensions (Accuracy, Robustness, Fairness, Explainability)
- Portfolio metrics: Sharpe ratio, Max Drawdown (absolute), Average Turnover
- Statistical test battery: Spearman rank correlation + bootstrap 95% CI (n_boot=1000), Kendall τ, Jonckheere–Terpstra monotonicity test (10 compliance bins), Kruskal–Wallis across Low/Mid/High compliance tertiles, Dunn post-hoc pairwise (Bonferroni)

---

**Spearman correlations — overall (n=150):**

| SAFE score | Portfolio metric | r | 95% CI | p-value | Kendall τ | Label |
|---|---|---|---|---|---|---|
| Arithmetic | Sharpe     | +0.5968 | [+0.4558, +0.7054] | 0.0000(*) | +0.4188 | moderate alignment |
| Arithmetic | Max DD     | −0.3497 | [−0.5061, −0.1699] | 0.0000(*) | −0.2215 | moderate alignment |
| Arithmetic | Avg TO     | +0.6108 | [+0.4909, +0.6945] | 0.0000(*) | +0.3853 | strong trade-off |
| Geometric  | Sharpe     | +0.6175 | [+0.4831, +0.7265] | 0.0000(*) | +0.4524 | strong alignment |
| Geometric  | Max DD     | −0.3389 | [−0.4953, −0.1570] | 0.0000(*) | −0.2295 | moderate alignment |
| Geometric  | Avg TO     | +0.5980 | [+0.4717, +0.6847] | 0.0000(*) | +0.3936 | moderate trade-off |
| RMS        | Sharpe     | +0.4170 | [+0.2528, +0.5611] | 0.0000(*) | +0.2998 | moderate alignment |
| RMS        | Max DD     | −0.3545 | [−0.5209, −0.1724] | 0.0000(*) | −0.2401 | moderate alignment |
| RMS        | Avg TO     | +0.6687 | [+0.5565, +0.7485] | 0.0000(*) | +0.4539 | strong trade-off |

**Primary headline (Arithmetic × Sharpe): r = +0.5968, 95% CI [+0.4558, +0.7054], p < 0.0001**

---

**Spearman correlations — by model family (Arithmetic score only):**

| Family  | × Sharpe          | × Max DD          | × Avg TO           |
|---|---|---|---|
| Ridge   | r=+0.4983, p=0.0002(*) | r=+0.3934, p=0.0047(*) | r=−0.3391, p=0.0160(*) |
| XGBoost | r=+0.0895, p=0.5367    | r=−0.2481, p=0.0823    | r=+0.7787, p=0.0000(*) |
| MLP     | r=+0.1524, p=0.2908    | r=−0.5426, p=0.0000(*) | r=+0.7718, p=0.0000(*) |

*Note: Ridge is the only family with a significant SAFE–Sharpe correlation. XGBoost and MLP within-family SAFE–Sharpe correlations are not significant.*

---

**Jonckheere–Terpstra monotonicity (10 compliance bins, H1: monotone increasing for Sharpe/decreasing for Max DD/decreasing for Avg TO):**

| SAFE score | × Sharpe (H1: ↑) | × Max DD (H1: ↓) | × Avg TO (H1: ↓) |
|---|---|---|---|
| Arithmetic | z=+7.4261, p=0.0000(*) | z=−4.0766, p=0.0000(*) | z=+6.8820, p=1.0000 |
| Geometric  | z=+7.8300, p=0.0000(*) | z=−4.1307, p=0.0000(*) | z=+7.0800, p=1.0000 |
| RMS        | z=+5.4130, p=0.0000(*) | z=−4.2101, p=0.0000(*) | z=+7.9185, p=1.0000 |

*Avg TO: p=1.0000 for H1 monotone decreasing means turnover increases monotonically with SAFE compliance — higher SAFE → higher turnover (performance–sustainability trade-off confirmed).*

---

**Kruskal–Wallis (Low/Mid/High compliance tertiles, Arithmetic × Sharpe):**

| Metric  | H statistic | p-value   | Median Low | Median Mid | Median High |
|---|---|---|---|---|---|
| Sharpe  | 69.16 | 0.0000(*) | 0.6351 | 1.1063 | 1.0562 |
| Max DD  | 21.98 | 0.0000(*) | 0.2254 | 0.2210 | 0.1921 |
| Avg TO  | 77.26 | 0.0000(*) | 0.7992 | 0.7803 | 0.9213 |

**Dunn post-hoc (Bonferroni, selected significant pairs):**
- Arithmetic × Sharpe: Low vs Mid p=0.0000(*); Low vs High p=0.0000(*) → Mid and High both significantly better than Low
- Arithmetic × Max DD: Low vs High p=0.0001(*); Mid vs High p=0.0003(*)
- Arithmetic × Avg TO: Low vs High p=0.0000(*); Mid vs High p=0.0000(*)

---

**Model-level SAFE–Sharpe summary (by family, n=50 each):**

| Family  | n  | Spearman ρ | p-value   | Best-SAFE config | Best-SAFE Sharpe | Best-Sharpe config | Best-Sharpe | Same config? |
|---|---|---|---|---|---|---|---|---|
| Ridge   | 50 | +0.4983 | 0.0002(*) | ridge_32  | 0.632  | ridge_01  | 0.6353 | No |
| XGBoost | 50 | +0.0895 | 0.5367    | xgboost_47 | **1.8122** | xgboost_47 | **1.8122** | **Yes** |
| MLP     | 50 | +0.1524 | 0.2908    | mlp_30    | 1.1904 | mlp_14    | 1.7729 | No |

*XGBoost: the single highest-SAFE-compliance configuration (xgboost_47) is identical to the highest-Sharpe configuration. This is the Pareto-dominant result.*

---

**Pareto-dominant configuration (Arithmetic × Sharpe):**

| Model    | Config ID   | Params | Compliance (Arith) | Sharpe | Max DD | Avg TO | Ann Ret | Ann Vol |
|---|---|---|---|---|---|---|---|---|
| XGBoost  | xgboost_47  | md=5, lr=0.15, ne=500, ss=0.6, cb=0.6, ra=0.0, rl=0.1 | 0.6289 | **1.8122** | 0.1628 | 0.9196 | 35.24% | 19.44% |

*Accuracy score=0.4647, Robustness score=0.5760, Fairness score=0.8277, Explainability score=0.6471.*
*Sortino=2.5992, Calmar=2.1647.*

---

**Figures produced (step10):**
- frontier_figures/: raw, binned, cumulative, pareto scatter plots for all 9 combinations (3 SAFE aggregations × 3 portfolio metrics)
- frontier_figures/model_level_safe_sharpe_by_family.png, model_level_safe_sharpe_combined.png
- frontier_stats/: bootstrap_frontier_{arithmetic/geometric/rms}_{sharpe/maxdrawdown/avgturnover}.png
- frontier_stats/stats_correlations.csv, stats_monotonicity.csv, stats_kruskal.csv, stats_dunn.csv (+ .tex)

---

## Figure Inventory by Step

| Step | Folder | Key figures |
|---|---|---|
| 2 | figures/step2/ | corr_distribution, return_distribution, sector_breakdown, vol_distribution |
| 3 | figures/step3/ | baseline_cumulative_returns, baseline_drawdown, baseline_rolling_sharpe, baseline_sector_heatmap, baseline_turnover, efficient_frontier |
| 6 | figures/step6/ | portfolio_cumulative_wealth_comparison, portfolio_drawdown_comparison, portfolio_rolling_sharpe_comparison, portfolio_sharpe_turnover_scatter, portfolio_average_sector_allocation_comparison, portfolio_average_turnover_barplot, portfolio_turnover_comparison, baseline_sector_heatmap, xgboost_sector_heatmap |
| 7/accuracy | figures/step7/accuracy/ | ic_timeseries, ic_boxplot, decile_returns, rga_timeseries, top_bottom_spread, rank_summary |
| 7/explainability | figures/step7/explainability/ | shap_summary, shap_global_importance_bar, shap_group_importance_bar, shap_dependence_* (5 plots) |
| 7/fairness | figures/step7/fairness/ | rga_parity_sector, rga_parity_size, rank_error_sector, rank_error_size, top10_sector_deviation, top10_size_deviation, rank_summary |
| 7/sustainability | figures/step7/sustainability/ | rank_stability, noise_robustness, topk_overlap, mu_dispersion, winsorization_robustness, rank_summary |
| 7/safe_summary | figures/step7/safe_summary/ | model_safe_dimension_ranks, model_safe_composite_score, xgboost_safe_profile |
| 8/performance | figures/step8/portfolio_performance/ | portfolio_cumulative_wealth, portfolio_drawdowns, portfolio_risk_return_scatter, portfolio_performance_bars |
| 8/implementability | figures/step8/portfolio_implementability/ | portfolio_turnover_timeseries, holding_overlap_ratio, net_sharpe_by_cost, sharpe_decay_30bps |
| 8/diversification | figures/step8/portfolio_diversification/ | weight_hhi_by_model, effective_holdings_by_model, top3_weight_share_by_model, sector_hhi_by_model, active_sector_deviation_by_model |
| 8/comparative | figures/step8/comparative/ | portfolio_comparative_summary, safe_rank_concordance, sharpe_significance_heatmap |
| 10 | data/results/step10/frontier_figures/ | raw/binned/cumulative/pareto scatters (9 metric combos), model_level_safe_sharpe_by_family, model_level_safe_sharpe_combined |
| 10 | data/results/step10/frontier_stats/ | bootstrap_frontier_* (9 files), stats_correlations, stats_monotonicity, stats_kruskal, stats_dunn |
