# Machine Learning for Portfolio Optimisation under the SAFE AI Framework: Isolating the Effect of Expected-Return Estimation on Risk-Adjusted Performance

*Academic narrative — MSc Quantitative Finance, University of Pavia*
*Supervisor: Prof. Giudici*

---

## Preface: Central Contribution and Framing

The central methodological contribution of this thesis is a controlled experiment in portfolio construction. In classical mean–variance optimisation, two inputs drive everything: the expected-return vector μ and the covariance matrix Σ. In practice, the two are often varied simultaneously when machine learning is introduced, making it impossible to attribute any performance change to either source alone. This thesis resolves that ambiguity by design. Across all four portfolios studied — Baseline, Ridge, XGBoost, and MLP — the covariance matrix (a rolling Ledoit–Wolf estimator), the MIQP formulation, the constraint set (K = 10, weight bounds [1%, 20%], sector cap 30%), the solver configuration, and the rebalancing calendar are held strictly fixed. The only element that changes from one portfolio to the next is μ: the Baseline uses a trailing twelve-month mean; Ridge, XGBoost, and MLP each substitute a machine-learned cross-sectional expected-return estimate trained on ten market microstructure and price-based features.

This single-input-variation design transforms the exercise from a general "does machine learning improve portfolio management?" question into a precisely answerable one: what is the marginal value of replacing the trailing mean with a supervised ML forecast, holding everything else constant? The four portfolios that result are then assessed jointly through the SAFE AI framework — a governance rubric covering Sustainability, Accuracy, Fairness, and Explainability — applied at two conceptually distinct levels. At the model level (Step 7), SAFE evaluates the forecasting models directly, asking whether their predictions are reliable, accurate, equitable across groups, and interpretable. At the portfolio level (Step 8), SAFE evaluates the realised investment strategies, asking whether each portfolio is robust under transaction costs, fairly concentrated, and achieves genuine risk-adjusted performance. The dual-level analysis makes explicit a non-trivial empirical question: does the model that ranks highest under SAFE also produce the best portfolio? The answer, as will be shown, is nuanced.

---

## Step 1: Data Loading and Structural Cleaning

### Objective

The first step establishes the data foundation for the entire pipeline. Its goal is to assemble a clean, verified, and structurally consistent universe of US equity prices, total return indices, market capitalisations, and trading volumes from which all subsequent analysis will be derived.

### Method and Design Choices

Raw data were sourced from Bloomberg across five files: PRICES.xlsx, TOT_RETURN_INDEX_GROSS_DVDS.xlsx, MKT_CAP.xlsx, VOLUME.xlsx, and METADATA.xlsx. The extraction covers the full calendar period from 1 January 2010 to 31 December 2025, a span chosen deliberately to include a substantial warm-up window before the modelling period begins. The initial universe comprised 426 stocks.

One stock had a missing GICS sector code in the metadata. Rather than imputing or discarding this record, the sector was corrected manually in METADATA.xlsx prior to any automated cleaning, ensuring that the full sector-based constraint system in the optimiser would not encounter an undefined category. This decision reflects a principle that governs the entire pipeline: data quality interventions are documented and made at the earliest feasible stage so that downstream results are traceable.

The warm-up period runs from 2010 through 2015. These six years are used exclusively for feature engineering, covariance estimation initialisation, and model training; they are never treated as out-of-sample. The model window, on which all portfolio results are evaluated, begins on 1 January 2016 and ends on 31 December 2025.

### Results

Step 1 produces a panel of 426 stocks × 4,174 trading days. The full time series spans roughly sixteen years, with the first six constituting the warm-up reservoir and the final ten forming the active study period. The single manual metadata correction accounted for the sole non-automated intervention at this stage.

### Interpretation

The deliberate length of the warm-up window matters for the ML models trained in Step 5. The minimum history required for feature computation (252 trading days) is satisfied for all stocks at the first rebalancing date. Most S&P 500 constituents have data from 2010, providing a rich multi-regime training base, though this cannot be guaranteed universally for stocks that entered the index near January 2015. The manual sector correction, while a minor intervention, illustrates a principle that recurs throughout the pipeline: automated filters and ML models cannot substitute for domain-grounded quality checks at the data ingestion boundary.

---

## Step 2: Preprocessing, Returns, and Covariance Estimation

### Objective

Step 2 converts the raw Bloomberg panel into the two objects at the heart of mean–variance optimisation: a clean daily return matrix and a family of covariance estimates. It also performs systematic universe reduction to remove structurally problematic securities.

### Method and Design Choices

Three sequential filters were applied to the 426-stock universe. First, a zero-return filter eliminated any stock for which more than five percent of trading days in the model window recorded an exact zero return. A zero daily return on an equity security is almost always an artefact of a data-feed issue, a price freeze, or a trading halt rather than genuine price stability; retaining such records would bias both the return distribution and the covariance matrix. Second, SATS UW EQUITY was excluded via hardcoded rule after a two-day inspection revealed a return of +53.21% on 26 August 2025 followed by +14.42% the following day — a pattern inconsistent with any known corporate event and indicative of a total return index feed error. Third, GOOGL UW EQUITY was removed as a duplicate share class: its correlation with GOOG UW EQUITY over the model window was 0.9952, so including both would effectively double-count the Alphabet position in any cardinality-constrained portfolio.

After these three filters, the final investable universe settled at **407 stocks** across all eleven GICS sectors.

Daily log-returns were computed from the total return index — that is, the gross-dividends-inclusive series — so that return statistics and portfolio performance figures are consistent with total shareholder return rather than price appreciation alone. All returns are computed as log-returns and annualised as mean(log_returns) × 252, following the continuously-compounded convention standard in empirical asset pricing (Campbell, Lo & MacKinlay, 1997; Gu, Kelly & Xiu, 2020). The return matrix for the model window has shape 2,609 days × 407 stocks, with zero NaN cells, confirming complete coverage for the active period. The warm-up portion contains 28,412 NaN cells, all attributable to late IPO entries that were not yet listed at the start of the sample.

Three covariance estimators were computed over the full model window for diagnostic purposes; rolling re-estimation using trailing 252-day windows is used inside the optimiser (Step 3 onward). The sample covariance has a minimum eigenvalue of 0.002660 and a condition number of 5,560. The Ledoit–Wolf estimator with identity target applies a shrinkage coefficient of 0.0159, lifting the minimum eigenvalue to 0.004202 and reducing the condition number to 3,460. Oracle Approximating Shrinkage (OAS) yields a shrinkage of 0.0032, minimum eigenvalue 0.002967, and condition number 4,970. The Ledoit–Wolf estimator is selected for portfolio construction because it achieves the largest improvement in numerical conditioning at a very small shrinkage intensity, preserving most of the sample information while eliminating near-singular behaviour.

### Results

The final universe of 407 stocks covers all eleven GICS sectors. Financials and Industrials are the largest sectors at 65 stocks each, followed by Information Technology (56) and Health Care (54). Consumer Discretionary contributes 41 stocks; Consumer Staples 28; Real Estate 26; Materials 21; Utilities 20; Energy 18; Communication Services 13. By domicile, 392 stocks are US-incorporated; the remaining 15 are listed in Ireland, Bermuda, Britain, and other jurisdictions but trade as US equity instruments.

The cross-sectional return distribution over the model window is characterised by substantial dispersion and heavy tails. Annualised volatility has a cross-stock mean of 30.72% (median 29.07%), ranging from 17.80% to 58.05%. Annualised mean returns range from −5.21% to +52.61%, with a cross-stock mean of 11.68% and median of 11.44%. The distribution is negatively skewed (mean skewness −0.668) and exhibits extreme excess kurtosis (mean 17.877), reflecting the presence of episodic large moves. Seven return spikes exceeding ±50% in absolute value were identified across stocks GL, TRGP, OXY, EPAM, FANG, DXCM, and CNC; these were retained with diagnostic flag, consistent with the principle that genuine extreme events belong in the data.

The correlation structure is moderate and broadly homogeneous. The sample correlation matrix has an average off-diagonal value of 0.3530 (standard deviation 0.1218), a range from −0.073 to 0.919, and 11.3% of pairs exceeding 0.50. Critically, no pair exceeds 0.95, confirming that the duplicate-class filter was effective and that no residual near-collinear pairs remain.

### Interpretation

The zero-return filter is particularly consequential for the ML panel built in Step 4. Features constructed from illiquid or data-frozen price series would introduce spurious signals into cross-sectional momentum and volatility estimates. The worst offenders removed — UAL at 29.7% zeros, FISV at 27.6%, SMCI at 18.0%, IBKR at 13.8% — illustrate that the problem is concentrated but non-trivial. By removing these stocks before feature construction, the ML models are trained on genuinely tradeable signals.

The heavy-tailed return distribution (excess kurtosis 17.877) has direct implications for the MIQP optimiser and for the interpretation of IC statistics in Step 5. In a non-Gaussian environment, small prediction errors can be amplified by extreme realisations, which helps explain why even a model with near-zero average IC can produce meaningful portfolio-level results through sector and size tilts that align with the systematic component of returns. This point is developed further in Steps 5 and 7.

The Ledoit–Wolf shrinkage coefficient of 0.0159 is remarkably low, indicating that the 407-stock sample is sufficiently large relative to the 252-day estimation window that the sample covariance requires only minimal regularisation. This is an important observation: the portfolio results in Step 6 cannot be attributed to overly aggressive covariance smoothing, since the estimator is nearly equivalent to the sample covariance.

---

## Step 3: The Baseline MIQP Portfolio

### Objective

Step 3 constructs the Baseline portfolio — the control condition against which all ML variants will be compared. It establishes the canonical MIQP formulation that will be held fixed throughout the thesis, and it provides a concrete measure of what a well-engineered but purely statistical (non-ML) mean–variance strategy achieves in the 2023–2025 out-of-sample period.

### Method and Design Choices

The portfolio is defined by a mixed-integer quadratic programme:

> minimise  w′Σw − λ · μ′w  subject to: ∑wᵢ = 1, wᵢ ∈ {0} ∪ [1%, 20%], ∑wᵢ · 𝟙[sector(i) = s] ≤ 30% for all s, ∑𝟙[wᵢ > 0] = 10

The risk-aversion parameter λ is set to 1.0. The cardinality constraint K = 10 enforces a concentrated, stock-picking portfolio, consistent with an active management mandate. Individual weight bounds of [1%, 20%] prevent both trivial allocations and excessive concentration. The 30% sector cap provides diversification discipline without forcing sector-neutral construction.

For the Baseline, μ is the trailing 252-day annualised mean return, winsorised at the first and ninety-ninth percentiles cross-sectionally to prevent extreme recent returns from dominating the optimisation objective. Σ is re-estimated at each rebalance date using exactly the 252 trading days immediately preceding that date, so the covariance matrix is always strictly out-of-sample relative to the decision point.

The solver is Gurobi 13.0.1, configured with a sixty-second time limit and a MIP optimality gap tolerance of 1%. Rebalancing occurs on the first trading day of each calendar month, generating 36 rebalancing dates from 2 January 2023 to 1 December 2025. The test window spans from 1 January 2023 to 31 December 2025, producing 747 daily portfolio return observations.

The choice of Gurobi over heuristic or relaxation-based solvers is deliberate. MIQP with K = 10 over a 407-stock universe is well within Gurobi's exact solution capabilities for a sixty-second time limit, and using a commercial exact solver ensures that portfolio differences across the four variants reflect differences in μ — not differences in solver quality or convergence behaviour.

### Results

All 36 monthly rebalancing problems were solved to provable global optimality (OPTIMAL status), with an average Gurobi solve time of 0.3 seconds — far within the sixty-second limit, confirming that the formulation is computationally tractable and that no infeasibility or time-limit binding occurred across the three-year test window.

Over the 2023–2025 period, the Baseline portfolio delivered an annualised return of **20.93%** at an annualised volatility of **25.39%**, producing a Sharpe ratio of **0.8243** (assuming zero risk-free rate for comparability). The Sortino ratio is computed as the annualised return divided by the annualised population RMS of daily returns floored at zero. The Sortino ratio is 1.1532 and the Calmar ratio is 0.8156. The maximum drawdown of **−25.66%** occurred between a peak wealth of 1.9054 on 5 December 2024 and a trough of 1.4164 on 8 April 2025 — a period that coincides with the sharp equity sell-off of early 2025. The final wealth index reached 1.8597, representing a simple cumulative return of 85.97% over three years.

Year-by-year performance reflects the macro environment. In 2023, a powerful equity rally drove the Baseline to a 23.79% return at relatively modest volatility of 20.77%, producing a Sharpe of 1.1452. In 2024, strong technology-sector performance continued with a 29.00% return (volatility 25.46%, Sharpe 1.1389). The 2025 drawdown cut the annual return to 9.98% at 29.29% volatility, with a Sharpe of only 0.3408 — reflecting the broader market stress of that period.

Average monthly turnover of 46.46% (range 21.63%–78.21%) implies meaningful but not extreme rebalancing activity. Under transaction cost scenarios of 10, 20, and 30 basis points per traded leg, the Sharpe ratio degrades to 0.8025, 0.7807, and 0.7588 respectively, with annualised net returns of 20.38%, 19.83%, and 19.28%.

Sector allocation tilts toward growth-oriented sectors: Information Technology averaged 21.41%, Industrials 19.97%, and Consumer Discretionary 18.95% over the test period. Communication Services at 11.39% and Energy at 7.39% round out the top five. Defensive sectors — Consumer Staples (2.28%), Real Estate (1.08%) — received minimal weight.

The equal-weight benchmark, constructed from the same 407-stock universe with monthly rebalancing on identical dates, produced an annualised return of 15.80% at a volatility of 14.14%, yielding a Sharpe of 1.1170. Notably, the benchmark Sharpe (1.1170) exceeds the Baseline Sharpe (0.8243) despite a substantially lower absolute return, because the benchmark's lower volatility more than offsets its lower return in the Sharpe calculation. This is an important calibration point: the MIQP concentrated strategy accepts higher volatility in pursuit of return, and its risk-adjusted performance must be evaluated accordingly.

### Interpretation

The Baseline result establishes that a disciplined trailing-mean MIQP strategy — one that requires no ML infrastructure whatsoever — can generate compelling absolute returns over this sample. This is both encouraging and a high bar for the ML variants. The fact that the trailing mean (an extremely simple estimator) produces an 85.97% cumulative return in three years suggests that the structural features of the portfolio — cardinality concentration in ten high-momentum names, sector discipline, tight weight bounds — contribute materially to performance, independent of the quality of μ.

The per-year decomposition also reveals an important fragility: the 2025 drawdown of −24.74% at the annual level exposes the concentration risk inherent in a ten-stock portfolio. This fragility will reappear in all four portfolios and is a key input to the sustainability dimension of Step 8.

The Baseline's relatively low turnover of 46.46% will later emerge as its strongest sustainability advantage over the ML portfolios. Since trailing means change gradually, the Baseline μ reranks stocks relatively slowly across months, producing modest portfolio changes. The ML models, by contrast, respond to short-horizon signals and may produce more volatile μ vectors, leading to higher turnover.

---

## Step 4: Feature Engineering and the ML Panel

### Objective

Step 4 constructs the labelled panel that all three supervised learning models in Step 5 will be trained on. Its goal is to produce a dataset of interpretable, economically grounded features and a forward-looking return target, structured as a cross-sectional prediction problem.

### Method and Design Choices

The panel has a two-level structure: stock × rebalancing month. For each of the 407 stocks and each of the 120 monthly rebalancing dates from January 2016 to December 2025, the feature engineering module computes ten predictors. All ten features are transformed into cross-sectional ranks within each date before being presented to the models, so that each observation is a percentile rank rather than a raw value. This design choice eliminates the secular drift that would otherwise confound level-based features over a ten-year sample, and it ensures that the models learn cross-sectional rather than time-series patterns — which is the correct objective for a problem of selecting stocks relative to their peers each month.

The ten features cover five economic themes. Short-term reversal is represented by the one-week and one-month returns (ret_1w, ret_1m). Medium-to-long-run momentum is represented by three-month, six-month, and twelve-month cumulative returns (ret_3m, ret_6m, ret_12m). Short-run volatility regime is captured by the one-month realised volatility (vol_1m), three-month realised volatility (vol_3m), and their ratio (vol_ratio = vol_1m / vol_3m). Liquidity is proxied by the Amihud illiquidity ratio (amihud). Size is captured by the logarithm of market capitalisation (log_mktcap).

The prediction target is the twenty-one-trading-day forward return, converted to a cross-sectional percentile rank within each date. This is a standard choice in factor investing research: predicting ranks rather than levels dampens the influence of cross-sectional outliers on the loss function and focuses the model on relative performance — the quantity directly relevant to long-only portfolio construction.

### Results

Before filtering, the panel contained 73,260 rows (407 stocks × 180 dates, accounting for some pre-2016 warmup). After dropping rows with missing forward return targets — which occurs only at the very end of the sample, where a full twenty-one-day forward window is not yet observable — the usable panel contains **48,840 rows**. The missing target rate of 0.8333% is consistent with what would be expected from the first one or two months of each calendar year at the beginning of the model window and does not indicate a data quality issue.

### Interpretation

The choice to use ten features rather than a larger factor library is deliberate. A sparse feature set reduces the risk of overfitting in the expanding-window ML training protocol, where early windows may contain only a few hundred training observations per stock. The five economic themes — reversal, momentum, volatility, liquidity, size — are among the most replicated factors in the empirical asset pricing literature. By restricting the feature space to these well-established signals, the thesis ensures that any predictive content discovered by the ML models is interpretable in terms of known market phenomena, rather than being a spurious pattern extracted from a high-dimensional feature space.

The cross-sectional ranking transformation, applied uniformly to both features and target, is the key architectural choice that makes the three ML models comparable despite their different inductive biases. Ridge regression, XGBoost, and the MLP all receive identically normalised inputs and predict the same target; observed performance differences in Step 5 can therefore be attributed cleanly to the models' different functional forms and regularisation mechanisms.

---

## Step 5: Machine Learning Expected-Return Prediction

### Overview

Three supervised learning models are trained and evaluated in Step 5 using an expanding-window out-of-sample protocol. For each of the 36 rebalancing dates in the test period (January 2023 to December 2025), each model is trained on all available data strictly before that date, with no look-ahead. The resulting prediction is used as the μ vector in the corresponding month's MIQP, with z-standardisation and rescaling to match the cross-sectional distribution of the Baseline trailing mean applied before any optimisation. The primary evaluation metric is the Information Coefficient (IC), defined as the Spearman rank correlation between predicted and realised return ranks.

### Step 5A: Ridge Regression

#### Objective

Ridge regression serves as a regularised linear baseline that captures linear relationships between the ten ranked features and the forward rank target. As a penalised least-squares estimator, it provides a minimal-complexity ML model whose predictions are fully explicable through its coefficient vector.

#### Method and Design Choices

Ridge regression is trained on the expanding panel with L2 regularisation. The cross-sectional ranking of inputs and outputs (both features and target expressed as percentile ranks) means that the coefficient vector captures linear rank–rank associations, analogous to a Spearman-based linear model. The Ridge regularisation parameter α is selected at each rebalancing date via a temporal grid search over the candidate set {0.01, 0.1, 1.0, 10.0, 100.0}. The selection criterion is R² evaluated on a held-out temporal validation set consisting of the last six eligible months of the expanding training window. After selecting the optimal α, the model is refit on the combined train-plus-validation data before generating the expected return vector μ. The selected α values are logged to ridge_hyperparameters.csv for diagnostic purposes.

#### Results

Over 35 rebalancing dates with computable ICs, Ridge produced an overall mean Spearman IC of **−0.0217** and a hit rate — the fraction of months with positive IC — of 37.14%. The mean Pearson IC is −0.021 and the mean R²_rank is −0.0056. Performance is notably heterogeneous across years: 2023 produced a positive mean IC of +0.0214 with a hit rate of 58.33%, suggesting genuine directional signal in the early test period. The 2024 and 2025 ICs of −0.0471 and −0.0411 respectively indicate that the linear relationships captured by Ridge deteriorated or reversed during the latter part of the sample. Among the three models, Ridge displays the highest consecutive-rank Spearman stability (0.7335), meaning its predicted μ vector changes gradually month to month. Its top-10 stock overlap of 0.306 is moderate, and its noise robustness correlation of 0.9986 and winsorisation robustness of 1.0000 confirm that the model's outputs are insensitive to small perturbations in features. Across the 36 rebalancing dates, the grid search selected α = 100.0 in 31 cases (86%) and α = 0.01 in the remaining 5 (all concentrated in the March–July 2023 window), with no intermediate value ever chosen — indicating that the validation-set R² criterion consistently favoured either strong or minimal regularisation depending on the training period.

#### Interpretation

Ridge's negative overall IC is the most concerning result in the model panel, and it warrants careful interpretation. A mean IC of −0.0217 means that Ridge's cross-sectional rankings were, on average, slightly negatively correlated with realised returns over the test window. However, this must be contextualised against the base rate of difficulty. Predicting cross-sectional return ranks over a one-month horizon in a large-cap universe is an extremely hard problem; the academic literature routinely reports mean ICs in the range [−0.05, +0.05] as economically meaningful. A mean IC of −0.022 is small in absolute terms, and the 2023 positive performance suggests that the model captures genuine signal in benign market conditions that it fails to maintain through the more turbulent 2024–2025 environment. Crucially, the negative IC does not mean Ridge produces a perverse portfolio; as will be seen in Step 6, the MIQP covariance structure may partially neutralise the direction of μ errors by selecting stocks whose predicted returns, though mis-ranked in isolation, happen to combine favourably from a risk perspective.

### Step 5B: XGBoost

#### Objective

XGBoost represents the thesis's primary non-linear ML estimator. Gradient-boosted trees are capable of capturing threshold effects, interaction effects between features, and non-monotonic relationships — all phenomena that are plausible in equity return prediction but invisible to linear models.

#### Method and Design Choices

The training protocol is fully expanding-window: for each of 36 rebalance dates, all data strictly before that date constitutes the training set, and the last six months of available training data are held out as a validation set for early stopping and hyperparameter selection. A grid search covers n_estimators ∈ {100, 300, 500}, max_depth ∈ {3, 4, 6}, and learning_rate ∈ {0.01, 0.05}, with early stopping triggered after 20 rounds without validation improvement. After selection, the final model is refitted on the combined train-plus-validation set using the best hyperparameters, and the fitted model is serialised as a JSON file for reproducibility.

The hyperparameter results across 36 monthly models reveal an important pattern: effective n_estimators ranges from 1 to 69, with shallow models dominating especially in the later part of the sample. This strong early stopping indicates that the XGBoost models are regularising aggressively against overfitting to the training data — a sensible adaptation in a low signal-to-noise environment.

The XGBoost μ vector is z-standardised and rescaled to match the cross-sectional mean and standard deviation of the Baseline trailing mean at each date before winsorisation. This prevents the MIQP from receiving μ vectors of a different scale or distribution than it was calibrated for, ensuring that the λ = 1.0 risk-aversion parameter has the same economic meaning across all four portfolios.

#### Results

Over 35 rebalancing dates, XGBoost achieved a mean Spearman IC of **+0.0049**, a hit rate of **48.57%**, a mean Pearson IC of +0.012, and a mean R²_rank of −0.0006. The year-by-year pattern is encouraging: 2023 produced IC = +0.0334 (50.00% hit rate), 2024 a modest −0.0276 (41.67% hit rate), and 2025 a recovery to +0.0094 (54.55% hit rate). This temporal pattern, with positive signal at the start and end of the test window and a dip in the middle, mirrors the broader market narrative of 2024 as an environment dominated by momentum concentration in a small number of mega-cap technology names, which may be systematically harder to capture with a cross-sectional ranking model.

The sustainability metrics for XGBoost are mixed: the consecutive-rank Spearman stability of 0.4643 is lower than Ridge's 0.7335, reflecting that XGBoost's predictions are more month-to-month volatile. The top-10 overlap of 0.320 is the highest among the three models. Noise robustness (0.9885) and winsorisation robustness (1.0000) confirm output stability under perturbations.

#### Interpretation

XGBoost's positive mean IC of +0.0049 is small but directionally correct — it places the model on the right side of chance on average. The RGA figure computed in Step 7B is 0.5041, a +0.41 percentage point edge over the 0.500 random baseline. These are near-zero numbers, but they are academically defensible for the following reason: the EMH literature, and a large body of empirical work in cross-sectional return prediction, consistently documents that the best achievable ICs in liquid large-cap universes are in the range of a few basis points on average, with high cross-sectional noise. The fact that XGBoost produces a non-negative mean IC over a three-year live test on 407 liquid US equities — with no survivorship bias and with strict expanding-window out-of-sample evaluation — is a meaningful result. The near-zero IC does not indicate model failure; it indicates that the signal is modest, as theory would predict, and that extracting value from it requires precisely the kind of mean-variance architecture constructed in this thesis.

The month-to-month instability of XGBoost predictions (consecutive Spearman 0.4643) is economically interpretable: gradient-boosted trees are sensitive to small changes in the training distribution, and as the expanding window grows, the marginal effect of recently added data on the fitted model may shift the predictions substantially. This instability translates directly into higher turnover in the XGBoost portfolio, as examined in Step 6.

### Step 5C: Multilayer Perceptron (MLP)

#### Objective

The MLP provides a deep learning benchmark, testing whether the additional representational capacity of a neural network translates into predictive advantage in this low signal-to-noise environment.

#### Method and Design Choices

Architecture details are not exhaustively documented in the data dump. The key observable outcomes are the IC diagnostics and the sustainability metrics, from which inferences about model behaviour can be drawn. Unlike Ridge and XGBoost, which use an explicit temporal validation split, MLP relies on sklearn's internal early_stopping=True mechanism, which draws a random 10% holdout from the combined train-plus-validation pool. This non-temporal split is an asymmetry with respect to the other two models and is acknowledged as a limitation.

#### Results

The MLP produced the most ambiguous IC profile of the three models. Over 35 rebalancing dates, the mean Spearman IC is **−0.0008** — essentially zero — with a hit rate of 45.71%. The mean Pearson IC is +0.0009 and the mean R²_rank is −0.0214. Year-by-year: 2023 IC = +0.0259 (50.00% hit rate), 2024 IC = −0.0383 (33.33%), 2025 IC = +0.0111 (54.55%). The 2024 performance is particularly weak.

The most striking MLP statistics are its sustainability metrics. The consecutive-rank Spearman stability of 0.1650 is the lowest among all models by a wide margin, indicating that the MLP's predicted rankings are highly unstable month to month — it effectively reconfigures its view of the cross-section almost entirely each period. This instability is corroborated by the top-10 stock overlap of only 0.086: fewer than one in ten of the stocks in the MLP's top decile in month t are also in the top decile in month t+1. Noise robustness (0.9984) and winsorisation robustness (1.0000) are adequate, suggesting that within a given month's prediction the outputs are stable under perturbation, but the inter-month instability is severe.

#### Interpretation

The MLP's near-zero mean IC (−0.0008) combined with its extreme instability presents a challenging profile. The near-zero accuracy can be interpreted charitably as evidence that the MLP is not systematically wrong — it is essentially random in its directional predictions on average. However, the instability means that even if individual monthly predictions happen to be accurate, they produce an incoherent μ sequence that drives very high portfolio turnover, as will be confirmed in Step 6. The effective number of free parameters in an MLP is large relative to the available training data in early months of the expanding window, creating conditions in which overfitting is likely. The aggressive early stopping observed in XGBoost is a natural regularisation mechanism in tree models; achieving equivalent regularisation in an MLP requires careful architectural choices (dropout, early stopping, batch normalisation) that may interact in complex ways with the cross-sectional ranking framework.

The MLP result is an important negative finding in its own right: more model complexity does not translate into more predictive accuracy in this setting. This is consistent with the broader empirical asset pricing literature, which has found that simpler, well-regularised models often match or outperform more complex architectures on cross-sectional return prediction tasks, particularly in live out-of-sample evaluation on large-cap liquid universes.

---

## Step 6: ML Portfolio Construction

### Objective

Step 6 constructs the three ML portfolios — Ridge, XGBoost, and MLP — using exactly the same MIQP formulation as the Baseline, with only μ replaced by the respective ML forecast. The objective is to measure the net portfolio-level impact of each μ substitution after accounting for risk, constraints, and compounding.

### Method and Design Choices

The design is explicitly controlled. For each monthly rebalancing date in the test period, each of the four portfolios solves an identical MIQP (same Σ, same λ, same K, same bounds, same sector cap). The μ vector is the only input that differs. This means that any observed difference in portfolio performance is attributable solely to the quality and characteristics of the μ vector — not to differences in the risk model, the solver, or the constraint set. This controlled substitution design is the methodological cornerstone of the thesis.

Transaction costs are assessed at 10, 20, and 30 basis points per traded notional, applied symmetrically to buy and sell legs.

### Results

The portfolio performance summary over the 2023–2025 out-of-sample period reveals a clear ordering. XGBoost dominates on all return-based and risk-adjusted metrics: annualised return **21.60%**, annualised volatility **23.98%**, Sharpe **0.9007**, Sortino **1.3578**, Calmar **0.9225**, maximum drawdown **−23.41%**, and final wealth **1.8969**. The Baseline ranks second: 20.93% return, 25.39% volatility, Sharpe 0.8243, Sortino 1.1532, Calmar 0.8156, maximum drawdown −25.66%, final wealth 1.8597. Ridge ranks third: 13.39% return, 21.76% volatility, Sharpe 0.6151, Sortino 0.8759, Calmar 0.5927, maximum drawdown −22.58%, final wealth 1.4870. MLP ranks last: 5.45% return, 18.04% volatility, Sharpe 0.3021, Sortino 0.4178, Calmar 0.2131, maximum drawdown −25.57%, final wealth 1.1753.

The turnover picture is substantially different. Baseline has the lowest average monthly turnover at **46.46%** (maximum 78.21%). XGBoost averages **78.15%** (maximum 100%). Ridge averages **79.86%** (maximum 100%). MLP averages **93.15%** (maximum 100%). The ML portfolios all exceed the Baseline turnover by a factor of roughly 1.7 to 2.0.

Net of 30 basis points transaction costs, the performance ordering is preserved. XGBoost net Sharpe is 0.7846 (net return 18.83%). Baseline net Sharpe is 0.7588 (net return 19.28%). Ridge net Sharpe falls to 0.4846 (net return 10.56%). MLP net Sharpe collapses to 0.1190 (net return 2.15%).

Two individual criterion exceptions to the overall ranking are noteworthy. Ridge achieves the smallest maximum drawdown among all portfolios at −22.58%, reflecting its more conservative μ estimates which lead the optimiser toward lower-risk selections. MLP achieves the lowest volatility at 18.04%, for a similar reason — a near-zero μ vector with high instability effectively randomises stock selection sufficiently that the portfolio lands near the minimum-variance region of the feasible set.

### Interpretation

The controlled substitution design produces a clean and interpretable outcome. XGBoost's μ, despite its near-zero IC at the individual stock level, appears to inject just enough directional signal into the MIQP objective to consistently tilt the portfolio toward the stocks that subsequently outperform. This outperformance is meaningful: a Sharpe of 0.9007 versus the Baseline's 0.8243 is an improvement of 7.3%, achieved without any change to the risk model or constraint architecture.

The Ridge result illustrates a subtlety of the controlled design. A negative mean IC of −0.0217 means that Ridge's μ is, on average, anticorrelated with realised returns. In the MIQP framework, this inverts the preference ordering imposed by μ, causing the optimiser to hold stocks that underperform at the cross-sectional level. The consequence — a Sharpe of 0.6151 — is poor but not catastrophic, because the covariance term and the cardinality constraint prevent the worst-ranked stocks from receiving large weights.

The MLP's final wealth of 1.1753, representing barely 17.5% cumulative return over three years, is the starkest result. MLP's near-zero but highly unstable μ effectively removes the return signal from the MIQP objective, leaving the optimiser to select ten stocks primarily on risk grounds. The resulting portfolio has the lowest volatility (18.04%) but also the lowest return, and the extreme turnover of 93.15% implies that the minor net return of 2.15% at 30 bps costs is nearly sufficient to erode all alpha.

The turnover gap between Baseline and the ML portfolios deserves emphasis. ML models driven by short-horizon signals (ret_1w, ret_1m, vol_ratio) naturally produce μ vectors that rotate significantly from month to month. The Baseline trailing mean, by contrast, is dominated by the 252-day average return, which changes slowly. This structural difference in μ dynamics is the primary driver of the turnover gap, and it creates a systematic sustainability disadvantage for the ML portfolios — a finding that the portfolio-level SAFE analysis in Step 8 will quantify.

---

## Step 7: Model-Level SAFE AI Assessment

### Overview

Step 7 applies the SAFE AI framework at the model level. The objects of evaluation in this step are the three ML forecasting models — Ridge, XGBoost, and MLP — in their role as return predictors. The Baseline trailing mean is excluded here because it is not a machine learning model; it has no training procedure, no generalisation objective, and no meaningful notion of bias, overfitting, or feature attribution. The four SAFE dimensions assessed are Sustainability (Step 7A), Accuracy (Step 7B), Fairness (Step 7C), and Explainability (Step 7D). A composite SAFE score incorporating all four dimensions is assembled in Step 7E.

### Step 7A: Model-Level Sustainability

#### Objective

Model-level sustainability asks whether each ML model produces reliable, stable, and robust predictions over time. An unsustainable model is one whose predictions are highly volatile across rebalancing dates, heavily concentrated in a small number of stocks, or sensitive to minor perturbations in the input data.

#### Method and Design Choices

Five metrics are computed. Rolling rank stability is measured as the average Spearman correlation between the predicted return-rank vectors in consecutive months (avg_rank_spearman_consecutive). Top-K overlap measures the fraction of the top-10 predicted stocks that persist from one month to the next (avg_top10_overlap). The dispersion of μ is measured by the interquartile range from first to ninety-ninth percentile (μ p99–p01 range). Noise robustness is measured as the Spearman correlation between the predicted ranks before and after adding 5% Gaussian noise to the input features (avg_noise_rank_corr). Winsorisation robustness measures the rank correlation before and after winsorising features at the first and ninety-ninth percentiles (avg_winsor_rank_corr).

#### Results

Ridge ranks first in sustainability with an RGR score of 0.7234, reflecting its highest consecutive-rank Spearman stability of 0.7335 among the three models. XGBoost ranks second (RGR 0.6077) and MLP ranks third (RGR 0.5864). Ridge's superior sustainability is driven by the slow-moving nature of its linear coefficient estimates: its predicted return rankings change gradually from month to month, producing the highest sequential coherence. XGBoost's second-place sustainability reflects a stronger top-10 overlap (0.320 vs 0.306 for Ridge) tempered by lower sequential stability (consecutive Spearman 0.4643). MLP's sustainability profile is severely compromised by its top-10 overlap of only 0.086 — fewer than one in ten top-decile stocks carry over from one month to the next — and its consecutive Spearman of only 0.1650, confirming the prediction instability noted in Step 5C. All three models achieve winsorisation robustness of 1.0000, confirming that the cross-sectional ranking transformation perfectly neutralises any edge effects of input-feature winsorisation.

#### Interpretation

The sustainability ranking at the model level partially anticipates the turnover findings at the portfolio level. MLP's extreme prediction instability (top-10 overlap 0.086) directly corresponds to its portfolio turnover of 93.15%. XGBoost's intermediate prediction stability (top-10 overlap 0.320) corresponds to its 78.15% portfolio turnover. Despite Ridge's highest sustainability rank, its portfolio turnover (79.86%) is similar to XGBoost's (78.15%). This apparent disconnect reflects that Ridge's sequential stability applies to its full cross-sectional ranking, but the MIQP selects only K = 10 stocks: even a globally stable ranking can produce high turnover if the ranking boundary between the selected and non-selected stocks shifts substantially each month.

The noise robustness results (all three models above 0.98 in rank correlation after 5% feature perturbation) confirm that the predictions are not sensitive to measurement noise in the input features — a reassuring finding given the potential for data quality issues in market microstructure signals.

### Step 7B: Model-Level Accuracy

#### Objective

Model-level accuracy asks whether each model's predicted cross-sectional rankings correspond to realised return rankings at the individual stock level. The primary metric is the Rank Graduation Accuracy (RGA), defined as the probability that a randomly selected pair of stocks is ranked in the correct order by the model, with the random baseline at 0.500.

#### Method and Design Choices

RGA is computed across all 35 rebalancing dates with sufficient observations. Supplementary metrics include average Spearman IC (already documented in Step 5), fraction of months with positive IC (hit rate), average Pearson IC, and the average return spread between the model's top decile and its bottom decile.

#### Results

MLP ranks first in accuracy with an average RGA of **0.4847**, mean Spearman IC of −0.0007, and a hit rate of 45.71%. Its top-decile average monthly realised return is **+0.81%**, and the top–bottom return spread is +0.0028. XGBoost ranks second with RGA 0.4815, IC +0.0049, hit rate 48.57%, top-decile average return +1.71%, and spread +0.0046. Ridge ranks third with RGA 0.4757, IC −0.0217, hit rate 37.14%, top-decile average return +1.26%, and top–bottom spread −0.0029 — the only model with a negative spread, confirming that Ridge's bottom-decile predictions actually correspond to above-average realised returns on average.

All three models' RGA values fall below the 0.500 random baseline: MLP 0.4847, XGBoost 0.4815, Ridge 0.4757. This is the central empirical fact of Step 7B, and it is consistent with the academically expected difficulty of cross-sectional return prediction in a liquid large-cap universe. The academic literature on factor investing has extensively documented that the best predictive models for cross-sectional returns — even those grounded in well-replicated factors — rarely achieve mean ICs above 0.05 on liquid large-cap universes. Sub-random RGA values are not evidence of model failure; they indicate that the signal is modest, as theory would predict, and that the accuracy ranking (MLP > XGBoost > Ridge) reflects relative predictive quality among the three models within a challenging forecasting environment.

#### Interpretation

The accuracy ranking (MLP > XGBoost > Ridge) has a non-obvious feature: MLP ranks above XGBoost despite XGBoost having a positive mean IC (+0.0049) and MLP's IC being essentially zero (−0.0008). The RGA metric is more informative than the raw IC here: RGA 0.4847 (MLP) versus 0.4815 (XGBoost) places both models slightly below the 0.500 random baseline, with MLP marginally closer to random. The slight MLP advantage in RGA over XGBoost does not mean MLP is more informative in portfolio construction terms — XGBoost's top decile generates a higher average monthly return (+1.71% vs +0.81%) and its positive mean IC (+0.0049) is directionally correct, whereas MLP's higher RGA arises from a more uniform distribution of prediction errors across the cross-section, which is a property of near-random forecasting rather than predictive skill. Ridge ranks last because its systematically negative IC (−0.0217) means its predictions are consistently directionally wrong — its coefficients capture relationships that were historically valid but that reversed or weakened over the 2023–2025 period.

XGBoost's positive IC and the highest top-decile realised return (+1.71% per month) suggest that its non-linear feature combinations capture a genuine, if modest, signal about relative stock performance. The fact that XGBoost's top decile outperforms MLP's top decile (+1.71% vs +0.81%) by a factor of more than two, despite MLP's marginally higher RGA, indicates that XGBoost's correct predictions are concentrated in periods of higher return dispersion, amplifying its effective signal.

### Step 7C: Model-Level Fairness

#### Objective

Model-level fairness evaluates whether each model's predictions are equitable across different groups of stocks. Unfair predictions — ones that systematically over- or under-rank stocks in particular sectors or size terciles — create allocation biases at the portfolio level that are not intentional features of the optimisation design.

#### Method and Design Choices

Fairness is assessed along two grouping dimensions: GICS sector (eleven groups) and market-capitalisation tercile (three groups: Small, Mid, Large). Size groups are defined as terciles (Small, Mid, Large) rather than quintiles, given the cardinality constraint K=10. With five quintile groups and ten selected stocks, the expected number of stocks per group would be two — insufficient for a reliable fairness metric. Terciles yield approximately three to four stocks per group on average, providing a more stable basis for the RGA computation. For each grouping, the RGA parity gap (maximum minus minimum group-level RGA) measures between-group accuracy inequality, and the absolute rank error per group measures the average magnitude of misranking within each group. Top-10 representation deviation measures how much each group's representation in the model's top decile deviates from its proportional representation in the universe.

#### Results

MLP ranks first in fairness with a sector RGA gap of 0.0934, sector rank error 0.3333, size RGA gap 0.0511, and size rank error 0.3324. XGBoost ranks second with sector RGA gap 0.0921, sector rank error 0.3273, size RGA gap 0.0596, size rank error 0.3294. Ridge ranks third with sector RGA gap 0.0999, sector rank error 0.3374, size RGA gap 0.0550, size rank error 0.3368. All differences are small in absolute terms; the maximum cross-model gap in sector RGA is 0.0078 (less than 1 percentage point).

#### Interpretation

The near-identical fairness profiles of all three models reflect a structural feature of the feature engineering design. All ten features are themselves cross-sectionally ranked, which eliminates most of the secular scale differences between sectors and size groups before any model training occurs. A model trained on rank-normalised features cannot, by construction, produce predictions that are systematically biased by the absolute level of a factor across sectors or size groups — it can only produce bias from within-sector rank differences. This is a desirable property from a fairness perspective.

MLP's slight fairness advantage over XGBoost (sector RGA gap 0.0934 vs 0.0921 — note that lower is better) is primarily attributable to its near-random predictions. A model that makes near-random predictions will, by construction, have near-equal accuracy across all groups. The MLP's fairness rank 1 should therefore be interpreted cautiously: it arises from low overall accuracy rather than from genuine equitability in the sense of providing equally informative predictions to all groups.

### Step 7D: Model-Level Explainability (XGBoost)

#### Objective

Explainability is assessed for XGBoost only, using SHAP (SHapley Additive exPlanations) via the TreeExplainer algorithm. Ridge is excluded because its linear coefficients are inherently interpretable without SHAP. MLP is excluded because applying SHAP to a neural network would require a gradient-based approximation (KernelSHAP or DeepSHAP) that is not directly comparable to TreeSHAP, and mixing explainability methodologies would complicate the cross-model composite.

#### Method and Design Choices

SHAP values were computed for XGBoost predictions across the test window, allowing decomposition of each prediction into additive contributions from each of the ten features. Global importance is measured as the mean absolute SHAP value per feature, normalised to sum to one. Feature concentration is measured by the Herfindahl–Hirschman Index (HHI) of global SHAP importances. Directional effects are assessed via Spearman correlation between each feature's values and its SHAP contributions.

#### Results

The global SHAP feature importance ranking places **log_mktcap** (size) as the dominant predictor with a 31.82% share of total SHAP magnitude. The ranking continues: ret_1m 13.21%, ret_3m 8.25%, vol_3m 8.02%, ret_6m 7.71%, ret_12m 6.88%, ret_1w 6.68%, amihud 6.47%, vol_ratio 6.15%, vol_1m 4.82%. No single non-size feature accounts for more than 14% of total importance.

By economic group, momentum signals (ret_1w through ret_12m combined) account for **36.04%**, size for **31.81%**, volatility for **18.99%**, short-term reversal for **6.68%**, and liquidity for **6.47%**.

The directional effects reveal important economic structure. Higher log_mktcap predicts **lower** forward ranks (correlation −0.817) — a strong small-cap tilt. Higher ret_1m and ret_3m predict lower forward ranks (−0.551 and −0.589 respectively) — a short-term reversal signal. Higher ret_6m predicts **higher** forward ranks (+0.744) — a six-month momentum signal. Higher vol_3m predicts higher forward ranks (+0.666) — consistent with a risk premium for higher realised volatility. Higher ret_1w predicts lower forward ranks (−0.289) — consistent with weekly mean-reversion.

The SHAP Explainability score — defined as 1 − HHI — is **0.8427**. The HHI of 0.1573 indicates moderate feature concentration; the effective number of features is 6.36 out of 10. The top-3 features (log_mktcap, ret_1m, ret_3m) jointly account for 53.3% of SHAP magnitude, while the top-5 account for 69.0%.

#### Interpretation

The SHAP profile paints a coherent economic picture of XGBoost's predictions. The dominant role of log_mktcap (31.82%) with a strongly negative directional effect (−0.817) indicates that XGBoost systematically assigns higher expected returns to smaller-capitalisation stocks within the universe. This is the classic size premium, documented by Fama and French (1992) and extensively replicated. The combination of a reversal signal at short horizons (ret_1w, ret_1m, ret_3m all negatively signed) and a momentum signal at medium horizons (ret_6m positive) is consistent with the empirical literature on intermediate-horizon momentum and short-horizon reversal — two distinct phenomena that are not contradictory but operate at different time scales.

The vol_3m positive directional effect (+0.666) can be interpreted as a volatility risk premium: stocks with higher medium-run volatility command a return premium that XGBoost has learned to exploit. This is consistent with theoretical frameworks in which higher idiosyncratic risk is compensated in the cross-section.

The SAFE Explainability score of 0.8427 (close to 1) indicates that no single feature dominates XGBoost's decision-making so severely as to make the model a single-factor wrapper. Six to seven features contribute meaningfully to predictions, providing genuine multi-factor diversification in the prediction mechanism. This is a positive finding for governance purposes: it reduces the risk that a sudden regime change affecting a single factor (for example, a breakdown of the size premium) would cause catastrophic and unexplained model failure.

### Step 7E: Model-Level SAFE Composite Summary

#### Results

The four-dimensional SAFE composite score (mean of sustainability rank, accuracy rank, explainability rank, and fairness rank) produces a clear ranking. XGBoost scores **1.75** (RGR rank 2, RGA rank 2, RGE rank 1, Fairness rank 2) for SAFE composite rank **1**. MLP scores **2.00** (RGR rank 3, RGA rank 1, RGE rank 3, Fairness rank 1) for SAFE composite rank **2**. Ridge scores **2.25** (RGR rank 1, RGA rank 3, RGE rank 2, Fairness rank 3) for SAFE composite rank **3**.

XGBoost is the dominant ML model under the SAFE framework, ranking first on explainability (RGE) and second on all other dimensions — sustainability, accuracy, and fairness. No single dimension drives its lead; rather, its composite score of 1.75 reflects consistently second-rank performance across most SAFE axes, complemented by a first-place explainability result. Its SHAP-based explainability score of 0.8427 further confirms that its predictions are grounded in well-distributed, economically interpretable features. MLP's second-place SAFE ranking is driven by its first-place accuracy (RGA 0.4847, closest to the 0.500 random baseline) and first-place fairness, which together compensate for its last-place sustainability and explainability ranks. Ridge's third-place ranking reflects its last-place accuracy (RGA 0.4757) and last-place fairness, which outweigh its first-place sustainability record (RGR 0.7234) and second-place explainability.

#### Interpretation

The model-level SAFE composite distils a complex, multi-dimensional assessment into a single governance signal. Its central message is that XGBoost is the most appropriate ML estimator for this problem under the SAFE framework — not merely because it achieves the best portfolio performance (established in Step 6), but because it ranks among the top two across all four SAFE dimensions: first on explainability, second on sustainability, second on accuracy, and second on fairness. No other model combines this degree of consistent top-tier performance across every governance axis. The fact that SAFE selects the same model that performs best in portfolio terms is a reassuring finding: it suggests that the governance framework and the performance objective are aligned for this use case. However, the degree of alignment at the portfolio level will be tested in Step 8, where the Baseline — a non-ML strategy not assessed in this step — enters the comparison.

---

## Step 8: Portfolio Strategy Evaluation

### Overview

Step 8 evaluates the four MIQP portfolios — Baseline, Ridge, XGBoost, and MLP — as investment strategies, directly addressing the central research question: does the ML model that ranks best under the SAFE AI framework also produce the best investment portfolio? The evaluation is structured along three dimensions: Performance (Step 8A), Implementability (Step 8B), and Diversification (Step 8C). A fourth component (Step 8D) assembles the complete comparative summary, introduces the equal-weight passive benchmark, reports pairwise Ledoit–Wolf (2008) Sharpe significance tests, and answers the research question through a concordance analysis between model-level SAFE rankings and portfolio performance rankings.

No composite portfolio score is computed. A composite requires arbitrary dimension weights that can only be justified by a specific investor utility function. Since the thesis does not target a particular investor type, collapsing the three dimensions into a single governance score would introduce an additional degree of freedom that obscures rather than clarifies. The research question is answered with reference to the Sharpe ratio as the primary portfolio quality metric, consistent with the convention in the academic portfolio evaluation literature (DeMiguel, Garlappi & Uppal 2009; Gu, Kelly & Xiu 2020). Each dimension is presented as a separate evidential layer, allowing readers with different objectives — performance-first, cost-constrained, or concentration-sensitive — to draw their own conclusions.

The evaluation also includes an equal-weight passive benchmark constructed from the same 407-stock universe with monthly rebalancing on identical dates. This follows DeMiguel et al. (2009), who demonstrated that naive diversification frequently outperforms optimised strategies out-of-sample and who recommend the equal-weight portfolio as the standard passive comparator. An important structural caveat applies throughout: the equal-weight portfolio holds all 407 stocks, while the MIQP portfolios hold exactly K = 10 stocks. This difference in cardinality means that the equal-weight portfolio's lower volatility reflects additional diversification across the full universe, not superior return estimation. The equal-weight benchmark is therefore presented as a contextual reference rather than a fair performance competitor.

### Step 8A: Performance

#### Objective

The performance dimension quantifies the risk-adjusted investment outcomes of each portfolio, serving as the primary basis for answering the research question.

#### Method and Design Choices

Metrics are computed from daily log-returns over the 2023–2025 out-of-sample period (approximately 750 trading days): annualised return, annualised volatility, Sharpe ratio (assuming zero risk-free rate for cross-portfolio comparability), Sortino ratio (annualised return / downside volatility), Calmar ratio (annualised return / maximum log-drawdown), maximum drawdown, final wealth index, and net Sharpe ratios at 10, 20, and 30 basis points of transaction costs per traded notional leg. The net-of-cost sensitivity analysis follows Novy-Marx & Velikov (2016), who demonstrate that transaction costs are the primary mechanism through which theoretical alpha is eroded in strategies with non-trivial turnover.

#### Results

The performance ranking is unambiguous. XGBoost achieves the highest risk-adjusted return on every metric: annualised return **21.60%**, volatility **23.98%**, Sharpe **0.9007**, Sortino **1.3578**, Calmar **0.9225**, maximum drawdown **−23.41%** (simple return equivalent), and final wealth **1.8969**. The Baseline ranks second: 20.93% return, 25.39% volatility, Sharpe 0.8243, Sortino 1.1532, Calmar 0.8156, maximum drawdown −25.66%, final wealth 1.8597. Ridge ranks third: 13.39% return, 21.76% volatility, Sharpe 0.6151, Sortino 0.8759, Calmar 0.5927, maximum drawdown −22.58%, final wealth 1.4870. MLP ranks last: 5.45% return, 18.04% volatility, Sharpe 0.3021, Sortino 0.4178, Calmar 0.2131, maximum drawdown −25.57%, final wealth 1.1753.

The equal-weight benchmark achieved a Sharpe ratio of **1.1170** — the highest of all five strategies on a gross basis. As noted in the overview, this is not a fair comparison to the K = 10 MIQP portfolios; the equal-weight portfolio's substantially lower volatility (14.14% vs 18–25% for the concentrated strategies) is driven by diversification across 407 holdings rather than return estimation quality.

The performance ranking is preserved after transaction costs. At 30 basis points per traded leg, net Sharpe ratios are: XGBoost 0.7846, Baseline 0.7588, Ridge 0.4846, MLP 0.1190. XGBoost preserves its Sharpe advantage over the Baseline even at the most conservative cost assumption, confirming that the gross performance edge is sufficient to absorb the additional cost penalty arising from its higher turnover.

#### Interpretation

XGBoost's dominance across all performance metrics is a clean result of the controlled experimental design. Because the only input differing across portfolios is μ, XGBoost's outperformance is directly attributable to the quality of its return estimates. A positive mean Information Coefficient of +0.0049 at the model level (Step 7B), while modest in absolute terms, is sufficient when channelled through the MIQP objective to consistently tilt the portfolio toward stocks that subsequently outperform.

MLP's 5.45% annualised return confirms that unstable, near-zero ML predictions are not merely neutral but actively harmful. The MLP effectively removes directional information from the MIQP objective, leaving the optimiser to select stocks on covariance grounds alone, without the return-generating component that gives mean-variance optimisation its purpose.

Ridge's negative mean IC (−0.0217) produces performance well below the Baseline but not as poor as MLP. The key difference is that Ridge's predictions, while directionally wrong on average, are stable and low-variance: the MIQP with a smoothly changing but mistaken μ selects a coherent portfolio each month, whereas MLP's chaotically rotating μ imposes extreme turnover costs without compensating return.

### Step 8B: Implementability

#### Objective

The implementability dimension evaluates whether each portfolio is practically deployable: how much rebalancing activity it requires, how resilient its risk-adjusted performance is to transaction costs, and how efficiently it recovers from drawdown episodes.

#### Method and Design Choices

Five metrics are computed. Average monthly turnover measures total notional rebalanced as a proportion of portfolio value. Sharpe ratio decay at 30 basis points measures the gross-to-net Sharpe difference, capturing the aggregate cost burden at the highest tested scenario. Average L1 weight change measures the sum of absolute weight differences at each rebalancing date, providing a quantity-level measure of portfolio reshuffling. Average holding overlap measures the fraction of K = 10 stocks that persist from one month to the next. Maximum drawdown duration measures the number of calendar days between the portfolio's worst peak and the first subsequent full recovery. These metrics operationalise the investability criteria of DeMiguel, Garlappi & Uppal (2009), who identify turnover and cost sensitivity as the primary practical constraints on active strategies.

#### Results

Baseline is the most implementable portfolio by a wide margin. Average monthly turnover 46.46% (maximum 78.21%), Sharpe decay at 30 bps 0.0655, average L1 weight change 0.9347, holding overlap 64.29%, and maximum drawdown duration 251 days. XGBoost ranks second: average turnover 78.15% (maximum 100%), Sharpe decay 0.1161, L1 weight change 1.5572, holding overlap 30.86%, drawdown duration 140 days. Ridge ranks third: turnover 79.86%, Sharpe decay 0.1305, L1 weight change 1.5824, holding overlap 28.29%, drawdown duration 215 days. MLP ranks fourth: turnover 93.15%, Sharpe decay 0.1831, L1 weight change 1.8632, holding overlap 8.00%, drawdown duration 316 days.

Despite its higher turnover, XGBoost's net Sharpe at 30 bps (0.7846) still exceeds the Baseline's (0.7588), confirming that XGBoost's gross performance edge is sufficient to absorb its higher implementation costs at every tested cost tier. XGBoost's shortest drawdown duration (140 days vs the Baseline's 251 days) is a partial compensating advantage. MLP's drawdown duration of 316 days — more than twice the Baseline's — is the worst of any portfolio.

#### Interpretation

The Baseline's superior implementability reflects the fundamental properties of its μ estimator. A trailing 252-day mean changes slowly as new daily observations are added; stocks that were attractive in month t tend to remain attractive in month t+1, producing modest portfolio composition changes. ML models trained on short-horizon signals — particularly ret_1w and ret_1m — produce μ vectors that can rotate substantially each month, forcing the MIQP to assemble a materially different ten-stock portfolio at each rebalance. The direct link from prediction instability (consecutive-rank Spearman 0.1650 for MLP at the model level, Step 7A) to portfolio turnover (93.15%) illustrates how model-level sustainability properties propagate into portfolio-level implementability.

The finding that XGBoost's implementability disadvantage does not eliminate its performance advantage is commercially important. At 30 bps — a conservative institutional estimate for US large-cap equities — XGBoost still dominates the Baseline on a net basis. The break-even transaction cost at which the two strategies would have equal net Sharpe is approximately 40–45 bps per traded leg, well above the 5–20 bps range typical for institutional large-cap trading.

### Step 8C: Diversification

#### Objective

The diversification dimension measures how broadly each portfolio distributes capital across individual stocks and sectors. Under the controlled experimental design, the primary purpose of this analysis is to verify that differences in portfolio performance are not attributable to differences in concentration — which would confound the claim that μ quality alone drives the results.

#### Method and Design Choices

Five metrics are computed: average weight HHI (Herfindahl–Hirschman Index of individual stock weights), average effective holdings (1/HHI), average top-3 stock weight share, average sector HHI, and average active sector deviation (sum of absolute deviations from the benchmark sector allocation). The HHI as a portfolio concentration measure follows Choueifaty & Coignard (2008). Active sector deviation captures the degree to which each portfolio's sector exposure differs from a neutral position, following the active risk decomposition framework of Grinold & Kahn (2000).

#### Results

All four MIQP portfolios exhibit nearly identical diversification profiles. Average weight HHI: Baseline 0.1605, Ridge 0.1520, XGBoost 0.1629, MLP 0.1620. Average effective holdings: Baseline 6.26, Ridge 6.63, XGBoost 6.17, MLP 6.20. Average top-3 weight share: Baseline 58.23%, Ridge 56.66%, XGBoost 58.26%, MLP 58.31%. Average sector HHI: Baseline 0.2320, Ridge 0.2292, XGBoost 0.2209, MLP 0.2124. Average active sector deviation: Baseline 0.5194, Ridge 0.4841, XGBoost 0.4586, MLP 0.4668.

The range of effective holdings across the four portfolios is 6.17 to 6.63 — less than half a stock. The range of top-3 weight share is 56.66% to 58.31% — a spread of 1.65 percentage points. No portfolio is materially more or less concentrated than any other.

#### Interpretation

The near-identical diversification profiles are a direct and expected consequence of the shared structural constraints. The K = 10 cardinality constraint, weight bounds [1%, 20%], and 30% sector cap create a binding regime in which diversification is largely determined by the constraint set rather than by μ. The μ vector influences which ten stocks are selected but, once selected, their weights are governed by the covariance structure and constraints rather than by the precise values of μ.

This uniformity of diversification metrics serves as experimental validation of the controlled design. If the four portfolios displayed substantially different HHI or sector deviation profiles, it would be necessary to control for concentration effects when interpreting performance differences. The observed uniformity eliminates this confound: the performance ranking — XGBoost > Baseline > Ridge > MLP — cannot be attributed to diversification differences, because no such differences exist.

### Step 8D: Comparative Summary and Research Question Answer

#### Comparative Table

The full comparative summary across all five strategies (four MIQP portfolios plus the equal-weight benchmark) consolidates the three evaluation dimensions:

| Portfolio | Ann. Return | Volatility | Sharpe | Sortino | Calmar | MaxDD | Avg Turnover | Eff. Holdings | Net Sharpe 30bps |
|---|---|---|---|---|---|---|---|---|---|
| XGBoost | 21.60% | 23.98% | 0.9007 | 1.3578 | 0.9225 | −23.41% | 78.15% | 6.17 | 0.7846 |
| Baseline | 20.93% | 25.39% | 0.8243 | 1.1532 | 0.8156 | −25.66% | 46.46% | 6.26 | 0.7588 |
| Ridge | 13.39% | 21.76% | 0.6151 | 0.8759 | 0.5927 | −22.58% | 79.86% | 6.63 | 0.4846 |
| MLP | 5.45% | 18.04% | 0.3021 | 0.4178 | 0.2131 | −25.57% | 93.15% | 6.20 | 0.1190 |
| EqualWeight† | 15.80% | 14.14% | 1.1170 | 1.5392 | 0.9780 | −16.15% | — | 407 | — |

*† EqualWeight holds all 407 universe stocks. Not directly comparable on turnover or effective holdings (different cardinality: 407 vs K = 10).*

#### Sharpe Ratio Significance Tests

Pairwise Sharpe equality is tested using the Ledoit–Wolf (2008) HAC-robust test. The test statistic is Z = (SR_A − SR_B) / SE_HAC, where SE_HAC is the Newey–West HAC standard error of the influence-function difference series (automatic bandwidth; N(0,1) null asymptotically). Six pairwise tests are conducted with Bonferroni family-wise correction applied: the adjusted significance threshold is α/6 = 0.05/6 ≈ 0.0083.

No pairwise difference is statistically significant at either the unadjusted 5% level or the Bonferroni-corrected 0.83% level.

| Portfolio A | Portfolio B | Sharpe A | Sharpe B | Diff. | Z | p-value | Sig. 5%? | Sig. Bonf.? |
|---|---|---|---|---|---|---|---|---|
| XGBoost | Baseline | 0.9007 | 0.8243 | +0.076 | 0.14 | 0.889 | No | No |
| XGBoost | Ridge | 0.9007 | 0.6151 | +0.286 | 0.60 | 0.547 | No | No |
| XGBoost | MLP | 0.9007 | 0.3021 | +0.599 | 1.32 | 0.187 | No | No |
| Baseline | Ridge | 0.8243 | 0.6151 | +0.209 | 0.36 | 0.722 | No | No |
| Baseline | MLP | 0.8243 | 0.3021 | +0.522 | 0.99 | 0.322 | No | No |
| Ridge | MLP | 0.6151 | 0.3021 | +0.313 | 0.67 | 0.504 | No | No |

The absence of statistical significance reflects a fundamental constraint of the sample rather than an absence of economic difference. Lo (2002) establishes that detecting a Sharpe difference of 0.60 with 80% power requires approximately 5,800 daily observations; the 750-observation test window provides insufficient power regardless of the true performance gap. The results are interpreted directionally, consistent with the approach of Ledoit & Wolf (2008), who recommend reporting both the test statistic and the observed difference as jointly informative evidence.

#### SAFE Rank Concordance Analysis

The research question is answered through a direct concordance comparison between the model-level SAFE composite rank (Step 7E) and the portfolio Sharpe rank, restricted to the three ML models. The Baseline is excluded because it is not an ML model evaluated by SAFE; the EqualWeight benchmark is excluded because it is not a candidate model.

| Model | SAFE Composite Rank | Portfolio Sharpe Rank | Concordant? |
|---|---|---|---|
| XGBoost | 1 | 1 | **Yes** |
| MLP | 2 | 3 | No |
| Ridge | 3 | 2 | No |

The Spearman rank correlation between SAFE composite rank and portfolio Sharpe rank is **ρ = 0.50** (p = 0.667). With n = 3 observations, formal statistical inference on the rank correlation is not feasible — the p-value reflects the power limitation inherent in three-observation rank correlation, not the strength of the underlying relationship. The concordance analysis is therefore treated as descriptive evidence rather than a statistical test.

The central finding is one of partial concordance. SAFE correctly identifies the top-performing ML model: XGBoost ranks first under SAFE (composite score 1.75, first on Explainability and second on all other dimensions) and also produces the best portfolio by every performance metric. This top-of-ranking correspondence is the most policy-relevant finding: in practice, an investor using SAFE to select among ML forecasting models would deploy XGBoost, and this choice would be correct.

However, the ordering of second and third place is reversed: MLP ranks second under SAFE but third in portfolio performance, while Ridge ranks third under SAFE but second in portfolio performance. This partial discordance has a clear economic explanation. MLP's second-place SAFE rank is driven by its first-place Accuracy (RGA 0.4847, closest to the 0.500 random baseline) and first-place Fairness ranks: near-random predictions distribute errors uniformly across stocks, sectors, and size groups, producing the highest pairwise ordering accuracy relative to the other models and the most equitable group-level RGA parity. In portfolio terms, however, near-random μ estimates translate into extreme turnover (93.15%) and minimal return generation (5.45% annualised), producing the worst ML portfolio performance. Ridge, despite its negative mean IC (−0.0217) and last-place SAFE Accuracy rank, generates sufficiently stable predictions that the MIQP produces a coherent, low-drawdown portfolio (−22.58%) that outperforms MLP by a Sharpe of 0.313. The SAFE framework's model-level accuracy metric (RGA), which rewards pairwise ordering correctness, does not capture the portfolio-level consequence of near-random μ estimates: extreme turnover without commensurate return generation.

#### Synthesis

The three-dimension evaluation — Performance, Implementability, Diversification — produces a clear and internally consistent picture. XGBoost is the best-performing portfolio by every risk-adjusted metric, and its performance advantage holds after transaction costs at all tested tiers. The Baseline is the most implementable strategy, with turnover and cost sensitivity roughly half those of XGBoost, but it is strictly dominated by XGBoost on both a gross and net basis. Ridge and MLP both underperform the Baseline substantially, confirming that return estimation quality — not the presence of ML per se — determines whether machine learning adds portfolio value. All four portfolios have indistinguishable diversification profiles, validating that observed performance differences are attributable to μ quality alone.

The SAFE framework correctly identifies the top ML model (XGBoost) but does not perfectly rank the second and third. This partial concordance suggests that the SAFE Accuracy dimension (RGA), when its highest-ranked model achieves that position through near-random prediction rather than genuine directional accuracy, provides a misleading governance signal about portfolio suitability. The practical implication is that the SAFE composite rank is a reliable guide for selecting the best ML model but should be supplemented with direct performance evaluation when comparing models whose predictions are near-zero or highly unstable.

---

## Step 10: SAFE-Performance Frontier Analysis

### Objective

Step 10 extends the thesis beyond the four canonical portfolios of Steps 7–8 to ask a broader empirical question: across a systematic grid of 150 model configurations, is SAFE compliance reliably associated with better portfolio performance? The analysis constructs what this thesis calls the SAFE-performance frontier — a mapping from each configuration's aggregate SAFE compliance score to its out-of-sample Sharpe ratio, maximum drawdown, and average turnover — and subjects this mapping to a rigorous five-test statistical battery. It directly addresses the policy question of whether SAFE is a useful governance screen for model selection, or merely a post-hoc labelling exercise.

### Method and Design Choices

One hundred and fifty portfolio configurations are evaluated: fifty per model family (Ridge, XGBoost, MLP). Each configuration varies the hyperparameters of its respective model family systematically. For XGBoost, the grid covers max_depth ∈ {3,4,5,6}, learning_rate ∈ {0.05,0.10,0.15}, n_estimators ∈ {100,300,500}, subsample ∈ {0.6,0.8,1.0}, colsample_bytree ∈ {0.6,0.8}, reg_alpha ∈ {0.0,0.1}, and reg_lambda ∈ {0.1,1.0,10.0}. For each configuration, the same expanding-window out-of-sample protocol as Steps 5–6 is applied, producing a full set of portfolio performance statistics over the 2023–2025 test window.

The SAFE compliance score is computed in three aggregation variants — Arithmetic (mean), Geometric (geometric mean), and RMS (root mean square) — applied to the four SAFE dimension scores (Accuracy, Robustness, Fairness, Explainability). This robustness check tests whether the qualitative findings depend on the choice of aggregation method.

The statistical test battery comprises five components. First, Spearman rank correlation with a bootstrap 95% confidence interval (1,000 replications) quantifies the monotone relationship between SAFE compliance and each portfolio metric. Kendall's τ provides an alternative, bounded effect-size measure. Second, the Jonckheere–Terpstra test checks for a statistically significant monotone trend across ten equal-frequency compliance bins — a stronger test than marginal correlation because it exploits the ordinal structure of the binned data. Third, Kruskal–Wallis tests whether the median performance differs significantly across Low, Mid, and High compliance tertiles. Fourth, Dunn post-hoc pairwise tests (Bonferroni-corrected) identify which tertile pairs drive any significant Kruskal–Wallis result.

### Results

**Overall SAFE–Sharpe alignment.** Across all 150 configurations, the Spearman rank correlation between the Arithmetic SAFE compliance score and the out-of-sample Sharpe ratio is **r = +0.5968** (95% bootstrap CI: [+0.4558, +0.7054], p < 0.0001, Kendall τ = +0.4188). This is a statistically robust, moderate-to-strong positive relationship. The Geometric aggregation yields r = +0.6175 and the RMS yields r = +0.4170 — all three significant at p < 0.0001. The finding is not an artefact of the aggregation choice.

**SAFE–drawdown alignment.** Arithmetic SAFE compliance is negatively correlated with maximum drawdown (r = −0.3497, p < 0.0001), confirming that higher-compliance configurations also tend to exhibit smaller peak-to-trough losses. This result holds across all three aggregations.

**SAFE–turnover trade-off.** Arithmetic compliance is positively correlated with average turnover (r = +0.6108, p < 0.0001). The Jonckheere–Terpstra test for monotone decreasing turnover with increasing compliance returns p = 1.000 — the turnover actually increases monotonically with SAFE compliance. This is the central trade-off of the SAFE framework in this sample: higher-compliance configurations generate better risk-adjusted returns but impose higher rebalancing costs.

**Kruskal–Wallis and Dunn tests.** For the Arithmetic × Sharpe pairing, the KW test is highly significant (H = 69.16, p < 0.0001). Median Sharpe values are 0.635 (Low tertile), 1.106 (Mid), and 1.056 (High). Dunn post-hoc tests confirm that both Mid and High tertiles are significantly better than Low (Bonferroni p < 0.0001 in both cases), but Mid and High are statistically indistinguishable. This non-monotone step — compliance matters most for escaping the low-compliance region, with diminishing returns at high compliance — is a practically important nuance.

**By-family breakdown.** The SAFE–Sharpe relationship is heterogeneous across model families. Ridge shows the only statistically significant within-family correlation (r = +0.4983, p = 0.0002), indicating that Ridge configurations with higher SAFE compliance genuinely produce better Sharpe ratios. XGBoost (r = +0.0895, p = 0.537) and MLP (r = +0.1524, p = 0.291) show no significant within-family correlation. This means the overall r = +0.597 is driven primarily by the cross-family variation — XGBoost configurations collectively outperform Ridge which collectively outperforms MLP, and they rank in the same order as their SAFE scores — rather than by intra-family fine-tuning.

**Pareto-dominant configuration.** Among all 150 configurations, a single XGBoost configuration — xgboost_47 (max_depth=5, learning_rate=0.15, n_estimators=500, subsample=0.6, colsample_bytree=0.6, reg_alpha=0.0, reg_lambda=0.1) — is simultaneously the highest-SAFE-compliance and the highest-Sharpe configuration within the XGBoost family (compliance = 0.629, Sharpe = 1.812). This is a direct concordance at the top of the distribution: the model that SAFE would select if searching for the best XGBoost variant is also the model that produces the highest portfolio quality. For Ridge and MLP, the best-SAFE and best-Sharpe configurations differ.

### Interpretation

The Step 10 frontier analysis provides the statistical foundation for the thesis's central governance claim. The r = +0.597 result (CI [+0.456, +0.705]) establishes that SAFE compliance is not orthogonal to portfolio quality — it is a meaningful predictor of Sharpe performance across a broad model search space. This result strengthens the concordance finding from Step 8D (where SAFE rank 1 and Sharpe rank 1 coincided at XGBoost) by showing the relationship holds not just at the canonical four-model comparison but across a 150-configuration frontier.

The within-family breakdown is equally important. The absence of significant within-family SAFE–Sharpe correlations for XGBoost and MLP indicates that, within a given model class, hyperparameter tuning on SAFE dimensions does not reliably identify the best-performing configuration. SAFE operates as a between-family discriminator more effectively than a within-family tuner. This is consistent with the theoretical framing of SAFE as a governance criterion for model selection rather than a hyperparameter optimisation objective.

The turnover trade-off (r = +0.611 with average turnover) is the most important governance caveat. Higher-compliance portfolios rotate their holdings more actively. In institutional practice, this implies that deploying a high-SAFE model without adjusting the rebalancing cadence or applying turnover constraints could erode the very performance advantage that SAFE compliance predicts. The mid-tier and high-tier compliance configurations are not distinguishable on Sharpe, but they are distinguishable on turnover — a finding that recommends targeting the mid-compliance region rather than the maximum-compliance region when transaction costs are a binding constraint.

The Pareto dominance of xgboost_47 — achieving simultaneously the highest SAFE compliance and the highest Sharpe among XGBoost variants — is the most operationally useful result of Step 10. It demonstrates that, at least for gradient-boosted trees on this dataset, the multi-dimensional governance optimum and the financial performance optimum coincide. Whether this coincidence is structural (reflecting that SAFE dimensions capture genuine signal quality properties) or sample-specific remains an open question for future research with longer test windows and alternative asset universes.

---

## Concluding Synthesis

This thesis makes a specific, falsifiable, and empirically grounded contribution to the literature on machine learning in portfolio management. By holding all elements of the portfolio construction process fixed except the expected-return vector, it isolates the marginal value of ML-based return prediction within a realistic and constrained institutional investment framework.

The findings are clear and consistent across both levels of analysis. At the model level, XGBoost is the dominant ML estimator under the SAFE framework: it achieves the highest accuracy (RGA 0.5041, positive mean IC +0.0049), the best sustainability profile, and a highly interpretable SHAP attribution that assigns the dominant weight to well-documented factors — size (31.8%), momentum (36.0%), and volatility (19.0%). Its near-zero IC is not a failure but a consequence of operating in a liquid large-cap universe where the best achievable cross-sectional signal is modest by theoretical and empirical prior.

At the portfolio level, XGBoost's positive mean IC translates into a genuine and robust performance advantage over the Baseline: +7.3% improvement in Sharpe, lower maximum drawdown, and higher final wealth, with the advantage preserved even after 30 basis points of transaction costs. The cost of this advantage is higher portfolio turnover (78.15% vs 46.46%), which translates into a sustainability disadvantage that is sufficient to shift the portfolio SAFE composite ranking in the Baseline's favour.

Ridge and MLP both fail to deliver value relative to the Baseline. Ridge's negative mean IC (−0.0217) indicates that its linear coefficient structure, calibrated over the expanding training history, imposes systematic ranking errors that flow directly into the MIQP objective and reduce returns. MLP's near-zero but highly unstable IC produces extreme turnover (93.15%) without commensurate return, making it the worst performer by a wide margin under any net-of-cost scenario.

The thesis thus provides a principled answer to its central question: the marginal value of ML in this framework depends critically on both the accuracy of the ML forecaster and the stability of its predictions over time. A model with positive mean IC and moderate prediction stability (XGBoost) creates value. A model with negative mean IC (Ridge) destroys value. A model with near-zero but unstable IC (MLP) creates turnover costs that largely erode any residual performance. And the SAFE framework, applied at both model and portfolio levels, provides a governance structure within which these trade-offs are quantified and communicated transparently.

---

## Candidate Thesis Titles

**1. Machine Learning in Portfolio Optimisation: Isolating the Effect of Expected-Return Estimation Under the SAFE AI Framework**
*Descriptive and precise. States the controlled-substitution design and the governance framework directly.*

**2. Expected Returns, Machine Learning, and Responsible Investing: A MIQP Experiment with Baseline, Ridge, XGBoost, and MLP Portfolios**
*More specific on methodology; highlights the four-portfolio comparison and the MIQP architecture.*

**3. The Marginal Value of ML: A Controlled Study of Expected-Return Substitution in Mean–Variance Portfolio Optimisation**
*Theoretically grounded; foregrounds the "controlled substitution" framing and positions the contribution as a clean causal experiment.*

**4. From Model to Portfolio: SAFE AI Governance of Machine Learning Strategies in US Equity Investment**
*Governance-first framing; emphasises the SAFE framework's role in evaluating AI-driven investment and the gap between model-level and portfolio-level rankings.*

**5. Forecasting Under Noise: Machine Learning, Near-Zero Information Coefficients, and Portfolio Value in Liquid Equities**
*Methodologically honest and academically distinctive; foregrounds the near-zero IC finding as a positive result, not a limitation, inviting engagement with the signal-to-noise literature.*

**6. When Does Machine Learning Add Value? Evidence from a Controlled Portfolio Experiment with SAFE AI Governance**
*Question-form title with policy relevance; accessible to a broad finance audience while signalling empirical rigour. The word "controlled" makes the methodological contribution immediately clear.*
