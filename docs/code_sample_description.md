# Code Sample Description

This document is intended for a technical reviewer assessing this repository as a code sample. It explains what the code demonstrates, how the pipeline is organised, which files are most relevant for different areas of review, and what is absent because of data licensing constraints.

---

## What the code sample demonstrates

The repository contains the Python implementation of a quantitative finance research pipeline built for an MSc thesis in Quantitative Finance (University of Pavia, supervisor: Prof. Paolo Giudici). It covers:

1. **Financial data management** — loading, cleaning, and structuring a Bloomberg equity dataset (prices, total return index, market cap, volume, metadata) across approximately 400 large-cap US equities over 2010–2025.
2. **Return computation and covariance estimation** — daily log returns from total return index data; full-window and rolling covariance matrices using sample estimation, Ledoit-Wolf shrinkage, and Oracle Approximating Shrinkage (OAS).
3. **Portfolio optimisation** — constrained Markowitz mean-variance optimisation formulated as a mixed-integer quadratic program (MIQP), solved with Gurobi. Constraints include cardinality (K = 10 stocks), weight bounds (1%–20% per stock), and a sector cap (30%).
4. **Machine learning return forecasting** — three models trained in an expanding-window out-of-sample framework: Ridge regression, XGBoost (gradient boosting), and a multi-layer perceptron. Features are built from cross-sectional momentum, volatility, liquidity, and market cap signals, ranked at each rebalancing date to reduce look-ahead bias.
5. **Out-of-sample backtesting** — ML-predicted expected returns replace the historical mean in the MIQP, and four portfolio variants (baseline, Ridge, XGBoost, MLP) are compared on risk-adjusted performance metrics (Sharpe, Sortino, Calmar, maximum drawdown, turnover).
6. **SAFE AI evaluation** — an adaptation of the SAFE AI framework (Giudici 2024) assesses the investment pipeline along four dimensions: Sustainability/Security, Accuracy, Fairness, and Explainability.

The code is written in Python 3.11 and uses pandas, numpy, scikit-learn, XGBoost, scipy, matplotlib, and Gurobi.

---

## Pipeline structure

| Step | Script | Content |
|------|--------|---------|
| 1 | `step1_load_data.py` | Load and clean raw Bloomberg Excel files |
| 2 | `step2_preprocess_returns.py` | Compute log returns from TRI; estimate diagnostic covariance matrices |
| 3 | `step3_baseline_portfolio.py` | Markowitz MIQP baseline with rolling Ledoit-Wolf covariance and historical mean |
| 3.1 | `step3_1_visualize_baseline.py` | Baseline performance charts (cumulative returns, Sharpe, drawdown, turnover, sector heatmap) |
| 3.2 | `step3_2_visualize_frontier.py` | Efficient frontier visualisation |
| 4 | `step4_feature_engineering.py` | Cross-sectional ML panel with ranked features |
| 5a–5c | `step5a/b/c_train_*.py` | Train Ridge, XGBoost, and MLP in expanding-window OOS framework |
| 6a–6c | `step6a/b/c_portfolio_*.py` | ML-enhanced MIQP portfolios (same optimiser as step 3, ML mu replacing historical mean) |
| 6d | `step6d_compare_portfolios.py` | Aggregate and rank results across all four portfolios |
| 6e | `step6e_visualize_comparison.py` | Side-by-side comparison charts (cumulative returns, drawdown, rolling Sharpe, sector allocation) |

Scripts must be run in order from the repository root. Each script's docstring specifies its exact inputs and outputs, so the data flow is traceable without executing the code.

---

## Suggested review path

For a reviewer assessing Python and ML skills in a quantitative finance context, the following order is suggested:

1. **README.md** — project overview, research question, methodology, and how to run the pipeline.
2. **`src/step4_feature_engineering.py`** — financial feature construction on a stock-month panel; shows momentum, volatility, and liquidity signal design with look-ahead-free cross-sectional ranking.
3. **`src/step5b_train_xgboost.py`** — XGBoost return forecasting; shows the expanding-window OOS loop, 4-fold cross-validation, feature scaling, information coefficient computation, and feature importance extraction.
4. **`src/step3_baseline_portfolio.py`** — core portfolio construction; shows the MIQP formulation using Gurobi's MVar API, rolling covariance estimation, the backtest loop, and performance metric computation. This is the methodologically central script.
5. **`src/step6b_portfolio_xgboost.py`** — ML-enhanced portfolio; structurally identical to step 3 but with XGBoost predictions as the expected return input, which makes the experimental comparison explicit.
6. **`src/step6d_compare_portfolios.py`** — aggregates results across all four portfolios and computes the SAFE AI metrics.
7. **`docs/methodology_summary.md`** — concise description of all methodological choices, including the feature table, MIQP formulation, and performance metrics definitions.

---

## Areas of the code most relevant for Generali Asset Management

| Area | Relevant scripts |
|------|-----------------|
| Financial econometrics (return series, covariance estimation) | `step1`, `step2` |
| Portfolio construction (MIQP, constraints, backtesting) | `step3`, `step6a–6c` |
| ML forecasting (regression, gradient boosting, neural network) | `step5a–5c` |
| Feature engineering (cross-sectional financial signals) | `step4` |
| Performance and risk analysis | `step6d`, `step6e` |
| SAFE AI framework evaluation | `step6d`, `docs/safe_ai_framework.md` |

---

## What is excluded because of data licensing

The Bloomberg dataset cannot be shared publicly because of licensing restrictions. The following files are not included in the repository:

- `data/raw/*.xlsx` — raw Bloomberg files (prices, TRI, market cap, volume, metadata)
- `data/clean/*.csv` — processed data derived from Bloomberg (returns matrix, covariance matrices, feature panel)
- `data/predictions/*.csv` — ML model predictions derived from the above
- `data/results/` — portfolio weights, returns, and performance metrics generated during pipeline execution
- `data/figures/` — charts produced by the pipeline

To understand the pipeline without running it, read the scripts in order. Each docstring describes the exact inputs expected and the files produced, so the data flow can be followed without data access. `data/README_data.md` describes the expected file structure and how to replicate the pipeline with an equivalent dataset.

---

## How to understand the code without Bloomberg data

The docstrings at the top of each script describe the exact inputs, outputs, and methodology. Reading them alongside this document gives a clear picture of what each script does.

If you want to follow the portfolio construction logic, the key scripts are `step3_baseline_portfolio.py` and `step6a_portfolio_ridge.py`. The latter differs from the baseline only in the mu input, which makes the ML substitution explicit and easy to isolate.

To reproduce the pipeline with equivalent data, you need a dataset with the structure described in `data/README_data.md`. The code does not depend on any Bloomberg-specific format beyond what is described in `step1_load_data.py`.
