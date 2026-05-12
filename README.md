# Machine Learning-Enhanced Portfolio Optimization with SAFE AI Evaluation

MSc thesis project — Quantitative Finance, University of Pavia  
Supervisor: Prof. Paolo Giudici

---

## Overview

This repository contains the code developed for my MSc thesis in Quantitative Finance at the University of Pavia. The project studies whether machine learning-based expected return forecasts can improve constrained Markowitz portfolio construction compared with a classical historical-mean benchmark. The pipeline is then evaluated through an adaptation of the SAFE AI framework (Giudici 2024), which assesses portfolio systems along four dimensions: Sustainability/Security, Accuracy, Fairness, and Explainability.

The empirical analysis uses a Bloomberg dataset of approximately 400 large-cap US equities over the period 2010–2025, with out-of-sample portfolio evaluation from January 2023 to December 2025.

**This is research code developed for an academic thesis. It is not production software and some components are still being refined.**

---

## Motivation

Mean-variance portfolio construction is well understood in theory, but its practical implementation involves a series of estimation and modelling choices that are easy to underestimate. Expected returns in particular are notoriously difficult to estimate: the classical trailing mean is noisy and slow to adapt, and the literature has long documented that small improvements in return forecasting can translate into meaningful improvements in portfolio efficiency.

Machine learning offers a different approach. Rather than modelling expected returns directly, the models in this pipeline rank stocks cross-sectionally based on momentum, volatility, and liquidity signals -- signals that have known empirical support in the asset pricing literature. The question the thesis addresses is simple: does replacing the historical mean with ML-ranked expected returns produce better out-of-sample portfolios, after controlling for turnover and transaction costs?

The SAFE AI layer adds a second angle. ML models are often evaluated only on predictive accuracy. In a portfolio context, accuracy alone is insufficient: a model that predicts well but produces fragile, concentrated, or unexplainable allocation decisions is not useful in practice. SAFE AI provides a more complete evaluation framework, and adapting it to portfolio systems is one of the methodological contributions of the thesis.

---

## Project status

This repository is a code sample from my ongoing MSc thesis project in Quantitative Finance at the University of Pavia.

The current version implements the main quantitative pipeline: data cleaning, feature engineering, machine learning-based expected return forecasting, constrained Markowitz portfolio construction, and out-of-sample portfolio evaluation.

The next stage of the thesis is the integration of the SAFE AI evaluation framework. The repository already documents how SAFE AI is intended to be adapted to the portfolio optimization setting, but the empirical implementation of the SAFE dimensions is still under development.

Raw Bloomberg data, derived datasets, portfolio outputs and figures are not included because of data licensing restrictions.

---

## Research question

Can machine learning models (Ridge regression, XGBoost, MLP) produce expected return estimates that lead to better out-of-sample portfolio performance than trailing historical means, within a constrained Markowitz MIQP framework?

---

## Methodology

The pipeline runs in six numbered steps:

1. **Data loading and cleaning** — Load raw Bloomberg files (prices, total return index, market cap, volume, metadata) and apply structural cleaning.
2. **Return computation and covariance estimation** — Compute daily log returns from total return index data. Estimate full-window diagnostic covariance matrices (sample, Ledoit-Wolf, OAS). Rolling covariance matrices for the optimizer are re-estimated at each rebalancing date.
3. **Baseline portfolio** — Construct the benchmark portfolio using trailing historical mean expected returns and rolling Ledoit-Wolf covariance, solved via a mixed-integer quadratic program (MIQP) with cardinality and sector constraints (K=10 stocks, 1%–20% per stock, 30% sector cap). Solved with Gurobi. Also produces an efficient frontier visualization and baseline performance diagnostics.
4. **Feature engineering** — Build a monthly cross-sectional panel of momentum, volatility, liquidity, and market cap features. Features are ranked cross-sectionally at each rebalancing date to reduce look-ahead bias.
5. **ML return forecasting** — Train three models in an expanding-window out-of-sample framework: Ridge regression, XGBoost, and a multi-layer perceptron. Each model predicts next-month cross-sectional return ranks, which are then rescaled to the baseline mu distribution. No portfolio optimization is performed at this step.
6. **ML-enhanced portfolios and evaluation** — Run the same MIQP optimization as the baseline, replacing the historical mean with each ML model's predictions. Compare all four portfolios (baseline, Ridge, XGBoost, MLP) on out-of-sample performance metrics and through the SAFE AI framework.

---

## Repository structure

```
safe-ai-portfolio-optimization/
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   └── config_template.yaml        # configuration template (copy to config.yaml)
├── src/
│   ├── step1_load_data.py
│   ├── step2_preprocess_returns.py
│   ├── step3_baseline_portfolio.py
│   ├── step3_1_visualize_baseline.py
│   ├── step3_2_visualize_frontier.py
│   ├── step4_feature_engineering.py
│   ├── step5a_train_ridge.py
│   ├── step5b_train_xgboost.py
│   ├── step5c_train_mlp.py
│   ├── step6a_portfolio_ridge.py
│   ├── step6b_portfolio_xgboost.py
│   ├── step6c_portfolio_mlp.py
│   ├── step6d_compare_portfolios.py
│   └── step6e_visualize_comparison.py
├── docs/
│   ├── methodology_summary.md
│   ├── safe_ai_framework.md
│   └── code_sample_description.md
└── data/
    ├── README_data.md               # describes expected data format
    ├── raw/                         # Bloomberg raw files — gitignored (proprietary)
    ├── clean/                       # cleaned and processed data — gitignored
    ├── results/                     # portfolio optimization outputs — gitignored
    └── figures/                     # charts produced by the pipeline — gitignored
```

---

## How to run the pipeline

### Requirements

- Python 3.11+
- A valid Gurobi licence (steps 3 and 6 use MIQP optimization). Academic licences are available from [gurobi.com/academia](https://www.gurobi.com/academia/).
- Bloomberg raw data files (not included — see `data/README_data.md`)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

### Running the steps

Scripts are in `src/` and should be run from the **repository root**:

```bash
python src/step1_load_data.py
python src/step2_preprocess_returns.py
python src/step3_baseline_portfolio.py
python src/step3_1_visualize_baseline.py   # optional: baseline diagnostic charts
python src/step3_2_visualize_frontier.py   # optional: efficient frontier chart
python src/step4_feature_engineering.py
python src/step5a_train_ridge.py
python src/step5b_train_xgboost.py
python src/step5c_train_mlp.py
python src/step6a_portfolio_ridge.py
python src/step6b_portfolio_xgboost.py
python src/step6c_portfolio_mlp.py
python src/step6d_compare_portfolios.py
python src/step6e_visualize_comparison.py
```

Each step reads from and writes to `data/` directories using paths relative to the repository root. Steps must be run in order, as each step depends on outputs from the previous one.

---

## Main models

| Model | Script | Notes |
|-------|--------|-------|
| Ridge regression | `step5a_train_ridge.py` | Expanding-window OOS with cross-validation |
| XGBoost | `step5b_train_xgboost.py` | Gradient boosting, 4-fold CV |
| MLP (neural network) | `step5c_train_mlp.py` | scikit-learn MLPRegressor, expanding-window |

All models predict cross-sectional return ranks at each monthly rebalancing date. Predictions are rescaled to the baseline mu distribution before being passed to the optimizer.

---

## Portfolio optimization

The optimization uses a mixed-integer quadratic program (MIQP):

```
minimize  w'Σw − λ·μ'w
subject to
    Σ wᵢ = 1                         (fully invested)
    zᵢ ∈ {0,1},  Σ zᵢ = K = 10      (cardinality)
    0.01·zᵢ ≤ wᵢ ≤ 0.20·zᵢ          (weight bounds)
    Σ_{i∈sector s} wᵢ ≤ 0.30         (sector cap)
```

The covariance matrix Σ is estimated using rolling Ledoit-Wolf shrinkage on the trailing 252 trading days. The baseline uses trailing historical mean for μ. The ML variants replace μ with model predictions.

---

## SAFE AI evaluation

The SAFE AI framework (Giudici 2024) is adapted to evaluate the full investment pipeline:

- **Sustainability/Security** — portfolio robustness, drawdown, turnover, transaction cost sensitivity
- **Accuracy** — out-of-sample forecast quality (information coefficient) and portfolio performance metrics (Sharpe, Sortino, Calmar ratios)
- **Fairness** — sector allocation balance, stock concentration, diversification
- **Explainability** — feature importance and SHAP analysis for the XGBoost model

See `docs/safe_ai_framework.md` for details.

---

## Data

Raw Bloomberg data (prices, total return index, volume, market cap, metadata) are not included in this repository because of licensing restrictions. See `data/README_data.md` for the expected file format and instructions on how to replicate the pipeline with equivalent data.

---

## Limitations

- The out-of-sample test period (January 2023 to December 2025) is short and includes the post-COVID normalisation period and the aftermath of the 2022 rate shock. Results should not be read as evidence that the models will generalise to other regimes.
- ML models are trained on cross-sectional return ranks, not raw returns. This means they capture relative ordering across stocks but do not predict the overall market direction.
- The MIQP solver (Gurobi) runs with a 60-second time limit and a 1% MIP gap. On some rebalancing dates the solver terminates at the gap tolerance rather than at the proven global optimum.
- Feature engineering is kept intentionally simple (momentum, volatility, liquidity, market cap). More sophisticated signals or alternative data could change the results in either direction.
- The SAFE AI evaluation is still being developed. The metrics and the `safeaipackage` integration described in `docs/safe_ai_framework.md` are part of the ongoing thesis work, not a completed validation module.
- The pipeline depends on a proprietary Bloomberg dataset. It cannot be run without data access, and the results in the thesis are not independently reproducible without equivalent data.

---

## Notes

- Raw Bloomberg data and all derived datasets are not tracked in this repository (see `.gitignore`). Running the full pipeline requires access to Bloomberg and a valid Gurobi licence.
- Pipeline outputs -- portfolio weights, performance metrics, and figures -- are generated locally in `data/results/` and `data/figures/` when the scripts are executed, and are not committed to the repository.
- Reference PDFs used during the thesis research are excluded from version control.
- Gurobi solver logs and licence files are also excluded.

---

*Author: Anila Vata — University of Pavia, MSc Quantitative Finance*  
*Contact: anila.vata01@universitadipavia.it*
