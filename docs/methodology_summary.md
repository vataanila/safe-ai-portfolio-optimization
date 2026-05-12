# Methodology Summary

This document describes the main methodological choices made in the thesis pipeline. It is meant as a readable summary for anyone reviewing the code, not as a substitute for the thesis text.

---

## Data source

The dataset consists of daily Bloomberg data for approximately 400 large-cap US equities over the period 2010–2025. The following Bloomberg fields are used:

- `TOT_RETURN_INDEX_GROSS_DVDS` (TRI) — total return index including gross dividends, used to compute returns
- `PX_LAST` (closing prices) — used for the Amihud illiquidity ratio
- `CUR_MKT_CAP` — daily market capitalisation, used as a feature
- `PX_VOLUME` — daily trading volume, used for the Amihud ratio and volume momentum features
- `GICS_SECTOR_NAME` — sector classification, used for the sector concentration constraint

The raw data are stored as `.xlsx` files and are not included in the repository because of Bloomberg licensing restrictions.

---

## Return construction

Daily log returns are computed from the total return index (TRI) rather than from price data. The TRI accounts for dividend income and is the standard choice for empirical portfolio studies. The formula used is:

```
r_{i,t} = log(TRI_{i,t} / TRI_{i,t-1})
```

Returns are computed in `step2_preprocess_returns.py` and saved to `data/clean/returns.csv`. The full sample spans 2010–2025, with a warm-up period (2010–2015) used only for lagging features.

---

## Feature engineering

Features are constructed in `step4_feature_engineering.py`. All features are computed using only information available strictly before the rebalancing date (no look-ahead bias). Features are cross-sectionally ranked at each monthly rebalancing date to reduce the influence of outliers and make the signal more stable across time.

The feature set includes:

| Feature | Description |
|---------|-------------|
| `ret_1w`, `ret_1m` | Short-term momentum (1 week, 1 month) |
| `ret_3m`, `ret_6m`, `ret_12m` | Medium-term momentum |
| `vol_1m`, `vol_3m` | Realised volatility (1 month, 3 months) |
| `vol_ratio` | Ratio of short-term to medium-term volatility |
| `amihud` | Amihud (2002) illiquidity ratio |
| `log_mktcap` | Log of market capitalisation (if available) |

The panel is saved to `data/results/step4/ml_panel.csv`.

---

## Machine learning return forecasting

Three models are trained in `step5a_train_ridge.py`, `step5b_train_xgboost.py`, and `step5c_train_mlp.py`. XGBoost is the primary forecasting model of interest and the focus of the SAFE AI evaluation. Ridge regression and the MLP are included as additional ML benchmarks to assess whether XGBoost outperforms alternative forecasting approaches on this type of financial panel data.

- **Ridge regression** — linear model with L2 regularisation; hyperparameter selected on a rolling validation window
- **XGBoost** — gradient boosting trees; hyperparameters tuned by 4-fold cross-validation on the expanding training set
- **MLP (multi-layer perceptron)** — two hidden layers, trained with scikit-learn's `MLPRegressor`

All three models follow the same expanding-window out-of-sample (OOS) framework. At each monthly rebalancing date, the model is trained on all available historical data up to (but not including) the rebalancing date. Predictions are cross-sectional return ranks for the next month, which are then rescaled to match the distribution of the historical-mean baseline mu. This rescaling ensures that the scale of mu passed to the optimizer is comparable across all models.

Model forecast quality is reported as the information coefficient (IC), defined as the Spearman rank correlation between predicted and realised returns.

---

## Markowitz portfolio optimization

The baseline portfolio is constructed in `step3_baseline_portfolio.py`. The ML-enhanced portfolios are constructed in steps 6a–6c using the same optimizer with different mu inputs.

The optimization problem is a mixed-integer quadratic program (MIQP):

```
minimize  w'Σw − λ·μ'w
subject to
    Σ wᵢ = 1
    zᵢ ∈ {0,1},  Σ zᵢ = K = 10
    0.01·zᵢ ≤ wᵢ ≤ 0.20·zᵢ
    Σ_{i∈sector s} wᵢ ≤ 0.30  for each sector s
```

where:
- μ is the expected return vector (historical mean for the baseline, ML predictions for the ML variants)
- Σ is the rolling Ledoit-Wolf shrinkage covariance matrix
- K = 10 is the cardinality constraint (number of stocks held)
- Weight bounds are 1%–20% per stock
- The sector cap prevents any single GICS sector from exceeding 30% of the portfolio

The solver is Gurobi (via `gurobipy`), with a 60-second time limit and a 1% MIP gap tolerance. Rebalancing is done monthly.

The only difference between the baseline and the ML portfolios is the mu estimator. Everything else (Sigma, constraints, solver) is held identical so that any difference in performance can be attributed to the mu signal.

---

## Out-of-sample backtesting

The out-of-sample test period runs from January 2023 to December 2025. This period was not used in any model training or parameter calibration.

At each monthly rebalancing date:
1. Estimate Σ using trailing 252-day returns
2. Compute μ (historical mean or ML prediction)
3. Solve the MIQP
4. Hold the portfolio until the next rebalancing date
5. Compute daily portfolio returns using buy-and-hold return attribution

Portfolio turnover is computed as `0.5 × Σ|w_target − w_pre_trade|` at each rebalancing date. The first rebalancing date is excluded from the turnover calculation.

---

## Performance and risk metrics

All portfolios are evaluated on the following metrics over the out-of-sample period:

| Metric | Definition |
|--------|-----------|
| Annualised return | `mean(r) × 252` |
| Annualised volatility | `std(r) × sqrt(252)` |
| Sharpe ratio | `annualised return / annualised volatility` |
| Sortino ratio | `annualised return / downside deviation` |
| Calmar ratio | `annualised return / max drawdown` |
| Maximum drawdown | largest peak-to-trough loss |
| Average turnover | mean monthly turnover across rebalancing dates |

Net-of-cost performance is also computed at 10, 20, and 30 basis points per unit of traded turnover.

---

## Main limitations

- The sample period (2010–2025) includes several unusual episodes (2020 COVID crash, 2022 rate shock). Results for the short test period (2023–2025) should be interpreted cautiously.
- The ML models are trained on cross-sectional ranks, which limits their ability to capture the overall market direction.
- The MIQP solver uses a fixed time limit of 60 seconds. For some rebalancing dates, the solver terminates at the MIP gap tolerance rather than at the proven global optimum.
- Feature engineering is kept simple (momentum, volatility, liquidity). More sophisticated features or alternative data could change the results.
- The SAFE AI evaluation is planned for the XGBoost return forecasting model using the Rank Graduation Box metrics: RGA (Accuracy), RGR (Robustness/Sustainability), RGE (Explainability), and RGA parity across economic groups (Fairness). These metrics are implemented in `safeaipackage`. The empirical computation of these metrics is still under development. Portfolio performance metrics (Sharpe, Sortino, Calmar, drawdown, turnover) are downstream economic diagnostics evaluated separately, not SAFE AI metrics.
