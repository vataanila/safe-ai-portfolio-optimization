# SAFE AI Framework — Application to Portfolio Optimization

This document explains how the SAFE AI framework (Giudici 2024; Babaei, Giudici & Raffinetti 2025) is adapted and applied in this thesis to evaluate the machine learning-enhanced portfolio pipeline.

---

## Background

The SAFE AI framework is a model evaluation framework that assesses AI systems along four dimensions: **Sustainability/Security**, **Accuracy**, **Fairness**, and **Explainability**. It was originally developed to evaluate machine learning models in financial applications.

In this thesis, the framework is applied not just to the ML model in isolation, but to the full investment pipeline — from data inputs and feature construction through to portfolio construction, rebalancing, and out-of-sample performance. The object of evaluation is therefore the complete system, not only the return forecasting model.

The SAFE metrics are computed using the `safeaipackage` Python library and supplemented with additional portfolio-level diagnostics.

---

## Dimensions

### Sustainability / Security

In the original SAFE framework, Sustainability refers to model robustness and stability. In a portfolio management context, this dimension is extended to cover:

- **Drawdown**: the maximum peak-to-trough loss during the out-of-sample period. A portfolio with large drawdowns is fragile and imposes significant implementation risk.
- **Turnover**: how frequently and how much the portfolio changes composition. High turnover increases transaction costs and makes the strategy harder to implement in practice.
- **Transaction cost sensitivity**: net Sharpe ratios at 10, 20, and 30 basis points per unit of traded turnover. This gives a rough sense of how quickly performance degrades as costs increase.
- **Stability under changing conditions**: the portfolio's behaviour during the 2022 rate shock and the post-COVID normalisation period. These are not formal stress tests but qualitative observations from the backtest period.

The RGA (Relative Gini Accuracy) metric from `safeaipackage` is also used to evaluate forecast robustness under perturbations.

---

### Accuracy

Accuracy is assessed at two levels:

**Forecast accuracy (model level)**  
The information coefficient (IC) measures the Spearman rank correlation between predicted and realised returns at each rebalancing date. A positive IC indicates that the model ranks stocks in roughly the right order on average, though individual predictions may be noisy. Summary statistics (mean IC, IC standard deviation, fraction of positive IC dates) are computed for each model.

**Portfolio accuracy (pipeline level)**  
Out-of-sample portfolio performance metrics are compared across all four portfolios (baseline, Ridge, XGBoost, MLP):
- Annualised return, annualised volatility
- Sharpe, Sortino, and Calmar ratios
- Maximum drawdown
- Average monthly turnover

The question is whether using ML-predicted returns leads to better risk-adjusted performance than the classical trailing mean, after accounting for transaction costs.

The RGR (Relative Gini Reliability) metric from `safeaipackage` is used to measure forecast reliability.

---

### Fairness

In a standard supervised learning context, fairness refers to avoiding systematic bias against protected groups. In a portfolio management setting, a direct analogy does not apply. The framework is therefore reinterpreted as follows:

- **Sector concentration**: does the optimizer systematically overweight or underweight specific GICS sectors? A portfolio that concentrates heavily in one or two sectors may reflect estimation artefacts rather than genuine signal.
- **Stock concentration**: beyond the cardinality and weight constraints, does the portfolio tend to hold the same few stocks repeatedly?
- **Allocation bias**: is there any systematic pattern in which stocks are selected, for example a persistent preference for large-cap or high-momentum names?

Fairness is evaluated via sector allocation heatmaps (average sector weights over time) and summary statistics on allocation concentration.

The RGF (Relative Gini Fairness) metric from `safeaipackage` is applied to check for systematic over- or under-representation in model predictions.

---

### Explainability

The XGBoost model provides two sources of explainability:

- **Feature importance** — the gain-based feature importance computed by XGBoost at each training window, averaged across all rebalancing dates. This indicates which features the model relies on most across the out-of-sample period.
- **SHAP values** — SHapley Additive exPlanations decompose individual predictions into contributions from each feature. This allows more granular inspection of why the model predicted high or low expected returns for specific stocks at specific dates.

Economic interpretation is given particular attention: features with high importance should, ideally, correspond to known return predictors from the empirical asset pricing literature (momentum, low volatility, liquidity effects). Discrepancies are discussed as limitations.

The Ridge regression model does not provide native feature importance in the same way as tree-based models. Feature importance for Ridge is based on standardised coefficients and interpreted alongside the XGBoost results.

The RGE (Relative Gini Explainability) metric from `safeaipackage` is also applied.

---

## Implementation status

The quantitative portfolio optimization pipeline has already been implemented in the repository, including the Markowitz benchmark, the machine learning forecasting models, portfolio construction and out-of-sample comparison.

The SAFE AI component is the next stage of the thesis work. At this stage, this document explains how the SAFE AI framework is intended to be adapted to a portfolio optimization setting.

The empirical implementation of the SAFE dimensions is still being developed. For this reason, this section should be read as the evaluation framework guiding the next part of the thesis, rather than as a fully completed validation module.

---

## References

- Giudici, P. (2024). SAFE AI. *Annals of Operations Research*, forthcoming.
- Babaei, G., Giudici, P., & Raffinetti, E. (2025). SAFE AI for financial risk assessment.
- Markowitz, H. (1952). Portfolio selection. *Journal of Finance*, 7(1), 77–91.
- Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies*, 33(5), 2223–2273.
