# SAFE AI Framework — Application to Return Forecasting Evaluation

This document explains how the SAFE AI framework (Giudici 2024; Babaei, Giudici & Raffinetti 2025) is adapted and applied in this thesis to evaluate the XGBoost return forecasting model.

---

## Background

The SAFE AI framework is a model evaluation framework that assesses AI systems along four dimensions: **Sustainability/Security**, **Accuracy**, **Fairness**, and **Explainability**. It was originally developed to evaluate machine learning models in financial applications.

In this thesis, the framework is applied to the XGBoost return forecasting model, which is the main machine learning component of the pipeline. The object of evaluation is the model's forecast quality, robustness, and interpretability — not the Markowitz portfolio optimizer or the resulting portfolio allocations directly.

Portfolio-level diagnostics form a separate evaluation layer. Metrics such as annualised return, Sharpe ratio, maximum drawdown, and turnover are used to assess the practical and economic consequences of using XGBoost forecasts within the constrained Markowitz optimizer. These portfolio metrics are compared across all four strategies — the classical baseline and the three ML-enhanced variants (Ridge, XGBoost, MLP) — to determine whether XGBoost improves on the benchmark and how it compares with the other forecasting approaches.

The SAFE metrics are computed using the `safeaipackage` Python library and supplemented with XGBoost explainability tools (SHAP values and feature importance).

---

## Dimensions

### Sustainability / Security

In the original SAFE framework, Sustainability refers to model robustness and stability. Applied to the XGBoost return forecasting model, this dimension covers:

- **Forecast stability**: the consistency of XGBoost predictions across rebalancing dates, assessed through the distribution and time-series behaviour of the information coefficient (IC).
- **Robustness under perturbations**: how sensitive XGBoost forecasts are to changes in the training window or feature inputs. The RGA (Relative Gini Accuracy) metric from `safeaipackage` is used to evaluate this.

At the portfolio level — as a separate diagnostic — the economic implications of forecast stability are assessed through maximum drawdown, average turnover, and transaction cost sensitivity (net Sharpe ratios at 10, 20, and 30 basis points per unit of traded turnover). These portfolio diagnostics reflect how forecast instability propagates into allocation decisions, but they are not themselves part of the SAFE AI model evaluation.

---

### Accuracy

Within the SAFE AI evaluation, Accuracy refers to the predictive quality of the XGBoost return forecasting model:

**Forecast accuracy (model-level SAFE evaluation)**  
The information coefficient (IC) measures the Spearman rank correlation between predicted and realised returns at each rebalancing date. A positive IC indicates that the model ranks stocks in roughly the right order on average, though individual predictions may be noisy. Summary statistics (mean IC, IC standard deviation, fraction of positive IC dates) are reported for the XGBoost model. The RGR (Relative Gini Reliability) metric from `safeaipackage` is used to assess forecast reliability.

**Portfolio performance (separate evaluation layer)**  
Out-of-sample portfolio performance metrics are compared across all four portfolios (baseline, Ridge, XGBoost, MLP) as a distinct step from the SAFE AI model evaluation:
- Annualised return, annualised volatility
- Sharpe, Sortino, and Calmar ratios
- Maximum drawdown
- Average monthly turnover

These metrics address the question of whether XGBoost forecasts translate into better risk-adjusted portfolio performance relative to the classical baseline and the other ML approaches, after accounting for transaction costs. They are not direct SAFE AI metrics but complement the model-level evaluation by connecting forecast quality to investment outcomes.

---

### Fairness

In a standard supervised learning context, fairness refers to avoiding systematic bias against protected groups. In a return forecasting setting, a direct analogy does not apply in the same way. The dimension is therefore reinterpreted to assess whether the XGBoost model produces systematically biased predictions across stocks or market segments:

- **Prediction bias across segments**: does XGBoost systematically assign higher predicted returns to certain types of stocks — for example, by sector, size, or liquidity tier — regardless of actual signal quality? Such bias could reflect overfitting to specific historical patterns rather than genuine cross-sectional predictability.
- **Over-representation in predictions**: the RGF (Relative Gini Fairness) metric from `safeaipackage` is applied to check for systematic over- or under-representation in the model's predicted return distribution.

At the portfolio level, sector allocation heatmaps and stock concentration statistics are used as a complementary diagnostic to examine whether any model-level bias propagates into the portfolio's composition — for instance, as persistent overweighting of specific GICS sectors or repeated selection of the same stocks across rebalancing dates. These portfolio diagnostics reflect the downstream consequences of model behaviour rather than being part of the SAFE AI evaluation proper.

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
