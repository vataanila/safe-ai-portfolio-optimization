# SAFE AI Framework — Application to Return Forecasting Evaluation

This document explains how the SAFE AI framework (Babaei, Giudici & Raffinetti 2024) is adapted and applied in this thesis to evaluate the XGBoost return forecasting model.

---

## Background

The SAFE AI framework is a model evaluation framework that assesses AI systems along four dimensions: **Sustainability/Security**, **Accuracy**, **Fairness**, and **Explainability**. The underlying methodology is the Rank Graduation Box (RGB), introduced by Babaei, Giudici & Raffinetti (2024), which provides a family of metrics built on a common statistical foundation derived from Lorenz curves and concordance curves. The four metrics of the framework — RGA, RGR, RGE, and fairness assessed through RGA parity — are unified by this shared foundation and are implemented in the `safeaipackage` Python library.

In this thesis, the framework is applied to the XGBoost return forecasting model, which is the main machine learning component of the pipeline. The object of evaluation is the model's forecast quality, robustness, and interpretability — not the Markowitz portfolio optimizer or the resulting portfolio allocations.

Portfolio-level diagnostics form a separate evaluation layer. Metrics such as annualised return, Sharpe ratio, maximum drawdown, and turnover are used to assess the practical and economic consequences of using XGBoost forecasts within the constrained Markowitz optimizer. These portfolio metrics are compared across all four strategies — the classical baseline and the three ML-enhanced variants (Ridge, XGBoost, MLP) — but they are downstream economic diagnostics, not SAFE AI metrics.

The SAFE AI component is the next planned stage of the thesis. The quantitative pipeline (data cleaning, feature engineering, ML forecasting, portfolio construction, and out-of-sample comparison) has been implemented. The empirical computation of SAFE metrics using `safeaipackage` is still under development. This document describes the intended evaluation framework.

---

## Dimensions

### Accuracy

Accuracy is the central dimension of the Rank Graduation Box framework. The primary SAFE metric for this dimension is:

**RGA — Rank Graduation Accuracy** (Giudici & Raffinetti 2024; Raffinetti 2023)  
RGA measures the concordance between the ranks of predicted values and the ranks of actual values. It extends the logic of the AUC beyond binary classification to handle regression and ordinal outcomes, making it well suited to cross-sectional return forecasting where predictions are rank-ordered across stocks at each rebalancing date. An RGA of 1 indicates perfect rank concordance; values near 0.5 indicate performance no better than random. RGA is computed using the core module of `safeaipackage`.

**Information Coefficient — complementary finance diagnostic**  
The information coefficient (IC) measures the Spearman rank correlation between predicted and realised returns at each rebalancing date. It is a standard diagnostic in quantitative asset management and carries direct economic interpretability in a cross-sectional forecasting context. Summary statistics (mean IC, IC standard deviation, fraction of positive IC dates) are reported for the XGBoost model. The IC is a complementary metric specific to the financial application; it is not the SAFE-native accuracy measure, which is RGA.

---

### Sustainability / Security

In the Rank Graduation Box, Sustainability refers to model robustness to perturbations of the input variables. The primary SAFE metric for this dimension is:

**RGR — Rank Graduation Robustness** (Babaei, Giudici & Raffinetti 2024)  
RGR assesses whether the XGBoost model's output rankings remain stable when the input features are modified or perturbed. It compares model predictions under the original inputs with predictions obtained after applying perturbations to the explanatory variables, using RGA as the underlying measure of concordance. Low RGR values indicate that the model's rank predictions are sensitive to changes in inputs — a concern in a financial setting where features such as momentum or liquidity can shift substantially across market regimes. RGR is computed using the `check_robustness` module of `safeaipackage`.

**Portfolio-level diagnostics — separate economic layer**  
Maximum drawdown, average turnover, and transaction cost sensitivity (net Sharpe ratios at 10, 20, and 30 basis points per unit of traded turnover) are used as downstream diagnostics to assess how any instability in XGBoost forecasts propagates into portfolio allocations. These are not SAFE AI metrics; they measure the economic consequences of forecast behaviour within the Markowitz optimizer.

---

### Fairness

In the original SAFE framework, fairness is assessed by examining whether the model produces systematically different predictions across protected groups, using RGA parity as the measure of disparity. In a return forecasting setting, there are no human protected groups in the standard sense. The concept is therefore adapted to economically meaningful partitions of the stock universe:

- **Sector groups** — is the XGBoost model systematically more accurate (higher RGA) for stocks in certain GICS sectors than in others? Persistent sector-level differences in RGA could indicate that the model's signal is concentrated in particular industries rather than reflecting broad cross-sectional predictability.
- **Size buckets** — does forecast accuracy vary across large-cap, mid-cap, and small-cap stocks? Given that the training universe is large-cap US equities, this is a relevant robustness check within the sample.
- **Liquidity buckets** — are predictions more reliable for liquid stocks (low Amihud ratio) than for less liquid names?

Fairness is assessed by comparing RGA values across these groups and examining whether imparity is systematic and persistent across the out-of-sample period. The `check_fairness` module of `safeaipackage` provides the statistical infrastructure for this comparison; the group definitions are adapted to the financial context.

At the portfolio level, sector allocation heatmaps and stock concentration statistics serve as complementary diagnostics to examine whether any model-level prediction bias propagates into the portfolio's composition. These are interpreted as downstream consequences, not as fairness metrics in the SAFE AI sense.

---

### Explainability

The primary SAFE metric for this dimension is:

**RGE — Rank Graduation Explainability** (Babaei, Giudici & Raffinetti 2024)  
RGE quantifies the relative contribution of each input feature to the XGBoost model's predictions. It compares the full-model RGA with the RGA obtained from reduced models that exclude individual variables, one at a time. A feature with high RGE contributes substantially to the model's rank concordance and can therefore be considered important for its predictive performance. RGE is computed using the `check_explainability` module of `safeaipackage`.

**SHAP values and gain-based feature importance — complementary tools**  
XGBoost provides two additional sources of interpretability that complement RGE:

- **Gain-based feature importance** — the average gain contributed by each feature across all splits, computed at each training window and averaged over rebalancing dates.
- **SHAP values** — SHapley Additive exPlanations decompose individual predictions into additive contributions from each feature, allowing inspection of why the model assigned high or low expected returns to specific stocks at specific dates.

These tools add granularity beyond what RGE provides, particularly in tracing how economic signals (momentum, volatility, liquidity) enter individual predictions. Economic interpretation is given particular attention: features with high RGE and high SHAP contributions should, where possible, correspond to established return predictors from the empirical asset pricing literature. Discrepancies are treated as limitations.

The Ridge regression and MLP models are not the focus of the SAFE AI explainability analysis. Feature importance for Ridge (based on standardised coefficients) is reported for reference in the broader comparative discussion.

---

## References

- Babaei, G., Giudici, P., & Raffinetti, E. (2024). A Rank Graduation Box for SAFE AI. *Expert Systems with Applications*. https://doi.org/10.1016/j.eswa.2024.125239
- Giudici, P., & Raffinetti, E. (2024). RGA: a unified measure of predictive accuracy. *Annals of Operations Research*.
- Raffinetti, E. (2023). A rank graduation accuracy measure to mitigate artificial intelligence risks. *Quality & Quantity*, 57, 2355–2374.
- Markowitz, H. (1952). Portfolio selection. *Journal of Finance*, 7(1), 77–91.
- Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies*, 33(5), 2223–2273.
