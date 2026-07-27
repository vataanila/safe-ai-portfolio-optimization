# Machine Learning in Portfolio Optimization under the SAFE AI Framework

**An Empirical Study of Expected Return Estimation**

Master's thesis in Finance — University of Pavia, Department of Economics and Management
Author: Anila Vata · Supervisor: Prof. Paolo Giudici · A.Y. 2025–2026

---

## Overview

This project asks two questions: can machine-learning-based expected return
estimates beat a simple historical-mean benchmark in out-of-sample portfolio
performance, and — if a model wins on performance — is it also the most
*responsible* model, in the sense defined by the **SAFE AI** framework
(Giudici & Raffinetti, 2023), which evaluates models on robustness, accuracy,
fairness, and explainability?

Four return-estimation methods are compared inside the identical constrained
portfolio-optimization pipeline (same covariance estimator, constraints,
rebalancing rule, and transaction-cost assumptions):

- Historical Mean (benchmark)
- Ridge Regression
- Neural Network (MLP)
- XGBoost

The empirical analysis uses a stock universe derived from the S&P 500
(Bloomberg data, 407 stocks after cleaning, Jan 2010–Dec 2025): a 2010–2015
warm-up, 2016–2022 in-sample training window, and a 2023–2025 out-of-sample
test period with 36 monthly expanding-window rebalances.

The results are organized in three layers, mirroring the thesis itself:
portfolio-level financial performance, model-level SAFE AI evaluation, and
the SAFE-Performance Frontier that connects the two — the thesis's core
contribution.

## Key results

### 1. Portfolio performance

At the base risk-aversion level (λ = 1), XGBoost delivers the highest gross
and net Sharpe ratio among all four estimators:

| Estimator | Ann. Return | Volatility | Sharpe (gross) | Sharpe (net, 30bps) | Max DD | Avg. turnover |
|---|---|---|---|---|---|---|
| XGBoost | 21.60% | 23.98% | 0.90 | 0.78 | 23.41% | 78.2% |
| Historical Mean | 20.93% | 25.39% | 0.82 | 0.76 | 25.66% | 46.5% |
| Ridge | 13.39% | 21.76% | 0.62 | 0.48 | 22.58% | 79.9% |
| Neural Network | 5.45% | 18.04% | 0.30 | 0.12 | 25.57% | 93.2% |

The edge is conditional, not unconditional: XGBoost leads on Sharpe only at
λ = 1 and 2 — at other risk-aversion levels the Historical Mean is more
competitive — and XGBoost's much higher turnover (78.2% vs. 46.5%) erodes
most of its net-of-cost advantage over the simple benchmark.

### 2. SAFE AI evaluation

Only the three ML models are scored on SAFE AI (the Historical Mean is a
benchmark, not a trained model, so it's excluded from this part). No single
model wins on every dimension:

| Dimension | Scalar metric | Compliance vector | Leader |
|---|---|---|---|
| Accuracy | Mean RGA | RGA vector | XGBoost |
| Robustness | Group RGR at 5% | RGR vector | XGBoost |
| Fairness | 1 − mean sector RGA gap | RGF* | Neural Network |
| Explainability | Max single-feature RGE | RGE* | Ridge |

The four dimensions are aggregated into a four-dimensional Integrated
Compliance Score (CS4). Despite no model sweeping every dimension, XGBoost
ranks first under all three aggregation methods (arithmetic, geometric, and
RMS) — the strongest overall balance across dimensions.

### 3. SAFE-Performance Frontier (core contribution)

The thesis's central contribution links the two levels above. At the
aggregate level, SAFE AI compliance (CS4) correlates positively with Sharpe
ratio (ρ = 0.597 / 0.618 / 0.417 for arithmetic / geometric / RMS
aggregation, all p < 0.001) and negatively with maximum drawdown
(ρ ≈ −0.35, all p < 0.01) — trustworthy and profitable align, at least on
average. Higher compliance is also associated with higher turnover
(ρ = 0.611 / 0.598 / 0.669, all p < 0.01): the implementation trade-off is
that more compliant models are typically harder (costlier) to run.

This aggregate relationship comes with an honest caveat: it holds strongly
*within* the Ridge family (ρ = 0.498, p < 0.001) but is not statistically
significant within XGBoost (ρ ≈ 0.09, p = 0.54) or the Neural Network
(ρ = 0.15, p = 0.29). Part of the aggregate pattern reflects differences
*between* model families rather than a relationship that holds *within*
each one — a limitation the thesis states explicitly rather than glossing
over.

## Selected results

| | |
|---|---|
| ![Efficient frontier](tesi_SAFE_optimization/figures/step3/efficient_frontier.png) | ![Portfolio wealth comparison](tesi_SAFE_optimization/figures/step6/portfolio_cumulative_wealth_comparison.png) |
| ![SAFE AI dimensions](tesi_SAFE_optimization/figures/step7/safe_dimensions_grouped_bar.png) | ![SAFE-Performance Frontier](tesi_SAFE_optimization/outputs/step11/figure_A2_safe_performance_frontier.png) |

## Repository structure

```
.
├── Thesis_Anila_Vata.pdf                    # full thesis
├── Anila_Vata_Thesis_Defense_21_July_2026.pdf  # defense slides
└── tesi_SAFE_optimization/
    ├── run_pipeline.py                      # runs the full pipeline end to end
    ├── config.py                            # shared paths/constants
    ├── 1a_load.py                           # load raw Bloomberg data
    ├── 2a_preprocess.py                     # cleaning, universe construction
    ├── 3a_baseline.py / 3b-d_*.py            # historical-mean baseline + viz
    ├── 4a_features.py                       # feature engineering for ML models
    ├── 5a/b/c_*.py                           # Ridge / XGBoost / MLP training
    ├── 6a-f_*.py                             # portfolio construction & comparison
    ├── 7a-f_*.py                             # SAFE AI scalar metrics (RGA, RGR, RGF*, RGE*)
    ├── 8a-e_*.py                             # SAFE AI compliance vectors + Integrated Compliance Score
    ├── 9a/b_*.py                             # SAFE-Performance Frontier
    ├── 10a_appendix.py                       # appendix tables & figures
    ├── build_report.py / generate_report.py  # automated report generation
    ├── figures/                              # all generated charts, by pipeline step
    ├── outputs/step11/                       # appendix tables & figures
    └── data/
        ├── raw/        # Bloomberg source files — NOT included (license)
        ├── clean/      # cleaned/derived market data — NOT included (license)
        └── results/    # model outputs, metrics, compliance scores (included)
```

> **Data availability.** The raw and cleaned market data (`data/raw/`,
> `data/clean/`) are licensed from Bloomberg L.P. and are excluded from this
> repository (see `.gitignore`). What's included is all analysis code, the
> resulting metrics/figures, and the thesis documents themselves. To
> reproduce the pipeline you would need your own Bloomberg data pull matching
> the schema expected in `1a_load.py`.

## Methodology pipeline

1. **Data & preprocessing** — S&P 500-derived universe, prices/volume/market
   cap/total-return index from Bloomberg, cleaned and aligned (2010–2025).
2. **Baseline** — mean-variance optimization with historical mean returns,
   Ledoit-Wolf / OAS covariance shrinkage, transaction costs, turnover
   constraints.
3. **ML return estimation** — Ridge, XGBoost, and a small MLP trained on
   engineered features over an expanding in-sample window (2016–2022),
   re-estimated monthly and evaluated out-of-sample from 2023 to 2025.
4. **Portfolio construction** — same optimizer, constraints and rebalancing
   rule applied to each model's return forecasts; performance compared on
   Sharpe ratio, drawdown, turnover and net returns.
5. **SAFE AI assessment** — accuracy (RGA), robustness (RGR), fairness
   across GICS sectors (RGF*), and explainability (RGE*), aggregated into a
   four-dimensional Integrated Compliance Score (CS4) per model.
6. **SAFE-Performance Frontier** — joint view of financial performance vs.
   compliance score across models and risk-aversion levels (λ).

## Tech stack

Python · pandas / numpy / scipy · scikit-learn · XGBoost · matplotlib ·
[`safeaipackage`](https://pypi.org/project/safeaipackage/) (SAFE AI metrics,
developed by Prof. Paolo Giudici's group) · python-docx / reportlab for
automated report generation.

## Reproducing the pipeline

```bash
cd tesi_SAFE_optimization
pip install -r ../requirements.txt
python run_pipeline.py
```

`run_pipeline.py` executes each step in order (data load → preprocessing →
baseline → ML models → portfolios → SAFE AI metrics → SAFE-Performance
Frontier) and logs timing for each stage. Note this requires the Bloomberg
data described above.

## Author

**Anila Vata** — Master's in Finance, University of Pavia
[anila.vata01@universitadipavia.it](mailto:anila.vata01@universitadipavia.it)
