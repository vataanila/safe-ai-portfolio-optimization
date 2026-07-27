"""
generate_report.py
==================
Generates a PDF progress report covering Steps 7, 8, 10c and 10h.
Run from the project root:  python generate_report.py

Requires: reportlab   (pip install reportlab)
Output  : report_steps7_8_10c_10h.pdf
"""

import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

BASE = os.path.dirname(os.path.abspath(__file__))

def p(path): return os.path.join(BASE, path)

# ── Styles ────────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

title_style = ParagraphStyle("ReportTitle",
    parent=styles["Title"],
    fontSize=20, leading=26, spaceAfter=6, alignment=TA_CENTER,
    textColor=colors.HexColor("#1a1a2e"))

subtitle_style = ParagraphStyle("Subtitle",
    parent=styles["Normal"],
    fontSize=11, leading=14, spaceAfter=20, alignment=TA_CENTER,
    textColor=colors.HexColor("#444444"))

h1_style = ParagraphStyle("H1",
    parent=styles["Heading1"],
    fontSize=14, leading=18, spaceBefore=18, spaceAfter=6,
    textColor=colors.HexColor("#1a1a2e"),
    borderPad=4)

h2_style = ParagraphStyle("H2",
    parent=styles["Heading2"],
    fontSize=11, leading=14, spaceBefore=12, spaceAfter=4,
    textColor=colors.HexColor("#2c3e50"))

body_style = ParagraphStyle("Body",
    parent=styles["Normal"],
    fontSize=10, leading=15, spaceAfter=8, alignment=TA_JUSTIFY,
    fontName="Times-Roman")

caption_style = ParagraphStyle("Caption",
    parent=styles["Normal"],
    fontSize=8.5, leading=11, spaceAfter=10, alignment=TA_CENTER,
    textColor=colors.HexColor("#555555"), fontName="Times-Italic")

note_style = ParagraphStyle("Note",
    parent=styles["Normal"],
    fontSize=9, leading=12, spaceAfter=6,
    textColor=colors.HexColor("#555555"), fontName="Times-Italic")

def B(text): return f"<b>{text}</b>"
def I(text): return f"<i>{text}</i>"

def body(text): return Paragraph(text, body_style)
def h1(text):   return Paragraph(text, h1_style)
def h2(text):   return Paragraph(text, h2_style)
def cap(text):  return Paragraph(text, caption_style)
def note(text): return Paragraph(text, note_style)
def sp(h=0.3):  return Spacer(1, h*cm)
def hr():       return HRFlowable(width="100%", thickness=0.5,
                                  color=colors.HexColor("#cccccc"), spaceAfter=6)

def fig(rel_path, width_cm=14, caption_text=None):
    full = p(rel_path)
    if not os.path.exists(full):
        return [body(f"[Figure not found: {rel_path}]")]
    items = [Image(full, width=width_cm*cm, height=width_cm*cm*0.55)]
    if caption_text:
        items.append(cap(caption_text))
    return items

def make_table(headers, rows, col_widths=None):
    data = [headers] + rows
    style = TableStyle([
        ("BACKGROUND",  (0,0), (-1,0),  colors.HexColor("#2c3e50")),
        ("TEXTCOLOR",   (0,0), (-1,0),  colors.white),
        ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 8.5),
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f2f4f7")]),
        ("GRID",        (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
    ])
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(style)
    return t

story = []

# ── Cover ─────────────────────────────────────────────────────────────────────
story += [
    sp(3),
    Paragraph("MSc Thesis – Progress Report", subtitle_style),
    sp(0.5),
    Paragraph("SAFE AI Evaluation and Portfolio Optimization", title_style),
    sp(0.5),
    Paragraph("Steps 7, 8, 10c and 10h", subtitle_style),
    sp(1),
    hr(),
    sp(0.5),
    Paragraph("Anila Vata", subtitle_style),
    Paragraph("University of Pavia, MSc Quantitative Finance", subtitle_style),
    Paragraph("Supervisor: Prof. Paolo Giudici", subtitle_style),
    sp(0.5),
    hr(),
    sp(2),
    body(
        "This report summarises what has been done in the four analytical steps "
        "that form the core of the empirical part of my thesis. The goal of the "
        "project is to evaluate three machine-learning models -- Ridge regression, "
        "XGBoost, and a Multilayer Perceptron (MLP) - under the SAFE AI framework "
        "(Security, Accuracy, Fairness, Explainability) and to study whether the "
        "model that scores highest on SAFE compliance also builds the best "
        "investment portfolio."
    ),
    body(
        "The four steps covered here are: "
        "(7) computation of the four SAFE dimensions on the out-of-sample test set; "
        "(8) construction of an extended 4-dimensional SAFE Compliance Score that "
        "adds sector-level group parity to the original three dimensions; "
        "(10c) expansion of the SAFE–performance frontier across 150 model "
        "configurations (50 per family); and "
        "(10h) scatter analysis and visualisation of the compliance–performance "
        "relationship across all configurations."
    ),
    PageBreak(),
]

story += [
    h1("1.  Step 7 -- SAFE Dimensions on the Out-of-Sample Test Set"),
    hr(),
    sp(0.3),
    body(
        "The first thing I had to do was prepare the data properly. "
        "Step 7a takes the ML panel built in step 4 and the return predictions "
        "produced by each model in step 5, aligns them by year-month (because "
        "prediction files use first-trading-day dates while the panel uses "
        "calendar dates), and splits the data into a training set (before "
        "January 2023) and a test set (from January 2023 onward). "
        "The test panel contains 35 rebalancing dates and around 407 assets per date."
    ),
    body(
        "Steps 7b through 7e then compute the four SAFE dimensions separately "
        "on the test set using the safeaipackage library developed by Babaei and "
        "Giudici. Each dimension captures a different aspect of model behaviour:"
    ),
    body(
        "<b>RGA – Accuracy.</b> The Rank Graduation Accuracy measures how well each "
        "model's predicted return ranking agrees with the realised cross-sectional "
        "ranking of assets. A value of 0.5 means the model does no better than "
        "random; a value above 0.5 means the model adds predictive value."
    ),
    body(
        "<b>RGR – Robustness.</b> The Rank Graduation Robustness measures how stable "
        "the model's ranking predictions are as the training data is progressively "
        "perturbed. The RGR curve starts at 1.0 (no perturbation) and should "
        "stay as high as possible as perturbation increases."
    ),
    body(
        "<b>RGE – Explainability.</b> The Rank Graduation Explainability measures "
        "whether the model's predictions can be reproduced by progressively removing "
        "features. I report the maximum RGE value across features, which identifies "
        "which feature is most important for each model."
    ),
    body(
        "<b>RGF – Group Parity.</b> I extended the original three-dimensional SAFE "
        "framework by adding RGF, which measures whether the model's accuracy is "
        "consistent across GICS sectors. A large gap between the best and worst "
        "sector means the model is unfair in the sense that it predicts much better "
        "for some sectors than others."
    ),
    sp(0.5),
    h2("1.1  Results - SAFE Dimension Summary"),
    sp(0.2),
]

# Table 1: RGA summary
story += [
    make_table(
        ["Model", "Mean RGA", "Std RGA", "Min RGA", "Max RGA", "N Dates"],
        [
            ["Ridge",   "0.489", "0.063", "0.394", "0.709", "35"],
            ["XGBoost", "0.502", "0.057", "0.401", "0.695", "35"],
            ["MLP",     "0.500", "0.045", "0.425", "0.590", "35"],
        ],
        col_widths=[3*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2*cm]
    ),
    cap("Table 1. RGA Accuracy scores across the 35 out-of-sample rebalancing dates."),
    sp(0.5),
]

story += [
    body(
        "The accuracy scores are very close to each other and only slightly above "
        "the random baseline of 0.5. This is actually quite typical in cross-sectional "
        "return prediction: financial data is noisy and predicting the exact rank of "
        "hundreds of assets is hard. XGBoost has the highest mean RGA (0.502), "
        "followed by MLP (0.500) and Ridge (0.489). The differences are small "
        "but systematic."
    ),
]

# Table 2: Full SAFE dimension summary
story += [
    sp(0.3),
    make_table(
        ["Model", "RGA (Acc.)", "RGR (Rob.)", "RGE* (Expl.)", "RGF* (Parity)", "Sector Gap"],
        [
            ["Ridge",   "0.489", "0.737", "0.480", "0.615", "0.385"],
            ["XGBoost", "0.502", "0.745", "0.221", "0.647", "0.353"],
            ["MLP",     "0.500", "0.709", "0.202", "0.659", "0.341"],
        ],
        col_widths=[2.5*cm, 2.5*cm, 2.5*cm, 2.8*cm, 2.8*cm, 2.4*cm]
    ),
    cap("Table 2. SAFE dimension summary. RGE* = max RGE across features. "
        "RGF* = 1 − sector gap (positively oriented). Sector gap = mean absolute "
        "difference between best and worst sector RGA."),
    sp(0.5),
]

story += [
    body(
        "A few things stand out from Table 2. First, robustness (RGR) is the "
        "highest-scoring dimension for all three models, around 0.71–0.74, which "
        "means the ranking predictions remain reasonably stable even when the "
        "training data is perturbed. XGBoost is the most robust, Ridge and MLP "
        "slightly less so."
    ),
    body(
        "Second, explainability (RGE*) is where the models differ most. Ridge "
        "scores 0.48, meaning that the log_mktcap feature alone can roughly "
        "replicate most of the model's ranking behaviour. XGBoost and MLP score "
        "only 0.22 and 0.20 respectively, their predictions are harder to reduce "
        "to a single feature, which reflects their higher complexity."
    ),
    body(
        "Third, group parity (RGF*) is moderate for all models, between 0.61 "
        "and 0.66. The sector gap is around 0.34–0.39, meaning the worst sector "
        "has an RGA that is roughly 35–39 percentage points lower than the best "
        "sector. Real Estate consistently comes out as the hardest sector to "
        "predict across all three models."
    ),
    sp(0.3),
]

story += fig("figures/step7/safe_dimensions_grouped_bar.png", width_cm=15,
             caption_text="Figure 1. SAFE AI dimensions by model. Each bar shows the "
             "mean value of a dimension across the 35 out-of-sample dates. "
             "RGE* is the maximum RGE across features; RGF* = 1 − sector gap.")

story += [sp(0.5),
    h2("1.2  RGA Curve and RGR Curve"),
    sp(0.2),
    body(
        "The RGA curve in Figure 2 shows the accuracy as a function of how much "
        "data is removed. All three models track the random baseline closely until "
        "about 90% of the data is removed, at which point they collapse to zero "
        "because there is simply not enough data left to rank assets. This confirms "
        "that the models are not relying on a small subset of observations."
    ),
]

story += fig("figures/step8/rga/rga_curve_by_model.png", width_cm=14,
             caption_text="Figure 2. RGA curve by model. The x-axis is the fraction "
             "of training data removed; the dashed orange line is the random baseline.")

story += [
    sp(0.4),
    body(
        "The RGR curve in Figure 3 is more informative. It shows how the ranking "
        "agreement decays as input noise increases. Ridge degrades the fastest "
        "and crosses below the random baseline at around 60–70% perturbation. "
        "MLP is the most robust at high perturbation levels, maintaining an RGR "
        "around 0.42 even at 100% normalized perturbation. XGBoost sits in between."
    ),
]

story += fig("figures/step8/rgr/rgr_curve_by_model.png", width_cm=14,
             caption_text="Figure 3. RGR curve by model. Higher is better. "
             "The dashed red line is the random baseline (RGR = 0.50).")

story += [PageBreak()]

story += [
    h1("2.  Step 8 - Extended 4D SAFE Compliance Score"),
    hr(),
    sp(0.3),
    body(
        "Once the four SAFE dimensions are computed as scalar summaries, the next "
        "step is to aggregate them into a single Compliance Score that can be used "
        "to compare models and -- later in steps 10c and 10h, to plot against "
        "portfolio performance."
    ),
    body(
        "The original SAFE framework by Giudici and Kolesnikov integrates three "
        "dimensions: accuracy (RGA), robustness (RGR), and explainability (RGE). "
        "In this thesis I added a fourth dimension, group parity (RGF*), which "
        "captures whether the model is equally accurate across sectors. This is "
        "a natural extension because fairness across subgroups is a standard "
        "requirement in responsible AI."
    ),
    body(
        "The Compliance Score is defined as the mean of an aggregation function "
        "applied to all combinations of values from the four SAFE vectors. "
        "I compute it three ways to check robustness of the aggregation choice: "
        "arithmetic mean, geometric mean, and root-mean-square (RMS). The "
        "arithmetic mean penalises low values leniently; the geometric mean "
        "collapses to zero if any dimension is zero; the RMS gives extra weight "
        "to extreme values."
    ),
    sp(0.4),
    h2("2.1  SAFE Vector Means by Dimension"),
    sp(0.2),
]

story += fig("figures/step8/compliance/compliance_dimension_means_by_model.png",
             width_cm=15,
             caption_text="Figure 4. Mean value of each SAFE vector dimension by model. "
             "The dashed line at 0.50 is a reference for the random baseline. "
             "RGE* and RGF* are the highest-scoring dimensions for all models.")

story += [
    sp(0.4),
    body(
        "Figure 4 shows that the two strongest dimensions across all models are "
        "explainability (RGE* ≈ 0.70–0.90) and group parity (RGF* ≈ 0.82–0.84). "
        "Accuracy and robustness sit closer to 0.32–0.53. Ridge has the highest "
        "explainability but the lowest robustness; MLP has the highest robustness "
        "and group parity but the lowest explainability."
    ),
    sp(0.3),
    h2("2.2  Extended 4D Compliance Score"),
    sp(0.2),
]

story += fig("figures/step8/compliance/compliance_score_4d_by_model.png",
             width_cm=15,
             caption_text="Figure 5. Extended 4D SAFE Compliance Score by model and "
             "aggregation method. Higher is better.")

story += [
    sp(0.3),
    make_table(
        ["Model", "CS4 Arith.", "CS4 Geom.", "CS4 RMS", "CS3 Arith.", "CS3 Geom.", "CS3 RMS", "Δ Arith."],
        [
            ["Ridge",   "0.619", "0.433", "0.690", "0.554", "0.384", "0.638", "+0.066"],
            ["XGBoost", "0.659", "0.571", "0.704", "0.603", "0.523", "0.653", "+0.056"],
            ["MLP",     "0.630", "0.563", "0.659", "0.561", "0.511", "0.584", "+0.069"],
        ],
        col_widths=[2.2*cm, 2*cm, 2*cm, 2*cm, 2.2*cm, 2.2*cm, 2.2*cm, 2.2*cm]
    ),
    cap("Table 3. 4D vs 3D SAFE Compliance Scores. Δ Arith. = CS4 − CS3 (arithmetic). "
        "Adding RGF* consistently raises the score for all models."),
    sp(0.4),
]

story += fig("figures/step8/compliance/compliance_score_3d_vs_4d.png",
             width_cm=13,
             caption_text="Figure 6. 3D vs 4D Compliance Score (arithmetic aggregation). "
             "Adding sector parity raises the score for all three models by 5–7 points.")

story += [
    sp(0.4),
    body(
        "XGBoost has the highest 4D Compliance Score under both arithmetic (0.659) "
        "and RMS (0.704) aggregation. Under geometric aggregation XGBoost also leads "
        "(0.571), though all models score lower here because the geometric mean is "
        "more sensitive to dimensions with low values."
    ),
    body(
        "Adding group parity (the fourth dimension) consistently raises the score "
        "for all models by 5.6 to 6.9 percentage points under arithmetic aggregation. "
        "This makes sense because RGF* scores are high (0.82–0.84) and pull the "
        "aggregate upward. The ranking of models is preserved: XGBoost > MLP > Ridge "
        "under arithmetic and RMS."
    ),
    sp(0.3),
    h2("2.3  Sector-Level Group Parity"),
    sp(0.2),
]

story += fig("figures/step8/rgf/sector_rga_heatmap_by_model.png",
             width_cm=10,
             caption_text="Figure 7. Mean sector RGA by model and GICS sector. "
             "Values close to 0.5 (yellow) are near-random; higher (green) is better. "
             "Real Estate is consistently the hardest sector to predict.")

story += [
    sp(0.4),
    body(
        "Figure 7 shows that no model is dramatically better or worse across "
        "sectors -- most cells are in the 0.47–0.54 range, which is close to the "
        "random baseline of 0.5. Real Estate is the only sector where all three "
        "models score below 0.48. Utilities and Energy are relatively easier to "
        "predict. Overall, the sector parity picture is reassuring: the models are "
        "not systematically biased against any particular sector."
    ),
    PageBreak(),
]

story += [
    h1("3.  Step 10c, SAFE–Performance Frontier (50 × 3 Configurations)"),
    hr(),
    sp(0.3),
    body(
        "Steps 7 and 8 evaluated each model family with a single default "
        "configuration. Step 10c expands this to 150 configurations -- 50 per "
        "model family - to build a proper SAFE–performance frontier. The idea is "
        "to see whether different hyperparameter choices lead to a systematic "
        "trade-off or alignment between SAFE compliance and portfolio performance."
    ),
    body(
        "The three grids are: for Ridge, 50 values of the regularisation parameter "
        "α on a logarithmic scale from 10⁻⁴ to 10⁴; for XGBoost and MLP, 50 "
        "random configurations drawn with seed 42, covering key hyperparameters "
        "(learning rate, depth, number of estimators for XGBoost; hidden layer "
        "sizes, learning rate, regularisation for MLP). Lambda is fixed at 1.0 "
        "throughout, so the only varying dimension is the model configuration."
    ),
    body(
        "For each of the 150 configurations, the full pipeline is run: "
        "expanding-window prediction, score rescaling, MIQP portfolio optimisation "
        "with Gurobi (K=10 assets, weight bounds 1%–20%, sector cap 30%), "
        "portfolio backtest, and SAFE compliance scoring. "
        "Results are checkpointed immediately so the script can be restarted "
        "if interrupted."
    ),
    sp(0.3),
    h2("3.1  Output Structure"),
    sp(0.2),
    body(
        "The output is a single CSV file, "
        "data/results/step10/safe_performance_frontier_50x3.csv -- with one row "
        "per configuration and columns for: model family, configuration ID, "
        "hyperparameter string, three compliance scores (arithmetic, geometric, RMS), "
        "four SAFE sub-scores (accuracy, robustness, fairness, explainability), "
        "and five performance metrics (Sharpe ratio, Sortino ratio, Calmar ratio, "
        "maximum drawdown, average turnover)."
    ),
    sp(0.3),
    h2("3.2  Key Configurations"),
    sp(0.2),
    body(
        "The most notable configuration that emerged from this step is "
        "<b>xgboost_47</b>, which achieves the highest compliance score and the "
        "highest Sharpe ratio simultaneously across all three compliance score "
        "aggregations. This makes it a Pareto-dominant point on the "
        "compliance–Sharpe frontier, more on this in Step 10h."
    ),
    sp(0.2),
    make_table(
        ["Config. ID", "Family", "CS Arith.", "CS Geom.", "CS RMS", "Sharpe", "Max DD", "Avg Turn."],
        [
            ["xgboost_47", "XGBoost", "0.629", "0.528", "0.661", "1.812", "0.163", "0.920"],
            ["xgboost_18", "XGBoost", "0.573", "0.467", "0.617", " -- ",     " - ",     "0.557"],
            ["mlp_14",     "MLP",     "0.603", "0.502", "0.638", "1.773", " -- ",     ", "],
            ["mlp_45",     "MLP",     "0.609", "0.509", "0.643", "—",     "0.155", " - "],
            ["mlp_27",     "MLP",     "0.551", "0.381", "0.621", " -- ",     ", ",     "0.769"],
            ["ridge_32",   "Ridge",   "0.549", "0.367", "0.614", "0.632", " - ",     ", "],
            ["ridge_49",   "Ridge",   "0.528", "0.337", "0.596", " -- ",     "0.181", ", "],
        ],
        col_widths=[2.5*cm, 2*cm, 2*cm, 2*cm, 2*cm, 1.8*cm, 1.8*cm, 2.2*cm]
    ),
    cap("Table 4. Selected notable configurations from the 50×3 frontier. "
        "Dashes indicate the configuration was not optimal on that metric. "
        "xgboost_47 is the only configuration that simultaneously achieves "
        "Best SAFE and Best Performance on Sharpe."),
    PageBreak(),
]

story += [
    h1("4.  Step 10h - SAFE–Performance Scatter Analysis"),
    hr(),
    sp(0.3),
    body(
        "Step 10h is the visualisation and analysis step. It reads the 150-row "
        "CSV produced by step 10c and generates 12 three-panel figures, one for "
        "each combination of scope (all models together, Ridge only, XGBoost only, "
        "MLP only) and performance metric (Sharpe ratio, maximum drawdown, average "
        "turnover). Each figure has three panels side by side, one for each "
        "compliance score aggregation (arithmetic, geometric, RMS)."
    ),
    body(
        "In each panel, the x-axis is the SAFE Compliance Score and the y-axis "
        "is the performance metric. The scatter points are the 50 configurations "
        "of that model family. A LOWESS smooth curve (locally weighted scatterplot "
        "smoothing) shows the average performance as a function of compliance. "
        "If the compliance score range is too narrow for a stable LOWESS fit "
        "(less than 10 percentage points), the curve falls back to a linear trend. "
        "The Spearman rank correlation ρ and its p-value are shown in each panel. "
        "The Best SAFE and Best Performance configurations are highlighted."
    ),
    sp(0.3),
    h2("4.1  Sharpe Ratio"),
    sp(0.2),
]

story += fig("figures/step10h/sharpe/all_models.png", width_cm=16,
             caption_text="Figure 8. SAFE Compliance vs Annualized Sharpe Ratio - "
             "all model families. Each colour is a model family; each curve is a "
             "family-specific LOWESS smooth. ρ ≈ +0.60, p = 0.000 across all panels. "
             "xgboost_47 (★) is simultaneously Best SAFE and Best Performance.")

story += [
    sp(0.3),
    body(
        "The all-models Sharpe figure (Figure 8) is probably the most important "
        "single result. The Spearman correlation is around +0.60 and highly "
        "significant. The XGBoost cloud sits at the top-right of each panel, "
        "higher compliance and higher Sharpe than Ridge and MLP. xgboost_47 stands "
        "out as a gold star at the top-right, being both the most compliant "
        "configuration and the highest-Sharpe one."
    ),
]

story += fig("figures/step10h/sharpe/ridge.png", width_cm=16,
             caption_text="Figure 9. Sharpe vs Compliance - Ridge only. "
             "The linear trend rises sharply; ρ ≈ +0.50–0.70 depending on aggregation. "
             "Best SAFE and Best Performance are different configurations.")

story += fig("figures/step10h/sharpe/xgboost.png", width_cm=16,
             caption_text="Figure 10. Sharpe vs Compliance -- XGBoost only. "
             "The cloud is wide and the association is not significant (ρ ≈ +0.09, p > 0.05). "
             "xgboost_47 is a clear outlier dominating both axes.")

story += fig("figures/step10h/sharpe/mlp.png", width_cm=16,
             caption_text="Figure 11. Sharpe vs Compliance, MLP only. "
             "No significant association (ρ ≈ +0.15). Best Performance (mlp_14) "
             "sits far from Best SAFE (mlp_30).")

story += [
    sp(0.3),
    body(
        "Looking at the individual model figures, the story becomes more nuanced. "
        "Ridge shows a clear positive trend -- more compliant Ridge configurations "
        "genuinely tend to have higher Sharpe. XGBoost, on the other hand, shows "
        "no significant association across its 50 configurations: the cloud is "
        "flat, and xgboost_47 looks like an outlier rather than the result of a "
        "systematic pattern. MLP is similar to XGBoost, the trend is nearly flat "
        "and not statistically significant."
    ),
    sp(0.3),
    h2("4.2  Maximum Drawdown"),
    sp(0.2),
]

story += fig("figures/step10h/max_drawdown/all_models.png", width_cm=16,
             caption_text="Figure 12. SAFE Compliance vs Maximum Drawdown - all models. "
             "Lower is better on y-axis. ρ ≈ −0.35, p = 0.000. Higher compliance "
             "is associated with lower drawdown on average.")

story += fig("figures/step10h/max_drawdown/ridge.png", width_cm=16,
             caption_text="Figure 13. Maximum Drawdown vs Compliance, Ridge only. "
             "Counterintuitive result: ρ ≈ +0.39, p = 0.005. More compliant Ridge "
             "configurations have higher drawdown. Best Perf (ridge_49) is the least "
             "compliant configuration.")

story += fig("figures/step10h/max_drawdown/mlp.png", width_cm=16,
             caption_text="Figure 14. Maximum Drawdown vs Compliance -- MLP only. "
             "Strongest result in this metric: ρ ≈ −0.51, p = 0.000. Higher compliance "
             "strongly associates with lower drawdown.")

story += [
    sp(0.3),
    body(
        "The maximum drawdown results are mixed. At the pooled level (Figure 12), "
        "the association is negative and significant - higher compliance tends to "
        "come with lower drawdown. But this hides important heterogeneity. "
        "For Ridge (Figure 13), the trend actually goes the wrong way: more "
        "compliant Ridge configurations have larger drawdowns. The best-performing "
        "Ridge configuration on drawdown (ridge_49) is also one of the least "
        "compliant. This is the most interesting anomaly in the dataset. "
        "For MLP (Figure 14), the opposite is true: the trend is strongly negative "
        "and significant, meaning MLP configurations that score high on compliance "
        "also tend to protect against large drawdowns."
    ),
    sp(0.3),
    h2("4.3  Average Turnover"),
    sp(0.2),
]

story += fig("figures/step10h/avg_turnover/all_models.png", width_cm=16,
             caption_text="Figure 15. SAFE Compliance vs Average Turnover -- all models. "
             "Lower is better on y-axis. ρ ≈ +0.61, p = 0.000. Higher compliance "
             "is associated with higher (worse) turnover for XGBoost and MLP.")

story += fig("figures/step10h/avg_turnover/xgboost.png", width_cm=16,
             caption_text="Figure 16. Turnover vs Compliance - XGBoost only. "
             "ρ ≈ +0.78, p = 0.000. Very strong trade-off: xgboost_47 is the most "
             "compliant but among the highest-turnover configurations.")

story += fig("figures/step10h/avg_turnover/ridge.png", width_cm=16,
             caption_text="Figure 17. Turnover vs Compliance—Ridge only. "
             "ρ ≈ −0.34 (Arithmetic), p = 0.016. Ridge is the only model where "
             "higher compliance associates with lower (better) turnover.")

story += [
    sp(0.3),
    body(
        "Turnover is where the sharpest trade-off appears. For XGBoost and MLP, "
        "higher compliance strongly predicts higher turnover (ρ ≈ +0.78 and +0.77 "
        "respectively, both highly significant). This means that the configurations "
        "the SAFE framework prefers are also the ones that trade more aggressively, "
        "which would reduce net returns in a real implementation after transaction "
        "costs. xgboost_47, the Pareto-dominant configuration on Sharpe - turns "
        "out to have one of the highest turnover values in the XGBoost family."
    ),
    body(
        "Ridge is the only exception: its compliance score is actually slightly "
        "negatively correlated with turnover, meaning more compliant Ridge "
        "configurations trade a little less. This is consistent with Ridge's "
        "nature as a regularised linear model that tends to produce smoother, "
        "more stable predictions."
    ),
    PageBreak(),
]

story += [
    h1("5.  Summary and Main Findings"),
    hr(),
    sp(0.3),
    body(
        "The table below summarises the direction of the SAFE–performance "
        "association for each model and each metric. A '+' means higher compliance "
        "tends to produce better performance (significant); '−' means a trade-off "
        "exists; '~' means no significant relationship was found."
    ),
    sp(0.3),
    make_table(
        ["", "Sharpe", "Max Drawdown", "Avg Turnover"],
        [
            ["Ridge",   "+ (sig.)",  "− (sig., wrong dir.)", "+ weak (sig.)"],
            ["XGBoost", "~ (n.s.)",  "~ (borderline)",       "− strong (sig.)"],
            ["MLP",     "~ (n.s.)",  "+ strong (sig.)",      "− strong (sig.)"],
            ["Pooled",  "+ (sig.)",  "+ (sig.)",              "− (sig.)"],
        ],
        col_widths=[3.5*cm, 4*cm, 4.5*cm, 4.5*cm]
    ),
    cap("Table 5. Direction of compliance–performance association. + = higher compliance "
        "associates with better performance. − = trade-off. ~ = not significant. "
        "For Max Drawdown and Turnover, lower is better, so a negative Spearman ρ "
        "means higher compliance → better portfolio."),
    sp(0.5),
    body(
        "The main finding of this empirical work is that SAFE compliance and "
        "portfolio performance are not universally aligned, but the relationship "
        "is not uniformly negative either. The answer depends strongly on the "
        "model family and the performance metric considered."
    ),
    body(
        "The clearest alignment is for Ridge on Sharpe and MLP on drawdown. "
        "The clearest trade-off is for XGBoost and MLP on turnover. The pooled "
        "analysis (all 150 configurations together) shows a significant positive "
        "association on Sharpe and drawdown, but this is partly driven by the "
        "fact that XGBoost configurations -- which are on average more compliant "
        "and also better performing on Sharpe, pull the pooled correlation upward."
    ),
    body(
        "The most remarkable individual result is xgboost_47, which is the only "
        "configuration in the entire 150-configuration sample that simultaneously "
        "achieves the highest compliance score and the highest Sharpe ratio. "
        "This makes it a Pareto-dominant point on the SAFE–performance frontier "
        "for that metric. However, it also has one of the highest turnover values "
        "in the XGBoost family, which would need to be accounted for in a "
        "practical implementation."
    ),
    sp(0.5),
    note(
        "Note on methodology. The LOWESS curves in step 10h are fitted "
        "separately for each model family (not on the pooled sample), using a "
        "bandwidth parameter frac = 0.55 throughout, which means each local "
        "estimate uses approximately 27 of the 50 available points. When the "
        "compliance score range for a family is narrower than 10 percentage "
        "points, the curve falls back to a linear trend to avoid instability. "
        "The Spearman ρ is used rather than Pearson correlation because we do "
        "not assume a linear relationship or normally distributed variables."
    ),
]

# ── Build PDF ─────────────────────────────────────────────────────────────────
out_path = p("report_steps7_8_10c_10h.pdf")
doc = SimpleDocTemplate(
    out_path,
    pagesize=A4,
    leftMargin=2.5*cm, rightMargin=2.5*cm,
    topMargin=2.5*cm, bottomMargin=2.5*cm,
    title="SAFE AI Evaluation - Progress Report",
    author="Anila Vata",
)
doc.build(story)
print(f"\nReport saved to: {out_path}")
