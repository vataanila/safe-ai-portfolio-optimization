"""
run_tradeoff_experiments.py

Automates the extraction of (Net Sharpe, MDD, Turnover, CS_geometric) data
points for the Trade-off Curves by forcing specific hyperparameters in each
step5 script.

Experiment grid
---------------
Ridge   : alpha              in [100.0, 10.0, 1.0, 0.1]
XGBoost : max_depth          in [2, 3, 4, 6]
MLP     : hidden_layer_sizes in [(16,), (32,), (64,), (128,)]

Output
------
data/results/step7/tradeoff_data.json
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent.resolve()

# Force UTF-8 I/O in every subprocess so Greek/Unicode chars in step6 don't crash
_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}

STEP7_SCRIPTS = [
    "7a_prepare_panel.py",
    "7b_rga.py",
    "7c_rge.py",
    "7d_rgr.py",
    "7e_rgf.py",
    "7f_summary.py",
    "8a_rga_vector.py",
    "8b_rgr_vector.py",
    "8c_rge_vector.py",
    "8d_rgf_vector.py",
    "8e_compliance_score.py",
]

def run_script(script_name: str) -> None:
    print(f"    running {script_name} ...", flush=True)
    result = subprocess.run(
        [sys.executable, str(BASE_DIR / script_name)],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        env=_ENV,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{script_name} failed (exit {result.returncode}).\n"
            f"--- STDOUT (last 3000 chars) ---\n{result.stdout[-3000:]}\n"
            f"--- STDERR (last 3000 chars) ---\n{result.stderr[-3000:]}"
        )

def _perf_dir(model: str) -> Path:
    return BASE_DIR / "data" / "results" / "step6" / model / "performance"

def read_net_sharpe(model: str) -> float:
    df = pd.read_csv(_perf_dir(model) / f"{model}_net_cost_summary.csv")
    row = df[df["cost_bps"] == 20]
    if row.empty:
        raise ValueError(f"No cost_bps=20 row for {model}")
    return float(row["sharpe"].iloc[0])

def read_mdd(model: str) -> float:
    """Max drawdown from the 20 bps net-cost scenario (positive fraction, e.g. 0.23)."""
    df = pd.read_csv(_perf_dir(model) / f"{model}_net_cost_summary.csv")
    row = df[df["cost_bps"] == 20]
    if row.empty:
        raise ValueError(f"No cost_bps=20 row for {model}")
    return float(row["max_drawdown"].iloc[0])

def read_turnover(model: str) -> float:
    """Average monthly (rebalancing) turnover from the gross summary."""
    df = pd.read_csv(_perf_dir(model) / f"{model}_summary.csv")
    return float(df["avg_turnover"].iloc[0])

def read_cs_geometric(model: str) -> float:
    csv_path = BASE_DIR / "data" / "results" / "step8" / "compliance_score_4d.csv"
    df = pd.read_csv(csv_path)
    row = df[df["model"] == model]
    if row.empty:
        raise ValueError(f"No row for model={model} in compliance_score_4d.csv")
    return float(row["CS4_geometric"].iloc[0])

def collect(model: str) -> dict:
    net_sharpe = read_net_sharpe(model)
    mdd        = read_mdd(model)
    turnover   = read_turnover(model)
    cs_geo     = read_cs_geometric(model)
    print(
        f"    net_sharpe={net_sharpe:.4f}  mdd={mdd:.4f}"
        f"  turnover={turnover:.4f}  CS_geometric={cs_geo:.4f}"
    )
    return {
        "net_sharpe": net_sharpe,
        "mdd": mdd,
        "turnover": turnover,
        "cs_geometric": cs_geo,
    }

results: dict = {"ridge": [], "xgboost": [], "mlp": []}

# ─────────────────────────────────────────────────────────────────────────────
# Ridge, vary alpha
# ─────────────────────────────────────────────────────────────────────────────
ridge_path = BASE_DIR / "5a_ridge.py"
ridge_original = ridge_path.read_text(encoding="utf-8")

RIDGE_PATCH_OLD = "for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:"
if RIDGE_PATCH_OLD not in ridge_original:
    raise RuntimeError(f"Ridge patch target not found.\nExpected: {RIDGE_PATCH_OLD!r}")

try:
    for alpha in [100.0, 10.0, 1.0, 0.1]:
        print(f"\n[Ridge] alpha={alpha}", flush=True)
        patched = ridge_original.replace(RIDGE_PATCH_OLD, f"for alpha in [{alpha}]:")
        ridge_path.write_text(patched, encoding="utf-8")

        run_script("5a_ridge.py")
        run_script("6a_ridge_portfolio.py")
        for s in STEP7_SCRIPTS:
            run_script(s)

        entry = collect("ridge")
        entry.update({"param": "alpha", "value": alpha, "label": f"α={alpha}"})
        results["ridge"].append(entry)

finally:
    ridge_path.write_text(ridge_original, encoding="utf-8")
    print("\n[Ridge] original file restored.")

# ─────────────────────────────────────────────────────────────────────────────
# XGBoost—vary max_depth
# ─────────────────────────────────────────────────────────────────────────────
xgb_path = BASE_DIR / "5b_xgboost.py"
xgb_original = xgb_path.read_text(encoding="utf-8")

XGB_PATCH_OLD = "        for depth in [3, 4, 6]:"
if XGB_PATCH_OLD not in xgb_original:
    raise RuntimeError(f"XGBoost patch target not found.\nExpected: {XGB_PATCH_OLD!r}")

try:
    for depth in [2, 3, 4, 6]:
        print(f"\n[XGBoost] max_depth={depth}", flush=True)
        patched = xgb_original.replace(XGB_PATCH_OLD, f"        for depth in [{depth}]:")
        xgb_path.write_text(patched, encoding="utf-8")

        run_script("5b_xgboost.py")
        run_script("6b_xgboost_portfolio.py")
        for s in STEP7_SCRIPTS:
            run_script(s)

        entry = collect("xgboost")
        entry.update({"param": "max_depth", "value": depth, "label": f"max_depth={depth}"})
        results["xgboost"].append(entry)

finally:
    xgb_path.write_text(xgb_original, encoding="utf-8")
    print("\n[XGBoost] original file restored.")

# ─────────────────────────────────────────────────────────────────────────────
# MLP - vary hidden_layer_sizes
# ─────────────────────────────────────────────────────────────────────────────
mlp_path = BASE_DIR / "5c_mlp.py"
mlp_original = mlp_path.read_text(encoding="utf-8")

MLP_PATCH_OLD = "        hidden_layer_sizes = (64, 32),"
if MLP_PATCH_OLD not in mlp_original:
    raise RuntimeError(f"MLP patch target not found.\nExpected: {MLP_PATCH_OLD!r}")

try:
    for hls in [(16,), (32,), (64,), (128,)]:
        print(f"\n[MLP] hidden_layer_sizes={hls}", flush=True)
        patched = mlp_original.replace(MLP_PATCH_OLD, f"        hidden_layer_sizes = {hls},")
        mlp_path.write_text(patched, encoding="utf-8")

        run_script("5c_mlp.py")
        run_script("6c_mlp_portfolio.py")
        for s in STEP7_SCRIPTS:
            run_script(s)

        entry = collect("mlp")
        entry.update({
            "param": "hidden_layer_sizes",
            "value": str(hls),
            "label": f"layers={hls}",
        })
        results["mlp"].append(entry)

finally:
    mlp_path.write_text(mlp_original, encoding="utf-8")
    print("\n[MLP] original file restored.")

# ─────────────────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────────────────
output_path = BASE_DIR / "data" / "results" / "step7" / "tradeoff_data.json"
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\nAll experiments complete. Results saved to:\n  {output_path}")
print(json.dumps(results, indent=2))
