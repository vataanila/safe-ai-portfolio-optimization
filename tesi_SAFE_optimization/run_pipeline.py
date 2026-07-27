import subprocess
import time
from datetime import datetime

STEPS = [
    "1a_load.py",
    "2a_preprocess.py",
    "3a_baseline.py",
    "3c_viz_baseline.py",
    "3d_viz_frontier.py",
    "4a_features.py",
    "5a_ridge.py",
    "5b_xgboost.py",
    "5c_mlp.py",
    "6a_ridge_portfolio.py",
    "6b_xgboost_portfolio.py",
    "6c_mlp_portfolio.py",
    "6d_compare_portfolios.py",
    "6e_visualize_portfolio_comparison.py",
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
    "9a_safe_frontier_50x3.py",
]

elapsed_times = {}

print(f"Pipeline started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

for script in STEPS:
    print(f"\n[START] {script}  ({datetime.now().strftime('%H:%M:%S')})")
    t0 = time.time()
    try:
        subprocess.run(["python", script], check=True)
    except subprocess.CalledProcessError:
        elapsed = time.time() - t0
        print(f"\n[FAILED] {script} after {elapsed:.1f}s")
        print("Pipeline aborted.")
        break
    elapsed = time.time() - t0
    elapsed_times[script] = elapsed
    print(f"[DONE]  {script}  ({elapsed:.1f}s)")
else:
    print("\n" + "=" * 60)
    print("Pipeline completed successfully.")

print("\n--- Elapsed times ---")
for script, elapsed in elapsed_times.items():
    print(f"  {script:<45} {elapsed:>8.1f}s")

total = sum(elapsed_times.values())
print(f"\n  {'TOTAL':<45} {total:>8.1f}s")
print(f"\nPipeline ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
