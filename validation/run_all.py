"""
Validation -- Orchestrator
==========================
Runs all four validation steps in order and prints a final summary table.

Usage
-----
    python validation/run_all.py      (from repo root)
    python run_all.py                 (from inside validation/)

Output
------
  All artefacts written to validation/results/
  Final summary printed to stdout and saved to results/run_summary.txt
"""

import json
import os
import sys
import time
from datetime import datetime

# Ensure repo root is importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, VAL_DIR)

RESULTS_DIR = os.path.join(VAL_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DIVIDER = "=" * 65


def section(title):
    """Print a banner so each pipeline stage is easy to spot in the console output."""
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def main():
    """Run validation steps 1-4 in order and print a timed run summary."""
    start_all = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{DIVIDER}")
    print(f"  IntelliCredit Validation Pipeline")
    print(f"  Started: {timestamp}")
    print(DIVIDER)

    # -- Step 1: Feature Crosswalk --------------------------------------------
    section("Step 1 / 4 -- Feature Crosswalk  (UCI -> capstone schema)")
    t0 = time.time()
    from step1_feature_crosswalk import main as step1
    step1()
    print(f"  [OK] Done in {time.time() - t0:.1f}s")

    # -- Step 2: TSTR / TRTS --------------------------------------------------
    section("Step 2 / 4 -- TSTR / TRTS Cross-Train Experiments")
    t0 = time.time()
    from step2_tstr_trts import run_experiments
    exp_results, preds = run_experiments()
    print(f"  [OK] Done in {time.time() - t0:.1f}s")

    # -- Step 3: Calibration & Policy -----------------------------------------
    section("Step 3 / 4 -- Calibration & Policy Check")
    t0 = time.time()
    from step3_calibration_policy import run_calibration
    exp_results, policy_all = run_calibration(preds=preds, results=exp_results)
    print(f"  [OK] Done in {time.time() - t0:.1f}s")

    # -- Step 4: Fidelity Audit -----------------------------------------------
    section("Step 4 / 4 -- Synthetic Data Fidelity Audit")
    t0 = time.time()
    from step4_fidelity_audit import run_fidelity_audit
    fidelity = run_fidelity_audit()
    print(f"  [OK] Done in {time.time() - t0:.1f}s")

    # -- Final Summary --------------------------------------------------------
    section("FINAL SUMMARY")

    json_path = os.path.join(RESULTS_DIR, "experiment_results.json")
    with open(json_path) as f:
        results = json.load(f)

    summary_lines = [
        f"IntelliCredit Validation Run  |  {timestamp}",
        "",
        "-- Transfer Experiments ------------------------------------------",
        f"{'Experiment':<22} {'AUC':>7} {'PR-AUC':>8} {'KS':>7} {'Brier':>8} {'ECE':>8}",
        "-" * 65,
    ]

    for exp in ["S->S", "U->U", "S->U (TSTR)", "U->S (TRTS)", "S->U ablated"]:
        # Try both arrow styles
        r = results.get(exp) or results.get(exp.replace("->", "->")) or {}
        line = (
            f"{exp:<22} "
            f"{str(r.get('auc', '-')):>7} "
            f"{str(r.get('pr_auc', '-')):>8} "
            f"{str(r.get('ks', '-')):>7} "
            f"{str(r.get('brier', '-')):>8} "
            f"{str(r.get('ece', '-')):>8}"
        )
        summary_lines.append(line)

    auc_gap = results.get("_summary", {}).get("auc_gap_SS_vs_TSTR", "N/A")
    interp = results.get("_summary", {}).get("interpretation", "")
    summary_lines += [
        "",
        f"AUC Gap (S->S vs TSTR): {auc_gap}  ->  {interp}",
        "",
        "-- Policy Outcomes (break-even rule) --------------------------------",
        f"{'Experiment':<22} {'Approval':>10} {'DR(Approved)':>14} {'DR(Declined)':>14}",
        "-" * 65,
    ]

    for exp in ["S->S", "U->U", "S->U (TSTR)", "U->S (TRTS)"]:
        r = results.get(exp) or results.get(exp.replace("->", "->")) or {}
        p = r.get("policy", {})
        line = (
            f"{exp:<22} "
            f"{str(p.get('approval_rate', '-')):>10} "
            f"{str(p.get('default_rate_approved', '-')):>14} "
            f"{str(p.get('default_rate_declined', '-')):>14}"
        )
        summary_lines.append(line)

    summary_lines += [
        "",
        "-- Fidelity (Synthetic vs UCI) --------------------------------------",
        f"Frobenius dist (corr matrices): {fidelity['correlation_matrix_frobenius_distance']}",
        f"DCR mean: {fidelity['dcr']['mean']}   DCR p5: {fidelity['dcr']['p5']}   "
        f"Privacy: {fidelity['dcr']['privacy_flag']}",
        f"High-KS features: {fidelity['overall_fidelity']['high_ks_features']}",
        f"Good-KS features:  {fidelity['overall_fidelity']['good_ks_features']}",
        "",
        "-- Outputs -----------------------------------------------------------",
        "  validation/results/uci_mapped.csv",
        "  validation/results/crosswalk_sample.csv",
        "  validation/results/experiment_results.json",
        "  validation/results/auc_comparison.png",
        "  validation/results/ks_comparison.png",
        "  validation/results/calibration_tstr.png",
        "  validation/results/policy_comparison.png",
        "  validation/results/fidelity_ks_wasserstein.csv",
        "  validation/results/fidelity_corr_synth.png",
        "  validation/results/fidelity_corr_uci.png",
        "  validation/results/fidelity_dcr_hist.png",
        "  validation/results/fidelity_summary.json",
        "",
        f"Total runtime: {time.time() - start_all:.1f}s",
    ]

    print("\n" + "\n".join(summary_lines))

    summary_path = os.path.join(RESULTS_DIR, "run_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"\n  Run summary saved -> {summary_path}")


if __name__ == "__main__":
    main()
