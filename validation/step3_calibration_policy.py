"""
Validation Step 3 -- Calibration & Policy Check
================================================
Uses the TSTR predictions from Step 2 to:
  1. Plot reliability diagrams (observed default rate vs mean predicted PD)
     for all four experiments.
  2. Apply the capstone break-even policy rule to TSTR predictions on UCI
     and compare approval / default rates.
  3. Report Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

Outputs
-------
  validation/results/calibration_tstr.png       reliability diagram (all experiments)
  validation/results/policy_comparison.png      approval/default bar chart
  validation/results/experiment_results.json    updated with calibration & policy fields
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import BORROWER_PROFILES_PATH, RANDOM_STATE, TEST_SIZE
from src.underwriting_model import break_even_pd, build_features

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Default values used for UCI break-even calculation
# (UCI records don't have interest_rate / tenure; we use capstone medians)
UCI_RATE = 22.0
UCI_TENURE = 12


def reliability_diagram(y_true, y_prob, n_bins=10, label="", ax=None, color="steelblue"):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(prob_pred, prob_true, "s-", label=label, color=color)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Reliability Diagram")
    ax.legend(fontsize=8)
    return ax


def calibration_metrics(y_true, y_prob, n_bins=10):
    """ECE and MCE."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    ece, mce = 0.0, 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() > 0:
            diff = abs(y_true[mask].mean() - y_prob[mask].mean())
            ece += mask.sum() / n * diff
            mce = max(mce, diff)
    return round(ece, 5), round(mce, 5)


def policy_check(y_true, y_prob, rate=UCI_RATE, tenure=UCI_TENURE):
    """Apply break-even policy and return approval / default stats."""
    be = break_even_pd(rate, tenure)
    approve = y_prob < be
    n = len(y_true)
    n_approved = int(approve.sum())
    n_declined = n - n_approved
    dr_approved = float(y_true[approve].mean()) if n_approved > 0 else float("nan")
    dr_declined = float(y_true[~approve].mean()) if n_declined > 0 else float("nan")
    return {
        "break_even_pd": round(float(be), 4),
        "approval_rate": round(n_approved / n, 4),
        "default_rate_approved": round(dr_approved, 4) if not np.isnan(dr_approved) else None,
        "default_rate_declined": round(dr_declined, 4) if not np.isnan(dr_declined) else None,
        "n_approved": n_approved,
        "n_declined": n_declined,
    }


def run_calibration(preds=None, results=None):
    """
    Parameters
    ----------
    preds   : dict with keys p_ss, y_s_te, p_uu, y_u_te, p_su, p_us
              If None, re-runs step 2 to generate them.
    results : existing results dict. If None, loads from JSON.
    """
    print("=" * 60)
    print("  Step 3: Calibration & Policy Check")
    print("=" * 60)

    # -- Load pre-computed predictions from step 2 or re-run ------------------
    if preds is None:
        mapped_path = os.path.join(RESULTS_DIR, "uci_mapped.csv")
        if not os.path.exists(mapped_path):
            raise FileNotFoundError("Run step1_feature_crosswalk.py first.")
        sys.path.insert(0, os.path.dirname(__file__))
        from step2_tstr_trts import run_experiments
        print("\n  Re-running step 2 to get predictions ...")
        results, preds = run_experiments()

    p_ss, y_s_te = preds["p_ss"], preds["y_s_te"]
    p_uu, y_u_te = preds["p_uu"], preds["y_u_te"]
    p_su = preds["p_su"]   # TSTR predictions on UCI
    p_us = preds["p_us"]   # TRTS predictions on synth

    if results is None:
        json_path = os.path.join(RESULTS_DIR, "experiment_results.json")
        with open(json_path) as f:
            results = json.load(f)

    # -- Calibration ----------------------------------------------------------
    print("\n  Computing calibration metrics ...")
    cal_data = {
        "S->S": (y_s_te, p_ss),
        "U->U": (y_u_te, p_uu),
        "S->U (TSTR)": (y_u_te, p_su),
        "U->S (TRTS)": (y_s_te, p_us),
    }
    colors = {"S->S": "#4C72B0", "U->U": "#DD8452", "S->U (TSTR)": "#55A868", "U->S (TRTS)": "#C44E52"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for idx, (exp_label, (y_true, y_prob)) in enumerate(cal_data.items()):
        ece, mce = calibration_metrics(y_true, y_prob)
        # Map label to results key (may use arrow notation)
        rkey = exp_label.replace("->", "->")
        for k in results:
            if k.replace("->", "->") == rkey or k == exp_label:
                results[k]["ece"] = ece
                results[k]["mce"] = mce
                break
        reliability_diagram(y_true, y_prob, label=exp_label, ax=axes[idx], color=colors[exp_label])
        axes[idx].set_title(f"{exp_label}\nECE={ece:.4f}  MCE={mce:.4f}")
        print(f"    {exp_label}: ECE={ece:.4f}  MCE={mce:.4f}")

    fig.suptitle("Reliability Diagrams -- IntelliCredit Validation", fontsize=13)
    fig.tight_layout()
    cal_path = os.path.join(RESULTS_DIR, "calibration_tstr.png")
    fig.savefig(cal_path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved calibration plot -> {cal_path}")

    # -- Policy check ---------------------------------------------------------
    print("\n  Running policy check (break-even rule) ...")

    # Synthetic test: use actual rates from capstone data
    df_synth = pd.read_csv(BORROWER_PROFILES_PATH)
    si_tr, si_te = train_test_split(np.arange(len(df_synth)), test_size=TEST_SIZE,
                                     stratify=df_synth["loan_default_status"], random_state=RANDOM_STATE)
    rates_s = df_synth.loc[si_te, "interest_rate_annual"].values
    tenures_s = df_synth.loc[si_te, "loan_tenure_months"].values

    be_s = break_even_pd(rates_s, tenures_s)
    approve_ss = p_ss < be_s
    approve_us = p_us < be_s

    policy_ss = {
        "approval_rate": round(float(approve_ss.mean()), 4),
        "default_rate_approved": round(float(y_s_te[approve_ss].mean()), 4),
        "default_rate_declined": round(float(y_s_te[~approve_ss].mean()), 4),
    }
    policy_us = {
        "approval_rate": round(float(approve_us.mean()), 4),
        "default_rate_approved": round(float(y_s_te[approve_us].mean()), 4) if approve_us.sum() > 0 else None,
        "default_rate_declined": round(float(y_s_te[~approve_us].mean()), 4) if (~approve_us).sum() > 0 else None,
    }

    # UCI test: imputed rate / tenure
    policy_su = policy_check(y_u_te, p_su, rate=UCI_RATE, tenure=UCI_TENURE)
    policy_uu = policy_check(y_u_te, p_uu, rate=UCI_RATE, tenure=UCI_TENURE)

    policy_all = {
        "S->S": policy_ss,
        "U->U": policy_uu,
        "S->U (TSTR)": policy_su,
        "U->S (TRTS)": policy_us,
    }
    for k, v in policy_all.items():
        print(f"    {k}: approval={v['approval_rate']:.3f}  "
              f"DR(approved)={v.get('default_rate_approved', 'N/A')}  "
              f"DR(declined)={v.get('default_rate_declined', 'N/A')}")
        # Update matching results key
        for rk in results:
            if rk == k or rk.replace("->", "->") == k:
                results[rk]["policy"] = v
                break

    # -- Policy bar chart -----------------------------------------------------
    exp_labels = list(policy_all.keys())
    approval_rates = [policy_all[k]["approval_rate"] for k in exp_labels]
    dr_approved = [policy_all[k].get("default_rate_approved") or 0 for k in exp_labels]

    x = np.arange(len(exp_labels))
    width = 0.35
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.bar(x - width / 2, approval_rates, width, label="Approval Rate", color="#4C72B0")
    ax2.bar(x + width / 2, dr_approved, width, label="Default Rate (Approved)", color="#C44E52")
    ax2.set_xticks(x)
    ax2.set_xticklabels(exp_labels, rotation=10, ha="right")
    ax2.set_ylabel("Rate")
    ax2.set_title("Policy Outcomes by Experiment\n(Break-Even Decision Rule)")
    ax2.legend()
    ax2.set_ylim(0, 1)
    fig2.tight_layout()
    pol_path = os.path.join(RESULTS_DIR, "policy_comparison.png")
    fig2.savefig(pol_path, dpi=150)
    plt.close(fig2)
    print(f"  Saved policy comparison chart -> {pol_path}")

    # -- Update JSON ----------------------------------------------------------
    json_path = os.path.join(RESULTS_DIR, "experiment_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Updated experiment_results.json -> {json_path}")

    return results, policy_all


if __name__ == "__main__":
    run_calibration()
