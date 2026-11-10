"""
Validation Step 2 -- TSTR / TRTS Cross-Train Experiments
=========================================================
Runs all four cross-train / cross-test combinations:

  S->S  Train on Synthetic, Test on Synthetic  (capstone baseline)
  U->U  Train on UCI,       Test on UCI        (real-data ceiling)
  S->U  Train on Synthetic, Test on UCI        (TSTR – key transfer test)
  U->S  Train on UCI,       Test on Synthetic  (TRTS – reverse transfer)

Also runs a feature-ablation TSTR that keeps only the well-mapped features
(dropping imputed constants) to separate pipeline quality from feature absence.

Outputs
-------
  validation/results/experiment_results.json   all numeric metrics
  validation/results/auc_comparison.png        grouped bar chart
  validation/results/ks_comparison.png         KS statistic bar chart
"""

import json
import os
import sys

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# Allow running from repo root or from validation/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import (
    BORROWER_PROFILES_PATH,
    ORDINAL_MAPS,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.underwriting_model import build_features

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── LightGBM hyperparameters (identical to underwriting_model.py) ─────────────
LGB_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.02,
    num_leaves=15,
    min_child_samples=80,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    random_state=RANDOM_STATE,
    verbose=-1,
)

# Features that are well-mapped from UCI (not imputed constants)
REAL_FEATURES = [
    "age",
    "education_code",
    "avg_monthly_income",
    "income_volatility_index",
    "income_trend_slope",
    "utility_reliability_score",
    "avg_utility_delay_days",
    "max_utility_delay_days",
    "debt_to_inflow_ratio",
    "savings_buffer_ratio",
    "repayment_ratio",
]


def ks_statistic(y_true, y_prob):
    """KS statistic: max separation between TPR and FPR curves."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def compute_metrics(y_true, y_prob, label=""):
    auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    ks = ks_statistic(y_true, y_prob)
    # Expected calibration error (10 bins)
    bins = np.linspace(0, 1, 11)
    bin_ids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for b in range(10):
        mask = bin_ids == b
        if mask.sum() > 0:
            ece += mask.sum() / len(y_true) * abs(y_true[mask].mean() - y_prob[mask].mean())

    result = {
        "label": label,
        "n_test": int(len(y_true)),
        "default_rate_test": round(float(np.mean(y_true)), 4),
        "mean_predicted_pd": round(float(np.mean(y_prob)), 4),
        "auc": round(auc, 4),
        "pr_auc": round(pr_auc, 4),
        "ks": round(ks, 4),
        "brier": round(brier, 5),
        "ece": round(ece, 5),
    }
    return result


def _load_synthetic():
    """Load synthetic capstone data and build model features."""
    df = pd.read_csv(BORROWER_PROFILES_PATH)
    X = build_features(df)
    y = df["loan_default_status"].values
    return X, y


def _load_uci_mapped():
    """Load UCI mapped data (output of step1) and build model features."""
    mapped_path = os.path.join(RESULTS_DIR, "uci_mapped.csv")
    if not os.path.exists(mapped_path):
        raise FileNotFoundError(
            f"Mapped UCI file not found at {mapped_path}.\n"
            "Run step1_feature_crosswalk.py first."
        )
    df = pd.read_csv(mapped_path)
    X = build_features(df)
    y = df["loan_default_status"].values
    return X, y


def train_and_predict(X_train, y_train, X_test):
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1], model


def run_experiments():
    print("=" * 60)
    print("  Step 2: TSTR / TRTS Cross-Train Experiments")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n  Loading synthetic capstone data …")
    X_synth, y_synth = _load_synthetic()
    print(f"    Shape: {X_synth.shape}, Default rate: {y_synth.mean():.3f}")

    print("  Loading UCI mapped data …")
    X_uci, y_uci = _load_uci_mapped()
    print(f"    Shape: {X_uci.shape}, Default rate: {y_uci.mean():.3f}")

    # ── Splits (stratified 80 / 20, seed 42) ─────────────────────────────────
    si_tr, si_te = train_test_split(np.arange(len(y_synth)), test_size=TEST_SIZE,
                                     stratify=y_synth, random_state=RANDOM_STATE)
    ui_tr, ui_te = train_test_split(np.arange(len(y_uci)), test_size=TEST_SIZE,
                                     stratify=y_uci, random_state=RANDOM_STATE)

    X_s_tr, y_s_tr = X_synth.iloc[si_tr], y_synth[si_tr]
    X_s_te, y_s_te = X_synth.iloc[si_te], y_synth[si_te]
    X_u_tr, y_u_tr = X_uci.iloc[ui_tr], y_uci[ui_tr]
    X_u_te, y_u_te = X_uci.iloc[ui_te], y_uci[ui_te]

    results = {}

    # ── S->S  (capstone baseline) ──────────────────────────────────────────────
    print("\n  [1/5] S->S  Train Synthetic -> Test Synthetic …")
    p_ss, m_ss = train_and_predict(X_s_tr, y_s_tr, X_s_te)
    results["S->S"] = compute_metrics(y_s_te, p_ss, "S->S (baseline)")
    print(f"    AUC: {results['S->S']['auc']}  PR-AUC: {results['S->S']['pr_auc']}  KS: {results['S->S']['ks']}")

    # ── U->U  (UCI ceiling) ────────────────────────────────────────────────────
    print("\n  [2/5] U->U  Train UCI -> Test UCI …")
    p_uu, m_uu = train_and_predict(X_u_tr, y_u_tr, X_u_te)
    results["U->U"] = compute_metrics(y_u_te, p_uu, "U->U (UCI ceiling)")
    print(f"    AUC: {results['U->U']['auc']}  PR-AUC: {results['U->U']['pr_auc']}  KS: {results['U->U']['ks']}")

    # ── S->U  (TSTR – key transfer test) ──────────────────────────────────────
    print("\n  [3/5] S->U  Train Synthetic -> Test UCI (TSTR) …")
    p_su, m_su = train_and_predict(X_s_tr, y_s_tr, X_u_te)
    results["S->U (TSTR)"] = compute_metrics(y_u_te, p_su, "S->U (TSTR)")
    print(f"    AUC: {results['S->U (TSTR)']['auc']}  PR-AUC: {results['S->U (TSTR)']['pr_auc']}  KS: {results['S->U (TSTR)']['ks']}")

    # ── U->S  (TRTS – reverse transfer) ───────────────────────────────────────
    print("\n  [4/5] U->S  Train UCI -> Test Synthetic (TRTS) …")
    p_us, m_us = train_and_predict(X_u_tr, y_u_tr, X_s_te)
    results["U->S (TRTS)"] = compute_metrics(y_s_te, p_us, "U->S (TRTS)")
    print(f"    AUC: {results['U->S (TRTS)']['auc']}  PR-AUC: {results['U->S (TRTS)']['pr_auc']}  KS: {results['U->S (TRTS)']['ks']}")

    # ── S->U ablated  (only real features, no imputed constants) ───────────────
    print("\n  [5/5] S->U ablated  TSTR with real features only …")
    real_cols = [c for c in REAL_FEATURES if c in X_synth.columns and c in X_uci.columns]
    p_su_ab, _ = train_and_predict(X_s_tr[real_cols], y_s_tr, X_u_te[real_cols])
    results["S->U ablated"] = compute_metrics(y_u_te, p_su_ab, "S->U ablated (real features only)")
    print(f"    AUC: {results['S->U ablated']['auc']}  PR-AUC: {results['S->U ablated']['pr_auc']}  KS: {results['S->U ablated']['ks']}")
    print(f"    Features used: {real_cols}")

    # ── AUC gap summary ───────────────────────────────────────────────────────
    auc_gap = round(results["S->S"]["auc"] - results["S->U (TSTR)"]["auc"], 4)
    gap_interp = (
        "Good transfer (<0.05)" if auc_gap < 0.05
        else "Moderate mismatch (0.05-0.15)" if auc_gap < 0.15
        else "Strong domain mismatch (>0.15)"
    )
    results["_summary"] = {
        "auc_gap_SS_vs_TSTR": auc_gap,
        "interpretation": gap_interp,
        "note": "AUC gap = S->S AUC minus S->U AUC. High gap = generator or domain mismatch.",
    }
    print(f"\n  AUC gap (S->S vs TSTR): {auc_gap} -> {gap_interp}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    json_path = os.path.join(RESULTS_DIR, "experiment_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved results -> {json_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_auc_comparison(results)
    _plot_ks_comparison(results)

    return results, {
        "p_ss": p_ss, "y_s_te": y_s_te,
        "p_uu": p_uu, "y_u_te": y_u_te,
        "p_su": p_su,
        "p_us": p_us,
    }


def _plot_auc_comparison(results):
    labels = ["S->S", "U->U", "S->U (TSTR)", "U->S (TRTS)", "S->U ablated"]
    auc_vals = [results[k]["auc"] for k in labels]
    pr_vals = [results[k]["pr_auc"] for k in labels]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, auc_vals, width, label="ROC-AUC", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, pr_vals, width, label="PR-AUC", color="#DD8452")

    ax.set_xlabel("Experiment")
    ax.set_ylabel("Score")
    ax.set_title("Cross-Train / Cross-Test AUC Comparison\n(IntelliCredit Pipeline Validation)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)

    for bar in bars1:
        ax.annotate(f"{bar.get_height():.3f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    for bar in bars2:
        ax.annotate(f"{bar.get_height():.3f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8, label="Random (AUC=0.5)")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "auc_comparison.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved AUC comparison chart -> {out}")


def _plot_ks_comparison(results):
    labels = ["S->S", "U->U", "S->U (TSTR)", "U->S (TRTS)", "S->U ablated"]
    ks_vals = [results[k]["ks"] for k in labels]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, ks_vals, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"])
    ax.set_xlabel("Experiment")
    ax.set_ylabel("KS Statistic")
    ax.set_title("KS Statistic by Experiment")
    ax.set_ylim(0, 1)
    for bar in bars:
        ax.annotate(f"{bar.get_height():.3f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "ks_comparison.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved KS comparison chart -> {out}")


if __name__ == "__main__":
    run_experiments()

