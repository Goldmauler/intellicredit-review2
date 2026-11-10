"""
Validation Step 4 -- Synthetic Data Fidelity Audit
===================================================
Checks how well the synthetic capstone dataset matches the UCI reference on
the features that *can* be mapped.  Uses:
  - Kolmogorov–Smirnov (KS) test per feature
  - Wasserstein (Earth Mover's) distance per feature
  - Correlation matrix comparison (Frobenius distance)
  - Privacy proxy: distance-to-closest-record (DCR) on a random sample

Outputs
-------
  validation/results/fidelity_ks_wasserstein.csv    per-feature KS / Wasserstein
  validation/results/fidelity_corr_synth.png        synth correlation heatmap
  validation/results/fidelity_corr_uci.png          UCI correlation heatmap
  validation/results/fidelity_dcr_hist.png          DCR histogram
  validation/results/fidelity_summary.json          summary statistics
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp, wasserstein_distance

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import BORROWER_PROFILES_PATH
from src.underwriting_model import build_features

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Features that exist in both datasets (non-imputed or meaningfully mapped)
COMPARABLE_FEATURES = [
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
]

DCR_SAMPLE_SIZE = 500   # records sampled for DCR (full 30 k is slow)
DCR_N_REAL = 500        # real rows to compare against


def run_fidelity_audit():
    """Compare synthetic vs UCI feature distributions (KS, Wasserstein, correlations, DCR) and save results."""
    print("=" * 60)
    print("  Step 4: Synthetic Data Fidelity Audit")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n  Loading synthetic data …")
    df_synth_raw = pd.read_csv(BORROWER_PROFILES_PATH)
    X_synth = build_features(df_synth_raw)

    print("  Loading UCI mapped data …")
    mapped_path = os.path.join(RESULTS_DIR, "uci_mapped.csv")
    if not os.path.exists(mapped_path):
        raise FileNotFoundError("Run step1_feature_crosswalk.py first.")
    df_uci = pd.read_csv(mapped_path)
    X_uci = build_features(df_uci)

    # ── KS + Wasserstein per feature ──────────────────────────────────────────
    print("\n  Computing per-feature KS and Wasserstein distances …")
    rows = []
    for feat in COMPARABLE_FEATURES:
        if feat not in X_synth.columns or feat not in X_uci.columns:
            print(f"    [WARN] Feature '{feat}' not found -- skipping")
            continue
        s = X_synth[feat].dropna().values
        u = X_uci[feat].dropna().values
        if len(s) == 0 or len(u) == 0:
            print(f"    [WARN] Feature '{feat}' has no values after dropna -- skipping")
            continue
        # Normalise to [0, 1] for Wasserstein comparability
        lo, hi = min(s.min(), u.min()), max(s.max(), u.max())
        rng = hi - lo if hi != lo else 1.0
        s_norm = (s - lo) / rng
        u_norm = (u - lo) / rng
        ks_stat, ks_pval = ks_2samp(s, u)
        wass = wasserstein_distance(s_norm, u_norm)
        rows.append({
            "feature": feat,
            "synth_mean": round(float(s.mean()), 4),
            "uci_mean": round(float(u.mean()), 4),
            "synth_std": round(float(s.std()), 4),
            "uci_std": round(float(u.std()), 4),
            "ks_statistic": round(ks_stat, 4),
            "ks_pvalue": round(ks_pval, 6),
            "wasserstein_normalized": round(wass, 5),
            "fidelity_flag": "[WARN] HIGH" if ks_stat > 0.3 else ("OK" if ks_stat < 0.15 else "MODERATE"),
        })
        print(f"    {feat:35s}  KS={ks_stat:.3f}  W={wass:.4f}")

    df_fid = pd.DataFrame(rows)
    fid_path = os.path.join(RESULTS_DIR, "fidelity_ks_wasserstein.csv")
    df_fid.to_csv(fid_path, index=False)
    print(f"\n  Saved per-feature fidelity table -> {fid_path}")

    # ── Correlation matrix comparison ─────────────────────────────────────────
    print("\n  Comparing correlation matrices …")
    feats_in_both = [f for f in COMPARABLE_FEATURES if f in X_synth.columns and f in X_uci.columns]
    corr_synth = X_synth[feats_in_both].corr()
    corr_uci = X_uci[feats_in_both].corr()
    frob_dist = float(np.linalg.norm(corr_synth.values - corr_uci.values, "fro"))
    print(f"    Frobenius distance between correlation matrices: {frob_dist:.4f}")

    for label, corr, cmap, fname in [
        ("Synthetic", corr_synth, "Blues", "fidelity_corr_synth.png"),
        ("UCI (mapped)", corr_uci, "Oranges", "fidelity_corr_uci.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, ax=ax,
                    xticklabels=[f.replace("_", "\n") for f in feats_in_both],
                    yticklabels=[f.replace("_", "\n") for f in feats_in_both],
                    annot_kws={"size": 7})
        ax.set_title(f"Feature Correlation Matrix -- {label}")
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, fname), dpi=150)
        plt.close(fig)
        print(f"    Saved {fname}")

    # ── Distance-to-closest-record (DCR) ──────────────────────────────────────
    print("\n  Computing Distance-to-Closest-Record (privacy proxy) …")
    rng = np.random.default_rng(42)
    synth_sample = X_synth[feats_in_both].dropna().sample(n=min(DCR_SAMPLE_SIZE, len(X_synth)), random_state=42)
    uci_sample = X_uci[feats_in_both].dropna().sample(n=min(DCR_N_REAL, len(X_uci)), random_state=42)

    # Normalise jointly
    all_data = pd.concat([synth_sample, uci_sample], axis=0)
    mins = all_data.min()
    maxs = all_data.max()
    rngs = (maxs - mins).replace(0, 1)
    synth_norm = ((synth_sample - mins) / rngs).values
    uci_norm = ((uci_sample - mins) / rngs).values

    # DCR: for each synthetic record, find closest real record
    dists = cdist(synth_norm, uci_norm, metric="euclidean")
    dcr = dists.min(axis=1)
    dcr_mean = float(dcr.mean())
    dcr_p5 = float(np.percentile(dcr, 5))
    print(f"    DCR mean: {dcr_mean:.4f}   5th-pct: {dcr_p5:.4f}")
    if dcr_p5 < 0.05:
        print("    [WARN] Some synthetic rows sit very close to real records (possible memorisation).")
    else:
        print("    [OK] Synthetic rows are adequately distant from real records.")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(dcr, bins=40, color="#4C72B0", edgecolor="white")
    ax.axvline(dcr_p5, color="red", linestyle="--", label=f"5th pct={dcr_p5:.3f}")
    ax.axvline(dcr_mean, color="orange", linestyle="--", label=f"Mean={dcr_mean:.3f}")
    ax.set_xlabel("Distance to Closest Real Record (normalized)")
    ax.set_ylabel("Count (synthetic records)")
    ax.set_title("DCR Distribution -- Privacy Proxy\n(lower = closer to real data)")
    ax.legend()
    fig.tight_layout()
    dcr_path = os.path.join(RESULTS_DIR, "fidelity_dcr_hist.png")
    fig.savefig(dcr_path, dpi=150)
    plt.close(fig)
    print(f"    Saved DCR histogram -> {dcr_path}")

    # ── Summary JSON ──────────────────────────────────────────────────────────
    summary = {
        "features_compared": feats_in_both,
        "features_missing": [f for f in COMPARABLE_FEATURES if f not in feats_in_both],
        "per_feature": df_fid.set_index("feature").to_dict(orient="index"),
        "correlation_matrix_frobenius_distance": round(frob_dist, 4),
        "dcr": {
            "sample_size_synthetic": len(synth_sample),
            "sample_size_real": len(uci_sample),
            "mean": round(dcr_mean, 4),
            "p5": round(dcr_p5, 4),
            "privacy_flag": "[WARN] Possible memorisation" if dcr_p5 < 0.05 else "OK",
        },
        "overall_fidelity": {
            "high_ks_features": df_fid[df_fid["ks_statistic"] > 0.3]["feature"].tolist(),
            "moderate_ks_features": df_fid[(df_fid["ks_statistic"] >= 0.15) & (df_fid["ks_statistic"] <= 0.3)]["feature"].tolist(),
            "good_ks_features": df_fid[df_fid["ks_statistic"] < 0.15]["feature"].tolist(),
        },
    }
    summ_path = os.path.join(RESULTS_DIR, "fidelity_summary.json")
    with open(summ_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved fidelity summary -> {summ_path}")
    return summary


if __name__ == "__main__":
    run_fidelity_audit()

