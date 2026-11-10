"""
IntelliCredit - Model Monitoring & Population Stability Index (PSI) Engine
==========================================================================
Provides statistical drift detection and stability auditing for credit scorecards
and alternative underwriting features in production environments.

Key Metrics:
  * Population Stability Index (PSI):
      PSI = sum_{b=1}^B (Actual_b - Expected_b) * ln(Actual_b / Expected_b)
  * Basel / EBA Traffic-Light Standards:
      - Green (PSI < 0.10)       : Insignificant drift; score distribution is stable.
      - Amber (0.10 <= PSI <= 0.25): Moderate drift; investigate incoming population.
      - Red   (PSI > 0.25)       : Significant shift; trigger model recalibration.
  * Characteristic Analysis:
      - Variable-level PSI across numerical and categorical features.
      - Kolmogorov-Smirnov (KS) distribution difference test.

Author: Interactive Dashboard & UI Team
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.config import DATA_DIR, PROCESSED_DIR

TRAIN_DATA_PATH = os.path.join(PROCESSED_DIR, "train.csv")
TEST_DATA_PATH = os.path.join(PROCESSED_DIR, "test.csv")
BORROWER_PROFILES_PATH = os.path.join(DATA_DIR, "borrower_profiles.csv")
SAMPLE_BORROWERS_PATH = os.path.join(DATA_DIR, "sample", "sample_borrower_profiles.csv")

# Basel Traffic Light Regulatory Standards for Credit Scorecard Drift
PSI_THRESHOLD_AMBER = 0.10
PSI_THRESHOLD_RED = 0.25

ALERT_GREEN = "Stable (Green)"
ALERT_AMBER = "Moderate Shift (Amber)"
ALERT_RED = "Significant Drift (Red)"


def classify_psi_status(psi_value: float) -> str:
    """Classifies a PSI value into Basel supervisory traffic-light tiers."""
    if psi_value < PSI_THRESHOLD_AMBER:
        return ALERT_GREEN
    elif psi_value <= PSI_THRESHOLD_RED:
        return ALERT_AMBER
    return ALERT_RED


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> Tuple[float, pd.DataFrame]:
    """
    Calculates Population Stability Index (PSI) between baseline and evaluation samples.

    Parameters:
        expected: Baseline / benchmark distribution (e.g. training set or origination cohort).
        actual: Current / scoring distribution (e.g. recent applicants).
        num_bins: Number of quantiles for binning (default 10).
        epsilon: Small smoothing constant to avoid log(0) and division by zero.

    Returns:
        Tuple of (scalar PSI value, detailed bin breakdown DataFrame).
    """
    exp_clean = np.asarray(expected, dtype=float)
    act_clean = np.asarray(actual, dtype=float)

    # Remove NaNs
    exp_clean = exp_clean[~np.isnan(exp_clean)]
    act_clean = act_clean[~np.isnan(act_clean)]

    if len(exp_clean) == 0 or len(act_clean) == 0:
        return 0.0, pd.DataFrame()

    # Determine quantile bin edges from expected distribution
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(exp_clean, percentiles)
    bin_edges = np.unique(bin_edges)  # Prevent duplicate edges

    if len(bin_edges) < 2:
        return 0.0, pd.DataFrame()

    # Expand outer edges slightly to capture boundary values
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Bin counts
    exp_counts, _ = np.histogram(exp_clean, bins=bin_edges)
    act_counts, _ = np.histogram(act_clean, bins=bin_edges)

    # Convert to proportions with epsilon smoothing
    exp_pct = np.maximum(exp_counts / len(exp_clean), epsilon)
    act_pct = np.maximum(act_counts / len(act_clean), epsilon)

    # Re-normalize after smoothing
    exp_pct /= np.sum(exp_pct)
    act_pct /= np.sum(act_pct)

    # PSI components
    psi_components = (act_pct - exp_pct) * np.log(act_pct / exp_pct)
    total_psi = float(np.sum(psi_components))

    # Detailed bin table
    bin_labels = [f"Bin {i+1}" for i in range(len(psi_components))]
    detail_df = pd.DataFrame({
        "bin": bin_labels,
        "expected_count": exp_counts,
        "actual_count": act_counts,
        "expected_pct": exp_pct,
        "actual_pct": act_pct,
        "psi_contribution": psi_components,
    })

    return round(total_psi, 4), detail_df


def audit_feature_drift(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Evaluates characteristic drift (PSI and KS statistic) across key numerical features.

    Parameters:
        baseline_df: Baseline reference dataset.
        current_df: Evaluation dataset.
        features: List of column names to monitor. If None, auto-selects common numeric columns.

    Returns:
        Summary DataFrame sorted by PSI descending.
    """
    if features is None:
        common_num = baseline_df.select_dtypes(include=[np.number]).columns.intersection(
            current_df.select_dtypes(include=[np.number]).columns
        ).tolist()
        # Filter out IDs or binary flags
        features = [
            c for c in common_num
            if not c.endswith("_id") and not c.startswith("is_") and baseline_df[c].nunique() > 2
        ][:15]

    results = []
    for feat in features:
        if feat not in baseline_df or feat not in current_df:
            continue

        exp_vals = baseline_df[feat].dropna().to_numpy()
        act_vals = current_df[feat].dropna().to_numpy()

        if len(exp_vals) == 0 or len(act_vals) == 0:
            continue

        psi_val, _ = calculate_psi(exp_vals, act_vals)
        ks_stat, ks_pval = stats.ks_2samp(exp_vals, act_vals)
        status = classify_psi_status(psi_val)

        results.append({
            "feature": feat,
            "psi": psi_val,
            "status": status,
            "ks_statistic": round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_pval), 4),
            "baseline_mean": round(float(np.mean(exp_vals)), 2),
            "current_mean": round(float(np.mean(act_vals)), 2),
            "mean_shift_pct": round(
                ((np.mean(act_vals) - np.mean(exp_vals)) / (np.mean(exp_vals) + 1e-6)) * 100.0, 2
            ),
        })

    summary_df = pd.DataFrame(results)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(by="psi", ascending=False).reset_index(drop=True)
    return summary_df


def run_model_monitoring_suite() -> Dict[str, Any]:
    """Runs a complete drift monitoring check using train vs test partitions."""
    b_path = TRAIN_DATA_PATH if os.path.exists(TRAIN_DATA_PATH) else SAMPLE_BORROWERS_PATH
    c_path = TEST_DATA_PATH if os.path.exists(TEST_DATA_PATH) else SAMPLE_BORROWERS_PATH

    baseline_df = pd.read_csv(b_path)
    current_df = pd.read_csv(c_path)

    drift_report = audit_feature_drift(baseline_df, current_df)
    amber_count = int((drift_report["status"] == ALERT_AMBER).sum())
    red_count = int((drift_report["status"] == ALERT_RED).sum())

    return {
        "features_monitored": len(drift_report),
        "stable_green_count": len(drift_report) - amber_count - red_count,
        "amber_alert_count": amber_count,
        "red_alert_count": red_count,
        "top_drifting_feature": drift_report.iloc[0]["feature"] if not drift_report.empty else None,
        "drift_table": drift_report,
    }


if __name__ == "__main__":
    print("IntelliCredit - Executing Model Monitoring & Population Stability Index (PSI)...")
    summary = run_model_monitoring_suite()
    print(f"\nMonitored {summary['features_monitored']} features:")
    print(f"  * Stable (Green) : {summary['stable_green_count']}")
    print(f"  * Amber Alerts   : {summary['amber_alert_count']}")
    print(f"  * Red Alerts     : {summary['red_alert_count']}")
    print("\nTop Feature Drift Summary:")
    print(summary["drift_table"][["feature", "psi", "status", "ks_statistic", "mean_shift_pct"]].head(8))
