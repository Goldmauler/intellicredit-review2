"""
Validation Step 1 -- Feature Crosswalk
======================================
Maps the UCI Default of Credit Card Clients dataset to the IntelliCredit
capstone feature schema so the same LightGBM pipeline can be run unchanged.

Outputs
-------
  validation/results/uci_mapped.csv        full mapped dataset (all 30 k rows)
  validation/results/crosswalk_sample.csv  first 10 rows for manual inspection

Feature mapping rationale is documented in validation_plan.md.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import linregress

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data_compare", "UCI_Credit_Card.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Capstone medians used for imputation of UCI-absent features ──────────────
IMPUTE = {
    "gig_tenure_months": 36,
    "has_bank_account": 1,          # UCI records are credit-card holders
    "num_dependents": 1,
    "avg_days_between_recharges": 18,
    "avg_app_logins": 12,
    "avg_upi_transactions": 25,
    "loan_tenure_months": 12,
    "interest_rate_annual": 22.0,
    "previous_micro_loans": 0,
}


def _payment_months_on_time(row):
    """Number of months (out of 6) where PAY_i <= 0 (paid on time or early)."""
    pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    return sum(row[c] <= 0 for c in pay_cols)


def _billing_trend_slope(row):
    """OLS slope of BILL_AMT1..6 over 6 months (most recent = month 1)."""
    bill_cols = ["BILL_AMT6", "BILL_AMT5", "BILL_AMT4", "BILL_AMT3", "BILL_AMT2", "BILL_AMT1"]
    vals = np.array([row[c] for c in bill_cols], dtype=float)
    if vals.std() == 0:
        return 0.0
    slope, _, _, _, _ = linregress(np.arange(6), vals)
    return slope


def _billing_volatility(row):
    """Coefficient of variation of monthly bill amounts."""
    bill_cols = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
    vals = np.array([row[c] for c in bill_cols], dtype=float)
    mean = vals.mean()
    return vals.std() / max(abs(mean), 1.0)


def _avg_delay_days(row):
    """Mean delay months (PAY_i > 0) × 30 -> approximate days."""
    pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    delays = [max(row[c], 0) for c in pay_cols]
    return np.mean(delays) * 30.0


def _max_delay_days(row):
    """Worst delay months × 30 -> approximate days."""
    pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    delays = [max(row[c], 0) for c in pay_cols]
    return max(delays) * 30.0


def _payment_ratio(row):
    """Mean monthly payment / mean monthly bill (savings buffer proxy)."""
    bill_cols = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
    pay_cols = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
    mean_bill = np.mean([row[c] for c in bill_cols])
    mean_pay = np.mean([row[c] for c in pay_cols])
    return np.clip(mean_pay / max(abs(mean_bill), 1.0), 0, 5)


def _credit_utilisation(row):
    """Mean bill amount / credit limit -> debt-to-inflow proxy."""
    bill_cols = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
    mean_bill = np.mean([row[c] for c in bill_cols])
    return np.clip(mean_bill / max(row["LIMIT_BAL"], 1.0), 0, 2)


# ── Education mapping ─────────────────────────────────────────────────────────
# UCI: 1=graduate school, 2=university, 3=high school, 4–6=others/unknown
# Capstone: 0=Below 10th, 1=10th, 2=12th, 3=Diploma, 4=UG, 5=PG
UCI_EDU_MAP = {
    0: 2,   # undocumented -> 12th pass
    1: 5,   # graduate school -> Postgraduate
    2: 4,   # university -> Undergraduate
    3: 2,   # high school -> 12th Pass
    4: 1,   # others -> 10th Pass
    5: 1,
    6: 1,
}


def build_uci_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map UCI columns -> capstone model features.

    Parameters
    ----------
    df : raw UCI DataFrame (output of pd.read_csv)

    Returns
    -------
    DataFrame with exactly the columns expected by build_features() in
    underwriting_model.py, plus 'loan_default_status' target column.
    """
    n = len(df)
    out = pd.DataFrame(index=df.index)

    # ── Directly mappable ────────────────────────────────────────────────────
    out["age"] = df["AGE"].clip(18, 80)
    # Map UCI EDUCATION code -> capstone string label (build_features uses ORDINAL_MAPS which expects strings)
    UCI_EDU_TO_LABEL = {
        0: "12th Pass",          # undocumented -> 12th Pass
        1: "Postgraduate",       # graduate school
        2: "Undergraduate",      # university
        3: "12th Pass",          # high school
        4: "10th Pass",          # others
        5: "10th Pass",
        6: "10th Pass",
    }
    out["education_level"] = df["EDUCATION"].map(UCI_EDU_TO_LABEL).fillna("12th Pass")

    # Income proxy: credit limit / 6  (rough monthly income analogue)
    income_proxy = (df["LIMIT_BAL"] / 6).clip(lower=1.0)
    out["avg_monthly_income"] = income_proxy

    # ── Computed proxies ─────────────────────────────────────────────────────
    # On-time payment share (utility reliability proxy)
    pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    on_time_counts = (df[pay_cols] <= 0).sum(axis=1)
    out["utility_on_time_count"] = on_time_counts  # out of 6; capstone divides by 12

    # Delay proxies
    max_delays = df[pay_cols].clip(lower=0).max(axis=1) * 30.0
    mean_delays = df[pay_cols].clip(lower=0).mean(axis=1) * 30.0
    out["avg_utility_delay_days"] = mean_delays
    out["max_utility_delay_days"] = max_delays

    # Billing trend slope
    bill_cols_asc = ["BILL_AMT6", "BILL_AMT5", "BILL_AMT4", "BILL_AMT3", "BILL_AMT2", "BILL_AMT1"]
    bill_matrix = df[bill_cols_asc].values.T.astype(float)      # shape (6, n)
    x = np.arange(6, dtype=float)
    x_mean = x.mean()
    y_mean = bill_matrix.mean(axis=0)                            # (n,)
    cov = ((x[:, None] - x_mean) * (bill_matrix - y_mean)).sum(axis=0)
    var_x = ((x - x_mean) ** 2).sum()
    slopes = np.where(var_x > 0, cov / var_x, 0.0)
    out["income_trend_slope"] = slopes

    # Income volatility index
    bill_std = df[["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]].std(axis=1)
    bill_mean = df[["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]].mean(axis=1).abs()
    out["std_monthly_income"] = bill_std    # used in build_features for volatility ratio

    # Debt-to-inflow: mean bill / credit limit
    mean_bill = df[["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]].mean(axis=1)
    out["avg_bnpl_monthly_amount"] = (mean_bill.clip(lower=0) * income_proxy / df["LIMIT_BAL"].clip(lower=1)).clip(0)
    # This sets up debt_to_inflow_ratio = avg_bnpl_monthly_amount / income_proxy ≈ utilisation

    # Savings buffer: mean pay_amt / mean_outflow
    mean_pay = df[["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]].mean(axis=1)
    out["avg_monthly_outflow"] = mean_bill.clip(lower=1)
    out["avg_eom_balance"] = (mean_pay * np.clip(mean_bill / mean_bill.clip(lower=1), 0, 5)).clip(lower=0)

    # Repayment history proxy
    out["previous_micro_loans"] = IMPUTE["previous_micro_loans"]
    out["previous_loans_repaid_on_time"] = 0

    # ── Imputed constants (gig-specific, absent from UCI) ────────────────────
    out["gig_tenure_months"] = IMPUTE["gig_tenure_months"]
    out["has_bank_account"] = IMPUTE["has_bank_account"]
    out["num_dependents"] = IMPUTE["num_dependents"]
    out["location_type"] = "Tier-2 City"          # middle of ordinal scale
    out["avg_days_between_recharges"] = IMPUTE["avg_days_between_recharges"]
    out["avg_app_logins"] = IMPUTE["avg_app_logins"]
    out["avg_upi_transactions"] = IMPUTE["avg_upi_transactions"]
    out["loan_amount_requested"] = income_proxy * 6  # 6 months income as loan proxy
    out["loan_tenure_months"] = IMPUTE["loan_tenure_months"]
    out["interest_rate_annual"] = IMPUTE["interest_rate_annual"]

    # ── Target ───────────────────────────────────────────────────────────────
    out["loan_default_status"] = df["default.payment.next.month"].values

    return out


def main():
    print("=" * 60)
    print("  Step 1: Feature Crosswalk -- UCI -> IntelliCredit Schema")
    print("=" * 60)

    print(f"\n  Loading UCI data from: {DATA_PATH}")
    df_raw = pd.read_csv(DATA_PATH)
    print(f"  Raw shape: {df_raw.shape}")
    print(f"  Default rate (raw): {df_raw['default.payment.next.month'].mean():.3f}")

    print("\n  Building capstone-compatible features …")
    df_mapped = build_uci_features(df_raw)
    print(f"  Mapped shape: {df_mapped.shape}")
    print(f"  Default rate (mapped): {df_mapped['loan_default_status'].mean():.3f}")
    print(f"  Columns: {list(df_mapped.columns)}")

    out_path = os.path.join(RESULTS_DIR, "uci_mapped.csv")
    df_mapped.to_csv(out_path, index=False)
    print(f"\n  Saved full mapped dataset -> {out_path}")

    sample_path = os.path.join(RESULTS_DIR, "crosswalk_sample.csv")
    df_mapped.head(10).to_csv(sample_path, index=False)
    print(f"  Saved 10-row sample -> {sample_path}")

    # Quick null check
    null_counts = df_mapped.isnull().sum()
    if null_counts.sum() == 0:
        print("\n  [OK] No nulls in mapped dataset.")
    else:
        print("\n  [WARN] Nulls detected:")
        print(null_counts[null_counts > 0])

    return df_mapped


if __name__ == "__main__":
    main()

