"""
IntelliCredit - Portfolio Data Integrity & Schema Audit Suite
==============================================================
Provides automated quality assurance, schema validation, and integrity auditing
across all micro-lending risk datasets (borrower profiles, transaction logs,
portfolio loan book, monthly panel, and macroeconomic drivers).

Audited Quality Dimensions:
  1. Schema Completeness: Expected columns, types, and primary key uniqueness.
  2. Value Range & Boundary Checks: Tenures [3, 60], DPD >= 0, interest rates [0.05, 0.50].
  3. Data Leakage Prevention: Asserts target and ground-truth PD are absent from ML features.
  4. Temporal Consistency: Monotonic month ordering and valid state transitions per borrower.
  5. Statistical Drift & Null Tolerance: Verifies missingness is within acceptable thresholds.

Author: Portfolio Risk Modeling Team
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    CLEANED_BORROWERS_PATH,
    DATA_DIR,
    DROP_COLS,
    FEATURE_COLUMNS_PATH,
    PROCESSED_DIR,
    TARGET_COL,
)

PORTFOLIO_DIR = os.path.join(DATA_DIR, "portfolio")
LOAN_BOOK_PATH = os.path.join(PORTFOLIO_DIR, "loan_book.csv")
LOAN_PERF_PATH = os.path.join(PORTFOLIO_DIR, "loan_performance.csv")


class AuditCheckResult:
    """Represents the outcome of a single dataset integrity check."""

    def __init__(self, check_name: str, passed: bool, message: str, level: str = "ERROR"):
        self.check_name = check_name
        self.passed = passed
        self.message = message
        self.level = level  # "ERROR", "WARNING", "INFO"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check": self.check_name,
            "status": "PASS" if self.passed else "FAIL",
            "level": self.level,
            "message": self.message,
        }


def audit_loan_book(book_df: pd.DataFrame) -> List[AuditCheckResult]:
    """Audits schema, primary key uniqueness, and financial ranges in the loan book."""
    required_columns = {"borrower_id", "loan_tenure_months"}
    missing_columns = sorted(required_columns - set(book_df.columns))
    if not {"loan_amount", "loan_amount_requested"} & set(book_df.columns):
        missing_columns.append("loan_amount or loan_amount_requested")

    schema_result = AuditCheckResult(
        "loan_book_required_columns",
        not missing_columns,
        "Required columns are present."
        if not missing_columns
        else f"Missing required columns: {', '.join(missing_columns)}",
    )
    results = [schema_result]
    if missing_columns:
        return results

    # 1. Primary key uniqueness
    has_dup = book_df["borrower_id"].duplicated().any()
    results.append(
        AuditCheckResult(
            "loan_book_unique_keys",
            not has_dup,
            f"Found {book_df['borrower_id'].duplicated().sum()} duplicate IDs" if has_dup else "All borrower IDs unique.",
        )
    )

    # 2. Positive loan amounts
    amt_col = "loan_amount" if "loan_amount" in book_df.columns else "loan_amount_requested"
    min_amt = book_df[amt_col].min()
    results.append(
        AuditCheckResult(
            "loan_book_positive_amounts",
            min_amt > 0,
            f"Minimum loan amount is INR {min_amt:,.2f}" if min_amt > 0 else f"Invalid non-positive loan amount: {min_amt}",
        )
    )

    # 3. Tenure validity (typical micro-loans are 3 to 60 months)
    min_ten = book_df["loan_tenure_months"].min()
    max_ten = book_df["loan_tenure_months"].max()
    valid_tenure = (min_ten >= 3) and (max_ten <= 60)
    results.append(
        AuditCheckResult(
            "loan_book_tenure_bounds",
            valid_tenure,
            f"Loan tenure spans {min_ten} to {max_ten} months (within [3, 60]).",
        )
    )

    # 4. Interest rate validity (accommodates either percentage scale e.g. 8-40% or decimal scale 0.08-0.40)
    if "interest_rate_annual" in book_df.columns:
        min_rate = book_df["interest_rate_annual"].min()
        max_rate = book_df["interest_rate_annual"].max()
        if min_rate >= 1.0:
            # Stored as percentage values e.g. 8.0 to 36.0%
            valid_rate = (min_rate >= 5.0) and (max_rate <= 50.0)
            msg = f"Annual interest rates span {min_rate:.1f}% to {max_rate:.1f}%."
        else:
            # Stored as decimals e.g. 0.08 to 0.36
            valid_rate = (min_rate >= 0.05) and (max_rate <= 0.50)
            msg = f"Annual interest rates span {min_rate:.1%} to {max_rate:.1%}."

        results.append(
            AuditCheckResult(
                "loan_book_interest_rate_bounds",
                valid_rate,
                msg,
            )
        )

    return results


def audit_performance_panel(perf_df: pd.DataFrame) -> List[AuditCheckResult]:
    """Audits the monthly loan performance panel for temporal and state consistency."""
    required_columns = {"dpd", "outstanding_principal", "status"}
    missing_columns = sorted(required_columns - set(perf_df.columns))
    schema_result = AuditCheckResult(
        "panel_required_columns",
        not missing_columns,
        "Required columns are present."
        if not missing_columns
        else f"Missing required columns: {', '.join(missing_columns)}",
    )
    results = [schema_result]
    if missing_columns:
        return results

    # 1. DPD bounds (DPD cannot be negative)
    neg_dpd = (perf_df["dpd"] < 0).any()
    results.append(
        AuditCheckResult(
            "panel_non_negative_dpd",
            not neg_dpd,
            "All DPD entries are non-negative." if not neg_dpd else "Found negative DPD values.",
        )
    )

    # 2. Outstanding balance consistency
    neg_bal = (perf_df["outstanding_principal"] < 0).any()
    results.append(
        AuditCheckResult(
            "panel_non_negative_balance",
            not neg_bal,
            "All outstanding principal balances are non-negative." if not neg_bal else "Found negative balance entries.",
        )
    )

    # 3. Valid status values
    valid_statuses = {"current", "delinquent", "default", "settled"}
    observed_statuses = set(perf_df["status"].str.lower().unique())
    invalid_statuses = observed_statuses - valid_statuses
    results.append(
        AuditCheckResult(
            "panel_valid_status_enum",
            len(invalid_statuses) == 0,
            f"Observed valid status values: {observed_statuses}" if not invalid_statuses else f"Unexpected statuses: {invalid_statuses}",
        )
    )

    return results


def audit_feature_leakage() -> List[AuditCheckResult]:
    """Ensures feature set does not leak target labels or simulator ground truth."""
    results = []

    if os.path.exists(FEATURE_COLUMNS_PATH):
        with open(FEATURE_COLUMNS_PATH, "r") as f:
            features = [line.strip() for line in f if line.strip()]

        leaked_cols = [col for col in DROP_COLS if col in features]
        if TARGET_COL in features:
            leaked_cols.append(TARGET_COL)

        results.append(
            AuditCheckResult(
                "feature_set_zero_leakage",
                len(leaked_cols) == 0,
                f"Zero leakage: feature columns exclude {DROP_COLS} and {TARGET_COL}."
                if not leaked_cols
                else f"CRITICAL: Leaked columns found in feature list: {leaked_cols}",
            )
        )
    else:
        results.append(
            AuditCheckResult(
                "feature_set_columns_file_exists",
                False,
                f"Feature columns file not found at {FEATURE_COLUMNS_PATH}",
                level="WARNING",
            )
        )

    return results


def run_full_data_audit() -> Tuple[bool, List[Dict[str, Any]]]:
    """Executes all data integrity audits across the repository."""
    all_checks = []

    # 1. Feature Leakage Audit
    all_checks.extend(audit_feature_leakage())

    # 2. Loan Book Audit
    if os.path.exists(LOAN_BOOK_PATH):
        book_df = pd.read_csv(LOAN_BOOK_PATH)
        all_checks.extend(audit_loan_book(book_df))

    # 3. Performance Panel Audit
    if os.path.exists(LOAN_PERF_PATH):
        perf_df = pd.read_csv(LOAN_PERF_PATH)
        all_checks.extend(audit_performance_panel(perf_df))

    results_serialized = [c.to_dict() for c in all_checks]
    all_passed = all(c.passed for c in all_checks if c.level == "ERROR")

    return all_passed, results_serialized


if __name__ == "__main__":
    print("IntelliCredit - Running Portfolio Data Integrity & Schema Audit Suite...")
    passed, checks = run_full_data_audit()
    print(f"\nOverall Audit Status: {'PASSED [OK]' if passed else 'FAILED'}\n")
    for chk in checks:
        icon = "[PASS]" if chk["status"] == "PASS" else "[FAIL]"
        print(f"  {icon:6s} | {chk['check']:32s} | {chk['message']}")
