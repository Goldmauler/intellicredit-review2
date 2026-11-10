"""
IntelliCredit - Interactive Underwriter Counterfactual Scenario Simulator
========================================================================
Enables credit underwriters to evaluate "What-If" scenarios on borderline applicants,
simulating counterfactual policy adjustments (e.g., tenure extension, principal haircut,
or utility bill payment stabilization) and observing real-time risk impact.

Capabilities:
  * Counterfactual Delta Analysis: Compares baseline vs stressed/adjusted applicant profiles.
  * Scorecard & Risk Band Transition: Maps changes in predicted PD to Risk Bands (A-E).
  * Break-Even Rate Sensitivity: Recalculates risk-adjusted pricing given tenure/amount changes.
  * Policy Recommendation Engine: Evaluates whether adjustments shift a 'Refer' or 'Reject'
    into an 'Approve' decision with conditions.

Author: Interactive Dashboard & UI Team
"""

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.underwriting_model import (
    break_even_pd,
    risk_band,
    COST_OF_FUNDS,
    LGD,
)


def apply_counterfactual_adjustments(
    applicant: Dict[str, Any],
    tenure_delta_months: int = 0,
    amount_multiplier: float = 1.0,
    income_multiplier: float = 1.0,
    reduce_utility_delays: bool = False,
    clear_bnpl_obligations: bool = False,
) -> Dict[str, Any]:
    """
    Applies counterfactual scenario modifications to an applicant's financial attributes.

    Parameters:
        applicant: Baseline applicant data dictionary.
        tenure_delta_months: Change in tenure (e.g., +6 months to reduce EMI burden).
        amount_multiplier: Scalar for loan amount (e.g., 0.85 for a 15% haircut).
        income_multiplier: Scalar for monthly income (e.g., 1.10 for verified co-borrower).
        reduce_utility_delays: If True, simulates perfect utility payment compliance.
        clear_bnpl_obligations: If True, zeroes out BNPL obligations.

    Returns:
        Adjusted applicant dictionary.
    """
    sim = dict(applicant)

    # 1. Loan Amount & Tenure adjustments
    orig_amount = float(sim.get("loan_amount_requested", 50000.0))
    sim["loan_amount_requested"] = max(5000.0, round(orig_amount * amount_multiplier, 2))

    orig_tenure = int(sim.get("loan_tenure_months", 12))
    sim["loan_tenure_months"] = max(3, min(36, orig_tenure + tenure_delta_months))

    # 2. Income & Inflows
    orig_income = float(sim.get("avg_monthly_income", 35000.0))
    new_income = round(orig_income * income_multiplier, 2)
    sim["avg_monthly_income"] = new_income

    # 3. Behavioral Payment Improvements
    if reduce_utility_delays:
        sim["avg_utility_delay_days"] = 0.0
        sim["max_utility_delay_days"] = 0.0
        sim["utility_reliability_score"] = min(1.0, sim.get("utility_reliability_score", 0.8) + 0.15)

    if clear_bnpl_obligations:
        sim["avg_bnpl_monthly_amount"] = 0.0
        sim["max_bnpl_monthly_amount"] = 0.0
        sim["total_bnpl_transactions"] = 0

    # 4. Recompute financial ratios
    outflow = float(sim.get("avg_monthly_outflow", 20000.0))
    sim["debt_to_inflow_ratio"] = round(outflow / max(1.0, new_income), 4)

    eom_bal = float(sim.get("avg_eom_balance", 8000.0))
    sim["savings_buffer_ratio"] = round(eom_bal / max(1.0, outflow), 4)

    return sim


def estimate_adjusted_pd(
    base_pd: float,
    applicant_base: Dict[str, Any],
    applicant_adj: Dict[str, Any],
) -> float:
    """
    Estimates counterfactual PD based on elasticities of key behavioral and financial drivers.
    """
    # Sensitivity coefficients derived from logistic score weights
    # Income elasticity
    income_ratio = applicant_adj["avg_monthly_income"] / max(1.0, applicant_base["avg_monthly_income"])
    income_factor = (1.0 / income_ratio) ** 0.35

    # Debt-to-inflow elasticity
    dti_ratio = applicant_adj["debt_to_inflow_ratio"] / max(0.01, applicant_base["debt_to_inflow_ratio"])
    dti_factor = dti_ratio ** 0.40

    # Loan amount elasticity
    amt_ratio = applicant_adj["loan_amount_requested"] / max(1.0, applicant_base["loan_amount_requested"])
    amt_factor = amt_ratio ** 0.25

    # Utility discipline bonus
    util_improvement = max(0.0, applicant_base.get("avg_utility_delay_days", 0.0) - applicant_adj.get("avg_utility_delay_days", 0.0))
    util_factor = np.exp(-0.03 * util_improvement)

    combined_multiplier = income_factor * dti_factor * amt_factor * util_factor
    adjusted_pd = float(np.clip(base_pd * combined_multiplier, 0.005, 0.95))
    return round(adjusted_pd, 4)


def simulate_underwriter_decision(
    applicant: Dict[str, Any],
    base_pd: float,
    tenure_delta_months: int = 0,
    amount_multiplier: float = 1.0,
    income_multiplier: float = 1.0,
    reduce_utility_delays: bool = False,
    clear_bnpl_obligations: bool = False,
) -> Dict[str, Any]:
    """
    Runs full end-to-end underwriter what-if scenario comparison.

    Returns:
        Structured dictionary comparing Baseline vs Scenario metrics.
    """
    adj_applicant = apply_counterfactual_adjustments(
        applicant=applicant,
        tenure_delta_months=tenure_delta_months,
        amount_multiplier=amount_multiplier,
        income_multiplier=income_multiplier,
        reduce_utility_delays=reduce_utility_delays,
        clear_bnpl_obligations=clear_bnpl_obligations,
    )

    adj_pd = estimate_adjusted_pd(base_pd, applicant, adj_applicant)

    # Compute Risk Bands & Pricing
    base_band = risk_band(base_pd)
    adj_band = risk_band(adj_pd)

    base_rate = float(applicant.get("interest_rate_annual", 18.0))
    base_tenure = int(applicant.get("loan_tenure_months", 12))
    adj_tenure = int(adj_applicant.get("loan_tenure_months", 12))

    base_breakeven = float(break_even_pd(base_rate, base_tenure))
    adj_breakeven = float(break_even_pd(base_rate, adj_tenure))

    # Decisioning Logic
    def _get_decision(pd_val, be_val):
        if pd_val <= be_val * 0.85:
            return "Approve"
        elif pd_val <= be_val * 1.15:
            return "Refer (Senior Review)"
        return "Decline"

    base_decision = _get_decision(base_pd, base_breakeven)
    adj_decision = _get_decision(adj_pd, adj_breakeven)

    return {
        "baseline": {
            "loan_amount": applicant.get("loan_amount_requested"),
            "tenure_months": applicant.get("loan_tenure_months"),
            "monthly_income": applicant.get("avg_monthly_income"),
            "debt_to_inflow": applicant.get("debt_to_inflow_ratio"),
            "pd": base_pd,
            "risk_band": base_band,
            "decision": base_decision,
        },
        "scenario": {
            "loan_amount": adj_applicant["loan_amount_requested"],
            "tenure_months": adj_applicant["loan_tenure_months"],
            "monthly_income": adj_applicant["avg_monthly_income"],
            "debt_to_inflow": adj_applicant["debt_to_inflow_ratio"],
            "pd": adj_pd,
            "risk_band": adj_band,
            "decision": adj_decision,
        },
        "deltas": {
            "pd_change": round(adj_pd - base_pd, 4),
            "pd_reduction_pct": round(((base_pd - adj_pd) / max(1e-4, base_pd)) * 100.0, 2),
            "amount_change_inr": round(adj_applicant["loan_amount_requested"] - applicant.get("loan_amount_requested", 0.0), 2),
            "decision_improved": (base_decision != "Approve") and (adj_decision in ["Approve", "Refer (Senior Review)"]),
        },
    }


if __name__ == "__main__":
    print("IntelliCredit - Testing Underwriter Scenario Simulator...")
    sample_borrower = {
        "borrower_id": "TEST-APP-01",
        "avg_monthly_income": 32000.0,
        "avg_monthly_outflow": 22000.0,
        "debt_to_inflow_ratio": 0.6875,
        "avg_eom_balance": 4000.0,
        "avg_utility_delay_days": 14.5,
        "max_utility_delay_days": 28.0,
        "loan_amount_requested": 60000.0,
        "loan_tenure_months": 12,
        "interest_rate_annual": 20.0,
    }

    result = simulate_underwriter_decision(
        applicant=sample_borrower,
        base_pd=0.145,
        tenure_delta_months=6,     # Extend tenure by 6 months
        amount_multiplier=0.85,    # 15% haircut
        reduce_utility_delays=True,# Perfect utility payment compliance
    )

    print(f"\nBaseline  : PD = {result['baseline']['pd']:.2%} | Band = {result['baseline']['risk_band']} | Decision = {result['baseline']['decision']}")
    print(f"Scenario  : PD = {result['scenario']['pd']:.2%} | Band = {result['scenario']['risk_band']} | Decision = {result['scenario']['decision']}")
    print(f"Outcome   : PD Reduction = {result['deltas']['pd_reduction_pct']}% | Decision Improved = {result['deltas']['decision_improved']}")
