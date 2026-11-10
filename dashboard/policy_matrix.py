"""
IntelliCredit - Credit Policy Matrix & Underwriting Rules Engine
================================================================
Defines institutional credit risk rules, exposure caps, and policy cut-offs
for micro-lending facilities, providing an interactive compliance checklist
for loan underwriters.

Policy Guardrails:
  1. Maximum Debt-to-Inflow (DTI): Outflow cannot exceed 70% of monthly income.
  2. Loan-to-Income (LTI) Cap: Requested principal cannot exceed 3.0x monthly income.
  3. Utility Discipline Cap: Average utility delay cannot exceed 25 days.
  4. Savings Buffer Floor: End-of-month buffer must cover at least 15% of monthly outflow.
  5. Maximum Risk Cut-Off: Calibrated PD must not exceed Break-Even PD + 15%.

Author: Interactive Dashboard & UI Team
"""

from typing import Any, Dict, List, Tuple
import pandas as pd

# Standard Credit Policy Guardrails for Retail Micro-Lending
POLICY_MAX_DTI = 0.70          # 70% Debt-to-Inflow ceiling
POLICY_MAX_LTI = 3.0           # 3.0x Loan-to-Monthly-Income ceiling
POLICY_MAX_UTIL_DELAY = 25.0   # 25 days maximum average utility delay
POLICY_MIN_SAVINGS_BUFFER = 0.15 # 15% minimum savings buffer ratio
POLICY_MAX_TENURE = 36         # 36 months maximum tenure for unsecured micro-loans


def evaluate_policy_guardrails(
    applicant: Dict[str, Any],
    calibrated_pd: float,
    breakeven_pd: float,
) -> Dict[str, Any]:
    """
    Evaluates an applicant against institutional credit policy guardrails.

    Returns:
        Dictionary containing policy checklist results, breaches count, and disposition.
    """
    checklist = []
    breaches = 0

    # 1. Debt-to-Inflow Check
    dti = float(applicant.get("debt_to_inflow_ratio", 0.0))
    dti_pass = dti <= POLICY_MAX_DTI
    if not dti_pass:
        breaches += 1
    checklist.append({
        "rule": "Debt-to-Inflow (DTI) Cap",
        "threshold": f"<= {POLICY_MAX_DTI:.0%}",
        "applicant_value": f"{dti:.1%}",
        "passed": dti_pass,
        "severity": "CRITICAL",
    })

    # 2. Loan-to-Income (LTI) Check
    income = float(applicant.get("avg_monthly_income", 1.0))
    amount = float(applicant.get("loan_amount_requested", 0.0))
    lti = amount / max(1.0, income)
    lti_pass = lti <= POLICY_MAX_LTI
    if not lti_pass:
        breaches += 1
    checklist.append({
        "rule": "Loan-to-Income (LTI) Multiplier",
        "threshold": f"<= {POLICY_MAX_LTI:.1f}x",
        "applicant_value": f"{lti:.2f}x",
        "passed": lti_pass,
        "severity": "HIGH",
    })

    # 3. Utility Delay Check
    util_delay = float(applicant.get("avg_utility_delay_days", 0.0))
    util_pass = util_delay <= POLICY_MAX_UTIL_DELAY
    if not util_pass:
        breaches += 1
    checklist.append({
        "rule": "Utility Bill Payment Delay",
        "threshold": f"<= {POLICY_MAX_UTIL_DELAY:.0f} days",
        "applicant_value": f"{util_delay:.1f} days",
        "passed": util_pass,
        "severity": "MEDIUM",
    })

    # 4. Savings Buffer Floor
    buffer_ratio = float(applicant.get("savings_buffer_ratio", 0.0))
    buffer_pass = buffer_ratio >= POLICY_MIN_SAVINGS_BUFFER
    if not buffer_pass:
        breaches += 1
    checklist.append({
        "rule": "Savings Buffer Reserve Floor",
        "threshold": f">= {POLICY_MIN_SAVINGS_BUFFER:.0%}",
        "applicant_value": f"{buffer_ratio:.1%}",
        "passed": buffer_pass,
        "severity": "MEDIUM",
    })

    # 5. Economic Break-Even PD Constraint
    pd_pass = calibrated_pd <= (breakeven_pd * 1.15)
    if not pd_pass:
        breaches += 1
    checklist.append({
        "rule": "Economic Break-Even Risk Ceiling",
        "threshold": f"<= {breakeven_pd * 1.15:.2%}",
        "applicant_value": f"{calibrated_pd:.2%}",
        "passed": pd_pass,
        "severity": "CRITICAL",
    })

    # Recommended Policy Disposition
    if breaches == 0:
        disposition = "Policy Compliant (Eligible for Auto-Approval)"
    elif breaches == 1 and checklist[0]["passed"] and checklist[4]["passed"]:
        disposition = "Minor Exception (Senior Underwriter Discretion)"
    else:
        disposition = "Policy Breach (Hard Decline / Exception Committee Required)"

    return {
        "is_compliant": breaches == 0,
        "breaches_count": breaches,
        "disposition": disposition,
        "checklist": checklist,
    }


def get_policy_summary_table() -> pd.DataFrame:
    """Returns static credit policy rule definitions for display in dashboards."""
    rules = [
        {"Parameter": "Max Debt-to-Inflow (DTI)", "Limit": "70.0%", "Action on Breach": "Decline"},
        {"Parameter": "Max Loan-to-Income (LTI)", "Limit": "3.0x Monthly Income", "Action on Breach": "Haircut Principal"},
        {"Parameter": "Max Utility Delay", "Limit": "25.0 Days", "Action on Breach": "Senior Review"},
        {"Parameter": "Min Savings Buffer", "Limit": "15.0% Outflow", "Action on Breach": "Risk Premium +1.5%"},
        {"Parameter": "Max Tenor", "Limit": "36 Months", "Action on Breach": "Shorten Tenor"},
    ]
    return pd.DataFrame(rules)


if __name__ == "__main__":
    print("IntelliCredit - Credit Policy Matrix Evaluation...")
    mock_app = {
        "debt_to_inflow_ratio": 0.58,
        "avg_monthly_income": 40000.0,
        "loan_amount_requested": 90000.0,
        "avg_utility_delay_days": 12.0,
        "savings_buffer_ratio": 0.22,
    }

    res = evaluate_policy_guardrails(mock_app, calibrated_pd=0.08, breakeven_pd=0.11)
    print(f"Status      : {res['disposition']}")
    print(f"Breaches    : {res['breaches_count']}")
    for item in res["checklist"]:
        status = "[PASS]" if item["passed"] else "[FAIL]"
        print(f"  {status} | {item['rule']:35s} | Req: {item['threshold']:12s} | Actual: {item['applicant_value']}")
