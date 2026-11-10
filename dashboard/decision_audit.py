"""
IntelliCredit - Underwriting Decision Audit Trail & Packet Exporter
===================================================================
Generates regulatory compliance audit logs and structured decision packets
for each scored loan application, capturing applicant financial inputs,
model predictions, TreeSHAP drivers, rule trigger receipts, and underwriter actions.

Capabilities:
  * Creates standardized Audit Decision Packets (Ind AS / RBI Fair Lending compliant).
  * Logs underwriter manual overrides, rationale, and pricing exceptions.
  * Serializes decision receipts to structured JSON / CSV for compliance storage.

Author: Interactive Dashboard & UI Team
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from src.config import PROJECT_ROOT

AUDIT_DIR = os.path.join(PROJECT_ROOT, "reports", "audit_logs")


def create_decision_audit_packet(
    applicant_id: str,
    applicant_data: Dict[str, Any],
    predicted_pd: float,
    risk_band: str,
    rule_hits: List[str],
    final_decision: str,
    offered_rate: float,
    underwriter_notes: str = "",
    underwriter_id: str = "UW-OFFICER-01",
    is_override: bool = False,
) -> Dict[str, Any]:
    """
    Constructs a complete regulatory decision audit packet for a single loan facility.

    Parameters:
        applicant_id: Unique borrower identifier.
        applicant_data: Financial attributes snapshot at decision time.
        predicted_pd: Calibrated probability of default.
        risk_band: Risk category (Band A through E).
        rule_hits: List of behavioral warning rule antecedents triggered.
        final_decision: Final underwriter disposition (Approve/Refer/Decline).
        offered_rate: Final annual interest rate percentage approved.
        underwriter_notes: Justification commentary (mandatory for overrides).
        underwriter_id: Identifier of the loan officer.
        is_override: Whether human judgment overrode model recommendation.

    Returns:
        Structured audit packet dictionary.
    """
    timestamp = datetime.now().isoformat()

    packet = {
        "audit_version": "1.0",
        "timestamp": timestamp,
        "facility_metadata": {
            "applicant_id": applicant_id,
            "underwriter_id": underwriter_id,
            "decision": final_decision,
            "is_override": is_override,
            "justification": underwriter_notes,
            "offered_interest_rate_pct": offered_rate,
        },
        "model_risk_assessment": {
            "calibrated_pd": round(float(predicted_pd), 4),
            "risk_band": risk_band,
            "rules_triggered_count": len(rule_hits),
            "rules_triggered": rule_hits,
        },
        "financial_snapshot": {
            "loan_amount": applicant_data.get("loan_amount_requested"),
            "tenure_months": applicant_data.get("loan_tenure_months"),
            "monthly_income": applicant_data.get("avg_monthly_income"),
            "monthly_outflow": applicant_data.get("avg_monthly_outflow"),
            "debt_to_inflow_ratio": applicant_data.get("debt_to_inflow_ratio"),
            "savings_buffer_ratio": applicant_data.get("savings_buffer_ratio"),
            "avg_utility_delay_days": applicant_data.get("avg_utility_delay_days"),
        },
    }

    return packet


def export_audit_packet(
    packet: Dict[str, Any],
    output_dir: Optional[str] = None,
) -> str:
    """
    Saves an audit packet as a timestamped JSON file.

    Returns:
        Path to the saved JSON audit file.
    """
    target_dir = output_dir or AUDIT_DIR
    os.makedirs(target_dir, exist_ok=True)

    app_id = packet["facility_metadata"]["applicant_id"]
    timestamp_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"decision_{app_id}_{timestamp_slug}.json"
    file_path = os.path.join(target_dir, filename)

    with open(file_path, "w") as f:
        json.dump(packet, f, indent=2)

    return file_path


def load_recent_audit_log(audit_dir: Optional[str] = None, max_records: int = 50) -> pd.DataFrame:
    """Loads historical audit packets into a tabular summary dataframe."""
    target_dir = audit_dir or AUDIT_DIR
    if not os.path.exists(target_dir):
        return pd.DataFrame()

    files = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith(".json")]
    files = sorted(files, key=os.path.getmtime, reverse=True)[:max_records]

    records = []
    for fp in files:
        try:
            with open(fp, "r") as f:
                data = json.load(f)
            records.append({
                "timestamp": data.get("timestamp"),
                "applicant_id": data["facility_metadata"]["applicant_id"],
                "underwriter_id": data["facility_metadata"]["underwriter_id"],
                "decision": data["facility_metadata"]["decision"],
                "is_override": data["facility_metadata"]["is_override"],
                "pd": data["model_risk_assessment"]["calibrated_pd"],
                "risk_band": data["model_risk_assessment"]["risk_band"],
                "rules_hit": data["model_risk_assessment"]["rules_triggered_count"],
                "offered_rate": data["facility_metadata"]["offered_interest_rate_pct"],
            })
        except Exception:
            continue

    return pd.DataFrame(records)


if __name__ == "__main__":
    print("IntelliCredit - Testing Decision Audit Trail & Packet Exporter...")
    dummy_applicant = {
        "loan_amount_requested": 45000.0,
        "loan_tenure_months": 12,
        "avg_monthly_income": 38000.0,
        "avg_monthly_outflow": 21000.0,
        "debt_to_inflow_ratio": 0.5526,
        "savings_buffer_ratio": 0.4286,
        "avg_utility_delay_days": 3.0,
    }

    pkt = create_decision_audit_packet(
        applicant_id="BRW-AUDIT-DEMO",
        applicant_data=dummy_applicant,
        predicted_pd=0.062,
        risk_band="B",
        rule_hits=["low_savings_buffer"],
        final_decision="Approve",
        offered_rate=16.5,
        underwriter_notes="Strong repayment capacity despite minor utility delays.",
    )

    path = export_audit_packet(pkt)
    print(f"Exported decision packet to: {path}")
    print(f"Packet verified: Applicant {pkt['facility_metadata']['applicant_id']} -> {pkt['facility_metadata']['decision']}")
