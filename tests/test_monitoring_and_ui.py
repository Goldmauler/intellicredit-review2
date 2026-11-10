"""
Unit Tests - Model Monitoring, UI Utilities & Underwriter Simulator
===================================================================
Validates Population Stability Index (PSI) calculations, Basel alert tiering,
underwriter counterfactual scenario logic, and INR formatting utilities.
"""

import unittest
import numpy as np
import pandas as pd

from dashboard.monitoring_engine import (
    calculate_psi,
    classify_psi_status,
    ALERT_GREEN,
    ALERT_AMBER,
    ALERT_RED,
    PSI_THRESHOLD_AMBER,
    PSI_THRESHOLD_RED,
)
from dashboard.scenario_simulator import (
    apply_counterfactual_adjustments,
    estimate_adjusted_pd,
    simulate_underwriter_decision,
)
from dashboard.ui_utils import format_inr, render_risk_band_badge, render_decision_badge
from dashboard.policy_matrix import evaluate_policy_guardrails


class TestMonitoringEngine(unittest.TestCase):
    """Verifies statistical stability and PSI calculation properties."""

    def test_identical_distributions_zero_psi(self):
        """Verifies identical distributions yield PSI close to 0.0 and Green status."""
        rng = np.random.default_rng(42)
        dist_a = rng.normal(loc=50, scale=10, size=2000)
        dist_b = rng.normal(loc=50, scale=10, size=2000)

        psi, _ = calculate_psi(dist_a, dist_b, num_bins=10)
        self.assertLess(psi, 0.05)
        self.assertEqual(classify_psi_status(psi), ALERT_GREEN)

    def test_severely_shifted_distribution_triggers_red(self):
        """Verifies that an acute population distribution shift triggers a Red PSI alert."""
        rng = np.random.default_rng(42)
        dist_baseline = rng.normal(loc=30, scale=5, size=2000)
        dist_shifted = rng.normal(loc=65, scale=5, size=2000)  # Heavy right shift

        psi, _ = calculate_psi(dist_baseline, dist_shifted, num_bins=10)
        self.assertGreater(psi, PSI_THRESHOLD_RED)
        self.assertEqual(classify_psi_status(psi), ALERT_RED)

    def test_moderate_shift_triggers_amber(self):
        """Verifies moderate shift falls into Amber tier [0.10, 0.25]."""
        rng = np.random.default_rng(42)
        dist_baseline = rng.normal(loc=40, scale=8, size=3000)
        dist_moderate = rng.normal(loc=42.5, scale=8, size=3000)

        psi, _ = calculate_psi(dist_baseline, dist_moderate, num_bins=10)
        status = classify_psi_status(psi)
        self.assertEqual(status, ALERT_AMBER)


class TestScenarioSimulator(unittest.TestCase):
    """Verifies counterfactual underwriter what-if adjustments."""

    def setUp(self):
        self.applicant = {
            "borrower_id": "TEST-APP-01",
            "avg_monthly_income": 30000.0,
            "avg_monthly_outflow": 21000.0,
            "debt_to_inflow_ratio": 0.70,
            "avg_eom_balance": 3000.0,
            "avg_utility_delay_days": 18.0,
            "loan_amount_requested": 60000.0,
            "loan_tenure_months": 12,
            "interest_rate_annual": 20.0,
        }

    def test_tenure_and_amount_counterfactuals(self):
        """Validates that tenure extension and amount haircut improve debt burden."""
        adj = apply_counterfactual_adjustments(
            applicant=self.applicant,
            tenure_delta_months=6,
            amount_multiplier=0.80,
            reduce_utility_delays=True,
        )

        self.assertEqual(adj["loan_tenure_months"], 18)
        self.assertEqual(adj["loan_amount_requested"], 48000.0)
        self.assertEqual(adj["avg_utility_delay_days"], 0.0)

    def test_counterfactual_pd_monotonic_reduction(self):
        """Verifies that favorable adjustments reduce estimated PD."""
        adj = apply_counterfactual_adjustments(
            applicant=self.applicant,
            income_multiplier=1.20,
            reduce_utility_delays=True,
        )
        base_pd = 0.15
        adj_pd = estimate_adjusted_pd(base_pd, self.applicant, adj)
        self.assertLess(adj_pd, base_pd)


class TestUIAndPolicyUtilities(unittest.TestCase):
    """Verifies currency formatters, badges, and credit policy checklists."""

    def test_inr_formatting(self):
        """Validates standard and compact Indian Rupee currency representations."""
        self.assertEqual(format_inr(50000), "₹50,000")
        self.assertEqual(format_inr(1500000, compact=True), "₹15.00 L")
        self.assertEqual(format_inr(25000000, compact=True), "₹2.50 Cr")

    def test_badge_html_generation(self):
        """Verifies HTML styling strings for risk bands and decision badges."""
        b_html = render_risk_band_badge("A")
        self.assertIn("Band A (Prime)", b_html)
        self.assertIn("style=", b_html)

        d_html = render_decision_badge("Approve")
        self.assertIn("APPROVE", d_html)

    def test_policy_matrix_breach_detection(self):
        """Verifies that an applicant exceeding DTI or LTI triggers policy breaches."""
        overleveraged_app = {
            "debt_to_inflow_ratio": 0.85,  # Breaches 70% cap
            "avg_monthly_income": 20000.0,
            "loan_amount_requested": 100000.0,  # 5x LTI, breaches 3x cap
            "avg_utility_delay_days": 10.0,
            "savings_buffer_ratio": 0.20,
        }
        res = evaluate_policy_guardrails(overleveraged_app, calibrated_pd=0.10, breakeven_pd=0.12)
        self.assertFalse(res["is_compliant"])
        self.assertGreaterEqual(res["breaches_count"], 2)


if __name__ == "__main__":
    unittest.main()
