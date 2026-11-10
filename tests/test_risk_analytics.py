"""
Unit Tests - Advanced Credit Risk Analytics & Staged Loss Modeling
==================================================================
Tests Markov transition matrices, IFRS 9 / Ind AS 109 ECL staging,
vintage cohort metrics, and supervisory stress test sensitivities.
"""

import unittest
import numpy as np
import pandas as pd

from src.transition_matrix import (
    map_dpd_to_state,
    compute_transition_matrix,
    project_absorbing_probabilities,
    STATE_CURRENT,
    STATE_30_DPD,
    STATE_60_DPD,
    STATE_DEFAULT,
    STATE_SETTLED,
    STANDARD_STATES,
)
from src.ecl_engine import (
    classify_ecl_stage,
    calculate_loan_ecl,
    compute_scenario_weighted_ecl,
    STAGE_1,
    STAGE_2,
    STAGE_3,
)
from src.vintage_analysis import (
    assign_origination_quarter,
    compute_vintage_matrix,
)
from src.stress_testing import (
    apply_macro_stress_to_pds,
    simulate_stress_scenario,
    SCENARIOS,
)


class TestTransitionMatrix(unittest.TestCase):
    """Verifies Markov transition matrix calculations and roll-rate properties."""

    def test_dpd_mapping_rules(self):
        """Validates that DPD and balances map to correct regulatory states."""
        self.assertEqual(map_dpd_to_state(0, 50000.0), STATE_CURRENT)
        self.assertEqual(map_dpd_to_state(30, 45000.0), STATE_30_DPD)
        self.assertEqual(map_dpd_to_state(60, 40000.0), STATE_60_DPD)
        self.assertEqual(map_dpd_to_state(90, 35000.0), STATE_DEFAULT)
        self.assertEqual(map_dpd_to_state(0, 0.0), STATE_SETTLED)

    def test_stochastic_transition_matrix_row_sums(self):
        """Verifies that transition matrix rows strictly sum to 1.0 (stochastic matrix)."""
        mock_transitions = pd.DataFrame([
            {"borrower_id": "B1", "from_state": STATE_CURRENT, "to_state": STATE_CURRENT},
            {"borrower_id": "B2", "from_state": STATE_CURRENT, "to_state": STATE_30_DPD},
            {"borrower_id": "B3", "from_state": STATE_30_DPD, "to_state": STATE_60_DPD},
            {"borrower_id": "B4", "from_state": STATE_30_DPD, "to_state": STATE_CURRENT},
            {"borrower_id": "B5", "from_state": STATE_60_DPD, "to_state": STATE_DEFAULT},
            {"borrower_id": "B6", "from_state": STATE_CURRENT, "to_state": STATE_SETTLED},
        ])

        mat = compute_transition_matrix(mock_transitions, states=STANDARD_STATES)
        row_sums = mat.sum(axis=1)
        for state, s in row_sums.items():
            self.assertAlmostEqual(s, 1.0, places=3, msg=f"Row {state} sum {s} != 1.0")

        # Default and Settled must be absorbing
        self.assertEqual(mat.loc[STATE_DEFAULT, STATE_DEFAULT], 1.0)
        self.assertEqual(mat.loc[STATE_SETTLED, STATE_SETTLED], 1.0)

    def test_multi_period_matrix_power_validity(self):
        """Verifies multi-period matrix power P^k preserves valid probability bounds [0, 1]."""
        mock_transitions = pd.DataFrame([
            {"borrower_id": "B1", "from_state": STATE_CURRENT, "to_state": STATE_CURRENT},
            {"borrower_id": "B2", "from_state": STATE_CURRENT, "to_state": STATE_30_DPD},
            {"borrower_id": "B3", "from_state": STATE_30_DPD, "to_state": STATE_CURRENT},
        ])
        mat = compute_transition_matrix(mock_transitions)
        projs = project_absorbing_probabilities(mat, horizons=[3, 6, 12])

        for h, p_h in projs.items():
            self.assertTrue((p_h.to_numpy() >= -1e-6).all())
            self.assertTrue((p_h.to_numpy() <= 1.0 + 1e-6).all())
            np.testing.assert_allclose(p_h.sum(axis=1), 1.0, atol=1e-3)


class TestECLEngine(unittest.TestCase):
    """Verifies Ind AS 109 / IFRS 9 staging rules and provision discounting."""

    def test_ecl_staging_classification(self):
        """Validates staging logic: 0 DPD -> Stage 1, 30 DPD -> Stage 2, 90 DPD -> Stage 3."""
        self.assertEqual(classify_ecl_stage(dpd=0, current_pd=0.04), STAGE_1)
        self.assertEqual(classify_ecl_stage(dpd=30, current_pd=0.08), STAGE_2)
        self.assertEqual(classify_ecl_stage(dpd=60, current_pd=0.12), STAGE_2)
        self.assertEqual(classify_ecl_stage(dpd=90, current_pd=0.50), STAGE_3)
        # SICR triggered by PD jump >= 2.0x
        self.assertEqual(classify_ecl_stage(dpd=0, current_pd=0.10, orig_pd=0.04), STAGE_2)

    def test_ecl_provision_hierarchy(self):
        """Verifies that Stage 3 provision exceeds Stage 1 provision for identical principal."""
        ead = 100000.0
        pd_val = 0.08
        lgd = 0.50
        ecl_stage1 = calculate_loan_ecl(ead=ead, pd_annual=pd_val, lgd=lgd, stage=STAGE_1)
        ecl_stage3 = calculate_loan_ecl(ead=ead, pd_annual=pd_val, lgd=lgd, stage=STAGE_3)

        self.assertGreater(ecl_stage1, 0.0)
        self.assertGreater(ecl_stage3, ecl_stage1)

    def test_scenario_weighted_ecl_bounds(self):
        """Checks that probability-weighted ECL lies between upside and downside values."""
        base_summary = {"total_ecl_provision_inr": 1000000.0, "total_ead_inr": 20000000.0}
        scen = compute_scenario_weighted_ecl(base_summary)

        self.assertLess(scen["upside_ecl_inr"], scen["base_ecl_inr"])
        self.assertGreater(scen["downside_ecl_inr"], scen["base_ecl_inr"])
        self.assertTrue(scen["upside_ecl_inr"] < scen["weighted_ecl_inr"] < scen["downside_ecl_inr"])


class TestVintageAndStress(unittest.TestCase):
    """Verifies vintage cohort mapping and stress test monotonicity."""

    def test_assign_origination_quarter(self):
        """Validates quarter bucketing from date strings."""
        self.assertEqual(assign_origination_quarter("2021-01"), "2021-Q1")
        self.assertEqual(assign_origination_quarter("2021-06"), "2021-Q2")
        self.assertEqual(assign_origination_quarter("2022-09"), "2022-Q3")
        self.assertEqual(assign_origination_quarter("2023-12"), "2023-Q4")

    def test_stress_testing_monotonic_loss(self):
        """Verifies that Severe scenario causes strictly greater losses than Moderate and Baseline."""
        mock_loan_book = pd.DataFrame([{
            "borrower_id": f"B{i}",
            "loan_amount": 50000.0,
            "default_probability_true": 0.07,
        } for i in range(100)])

        res_base = simulate_stress_scenario(mock_loan_book, "Baseline", SCENARIOS["Baseline"])
        res_mod = simulate_stress_scenario(mock_loan_book, "Moderate", SCENARIOS["Moderate_Adverse"])
        res_sev = simulate_stress_scenario(mock_loan_book, "Severe", SCENARIOS["Severe_Adverse"])

        self.assertGreater(res_mod["total_stressed_loss_inr"], res_base["total_stressed_loss_inr"])
        self.assertGreater(res_sev["total_stressed_loss_inr"], res_mod["total_stressed_loss_inr"])
        self.assertLess(res_sev["stressed_crar_pct"], res_base["stressed_crar_pct"])


if __name__ == "__main__":
    unittest.main()
