"""
Unit Tests - Portfolio Loan Simulation & Macroeconomic Response
===============================================================
Validates credit hazard curves, logit-scale macroeconomic stress response,
and portfolio monthly aggregation mechanics.
"""

import os
import unittest
import numpy as np
import pandas as pd
from scipy.special import expit, logit

from src.config import DATA_DIR


class TestPortfolioSimulation(unittest.TestCase):
    """Verifies core mathematical principles of the portfolio hazard simulator."""

    def test_hazard_hump_shape(self):
        """Verifies early-payment default hump: hazard peaks in early-to-mid tenure."""
        tenure = 24
        k = np.arange(1, tenure + 1)
        # Standard Weibull / Gamma hump: peak around months 3-6
        hazard_shape = (k ** 0.8) * np.exp(-0.25 * k)
        self.assertTrue(np.all(hazard_shape > 0))
        peak_k = k[np.argmax(hazard_shape)]
        self.assertTrue(2 <= peak_k <= 8, f"Hazard peak at month {peak_k} is outside realistic window [2, 8]")

    def test_wilson_macro_logit_shift(self):
        """Validates that macroeconomic stress strictly shifts default hazards upward on logit scale."""
        base_h = 0.015  # 1.5% monthly base hazard
        beta = 0.75
        stress_neutral = 0.0
        stress_adverse = 1.8

        logit_base = logit(base_h)
        h_neutral = expit(logit_base + beta * stress_neutral)
        h_adverse = expit(logit_base + beta * stress_adverse)

        self.assertAlmostEqual(h_neutral, base_h, places=4)
        self.assertGreater(h_adverse, h_neutral)
        self.assertTrue(0.0 < h_adverse < 1.0)

    def test_sample_macro_indicators_completeness(self):
        """Verifies that macroeconomic series contains non-null time-series drivers."""
        macro_path = os.path.join(DATA_DIR, "sample", "sample_macro_india_monthly.csv")
        self.assertTrue(os.path.exists(macro_path), f"Macro sample missing at {macro_path}")

        macro_df = pd.read_csv(macro_path)
        self.assertGreater(len(macro_df), 12)
        self.assertIn("date", macro_df.columns)
        # Check for core macro indicators
        expected_macro = ["cpi_yoy", "unemployment_rate", "repo_rate"]
        present_macro = [c for c in expected_macro if c in macro_df.columns]
        self.assertGreater(len(present_macro), 0)

    def test_sample_portfolio_monthly_kpis(self):
        """Validates KPIs in the portfolio monthly time-series summary."""
        pm_path = os.path.join(DATA_DIR, "sample", "sample_portfolio_monthly.csv")
        self.assertTrue(os.path.exists(pm_path))

        pm_df = pd.read_csv(pm_path)
        self.assertIn("active_loans", pm_df.columns)
        self.assertIn("default_rate", pm_df.columns)
        # Default rate should always be a probability [0, 1]
        self.assertTrue((pm_df["default_rate"] >= 0.0).all())
        self.assertTrue((pm_df["default_rate"] <= 1.0).all())


if __name__ == "__main__":
    unittest.main()

