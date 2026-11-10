"""
Unit Tests - Configuration Constants & Association Risk Rules
=============================================================
Verifies directory path structure, categorical encoding dictionaries,
leakage drop lists, and rule evaluation functions.
"""

import os
import unittest
import pandas as pd
import numpy as np

from src.config import (
    PROJECT_ROOT,
    DATA_DIR,
    PROCESSED_DIR,
    MODELS_DIR,
    TARGET_COL,
    DROP_COLS,
    ORDINAL_MAPS,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.risk_rules import load_rules, flag_items


class TestConfig(unittest.TestCase):
    """Verifies that project configuration parameters remain consistent and intact."""

    def test_directory_paths_exist(self):
        """Checks that root and standard subdirectories exist."""
        self.assertTrue(os.path.isdir(PROJECT_ROOT))
        self.assertTrue(os.path.isdir(DATA_DIR))
        self.assertTrue(os.path.isdir(MODELS_DIR))

    def test_target_and_leakage_definitions(self):
        """Verifies target column name and drop list constraints."""
        self.assertEqual(TARGET_COL, "loan_default_status")
        self.assertIn("borrower_id", DROP_COLS)
        self.assertIn("default_probability_true", DROP_COLS)

    def test_ordinal_maps_monotonicity(self):
        """Ensures that education and location mappings have strictly increasing integer codes."""
        edu_map = ORDINAL_MAPS["education_level"]
        self.assertEqual(edu_map["Below 10th"], 0)
        self.assertEqual(edu_map["Postgraduate"], 5)
        self.assertTrue(edu_map["Undergraduate"] > edu_map["12th Pass"])

        loc_map = ORDINAL_MAPS["location_type"]
        self.assertEqual(loc_map["Rural"], 0)
        self.assertEqual(loc_map["Metro"], 4)

    def test_split_and_seed_constants(self):
        """Validates standard modeling seeds and train/test ratio."""
        self.assertEqual(RANDOM_STATE, 42)
        self.assertAlmostEqual(TEST_SIZE, 0.20)


class TestRiskRules(unittest.TestCase):
    """Verifies behavioral warning rules and flag extraction logic."""

    @classmethod
    def setUpClass(cls):
        cls.item_defs, cls.rules = load_rules()

    def test_load_rules_integrity(self):
        """Verifies that rule catalog and item definitions load with expected schema."""
        self.assertIsInstance(self.item_defs, dict)
        self.assertGreater(len(self.item_defs), 0)
        self.assertIsInstance(self.rules, pd.DataFrame)
        self.assertIn("antecedent_items", self.rules.columns)
        self.assertIn("items", self.rules.columns)

    def test_flag_items_clean_applicant(self):
        """Tests that a borrower profile produces valid boolean flags."""
        sample_path = os.path.join(DATA_DIR, "sample", "sample_borrower_profiles.csv")
        self.assertTrue(os.path.exists(sample_path))
        sample_applicant = pd.read_csv(sample_path, nrows=2)

        flags = flag_items(sample_applicant, self.item_defs)
        self.assertIsInstance(flags, pd.DataFrame)
        self.assertEqual(len(flags), 2)
        # All flagged items should be boolean columns
        for col in flags.columns:
            self.assertTrue(flags[col].dtype == bool or np.issubdtype(flags[col].dtype, np.bool_))


if __name__ == "__main__":
    unittest.main()
