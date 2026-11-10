import unittest

import pandas as pd

from src.data_integrity import audit_loan_book, audit_performance_panel


class TestDataIntegritySchema(unittest.TestCase):
    def test_loan_book_reports_missing_required_columns(self):
        results = audit_loan_book(pd.DataFrame({"borrower_id": ["B1"]}))

        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].passed)
        self.assertEqual(results[0].check_name, "loan_book_required_columns")
        self.assertIn("loan_amount or loan_amount_requested", results[0].message)
        self.assertIn("loan_tenure_months", results[0].message)

    def test_loan_book_accepts_either_amount_column(self):
        loan_book = pd.DataFrame(
            {
                "borrower_id": ["B1"],
                "loan_amount_requested": [10000.0],
                "loan_tenure_months": [12],
            }
        )

        results = audit_loan_book(loan_book)

        self.assertTrue(results[0].passed)
        self.assertEqual(results[0].check_name, "loan_book_required_columns")

    def test_performance_panel_reports_missing_required_columns(self):
        results = audit_performance_panel(
            pd.DataFrame({"dpd": [0], "outstanding_principal": [5000.0]})
        )

        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].passed)
        self.assertEqual(results[0].check_name, "panel_required_columns")
        self.assertIn("status", results[0].message)

    def test_performance_panel_accepts_complete_schema(self):
        panel = pd.DataFrame(
            {
                "dpd": [0],
                "outstanding_principal": [5000.0],
                "status": ["current"],
            }
        )

        results = audit_performance_panel(panel)

        self.assertTrue(all(result.passed for result in results))
        self.assertEqual(results[0].check_name, "panel_required_columns")


if __name__ == "__main__":
    unittest.main()
