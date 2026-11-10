# IntelliCredit: Risk Analytics & Quality Assurance Validation Guide

> **Developer & Reviewer Guide**  
> Covers execution instructions for the Portfolio Risk Modeling Suite, Automated Test Suite, and Data Integrity Auditing.

---

## 1. Quick Verification Commands

From the repository root directory, run the following validation commands:

```bash
# 1. Run all unit tests (18 automated tests)
python -m unittest discover tests -v

# 2. Run data integrity & zero-leakage schema audit
python -m src.data_integrity

# 3. Compute delinquency transition matrices & roll rates
python -m src.transition_matrix

# 4. Calculate Ind AS 109 / IFRS 9 staged ECL provisions
python -m src.ecl_engine

# 5. Track vintage cohorts & Early Payment Default (EPD) curves
python -m src.vintage_analysis

# 6. Execute macroeconomic supervisory stress test
python -m src.stress_testing

# 7. Generate executive summary markdown report & JSON KPI snapshot
python -m scripts.generate_portfolio_risk_report
```

---

## 2. Test Suite Architecture (`tests/`)

| Test Module | Target Tested | Verification Scope |
|---|---|---|
| `tests/test_config_and_rules.py` | `src/config.py`, `src/risk_rules.py` | Directory paths, ordinal map monotonic rank, target definitions, rule evaluation schema |
| `tests/test_portfolio_simulation.py` | `src/portfolio_simulation.py` | Early hazard hump shape, Wilson logit-link macro stress shifts, monthly KPI bounds |
| `tests/test_risk_analytics.py` | `src/transition_matrix.py`, `src/ecl_engine.py`, `src/vintage_analysis.py`, `src/stress_testing.py` | Stochastic row-sum property ($\sum P_{ij} = 1.0$), multi-period bounds, Ind AS 109 staging criteria, ECL discounting hierarchy, stress loss monotonicity |

---

## 3. Data Integrity & Quality Checks (`src/data_integrity.py`)

The automated auditor runs eight rigorous assertions before model scoring:
1. `feature_set_zero_leakage`: Confirms target column (`loan_default_status`) and hidden ground-truth (`default_probability_true`) are never present in feature matrices.
2. `loan_book_unique_keys`: Asserts zero primary key duplication across borrower profiles.
3. `loan_book_positive_amounts`: Validates principal exposure amounts are positive.
4. `loan_book_tenure_bounds`: Validates loan tenures remain within regulatory boundaries (3 to 60 months).
5. `loan_book_interest_rate_bounds`: Confirms annual interest rates fall within micro-lending rate caps.
6. `panel_non_negative_dpd`: Verifies absence of negative delinquency counters.
7. `panel_non_negative_balance`: Confirms outstanding principal balances never drop below zero.
8. `panel_valid_status_enum`: Enforces standardized delinquency state enumeration (`Current`, `Delinquent`, `Default`, `Settled`).

---

## 4. Continuous Integration (CI) Workflow

The automated GitHub Actions workflow (`.github/workflows/ci.yml`) triggers on every push and pull request to `main` across a multi-version Python matrix (`3.10`, `3.11`, `3.12`):
* Installs project dependencies.
* Executes the data integrity auditor.
* Runs the complete unit test suite.
* Verifies automated portfolio report generation.

