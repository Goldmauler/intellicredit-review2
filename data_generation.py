"""
==============================================================================
IntelliCredit - Synthetic Micro-Lending Data Generator (Vectorized)
==============================================================================
Project : IntelliCredit: Alternative Credit Scoring & Portfolio Risk Engine
Purpose : Generates realistic synthetic borrower transaction data using
          fully vectorized NumPy operations for fast execution.

Dataset Policy Compliance:
    - NO pre-built datasets from Kaggle, UCI, or GitHub.
    - All data is programmatically generated using statistical distributions
      with realistic inter-feature correlations.

Usage:
    python data_generation.py

Output:
    - data/borrower_profiles.csv        (30,000 borrower records)
    - data/monthly_transactions.csv     (360,000 monthly transaction records)
    - data/macroeconomic_indicators.csv (36 months of macro data)
    - data/data_dictionary.txt          (Feature documentation)
==============================================================================
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from faker import Faker
import warnings
import time

warnings.filterwarnings("ignore")
np.random.seed(42)
fake = Faker("en_IN")
Faker.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================
NUM_BORROWERS = 30000
NUM_MONTHS = 12
OUTPUT_DIR = os.environ.get("INTELLICREDIT_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))

GIG_PLATFORMS = [
    "Ride-Hailing (Uber/Ola)",
    "Food Delivery (Swiggy/Zomato)",
    "Freelancing (Upwork/Fiverr)",
    "E-Commerce Seller (Amazon/Flipkart)",
    "Content Creator (YouTube/Instagram)",
    "Tutoring (Byju's/Unacademy)",
    "Home Services (UrbanClap/TaskRabbit)",
    "Logistics (Porter/Dunzo)",
]

LOCATION_TYPES = ["Metro", "Tier-1 City", "Tier-2 City", "Semi-Urban", "Rural"]
LOCATION_WEIGHTS = [0.30, 0.25, 0.20, 0.15, 0.10]
LOCATION_MULTIPLIERS = {"Metro": 1.4, "Tier-1 City": 1.2, "Tier-2 City": 1.0, "Semi-Urban": 0.8, "Rural": 0.6}

EDUCATION_LEVELS = ["Below 10th", "10th Pass", "12th Pass", "Diploma", "Undergraduate", "Postgraduate"]
EDUCATION_WEIGHTS = [0.05, 0.10, 0.25, 0.20, 0.30, 0.10]
EDUCATION_MULTIPLIERS = {"Below 10th": 0.7, "10th Pass": 0.8, "12th Pass": 0.9, "Diploma": 1.0, "Undergraduate": 1.1, "Postgraduate": 1.25}


def main():
    start_time = time.time()
    print("=" * 70)
    print("  IntelliCredit - Synthetic Data Generation Pipeline (Vectorized)")
    print("=" * 70)
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    n = NUM_BORROWERS
    m = NUM_MONTHS

    # ==================================================================
    # STEP 1: Generate Borrower Demographics
    # ==================================================================
    print(f"[1/6] Generating {n:,} borrower demographic profiles...")

    borrower_ids = np.array([f"BRW-{str(i).zfill(6)}" for i in range(1, n + 1)])
    borrower_names = np.array([fake.name() for _ in range(n)])

    ages = np.clip(np.random.normal(32, 8, n).astype(int), 21, 60)
    genders = np.random.choice(["Male", "Female", "Other"], n, p=[0.58, 0.40, 0.02])
    locations = np.random.choice(LOCATION_TYPES, n, p=LOCATION_WEIGHTS)
    education = np.random.choice(EDUCATION_LEVELS, n, p=EDUCATION_WEIGHTS)
    platforms = np.random.choice(GIG_PLATFORMS, n)
    tenures = np.clip(np.random.exponential(18, n).astype(int), 1, 120)
    has_bank = np.random.choice([1, 0], n, p=[0.85, 0.15])
    dependents = np.random.choice([0, 1, 2, 3, 4, 5], n, p=[0.20, 0.25, 0.25, 0.15, 0.10, 0.05])

    # Vectorized multiplier lookup
    loc_mult = np.array([LOCATION_MULTIPLIERS[loc] for loc in locations])
    edu_mult = np.array([EDUCATION_MULTIPLIERS[edu] for edu in education])

    print(f"    -> Done in {time.time() - start_time:.1f}s")

    # ==================================================================
    # STEP 2: Generate Monthly Transactions (Fully Vectorized)
    # ==================================================================
    step2_start = time.time()
    print(f"[2/6] Generating {n * m:,} monthly transaction records (vectorized)...")

    # Base income per borrower (shape: n)
    base_incomes = np.random.uniform(8000, 55000, n) * loc_mult * edu_mult

    # Income volatility factor per borrower (higher for newer gig workers)
    volatility_factors = np.maximum(0.05, 0.35 - (tenures * 0.003))

    # Generate all monthly incomes at once (shape: n x m)
    income_noise = np.random.normal(0, 1, (n, m))
    monthly_incomes = np.clip(
        base_incomes[:, None] + income_noise * (base_incomes * volatility_factors)[:, None],
        2000, 150000
    )

    # Utility bill amounts (shape: n x m)
    utility_bills = np.random.uniform(800, 4500, (n, m)) * loc_mult[:, None]

    # On-time payment probability (correlated with tenure and income stability)
    on_time_base_prob = np.clip(
        0.6 + (tenures * 0.005)[:, None] + (monthly_incomes / base_incomes[:, None] - 1) * 0.3,
        0.3, 0.98
    )
    utility_on_time = (np.random.uniform(0, 1, (n, m)) < on_time_base_prob).astype(int)
    utility_delay = np.where(utility_on_time == 1, 0, np.clip(np.random.exponential(12, (n, m)).astype(int) + 1, 1, 90))

    # Mobile recharge
    recharge_amounts = np.random.uniform(149, 999, (n, m))
    days_between_recharges = np.clip(np.random.normal(28, 8, (n, m)).astype(int), 10, 60)

    # BNPL usage
    bnpl_prob = np.clip(0.3 + (0.02 * (35 - ages)) - (tenures * 0.002), 0.05, 0.70)
    bnpl_counts = np.random.poisson((3 * bnpl_prob)[:, None] * np.ones((1, m)))
    bnpl_per_item = np.random.uniform(500, 5000, (n, m))
    bnpl_amounts = np.where(bnpl_counts > 0, bnpl_counts * bnpl_per_item, 0)

    # Monthly outflow & balance
    expense_ratios = np.clip(np.random.normal(0.70, 0.15, (n, m)), 0.30, 1.05)
    monthly_outflows = monthly_incomes * expense_ratios
    eom_balances = np.maximum(0, (monthly_incomes - monthly_outflows) + np.random.uniform(0, 5000, (n, m)))

    # Digital engagement
    app_logins = np.clip(np.random.poisson((15 + tenures * 0.2)[:, None] * np.ones((1, m))), 0, 90)
    upi_txns = np.clip(np.random.poisson((12 * loc_mult)[:, None] * np.ones((1, m))), 0, 80)

    # Build month labels
    month_dates = pd.date_range(start="2025-01-01", periods=m, freq="MS").strftime("%Y-%m").tolist()

    # Flatten into transaction DataFrame
    rep_ids = np.repeat(borrower_ids, m)
    rep_month_idx = np.tile(np.arange(1, m + 1), n)
    rep_month_labels = np.tile(month_dates, n)

    txn_df = pd.DataFrame({
        "borrower_id": rep_ids,
        "month_index": rep_month_idx,
        "month_label": rep_month_labels,
        "gig_income": monthly_incomes.ravel().round(2),
        "utility_bill_amount": utility_bills.ravel().round(2),
        "utility_paid_on_time": utility_on_time.ravel(),
        "utility_delay_days": utility_delay.ravel(),
        "mobile_recharge_amount": recharge_amounts.ravel().round(2),
        "days_between_recharges": days_between_recharges.ravel(),
        "bnpl_transaction_count": bnpl_counts.ravel(),
        "bnpl_total_amount": bnpl_amounts.ravel().round(2),
        "monthly_outflow": monthly_outflows.ravel().round(2),
        "end_of_month_balance": eom_balances.ravel().round(2),
        "app_login_count": app_logins.ravel(),
        "upi_transaction_count": upi_txns.ravel(),
    })

    print(f"    -> Done in {time.time() - step2_start:.1f}s")

    # ==================================================================
    # STEP 3: Aggregate Borrower-Level Features (Vectorized)
    # ==================================================================
    step3_start = time.time()
    print("[3/6] Aggregating borrower-level features...")

    # Compute aggregates directly from the 2D arrays (no groupby needed)
    avg_monthly_income = monthly_incomes.mean(axis=1)
    std_monthly_income = monthly_incomes.std(axis=1)
    min_monthly_income = monthly_incomes.min(axis=1)
    max_monthly_income = monthly_incomes.max(axis=1)

    utility_on_time_count = utility_on_time.sum(axis=1)
    avg_utility_delay = utility_delay.mean(axis=1)
    max_utility_delay = utility_delay.max(axis=1)

    avg_recharge_gap = days_between_recharges.mean(axis=1)
    total_bnpl_txns = bnpl_counts.sum(axis=1)
    avg_bnpl_amount = bnpl_amounts.mean(axis=1)
    max_bnpl_amount = bnpl_amounts.max(axis=1)

    avg_outflow = monthly_outflows.mean(axis=1)
    avg_eom_bal = eom_balances.mean(axis=1)
    min_eom_bal = eom_balances.min(axis=1)

    avg_logins = app_logins.mean(axis=1)
    avg_upi = upi_txns.mean(axis=1)

    # Engineered Features
    income_volatility_index = np.round(std_monthly_income / avg_monthly_income, 4)
    utility_reliability_score = np.round(utility_on_time_count / m, 4)
    debt_to_inflow_ratio = np.clip(np.round(avg_bnpl_amount / avg_monthly_income, 4), 0, 2)
    savings_buffer_ratio = np.clip(np.round(avg_eom_bal / np.where(avg_outflow == 0, 1, avg_outflow), 4), 0, 5)

    # Income trend slope (vectorized linear regression)
    x = np.arange(m)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()
    income_trend_slope = np.round(((monthly_incomes - monthly_incomes.mean(axis=1, keepdims=True)) * (x - x_mean)).sum(axis=1) / x_var, 2)

    print(f"    -> Done in {time.time() - step3_start:.1f}s")

    # ==================================================================
    # STEP 4: Generate Loan Details & Default Labels
    # ==================================================================
    step4_start = time.time()
    print("[4/6] Generating loan details and computing default labels...")

    loan_amounts = np.clip(avg_monthly_income * np.random.uniform(1.5, 6.0, n), 5000, 300000).round(0)
    loan_tenures = np.random.choice([3, 6, 9, 12, 18, 24], n, p=[0.15, 0.25, 0.20, 0.20, 0.12, 0.08])
    interest_rates = np.clip(np.random.normal(18, 5, n), 8, 36).round(2)
    prev_loans = np.clip(np.random.poisson(2, n), 0, 15)
    prev_repaid = (prev_loans * np.clip(np.random.beta(5, 1.5, n), 0.2, 1.0)).astype(int)

    # Default probability via logistic function
    repayment_ratio = np.where(prev_loans == 0, 0.5, prev_repaid / prev_loans)

    log_odds = (
        -1.2
        + 3.0  * income_volatility_index
        - 2.5  * utility_reliability_score
        + 2.0  * debt_to_inflow_ratio
        - 1.5  * np.clip(savings_buffer_ratio, 0, 2)
        - 0.02 * tenures
        + 0.5  * (avg_recharge_gap > 35).astype(float)
        - 0.03 * avg_logins
        + 0.8  * (income_trend_slope < -500).astype(float)
        + 0.000005 * loan_amounts
        - 0.5  * np.clip(repayment_ratio, 0, 1)
        + np.random.normal(0, 0.5, n)
    )

    default_probability = 1 / (1 + np.exp(-log_odds))
    default_status = (np.random.uniform(0, 1, n) < default_probability).astype(int)

    actual_rate = default_status.mean()
    print(f"    -> Generated default rate: {actual_rate:.2%} ({default_status.sum():,} defaults out of {n:,})")
    print(f"    -> Done in {time.time() - step4_start:.1f}s")

    # ==================================================================
    # Build final borrower DataFrame
    # ==================================================================
    borrower_df = pd.DataFrame({
        "borrower_id": borrower_ids,
        "borrower_name": borrower_names,
        "age": ages,
        "gender": genders,
        "location_type": locations,
        "education_level": education,
        "primary_gig_platform": platforms,
        "gig_tenure_months": tenures,
        "has_bank_account": has_bank,
        "num_dependents": dependents,
        "avg_monthly_income": avg_monthly_income.round(2),
        "std_monthly_income": std_monthly_income.round(2),
        "min_monthly_income": min_monthly_income.round(2),
        "max_monthly_income": max_monthly_income.round(2),
        "total_utility_bills": m,
        "utility_on_time_count": utility_on_time_count,
        "avg_utility_delay_days": avg_utility_delay.round(2),
        "max_utility_delay_days": max_utility_delay,
        "avg_days_between_recharges": avg_recharge_gap.round(2),
        "total_bnpl_transactions": total_bnpl_txns,
        "avg_bnpl_monthly_amount": avg_bnpl_amount.round(2),
        "max_bnpl_monthly_amount": max_bnpl_amount.round(2),
        "avg_monthly_outflow": avg_outflow.round(2),
        "avg_eom_balance": avg_eom_bal.round(2),
        "min_eom_balance": min_eom_bal.round(2),
        "avg_app_logins": avg_logins.round(2),
        "avg_upi_transactions": avg_upi.round(2),
        "income_volatility_index": income_volatility_index,
        "utility_reliability_score": utility_reliability_score,
        "debt_to_inflow_ratio": debt_to_inflow_ratio,
        "savings_buffer_ratio": savings_buffer_ratio,
        "income_trend_slope": income_trend_slope,
        "loan_amount_requested": loan_amounts,
        "loan_tenure_months": loan_tenures,
        "interest_rate_annual": interest_rates,
        "previous_micro_loans": prev_loans,
        "previous_loans_repaid_on_time": prev_repaid,
        "default_probability_true": default_probability.round(4),
        "loan_default_status": default_status,
    })

    # ==================================================================
    # STEP 5: Generate Macroeconomic Time-Series Data
    # ==================================================================
    step5_start = time.time()
    print("[5/6] Generating macroeconomic time-series data (36 months)...")

    dates = pd.date_range(start="2023-01-01", periods=36, freq="MS")

    # CPI Inflation Rate (%)
    base_cpi = 4.5
    cpi_trend = np.linspace(0, 1.5, 36) + np.random.normal(0, 0.2, 36)
    cpi_inflation = np.clip(base_cpi + cpi_trend + 0.3 * np.sin(np.linspace(0, 4 * np.pi, 36)), 3.0, 8.0)

    # Benchmark Interest Rate (%)
    base_rate = 6.5
    rate_steps = np.concatenate([
        np.full(8, 0), np.full(6, 0.25), np.full(6, 0.50),
        np.full(6, 0.75), np.full(6, 0.50), np.full(4, 0.25)
    ])[:36]
    interest_rate = base_rate + rate_steps + np.random.normal(0, 0.05, 36)

    # Unemployment Rate (%)
    base_unemployment = 7.5
    seasonal = 0.4 * np.sin(np.linspace(0, 4 * np.pi, 36))
    unemployment_rate = np.clip(base_unemployment + seasonal + np.random.normal(0, 0.15, 36), 5.5, 10.0)

    # Consumer Confidence Index
    cci = np.clip(75 - 2 * (cpi_inflation - 4.5) - 1.5 * (unemployment_rate - 7.5) + np.random.normal(0, 2, 36), 50, 110)

    # Portfolio Default Rate (correlated with macro conditions)
    portfolio_default_rate = np.clip(
        0.08 + 0.01 * (cpi_inflation - 4.5) + 0.005 * (interest_rate - 6.5)
        + 0.008 * (unemployment_rate - 7.5) + np.random.normal(0, 0.005, 36),
        0.03, 0.20,
    )

    macro_df = pd.DataFrame({
        "date": dates,
        "cpi_inflation_rate": np.round(cpi_inflation, 2),
        "benchmark_interest_rate": np.round(interest_rate, 2),
        "unemployment_rate": np.round(unemployment_rate, 2),
        "consumer_confidence_index": np.round(cci, 2),
        "portfolio_default_rate": np.round(portfolio_default_rate, 4),
    })

    print(f"    -> Done in {time.time() - step5_start:.1f}s")

    # ==================================================================
    # STEP 6: Save All Datasets & Data Dictionary
    # ==================================================================
    print("[6/6] Saving datasets...")

    borrower_path = os.path.join(OUTPUT_DIR, "borrower_profiles.csv")
    borrower_df.to_csv(borrower_path, index=False)
    print(f"    -> Borrower profiles   : {borrower_path} ({len(borrower_df):,} rows, {len(borrower_df.columns)} cols)")

    txn_path = os.path.join(OUTPUT_DIR, "monthly_transactions.csv")
    txn_df.to_csv(txn_path, index=False)
    print(f"    -> Monthly transactions: {txn_path} ({len(txn_df):,} rows, {len(txn_df.columns)} cols)")

    macro_path = os.path.join(OUTPUT_DIR, "macroeconomic_indicators.csv")
    macro_df.to_csv(macro_path, index=False)
    print(f"    -> Macro indicators    : {macro_path} ({len(macro_df):,} rows, {len(macro_df.columns)} cols)")

    # --- Data Dictionary ---
    dictionary = f"""
================================================================================
IntelliCredit - Data Dictionary
================================================================================
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Dataset Policy: All data programmatically generated (NO Kaggle/UCI/GitHub sources)

================================================================================
FILE: borrower_profiles.csv  ({len(borrower_df):,} records)
================================================================================

DEMOGRAPHIC FEATURES:
  borrower_id                   : Unique borrower identifier (BRW-XXXXXX)
  borrower_name                 : Synthetic name (Faker library, en_IN locale)
  age                           : Age in years (21-60, Normal dist, mean=32)
  gender                        : Male / Female / Other
  location_type                 : Metro / Tier-1 City / Tier-2 City / Semi-Urban / Rural
  education_level               : Below 10th to Postgraduate
  primary_gig_platform          : Primary gig-economy platform category
  gig_tenure_months             : Months active on gig platform (1-120)
  has_bank_account              : Whether borrower has a formal bank account (0/1)
  num_dependents                : Number of financial dependents (0-5)

AGGREGATED BEHAVIORAL FEATURES (from 12-month transaction history):
  avg_monthly_income            : Mean monthly gig platform earnings (INR)
  std_monthly_income            : Std deviation of monthly earnings
  min_monthly_income            : Lowest monthly earning in 12 months
  max_monthly_income            : Highest monthly earning in 12 months
  total_utility_bills           : Number of utility bill invoices (12)
  utility_on_time_count         : Bills paid on time out of 12
  avg_utility_delay_days        : Average delay in days when payment was late
  max_utility_delay_days        : Maximum delay recorded
  avg_days_between_recharges    : Average gap between mobile recharges (days)
  total_bnpl_transactions       : Total BNPL/micro-loan transactions in 12 months
  avg_bnpl_monthly_amount       : Average monthly BNPL obligation (INR)
  max_bnpl_monthly_amount       : Peak BNPL obligation month (INR)
  avg_monthly_outflow           : Mean monthly total spending (INR)
  avg_eom_balance               : Average end-of-month bank balance (INR)
  min_eom_balance               : Minimum EOM balance (liquidity floor)
  avg_app_logins                : Average monthly app login count
  avg_upi_transactions          : Average monthly UPI digital payments

ENGINEERED FINANCIAL FEATURES:
  income_volatility_index       : Coefficient of Variation (std/mean) of income
  utility_reliability_score     : On-time utility payments / Total bills
  debt_to_inflow_ratio          : Avg BNPL amount / Avg monthly income
  savings_buffer_ratio          : Avg EOM balance / Avg monthly outflow
  income_trend_slope            : Linear slope of 12-month income trajectory

LOAN REQUEST FEATURES:
  loan_amount_requested         : Requested micro-loan amount (INR)
  loan_tenure_months            : Requested loan duration (3-24 months)
  interest_rate_annual          : Assigned annual interest rate (%)
  previous_micro_loans          : Count of prior micro-loans taken
  previous_loans_repaid_on_time : Count of prior loans repaid without default

TARGET VARIABLE:
  default_probability_true      : True underlying default probability (for analysis)
  loan_default_status           : Binary (0=Non-Default, 1=Default/90-day delinquent)

================================================================================
FILE: monthly_transactions.csv  ({len(txn_df):,} records)
================================================================================
  borrower_id                   : Foreign key to borrower_profiles
  month_index                   : Month number (1-12)
  month_label                   : Calendar month (YYYY-MM)
  gig_income                    : Gig platform earnings for that month (INR)
  utility_bill_amount           : Utility bill amount (INR)
  utility_paid_on_time          : Whether paid on time (0/1)
  utility_delay_days            : Days delayed (0 if on time)
  mobile_recharge_amount        : Mobile recharge value (INR)
  days_between_recharges        : Days gap since last recharge
  bnpl_transaction_count        : BNPL transactions that month
  bnpl_total_amount             : Total BNPL spend that month (INR)
  monthly_outflow               : Total outgoing spending (INR)
  end_of_month_balance          : Closing bank balance (INR)
  app_login_count               : App engagement (login count)
  upi_transaction_count         : Digital payment transactions

================================================================================
FILE: macroeconomic_indicators.csv  ({len(macro_df)} monthly records)
================================================================================
  date                          : Month start date
  cpi_inflation_rate            : Consumer Price Index inflation (%)
  benchmark_interest_rate       : Central bank benchmark rate (%)
  unemployment_rate             : Civilian unemployment rate (%)
  consumer_confidence_index     : Consumer confidence (50-110 scale)
  portfolio_default_rate        : Aggregate micro-lending default rate

================================================================================
DATA GENERATION METHODOLOGY
================================================================================
- Demographics: Faker library (en_IN) + numpy statistical distributions
- Transaction correlations: Features interdependent via conditional probability
  (e.g., utility payment punctuality depends on income stability)
- Default labels: Logistic sigmoid function driven by behavioral feature
  combination (income volatility, utility reliability, debt ratio, savings
  buffer, gig tenure, digital engagement, income trend)
- Macroeconomic data: Synthetic time-series with trend, seasonality, and
  inter-variable correlation
- All operations fully vectorized using NumPy (no row-level Python loops)
================================================================================
"""
    dict_path = os.path.join(OUTPUT_DIR, "data_dictionary.txt")
    with open(dict_path, "w", encoding="utf-8") as f:
        f.write(dictionary)
    print(f"    -> Data dictionary     : {dict_path}")

    # ==================================================================
    # SUMMARY
    # ==================================================================
    total_time = time.time() - start_time
    print()
    print("=" * 70)
    print("  Generation Complete - Summary")
    print("=" * 70)
    print(f"  Total borrower profiles     : {len(borrower_df):,}")
    print(f"  Total monthly transactions  : {len(txn_df):,}")
    print(f"  Macro time-series months    : {len(macro_df):,}")
    print(f"  Default rate                : {borrower_df['loan_default_status'].mean():.2%}")
    print(f"  Defaulters                  : {borrower_df['loan_default_status'].sum():,}")
    print(f"  Non-defaulters              : {(borrower_df['loan_default_status'] == 0).sum():,}")
    print()
    print("  Feature Statistics (Borrower Profiles):")
    print(f"    Avg monthly income        : INR {borrower_df['avg_monthly_income'].mean():,.0f}")
    print(f"    Avg income volatility     : {borrower_df['income_volatility_index'].mean():.3f}")
    print(f"    Avg utility reliability   : {borrower_df['utility_reliability_score'].mean():.3f}")
    print(f"    Avg debt-to-inflow ratio  : {borrower_df['debt_to_inflow_ratio'].mean():.3f}")
    print(f"    Avg savings buffer ratio  : {borrower_df['savings_buffer_ratio'].mean():.3f}")
    print()
    print(f"  Total execution time        : {total_time:.1f} seconds")
    print(f"  Output directory            : {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
