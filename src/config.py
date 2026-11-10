"""
IntelliCredit - Shared Configuration
=====================================
Central configuration used by all 5 member notebooks.
"""

import os

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, "notebooks")

# Raw data files
BORROWER_PROFILES_PATH = os.path.join(DATA_DIR, "borrower_profiles.csv")
MONTHLY_TRANSACTIONS_PATH = os.path.join(DATA_DIR, "monthly_transactions.csv")
MACRO_INDICATORS_PATH = os.path.join(DATA_DIR, "macroeconomic_indicators.csv")

# Processed data files (output of M1, input to others)
CLEANED_BORROWERS_PATH = os.path.join(PROCESSED_DIR, "borrower_profiles_cleaned.csv")
CLEANED_TRANSACTIONS_PATH = os.path.join(PROCESSED_DIR, "monthly_transactions_cleaned.csv")
CLEANED_MACRO_PATH = os.path.join(PROCESSED_DIR, "macroeconomic_indicators_cleaned.csv")

# Feature-engineered data (output of M3, input to M4 & M5)
FEATURED_DATA_PATH = os.path.join(PROCESSED_DIR, "borrower_featured.csv")
FEATURE_COLUMNS_PATH = os.path.join(PROCESSED_DIR, "feature_columns.txt")

# Train/test splits (output of M3, used by M4 & M5)
TRAIN_DATA_PATH = os.path.join(PROCESSED_DIR, "train.csv")
TEST_DATA_PATH = os.path.join(PROCESSED_DIR, "test.csv")
TRAIN_SMOTE_PATH = os.path.join(PROCESSED_DIR, "train_smote.csv")

# ============================================================================
# COLUMN DEFINITIONS
# ============================================================================
TARGET_COL = "loan_default_status"
ID_COL = "borrower_id"
DROP_COLS = ["borrower_id", "borrower_name", "default_probability_true"]

CATEGORICAL_COLS = [
    "gender", "location_type", "education_level", "primary_gig_platform"
]

ORDINAL_MAPS = {
    "education_level": {
        "Below 10th": 0, "10th Pass": 1, "12th Pass": 2,
        "Diploma": 3, "Undergraduate": 4, "Postgraduate": 5
    },
    "location_type": {
        "Rural": 0, "Semi-Urban": 1, "Tier-2 City": 2,
        "Tier-1 City": 3, "Metro": 4
    }
}

NUMERIC_COLS_TO_WINSORIZE = [
    "avg_monthly_income", "std_monthly_income", "min_monthly_income",
    "max_monthly_income", "avg_utility_delay_days", "max_utility_delay_days",
    "avg_bnpl_monthly_amount", "max_bnpl_monthly_amount",
    "avg_monthly_outflow", "avg_eom_balance", "min_eom_balance",
    "income_volatility_index", "debt_to_inflow_ratio", "savings_buffer_ratio",
    "income_trend_slope", "loan_amount_requested", "interest_rate_annual"
]

# ============================================================================
# MODELLING PARAMETERS
# ============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
SMOTE_STRATEGY = "auto"
