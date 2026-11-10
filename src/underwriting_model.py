"""
IntelliCredit - Underwriting model used by the dashboard
========================================================
A compact, explainable PD model trained on the fields an underwriter can actually
type in for a new applicant (no target-encoded or transaction-panel features), so the
dashboard can score anyone in real time.

* Same 80/20 stratified split (seed 42) as the Review 1 notebooks.
* LightGBM trained WITHOUT class re-weighting so the output is a calibrated
  probability (needed for expected-loss and pricing maths, unlike the SMOTE models in M4).
* Explanations come from LightGBM's built-in TreeSHAP (pred_contrib=True).
* Decision rule: decline when PD exceeds the applicant's break-even PD,
      PD* = margin / (margin + LGD),  margin = (rate - cost_of_funds) * tenure_years
  i.e. the PD at which expected interest income equals expected credit loss.

Usage:
    python -m src.underwriting_model
"""

import json
import os

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import BORROWER_PROFILES_PATH, MODELS_DIR, ORDINAL_MAPS, RANDOM_STATE, TEST_SIZE

MODEL_PATH = os.path.join(MODELS_DIR, "underwriting_model.joblib")
CARD_PATH = os.path.join(MODELS_DIR, "underwriting_model_card.json")

# Loss given default: share of exposure not recovered after a default
LGD = 0.65
# Annual funding cost the lender pays; interest above this is margin
COST_OF_FUNDS = 0.09

# Raw application inputs (what the dashboard form asks for)
INPUT_FIELDS = [
    "age", "gig_tenure_months", "has_bank_account", "num_dependents", "location_type", "education_level",
    "avg_monthly_income", "std_monthly_income", "income_trend_slope",
    "utility_on_time_count", "avg_utility_delay_days", "max_utility_delay_days",
    "avg_days_between_recharges", "avg_bnpl_monthly_amount", "avg_monthly_outflow", "avg_eom_balance",
    "avg_app_logins", "avg_upi_transactions",
    "loan_amount_requested", "loan_tenure_months", "interest_rate_annual",
    "previous_micro_loans", "previous_loans_repaid_on_time",
]

FEATURE_LABELS = {
    "age": "Age", "gig_tenure_months": "Gig tenure (months)", "has_bank_account": "Has bank account",
    "num_dependents": "Dependents", "location_code": "Location tier", "education_code": "Education level",
    "avg_monthly_income": "Avg monthly income", "income_volatility_index": "Income volatility",
    "income_trend_slope": "Income trend", "utility_reliability_score": "Utility on-time share",
    "avg_utility_delay_days": "Avg utility delay", "max_utility_delay_days": "Worst utility delay",
    "avg_days_between_recharges": "Recharge gap (days)", "debt_to_inflow_ratio": "BNPL / income",
    "savings_buffer_ratio": "Savings buffer", "avg_app_logins": "App logins / month",
    "avg_upi_transactions": "UPI txns / month", "loan_to_income": "Loan / monthly income",
    "loan_tenure_months": "Loan tenure", "interest_rate_annual": "Interest rate",
    "previous_micro_loans": "Previous micro-loans", "repayment_ratio": "Prior on-time repayment share",
}


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw application fields -> model features (pure function, used by training and the dashboard)."""
    # ordinal encodings and ratios mirror the Review 1 feature engineering
    f = pd.DataFrame(index=raw.index)
    f["age"] = raw["age"]
    f["gig_tenure_months"] = raw["gig_tenure_months"]
    f["has_bank_account"] = raw["has_bank_account"]
    f["num_dependents"] = raw["num_dependents"]
    f["location_code"] = raw["location_type"].map(ORDINAL_MAPS["location_type"])
    f["education_code"] = raw["education_level"].map(ORDINAL_MAPS["education_level"])
    income = raw["avg_monthly_income"].clip(lower=1)  # floor at 1 to avoid divide-by-zero below
    f["avg_monthly_income"] = raw["avg_monthly_income"]
    f["income_volatility_index"] = raw["std_monthly_income"] / income
    f["income_trend_slope"] = raw["income_trend_slope"]
    f["utility_reliability_score"] = raw["utility_on_time_count"] / 12
    f["avg_utility_delay_days"] = raw["avg_utility_delay_days"]
    f["max_utility_delay_days"] = raw["max_utility_delay_days"]
    f["avg_days_between_recharges"] = raw["avg_days_between_recharges"]
    f["debt_to_inflow_ratio"] = (raw["avg_bnpl_monthly_amount"] / income).clip(0, 2)
    f["savings_buffer_ratio"] = (raw["avg_eom_balance"] / raw["avg_monthly_outflow"].clip(lower=1)).clip(0, 5)
    f["avg_app_logins"] = raw["avg_app_logins"]
    f["avg_upi_transactions"] = raw["avg_upi_transactions"]
    f["loan_to_income"] = raw["loan_amount_requested"] / income
    f["loan_tenure_months"] = raw["loan_tenure_months"]
    f["interest_rate_annual"] = raw["interest_rate_annual"]
    f["previous_micro_loans"] = raw["previous_micro_loans"]
    f["repayment_ratio"] = np.where(raw["previous_micro_loans"] > 0,
                                    raw["previous_loans_repaid_on_time"] / raw["previous_micro_loans"].clip(lower=1), -1)
    return f


def break_even_pd(rate_annual_pct, tenure_months):
    """PD at which expected interest income equals expected credit loss (decline above this)."""
    margin = np.maximum(np.asarray(rate_annual_pct) / 100 - COST_OF_FUNDS, 0.001) * np.asarray(tenure_months) / 12
    return margin / (margin + LGD)


def risk_band(pd_value):
    """Map a probability of default to a letter band A (safest) to E (riskiest)."""
    for upper, band in [(0.04, "A"), (0.07, "B"), (0.11, "C"), (0.16, "D")]:
        if pd_value < upper:
            return band
    return "E"


def train():
    """Train the LightGBM PD model, write the joblib + model card, return (model, card)."""
    df = pd.read_csv(BORROWER_PROFILES_PATH)
    X = build_features(df)
    y = df["loan_default_status"]
    train_idx, test_idx = train_test_split(df.index, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    model = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.02, num_leaves=15, min_child_samples=80,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=2.0,
        random_state=RANDOM_STATE, verbose=-1,
    )
    model.fit(X.loc[train_idx], y.loc[train_idx])

    p_test = model.predict_proba(X.loc[test_idx])[:, 1]
    y_test = y.loc[test_idx]
    oracle = roc_auc_score(y_test, df.loc[test_idx, "default_probability_true"])
    deciles = pd.qcut(p_test, 10, labels=False, duplicates="drop")
    calib = pd.DataFrame({"pred": p_test, "obs": y_test.values, "decile": deciles}).groupby("decile")[["pred", "obs"]].mean()

    # simulate the decision policy on the held-out set: approve when PD is below break-even
    be = break_even_pd(df.loc[test_idx, "interest_rate_annual"], df.loc[test_idx, "loan_tenure_months"])
    approve = p_test < be
    card = {
        "model": "LightGBM (calibrated, unweighted logloss)",
        "features": list(X.columns),
        "feature_labels": FEATURE_LABELS,
        "input_fields": INPUT_FIELDS,
        "test_auc": round(float(roc_auc_score(y_test, p_test)), 4),
        "test_pr_auc": round(float(average_precision_score(y_test, p_test)), 4),
        "test_brier": round(float(brier_score_loss(y_test, p_test)), 5),
        "oracle_auc_true_pd": round(float(oracle), 4),
        "mean_pd_test": round(float(p_test.mean()), 4),
        "observed_default_rate_test": round(float(y_test.mean()), 4),
        "calibration_by_decile": calib.round(4).to_dict(orient="index"),
        "policy": {
            "lgd": LGD, "cost_of_funds": COST_OF_FUNDS,
            "approval_rate_test": round(float(approve.mean()), 4),
            "default_rate_approved_test": round(float(y_test[approve].mean()), 4),
            "default_rate_declined_test": round(float(y_test[~approve].mean()), 4),
        },
        "risk_bands": {"A": "< 4%", "B": "4-7%", "C": "7-11%", "D": "11-16%", "E": ">= 16%"},
    }
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    with open(CARD_PATH, "w") as f:
        json.dump(card, f, indent=2)
    return model, card


def main():
    _, card = train()
    print("Underwriting model trained")
    for k in ["test_auc", "test_pr_auc", "test_brier", "oracle_auc_true_pd", "mean_pd_test", "observed_default_rate_test"]:
        print(f"  {k:<28}: {card[k]}")
    print("  policy:", card["policy"])
    print(f"Saved {MODEL_PATH} and {CARD_PATH}")


if __name__ == "__main__":
    main()
