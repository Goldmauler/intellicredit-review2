"""
IntelliCredit - IFRS 9 / Ind AS 109 Expected Credit Loss (ECL) Framework
========================================================================
Implements regulatory 3-stage Expected Credit Loss (ECL) provisioning for
micro-lending portfolios in compliance with Ind AS 109 / IFRS 9 standards.

Staging Architecture:
  * Stage 1 (Performing): Low credit risk (0 DPD, stable PD).
      Provision = 12-Month ECL = PD_12m * LGD * EAD * DF
  * Stage 2 (Underperforming / SICR): Significant Increase in Credit Risk
      (30-60 DPD or PD jump >= 2.0x origination baseline).
      Provision = Lifetime ECL across remaining loan tenure.
  * Stage 3 (Credit-Impaired): Non-performing / Defaulted (90+ DPD).
      Provision = Lifetime ECL with PD = 1.0 (impaired asset).

Forward-Looking Macroeconomic Overlays:
  IFRS 9 mandates probability-weighted ECL across multiple macroeconomic scenarios:
    - Base Case (50% weight): Neutral trend
    - Upside (20% weight): Macro tailwinds (GDP +1.5%, repo -50 bps)
    - Downside (30% weight): Adverse macro shock (GDP -2.5%, CPI +200 bps)

Author: Portfolio Risk Modeling Team
"""

import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import DATA_DIR

PORTFOLIO_DIR = os.path.join(DATA_DIR, "portfolio")
SAMPLE_DIR = os.path.join(DATA_DIR, "sample")
LOAN_BOOK_PATH = os.path.join(PORTFOLIO_DIR, "loan_book.csv")
SAMPLE_BOOK_PATH = os.path.join(SAMPLE_DIR, "sample_loan_book.csv")

# Regulatory Parameters for Micro-Lending Portfolios
DEFAULT_LGD = 0.50          # Loss Given Default (unsecured retail/fintech benchmark: 50%)
DEFAULT_DISCOUNT_RATE = 0.14 # Annual Effective Interest Rate (14%)
SICR_PD_RATIO = 2.0         # Relative PD threshold for Stage 2 classification
STAGE_1 = "Stage_1"
STAGE_2 = "Stage_2"
STAGE_3 = "Stage_3"


def classify_ecl_stage(
    dpd: int,
    current_pd: float,
    orig_pd: Optional[float] = None,
    sicr_ratio: float = SICR_PD_RATIO,
) -> str:
    """
    Classifies a loan into Stage 1, Stage 2, or Stage 3 based on DPD and SICR.

    Parameters:
        dpd: Days Past Due.
        current_pd: Current estimated probability of default.
        orig_pd: Baseline probability of default at origination.
        sicr_ratio: Relative PD increase threshold denoting SICR.

    Returns:
        One of 'Stage_1', 'Stage_2', 'Stage_3'.
    """
    if dpd >= 90:
        return STAGE_3
    
    # Check for Significant Increase in Credit Risk (SICR)
    has_sicr = False
    if dpd >= 30:
        has_sicr = True
    elif orig_pd is not None and orig_pd > 0:
        if (current_pd / orig_pd) >= sicr_ratio:
            has_sicr = True

    return STAGE_2 if has_sicr else STAGE_1


def calculate_loan_ecl(
    ead: float,
    pd_annual: float,
    lgd: float = DEFAULT_LGD,
    stage: str = STAGE_1,
    tenure_months: int = 12,
    annual_discount_rate: float = DEFAULT_DISCOUNT_RATE,
) -> float:
    """
    Calculates expected credit loss for an individual loan facility.

    ECL = EAD * LGD * Effective_PD * Discount_Factor
    """
    if ead <= 0.0:
        return 0.0

    monthly_rate = annual_discount_rate / 12.0

    if stage == STAGE_3:
        # Credit-impaired: PD = 100%, discounted recovery over 6 months workout
        df = 1.0 / ((1.0 + monthly_rate) ** 6)
        return float(round(ead * lgd * 1.0 * df, 2))

    if stage == STAGE_1:
        # 12-Month ECL
        df = 1.0 / ((1.0 + monthly_rate) ** 6)  # Midpoint discount
        effective_pd = min(1.0, max(0.0, pd_annual))
        return float(round(ead * lgd * effective_pd * df, 2))

    # Stage 2: Lifetime ECL across remaining tenure
    tenure_years = max(1.0, tenure_months / 12.0)
    # Lifetime PD = 1 - (1 - annual_PD)^tenure_years
    lifetime_pd = 1.0 - ((1.0 - min(0.99, max(0.0, pd_annual))) ** tenure_years)
    df = 1.0 / ((1.0 + monthly_rate) ** (tenure_months / 2.0))
    return float(round(ead * lgd * lifetime_pd * df, 2))


def compute_portfolio_ecl(
    loan_df: pd.DataFrame,
    perf_df: Optional[pd.DataFrame] = None,
    lgd: float = DEFAULT_LGD,
    discount_rate: float = DEFAULT_DISCOUNT_RATE,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Calculates portfolio-wide ECL provisions by stage.

    Parameters:
        loan_df: DataFrame containing loan book data.
        perf_df: Optional DataFrame of monthly loan performance panel.
        lgd: Assumed loss given default rate.
        discount_rate: Annual effective interest rate.

    Returns:
        Tuple of (annotated DataFrame, summary metrics dictionary).
    """
    df = loan_df.copy()

    # If performance panel is provided, extract latest status and DPD per borrower
    if perf_df is not None and not perf_df.empty:
        latest_perf = (
            perf_df.sort_values("month")
            .groupby("borrower_id")
            .last()
            .reset_index()
        )
        merge_cols = ["borrower_id"]
        for col in ["dpd", "outstanding_principal", "status"]:
            if col in latest_perf.columns:
                if col in df.columns:
                    df = df.drop(columns=[col])
                df = df.merge(latest_perf[["borrower_id", col]], on="borrower_id", how="left")

    # Resolve EAD column
    if "outstanding_principal" in df.columns:
        df["ead"] = df["outstanding_principal"].clip(lower=0.0)
    elif "loan_amount" in df.columns:
        df["ead"] = df["loan_amount"].astype(float)
    elif "loan_amount_requested" in df.columns:
        df["ead"] = df["loan_amount_requested"].astype(float)
    else:
        raise ValueError("Loan dataset missing principal/EAD column.")

    # Resolve PD column
    if "default_probability_true" in df.columns:
        df["pd_rate"] = df["default_probability_true"]
    elif "loan_default_status" in df.columns:
        df["pd_rate"] = df["loan_default_status"].astype(float) * 0.7 + 0.05
    else:
        df["pd_rate"] = 0.08  # Default baseline assumption

    dpd_vals = df["dpd"] if "dpd" in df.columns else np.zeros(len(df), dtype=int)
    tenure_vals = df["loan_tenure_months"] if "loan_tenure_months" in df.columns else np.full(len(df), 12)

    stages = [classify_ecl_stage(d, p) for d, p in zip(dpd_vals, df["pd_rate"])]
    df["ecl_stage"] = stages

    ecl_values = [
        calculate_loan_ecl(
            ead=ead,
            pd_annual=pd_val,
            lgd=lgd,
            stage=stg,
            tenure_months=int(ten),
            annual_discount_rate=discount_rate,
        )
        for ead, pd_val, stg, ten in zip(df["ead"], df["pd_rate"], stages, tenure_vals)
    ]
    df["ecl_provision"] = ecl_values

    # Aggregate summaries
    total_ead = float(df["ead"].sum())
    total_ecl = float(df["ecl_provision"].sum())

    summary = {
        "total_ead_inr": total_ead,
        "total_ecl_provision_inr": total_ecl,
        "overall_coverage_ratio": (total_ecl / total_ead) if total_ead > 0 else 0.0,
        "stage_1_loans": int((df["ecl_stage"] == STAGE_1).sum()),
        "stage_1_ecl_inr": float(df.loc[df["ecl_stage"] == STAGE_1, "ecl_provision"].sum()),
        "stage_2_loans": int((df["ecl_stage"] == STAGE_2).sum()),
        "stage_2_ecl_inr": float(df.loc[df["ecl_stage"] == STAGE_2, "ecl_provision"].sum()),
        "stage_3_loans": int((df["ecl_stage"] == STAGE_3).sum()),
        "stage_3_ecl_inr": float(df.loc[df["ecl_stage"] == STAGE_3, "ecl_provision"].sum()),
    }

    return df, summary


def compute_scenario_weighted_ecl(
    base_summary: Dict[str, float],
    upside_scalar: float = 0.85,
    downside_scalar: float = 1.35,
    weights: Tuple[float, float, float] = (0.20, 0.50, 0.30),
) -> Dict[str, float]:
    """
    Computes forward-looking weighted ECL across Upside, Base, and Downside economic states.

    Weights: (Upside 20%, Base 50%, Downside 30%)
    """
    base_ecl = base_summary["total_ecl_provision_inr"]
    upside_ecl = base_ecl * upside_scalar
    downside_ecl = base_ecl * downside_scalar

    w_up, w_base, w_down = weights
    weighted_ecl = (w_up * upside_ecl) + (w_base * base_ecl) + (w_down * downside_ecl)
    total_ead = base_summary["total_ead_inr"]

    return {
        "upside_ecl_inr": upside_ecl,
        "base_ecl_inr": base_ecl,
        "downside_ecl_inr": downside_ecl,
        "weighted_ecl_inr": weighted_ecl,
        "weighted_coverage_ratio": (weighted_ecl / total_ead) if total_ead > 0 else 0.0,
    }


def run_ecl_pipeline() -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    """Loads portfolio data and executes the full Ind AS 109 staged ECL assessment."""
    path = LOAN_BOOK_PATH if os.path.exists(LOAN_BOOK_PATH) else SAMPLE_BOOK_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Loan book data not found at {path}")

    loan_df = pd.read_csv(path)
    perf_path = os.path.join(PORTFOLIO_DIR, "loan_performance.csv")
    perf_df = pd.read_csv(perf_path) if os.path.exists(perf_path) else None

    scored_df, summary = compute_portfolio_ecl(loan_df, perf_df=perf_df)
    scenario_summary = compute_scenario_weighted_ecl(summary)
    return scored_df, summary, scenario_summary


if __name__ == "__main__":
    print("IntelliCredit - Calculating IFRS 9 / Ind AS 109 Expected Credit Loss Provisions...")
    _, summ, scen = run_ecl_pipeline()
    print(f"Total Exposure at Default (EAD) : INR {summ['total_ead_inr']:,.2f}")
    print(f"Total Base ECL Provision       : INR {summ['total_ecl_provision_inr']:,.2f} ({summ['overall_coverage_ratio']:.2%})")
    print(f"  * Stage 1 Provision (12M)     : INR {summ['stage_1_ecl_inr']:,.2f} ({summ['stage_1_loans']:,} loans)")
    print(f"  * Stage 2 Provision (Lifetime): INR {summ['stage_2_ecl_inr']:,.2f} ({summ['stage_2_loans']:,} loans)")
    print(f"  * Stage 3 Provision (Default) : INR {summ['stage_3_ecl_inr']:,.2f} ({summ['stage_3_loans']:,} loans)")
    print("\nForward-Looking Multi-Scenario ECL:")
    print(f"  * Upside (20%)   : INR {scen['upside_ecl_inr']:,.2f}")
    print(f"  * Base (50%)     : INR {scen['base_ecl_inr']:,.2f}")
    print(f"  * Downside (30%) : INR {scen['downside_ecl_inr']:,.2f}")
    print(f"  => Probability-Weighted ECL: INR {scen['weighted_ecl_inr']:,.2f} ({scen['weighted_coverage_ratio']:.2%})")
