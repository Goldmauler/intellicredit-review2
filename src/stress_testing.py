"""
IntelliCredit - Regulatory Macroeconomic Stress Testing & Capital Adequacy Engine
================================================================================
Simulates portfolio losses, credit default migration, and Capital Adequacy Ratio (CRAR)
depletion under RBI supervisory stress scenarios and macroeconomic shocks.

Methodology:
  * Employs Wilson (1997) CreditPortfolioView logit link:
      logit(PD_stressed) = logit(PD_base) + BETA_STRESS * Z_macro
  * Evaluates three regulatory macroeconomic scenarios:
      1. Baseline: Trend economic growth (GDP +6.5%, Repo 6.5%, CPI 5.0%)
      2. Moderate Adverse: Cyclical slowdown (GDP -200 bps, Repo +100 bps, CPI +150 bps)
      3. Severe Adverse: Stagflation shock (GDP -400 bps, Repo +250 bps, Unemployment +300 bps)
  * Translates stressed defaults into Expected Credit Losses and measures regulatory
    capital buffer absorption against RBI NBFC-MFI capital requirements (CRAR minimum 15.0%).

Author: Portfolio Risk Modeling Team
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from src.config import DATA_DIR

PORTFOLIO_DIR = os.path.join(DATA_DIR, "portfolio")
SAMPLE_DIR = os.path.join(DATA_DIR, "sample")
LOAN_BOOK_PATH = os.path.join(PORTFOLIO_DIR, "loan_book.csv")
SAMPLE_BOOK_PATH = os.path.join(SAMPLE_DIR, "sample_loan_book.csv")

# Regulatory Capital Constants
BASE_CRAR_PERCENT = 18.5     # Baseline Capital to Risk-Weighted Assets Ratio (18.5%)
MIN_REGULATORY_CRAR = 15.0   # RBI NBFC prudential minimum requirement (15.0%)
RWA_DENSITY_UNSECURED = 1.0  # 100% Risk-Weight for unsecured micro-loans
DEFAULT_LGD = 0.50           # Assumed 50% loss given default
BETA_STRESS = 0.75           # Sensitivity of default hazard to macro z-index


# Standard Supervisory Stress Scenarios
SCENARIOS = {
    "Baseline": {
        "macro_shock_z": 0.0,
        "gdp_growth_pct": 6.5,
        "repo_rate_pct": 6.50,
        "cpi_inflation_pct": 5.0,
        "description": "Baseline neutral economic growth and steady interest rates.",
    },
    "Moderate_Adverse": {
        "macro_shock_z": 1.25,
        "gdp_growth_pct": 4.5,
        "repo_rate_pct": 7.50,
        "cpi_inflation_pct": 6.5,
        "description": "Cyclical slowdown, rising borrowing costs, and elevated food/fuel inflation.",
    },
    "Severe_Adverse": {
        "macro_shock_z": 2.50,
        "gdp_growth_pct": 2.5,
        "repo_rate_pct": 9.00,
        "cpi_inflation_pct": 8.0,
        "description": "Acute stagflation shock, severe liquidity contraction, and consumer stress.",
    },
}


def apply_macro_stress_to_pds(
    base_pds: np.ndarray,
    macro_shock_z: float,
    beta: float = BETA_STRESS,
) -> np.ndarray:
    """
    Applies macroeconomic stress shock to baseline PDs via logit link.

    logit(PD_stressed) = logit(PD_base) + beta * shock_z
    """
    # Clip baseline PDs away from 0 and 1 for numerical stability
    clipped_pds = np.clip(base_pds, 1e-4, 0.9999)
    logit_pds = logit(clipped_pds)
    stressed_logits = logit_pds + (beta * macro_shock_z)
    return expit(stressed_logits)


def simulate_stress_scenario(
    loan_df: pd.DataFrame,
    scenario_name: str,
    scenario_params: Dict[str, float],
    lgd: float = DEFAULT_LGD,
    base_crar: float = BASE_CRAR_PERCENT,
) -> Dict[str, float]:
    """
    Simulates stressed defaults, credit losses, and capital adequacy depletion for one scenario.
    """
    ead_col = "loan_amount" if "loan_amount" in loan_df.columns else "loan_amount_requested"
    eads = loan_df[ead_col].to_numpy(dtype=float)
    total_ead = float(np.sum(eads))

    base_pds = (
        loan_df["default_probability_true"].to_numpy(dtype=float)
        if "default_probability_true" in loan_df.columns
        else np.full(len(loan_df), 0.08)
    )

    shock_z = scenario_params["macro_shock_z"]
    stressed_pds = apply_macro_stress_to_pds(base_pds, shock_z)

    # Stressed Expected Losses
    stressed_losses = eads * stressed_pds * lgd
    total_stressed_loss = float(np.sum(stressed_losses))
    stressed_default_rate = float(np.average(stressed_pds, weights=eads))

    # Capital Adequacy Impact
    total_rwa = total_ead * RWA_DENSITY_UNSECURED
    base_capital = total_rwa * (base_crar / 100.0)
    stressed_capital = max(0.0, base_capital - total_stressed_loss)
    stressed_crar = (stressed_capital / total_rwa) * 100.0 if total_rwa > 0 else 0.0
    capital_buffer_absorbed = base_capital - stressed_capital

    breached_minimum = stressed_crar < MIN_REGULATORY_CRAR

    return {
        "scenario": scenario_name,
        "macro_shock_z": shock_z,
        "stressed_default_rate_pct": stressed_default_rate * 100.0,
        "total_stressed_loss_inr": total_stressed_loss,
        "stressed_loss_ratio_pct": (total_stressed_loss / total_ead) * 100.0 if total_ead > 0 else 0.0,
        "stressed_crar_pct": stressed_crar,
        "crar_depletion_bps": (base_crar - stressed_crar) * 100.0,
        "capital_absorbed_inr": capital_buffer_absorbed,
        "breaches_rbi_minimum": breached_minimum,
    }


def run_portfolio_stress_test(
    loan_df: Optional[pd.DataFrame] = None,
    scenarios: Optional[Dict[str, Dict]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """Runs supervisory macroeconomic stress tests across all scenarios."""
    if loan_df is None:
        path = LOAN_BOOK_PATH if os.path.exists(LOAN_BOOK_PATH) else SAMPLE_BOOK_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"Loan book data not found at {path}")
        loan_df = pd.read_csv(path)

    test_scenarios = scenarios or SCENARIOS
    results_list = []
    results_dict = {}

    for name, params in test_scenarios.items():
        res = simulate_stress_scenario(loan_df, name, params)
        results_list.append(res)
        results_dict[name] = res

    summary_df = pd.DataFrame(results_list).set_index("scenario")
    return summary_df, results_dict


if __name__ == "__main__":
    print("IntelliCredit - Executing Macroeconomic Supervisory Stress Test...")
    summary_table, _ = run_portfolio_stress_test()
    print("\nStress Test Summary across Regulatory Scenarios:")
    cols_to_show = [
        "macro_shock_z", "stressed_default_rate_pct", "total_stressed_loss_inr",
        "stressed_loss_ratio_pct", "stressed_crar_pct", "breaches_rbi_minimum"
    ]
    print(summary_table[cols_to_show].round(2))

