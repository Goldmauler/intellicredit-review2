"""
IntelliCredit - Delinquency Transition Matrix & Roll-Rate Engine
================================================================
Analyzes month-over-month delinquency state transitions across borrowers
in the micro-lending portfolio panel.

Delinquency Buckets:
  - Current: 0 DPD (on schedule)
  - 30_DPD: 1 - 30 DPD (missed 1 EMI)
  - 60_DPD: 31 - 60 DPD (missed 2 EMIs)
  - Default: 90+ DPD (credit impaired, non-performing asset)
  - Settled: Loan matured and fully repaid

Key Outputs:
  1. Empirical 1-month Transition Probability Matrix P_{ij} = P(S_{t+1}=j | S_t=i)
  2. Roll-forward rates (Current -> 30, 30 -> 60, 60 -> Default)
  3. Cure rates (30 -> Current, 60 -> Current)
  4. Multi-period absorbing state projections P^k

Author: Portfolio Risk Modeling Team
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import DATA_DIR

PORTFOLIO_DIR = os.path.join(DATA_DIR, "portfolio")
SAMPLE_DIR = os.path.join(DATA_DIR, "sample")
LOAN_PERF_PATH = os.path.join(PORTFOLIO_DIR, "loan_performance.csv")
SAMPLE_PERF_PATH = os.path.join(SAMPLE_DIR, "sample_loan_performance.csv")

# Standardized Delinquency Buckets
STATE_CURRENT = "Current"
STATE_30_DPD = "30_DPD"
STATE_60_DPD = "60_DPD"
STATE_DEFAULT = "Default"
STATE_SETTLED = "Settled"

STANDARD_STATES = [STATE_CURRENT, STATE_30_DPD, STATE_60_DPD, STATE_DEFAULT, STATE_SETTLED]


def map_dpd_to_state(dpd: int, outstanding_principal: float = 1.0) -> str:
    """Maps Days Past Due (DPD) and balance to a standardized credit risk state."""
    if outstanding_principal <= 0.0:
        return STATE_SETTLED
    if dpd >= 90:
        return STATE_DEFAULT
    if dpd >= 60:
        return STATE_60_DPD
    if dpd >= 30:
        return STATE_30_DPD
    return STATE_CURRENT


def extract_monthly_transitions(
    perf_df: pd.DataFrame,
    id_col: str = "borrower_id",
    month_col: str = "month",
    dpd_col: str = "dpd",
    balance_col: str = "outstanding_principal",
) -> pd.DataFrame:
    """
    Extracts consecutive monthly state pairs (S_t, S_{t+1}) for each unique borrower.

    Parameters:
        perf_df: Historical loan performance panel.
        id_col: Column name identifying the loan/borrower.
        month_col: Column name for month/period.
        dpd_col: Column name for Days Past Due.
        balance_col: Column name for outstanding balance.

    Returns:
        DataFrame containing 'from_state', 'to_state', and 'borrower_id'.
    """
    df = perf_df.copy()
    if "risk_state" not in df.columns:
        df["risk_state"] = [
            map_dpd_to_state(d, b)
            for d, b in zip(df[dpd_col], df[balance_col])
        ]

    # Sort sequentially per borrower
    df = df.sort_values(by=[id_col, month_col])
    df["next_state"] = df.groupby(id_col)["risk_state"].shift(-1)

    # For loans that terminate after default, default is an absorbing state
    valid_transitions = df.dropna(subset=["next_state"]).copy()
    
    return valid_transitions[[id_col, "risk_state", "next_state"]].rename(
        columns={"risk_state": "from_state", "next_state": "to_state"}
    )


def compute_transition_matrix(
    transitions_df: pd.DataFrame,
    states: Optional[List[str]] = None,
    enforce_absorbing_default: bool = True,
) -> pd.DataFrame:
    """
    Calculates empirical transition probability matrix P_{ij} = N_{ij} / sum_k N_{ik}.

    Parameters:
        transitions_df: DataFrame with 'from_state' and 'to_state' columns.
        states: List of state names to index the matrix.
        enforce_absorbing_default: If True, sets Default -> Default = 1.0.

    Returns:
        Square DataFrame of transition probabilities summing to 1.0 across rows.
    """
    if states is None:
        observed_states = list(
            set(transitions_df["from_state"]).union(set(transitions_df["to_state"]))
        )
        states = [s for s in STANDARD_STATES if s in observed_states]
        # Include any other states observed
        for s in observed_states:
            if s not in states:
                states.append(s)

    # Frequency cross-tabulation
    counts = pd.crosstab(
        transitions_df["from_state"],
        transitions_df["to_state"]
    ).reindex(index=states, columns=states, fill_value=0)

    # Convert counts to conditional transition probabilities
    row_sums = counts.sum(axis=1)
    matrix = counts.divide(row_sums.replace(0, 1), axis=0)

    # Handle rows with no transitions (e.g. absorbing states)
    for s in states:
        if row_sums[s] == 0:
            matrix.loc[s, :] = 0.0
            matrix.loc[s, s] = 1.0

    if enforce_absorbing_default and STATE_DEFAULT in states:
        matrix.loc[STATE_DEFAULT, :] = 0.0
        matrix.loc[STATE_DEFAULT, STATE_DEFAULT] = 1.0

    if STATE_SETTLED in states:
        matrix.loc[STATE_SETTLED, :] = 0.0
        matrix.loc[STATE_SETTLED, STATE_SETTLED] = 1.0

    return matrix.round(4)


def compute_roll_rates(matrix_df: pd.DataFrame) -> Dict[str, float]:
    """
    Extracts key roll-forward and cure rates from the transition probability matrix.

    Returns:
        Dictionary containing:
          - roll_current_to_30: P(30_DPD | Current)
          - roll_30_to_60: P(60_DPD | 30_DPD)
          - roll_60_to_default: P(Default | 60_DPD)
          - cure_30_to_current: P(Current | 30_DPD)
          - cure_60_to_current: P(Current | 60_DPD)
    """
    def _safe_get(f: str, t: str) -> float:
        if f in matrix_df.index and t in matrix_df.columns:
            return float(matrix_df.loc[f, t])
        return 0.0

    return {
        "roll_current_to_30": _safe_get(STATE_CURRENT, STATE_30_DPD),
        "roll_30_to_60": _safe_get(STATE_30_DPD, STATE_60_DPD),
        "roll_60_to_default": _safe_get(STATE_60_DPD, STATE_DEFAULT),
        "cure_30_to_current": _safe_get(STATE_30_DPD, STATE_CURRENT),
        "cure_60_to_current": _safe_get(STATE_60_DPD, STATE_CURRENT),
    }


def project_absorbing_probabilities(
    matrix_df: pd.DataFrame, horizons: List[int] = [3, 6, 12]
) -> Dict[int, pd.DataFrame]:
    """
    Projects cumulative multi-period transition matrices using matrix exponentiation P^k.

    Parameters:
        matrix_df: 1-month transition probability matrix.
        horizons: List of forecast horizons in months.

    Returns:
        Dictionary mapping horizon -> P^k DataFrame.
    """
    p_np = matrix_df.to_numpy(dtype=float)
    results = {}
    for h in horizons:
        p_k = np.linalg.matrix_power(p_np, h)
        results[h] = pd.DataFrame(
            p_k, index=matrix_df.index, columns=matrix_df.columns
        ).round(4)
    return results


def run_transition_analysis(
    perf_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[int, pd.DataFrame]]:
    """Loads performance data and computes the full transition analysis."""
    path = perf_path or (LOAN_PERF_PATH if os.path.exists(LOAN_PERF_PATH) else SAMPLE_PERF_PATH)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Performance data not found at {path}")

    perf_df = pd.read_csv(path)
    transitions = extract_monthly_transitions(perf_df)
    matrix = compute_transition_matrix(transitions)
    roll_rates = compute_roll_rates(matrix)
    projections = project_absorbing_probabilities(matrix, horizons=[3, 6, 12])
    return matrix, roll_rates, projections


if __name__ == "__main__":
    print("IntelliCredit - Calculating Portfolio Delinquency Transition Matrix...")
    mat, rolls, projs = run_transition_analysis()
    print("\n1-Month Transition Probability Matrix P:")
    print(mat)
    print("\nCredit Roll & Cure Rates:")
    for k, v in rolls.items():
        print(f"  {k:22s}: {v:.2%}")
    print("\n12-Month Projected State Matrix P^12:")
    print(projs[12])

