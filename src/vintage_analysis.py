"""
IntelliCredit - Portfolio Vintage Analysis & Cumulative Net Loss (CNL)
======================================================================
Groups micro-lending facilities into origination cohorts (quarterly vintages)
and tracks cumulative default curves and payment retention over loan lifecycle age
(Months on Books - MOB).

Key Metrics:
  - Cumulative Default Rate (% of cohort defaulted by MOB k)
  - Cumulative Net Loss (CNL % of cohort origination principal)
  - Early Payment Default (EPD-30 at MOB 3, EPD-60 at MOB 6)
  - Vintage Maturation Index (default rate acceleration during early MOBs)

Author: Portfolio Risk Modeling Team
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import DATA_DIR

PORTFOLIO_DIR = os.path.join(DATA_DIR, "portfolio")
SAMPLE_DIR = os.path.join(DATA_DIR, "sample")
LOAN_BOOK_PATH = os.path.join(PORTFOLIO_DIR, "loan_book.csv")
LOAN_PERF_PATH = os.path.join(PORTFOLIO_DIR, "loan_performance.csv")
SAMPLE_BOOK_PATH = os.path.join(SAMPLE_DIR, "sample_loan_book.csv")
SAMPLE_PERF_PATH = os.path.join(SAMPLE_DIR, "sample_loan_performance.csv")


def assign_origination_quarter(month_str: str) -> str:
    """Converts YYYY-MM into quarterly vintage tag 'YYYY-Q1'..'YYYY-Q4'."""
    try:
        parts = month_str.split("-")
        year, month = int(parts[0]), int(parts[1])
        quarter = (month - 1) // 3 + 1
        return f"{year}-Q{quarter}"
    except Exception:
        return "Unknown"


def build_vintage_dataset(
    book_df: pd.DataFrame,
    perf_df: pd.DataFrame,
    max_mob: int = 24,
) -> pd.DataFrame:
    """
    Merges origination cohorts with monthly performance panel.

    Parameters:
        book_df: Loan book DataFrame containing 'borrower_id', 'origination_month', 'loan_amount'.
        perf_df: Monthly performance panel with 'borrower_id', 'loan_age', 'dpd', 'status'.
        max_mob: Maximum Months on Books to track.

    Returns:
        Consolidated DataFrame with vintage cohorts and MOB milestones.
    """
    b_df = book_df[["borrower_id", "origination_month", "loan_amount"]].copy()
    b_df["vintage_quarter"] = b_df["origination_month"].apply(assign_origination_quarter)

    p_df = perf_df[["borrower_id", "loan_age", "dpd", "status"]].copy()
    p_df = p_df[p_df["loan_age"] <= max_mob]

    merged = p_df.merge(b_df, on="borrower_id", how="inner")
    merged["is_default"] = (merged["dpd"] >= 90).astype(int)
    return merged


def compute_vintage_matrix(
    vintage_df: pd.DataFrame,
    metric: str = "default_rate",
    max_mob: int = 18,
) -> pd.DataFrame:
    """
    Computes a cohort-by-MOB matrix of cumulative default rates.

    Parameters:
        vintage_df: Merged vintage DataFrame from build_vintage_dataset.
        metric: 'default_rate' (percentage) or 'default_count'.
        max_mob: Maximum MOB columns to retain.

    Returns:
        Pivot table with vintage quarters as rows and MOB 1..N as columns.
    """
    # Track cumulative default event per borrower
    cohort_sizes = vintage_df.groupby("vintage_quarter")["borrower_id"].nunique()

    # Find earliest default MOB for each defaulting borrower
    defaults = vintage_df[vintage_df["is_default"] == 1].groupby(
        ["vintage_quarter", "borrower_id"]
    )["loan_age"].min().reset_index()

    vintages = sorted(vintage_df["vintage_quarter"].unique())
    mobs = list(range(1, max_mob + 1))

    matrix_rows = []
    for v in vintages:
        n_loans = cohort_sizes.get(v, 0)
        if n_loans == 0:
            continue
        v_defaults = defaults[defaults["vintage_quarter"] == v]

        row = {"vintage_quarter": v, "cohort_size": n_loans}
        cum_defs = 0
        for m in mobs:
            new_defs = (v_defaults["loan_age"] == m).sum()
            cum_defs += new_defs
            if metric == "default_rate":
                row[f"MOB_{m:02d}"] = round((cum_defs / n_loans) * 100.0, 2)
            else:
                row[f"MOB_{m:02d}"] = cum_defs
        matrix_rows.append(row)

    res_df = pd.DataFrame(matrix_rows).set_index("vintage_quarter")
    return res_df


def extract_epd_metrics(vintage_matrix_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts Early Payment Default (EPD) benchmarks:
      - EPD-30: Cumulative default rate at MOB 3
      - EPD-60: Cumulative default rate at MOB 6
      - Mature-12: Cumulative default rate at MOB 12
    """
    df = vintage_matrix_df.copy()
    epd_cols = {}
    if "MOB_03" in df.columns:
        epd_cols["EPD_MOB3_pct"] = df["MOB_03"]
    if "MOB_06" in df.columns:
        epd_cols["EPD_MOB6_pct"] = df["MOB_06"]
    if "MOB_12" in df.columns:
        epd_cols["Mature_MOB12_pct"] = df["MOB_12"]

    epd_df = pd.DataFrame(epd_cols, index=df.index)
    epd_df.insert(0, "cohort_size", df["cohort_size"])
    return epd_df


def run_vintage_analysis(
    max_mob: int = 18,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Executes the end-to-end vintage analysis pipeline."""
    b_path = LOAN_BOOK_PATH if os.path.exists(LOAN_BOOK_PATH) else SAMPLE_BOOK_PATH
    p_path = LOAN_PERF_PATH if os.path.exists(LOAN_PERF_PATH) else SAMPLE_PERF_PATH

    if not os.path.exists(b_path) or not os.path.exists(p_path):
        raise FileNotFoundError("Vintage analysis requires both loan_book.csv and loan_performance.csv")

    book_df = pd.read_csv(b_path)
    perf_df = pd.read_csv(p_path)

    vintage_df = build_vintage_dataset(book_df, perf_df, max_mob=max_mob)
    matrix = compute_vintage_matrix(vintage_df, metric="default_rate", max_mob=max_mob)
    epd = extract_epd_metrics(matrix)

    # Average metrics across mature cohorts
    summary = {
        "avg_epd_mob3_pct": float(epd["EPD_MOB3_pct"].mean()) if "EPD_MOB3_pct" in epd else 0.0,
        "avg_epd_mob6_pct": float(epd["EPD_MOB6_pct"].mean()) if "EPD_MOB6_pct" in epd else 0.0,
        "avg_mob12_pct": float(epd["Mature_MOB12_pct"].mean()) if "Mature_MOB12_pct" in epd else 0.0,
        "total_cohorts_tracked": len(epd),
    }

    return matrix, epd, summary


if __name__ == "__main__":
    print("IntelliCredit - Calculating Portfolio Vintage Curves & EPD...")
    mat, epd_tbl, summ = run_vintage_analysis(max_mob=12)
    print(f"\nTracked {summ['total_cohorts_tracked']} Quarterly Vintages across Months on Books (MOB):")
    print(f"  * Average EPD at MOB 3 : {summ['avg_epd_mob3_pct']:.2f}%")
    print(f"  * Average EPD at MOB 6 : {summ['avg_epd_mob6_pct']:.2f}%")
    print(f"  * Average Default MOB 12: {summ['avg_mob12_pct']:.2f}%")
    print("\nRecent Cohort EPD Benchmarks:")
    print(epd_tbl.tail(8))

