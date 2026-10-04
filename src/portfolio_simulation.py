"""
IntelliCredit - Loan Performance Panel Simulator
================================================
Turns the 30,000 applicant profiles from data_generation.py into a month-by-month
loan book (2019-01 .. 2024-12) whose delinquency is driven by the REAL macroeconomic
series collected by src/macro_api.py (FRED + World Bank).

Design (kept deliberately transparent so it can be explained in a viva):
  * Every borrower takes the loan they applied for (amount, tenure, rate) in an
    origination month between 2018-07 and 2024-12. Volumes grow ~3x over the period
    and dip during the COVID lockdown, as fintech originations did.
  * Each borrower's true lifetime PD (default_probability_true from Review 1) is split
    into monthly "through-the-cycle" hazards h_ik that follow an early-payment-default
    hump over the loan's age k, so that, in a neutral economy, P(default over the loan)
    equals the borrower's PD.
  * The economy moves every borrower's hazard together through a logit link - the
    macro-linked default model used by CreditPortfolioView (Wilson, 1997):
        logit(h_ikt) = logit(h_ik) + BETA * stress[t] + log(season[t])
    stress[t] is a z-score composite of unemployment, GDP growth, CPI inflation, INR
    depreciation and the call-money rate, plus COVID event months.
  * Delinquency path: a first missed EMI = 30 DPD, then 60 DPD, then 90+ DPD = default
    (the same 90-day definition as the Review 1 target). Performing loans also have
    short 30/60 DPD episodes that cure, more often in stressed months.
  * Panel outcomes are a fresh Monte-Carlo draw from the same true PDs, so they agree
    with the Review 1 label in distribution (higher PD -> more defaults), not row by row.

Usage:
    python -m src.portfolio_simulation

Outputs:
    data/portfolio/loan_book.csv          one row per loan
    data/portfolio/loan_performance.csv   one row per loan-month (DPD status, balance)
    data/portfolio/portfolio_monthly.csv  monthly portfolio KPIs + macro drivers
"""

import os

import numpy as np
import pandas as pd

from src.config import BORROWER_PROFILES_PATH, DATA_DIR
from src.macro_api import MACRO_MONTHLY_PATH

PORTFOLIO_DIR = os.path.join(DATA_DIR, "portfolio")
LOAN_BOOK_PATH = os.path.join(PORTFOLIO_DIR, "loan_book.csv")
LOAN_PERFORMANCE_PATH = os.path.join(PORTFOLIO_DIR, "loan_performance.csv")
PORTFOLIO_MONTHLY_PATH = os.path.join(PORTFOLIO_DIR, "portfolio_monthly.csv")

SEED = 2026
ORIGINATION_START, WINDOW_START, WINDOW_END = "2018-07-01", "2019-01-01", "2024-12-01"

STRESS_WEIGHTS = {            # sign: + means "higher value = more stress"
    "unemployment_rate": +0.35,
    "gdp_growth": -0.30,
    "cpi_inflation_yoy": +0.15,
    "inr_depreciation_yoy": +0.10,
    "call_money_rate": +0.10,
}
LOCKDOWN_STRESS, SECOND_WAVE_STRESS = 1.6, 0.9

BETA = 0.75           # sensitivity of the monthly default hazard to the stress index (logit scale)
EPISODE_BASE = 0.012  # monthly chance of a curing 30-DPD episode for an average borrower
BETA_EPISODE = 0.45

# Month-of-year multiplier on hazards: post-festive strain in Jan-Mar, monsoon disruption
# for delivery / ride-hailing workers in Jul-Aug, festive-season income in Oct-Nov.
SEASONALITY = {1: 1.15, 2: 1.12, 3: 1.08, 4: 1.00, 5: 0.95, 6: 0.97,
               7: 1.08, 8: 1.08, 9: 1.00, 10: 0.85, 11: 0.82, 12: 0.90}


def build_stress_index(macro):
    """z-score composite of the macro drivers (z computed over the simulation window)."""
    window = macro[(macro["date"] >= ORIGINATION_START) & (macro["date"] <= WINDOW_END)]
    stress = pd.Series(0.0, index=macro.index)
    for col, w in STRESS_WEIGHTS.items():
        z = (macro[col] - window[col].mean()) / window[col].std()
        stress += w * z.fillna(0)
    stress += LOCKDOWN_STRESS * macro["covid_lockdown"] + SECOND_WAVE_STRESS * macro["covid_second_wave"]
    return stress


def age_hazard(k):
    """Early-payment-default hump: risk peaks around the 3rd-5th instalment."""
    k = np.asarray(k, dtype=float)
    return k ** 1.3 * np.exp(-k / 3.5)


def logit(p):
    return np.log(p / (1 - p))


def expit(x):
    return 1 / (1 + np.exp(-x))


def simulate(seed=SEED):
    rng = np.random.default_rng(seed)
    borrowers = pd.read_csv(BORROWER_PROFILES_PATH)
    macro = pd.read_csv(MACRO_MONTHLY_PATH, parse_dates=["date"])
    macro["stress_index"] = build_stress_index(macro)

    months = pd.date_range(ORIGINATION_START, WINDOW_END, freq="MS")
    # Month index arithmetic is easier than dates: idx 0 = 2018-07
    window_start_idx = months.get_loc(pd.Timestamp(WINDOW_START))
    last_idx = len(months) - 1

    macro_w = macro.set_index("date").reindex(months)
    stress = macro_w["stress_index"].to_numpy()
    log_season = np.log([SEASONALITY[d.month] for d in months])

    # ------------------------------------------------------------------ originations
    t = np.arange(len(months))
    orig_weight = np.exp(0.016 * t)
    orig_weight[np.isin(months.strftime("%Y-%m"), ["2020-04", "2020-05"])] *= 0.20
    orig_weight[np.isin(months.strftime("%Y-%m"), ["2020-03", "2020-06", "2021-04", "2021-05"])] *= 0.55
    orig_weight /= orig_weight.sum()
    n = len(borrowers)
    orig_idx = rng.choice(len(months), size=n, p=orig_weight)

    pd_ttc = borrowers["default_probability_true"].to_numpy()
    label = borrowers["loan_default_status"].to_numpy()
    tenure = borrowers["loan_tenure_months"].to_numpy()
    principal = borrowers["loan_amount_requested"].to_numpy(dtype=float)
    rate_m = borrowers["interest_rate_annual"].to_numpy() / 1200
    emi = principal * rate_m * (1 + rate_m) ** tenure / ((1 + rate_m) ** tenure - 1)
    pd_rel = pd_ttc / pd_ttc.mean()

    perf_rows = []
    book_rows = []
    for i in range(n):
        o, T = orig_idx[i], tenure[i]
        k = np.arange(1, T + 1)                     # loan age at each instalment
        inst_idx = o + k                            # calendar index of each instalment

        # through-the-cycle monthly hazards whose product reproduces the lifetime PD
        a = age_hazard(k)
        h_ttc = 1 - (1 - pd_ttc[i]) ** (a / a.sum())
        h_ttc = np.clip(h_ttc, 1e-6, 0.5)

        balance = principal[i]
        arrears = 0             # missed EMIs in a curing episode
        first_miss_k = None     # loan age of the missed EMI that rolls to default
        default_idx = None
        for j, (kk, ci) in enumerate(zip(k, inst_idx)):
            if ci > last_idx:
                break
            if first_miss_k is None and arrears == 0:
                h = expit(logit(h_ttc[j]) + BETA * stress[ci] + log_season[ci])
                if rng.random() < h:
                    first_miss_k = kk
            if first_miss_k is not None:
                # on the road to default: 30 -> 60 -> 90+ DPD
                dpd, paid = min(90, 30 * (kk - first_miss_k + 1)), 0
            elif arrears > 0:
                # in a short episode: 75% cure after one missed EMI, everyone cures after two
                if rng.random() < (0.75 if arrears == 1 else 1.0):
                    for _ in range(arrears):     # catch up the missed instalments
                        balance = max(0.0, balance * (1 + rate_m[i]) - emi[i])
                    arrears, dpd, paid = 0, 0, 1
                else:
                    arrears += 1
                    dpd, paid = 30 * arrears, 0
            else:
                p_ep = EPISODE_BASE * pd_rel[i] ** 0.8 * np.exp(BETA_EPISODE * stress[ci] + log_season[ci])
                if rng.random() < min(p_ep, 0.5):
                    arrears, dpd, paid = 1, 30, 0
                else:
                    dpd, paid = 0, 1
            if paid:
                balance = max(0.0, balance * (1 + rate_m[i]) - emi[i])
            status = "default" if dpd >= 90 else ("delinquent" if dpd > 0 else "current")
            if ci >= window_start_idx:
                perf_rows.append((i, ci, kk, dpd, status, paid, round(balance, 2)))
            if dpd >= 90:
                default_idx = ci
                break

        censored = int(o + T > last_idx and default_idx is None)
        book_rows.append((i, o, T, default_idx, censored))

    # ------------------------------------------------------------------ tidy outputs
    month_str = months.strftime("%Y-%m")
    ids = borrowers["borrower_id"].to_numpy()

    book = pd.DataFrame(book_rows, columns=["i", "orig_idx", "loan_tenure_months", "default_idx", "censored_at_window_end"])
    book.insert(0, "borrower_id", ids[book["i"]])
    book["origination_month"] = month_str[book["orig_idx"]]
    book["maturity_month"] = [(months[o] + pd.DateOffset(months=int(T))).strftime("%Y-%m")
                              for o, T in zip(book["orig_idx"], book["loan_tenure_months"])]
    book["defaulted_in_panel"] = book["default_idx"].notna().astype(int)
    book["default_month"] = [month_str[int(d)] if pd.notna(d) else "" for d in book["default_idx"]]
    book["loan_amount"] = principal[book["i"]]
    book["interest_rate_annual"] = borrowers["interest_rate_annual"].to_numpy()[book["i"]]
    book["emi"] = emi[book["i"]].round(2)
    book["loan_default_status"] = label[book["i"]]
    book["default_probability_true"] = pd_ttc[book["i"]]
    book = book.drop(columns=["i", "orig_idx", "default_idx"])

    perf = pd.DataFrame(perf_rows, columns=["i", "month_idx", "loan_age", "dpd", "status",
                                            "emi_paid", "outstanding_principal"])
    perf.insert(0, "borrower_id", ids[perf["i"]])
    perf.insert(1, "month", month_str[perf["month_idx"]])
    perf = perf.drop(columns=["i"])

    # monthly portfolio KPIs
    g = perf.groupby("month_idx")
    monthly = pd.DataFrame({
        "active_loans": g.size(),
        "new_defaults": g["dpd"].apply(lambda s: int((s >= 90).sum())),
        "dpd30_plus": g["dpd"].apply(lambda s: int((s >= 30).sum())),
        "dpd60_plus": g["dpd"].apply(lambda s: int((s >= 60).sum())),
        "outstanding_principal": g["outstanding_principal"].sum(),
    }).reindex(range(window_start_idx, last_idx + 1), fill_value=0)
    orig_counts = pd.Series(orig_idx).value_counts().reindex(range(len(months)), fill_value=0)
    monthly["new_originations"] = orig_counts.loc[monthly.index].to_numpy()
    monthly["default_rate"] = monthly["new_defaults"] / monthly["active_loans"]
    monthly["dpd30_plus_rate"] = monthly["dpd30_plus"] / monthly["active_loans"]
    monthly["dpd60_plus_rate"] = monthly["dpd60_plus"] / monthly["active_loans"]
    monthly.insert(0, "date", months[window_start_idx:])

    macro_cols = ["cpi_inflation_yoy", "call_money_rate", "gsec_10y_yield", "inr_per_usd",
                  "inr_depreciation_yoy", "unemployment_rate", "gdp_growth", "lending_rate",
                  "covid_lockdown", "covid_second_wave", "stress_index"]
    monthly = monthly.merge(macro[["date"] + macro_cols], on="date", how="left").reset_index(drop=True)
    perf = perf.drop(columns=["month_idx"])
    return book, perf, monthly


def main():
    os.makedirs(PORTFOLIO_DIR, exist_ok=True)
    print("=" * 70)
    print("  IntelliCredit - Loan performance panel (macro-driven)")
    print("=" * 70)
    book, perf, monthly = simulate()
    book.to_csv(LOAN_BOOK_PATH, index=False)
    perf.to_csv(LOAN_PERFORMANCE_PATH, index=False)
    monthly.to_csv(PORTFOLIO_MONTHLY_PATH, index=False)

    matured = book[book["censored_at_window_end"] == 0]
    print(f"Loans                         : {len(book):,}")
    print(f"Loan-month rows               : {len(perf):,}")
    print(f"Defaults in panel             : {book['defaulted_in_panel'].sum():,}")
    print(f"Censored (still running)      : {book['censored_at_window_end'].sum():,}")
    print(f"Lifetime default rate (closed loans): {matured['defaulted_in_panel'].mean():.2%} "
          f"(Review 1 label rate {book['loan_default_status'].mean():.2%})")
    print(f"Panel default rate | label=1  : {book.loc[book.loan_default_status == 1, 'defaulted_in_panel'].mean():.2%}, "
          f"label=0: {book.loc[book.loan_default_status == 0, 'defaulted_in_panel'].mean():.2%}")
    print(f"Mean monthly default rate     : {monthly['default_rate'].mean():.3%}")
    print(f"Peak month                    : {monthly.loc[monthly['default_rate'].idxmax(), 'date']:%Y-%m} "
          f"({monthly['default_rate'].max():.3%})")
    print(f"Saved -> {PORTFOLIO_DIR}")


if __name__ == "__main__":
    main()
