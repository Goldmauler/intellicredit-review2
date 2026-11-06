"""
IntelliCredit - Counterfactual "path to approval" (actionable recourse)
=======================================================================
A declined applicant should be told *what would have to change* for the loan to be
approved, not only *why* it was declined. This module searches a small grid of
ACTIONABLE changes and returns the cheapest ones that turn the decision around.

Ideas from the literature:
  * Counterfactual explanations - the smallest change to the inputs that flips the
    decision (Wachter, Mittelstadt & Russell 2017).
  * Actionable recourse - only change what the applicant can actually change, never
    immutable attributes such as age, location or education (Ustun, Spangher & Liu 2019).
  * Diverse counterfactuals - offer several different options (Mothilal, Sharma & Tan 2020, DiCE).

Levers (and why they are allowed):
  * loan amount     - the applicant can ask for less, today
  * loan tenure     - a longer tenure spreads repayment (changes the break-even PD too)
  * BNPL spending   - can be cut, takes about a month to show in cash flow
  * utility on-time - behavioural, needs several months of history (highest effort)
  * month-end balance - building savings, takes months
The interest rate is deliberately NOT a lever: pricing a risky applicant out of a decline
is predatory under RBI's Fair Practices Code for lenders.
"""

import itertools

import numpy as np
import pandas as pd

from src.underwriting_model import break_even_pd, build_features

AMOUNT_FACTORS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
TENURES = [3, 6, 9, 12, 18, 24]
BNPL_FACTORS = [1.0, 0.5, 0.0]
ON_TIME_GAINS = [0, 2, 4]
BALANCE_FACTORS = [1.0, 1.5, 2.0]

# effort weights: how hard each change is for the applicant (higher = harder / slower)
EFFORT = {"amount": 1.0, "tenure": 0.4, "bnpl": 0.8, "on_time": 1.5, "balance": 1.2}


def _effort(base, row):
    e = EFFORT["amount"] * (1 - row["loan_amount_requested"] / base["loan_amount_requested"])
    e += EFFORT["tenure"] * abs(row["loan_tenure_months"] - base["loan_tenure_months"]) / 12
    if base["avg_bnpl_monthly_amount"] > 0:
        e += EFFORT["bnpl"] * (1 - row["avg_bnpl_monthly_amount"] / base["avg_bnpl_monthly_amount"])
    e += EFFORT["on_time"] * (row["utility_on_time_count"] - base["utility_on_time_count"]) / 4
    e += EFFORT["balance"] * (row["avg_eom_balance"] / max(base["avg_eom_balance"], 1.0) - 1)
    return e


def path_to_approval(model, raw_row: pd.Series, multiplier=1.0, max_rules_ok=None, top_k=3):
    """Return up to `top_k` cheapest actionable changes that bring point-in-time PD below break-even.

    raw_row: one applicant in the raw input schema used by the dashboard form.
    Returns a DataFrame (one row per option) with the changes, new PD, new break-even PD and effort.
    """
    base = raw_row.copy()
    grid = []
    for a, t, b, o, s in itertools.product(AMOUNT_FACTORS, TENURES, BNPL_FACTORS, ON_TIME_GAINS, BALANCE_FACTORS):
        r = base.copy()
        r["loan_amount_requested"] = round(base["loan_amount_requested"] * a, -2)
        r["loan_tenure_months"] = t
        r["avg_bnpl_monthly_amount"] = base["avg_bnpl_monthly_amount"] * b
        r["utility_on_time_count"] = min(12, int(base["utility_on_time_count"]) + o)
        r["avg_eom_balance"] = base["avg_eom_balance"] * s
        grid.append(r)
    cand = pd.DataFrame(grid).drop_duplicates()
    X = build_features(cand)
    pd_pit = np.minimum(model.predict_proba(X)[:, 1] * multiplier, 0.99)
    be = np.asarray(break_even_pd(cand["interest_rate_annual"].values, cand["loan_tenure_months"].values), dtype=float)
    cand = cand.assign(new_pd=pd_pit, new_break_even=be, margin=be - pd_pit)
    ok = cand[cand["margin"] > 0].copy()
    if ok.empty:
        return pd.DataFrame()
    ok["effort"] = ok.apply(lambda r: _effort(base, r), axis=1)
    ok = ok.sort_values(["effort", "new_pd"])

    options, seen = [], set()
    for _, r in ok.iterrows():
        changes = []
        if r["loan_amount_requested"] != base["loan_amount_requested"]:
            changes.append(f"borrow INR {r['loan_amount_requested']:,.0f} instead of INR {base['loan_amount_requested']:,.0f}")
        if r["loan_tenure_months"] != base["loan_tenure_months"]:
            changes.append(f"tenure {int(r['loan_tenure_months'])} months instead of {int(base['loan_tenure_months'])}")
        if r["avg_bnpl_monthly_amount"] < base["avg_bnpl_monthly_amount"]:
            changes.append(f"cut BNPL spend to INR {r['avg_bnpl_monthly_amount']:,.0f}/month")
        if r["utility_on_time_count"] > base["utility_on_time_count"]:
            changes.append(f"pay {int(r['utility_on_time_count'])} of 12 utility bills on time (needs a few months)")
        if r["avg_eom_balance"] > base["avg_eom_balance"]:
            changes.append(f"keep a month-end balance of about INR {r['avg_eom_balance']:,.0f}")
        key = frozenset(c.split(" ")[0] for c in changes)
        # diverse counterfactuals: skip an option whose levers include an earlier, cheaper option's levers
        if not changes or any(prev <= key for prev in seen):
            continue
        seen.add(key)
        options.append({"option": len(options) + 1, "what to change": "; ".join(changes),
                        "new PD": r["new_pd"], "break-even PD": r["new_break_even"], "effort score": round(r["effort"], 2)})
        if len(options) == top_k:
            break
    return pd.DataFrame(options)
