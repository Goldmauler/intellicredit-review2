"""
IntelliCredit - apply the mined warning rules to applicants
===========================================================
Re-applies the item definitions and validated rules saved by
notebooks/07_R2_M2_association_rule_mining.ipynb, so the dashboard and the
combined-insights notebook flag applicants exactly as the rules were mined.
"""

import json
import os

import numpy as np
import pandas as pd

from src.config import MODELS_DIR

ITEM_DEFS_PATH = os.path.join(MODELS_DIR, "arm_item_definitions.json")
RULES_PATH = os.path.join(MODELS_DIR, "association_rules.csv")

_OPS = {">=": np.greater_equal, "<=": np.less_equal, "<": np.less, ">": np.greater, "==": np.equal}


def load_rules():
    with open(ITEM_DEFS_PATH) as f:
        item_defs = json.load(f)
    rules = pd.read_csv(RULES_PATH)
    rules["items"] = rules["antecedent_items"].str.split("|")
    return item_defs, rules


def flag_items(df, item_defs):
    """Boolean frame: one column per warning flag, one row per applicant."""
    d = df.copy()
    # ratios stored in borrower_profiles.csv; derived with the generator's formulas when scoring a raw form
    if "income_volatility_index" not in d:
        d["income_volatility_index"] = d["std_monthly_income"] / d["avg_monthly_income"]
    if "debt_to_inflow_ratio" not in d:
        d["debt_to_inflow_ratio"] = (d["avg_bnpl_monthly_amount"] / d["avg_monthly_income"]).clip(0, 2)
    if "savings_buffer_ratio" not in d:
        d["savings_buffer_ratio"] = (d["avg_eom_balance"] / d["avg_monthly_outflow"].where(d["avg_monthly_outflow"] != 0, 1)).clip(0, 5)
    d["loan_to_income"] = d["loan_amount_requested"] / d["avg_monthly_income"]
    d["repayment_ratio"] = np.where(d["previous_micro_loans"] > 0,
                                    d["previous_loans_repaid_on_time"] / d["previous_micro_loans"].clip(lower=1), np.nan)
    out = pd.DataFrame(index=d.index)
    for it in item_defs["items"]:
        values = d[it["column"]]
        if isinstance(it["threshold"], str):
            out[it["item"]] = values == it["threshold"]
        else:
            out[it["item"]] = pd.Series(_OPS[it["op"]](values, it["threshold"]), index=d.index).fillna(False)
    return out.astype(bool)


def rules_hit_count(flags, rules):
    return sum(flags[items].all(axis=1).astype(int) for items in rules["items"])


def triggered_rules(flag_row, rules):
    """Rules whose every flag is present for one applicant (flag_row: boolean Series)."""
    mask = rules["items"].apply(lambda items: bool(all(flag_row[i] for i in items)))
    return rules[mask].sort_values("test_lift", ascending=False)
