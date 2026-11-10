"""
IntelliCredit - synthetic-data fidelity anchors
================================================
Recomputes every synthetic statistic that is compared with a published real-world figure and writes
models/validation/anchors.csv. The real-world numbers are typed in once here, each with its source link and
the date it was checked; the synthetic numbers come from the data files, never from memory.

    python scripts/build_fidelity_anchors.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "models", "validation")
os.makedirs(OUT, exist_ok=True)

b = pd.read_csv(os.path.join(ROOT, "data", "borrower_profiles.csv"))
pm = pd.read_csv(os.path.join(ROOT, "data", "portfolio", "portfolio_monthly.csv"), parse_dates=["date"])
real = json.load(open(os.path.join(ROOT, "models", "real", "real_data_results.json"), encoding="utf-8"))
bench = pd.read_csv(os.path.join(OUT, "realbench_results.csv")) if os.path.exists(os.path.join(OUT, "realbench_results.csv")) else None
card = json.load(open(os.path.join(ROOT, "models", "underwriting_model_card.json"), encoding="utf-8"))

y19 = pm.loc[pm.date.dt.year == 2019, "default_rate"].mean()
y20 = pm.loc[pm.date.dt.year == 2020, "default_rate"].mean()
real_lo_auc = real_hi_auc = None
if bench is not None:
    best = bench[(bench.dataset != "intellicredit") & (bench.model.str.startswith("LightGBM"))]
    real_lo_auc, real_hi_auc = best.auc.min(), best.auc.max()

W = {
    "mfin": "https://mfinindia.org/assets/upload_image/publications/MicrometerSynopsis/Micrometer%20Q1_FY_25-26_Synopsis.pdf",
    "sadhan": "https://www.sa-dhan.net/BMR/2025/",
    "trai": "https://www.dtnext.in/business/2022/01/28/now-prepaid-mobile-recharge-plans-to-offer-30day-validity",
    "survey": "research/BENCHMARK_TABLE.md (Economic Survey 2025-26, ref [65])",
    "rbi": "research/BENCHMARK_TABLE.md (RBI Trend & Progress 2020-21, ref [63])",
}

rows = [
    dict(check="Default rate",
         synthetic=f"{b.loan_default_status.mean():.2%} of borrowers reach 90+ days late",
         real="Microfinance portfolio-at-risk (31-180 days): 2.2% (Mar 2024), 6.3% (Mar 2025), 1.6% (Jun 2026)",
         source="MFIN Micrometer, via research/BENCHMARK_TABLE.md [67]", verdict="Roughly matches",
         note="Higher on purpose (thin-file segment outside MFI books), and the same order of magnitude as the 2025 stress peak."),
    dict(check="Share of workers earning under Rs 15,000 a month",
         synthetic=f"{(b.avg_monthly_income < 15000).mean():.1%}",
         real="About 40% of gig workers", source="Economic Survey 2025-26 (ref [65])", verdict="Does not match",
         note="The synthetic population is richer than reality. See the re-weighting test, which shows the model ranking is not sensitive to this."),
    dict(check="Typical loan size",
         synthetic=f"Median Rs {b.loan_amount_requested.median():,.0f}",
         real="Microfinance average ticket Rs 50,131 (FY25) and Rs 54,956 (Q1 FY26)", source=W["mfin"], verdict="Does not match",
         note=f"About {b.loan_amount_requested.median() / 50131:.1f} times larger. The generator sets loans at 1.5-6 months of income, capped at Rs 3 lakh."),
    dict(check="Interest rate",
         synthetic=f"Mean {b.interest_rate_annual.mean():.1f}% (5th-95th percentile {b.interest_rate_annual.quantile(.05):.0f}-{b.interest_rate_annual.quantile(.95):.0f}%)",
         real="NBFC-MFI average effective yield 19.1%; member-reported MFIN rates in the mid-to-high 20s", source=W["sadhan"],
         verdict="Roughly matches", note="Slightly below the reported range; fintech small-ticket loans are often priced higher still."),
    dict(check="Mobile recharge cycle",
         synthetic=f"Median {b.avg_days_between_recharges.median():.1f} days between recharges",
         real="Most prepaid plans ran on 28-day cycles until TRAI required 30-day plans in January 2022", source=W["trai"],
         verdict="Roughly matches", note="Consistent with 28-30 day plans. Weak evidence: it is a plausibility check, not a distribution test."),
    dict(check="How hard is it to predict default?",
         synthetic=f"Best possible AUC {card['oracle_auc_true_pd']:.3f}; model reaches {card['test_auc']:.3f}",
         real=(f"Same model recipe on 4 real public credit datasets: AUC {real_lo_auc:.2f} to {real_hi_auc:.2f}" if real_lo_auc else "see real-data benchmark")
         + "; Indian fintech lender, footprint data only: 0.60",
         source="Computed here (scripts/run_validation_suite.py); Agarwal et al. 2020 (ref [4])", verdict="Roughly matches",
         note="The synthetic data is as hard as real bureau-style data, but much easier than the Indian fintech case. Real Indian data would likely score lower."),
    dict(check="Default spike in 2020 (COVID)",
         synthetic=f"2020 default rate is {y20 / y19:.1f} times the 2019 rate",
         real="Bank bad-loan ratio fell about 11% over the same period (moratorium and write-offs hid the stress): x0.89", source=W["rbi"] + "; IMF series in notebook 09",
         verdict="Does not match", note="The spike is an assumption of the simulation, not something observed in Indian data."),
    dict(check="Economy and defaults move together the right way",
         synthetic="Unemployment, inflation and interest rate raise defaults; GDP growth lowers them",
         real=f"140-country panel: unemployment change {real['panel']['coefficients']['d_unemp']['β (no rate)']:+.2f} (p={real['panel']['coefficients']['d_unemp']['p (no rate)']:.3f}), GDP growth {real['panel']['coefficients']['gdp_growth']['β (no rate)']:+.2f} (p={real['panel']['coefficients']['gdp_growth']['p (no rate)']:.3f})",
         source="World Bank panel, notebook 09", verdict="Matches", note="Signs agree for all four drivers. Only unemployment, GDP and inflation are statistically significant in the real panel."),
    dict(check="Income volatility, bill punctuality, UPI activity",
         synthetic=f"Income variation {b.income_volatility_index.mean():.2f}; {b.utility_reliability_score.mean():.0%} of bills on time; median {b.avg_upi_transactions.median():.0f} UPI payments a month",
         real="No verified public figure for gig-worker borrowers", source="searched, none found", verdict="Cannot verify",
         note="Stated openly rather than claimed. These need a real lender's data."),
]
out = pd.DataFrame(rows)
out.to_csv(os.path.join(OUT, "anchors.csv"), index=False)
print(out[["check", "synthetic", "verdict"]].to_string())
