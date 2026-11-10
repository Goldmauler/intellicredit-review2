"""
Proof & comparison -> "How the data is made, step by step"
==========================================================
How data_generation.py builds borrower_profiles.csv, why each step exists, and the evidence behind it.

Live from files (nothing in a table or chart is typed in by hand unless it is a published figure with its source):
    data/borrower_profiles.csv                          our distributions, default by age and income, risk-term shares
    models/validation/effect_concordance_detail.csv     correlation with default: real public datasets against ours
Published figures (typed, each with its source and saved copy in research/papers/):
    [75] NCAER / PRI slides  [73] JPMorgan Chase Institute  [74] New York Fed  [76] CFPB  [64] NITI Aayog  [65] Economic Survey

validation_tab.py passes in its own table and tag helpers so the two files do not import each other.
"""
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

NAVY, OCHRE, BRICK, SLATE, GREEN = "#1f3a5f", "#b8862f", "#9b2c2c", "#5a5546", "#3d6b4f"
CFG = {"displayModeBar": False}

# NCAER-PRI workshop slides (research/papers/2023_NCAER_PRI_workshop_food_delivery_platform_workers.pdf), slides 13, 15, 17, 23
NCAER = {
    "age": [29.0, 36.0, 18.0, 10.0, 7.0],                 # 18-24, 25-30, 31-35, 36-40, over 40 (% of 924 workers)
    "edu": [6.0, 56.0, 38.0],                            # below Class X, Class X to below graduate, graduate and above
    "tenure": [round(28.1 / 57.7 * 100, 1), round(14.8 / 57.7 * 100, 1), round(14.8 / 57.7 * 100, 1)],   # active workers: <1y, 1-2y, >2y
    "mean_age": 29.1, "under35": 84.0, "earn": 20744, "plfs_youth": 13000, "plfs_all": 18600,
}
DS = {"gmsc": "Give Me Some Credit", "german": "German credit", "taiwan": "Taiwan cards", "heloc": "FICO HELOC"}


@st.cache_data
def _stats(path, mtime):
    d = pd.read_csv(path)
    y = d["loan_default_status"]
    out = {"n": len(d), "rate": float(y.mean())}
    out["age"] = [float(((d.age >= lo) & (d.age <= hi)).mean() * 100) for lo, hi in [(18, 24), (25, 30), (31, 35), (36, 40), (41, 200)]]
    out["mean_age"] = float(d.age.mean())
    out["under35"] = float((d.age < 35).mean() * 100)
    t = d.gig_tenure_months
    out["tenure"] = [float((t < 12).mean() * 100), float(((t >= 12) & (t <= 24)).mean() * 100), float((t > 24).mean() * 100)]
    e = d.education_level
    out["edu"] = [float((e == "Below 10th").mean() * 100), float(e.isin(["10th Pass", "12th Pass", "Diploma"]).mean() * 100),
                  float(e.isin(["Undergraduate", "Postgraduate"]).mean() * 100)]
    out["income_median"] = float(d.avg_monthly_income.median())
    out["income_u15"] = float((d.avg_monthly_income < 15000).mean() * 100)
    band = pd.cut(d.age, [20, 25, 30, 35, 40, 60], labels=["21-25", "26-30", "31-35", "36-40", "41-60"])
    out["by_age"] = (d.groupby(band, observed=True)["loan_default_status"].mean() * 100).round(2).to_dict()
    q = pd.qcut(d.avg_monthly_income, 4, labels=["Lowest 25%", "Second", "Third", "Highest 25%"])
    out["by_inc"] = (d.groupby(q, observed=True)["loan_default_status"].mean() * 100).round(2).to_dict()
    out["corr_age"] = float(d.age.corr(y))
    out["corr_inc"] = float(d.avg_monthly_income.corr(y))
    rr = np.where(d.previous_micro_loans == 0, 0.5, d.previous_loans_repaid_on_time / d.previous_micro_loans)
    terms = {
        "Utility on-time share (−2.5)": -2.5 * d.utility_reliability_score,
        "Loan amount (+0.000005 per ₹)": 0.000005 * d.loan_amount_requested,
        "Gig tenure (−0.02 per month)": -0.02 * d.gig_tenure_months,
        "Income falling, slope under −500 (+0.8)": 0.8 * (d.income_trend_slope < -500).astype(float),
        "Income volatility (+3.0)": 3.0 * d.income_volatility_index,
        "BNPL debt ÷ income (+2.0)": 2.0 * d.debt_to_inflow_ratio,
        "Savings buffer (−1.5)": -1.5 * np.clip(d.savings_buffer_ratio, 0, 2),
        "Past repayment ratio (−0.5)": -0.5 * np.clip(rr, 0, 1),
        "App logins (−0.03)": -0.03 * d.avg_app_logins,
        "Recharge gap over 35 days (+0.5)": 0.5 * (d.avg_days_between_recharges > 35).astype(float),
    }
    sd = {k: float(v.std()) for k, v in terms.items()}
    tot = sum(sd.values())
    out["terms"] = {k: v / tot * 100 for k, v in sorted(sd.items(), key=lambda kv: -kv[1])}
    return out


def _pct(v):
    return f"{v:.1f}%"


def render(root, explain, style_fig, table, tag):
    path = os.path.join(root, "data", "borrower_profiles.csv")
    if not os.path.exists(path):
        st.warning("data/borrower_profiles.csv was not found.")
        return
    S = _stats(path, os.path.getmtime(path))

    explain("How the 30,000 borrowers are built, step by step, with the reason for each step and the evidence behind it. "
            "Open a step to see what the code does, why, and where it matches real data and where it does not. "
            "The data is synthetic because real borrower files are closed by law (CICRA 2005, DPDP Act 2023). It proves the pipeline works against a known ceiling; "
            "it does not prove real gig workers behave this way.")
    st.markdown(" ".join([tag("Matches", "good"), "checked against a real figure &nbsp;", tag("Assumption", "mid"), "our judgement &nbsp;",
                          tag("Does not match", "bad"), "real data disagrees &nbsp;", tag("Cannot check", "na"), "no public figure"]), unsafe_allow_html=True)

    # =============================================================== step 1
    with st.expander("Step 1 · Draw the people (age, location, education, tenure, income)", expanded=True):
        st.markdown("**What.** 30,000 borrowers, each with demographics drawn independently from the distributions below (seed 42). "
                    "**Why.** A lender judges who the person is and how they earn, so the population must look like a gig workforce. Each variable gets a distribution a "
                    "verified source supports, or an openly labelled assumption where none exists. "
                    "**Limit.** Variables are drawn independently and linked later through income and tenure; real populations have richer dependence.")
        rows = [
            ("Age", "Normal, mean 32, sd 8, clipped 21 to 60", "Gig work skews young",
             f"NCAER survey of 924 delivery workers [75]: mean {NCAER['mean_age']}, {NCAER['under35']:.0f}% under 35. Ours: {S['mean_age']:.1f}, {S['under35']:.0f}%.", tag("Older", "mid")),
            ("Gender", "Male 58%, Female 40%, Other 2%", "Mixed gig population. Not a model input.", "None", tag("Assumption", "mid")),
            ("Location", "Metro 30, Tier-1 25, Tier-2 20, Semi-urban 15, Rural 10 (%). Income multipliers 1.4, 1.2, 1.0, 0.8, 0.6.",
             "Platform work is urban-heavy and city pay is higher", "None verified", tag("Assumption", "mid")),
            ("Education", "Below 10th 5, 10th 10, 12th 25, Diploma 20, Undergraduate 30, Postgraduate 10 (%)", "Platform workers are better educated than average workers",
             f"NCAER [75]: 6 / 56 / 38. Ours: {S['edu'][0]:.1f} / {S['edu'][1]:.1f} / {S['edu'][2]:.1f}.", tag("Matches", "good")),
            ("Platform", "Uniform over 8 kinds", "Gives the model no platform shortcut", "n/a", tag("By design", "na")),
            ("Gig tenure", "Exponential, mean 18 months, 1 to 120", "Many gig workers are new and many leave",
             f"NCAER active workers [75]: {NCAER['tenure'][0]} / {NCAER['tenure'][1]} / {NCAER['tenure'][2]}. Ours: {S['tenure'][0]:.1f} / {S['tenure'][1]:.1f} / {S['tenure'][2]:.1f}.", tag("Matches", "good")),
            ("Bank account", "85% yes", "Assumption", "None", tag("Assumption", "mid")),
            ("Dependents", "0 to 5, mean about 2", "Assumption", "None. The data shows no effect on default.", tag("Assumption", "mid")),
            ("Monthly income", "Uniform ₹8,000 to ₹55,000, times location (0.6 to 1.4), times education (0.7 to 1.25)", "Spans low to middle earners",
             f"Economic Survey [65]: about 40% under ₹15,000, ours {S['income_u15']:.1f}%. NCAER [75] ₹{NCAER['earn']:,} a month, our median ₹{S['income_median']:,.0f}.", tag("Too high", "bad")),
        ]
        table(pd.DataFrame(rows, columns=["Variable", "How it is drawn", "Why", "Real-world check", "Status"]), widths=[13, 24, 18, 30, 15])

        st.markdown("**How well the shapes match the NCAER survey** (one platform, 924 workers, 28 cities, April to May 2022).")
        fig = make_subplots(rows=1, cols=3, subplot_titles=("Education", "Time on platform (active workers)", "Age"), horizontal_spacing=0.09)
        panels = [("edu", ["Below Class X", "Class X to graduate", "Graduate+"]),
                  ("tenure", ["Under 1 year", "1 to 2 years", "Over 2 years"]),
                  ("age", ["18-24 (ours 21-24)", "25-30", "31-35", "36-40", "Over 40"])]
        for c, (k, labels) in enumerate(panels, 1):
            fig.add_trace(go.Bar(x=labels, y=NCAER[k], name="NCAER survey", marker_color=NAVY, showlegend=(c == 1), hovertemplate="%{x}: %{y:.1f}%<extra>NCAER</extra>"), row=1, col=c)
            fig.add_trace(go.Bar(x=labels, y=S[k], name="Our synthetic data", marker_color=OCHRE, showlegend=(c == 1), hovertemplate="%{x}: %{y:.1f}%<extra>Ours</extra>"), row=1, col=c)
        fig.update_layout(barmode="group", legend=dict(orientation="h", y=-0.22), yaxis_title="% of workers")
        fig.update_xaxes(tickangle=-30, tickfont=dict(size=10))
        st.plotly_chart(style_fig(fig, 340), width="stretch", config=CFG)
        st.caption("Education and tenure match closely. Ours is older: the NCAER delivery workers are 84% under 35. The NCAER source is one platform, so it may not represent all gig work.")

    # =============================================================== step 2
    with st.expander("Step 2 · Simulate 12 months of money (monthly_transactions.csv, 360,000 rows)"):
        st.markdown("**What.** For each borrower and month: income, utility bill and whether it was paid on time, recharge, buy-now-pay-later (BNPL), spending, month-end balance, app logins and UPI payments. "
                    "**Why.** A thin-file borrower has no bureau history, so the signal must come from how money moves: how steady income is, whether small bills are paid, whether savings absorb a bad month. "
                    "These are the alternative-data signals studied by Berg et al. (2020) and Björkegren and Grissen (2020). **Limit.** Directions are plausible or sourced; sizes are our judgement.")
        rows = [
            ("Monthly income", "Base income plus Normal noise with spread base × max(0.05, 0.35 − 0.003 × tenure)",
             "Newer gig workers have shakier earnings. The JPMorgan Chase Institute [73] found income swings are most marked among the young and the lowest earners."),
            ("Bill paid on time", "Probability 0.6 + 0.005 × tenure + 0.3 × (this month's income ÷ base − 1), kept 0.3 to 0.98",
             "Steadier, longer-serving workers pay more reliably, and a bad income month makes a late bill more likely."),
            ("Days between recharges", "Normal, mean 28, sd 8, kept 10 to 60",
             "Prepaid plans ran on 28-day cycles until 30-day plans were required in January 2022. A plausibility check, not a validation."),
            ("BNPL use", "Probability 0.3 + 0.02 × (35 − age) − 0.002 × tenure, kept 0.05 to 0.70",
             "BNPL borrowers skew young (CFPB slide 12 [76], US data). This is the only way age enters the data."),
            ("Spending", "Normal, 70% of income, sd 15 points", "Assumption: most earners spend most of what they make."),
            ("App logins and UPI", "Poisson: 15 + 0.2 × tenure logins, 12 × location multiplier UPI payments a month",
             "Engagement grows with time on the platform, and UPI use is higher in cities."),
        ]
        table(pd.DataFrame(rows, columns=["Quantity", "Rule in the code", "Reason"]), widths=[16, 40, 44])

    # =============================================================== step 3
    with st.expander("Step 3 · Summarise each person (12 rows become one profile)"):
        st.markdown("**What.** Each borrower's 12 months are collapsed into averages, spreads and five engineered ratios. "
                    "**Why.** A scorecard needs one row per applicant, and each ratio compares a behaviour with the person's own means, which is how an underwriter reasons.")
        rows = [
            ("income_volatility_index", "Standard deviation ÷ mean of monthly income", "Coefficient of variation: income instability in one number"),
            ("utility_reliability_score", "Bills paid on time ÷ 12", "Small, regular bills are a cheap test of reliability"),
            ("debt_to_inflow_ratio", "Average monthly BNPL amount ÷ average income, kept 0 to 2", "How much existing credit competes for the same income"),
            ("savings_buffer_ratio", "Average month-end balance ÷ average spending, kept 0 to 5", "Months of runway if income stops"),
            ("income_trend_slope", "Straight-line slope of income over 12 months, ₹ per month", "Falling earnings are an early warning"),
        ]
        table(pd.DataFrame(rows, columns=["Feature", "Definition", "Why it exists"]), widths=[24, 40, 36])

    # =============================================================== step 4
    with st.expander("Step 4 · Decide who defaults (the label every model must predict)", expanded=True):
        st.markdown("**What.** Each borrower gets a default probability from a logistic formula, and the default flag is a random draw from that probability. "
                    "**Why a formula.** No labelled gig-worker repayment data is open to us. A known formula gives a ground truth, so we can say exactly how close a model gets: "
                    "the best possible AUC is 0.781 and our model reaches 0.741, about 85% of what is achievable. **Why logistic.** It is how credit scorecards turn a weighted sum of risk factors into a probability. "
                    "**Limit.** The signs are supported by real data where it exists; the sizes are hand-set. A model trained on this label re-discovers our formula, so this tests the pipeline, not the world.")
        st.code("z = -1.2 + 3.0*income_volatility - 2.5*utility_on_time_share + 2.0*bnpl_debt_to_income\n"
                "    - 1.5*min(savings_buffer, 2) - 0.02*tenure_months + 0.5*[recharge_gap > 35 days]\n"
                "    - 0.03*app_logins + 0.8*[income_slope < -500] + 0.000005*loan_amount - 0.5*past_repayment_ratio + Normal(0, 0.5)\n"
                "P(default) = 1 / (1 + e^-z);   default = 1 if a uniform random number < P", language="text")
        st.caption(f"The intercept −1.2 was set so the overall rate lands near 8% (actual {S['rate'] * 100:.2f}%). The Normal(0, 0.5) term stands for everything we cannot observe; it caps the best possible AUC at 0.781.")
        st.markdown("**How much each term moves risk** (its standard deviation as a share of the total, computed from the generated file)")
        tt = S["terms"]
        fig = go.Figure(go.Bar(y=list(tt.keys())[::-1], x=list(tt.values())[::-1], orientation="h", marker_color=NAVY,
                               text=[f"{v:.1f}%" for v in list(tt.values())[::-1]], textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
        fig.update_xaxes(range=[0, max(tt.values()) * 1.2], title_text="Share of the combined spread of the terms (%)")
        st.plotly_chart(style_fig(fig, 380), width="stretch", config=CFG)
        rows = [
            ("Utility on-time share", "FICO HELOC −0.28, Taiwan cards −0.14", tag("Same direction", "good")),
            ("Loan amount", "German credit +0.09 (amount), +0.21 (duration). Our flaw: it is charged per rupee, not per unit of income.", tag("Direction yes, form no", "mid")),
            ("Gig tenure", "HELOC months on file −0.26 and −0.22", tag("Same direction", "good")),
            ("BNPL debt ÷ income", "All five real checks positive, +0.02 to +0.35. Weak in our data.", tag("Supported", "good")),
            ("Savings buffer", "Taiwan repayment proxy −0.14", tag("Same direction", "good")),
            ("Income volatility, falling income", "No public credit dataset records income volatility", tag("Cannot check", "na")),
            ("App logins, recharge gap, past repayment", "Not tested. Logins follow Berg et al. (2020) on digital footprints.", tag("Assumption", "mid")),
        ]
        table(pd.DataFrame(rows, columns=["Term", "Real-data check (our test on public datasets, Spearman correlation with default)", "Status"]), widths=[26, 56, 18])

    # =============================================================== step 5
    with st.expander("Step 5 · Link to the economy (loan book and monthly default series)"):
        st.markdown("**What.** The same 30,000 borrowers are given loans from mid-2018 to 2024 and followed month by month. Each borrower's monthly default hazard moves with a stress index built from "
                    "real macro series (FRED, World Bank) through a logit link, plus lockdown and second-wave flags. "
                    "**Why.** Forecasting needs a monthly default series and no real portfolio is available; linking hazards to real macro data gives the forecast something real to learn from "
                    "(the CreditPortfolioView idea, Wilson 1997). "
                    "**Limit.** The model that later finds the macro link is recovering the link we built in. The check on real data is notebook 09: signs agree for all four drivers in a 140-country panel, "
                    "and India's NPL series is explained better once RBI regime changes are added. The 2020 spike has no real counterpart (real bank bad-loan ratios fell).")

    # =============================================================== young / low income
    st.markdown("### Do young and low-income borrowers really struggle?")
    st.caption("Two separate answers: what is true in the world, and what is true in our data.")
    st.markdown("**In the world**")
    rows = [
        ("Young", "Credit-card delinquency (90+ days) is generally highest among the youngest borrowers and falls with each older group; the gap changes over time.", "New York Fed, Liberty Street Economics, 16 Nov 2018 [74]"),
        ("Young", "70% of people aged 18 to 24 had a month-to-month income change above 30%, against 55% nationally.", "JPMorgan Chase Institute, 2016 [73]"),
        ("Young", "On public credit data, age correlates with default at −0.117 (Give Me Some Credit), −0.112 (German credit) and +0.005 (Taiwan cards): younger borrowers default somewhat more in 2 of 3.", "Our test, scripts/run_validation_suite.py"),
        ("Low income", "74% of the bottom income quintile had a month-to-month income change above 30%.", "JPMorgan Chase Institute [73]"),
        ("Low income", f"Urban male workers aged 18 to 29 earn about ₹{NCAER['plfs_youth']:,} a month against ₹{NCAER['plfs_all']:,} for all aged 18+ (PLFS 2021-22).", "NCAER slides quoting PLFS [75]"),
        ("Low income", "On public credit data, income correlates with default at −0.067 (Give Me Some Credit) and credit limit at −0.170 (Taiwan cards): higher income, fewer defaults.", "Our test"),
    ]
    table(pd.DataFrame(rows, columns=["Group", "Evidence", "Source"]), widths=[10, 60, 30])
    st.markdown("These are **correlates of risk, not causes**. Young and low-income people tend to have shorter histories, less steady pay and thinner savings, which are the things the model measures. "
                "The effects on default are real but modest (a correlation of about 0.1).")

    st.markdown("**In our data**")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        ba = S["by_age"]
        fig = go.Figure(go.Bar(x=list(ba.keys()), y=list(ba.values()), marker_color=NAVY, text=[f"{v:.1f}%" for v in ba.values()], textposition="outside", cliponaxis=False))
        fig.add_hline(y=S["rate"] * 100, line_dash="dash", line_color=SLATE)
        fig.update_layout(title=f"Default rate by age band (dashed line = average {S['rate'] * 100:.2f}%)", yaxis_title="% defaulted")
        fig.update_yaxes(range=[0, 13])
        st.plotly_chart(style_fig(fig, 330), width="stretch", config=CFG)
        st.caption(f"Age has no direct term in the formula; it acts only through BNPL use. Correlation of age with default: {S['corr_age']:+.2f}.")
    with c2:
        bi = S["by_inc"]
        fig = go.Figure(go.Bar(x=list(bi.keys()), y=list(bi.values()), marker_color=BRICK, text=[f"{v:.1f}%" for v in bi.values()], textposition="outside", cliponaxis=False))
        fig.add_hline(y=S["rate"] * 100, line_dash="dash", line_color=SLATE)
        fig.update_layout(title=f"Default rate by income quartile (dashed line = average {S['rate'] * 100:.2f}%)", yaxis_title="% defaulted")
        fig.update_yaxes(range=[0, 13])
        st.plotly_chart(style_fig(fig, 330), width="stretch", config=CFG)
        st.caption(f"The reverse of real data: loans are sized to income and charged per rupee, so default rises with income. Correlation of income with default: {S['corr_inc']:+.2f}.")

    det_path = os.path.join(root, "models", "validation", "effect_concordance_detail.csv")
    if os.path.exists(det_path):
        det = pd.read_csv(det_path)
        pts = []
        for concept, label in [("Younger borrowers default more", "Age"), ("Higher income / credit capacity lowers default", "Income")]:
            g = det[det.concept == concept]
            for _, r in g.iterrows():
                pts.append((f"{label} · {DS.get(r.real_dataset, r.real_dataset)}", float(r.real_rho), "Real datasets"))
            if len(g):
                pts.append((f"{label} · Ours (synthetic)", float(g.synthetic_rho.iloc[0]), "Our synthetic data"))
        fig = go.Figure()
        for grp, colour in [("Real datasets", NAVY), ("Our synthetic data", BRICK)]:
            sel = [p for p in pts if p[2] == grp]
            fig.add_trace(go.Scatter(x=[p[1] for p in sel], y=[p[0] for p in sel], mode="markers+text", name=grp, marker=dict(size=13, color=colour),
                                     text=[f"{p[1]:+.3f}" for p in sel], textposition=["middle left" if p[1] < 0 else "middle right" for p in sel],
                                     hovertemplate="%{y}: %{x:+.3f}<extra></extra>"))
        fig.add_vline(x=0, line_color="#1f1f1f")
        fig.update_xaxes(range=[-0.25, 0.17], title_text="Correlation with default   (left: a higher value goes with fewer defaults, right: more defaults)")
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(title="Age and income against default: real datasets versus ours", legend=dict(orientation="h", y=-0.28))
        st.plotly_chart(style_fig(fig, 330), width="stretch", config=CFG)
        st.caption("Spearman correlation, from models/validation/effect_concordance_detail.csv. Taiwan uses credit limit as the income proxy.")

    st.markdown('<div class="callout ok"><b>What is safe to say.</b> Income instability and a short track record are the risk drivers in this data. Being young or low-income is a documented real-world correlate of them, '
                'supported by the New York Fed, the JPMorgan Chase Institute and our own tests on three public datasets. The real effects are modest.</div>', unsafe_allow_html=True)
    st.markdown('<div class="callout"><b>What not to say.</b> Do not say the synthetic data shows low-income people defaulting more: it shows the opposite, and the project documents it as a known gap. '
                'Do not call the BNPL-driven age effect a finding: it is an assumption backed by one US regulator slide.</div>', unsafe_allow_html=True)

    # =============================================================== proofs
    st.markdown("### The proofs behind these numbers")
    st.caption("Each source was opened and the figure read in a saved copy kept in the repository, except where marked. Numbers in brackets match research/REFERENCES_ACM.md.")
    link = lambda url, t: f'<a href="{url}" target="_blank">{t}</a>'
    rows = [
        ("NCAER and Policy Research Institute workshop slides, 2023 [75]", "Age, education and tenure mix; earnings against the national labour survey",
         "924 workers, 28 cities. Age 29.1, 84% under 35. Education 6 / 56 / 38. Earnings ₹20,744 against ₹22,494 (PLFS peers).", tag("Read in saved copy", "good"),
         link("https://www.mof.go.jp/pri/research/conference/indiaws/indiaws2023_02_02.pdf", "PDF") + "<br><code>research/papers/2023_NCAER_…pdf</code>"),
        ("JPMorgan Chase Institute, Farrell and Greig, 2016 [73]", "Income volatility is highest for the young and the lowest earners",
         "70% (aged 18-24), 74% (bottom quintile), 55% (national) had a monthly income change above 30%.", tag("Read in saved copy", "good"),
         link("https://www.jpmorganchase.com/institute/all-topics/careers-and-skills/report-paychecks-paydays-and-the-online-platform-economy", "Report") + "<br><code>research/papers/2016_JPMorganChaseInstitute_…pdf</code>"),
        ("New York Fed, Haughwout et al., 2018 [74]", "The youngest borrowers have the highest card delinquency",
         "Text: delinquency rates are generally highest among the youngest borrowers. No rate is given.", tag("Read in saved copy", "good"),
         link("https://libertystreeteconomics.newyorkfed.org/2018/11/just-released-a-look-at-borrowing-repayment-and-bankruptcy-rates-by-age.html", "Post") + "<br><code>research/papers/2018_NYFed_…html</code>"),
        ("CFPB, BNPL market trends, 2022 [76]", "Basis for the one age effect in our data", "Slide 12: BNPL borrowers skew young. US data.", tag("Read in saved copy", "good"),
         link("https://files.consumerfinance.gov/f/documents/cfpb_buy-now-pay-later-market-trends-and-consumer-impacts_presentation_2022-11.pdf", "PDF") + "<br><code>research/papers/2022_CFPB_BNPL_…pdf</code>"),
        ("NITI Aayog, 2022 [64]", "Size of the gig workforce in 2020-21", "7.7 million gig workers, projected 23.5 million by 2029-30.", tag("Read in saved copy", "good"),
         link("https://www.niti.gov.in/sites/default/files/2022-06/Policy_Brief_India's_Booming_Gig_and_Platform_Economy_27062022.pdf", "PDF") + "<br><code>research/papers/2022_NITI_Aayog_…pdf</code>"),
        ("Economic Survey 2025-26 [65]", "Pay of the gig workforce", "About 12 million gig workers in FY25, about 40% under ₹15,000 a month.", tag("News report only", "mid"),
         link("https://www.businesstoday.in/amp/union-budget/story/india-has-12-million-gig-workers-2-of-total-workforce-economic-survey-513429-2026-01-29", "Business Today") + "<br>The chapter itself was not found"),
        ("Four public credit datasets", "Real-data tests of the pipeline and of each risk-driver direction",
         "5-fold AUC: German 0.793, Taiwan 0.786, Give Me Some Credit 0.866, HELOC 0.794. Downloaded on demand, not stored.", tag("Re-run here", "good"),
         "<code>python scripts/run_validation_suite.py</code>"),
    ]
    table(pd.DataFrame(rows, columns=["Source", "What it proves for us", "Key figure", "Evidence level", "Open it"]), widths=[22, 20, 30, 12, 16])
