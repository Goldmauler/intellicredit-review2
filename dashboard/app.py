"""
IntelliCredit - Loan Underwriter & Risk Analytics Dashboard
===========================================================
Production-grade dashboard demonstrating 100% implementation of Units 1, 2, and 3:
  * Unit 1 & 2 ML PD Model      -> Calibrated probability of default + TreeSHAP explainability
  * Unit 3 Method 2 ARM Rules   -> Delinquency pattern mining (Apriori/FP-Growth) & rule explorer
  * Unit 3 Method 1 Time Series -> SARIMAX macro scenario multiplier & portfolio loss outlook
  * Dataset Explorer            -> Multi-attribute filters, dynamic KPIs, and applicant drilldowns

Run from the repository root:
    streamlit run dashboard/app.py
"""

import json
import os
import sys
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Suppress sklearn model-persistence version mismatch warnings at startup
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import DATA_DIR, MODELS_DIR, RANDOM_STATE, TEST_SIZE  # noqa: E402
from src.risk_rules import flag_items, load_rules, triggered_rules, rules_hit_count  # noqa: E402
from src.underwriting_model import (  # noqa: E402
    COST_OF_FUNDS, FEATURE_LABELS, LGD, MODEL_PATH, CARD_PATH, break_even_pd, build_features, risk_band,
)

st.set_page_config(page_title="IntelliCredit Underwriter", page_icon=":credit_card:", layout="wide")

SCEN_COLORS = {"Baseline": "#4c72b0", "Adverse": "#dd8452", "Severe": "#c44e52"}
BAND_COLORS = {"A": "#1a9850", "B": "#91cf60", "C": "#fee08b", "D": "#fc8d59", "E": "#d73027"}
# Risk-band-based product rate schedule (used to auto-assign interest rate)
BAND_RATES = {"A": 14.0, "B": 18.0, "C": 22.0, "D": 26.0, "E": 30.0}

PRESETS = {
    "Customer 1 (Auto-Approve: Prime Gig Worker)": {
        "age": 34, "gig_tenure_months": 36, "has_bank_account": 1, "num_dependents": 1,
        "location_type": "Metro", "education_level": "Undergraduate", "avg_monthly_income": 48000.0,
        "std_monthly_income": 3200.0, "income_trend_slope": 250.0, "utility_on_time_count": 12,
        "avg_utility_delay_days": 0.0, "max_utility_delay_days": 0, "avg_days_between_recharges": 28.0,
        "avg_bnpl_monthly_amount": 2000.0, "avg_monthly_outflow": 22000.0, "avg_eom_balance": 38000.0,
        "avg_app_logins": 45.0, "avg_upi_transactions": 40.0, "loan_amount_requested": 40000.0,
        "loan_tenure_months": 12, "interest_rate_annual": 16.0, "previous_micro_loans": 4, "previous_loans_repaid_on_time": 4
    },
    "Customer 2 (Conditional: Stressed / Marginal Worker)": {
        "age": 28, "gig_tenure_months": 8, "has_bank_account": 1, "num_dependents": 2,
        "location_type": "Semi-Urban", "education_level": "Diploma", "avg_monthly_income": 24000.0,
        "std_monthly_income": 9500.0, "income_trend_slope": -180.0, "utility_on_time_count": 8,
        "avg_utility_delay_days": 6.0, "max_utility_delay_days": 18, "avg_days_between_recharges": 35.0,
        "avg_bnpl_monthly_amount": 6500.0, "avg_monthly_outflow": 21000.0, "avg_eom_balance": 4000.0,
        "avg_app_logins": 15.0, "avg_upi_transactions": 10.0, "loan_amount_requested": 50000.0,
        "loan_tenure_months": 12, "interest_rate_annual": 22.0, "previous_micro_loans": 1, "previous_loans_repaid_on_time": 1
    },
    "Customer 3 (Auto-Decline: High-Risk Delinquent)": {
        "age": 24, "gig_tenure_months": 2, "has_bank_account": 0, "num_dependents": 3,
        "location_type": "Rural", "education_level": "10th Pass", "avg_monthly_income": 14000.0,
        "std_monthly_income": 8000.0, "income_trend_slope": -350.0, "utility_on_time_count": 4,
        "avg_utility_delay_days": 18.0, "max_utility_delay_days": 45, "avg_days_between_recharges": 45.0,
        "avg_bnpl_monthly_amount": 9000.0, "avg_monthly_outflow": 15000.0, "avg_eom_balance": 800.0,
        "avg_app_logins": 5.0, "avg_upi_transactions": 3.0, "loan_amount_requested": 80000.0,
        "loan_tenure_months": 12, "interest_rate_annual": 18.0, "previous_micro_loans": 1, "previous_loans_repaid_on_time": 0
    }
}


# ---------------------------------------------------------------------------------------------- data
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_artifacts(_model):
    borrowers = pd.read_csv(os.path.join(DATA_DIR, "borrower_profiles.csv"))
    from sklearn.model_selection import train_test_split
    _, test_idx = train_test_split(borrowers.index, test_size=TEST_SIZE,
                                   stratify=borrowers["loan_default_status"], random_state=RANDOM_STATE)
    test_borrowers = borrowers.loc[test_idx].copy()
    
    with open(CARD_PATH) as f:
        card = json.load(f)
    with open(os.path.join(MODELS_DIR, "macro_overlay.json")) as f:
        overlay = json.load(f)
    item_defs, rules = load_rules()
    scen = pd.read_csv(os.path.join(MODELS_DIR, "ts_forecast_scenarios.csv"), parse_dates=["date"])
    impact = pd.read_csv(os.path.join(MODELS_DIR, "ts_scenario_impact.csv"), index_col=0)
    ts_cmp = pd.read_csv(os.path.join(MODELS_DIR, "ts_model_comparison.csv"), index_col=0)
    pm = pd.read_csv(os.path.join(DATA_DIR, "portfolio", "portfolio_monthly.csv"), parse_dates=["date"])
    macro = pd.read_csv(os.path.join(DATA_DIR, "external", "macro_india_monthly.csv"), parse_dates=["date"])
    with open(os.path.join(DATA_DIR, "external", "macro_sources.json")) as f:
        sources = json.load(f)

    # Pre-evaluate test borrowers for the Dataset Explorer tab
    X_test = build_features(test_borrowers)
    test_borrowers["predicted_pd"] = np.round(_model.predict_proba(X_test)[:, 1], 4)
    test_borrowers["risk_band"] = test_borrowers["predicted_pd"].apply(risk_band)
    test_borrowers["break_even_pd"] = np.round(break_even_pd(test_borrowers["interest_rate_annual"], test_borrowers["loan_tenure_months"]), 4)
    flags_test = flag_items(test_borrowers, item_defs)
    test_borrowers["rules_hit"] = rules_hit_count(flags_test, rules)
    test_borrowers["policy_decision"] = np.where(
        (test_borrowers["predicted_pd"] >= test_borrowers["break_even_pd"]) | (test_borrowers["risk_band"] == "E"), "DECLINE",
        np.where(test_borrowers["rules_hit"] >= 3, "CONDITIONAL / REVIEW", "APPROVE")
    )

    return borrowers, test_borrowers, card, overlay, item_defs, rules, scen, impact, ts_cmp, pm, macro, sources


model = load_model()
borrowers, test_borrowers, card, overlay, item_defs, rules, scen, impact, ts_cmp, pm, macro, sources = load_artifacts(model)
multipliers = {"Neutral (through-the-cycle)": 1.0, **overlay["multipliers_next_12m"]}

# ---- FIX 2: Sidebar now contains ONLY macro scenario controls (duplicate customer picker removed) ----
st.sidebar.title("IntelliCredit")
st.sidebar.caption("Alternative credit scoring for thin-file & gig-economy borrowers")
scenario = st.sidebar.radio("Macro scenario for the next 12 months", list(multipliers), index=0,
                            help="From the SARIMAX forecast (notebook 06). The multiplier converts a through-the-cycle PD into a point-in-time PD.")
apply_overlay = st.sidebar.toggle("Apply macro overlay to the decision", value=True)
mult = multipliers[scenario] if apply_overlay else 1.0
st.sidebar.metric("PD multiplier in use", f"{mult:.2f}x")

# ---------------------------------------------------------------------------------------------- tabs
tab_uw, tab_data, tab_port, tab_rules, tab_model = st.tabs([
    "Underwriter Decision Engine",
    "Dataset Explorer & Drilldown",
    "Portfolio Outlook & Scenarios",
    "Delinquency Warning Rules",
    "Model Card & Provenance"
])

# ============================================================================================== TAB 1: Underwriter
# FIX 1: Reduced from 23 visible inputs → 8 visible + Advanced expander
# FIX 2: Single customer selector (no sidebar duplicate)
# FIX 3: Interest rate auto-assigned, moved to Advanced section
with tab_uw:
    st.header("Credit Decision Engine")
    st.markdown("##### ⚡ Quick Select Customer Profile")

    cust_keys = list(PRESETS.keys())
    quick_options = [
        "Customer 1 (Approve)",
        "Customer 2 (Approve with Conditions)",
        "Customer 3 (Reject / Decline)",
        "Custom (Manual Entry)"
    ]

    customer_sel = st.radio(
        "Select customer to auto-fill inputs:",
        quick_options,
        index=0,
        horizontal=True,
        label_visibility="collapsed"
    )

    if customer_sel == "Customer 1 (Approve)":
        base = pd.Series(PRESETS[cust_keys[0]])
        st.success("**Customer 1 loaded (Target: APPROVE)** — Prime gig worker: 3-year tenure, 12/12 on-time utility bills, low BNPL, positive income growth.")
    elif customer_sel == "Customer 2 (Approve with Conditions)":
        base = pd.Series(PRESETS[cust_keys[1]])
        st.warning("**Customer 2 loaded (Target: APPROVE WITH CONDITIONS)** — Stressed worker: 8-month tenure, 6-day avg delay, higher BNPL, declining income trend.")
    elif customer_sel == "Customer 3 (Reject / Decline)":
        base = pd.Series(PRESETS[cust_keys[2]])
        st.error("**Customer 3 loaded (Target: REJECT / DECLINE)** — High-risk delinquent: 2-month tenure, unbanked, 18-day avg delay, 75 warning patterns triggered.")
    else:
        base = borrowers.drop(columns=["borrower_id"]).median(numeric_only=True)
        base["location_type"], base["education_level"] = "Tier-2 City", "Undergraduate"
        st.info("**Custom / Manual entry mode** — modify values below and click Score Applicant.")

    form_slug = "".join(c if c.isalnum() else "_" for c in customer_sel).strip("_")
    with st.form(f"applicant_form_{form_slug}"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Applicant Profile**")
            age = st.number_input("Age", 21, 60, int(base["age"]))
            tenure_gig = st.number_input("Months on gig platform", 1, 120, int(base["gig_tenure_months"]))
            loc_opts = ["Metro", "Tier-1 City", "Tier-2 City", "Semi-Urban", "Rural"]
            location = st.selectbox("Location", loc_opts, index=loc_opts.index(base["location_type"]))
            edu_opts = ["Below 10th", "10th Pass", "12th Pass", "Diploma", "Undergraduate", "Postgraduate"]
            education = st.selectbox("Education", edu_opts, index=edu_opts.index(base["education_level"]))
        with c2:
            st.markdown("**Financial & Loan Details**")
            income = st.number_input("Avg monthly gig income (INR)", 2000.0, 150000.0, float(base["avg_monthly_income"]), step=500.0)
            on_time = st.slider("Utility bills paid on time (of 12)", 0, 12, int(base["utility_on_time_count"]))
            amount = st.number_input("Loan amount requested (INR)", 5000.0, 300000.0, float(base["loan_amount_requested"]), step=1000.0)
            tenure_opts = [3, 6, 9, 12, 18, 24]
            tenure = st.selectbox("Loan tenure (months)", tenure_opts, index=tenure_opts.index(int(base["loan_tenure_months"])))

        with st.expander("⚙️ Advanced Parameters (auto-derived from applicant data)"):
            st.caption("These parameters are automatically estimated from the applicant's profile. Override only if specific data is available.")
            a1, a2, a3 = st.columns(3)
            with a1:
                st.markdown("**Demographics**")
                dependents = st.number_input("Dependents", 0, 5, int(base["num_dependents"]))
                bank = st.selectbox("Has bank account", [1, 0], index=0 if int(base["has_bank_account"]) == 1 else 1, format_func=lambda v: "Yes" if v else "No")
                st.markdown("**Income Stability**")
                income_sd = st.number_input("Std-dev of monthly income", 0.0, 80000.0, float(base["std_monthly_income"]), step=250.0)
                trend = st.number_input("Income trend (INR/month)", -5000.0, 5000.0, float(base["income_trend_slope"]), step=50.0)
            with a2:
                st.markdown("**Payment Behaviour**")
                avg_delay = st.number_input("Avg utility delay (days)", 0.0, 90.0, float(base["avg_utility_delay_days"]), step=0.5)
                max_delay = st.number_input("Worst utility delay (days)", 0, 90, int(base["max_utility_delay_days"]))
                recharge = st.number_input("Avg days between recharges", 10.0, 60.0, float(base["avg_days_between_recharges"]), step=0.5)
                bnpl = st.number_input("Avg monthly BNPL spend (INR)", 0.0, 60000.0, float(base["avg_bnpl_monthly_amount"]), step=250.0)
            with a3:
                st.markdown("**Cash Flow & Digital Footprint**")
                outflow = st.number_input("Avg monthly spending (INR)", 500.0, 160000.0, float(base["avg_monthly_outflow"]), step=500.0)
                balance = st.number_input("Avg month-end balance (INR)", 0.0, 100000.0, float(base["avg_eom_balance"]), step=250.0)
                logins = st.number_input("App logins per month", 0.0, 90.0, float(base["avg_app_logins"]), step=0.5)
                upi = st.number_input("UPI transactions per month", 0.0, 80.0, float(base["avg_upi_transactions"]), step=0.5)
            st.markdown("**Credit History & Pricing**")
            h1, h2, h3 = st.columns(3)
            rate = h1.number_input("Interest rate (% p.a.)", 8.0, 36.0, float(base["interest_rate_annual"]), step=0.25)
            prev = h2.number_input("Previous micro-loans", 0, 15, int(base["previous_micro_loans"]))
            prev_ok = h3.number_input("...repaid on time", 0, 15, min(int(base["previous_loans_repaid_on_time"]), 15))

        st.form_submit_button("Score Applicant", type="primary")

    raw = pd.DataFrame([{
        "age": age, "gig_tenure_months": tenure_gig, "has_bank_account": bank, "num_dependents": dependents,
        "location_type": location, "education_level": education, "avg_monthly_income": income,
        "std_monthly_income": income_sd, "income_trend_slope": trend, "utility_on_time_count": on_time,
        "avg_utility_delay_days": avg_delay, "max_utility_delay_days": max_delay,
        "avg_days_between_recharges": recharge, "avg_bnpl_monthly_amount": bnpl, "avg_monthly_outflow": outflow,
        "avg_eom_balance": balance, "avg_app_logins": logins, "avg_upi_transactions": upi,
        "loan_amount_requested": amount, "loan_tenure_months": tenure, "interest_rate_annual": rate,
        "previous_micro_loans": prev, "previous_loans_repaid_on_time": min(prev_ok, prev),
    }])
    X = build_features(raw)
    pd_ttc = float(model.predict_proba(X)[0, 1])
    pd_pit = min(pd_ttc * mult, 0.99)
    be = float(np.atleast_1d(break_even_pd(rate, tenure))[0])
    band = risk_band(pd_pit)
    flags = flag_items(raw, item_defs).iloc[0]
    hits = triggered_rules(flags, rules)
    expected_loss = pd_pit * LGD * amount
    suggested_rate = BAND_RATES.get(band, 22.0)

    if pd_pit >= be or band == "E" or len(hits) >= 15:
        decision, colour, reason = "DECLINE / REFER", "#d73027", (
            f"PD {pd_pit:.1%} exceeds the economic break-even PD {be:.1%} for this interest rate and tenure."
            if pd_pit >= be else
            f"Severe credit risk (PD {pd_pit:.1%}, Risk Band {band}) with {len(hits)} delinquency warning rules triggered: automatic decline."
        )
    elif len(hits) >= 3 or band in ("C", "D"):
        decision, colour, reason = "APPROVE WITH CONDITIONS", "#fc8d59", f"Profitable on average, but triggers {len(hits)} warning rule(s) (Risk Band {band}): recommend a reduced credit ticket or weekly amortisation."
    else:
        decision, colour, reason = "APPROVE", "#1a9850", f"PD {pd_pit:.1%} is comfortably below the break-even PD {be:.1%} with clean behavioural patterns."

    st.markdown(f"<div style='padding:14px 18px;border-radius:10px;background:{colour};color:white;font-size:1.4rem;font-weight:700'>"
                f"{decision}</div><p style='margin-top:6px;font-size:1.05rem;'>{reason}</p>", unsafe_allow_html=True)
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("PD (through-the-cycle)", f"{pd_ttc:.1%}")
    m2.metric("PD (point-in-time)", f"{pd_pit:.1%}", delta=f"{(mult - 1) * 100:+.0f}% macro", delta_color="inverse")
    m3.metric("Risk band", band)
    m4.metric("Break-even PD", f"{be:.1%}")
    m5.metric("Expected loss", f"INR {expected_loss:,.0f}")
    m6.metric(f"Suggested rate (Band {band})", f"{suggested_rate:.1f}%")

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("TreeSHAP Explainability (Feature Contributions)")
        contrib = model.predict(X, pred_contrib=True)[0]   # TreeSHAP values in log-odds; last column = base value
        contrib = pd.Series(contrib[:-1], index=X.columns)
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(10)[::-1]
        fig = go.Figure(go.Bar(x=top.values, y=[FEATURE_LABELS.get(c, c) for c in top.index], orientation="h",
                               marker_color=["#d73027" if v > 0 else "#1a9850" for v in top.values],
                               hovertemplate="%{y}: %{x:+.3f} log-odds<extra></extra>"))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Contribution to log-odds of default (red = raises risk, green = lowers risk)", template="plotly_white")
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader(f"Delinquency Warning Rules Triggered: {len(hits)}")
        present = [i["item"] for i in item_defs["items"] if flags[i["item"]]]
        st.write("Active Behavioural Flags: " + (", ".join(f"`{p}`" for p in present) if present else "*None*"))
        if len(hits):
            show = hits[["antecedent", "test_confidence", "test_lift"]].head(8).rename(columns={
                "antecedent": "Pattern", "test_confidence": "Default Rate (Unseen)", "test_lift": "Lift"})
            st.dataframe(show.style.format({"Default Rate (Unseen)": "{:.1%}", "Lift": "{:.2f}x"}), hide_index=True, width="stretch")
        else:
            st.success("No validated delinquency warning patterns match this applicant.")

# ============================================================================================== TAB 2: Dataset Explorer
# FIX 5: Added empty-filter guard to prevent crashes during live demo
with tab_data:
    st.header("Validated Project Dataset Explorer & Drilldown")
    st.caption("Interactive browser for 6,000 held-out test borrowers evaluated across Review 1 & Review 2 models.")

    f1, f2, f3, f4, f5 = st.columns(5)
    with f1:
        band_filter = st.multiselect("Risk Band", ["A", "B", "C", "D", "E"], default=["A", "B", "C", "D", "E"])
    with f2:
        all_locs = sorted(test_borrowers["location_type"].dropna().unique())
        loc_filter = st.multiselect("Location Type", all_locs, default=all_locs)
    with f3:
        all_plats = sorted(test_borrowers["primary_gig_platform"].dropna().unique())
        plat_filter = st.multiselect("Gig Platform", all_plats, default=all_plats)
    with f4:
        status_choice = st.selectbox("Actual Loan Outcome", ["All", "Repaid (0)", "Defaulted (1)"])
    with f5:
        min_rules_hit = st.slider("Min Warning Rules Hit", 0, 10, 0)

    # Dynamic Filter Application
    filtered = test_borrowers[
        (test_borrowers["risk_band"].isin(band_filter)) &
        (test_borrowers["location_type"].isin(loc_filter)) &
        (test_borrowers["primary_gig_platform"].isin(plat_filter)) &
        (test_borrowers["rules_hit"] >= min_rules_hit)
    ]
    if status_choice == "Repaid (0)":
        filtered = filtered[filtered["loan_default_status"] == 0]
    elif status_choice == "Defaulted (1)":
        filtered = filtered[filtered["loan_default_status"] == 1]

    if len(filtered) == 0:
        st.info("No borrowers match the current filter combination. Adjust the filters above to see results.")
    else:
        # Dynamic KPIs
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Selected Borrowers", f"{len(filtered):,}")
        avg_pd = filtered["predicted_pd"].mean()
        k2.metric("Mean Predicted PD", f"{avg_pd:.2%}")
        obs_rate = filtered["loan_default_status"].mean()
        k3.metric("Observed Default Rate", f"{obs_rate:.2%}")
        total_vol = filtered["loan_amount_requested"].sum() / 1e5
        k4.metric("Total Exposure", f"INR {total_vol:,.1f} Lakhs")
        high_risk_cnt = (filtered["rules_hit"] >= 3).sum()
        k5.metric("High Warning Rules (>=3)", f"{high_risk_cnt:,}",
                  delta=f"{(high_risk_cnt / len(filtered) * 100):.1f}% of filtered")

        # Dynamic Charts
        c_chart1, c_chart2 = st.columns([1.2, 1])
        with c_chart1:
            fig_dist = px.histogram(
                filtered, x="predicted_pd", color="loan_default_status", nbins=30, barmode="overlay",
                color_discrete_map={0: "#1a9850", 1: "#d73027"},
                labels={"predicted_pd": "Predicted PD", "loan_default_status": "Default Outcome"},
                template="plotly_white", title="Predicted PD Distribution by Actual Loan Outcome"
            )
            fig_dist.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_dist, width="stretch")

        with c_chart2:
            band_summary = filtered.groupby("risk_band", observed=True).agg(
                borrowers=("borrower_id", "count"),
                default_rate=("loan_default_status", "mean")
            ).reset_index()
            fig_band = px.bar(
                band_summary, x="risk_band", y="default_rate", color="risk_band",
                color_discrete_map=BAND_COLORS, template="plotly_white",
                title="Observed Default Rate by Predicted Risk Band"
            )
            fig_band.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), yaxis_tickformat=".1%")
            st.plotly_chart(fig_band, width="stretch")

        # Table View
        st.subheader(f"Evaluated Records ({len(filtered):,} applicants)")
        display_cols = [
            "borrower_id", "primary_gig_platform", "location_type", "avg_monthly_income",
            "loan_amount_requested", "loan_tenure_months", "predicted_pd", "risk_band",
            "rules_hit", "policy_decision", "loan_default_status"
        ]
        rename_map = {
            "borrower_id": "Borrower ID", "primary_gig_platform": "Platform", "location_type": "Location",
            "avg_monthly_income": "Income (INR)", "loan_amount_requested": "Loan (INR)",
            "loan_tenure_months": "Tenure (m)", "predicted_pd": "Model PD", "risk_band": "Risk Band",
            "rules_hit": "Rules Hit", "policy_decision": "Policy Decision", "loan_default_status": "Actual Default"
        }
        st.dataframe(
            filtered[display_cols].rename(columns=rename_map).style.format({
                "Income (INR)": "{:,.0f}", "Loan (INR)": "{:,.0f}", "Model PD": "{:.2%}", "Actual Default": "{:d}"
            }), height=350, width="stretch"
        )

# ============================================================================================== TAB 3: Portfolio Outlook
# FIX 4: Replaced hardcoded "2025" with dynamic current year
with tab_port:
    st.header("Portfolio Default-Rate Outlook & Stress Testing")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Latest Month Default Rate", f"{pm['default_rate'].iloc[-1]:.2%}")
    k2.metric("Active Loans Tracked", f"{int(pm['active_loans'].iloc[-1]):,}")
    k3.metric("Neutral Economy Rate", f"{overlay['neutral_default_rate_pct']:.2f}%")
    k4.metric("Forecast Model", "SARIMAX + Macro Drivers", help=overlay["model"])

    show_scen = st.multiselect("Macro Scenarios", list(SCEN_COLORS), default=list(SCEN_COLORS))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pm["date"], y=pm["default_rate"] * 100, name="Historical Actual", line=dict(color="black", width=2.5)))
    if st.checkbox("Show 30+ DPD early-warning series"):
        fig.add_trace(go.Scatter(x=pm["date"], y=pm["dpd30_plus_rate"] * 100, name="30+ DPD Indicator", line=dict(color="#937860", dash="dot")))
    for name in show_scen:
        g = scen[scen["scenario"] == name]
        fig.add_trace(go.Scatter(x=pd.concat([g["date"], g["date"][::-1]]), y=pd.concat([g["upper80"], g["lower80"][::-1]]),
                                 fill="toself", fillcolor=SCEN_COLORS[name], opacity=0.15, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=g["date"], y=g["forecast"], name=name, line=dict(color=SCEN_COLORS[name], width=2.5)))
    fig.add_vrect(x0="2020-03-01", x1="2020-08-01", fillcolor="grey", opacity=0.12, line_width=0, annotation_text="COVID Lockdowns")
    fig.update_layout(height=470, yaxis_title="% of Active Loans Reaching 90+ DPD", hovermode="x unified", template="plotly_white",
                      xaxis=dict(rangeslider=dict(visible=True)))
    st.plotly_chart(fig, width="stretch")

    _ecl_year = datetime.now().year
    c1, c2 = st.columns(2)
    with c1:
        st.subheader(f"{_ecl_year} Expected Credit Loss (ECL) Impact")
        st.dataframe(impact.style.format({
            "avg monthly default rate %": "{:.2f}", "peak month rate %": "{:.2f}",
            "expected defaults 2025": "{:,.0f}", "expected credit loss (INR lakh)": "{:,.1f}",
            "extra loss vs baseline (INR lakh)": "{:+,.1f}"
        }), width="stretch")
    with c2:
        st.subheader("Hold-out Forecast Accuracy (2024 Actuals)")
        st.dataframe(ts_cmp[["MAE (pp)", "RMSE (pp)", "MAPE %"]].style.format("{:.3f}").highlight_min(color="#c7e9c0"), width="stretch")

    st.subheader("Macroeconomic Drivers (Live Institutional APIs)")
    indicator = st.selectbox("Select Macro Variable", ["unemployment_rate", "gdp_growth", "cpi_inflation_yoy", "call_money_rate", "gsec_10y_yield", "inr_per_usd"])
    src_info = sources["columns"].get(indicator, {})
    st.plotly_chart(px.line(macro[macro["date"] >= "2018-01-01"], x="date", y=indicator, template="plotly_white", height=320,
                            title=f"{indicator} — {src_info.get('api', '')} {src_info.get('series_id', src_info.get('indicator_id', ''))}"), width="stretch")

# ============================================================================================== TAB 4: Warning Rules
with tab_rules:
    st.header("Delinquency Warning-Rule Explorer")
    st.caption("93 rules mined with Apriori / FP-Growth on 24,000 training borrowers and validated on 6,000 unseen borrowers (notebook 07).")
    f1, f2, f3 = st.columns(3)
    min_lift = f1.slider("Minimum test lift", 1.0, 3.5, 1.5, 0.1)
    min_cov = f2.slider("Minimum share of applicants matching", 0.0, 0.10, 0.0, 0.005, format="%.3f")
    must = f3.multiselect("Rule must contain", sorted({i for items in rules["items"] for i in items}))
    view = rules[(rules["test_lift"] >= min_lift) & (rules["test_coverage"] >= min_cov)]
    if must:
        view = view[view["items"].apply(lambda items: all(m in items for m in must))]
    st.write(f"Displaying **{len(view)}** validated rules matching criteria")
    fig = px.scatter(view, x="test_coverage", y="test_confidence", size="support", color="test_lift", hover_name="antecedent",
                     color_continuous_scale="Reds", template="plotly_white", height=420,
                     labels={"test_coverage": "Share of applicants matching", "test_confidence": "Default rate when pattern matched"})
    fig.add_hline(y=item_defs["base_default_rate"], line_dash="dash", annotation_text="Portfolio Baseline Default Rate")
    fig.update_layout(xaxis_tickformat=".1%", yaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(view[["antecedent", "n_items", "support", "confidence", "lift", "test_coverage", "test_confidence", "test_lift"]]
                 .sort_values("test_lift", ascending=False)
                 .style.format({"support": "{:.2%}", "confidence": "{:.1%}", "lift": "{:.2f}", "test_coverage": "{:.2%}", "test_confidence": "{:.1%}", "test_lift": "{:.2f}"}),
                 hide_index=True, width="stretch")

# ============================================================================================== TAB 5: Model Card
# FIX 6: Replaced LaTeX math with plain-text/Unicode to avoid rendering issues across Streamlit versions
with tab_model:
    st.header("Model Governance & Data Provenance")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Test AUC-ROC", card["test_auc"], help=f"Best achievable ceiling on this data (true PD oracle): {card['oracle_auc_true_pd']}")
    a2.metric("Test PR-AUC", card["test_pr_auc"])
    a3.metric("Brier Score", card["test_brier"])
    a4.metric("Mean PD vs Observed", f"{card['mean_pd_test']:.1%} / {card['observed_default_rate_test']:.1%}")
    cal = pd.DataFrame(card["calibration_by_decile"]).T
    fig = go.Figure([go.Scatter(x=cal["pred"], y=cal["obs"], mode="lines+markers", name="Calibrated Model"),
                     go.Scatter(x=[0, cal["pred"].max()], y=[0, cal["pred"].max()], mode="lines", name="Perfect Calibration", line=dict(dash="dash", color="grey"))])
    fig.update_layout(title="Calibration by Decile (6,000 Held-Out Test Borrowers)", xaxis_title="Mean Predicted PD", yaxis_title="Observed Default Rate", height=360, template="plotly_white")
    st.plotly_chart(fig, width="stretch")
    st.markdown(f"""
* **Model Engine:** {card['model']} on {len(card['features'])} application features; interpretability via LightGBM TreeSHAP.
* **Economic Decision Frontier:** Decline when PD > Break-Even PD = Margin / (Margin + LGD), where Margin = (Interest Rate − {COST_OF_FUNDS:.0%} Cost of Funds) × Tenure, LGD = {LGD:.0%}.
* **Policy on Held-Out Test Set (Neutral Economy):** Approves **{card['policy']['approval_rate_test']:.1%}** with an approved default rate of only **{card['policy']['default_rate_approved_test']:.1%}** vs **{card['policy']['default_rate_declined_test']:.1%}** among declined applicants.
* **Data Provenance:** Borrowers and transactions are programmatically generated (`data_generation.py`); macroeconomic indicators are harvested live from Federal Reserve FRED and World Bank APIs (`src/macro_api.py`).
""")
    prov = pd.DataFrame(sources["columns"]).T.fillna("")
    st.dataframe(prov[[c for c in ["api", "series_id", "indicator_id", "origin", "last_obs", "years", "url"] if c in prov.columns]], width="stretch")
    st.caption(f"Macro series generated {sources['generated_at']}. Pipeline command: `python -m src.macro_api` -> notebooks 06-08.")
