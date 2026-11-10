"""
IntelliCredit - Loan Underwriter Dashboard
==========================================
Real-time credit decisions that combine all three analytical layers:
  * Review 1 PD model           -> calibrated probability of default + SHAP-style reasons
  * Review 2 association rules  -> which validated warning patterns the applicant triggers
  * Review 2 time-series model  -> macro scenario multiplier (point-in-time PD) and portfolio outlook

Run from the repository root:
    streamlit run dashboard/app.py
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import DATA_DIR, MODELS_DIR, RANDOM_STATE, TEST_SIZE  # noqa: E402
from src.risk_rules import flag_items, load_rules, triggered_rules  # noqa: E402
from src.underwriting_model import (  # noqa: E402
    COST_OF_FUNDS, FEATURE_LABELS, LGD, MODEL_PATH, CARD_PATH, break_even_pd, build_features, risk_band,
)

st.set_page_config(page_title="IntelliCredit Underwriter", page_icon=":credit_card:", layout="wide")

SCEN_COLORS = {"Baseline": "#4c72b0", "Adverse": "#dd8452", "Severe": "#c44e52"}
BAND_COLORS = {"A": "#1a9850", "B": "#91cf60", "C": "#fee08b", "D": "#fc8d59", "E": "#d73027"}


# ---------------------------------------------------------------------------------------------- data
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_artifacts():
    borrowers = pd.read_csv(os.path.join(DATA_DIR, "borrower_profiles.csv"))
    from sklearn.model_selection import train_test_split
    _, test_idx = train_test_split(borrowers.index, test_size=TEST_SIZE,
                                   stratify=borrowers["loan_default_status"], random_state=RANDOM_STATE)
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
    return borrowers, borrowers.loc[test_idx], card, overlay, item_defs, rules, scen, impact, ts_cmp, pm, macro, sources


model = load_model()
borrowers, test_borrowers, card, overlay, item_defs, rules, scen, impact, ts_cmp, pm, macro, sources = load_artifacts()
multipliers = {"Neutral (through-the-cycle)": 1.0, **overlay["multipliers_next_12m"]}

# ---------------------------------------------------------------------------------------------- sidebar
st.sidebar.title("IntelliCredit")
st.sidebar.caption("Alternative credit scoring for thin-file & gig-economy borrowers")
scenario = st.sidebar.radio("Macro scenario for the next 12 months", list(multipliers), index=1,
                            help="From the SARIMAX forecast (notebook 06). The multiplier converts a through-the-cycle PD into a point-in-time PD.")
apply_overlay = st.sidebar.toggle("Apply macro overlay to the decision", value=True)
mult = multipliers[scenario] if apply_overlay else 1.0
st.sidebar.metric("PD multiplier in use", f"{mult:.2f}x")

st.sidebar.divider()
st.sidebar.subheader("Applicant")
sample_ids = test_borrowers["borrower_id"].sort_values().head(400).tolist()
source = st.sidebar.selectbox("Pre-fill the form from", ["Hold-out applicant", "Blank (typical values)"])
if source == "Hold-out applicant":
    picked = st.sidebar.selectbox("Borrower ID (test set - never seen in training)", sample_ids, index=3)
    base = borrowers.set_index("borrower_id").loc[picked]
else:
    picked = None
    base = borrowers.drop(columns=["borrower_id"]).median(numeric_only=True)
    base["location_type"], base["education_level"] = "Tier-2 City", "Undergraduate"

# ---------------------------------------------------------------------------------------------- tabs
tab_uw, tab_port, tab_rules, tab_model = st.tabs(["Underwriter", "Portfolio outlook", "Warning rules", "Model & data"])

with tab_uw:
    st.header("Credit decision")
    with st.form("applicant"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Profile**")
            age = st.number_input("Age", 21, 60, int(base["age"]))
            tenure_gig = st.number_input("Months on gig platform", 1, 120, int(base["gig_tenure_months"]))
            dependents = st.number_input("Dependents", 0, 5, int(base["num_dependents"]))
            bank = st.selectbox("Has bank account", [1, 0], index=0 if int(base["has_bank_account"]) == 1 else 1, format_func=lambda v: "Yes" if v else "No")
            loc_opts = ["Metro", "Tier-1 City", "Tier-2 City", "Semi-Urban", "Rural"]
            location = st.selectbox("Location", loc_opts, index=loc_opts.index(base["location_type"]))
            edu_opts = ["Below 10th", "10th Pass", "12th Pass", "Diploma", "Undergraduate", "Postgraduate"]
            education = st.selectbox("Education", edu_opts, index=edu_opts.index(base["education_level"]))
        with c2:
            st.markdown("**Income & cash flow (12-month averages, INR)**")
            income = st.number_input("Avg monthly gig income", 2000.0, 150000.0, float(base["avg_monthly_income"]), step=500.0)
            income_sd = st.number_input("Std-dev of monthly income", 0.0, 80000.0, float(base["std_monthly_income"]), step=250.0)
            trend = st.number_input("Income trend (INR per month)", -5000.0, 5000.0, float(base["income_trend_slope"]), step=50.0)
            outflow = st.number_input("Avg monthly spending", 500.0, 160000.0, float(base["avg_monthly_outflow"]), step=500.0)
            balance = st.number_input("Avg month-end balance", 0.0, 100000.0, float(base["avg_eom_balance"]), step=250.0)
            bnpl = st.number_input("Avg monthly BNPL spend", 0.0, 60000.0, float(base["avg_bnpl_monthly_amount"]), step=250.0)
        with c3:
            st.markdown("**Payment behaviour & digital footprint**")
            on_time = st.slider("Utility bills paid on time (of 12)", 0, 12, int(base["utility_on_time_count"]))
            avg_delay = st.number_input("Avg utility delay (days)", 0.0, 90.0, float(base["avg_utility_delay_days"]), step=0.5)
            max_delay = st.number_input("Worst utility delay (days)", 0, 90, int(base["max_utility_delay_days"]))
            recharge = st.number_input("Avg days between mobile recharges", 10.0, 60.0, float(base["avg_days_between_recharges"]), step=0.5)
            logins = st.number_input("App logins per month", 0.0, 90.0, float(base["avg_app_logins"]), step=0.5)
            upi = st.number_input("UPI transactions per month", 0.0, 80.0, float(base["avg_upi_transactions"]), step=0.5)
        st.markdown("**Loan request & credit history**")
        d1, d2, d3, d4, d5 = st.columns(5)
        amount = d1.number_input("Loan amount (INR)", 5000.0, 300000.0, float(base["loan_amount_requested"]), step=1000.0)
        tenure_opts = [3, 6, 9, 12, 18, 24]
        tenure = d2.selectbox("Tenure (months)", tenure_opts, index=tenure_opts.index(int(base["loan_tenure_months"])))
        rate = d3.number_input("Interest rate (% p.a.)", 8.0, 36.0, float(base["interest_rate_annual"]), step=0.25)
        prev = d4.number_input("Previous micro-loans", 0, 15, int(base["previous_micro_loans"]))
        prev_ok = d5.number_input("...repaid on time", 0, 15, min(int(base["previous_loans_repaid_on_time"]), 15))
        st.form_submit_button("Score applicant", type="primary")

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
    required_rate = (COST_OF_FUNDS + (pd_pit * LGD / (1 - pd_pit)) / (tenure / 12)) * 100

    if pd_pit >= be:
        decision, colour, reason = "DECLINE / REFER", "#d73027", f"PD {pd_pit:.1%} is above the break-even PD {be:.1%} for this rate and tenure."
    elif len(hits) >= 3 or band in ("C", "D"):
        decision, colour, reason = "APPROVE WITH CONDITIONS", "#fc8d59", f"Profitable on average, but {len(hits)} warning rule(s) fire / band {band}: offer a smaller ticket or weekly instalments."
    else:
        decision, colour, reason = "APPROVE", "#1a9850", f"PD {pd_pit:.1%} is comfortably below the break-even PD {be:.1%}."

    st.markdown(f"<div style='padding:14px 18px;border-radius:10px;background:{colour};color:white;font-size:1.4rem;font-weight:700'>"
                f"{decision}</div><p style='margin-top:6px'>{reason}</p>", unsafe_allow_html=True)
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("PD (through-the-cycle)", f"{pd_ttc:.1%}")
    m2.metric("PD (point-in-time)", f"{pd_pit:.1%}", delta=f"{(mult - 1) * 100:+.0f}% macro", delta_color="inverse")
    m3.metric("Risk band", band)
    m4.metric("Break-even PD", f"{be:.1%}")
    m5.metric("Expected loss", f"INR {expected_loss:,.0f}")
    m6.metric("Min. rate to break even", f"{required_rate:.1f}%")
    if picked is not None:
        actual = int(borrowers.set_index("borrower_id").loc[picked, "loan_default_status"])
        st.caption(f"Hold-out applicant {picked}: actual outcome in the data = {'DEFAULTED' if actual else 'repaid'} (not used by the model).")

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Why this PD? (feature contributions)")
        contrib = model.predict(X, pred_contrib=True)[0]   # TreeSHAP values in log-odds; last column = base value
        contrib = pd.Series(contrib[:-1], index=X.columns)
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(10)[::-1]
        fig = go.Figure(go.Bar(x=top.values, y=[FEATURE_LABELS.get(c, c) for c in top.index], orientation="h",
                               marker_color=["#d73027" if v > 0 else "#1a9850" for v in top.values],
                               hovertemplate="%{y}: %{x:+.3f} log-odds<extra></extra>"))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="contribution to log-odds of default (red = raises risk)", template="plotly_white")
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader(f"Warning rules triggered: {len(hits)}")
        present = [i["item"] for i in item_defs["items"] if flags[i["item"]]]
        st.write("Flags present: " + (", ".join(f"`{p}`" for p in present) if present else "none"))
        if len(hits):
            show = hits[["antecedent", "test_confidence", "test_lift"]].head(8).rename(columns={
                "antecedent": "Pattern", "test_confidence": "Default rate (unseen borrowers)", "test_lift": "Lift"})
            st.dataframe(show.style.format({"Default rate (unseen borrowers)": "{:.1%}", "Lift": "{:.2f}x"}), hide_index=True, width="stretch")
        else:
            st.success("No validated warning pattern matches this applicant.")

with tab_port:
    st.header("Portfolio default-rate outlook")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Latest month default rate", f"{pm['default_rate'].iloc[-1]:.2%}")
    k2.metric("Active loans", f"{int(pm['active_loans'].iloc[-1]):,}")
    k3.metric("Neutral-economy rate", f"{overlay['neutral_default_rate_pct']:.2f}%")
    k4.metric("Forecast model", "SARIMAX", help=overlay["model"])

    show_scen = st.multiselect("Scenarios", list(SCEN_COLORS), default=list(SCEN_COLORS))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pm["date"], y=pm["default_rate"] * 100, name="Actual", line=dict(color="black", width=2.5)))
    if st.checkbox("Show 30+ DPD early-warning series"):
        fig.add_trace(go.Scatter(x=pm["date"], y=pm["dpd30_plus_rate"] * 100, name="30+ DPD", line=dict(color="#937860", dash="dot")))
    for name in show_scen:
        g = scen[scen["scenario"] == name]
        fig.add_trace(go.Scatter(x=pd.concat([g["date"], g["date"][::-1]]), y=pd.concat([g["upper80"], g["lower80"][::-1]]),
                                 fill="toself", fillcolor=SCEN_COLORS[name], opacity=0.15, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=g["date"], y=g["forecast"], name=name, line=dict(color=SCEN_COLORS[name], width=2.5)))
    fig.add_vrect(x0="2020-03-01", x1="2020-08-01", fillcolor="grey", opacity=0.12, line_width=0, annotation_text="COVID")
    fig.update_layout(height=470, yaxis_title="% of active loans reaching 90+ DPD", hovermode="x unified", template="plotly_white",
                      xaxis=dict(rangeslider=dict(visible=True)))
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("2025 credit-loss impact")
        st.dataframe(impact.style.format({"avg monthly default rate %": "{:.2f}", "peak month rate %": "{:.2f}", "expected defaults 2025": "{:,.0f}",
                                          "expected credit loss (INR lakh)": "{:,.1f}", "extra loss vs baseline (INR lakh)": "{:+,.1f}"}), width="stretch")
    with c2:
        st.subheader("Forecast accuracy (2024 hold-out)")
        st.dataframe(ts_cmp[["MAE (pp)", "RMSE (pp)", "MAPE %"]].style.format("{:.3f}").highlight_min(color="#c7e9c0"), width="stretch")

    st.subheader("Macro drivers (live API data)")
    indicator = st.selectbox("Indicator", ["unemployment_rate", "gdp_growth", "cpi_inflation_yoy", "call_money_rate", "gsec_10y_yield", "inr_per_usd"])
    src_info = sources["columns"].get(indicator, {})
    st.plotly_chart(px.line(macro[macro["date"] >= "2018-01-01"], x="date", y=indicator, template="plotly_white", height=320,
                            title=f"{indicator} - {src_info.get('api', '')} {src_info.get('series_id', src_info.get('indicator_id', ''))}"), width="stretch")

with tab_rules:
    st.header("Warning-rule explorer")
    st.caption("93 rules mined with Apriori / FP-Growth on 24,000 training borrowers and validated on 6,000 unseen borrowers (notebook 07).")
    f1, f2, f3 = st.columns(3)
    min_lift = f1.slider("Minimum test lift", 1.0, 3.5, 1.5, 0.1)
    min_cov = f2.slider("Minimum share of applicants matching", 0.0, 0.10, 0.0, 0.005, format="%.3f")
    must = f3.multiselect("Rule must contain", sorted({i for items in rules["items"] for i in items}))
    view = rules[(rules["test_lift"] >= min_lift) & (rules["test_coverage"] >= min_cov)]
    if must:
        view = view[view["items"].apply(lambda items: all(m in items for m in must))]
    st.write(f"{len(view)} rules")
    fig = px.scatter(view, x="test_coverage", y="test_confidence", size="support", color="test_lift", hover_name="antecedent",
                     color_continuous_scale="Reds", template="plotly_white", height=420,
                     labels={"test_coverage": "share of applicants matching", "test_confidence": "default rate when matched"})
    fig.add_hline(y=item_defs["base_default_rate"], line_dash="dash", annotation_text="portfolio average")
    fig.update_layout(xaxis_tickformat=".1%", yaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(view[["antecedent", "n_items", "support", "confidence", "lift", "test_coverage", "test_confidence", "test_lift"]]
                 .sort_values("test_lift", ascending=False)
                 .style.format({"support": "{:.2%}", "confidence": "{:.1%}", "lift": "{:.2f}", "test_coverage": "{:.2%}", "test_confidence": "{:.1%}", "test_lift": "{:.2f}"}),
                 hide_index=True, width="stretch")

with tab_model:
    st.header("Model card & data provenance")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Test AUC", card["test_auc"], help=f"Best achievable on this data (true PD): {card['oracle_auc_true_pd']}")
    a2.metric("Test PR-AUC", card["test_pr_auc"])
    a3.metric("Brier score", card["test_brier"])
    a4.metric("Mean PD vs observed", f"{card['mean_pd_test']:.1%} / {card['observed_default_rate_test']:.1%}")
    cal = pd.DataFrame(card["calibration_by_decile"]).T
    fig = go.Figure([go.Scatter(x=cal["pred"], y=cal["obs"], mode="lines+markers", name="model"),
                     go.Scatter(x=[0, cal["pred"].max()], y=[0, cal["pred"].max()], mode="lines", name="perfect", line=dict(dash="dash", color="grey"))])
    fig.update_layout(title="Calibration by decile (test set)", xaxis_title="mean predicted PD", yaxis_title="observed default rate", height=360, template="plotly_white")
    st.plotly_chart(fig, width="stretch")
    st.markdown(f"""
* **Model:** {card['model']} on {len(card['features'])} application features; explanations via LightGBM TreeSHAP.
* **Decision rule:** decline when PD > break-even PD = margin / (margin + LGD), margin = (rate − {COST_OF_FUNDS:.0%} cost of funds) × tenure in years, LGD = {LGD:.0%}.
* **Policy on the test set (neutral economy):** approves {card['policy']['approval_rate_test']:.1%}; default rate {card['policy']['default_rate_approved_test']:.1%} among approved vs {card['policy']['default_rate_declined_test']:.1%} among declined.
* **Data:** borrowers and transactions are synthetic (`data_generation.py`); macro indicators are collected live from the FRED and World Bank APIs (`src/macro_api.py`).
""")
    prov = pd.DataFrame(sources["columns"]).T.fillna("")
    st.dataframe(prov[[c for c in ["api", "series_id", "indicator_id", "origin", "last_obs", "years", "url"] if c in prov.columns]], width="stretch")
    st.caption(f"Macro table generated {sources['generated_at']}. Refresh: `python -m src.macro_api` -> re-run notebooks 06-08.")
