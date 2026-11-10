"""
IntelliCredit - Loan Decision Assistant
=======================================
A dashboard for lending to gig workers who have no credit-bureau history.

  * Score an applicant   -> chance of default (LightGBM) + decision + reasons (TreeSHAP) + path to approval
  * Test borrowers       -> 6,000 unseen borrowers: predicted risk vs what really happened
  * Review 1 models      -> the 11 first-round models compared (AUC, recall, calibration) and how the final model improved on them
  * Time series & rules  -> Methods 1 and 2 in one abstract page: SARIMAX forecast and 93 validated Apriori / FP-Growth rules
  * Proof & comparison   -> pipeline on 4 real credit datasets, realism tests, comparison with studies and products
  * Data explorer        -> browse, filter, profile and chart every dataset
  * Insights & Validation -> synthetic data tested against the UCI credit-card data (TSTR / TRTS, policy outcomes, fidelity audit)
  * Scenario lab         -> live what-if on the economy with the SARIMAX model, expected losses and lending decisions

Run from the repository root:
    streamlit run dashboard/app.py
"""

import json
import os
import sys
import warnings

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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import DATA_DIR, MODELS_DIR, RANDOM_STATE, TEST_SIZE  # noqa: E402
from src.risk_rules import flag_items, load_rules, triggered_rules, rules_hit_count  # noqa: E402
from src.recourse import path_to_approval  # noqa: E402
from validation_tab import render as render_proof  # noqa: E402
from explorer_tab import render as render_explorer  # noqa: E402
from scenario_tab import render as render_scenario  # noqa: E402
from methods_tab import render as render_methods  # noqa: E402
from review1_tab import render as render_review1  # noqa: E402
from uci_tab import render as render_uci  # noqa: E402
from src.underwriting_model import (  # noqa: E402
    COST_OF_FUNDS, FEATURE_LABELS, LGD, MODEL_PATH, CARD_PATH, break_even_pd, build_features, risk_band,
)

st.set_page_config(page_title="IntelliCredit - Loan Decision Assistant", layout="wide",
                   initial_sidebar_state="expanded")

# ============================================================================================== look & feel
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;0,8..60,700;1,8..60,400&display=swap');
html, body, .stApp, [data-testid="stMarkdownContainer"], button, input, textarea, select { font-family: 'Source Serif 4', Georgia, 'Times New Roman', serif; }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1260px; }
#MainMenu, footer { visibility: hidden; }
[data-testid="stToolbar"], .stDeployButton { display: none !important; }
[data-testid="stHeader"] { background: transparent; }
h1, h2, h3, h4 { color: #1f1f1f; font-weight: 700; letter-spacing: 0; }
h2 { font-size: 1.6rem !important; border-bottom: 1px solid #cfc7b3; padding-bottom: .35rem; margin-bottom: .4rem !important; }
h3 { font-size: 1.15rem !important; }

/* masthead */
.masthead { border-top: 4px double #1f1f1f; border-bottom: 1px solid #1f1f1f; padding: 12px 2px 10px 2px; margin: 0 0 12px 0; }
.mast-title { font-size: 2.1rem; font-weight: 700; letter-spacing: .01em; line-height: 1.1; color: #1f1f1f; }
.mast-sub { font-style: italic; font-size: 1.02rem; color: #3a3a3a; margin-top: 3px; }
.mast-meta { font-size: .8rem; letter-spacing: .07em; text-transform: uppercase; color: #6a6454; margin-top: 7px; }

/* tabs as a plain underlined menu */
.stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid #1f1f1f; flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] { padding: 8px 15px; font-weight: 600; font-size: .96rem; height: auto; color: #5a5546; border-radius: 0; margin-bottom: -1px; }
.stTabs [aria-selected="true"] { color: #1f1f1f; border-bottom: 3px solid #1f1f1f; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.1rem; }

/* notes and step titles */
.explain { border-left: 3px solid #8a8577; padding: 6px 14px; color: #3a3a3a; font-size: .98rem; line-height: 1.55; margin: 6px 0 16px 0; font-style: italic; }
.explain b { color: #1f1f1f; font-style: normal; }
.step { font-weight: 700; color: #1f1f1f; font-size: 1.1rem; margin: 8px 0 10px 0; border-bottom: 1px solid #d9d2c0; padding-bottom: 4px; }
.step span { font-variant: small-caps; letter-spacing: .06em; color: #6a6454; margin-right: 10px; font-weight: 600; }
.muted { color: #6a6454; font-size: .9rem; }

/* decision box */
.verdict { background: #fffdf8; border: 1px solid #cfc7b3; border-left-width: 9px; padding: 14px 20px; margin-bottom: 12px; }
.v-ok { border-left-color: #3d6b4f; } .v-cond { border-left-color: #b8862f; } .v-no { border-left-color: #9b2c2c; }
.v-label { font-size: .74rem; letter-spacing: .14em; text-transform: uppercase; color: #6a6454; }
.v-title { font-size: 1.9rem; font-weight: 700; line-height: 1.15; }
.v-ok .v-title { color: #2f5c41; } .v-cond .v-title { color: #8f6414; } .v-no .v-title { color: #8a2525; }
.v-sub { font-size: 1rem; color: #2d2d2d; margin-top: 4px; line-height: 1.5; }

/* metrics */
[data-testid="stMetric"] { background: #fffdf8; border: 1px solid #d9d2c0; border-radius: 2px; padding: 10px 14px; }
[data-testid="stMetricLabel"] p { font-size: .82rem; color: #6a6454; font-weight: 600; }
[data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 700; color: #1f1f1f; }

/* forms, containers, buttons */
[data-testid="stForm"] { background: #fffdf8; border: 1px solid #cfc7b3; border-radius: 2px; padding: 16px 18px 12px 18px; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 2px; }
[data-testid="stFormSubmitButton"] button, .stButton > button, .stDownloadButton > button { border-radius: 2px; font-weight: 600; padding: .5rem 1.4rem; }
[data-testid="stExpander"] { border-radius: 2px; background: #fffdf8; }
[data-testid="stDataFrame"] { border-radius: 2px; overflow: hidden; border: 1px solid #d9d2c0; }

/* radio groups as plain boxed options */
div[role="radiogroup"] { gap: 8px; }
div[role="radiogroup"] > label { background: #fffdf8; border: 1px solid #cfc7b3; border-radius: 2px; padding: 7px 14px; margin: 0; }
div[role="radiogroup"] > label:has(input:checked) { border-color: #1f1f1f; background: #f1ecdf; }
[data-testid="stSidebar"] div[role="radiogroup"] > label { width: 100%; }
.stTabs [data-baseweb="tab-panel"] h2:first-child { margin-top: 0; }

/* wrapping tables (glossary etc.) */
.gt { width: 100%; border-collapse: separate; border-spacing: 0; background: #fffdf8; border: 1px solid #cfc7b3; font-size: .95rem; }
.gt th { background: #f1ecdf; text-align: left; padding: 8px 14px; color: #1f1f1f; font-weight: 700; border-bottom: 1px solid #cfc7b3; }
.gt td { padding: 8px 14px; border-top: 1px solid #e6dfcf; vertical-align: top; color: #2d2d2d; line-height: 1.45; }
.gt td:first-child { font-weight: 600; white-space: nowrap; }

/* warning-pattern tags and cards */
.chip { display: inline-block; background: #fbf3f1; color: #8a2525; border: 1px solid #c9a5a2; border-radius: 2px; padding: 2px 9px;
        margin: 3px 4px 3px 0; font-size: .84rem; font-weight: 600; }
.chip.none { background: #f1f6ef; color: #2f5c41; border-color: #a9c4b0; }
.card { background: #fffdf8; border: 1px solid #cfc7b3; border-radius: 2px; padding: 16px 20px; min-height: 230px; }
.card h4 { margin: 0 0 6px 0; font-size: 1.08rem; }
.card p { margin: 0; color: #3a3a3a; font-size: .97rem; line-height: 1.55; }
.card .num { color: #6a6454; font-weight: 600; font-size: .8rem; letter-spacing: .12em; text-transform: uppercase; }
.who { background: #fffdf8; border: 1px solid #d9d2c0; border-left-width: 5px; border-radius: 2px; padding: 9px 16px; margin: 6px 0 12px 0; font-size: .97rem; color: #2d2d2d; }
.who.ok { border-left-color: #3d6b4f; } .who.cond { border-left-color: #b8862f; } .who.no { border-left-color: #9b2c2c; } .who.custom { border-left-color: #1f3a5f; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

PALETTE = ["#1f3a5f", "#b8862f", "#9b2c2c", "#3d6b4f", "#6b5b7b", "#5a5546"]
# Consistent colours across charts: scenarios and risk bands
SCEN_COLORS = {"Baseline": "#1f3a5f", "Adverse": "#b8862f", "Severe": "#9b2c2c"}
BAND_COLORS = {"A": "#5b8a6b", "B": "#9bb37c", "C": "#d8c46a", "D": "#d89b5a", "E": "#b34a4a"}
# Risk-band-based product rate schedule (used to auto-assign interest rate)
BAND_RATES = {"A": 14.0, "B": 18.0, "C": 22.0, "D": 26.0, "E": 30.0}
BAND_MEANING = {"A": "very low risk", "B": "low risk", "C": "medium risk", "D": "high risk", "E": "very high risk"}

SCEN_LABELS = {"Neutral (through-the-cycle)": "Normal economy", "Baseline": "Good times", "Adverse": "Mild slowdown", "Severe": "Severe downturn"}
SCEN_HELP = {
    "Neutral (through-the-cycle)": "Long-run average conditions. No adjustment to anyone's risk.",
    "Baseline": "The economy as forecast for next year: fewer defaults than the long-run average.",
    "Adverse": "Unemployment up 1.5 points and growth down 3 points.",
    "Severe": "A COVID-style shock with lockdown months. Defaults rise sharply.",
}

# Three demo applicants (approve / conditional / decline) used to auto-fill the form during a live demo
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


# ---------------------------------------------------------------------------------------------- UI helpers
def explain(text):
    """Plain-language 'what you are looking at' box at the top of a section."""
    st.markdown(f'<div class="explain">{text}</div>', unsafe_allow_html=True)


def step(n, text):
    st.markdown(f'<div class="step"><span>Step {n}</span>{text}</div>', unsafe_allow_html=True)


def style_fig(fig, height=340):
    """One consistent chart style across the whole app."""
    fig.update_layout(template="plotly_white", height=height, colorway=PALETTE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Georgia, 'Times New Roman', serif", size=13, color="#1f1f1f"), margin=dict(l=10, r=10, t=40, b=10),
                      hoverlabel=dict(font_size=13))
    fig.update_xaxes(gridcolor="#e6dfcf", zeroline=False, linecolor="#cfc7b3")
    fig.update_yaxes(gridcolor="#e6dfcf", zeroline=False, linecolor="#cfc7b3")
    return fig


def risk_gauge(pit, be):
    """Where the applicant's default chance sits relative to the risk grades and the break-even limit."""
    xmax = float(min(1.0, max(0.30, pit * 1.2, be * 1.6)))
    zones = [(0, .04, "A", "#dfe9dc"), (.04, .07, "B", "#ebeed8"), (.07, .11, "C", "#f3ecc9"), (.11, .16, "D", "#f2e0c8"), (.16, 1.0, "E", "#efd6d1")]
    fig = go.Figure()
    for lo, hi, lab, col in zones:
        if lo >= xmax:
            continue
        hi_c = min(hi, xmax)
        fig.add_shape(type="rect", x0=lo, x1=hi_c, y0=0, y1=1, fillcolor=col, line_width=0, layer="below")
        fig.add_annotation(x=(lo + hi_c) / 2, y=0.5, text=f"<b>{lab}</b>", showarrow=False, font=dict(size=15, color="#5a5546"))
    fig.add_shape(type="line", x0=be, x1=be, y0=-0.1, y1=1.1, line=dict(color="#1f1f1f", width=2.5, dash="dash"))
    fig.add_annotation(x=be, y=1.2, text=f"<b>Break-even limit {be:.1%}</b>", showarrow=False, yanchor="bottom", font=dict(size=12, color="#1f1f1f"))
    fig.add_trace(go.Scatter(x=[pit], y=[0.5], mode="markers", showlegend=False,
                             marker=dict(symbol="diamond", size=20, color="#1f3a5f", line=dict(color="#fffdf8", width=2)),
                             hovertemplate="This applicant: %{x:.1%}<extra></extra>"))
    fig.add_annotation(x=pit, y=-0.18, text=f"<b>This applicant {pit:.1%}</b>", showarrow=False, yanchor="top", font=dict(size=12, color="#1f3a5f"))
    fig.update_xaxes(range=[0, xmax], tickformat=".0%", showgrid=False, zeroline=False)
    fig.update_yaxes(range=[-0.75, 1.75], visible=False)
    style_fig(fig, 175)
    fig.update_layout(margin=dict(l=10, r=10, t=8, b=8))
    return fig


def html_table(df):
    """A table whose cells wrap (st.dataframe cuts long text off)."""
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>" for row in df.itertuples(index=False))
    st.markdown(f'<table class="gt"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>', unsafe_allow_html=True)


def pretty(pattern):
    """DECLINING_INCOME + INCOME_VOLATILE -> Declining income + Income volatile"""
    return " + ".join(p.strip().replace("_", " ").capitalize() for p in str(pattern).split("+"))


def colour_decision(v):
    return {"Decline": "background-color:#f3dedb;color:#8a2525;font-weight:700",
            "Approve with conditions": "background-color:#f5e8c8;color:#8f6414;font-weight:700",
            "Approve": "background-color:#dfeadb;color:#2f5c41;font-weight:700"}.get(v, "")


def decide(pit, be, band, n_rules):
    """Single decision policy used by the live Underwriter tab AND the Dataset Explorer.

    Returns "DECLINE", "CONDITIONS" or "APPROVE" (works on scalars or arrays).
      DECLINE    : PD at or above break-even, or risk band E, or 15+ warning rules fire
      CONDITIONS : 3+ warning rules fire, or risk band C/D
      APPROVE    : otherwise
    """
    pit, be, n_rules, band = np.asarray(pit), np.asarray(be), np.asarray(n_rules), np.asarray(band)
    return np.where((pit >= be) | (band == "E") | (n_rules >= 15), "DECLINE",
                    np.where((n_rules >= 3) | np.isin(band, ["C", "D"]), "CONDITIONS", "APPROVE"))


# ---------------------------------------------------------------------------------------------- data
@st.cache_resource
def load_model():
    """Load the trained LightGBM model once per server process (st.cache_resource)."""
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_artifacts(_model):
    """Load every data/model artifact the tabs need and pre-score the held-out test borrowers.
    The leading underscore tells Streamlit not to hash the model argument."""
    borrowers = pd.read_csv(os.path.join(DATA_DIR, "borrower_profiles.csv"))
    from sklearn.model_selection import train_test_split
    # rebuild the exact 80/20 split used in training so the explorer only shows unseen borrowers
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
    # Same policy as the live Underwriter tab
    _verdict = decide(test_borrowers["predicted_pd"], test_borrowers["break_even_pd"],
                      test_borrowers["risk_band"], test_borrowers["rules_hit"])
    test_borrowers["policy_decision"] = pd.Series(_verdict, index=test_borrowers.index).map(
        {"DECLINE": "DECLINE", "CONDITIONS": "CONDITIONAL / REVIEW", "APPROVE": "APPROVE"})

    return borrowers, test_borrowers, card, overlay, item_defs, rules, scen, impact, ts_cmp, pm, macro, sources


model = load_model()
borrowers, test_borrowers, card, overlay, item_defs, rules, scen, impact, ts_cmp, pm, macro, sources = load_artifacts(model)
# macro scenario -> PD multiplier (neutral = 1.0, others come from the SARIMAX overlay)
multipliers = {"Neutral (through-the-cycle)": 1.0, **overlay["multipliers_next_12m"]}

# ============================================================================================== sidebar
st.sidebar.markdown("### IntelliCredit")
st.sidebar.caption("Loan decisions for gig workers with no credit-bureau history.")
st.sidebar.markdown("**What will the economy do next year?**")
scenario = st.sidebar.radio("Economy scenario", list(multipliers), index=0, format_func=lambda k: SCEN_LABELS.get(k, k),
                            label_visibility="collapsed",
                            help="Comes from the time-series forecast (notebook 06). A weaker economy makes every applicant riskier.")
st.sidebar.caption(SCEN_HELP[scenario])
apply_overlay = st.sidebar.toggle("Let the economy change the decision", value=True,
                                  help="Switch off to see the decision for a normal economy only.")
mult = multipliers[scenario] if apply_overlay else 1.0
st.sidebar.metric("Risk multiplier", f"{mult:.2f}x",
                  help="Every applicant's default chance is multiplied by this number. Above 1x means a riskier economy, below 1x a safer one.")
st.sidebar.caption("This setting applies to the **Score applicant** tab.")
with st.sidebar.expander("What do the risk grades mean?"):
    for g, rng in card["risk_bands"].items():
        st.markdown(f"**Grade {g}** · default chance {rng} · {BAND_MEANING[g]}")

# ============================================================================================== header
st.markdown(
    '<div class="masthead"><div class="mast-title">IntelliCredit</div>'
    '<div class="mast-sub">Loan decisions for gig workers with no credit-bureau history: a clear '
    'Approve, Approve with conditions or Decline, and exactly why.</div>'
    f'<div class="mast-meta">{len(borrowers):,} borrowers &nbsp;·&nbsp; {len(rules)} warning patterns &nbsp;·&nbsp; '
    'tested on 6,000 unseen borrowers &nbsp;·&nbsp; checked on real data</div></div>',
    unsafe_allow_html=True)

tab_start, tab_explore, tab_uw, tab_scen, tab_data, tab_r1, tab_rules, tab_proof, tab_uci = st.tabs([
    "Start here",
    "Data explorer",
    "Score applicant",
    "Scenario lab",
    "Test borrowers",
    "Review 1 models",
    "Time series & rules",
    "Proof & comparison",
    "Insights & Validation",
])

# Shared context handed to the tab modules (they each expose render(ctx))
from types import SimpleNamespace  # noqa: E402

_ctx = SimpleNamespace(
    explain=explain, step=step, style_fig=style_fig, html_table=html_table, decide=decide, risk_band=risk_band, pretty=pretty,
    ROOT=ROOT, DATA_DIR=DATA_DIR, MODELS_DIR=MODELS_DIR, LGD=LGD, COST_OF_FUNDS=COST_OF_FUNDS,
    SCEN_COLORS=SCEN_COLORS, SCEN_LABELS=SCEN_LABELS, item_defs=item_defs, scen=scen,
    borrowers=borrowers, test_borrowers=test_borrowers, pm=pm, macro=macro, impact=impact, rules=rules, card=card,
)

# ============================================================================================== TAB 0: Start here
with tab_start:
    st.markdown("## Welcome")
    explain("IntelliCredit helps a lender decide whether to give a <b>small loan to a gig worker</b> (a delivery rider, cab driver, freelancer) "
            "who has no credit score. It looks at how the person earns, saves and pays bills, then answers: "
            "<b>how likely is this person to miss payments, and can the lender still make money?</b>")
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="card"><div class="num">Part 1</div><h4>Estimate the risk</h4><p>A machine-learning model turns a person\'s income, bills, savings and loan details into a '
                '<b>chance of default</b>, then compares it with the highest risk the lender can profitably accept (the <b>break-even limit</b>).</p></div>', unsafe_allow_html=True)
    c2.markdown('<div class="card"><div class="num">Part 2</div><h4>Read the economy</h4><p>A time-series forecast predicts how many loans will go bad in a good year, a slowdown, or a '
                'COVID-style shock. In a weaker economy <b>every applicant gets riskier</b>, so the same rule becomes stricter.</p></div>', unsafe_allow_html=True)
    c3.markdown('<div class="card"><div class="num">Part 3</div><h4>Spot warning patterns</h4><p>93 easy-to-read rules such as <i>"income falling + income bumpy + low savings"</i> '
                'flag people who default <b>about three times</b> as often as average.</p></div>', unsafe_allow_html=True)

    st.markdown("### Try it in 60 seconds")
    explain("<b>1.</b> Open the <b>Score applicant</b> tab and pick <b>Customer 3</b>. It is declined, and the screen shows why.<br>"
            "<b>2.</b> Pick <b>Customer 2</b>. It is approved <i>with conditions</i>.<br>"
            "<b>3.</b> In the left sidebar choose <b>Severe downturn</b>. Customer 2 flips to <b>Decline</b>: the same person is a worse risk when the economy is bad.")

    explain("<b>Defending the project?</b> Open <b>Proof &amp; comparison</b>. It shows the method running on four real credit datasets, how realistic the made-up borrower data is "
            "(including where it falls short), and how the work compares with published studies and existing products. <b>Insights &amp; Validation</b> tests the synthetic data against the UCI credit-card data.")

    explain("<b>Want to explore or test ideas?</b> <b>Data explorer</b> lets you browse and filter every dataset. <b>Review 1 models</b> compares the first-round models. <b>Time series &amp; rules</b> summarises the two Review 2 methods. "
            "<b>Scenario lab</b> asks what happens to defaults, losses and decisions if the economy changes, and explains why.")

    st.markdown("### The three sample customers")
    html_table(pd.DataFrame([
        ["Customer 1", "Prime worker: 3 years on the platform, pays every bill on time, steady income", "Approve"],
        ["Customer 2", "Stressed worker: 8 months, some late bills, falling income, heavy buy-now-pay-later", "Approve with conditions (Decline in a severe downturn)"],
        ["Customer 3", "High-risk worker: 2 months, no bank account, very late bills, tiny savings", "Decline in every scenario"],
    ], columns=["Sample", "Who they are", "Expected result"]))

    st.markdown("### Plain-English glossary")
    html_table(pd.DataFrame([
        ["Default", "The borrower stops paying and the loan becomes 90 or more days late."],
        ["Chance of default (PD)", "The model's estimate that this borrower will default. 10% means about 1 in 10 similar borrowers default."],
        ["Risk grade A to E", "A summary of the default chance. A is the safest (under 4%), E the riskiest (16% or more)."],
        ["Break-even limit", "The highest default chance at which the interest earned still covers the expected loss. Above it, the lender loses money on average."],
        ["Expected loss", "Chance of default × loan amount × share of the loan that is never recovered (65%)."],
        ["Economy scenario", "A forecast of next year's economy. It multiplies every applicant's default chance."],
        ["Warning pattern (rule)", "A combination of behaviours, found in past data, that goes with a much higher default rate."],
        ["Lift", "How many times riskier a group is than the average borrower. Lift 3 means three times as risky."],
        ["Explanation chart (SHAP)", "Shows which facts about the applicant pushed the risk up (red) or down (green), and by how much."],
        ["Path to approval", "The smallest realistic changes (smaller loan, fewer late bills, more savings) that would turn a decline into an approval."],
        ["AUC", "How well the model ranks risky borrowers above safe ones. 0.5 is a coin flip, 1.0 is perfect. The best possible on this data is 0.781."],
        ["NPL", "Non-performing loan: a loan that is no longer being repaid. The NPL ratio is a bank's share of such loans."],
    ], columns=["Term", "What it means"]))

# ============================================================================================== TAB 1: Score an applicant
with tab_uw:
    @st.fragment
    def _score_tab():
        st.markdown("## Score an applicant")
        explain("Pick a person and change any detail. The decision on the right <b>updates instantly</b>. Scroll down for <b>why</b> and <b>what could change it</b>.")

        step(1, "Who is applying?")
        cust_keys = list(PRESETS.keys())
        quick_options = ["Customer 1 · Prime", "Customer 2 · Stressed", "Customer 3 · High-risk", "Custom · your own"]
        customer_sel = st.radio("Select an applicant", quick_options, index=0, horizontal=True, label_visibility="collapsed")

        if customer_sel == quick_options[0]:
            base = pd.Series(PRESETS[cust_keys[0]])
            st.markdown('<div class="who ok"><b>Expected: Approve.</b> Three years on the platform, pays all 12 bills on time, low buy-now-pay-later, income growing.</div>', unsafe_allow_html=True)
        elif customer_sel == quick_options[1]:
            base = pd.Series(PRESETS[cust_keys[1]])
            st.markdown('<div class="who cond"><b>Expected: Approve with conditions.</b> Eight months on the platform, about 6 days late on bills, higher buy-now-pay-later, income falling.</div>', unsafe_allow_html=True)
        elif customer_sel == quick_options[2]:
            base = pd.Series(PRESETS[cust_keys[2]])
            st.markdown('<div class="who no"><b>Expected: Decline.</b> Two months on the platform, no bank account, about 18 days late on bills, 75 warning patterns triggered.</div>', unsafe_allow_html=True)
        else:
            base = borrowers.drop(columns=["borrower_id"]).median(numeric_only=True)
            base["location_type"], base["education_level"] = "Tier-2 City", "Undergraduate"
            st.markdown('<div class="who custom"><b>Custom applicant.</b> Starts from the median borrower. Change any value below and the decision updates straight away.</div>', unsafe_allow_html=True)

        form_col, result_col = st.columns([5, 6.4], gap="large")
        form_slug = "".join(c if c.isalnum() else "_" for c in customer_sel).strip("_")
        _nonce = st.session_state.get("score_reset", 0)
        _k = lambda name: f"sc_{form_slug}_{_nonce}_{name}"

        with form_col:
            step(2, "Check the details")
            st.caption("The decision on the right updates the moment you change a value.")
            with st.container():
                a, b = st.columns(2)
                age = a.number_input("Age", 21, 60, int(base["age"]), key=_k("age"))
                tenure_gig = b.number_input("Months on gig platforms", 1, 120, int(base["gig_tenure_months"]), help="How long the person has been earning from gig work. Newer workers are riskier.", key=_k("tenure_gig"))
                loc_opts = ["Metro", "Tier-1 City", "Tier-2 City", "Semi-Urban", "Rural"]
                edu_opts = ["Below 10th", "10th Pass", "12th Pass", "Diploma", "Undergraduate", "Postgraduate"]
                location = a.selectbox("Location", loc_opts, index=loc_opts.index(base["location_type"]), key=_k("location"))
                education = b.selectbox("Education", edu_opts, index=edu_opts.index(base["education_level"]), key=_k("education"))
                income = a.number_input("Monthly gig income (₹)", 2000.0, 150000.0, float(base["avg_monthly_income"]), step=500.0, key=_k("income"))
                on_time = b.slider("Bills paid on time (of 12)", 0, 12, int(base["utility_on_time_count"]), help="Out of the last 12 utility bills, how many were paid on time.", key=_k("on_time"))
                amount = a.number_input("Loan amount asked (₹)", 5000.0, 300000.0, float(base["loan_amount_requested"]), step=1000.0, key=_k("amount"))
                tenure_opts = [3, 6, 9, 12, 18, 24]
                tenure = b.selectbox("Loan length (months)", tenure_opts, index=tenure_opts.index(int(base["loan_tenure_months"])), key=_k("tenure"))

                with st.expander("More details (filled in automatically; change only if you know them)"):
                    a1, a2 = st.columns(2)
                    dependents = a1.number_input("Dependents", 0, 5, int(base["num_dependents"]), key=_k("dependents"))
                    bank = a2.selectbox("Has a bank account", [1, 0], index=0 if int(base["has_bank_account"]) == 1 else 1, format_func=lambda v: "Yes" if v else "No", key=_k("bank"))
                    income_sd = a1.number_input("How much income varies (₹, std-dev)", 0.0, 80000.0, float(base["std_monthly_income"]), step=250.0, help="Bigger means a bumpier income from month to month.", key=_k("income_sd"))
                    trend = a2.number_input("Income trend (₹ per month)", -5000.0, 5000.0, float(base["income_trend_slope"]), step=50.0, help="Negative means income is falling.", key=_k("trend"))
                    avg_delay = a1.number_input("Average bill delay (days)", 0.0, 90.0, float(base["avg_utility_delay_days"]), step=0.5, key=_k("avg_delay"))
                    max_delay = a2.number_input("Worst bill delay (days)", 0, 90, int(base["max_utility_delay_days"]), key=_k("max_delay"))
                    recharge = a1.number_input("Days between phone recharges", 10.0, 60.0, float(base["avg_days_between_recharges"]), step=0.5, key=_k("recharge"))
                    bnpl = a2.number_input("Buy-now-pay-later spend (₹ / month)", 0.0, 60000.0, float(base["avg_bnpl_monthly_amount"]), step=250.0, key=_k("bnpl"))
                    outflow = a1.number_input("Monthly spending (₹)", 500.0, 160000.0, float(base["avg_monthly_outflow"]), step=500.0, key=_k("outflow"))
                    balance = a2.number_input("Month-end balance (₹)", 0.0, 100000.0, float(base["avg_eom_balance"]), step=250.0, key=_k("balance"))
                    logins = a1.number_input("App logins per month", 0.0, 90.0, float(base["avg_app_logins"]), step=0.5, key=_k("logins"))
                    upi = a2.number_input("UPI payments per month", 0.0, 80.0, float(base["avg_upi_transactions"]), step=0.5, key=_k("upi"))
                    rate = a1.number_input("Interest rate (% a year)", 8.0, 36.0, float(base["interest_rate_annual"]), step=0.25, key=_k("rate"))
                    prev = a2.number_input("Earlier micro-loans", 0, 15, int(base["previous_micro_loans"]), key=_k("prev"))
                    prev_ok = a1.number_input("…of which repaid on time", 0, 15, min(int(base["previous_loans_repaid_on_time"]), 15), key=_k("prev_ok"))

                if st.button("Reset to this applicant's original values", key=f"sc_reset_btn_{form_slug}"):
                    st.session_state["score_reset"] = _nonce + 1
                    st.rerun(scope="fragment")


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

        verdict = str(decide(pd_pit, be, band, len(hits)))
        if verdict == "DECLINE":
            decision, css, reason = "Decline", "v-no", (
                f"The chance of default ({pd_pit:.1%}) is above the break-even limit ({be:.1%}) for this interest rate and loan length, so the lender would lose money on average."
                if pd_pit >= be else
                f"Severe credit risk (risk grade {band}, default chance {pd_pit:.1%}) with {len(hits)} warning patterns triggered: automatic decline.")
        elif verdict == "CONDITIONS":
            decision, css, reason = "Approve with conditions", "v-cond", (
                f"Profitable on average, but {len(hits)} warning pattern(s) fire (risk grade {band}). Offer a smaller loan or weekly repayments.")
        else:
            decision, css, reason = "Approve", "v-ok", (
                f"The chance of default ({pd_pit:.1%}) is comfortably below the break-even limit ({be:.1%}), with no worrying behaviour patterns.")

        with result_col:
            step(3, "The decision")
            st.markdown(f'<div class="verdict {css}"><div class="v-label">Decision</div><div class="v-title">{decision}</div>'
                        f'<div class="v-sub">{reason}</div></div>', unsafe_allow_html=True)
            st.plotly_chart(risk_gauge(pd_pit, be), width="stretch", config={"displayModeBar": False})
            st.caption("The diamond is this applicant's chance of default. Left of the dashed line the lender makes money on average; right of it the lender loses. Letters are the risk grades.")
            m1, m2, m3 = st.columns(3)
            m1.metric("Chance of default", f"{pd_pit:.1%}", delta=f"{(mult - 1) * 100:+.0f}% vs normal", delta_color="inverse",
                      help="Chance this applicant defaults, after adjusting for the economy scenario chosen in the sidebar.")
            m2.metric("Normal economy", f"{pd_ttc:.1%}", help="The model's raw estimate before any economy adjustment.")
            m3.metric("Risk grade", f"{band}", delta=BAND_MEANING[band], delta_color="off", help="A is safest (under 4%), E is riskiest (16% or more).")
            m4, m5, m6 = st.columns(3)
            m4.metric("Break-even limit", f"{be:.1%}", help="Highest default chance the lender can profitably accept at this interest rate and loan length.")
            m5.metric("Expected loss", f"₹{expected_loss:,.0f}", help="Default chance × loan amount × 65% (the share of a defaulted loan not recovered).")
            m6.metric("Suggested rate", f"{suggested_rate:.1f}%", help=f"Interest rate the lender's price list suggests for risk grade {band}.")

            # --- which test decided it (so the verdict is never a black box)
            n_hits = len(hits)
            tests = [
                ("Profit test", "fail" if pd_pit >= be else "pass",
                 f"Default chance {pd_pit:.1%} is {'at or above' if pd_pit >= be else 'below'} the break-even limit {be:.1%} (at or above = Decline)."),
                ("Risk grade", "fail" if band == "E" else ("warn" if band in ("C", "D") else "pass"),
                 f"Grade {band} ({BAND_MEANING[band]}). Grade E = Decline; C or D = Approve with conditions."),
                ("Warning patterns", "fail" if n_hits >= 15 else ("warn" if n_hits >= 3 else "pass"),
                 f"{n_hits} pattern(s) fire. 15 or more = Decline; 3 or more = Approve with conditions."),
            ]
            icon = {"pass": "✅", "warn": "⚠️", "fail": "⛔"}
            st.markdown("**How the decision was reached**")
            st.markdown("\n".join(f"- {icon[s]} **{n}:** {t}" for n, s, t in tests))
            st.caption("The worst result of the three tests becomes the decision. This is why changing one detail sometimes does not change the decision: "
                       "the applicant has to move across a threshold above, not just move a little.")

        # -------------------------------------------------------------------- same applicant, every economy
        st.markdown("### Same applicant, different economies")
        explain("The same person can get a different answer when the economy changes. This is what the sidebar switch does.")
        rows = []
        for k, m_ in multipliers.items():
            p_ = min(pd_ttc * m_, 0.99); b_ = risk_band(p_)
            d_ = str(decide(p_, be, b_, len(hits)))
            rows.append({"Economy": SCEN_LABELS.get(k, k), "Risk multiplier": f"{m_:.2f}x", "Default chance": f"{p_:.1%}", "Risk grade": b_,
                         "Decision": {"DECLINE": "Decline", "CONDITIONS": "Approve with conditions", "APPROVE": "Approve"}[d_]})
        _sty = pd.DataFrame(rows).style
        _sty = (_sty.map if hasattr(_sty, "map") else _sty.applymap)(colour_decision, subset=["Decision"])
        st.dataframe(_sty, hide_index=True, width="stretch")

        # -------------------------------------------------------------------- why
        st.markdown("### Why this decision?")
        left, right = st.columns([1.15, 1], gap="large")
        with left:
            st.markdown("**What pushed the risk up or down**")
            st.caption("Red bars raise the chance of default, green bars lower it. A longer bar means a bigger effect. (TreeSHAP explanation of the model.)")
            contrib = model.predict(X, pred_contrib=True)[0]   # TreeSHAP values in log-odds; last column = base value
            contrib = pd.Series(contrib[:-1], index=X.columns)
            top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(10)[::-1]
            fig = go.Figure(go.Bar(x=top.values, y=[FEATURE_LABELS.get(c, c) for c in top.index], orientation="h",
                                   marker_color=["#9b2c2c" if v > 0 else "#3d6b4f" for v in top.values],
                                   hovertemplate="%{y}: %{x:+.3f}<extra></extra>"))
            fig.update_layout(xaxis_title="Lowers risk (left)  ·  Raises risk (right)")
            style_fig(fig, 400)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        with right:
            st.markdown(f"**Warning patterns found: {len(hits)}**")
            st.caption("Behaviour combinations that, in past data, went with many more defaults than average.")
            present = [i["item"] for i in item_defs["items"] if flags[i["item"]]]
            if present:
                st.markdown("".join(f'<span class="chip">{p.replace("_", " ").title()}</span>' for p in present), unsafe_allow_html=True)
            else:
                st.markdown('<span class="chip none">No warning signs</span>', unsafe_allow_html=True)
            if len(hits):
                show = hits[["antecedent", "test_confidence", "test_lift"]].head(8).copy()
                show["antecedent"] = show["antecedent"].map(pretty)
                show = show.rename(columns={"antecedent": "Pattern", "test_confidence": "Default rate among similar people", "test_lift": "Times riskier than average"})
                st.dataframe(show.style.format({"Default rate among similar people": "{:.1%}", "Times riskier than average": "{:.2f}x"}), hide_index=True, width="stretch")
            else:
                st.success("No validated warning pattern matches this applicant.")

        # -------------------------------------------------------------------- recourse
        st.markdown("### What could change the outcome?")
        if pd_pit >= be:
            with st.spinner("Searching for realistic changes…"):
                recourse = path_to_approval(model, raw.iloc[0], multiplier=mult)
            if recourse.empty:
                st.warning("No realistic change (smaller loan, other loan length, less buy-now-pay-later, better bill payment, more savings) brings this applicant "
                           "below the break-even limit. Refer to a secured or group-lending product instead.")
            else:
                explain("These are the <b>cheapest different ways</b> this applicant could become profitable for the lender. Only things the applicant can change are used: "
                        "age, location, education and the interest rate are never changed.")
                st.dataframe(recourse.style.format({"new PD": "{:.1%}", "break-even PD": "{:.1%}"}), hide_index=True, width="stretch")
                st.caption("Counterfactual explanations (Wachter et al. 2017) restricted to actionable features (Ustun et al. 2019) and kept diverse (Mothilal et al. 2020). Code: src/recourse.py")
        else:
            st.success(f"Nothing to change: the default chance ({pd_pit:.1%}) is already {100 * (be - pd_pit):.1f} points below the break-even limit. "
                       "Pick **Severe downturn** in the sidebar to see this section work.")

    _score_tab()

# ============================================================================================== TAB 2: Test borrowers
with tab_data:
    st.markdown("## Test borrowers")
    explain("These are <b>6,000 borrowers the model never saw while learning</b>, so they show how it performs on new people. "
            "Filter them, and compare the model's predicted risk with what <b>actually happened</b>.")
    with st.container(border=True):
        f1, f2, f3, f4, f5 = st.columns(5)
        band_filter = f1.multiselect("Risk grade", ["A", "B", "C", "D", "E"], default=["A", "B", "C", "D", "E"])
        all_locs = sorted(test_borrowers["location_type"].dropna().unique())
        loc_filter = f2.multiselect("Location", all_locs, default=all_locs)
        all_plats = sorted(test_borrowers["primary_gig_platform"].dropna().unique())
        plat_filter = f3.multiselect("Gig platform", all_plats, default=all_plats)
        status_choice = f4.selectbox("What actually happened", ["All", "Repaid (0)", "Defaulted (1)"])
        min_rules_hit = f5.slider("Warning patterns hit (at least)", 0, 10, 0)

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
        st.info("No borrowers match this combination of filters. Loosen a filter to see results.")
    else:
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Borrowers selected", f"{len(filtered):,}")
        k2.metric("Predicted default rate", f"{filtered['predicted_pd'].mean():.2%}", help="Average of the model's predicted default chances.")
        k3.metric("Actual default rate", f"{filtered['loan_default_status'].mean():.2%}", help="Share that really defaulted. When this is close to the predicted rate, the model is well calibrated.")
        k4.metric("Total loans asked", f"₹{filtered['loan_amount_requested'].sum() / 1e7:,.1f} Cr", help="Crore (Cr) = 10 million rupees.")
        high_risk_cnt = (filtered["rules_hit"] >= 3).sum()
        k5.metric("Hit 3+ patterns", f"{high_risk_cnt:,}", delta=f"{(high_risk_cnt / len(filtered) * 100):.1f}% of selection", delta_color="off",
                  help="Counts all 93 overlapping rules, so this is a loose threshold (53% of all test applicants). The report's 14% / 34% figures use a top-25 rule score instead.")

        c_chart1, c_chart2 = st.columns([1.2, 1], gap="large")
        with c_chart1:
            fig_dist = px.histogram(filtered, x="predicted_pd", color="loan_default_status", nbins=30, barmode="overlay",
                                    color_discrete_map={0: "#5b8a6b", 1: "#9b2c2c"},
                                    labels={"predicted_pd": "Predicted chance of default", "loan_default_status": "Defaulted?"},
                                    title="Defaulters (red) sit further right than repayers (green)")
            fig_dist.update_layout(xaxis_tickformat=".0%", yaxis_title="Borrowers")
            st.plotly_chart(style_fig(fig_dist, 330), width="stretch", config={"displayModeBar": False})
        with c_chart2:
            band_summary = filtered.groupby("risk_band", observed=True).agg(borrowers=("borrower_id", "count"), default_rate=("loan_default_status", "mean")).reset_index()
            fig_band = px.bar(band_summary, x="risk_band", y="default_rate", color="risk_band", color_discrete_map=BAND_COLORS,
                              labels={"risk_band": "Risk grade", "default_rate": "Actual default rate"}, title="Riskier grades really do default more")
            fig_band.update_layout(yaxis_tickformat=".0%", showlegend=False)
            st.plotly_chart(style_fig(fig_band, 330), width="stretch", config={"displayModeBar": False})

        st.markdown(f"### Borrowers ({len(filtered):,})")
        display_cols = ["borrower_id", "primary_gig_platform", "location_type", "avg_monthly_income", "loan_amount_requested", "loan_tenure_months",
                        "predicted_pd", "risk_band", "rules_hit", "policy_decision", "loan_default_status"]
        rename_map = {"borrower_id": "Borrower", "primary_gig_platform": "Platform", "location_type": "Location", "avg_monthly_income": "Income (₹)",
                      "loan_amount_requested": "Loan (₹)", "loan_tenure_months": "Months", "predicted_pd": "Predicted default chance", "risk_band": "Grade",
                      "rules_hit": "Patterns hit", "policy_decision": "Decision", "loan_default_status": "Actually defaulted"}
        table = filtered[display_cols].rename(columns=rename_map)
        st.dataframe(table.style.format({"Income (₹)": "{:,.0f}", "Loan (₹)": "{:,.0f}", "Predicted default chance": "{:.2%}", "Actually defaulted": "{:d}"}),
                     height=350, width="stretch")
        st.download_button("Download these borrowers as CSV", table.to_csv(index=False).encode("utf-8"), "intellicredit_test_borrowers.csv", "text/csv")

# ============================================================================================== Warning-pattern explorer (shown in an expander on the Time series & rules tab)
def _warning_explorer():
    explain("A warning pattern is a <b>combination of behaviours</b> that went with far more defaults than average, for example "
            "<i>declining income + bumpy income + low savings</i>. They were mined from 24,000 borrowers and then <b>checked on 6,000 different borrowers</b> "
            "(Apriori / FP-Growth, notebook 07). Higher on the chart means riskier; further right means it affects more people.")
    f1, f2, f3 = st.columns(3)
    min_lift = f1.slider("Show patterns at least this many times riskier than average", 1.0, 3.5, 1.5, 0.1)
    min_cov = f2.slider("…that match at least this share of applicants", 0.0, 0.10, 0.0, 0.005, format="%.3f")
    must = f3.multiselect("Must include this behaviour", sorted({i for items in rules["items"] for i in items}), format_func=lambda s: s.replace("_", " ").title())
    view = rules[(rules["test_lift"] >= min_lift) & (rules["test_coverage"] >= min_cov)]
    if must:
        view = view[view["items"].apply(lambda items: all(m in items for m in must))]
    st.markdown(f"Showing **{len(view)}** of {len(rules)} validated patterns")
    fig = px.scatter(view.assign(pattern=view["antecedent"].map(pretty)), x="test_coverage", y="test_confidence", size="support", color="test_lift", hover_name="pattern",
                     color_continuous_scale="Reds", labels={"test_coverage": "Share of unseen applicants matching", "test_confidence": "Default rate when the pattern matches",
                                                            "test_lift": "Times riskier"})
    fig.add_hline(y=item_defs["base_default_rate"], line_dash="dash", line_color="#1f3a5f", annotation_text="Average default rate")
    fig.update_layout(xaxis_tickformat=".1%", yaxis_tickformat=".0%")
    st.plotly_chart(style_fig(fig, 420), width="stretch", config={"displayModeBar": False})
    tbl = (view[["antecedent", "test_confidence", "test_lift", "test_coverage", "n_items", "confidence", "lift", "support"]]
           .sort_values("test_lift", ascending=False).copy())
    tbl["antecedent"] = tbl["antecedent"].map(pretty)
    tbl = tbl.rename(columns={"antecedent": "Warning pattern", "test_confidence": "Default rate (unseen)", "test_lift": "Times riskier (unseen)",
                              "test_coverage": "Share of unseen applicants", "n_items": "Signs combined", "confidence": "Default rate (training)",
                              "lift": "Times riskier (training)", "support": "Share of training borrowers"})
    st.dataframe(tbl.style.format({"Default rate (unseen)": "{:.1%}", "Times riskier (unseen)": "{:.2f}x", "Share of unseen applicants": "{:.2%}",
                                   "Default rate (training)": "{:.1%}", "Times riskier (training)": "{:.2f}x", "Share of training borrowers": "{:.2%}"}),
                 hide_index=True, width="stretch", height=420)

# ============================================================================================== Real-data results loader (used by the Proof & comparison tab)
REAL_DIR = os.path.join(DATA_DIR, "real")
REAL_OUT = os.path.join(MODELS_DIR, "real")


@st.cache_data
def load_real():
    with open(os.path.join(REAL_OUT, "real_data_results.json"), encoding="utf-8") as f:
        res = json.load(f)
    with open(os.path.join(REAL_DIR, "real_sources.json"), encoding="utf-8") as f:
        srcs = json.load(f)
    npl = pd.read_csv(os.path.join(REAL_DIR, "india_bank_npl_quarterly.csv"), parse_dates=["quarter_start"])
    fc = pd.read_csv(os.path.join(REAL_OUT, "india_npl_test_forecasts.csv"), index_col=0, parse_dates=True)
    us_bt = pd.read_csv(os.path.join(REAL_OUT, "us_delinquency_rolling_forecasts.csv"), index_col=0, parse_dates=True)
    hr = pd.read_csv(os.path.join(REAL_OUT, "hmda_denial_rules.csv"))
    return res, srcs, npl, fc, us_bt, hr


# ============================================================================================== TAB 7: Proof & comparison
with tab_proof:
    if os.path.exists(os.path.join(REAL_OUT, "real_data_results.json")):
        _res, _srcs, _npl, *_rest = load_real()
        render_proof(ROOT, explain, style_fig, html_table, pm, _res, _srcs, _npl)
    else:
        st.warning("Run `python -m src.real_data` and notebook 09 first.")


# ============================================================================================== Data explorer, Scenario lab, Review 1 models, Insights & Validation
with tab_explore:
    render_explorer(_ctx)
with tab_scen:
    render_scenario(_ctx)
with tab_r1:
    render_review1(_ctx)
with tab_rules:
    render_methods(_ctx)
    with st.expander("Explore all 93 warning patterns (interactive)"):
        _warning_explorer()
with tab_uci:
    render_uci(_ctx)
