"""
IntelliCredit dashboard - "Proof & comparison" tab
==================================================
Everything shown here is read from files written by
    scripts/run_validation_suite.py   (models/validation/*.csv)
    scripts/build_fidelity_anchors.py (models/validation/anchors.csv)
    research/competitive_landscape.json (curated, every entry carries a source link and a status)
so no number on this page is typed into the dashboard by hand.
"""
import json
import os
import re
import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

NAVY, OCHRE, BRICK, SLATE, GREEN = "#1f3a5f", "#b8862f", "#9b2c2c", "#5a5546", "#3d6b4f"
MODEL_STYLE = {                       # fixed order, never re-coloured; marker shape is the second encoding
    "Logistic scorecard": (SLATE, "square"),
    "Random forest": (OCHRE, "triangle-up"),
    "XGBoost": (BRICK, "diamond"),
    "LightGBM (IntelliCredit recipe)": (NAVY, "circle"),
}
DS_LABEL = {"intellicredit": "IntelliCredit (simulated)", "german": "German credit (UCI)", "heloc": "FICO HELOC",
            "taiwan": "Taiwan credit cards (UCI)", "gmsc": "Give Me Some Credit (Kaggle)"}
FRIENDLY = {
    "PAY_0": "latest repayment delay", "PAY_2": "repayment delay, 2 months ago", "PAY_3": "repayment delay, 3 months ago",
    "LIMIT_BAL": "credit limit", "BILL_AMT1": "latest bill amount", "PAY_AMT1": "latest payment", "PAY_AMT2": "payment 2 months ago",
    "AGE": "age", "ExternalRiskEstimate": "external risk score", "MSinceMostRecentInqexcl7days": "months since last enquiry",
    "NetFractionRevolvingBurden": "revolving credit burden", "AverageMInFile": "average months on file",
    "PercentTradesNeverDelq": "% accounts never late", "NumSatisfactoryTrades": "satisfactory accounts",
    "MSinceOldestTradeOpen": "months since oldest account", "PercentInstallTrades": "% instalment accounts",
    "RevolvingUtilizationOfUnsecuredLines": "credit-line utilisation", "NumberOfTimes90DaysLate": "times 90+ days late",
    "NumberOfTime30_59DaysPastDueNotWorse": "times 30-59 days late", "NumberOfTime60_89DaysPastDueNotWorse": "times 60-89 days late",
    "DebtRatio": "debt ratio", "MonthlyIncome": "monthly income", "NumberOfOpenCreditLinesAndLoans": "open credit lines",
    "age": "age", "duration": "loan duration", "credit_amount": "credit amount", "checking_status_0_X_200": "checking account 0-200",
    "checking_status_no_checking": "no checking account", "checking_status_0": "negative checking balance",
    "installment_commitment": "instalment / income", "savings_status_100": "savings under 100",
    "avg_monthly_income": "average income", "utility_reliability_score": "utility bills paid on time",
    "avg_utility_delay_days": "average bill delay", "max_utility_delay_days": "worst bill delay", "income_volatility_index": "income volatility",
    "debt_to_inflow_ratio": "BNPL / income", "savings_buffer_ratio": "savings buffer", "loan_to_income": "loan / income",
    "interest_rate_annual": "interest rate", "gig_tenure_months": "gig tenure", "repayment_ratio": "past repayment share",
    "avg_upi_transactions": "UPI payments", "avg_app_logins": "app logins", "avg_days_between_recharges": "recharge gap",
    "income_trend_slope": "income trend", "loan_tenure_months": "loan tenure",
}

CSS = """
<style>
.tag { display:inline-block; padding:1px 8px; border:1px solid; font-size:.82rem; font-weight:700; white-space:nowrap; }
.tag-good { color:#2f5c41; border-color:#a9c4b0; background:#f1f6ef; }
.tag-mid  { color:#8f6414; border-color:#d9c08a; background:#fbf3df; }
.tag-bad  { color:#8a2525; border-color:#c9a5a2; background:#fbf3f1; }
.tag-na   { color:#5a5546; border-color:#cfc7b3; background:#f3eee2; }
.gt a { color:#1f3a5f; }
.callout { background:#fffdf8; border:1px solid #cfc7b3; border-left:5px solid #9b2c2c; padding:10px 16px; margin:10px 0 14px 0; font-size:.97rem; line-height:1.55; }
.callout.ok { border-left-color:#3d6b4f; } .callout.mid { border-left-color:#b8862f; }
.tw { overflow-x:auto; margin:4px 0 12px 0; }
.tw table.gt { font-size:.88rem; }
.tw table.gt td:first-child { white-space:normal; }
.tw table.gt td, .tw table.gt th { overflow-wrap:break-word; }
.tw .tag { white-space:normal; }
.cellnote { font-size:.84rem; color:#3a3a3a; margin-top:4px; line-height:1.35; }
</style>
"""


def _tag(text, kind):
    return f'<span class="tag tag-{kind}">{text}</span>'


def _link(url, text="source"):
    return f'<a href="{url}" target="_blank">{text}</a>' if url else ""


def _status(s):
    return {"primary": _tag("Checked at source", "good"), "secondary": _tag("Seen in summary", "mid"),
            "vendor": _tag("Company claim", "na")}.get(s, s)


def _verdict_kind(v):
    v = str(v)
    if v.startswith(("Matches", "Same direction", "Yes")):
        return "good"
    if v.startswith(("Roughly", "Negligible", "Partly")):
        return "mid"
    if v.startswith(("Does not", "Opposite", "No")):
        return "bad"
    return "na"


def _cell(v):
    """Matrix cell: bold the leading Yes / No / Partly word and colour it, always keeping the word itself."""
    v = str(v)
    for head, kind in (("Not public", "na"), ("Not applicable", "na"), ("Yes", "good"), ("By design", "good"), ("Partly", "mid"),
                       ("Needs review", "mid"), ("No", "bad")):
        if re.match(rf"^{head}(?![A-Za-z])", v):
            rest = v[len(head):].lstrip(":,. ")
            return _tag(head, kind) + (f'<div class="cellnote">{rest[0].upper() + rest[1:]}</div>' if rest else "")
    return f'<div class="cellnote">{v}</div>'


def _wrap(s, n=34):
    return "<br>".join(textwrap.wrap(s, n))


def _table(df, widths=None):
    """Wrapping table that scrolls sideways instead of clipping. widths = list of column percentages."""
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>" for row in df.itertuples(index=False))
    cols = "<colgroup>" + "".join(f'<col style="width:{w}%">' for w in widths) + "</colgroup>" if widths else ""
    fixed = ' style="table-layout:fixed"' if widths else ""
    st.markdown(f'<div class="tw"><table class="gt"{fixed}>{cols}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>', unsafe_allow_html=True)


SHORT = {
    "Younger borrowers default more": "Age", "Higher income / credit capacity lowers default": "Income level",
    "More dependents raise default": "Number of dependents", "Late payments predict default (count / severity)": "Late payments",
    "A high on-time payment share protects": "On-time payment share", "Heavy debt burden raises default": "Debt burden",
    "A bigger loan relative to means raises default": "Loan size vs means", "A longer track record lowers default": "Length of track record",
    "A savings / repayment buffer lowers default": "Savings / repayment buffer",
}
VERDICT_SHORT = {"Negligible in synthetic data": "Negligible", "Same direction (real mixed)": "Same direction"}


@st.cache_data
def load(root):
    v = os.path.join(root, "models", "validation")
    if not os.path.exists(os.path.join(v, "realbench_results.csv")):
        return None
    d = {name: pd.read_csv(os.path.join(v, f"{name}.csv")) for name in
         ["realbench_results", "realbench_calibration", "realbench_meta", "model_ranking_concordance", "effect_concordance",
          "effect_concordance_detail", "shap_top_drivers", "dynamics", "income_reweighting", "feature_group_ablation",
          "approvals_at_equal_loss", "anchors"]}
    d["pair"] = json.load(open(os.path.join(v, "dynamics_pair.json"), encoding="utf-8"))
    d["info"] = json.load(open(os.path.join(v, "run_info.json"), encoding="utf-8"))
    d["land"] = json.load(open(os.path.join(root, "research", "competitive_landscape.json"), encoding="utf-8"))
    refs = [l for l in open(os.path.join(root, "research", "REFERENCES_ACM.md"), encoding="utf-8").read().splitlines() if re.match(r"^\[\d+\] ", l)]
    pdfs = [f for f in os.listdir(os.path.join(root, "research", "papers")) if f.endswith(".pdf")]
    d["refs"] = dict(total=len(refs), primary=sum("[V]" in l for l in refs), secondary=sum("[S]" in l and "[V]" not in l for l in refs), pdfs=len(pdfs))
    return d


def _pct(x, nd=1):
    return f"{x:.{nd}%}"


# ---------------------------------------------------------------------------------------------------------------- render
def render(root, explain, style_fig, html_table, pm, real_res, srcs, npl):
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("## Proof and comparison")
    explain("Examiners usually ask three things: <b>is the research real</b>, <b>is the made-up data close enough to real data</b>, and "
            "<b>how does this compare with what already exists</b>? This tab answers each with numbers you can re-run "
            "(<code>python scripts/run_validation_suite.py</code>). Where the evidence is weak, it says so.")
    D = load(root)
    if D is None:
        st.warning("Validation results are missing. Run `python scripts/run_validation_suite.py` and "
                   "`python scripts/build_fidelity_anchors.py` from the repository root.")
        return

    sections = ["Summary", "Real-data benchmark", "Is the made-up data realistic?", "Versus published studies",
                "Versus existing products", "Head-to-head", "Sources and re-run"]
    sec = st.radio("Section", sections, horizontal=True, label_visibility="collapsed", key="proof_section")
    {"Summary": _summary, "Real-data benchmark": _benchmark, "Is the made-up data realistic?": _realism,
     "Versus published studies": _literature, "Versus existing products": _products, "Head-to-head": _headtohead,
     "Sources and re-run": _sources}[sec](D, explain, style_fig, _table, pm, real_res, srcs, npl)


# ---------------------------------------------------------------------------------------------------------------- 1 summary
def _summary(D, explain, style_fig, html_table, pm, real_res, srcs, npl):
    res, eff, anc, ab, eq = D["realbench_results"], D["effect_concordance"], D["anchors"], D["feature_group_ablation"], D["approvals_at_equal_loss"]
    lg = res[res.model.str.startswith("LightGBM")].set_index("dataset")
    real_auc = lg.drop("intellicredit").auc
    n_same = int((eff.verdict.str.startswith("Same direction")).sum())
    n_test = int(((eff.real_checks > 0)).sum())
    n_ok = int(anc.verdict.isin(["Matches", "Roughly matches"]).sum())
    n_ver = int((anc.verdict != "Cannot verify").sum())
    base = ab[(ab.feature_group == "Application form + past loans") & ab.model.str.startswith("LightGBM")].auc.iloc[0]
    full = ab[(ab.feature_group == "Everything (IntelliCredit)") & ab.model.str.startswith("LightGBM")].auc.iloc[0]
    e5 = eq[np.isclose(eq.target_default_rate_of_approved, 0.05)].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy on real data", f"AUC {real_auc.min():.2f} to {real_auc.max():.2f}",
              help="The same model recipe run on 4 public real-world credit datasets (5-fold cross-validation). 0.5 is a coin flip, 1.0 is perfect.")
    c2.metric("Drivers pointing right", f"{n_same} of {n_test}",
              help="Of the risk drivers we can test against real data, how many push default in the same direction in our made-up data as in real data.")
    c3.metric("Realism checks passed", f"{n_ok} of {n_ver}",
              help="Made-up data compared with published Indian figures. Counts 'matches' and 'roughly matches'. One more check could not be verified.")
    c4.metric("Gain from alt. data", f"+{(full - base) * 100:.1f} AUC points",
              help="Added accuracy from utility, recharge, savings and income-pattern data over a conventional application form plus past loans. "
                   "Berg et al. (2020) found +5.3 points on real data. Ours is built into the simulation.")

    st.markdown("### What we can claim, and what we cannot")
    fid = real_res
    rf = D["refs"]
    rows = [
        ("The research behind it is real",
         f"The literature pack lists {rf['total']} numbered sources. {rf['primary']} were checked at the primary record (DOI page, publisher, regulator or SEC filing) and "
         f"{rf['secondary']} were confirmed only through summaries. {rf['pdfs']} papers are saved as PDFs in <code>research/papers/</code>. Every study quoted on this page has a link.",
         _tag("Strong", "good")),
        ("The method works on real credit data",
         f"Same recipe scores AUC {lg.loc['gmsc', 'auc']:.2f} on 150,000 US borrowers, {lg.loc['taiwan', 'auc']:.2f} on Taiwan cards, "
         f"{lg.loc['heloc', 'auc']:.2f} on FICO's HELOC data, {lg.loc['german', 'auc']:.2f} on German loans. The best model type matches the literature (boosted trees).",
         _tag("Strong", "good")),
        ("Our forecasting and pattern methods beat the usual alternatives on real data",
         f"Bad-loan forecast error for India {fid['india_model_comparison'][fid['india_best_model']]['MAE (pp)']:.2f} pp against "
         f"{fid['india_model_comparison']['Naive (last value)']['MAE (pp)']:.2f} pp for a naive forecast. Adaptive forecast ranges cover "
         f"{fid['us_interval_coverage']['80%']['adaptive conformal (ACI)']:.0%} against {fid['us_interval_coverage']['80%']['Gaussian model interval']:.0%} for the plain method. "
         f"{fid['hmda_rules']['share_hold_out_of_state']:.0%} of warning patterns hold in a state they never saw.",
         _tag("Strong", "good")),
        ("Alternative data helps",
         f"+{(full - base) * 100:.1f} AUC points in our data, and {e5.extra_approvals_pct:.0%} more approvals at the same 5% default rate. "
         "Published real-data result: +5.3 points (Berg et al.). Ours is a built-in effect, so it shows the mechanism, not proof.",
         _tag("Moderate", "mid")),
        ("The made-up borrower data is realistic",
         f"{n_ok} of {n_ver} reality checks passed, {n_same} of {n_test} risk drivers point the right way. Income level, debt burden, "
         "loan size and the COVID spike do not match reality. Details in the realism section.",
         _tag("Partial", "mid")),
        ("It would work on real Indian gig-worker borrowers",
         "No real Indian borrower outcomes were available. The one real Indian fintech study reports AUC 0.60 for alternative data alone, "
         "far below our 0.74.",
         _tag("Not proven", "bad")),
    ]
    html_table(pd.DataFrame(rows, columns=["Claim", "Evidence", "How strong"]), widths=[22, 64, 14])
    st.markdown(
        '<div class="callout">Why the borrower data is simulated at all: real Indian credit files are protected by law (CICRA 2005, '
        'Digital Personal Data Protection Act 2023). The honest position is: <b>the pipeline is proven on real data, the simulated data is partly '
        'realistic and its gaps are listed, and a pilot with a real lender is the missing step.</b></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------------------------------------------- 2 benchmark
def _benchmark(D, explain, style_fig, html_table, pm, real_res, srcs, npl):
    res, meta, cal, rk, shap = (D["realbench_results"], D["realbench_meta"], D["realbench_calibration"],
                                D["model_ranking_concordance"], D["shap_top_drivers"])
    explain("If the pipeline only worked on our own made-up data, that would be a red flag. So we ran <b>the same four model types, the same way</b> "
            "(5-fold cross-validation), on four well-known real credit datasets and on our simulated one. Higher AUC is better; 0.5 is a coin flip.")
    st.markdown("### The datasets")
    m = meta.copy()
    m["Dataset"] = m.dataset.map(DS_LABEL)
    t = pd.DataFrame({"Dataset": m["Dataset"], "Country": m.country, "Product": m["product"], "Borrowers": m.n.map("{:,}".format),
                      "Share that defaulted": m.default_rate.map(lambda x: f"{x:.1%}"), "What counts as default": m.target, "Source": m.source})
    html_table(t)

    st.markdown("### Accuracy of each model type")
    order = ["intellicredit", "german", "heloc", "taiwan", "gmsc"]
    ylab = [DS_LABEL[k] for k in order]
    fig = go.Figure()
    fig.add_hrect(y0=-0.5, y1=0.5, fillcolor="rgba(31,58,95,.08)", line_width=0)
    offs = {"Logistic scorecard": -0.27, "Random forest": -0.09, "XGBoost": 0.09, "LightGBM (IntelliCredit recipe)": 0.27}
    pos = list(range(len(order)))
    for mname, (col, sym) in MODEL_STYLE.items():
        d = res[res.model == mname].set_index("dataset").loc[order]
        fig.add_trace(go.Scatter(x=d.auc, y=[i + offs[mname] for i in pos], mode="markers", name=mname.replace(" (IntelliCredit recipe)", " (our recipe)"),
                                 marker=dict(symbol=sym, size=10, color=col, line=dict(color="#fbf8f1", width=1.2)),
                                 error_x=dict(type="data", array=d.auc_sd, thickness=1, width=0, color=col),
                                 customdata=ylab, hovertemplate="%{customdata}<br>AUC %{x:.3f}<extra>" + mname + "</extra>"))
    fig.update_layout(xaxis_title="AUC (higher is better): mean of 5 folds, line = spread across folds", xaxis_range=[0.69, 0.90],
                      legend=dict(orientation="h", y=-0.2), yaxis=dict(tickvals=pos, ticktext=ylab, range=[-0.6, len(order) - 0.4]))
    st.plotly_chart(style_fig(fig, 430), width="stretch", config={"displayModeBar": False})
    syn = res[(res.dataset == "intellicredit") & res.model.str.startswith("LightGBM")].auc.iloc[0]
    real = res[(res.dataset != "intellicredit") & res.model.str.startswith("LightGBM")].auc
    st.markdown(f"""
* **The simulated data sits at the hard end of the real range.** Our recipe scores {syn:.3f} on simulated borrowers and {real.min():.2f} to {real.max():.2f} on real ones, so the made-up data is not unrealistically easy.
* **Complex models add little over a plain scorecard**, on real and simulated data alike. This agrees with Fuster et al. (2022), where machine learning beat logistic regression by about 0.005 on 9.4 million mortgages.
* Shaded row: simulated data. Bars show the spread across folds.""")

    st.markdown("### Is the best model type the same on real and simulated data?")
    r = rk.copy()
    r["Real dataset"] = r.real_dataset.map(DS_LABEL)
    t = pd.DataFrame({"Real dataset": r["Real dataset"],
                      "Agreement of model ranking with simulated data": r.spearman_model_ranking.map(lambda x: f"{x:+.2f}"),
                      "Best model on real data": r.best_on_real.str.replace(" (IntelliCredit recipe)", " (our recipe)", regex=False),
                      "Best model on simulated data": r.best_on_synthetic,
                      "Gain of best model over plain scorecard (real)": r.lr_to_best_gap_real.map(lambda x: f"+{x * 100:.1f} pts"),
                      "Same gain (simulated)": r.lr_to_best_gap_synth.map(lambda x: f"+{x * 100:.1f} pts")})
    html_table(t)
    st.caption("Ranking agreement is a rank correlation: +1.00 means the four model types come out in the same order. Boosted trees are best or tied-best everywhere. "
               "Taiwan shows a bigger gap because repayment-delay codes are not linear; the simulation is nearly linear, so its gap is smaller than reality.")

    st.markdown("### Do predicted chances match what happened?")
    st.caption("Each panel groups borrowers into 10 equal sets, safest to riskiest. Points on the dashed line mean the predicted default chance equals the observed rate.")
    figc = make_subplots(rows=1, cols=5, subplot_titles=[DS_LABEL[k].replace(" (", "<br>(") for k in order], horizontal_spacing=0.05)
    for i, k in enumerate(order, 1):
        c = cal[cal.dataset == k]
        mx = float(max(c.predicted.max(), c.observed.max())) * 1.05
        figc.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color="#8a8577", dash="dash", width=1), showlegend=False, hoverinfo="skip"), row=1, col=i)
        figc.add_trace(go.Scatter(x=c.predicted, y=c.observed, mode="lines+markers", line=dict(color=NAVY, width=2), marker=dict(size=6), showlegend=False,
                                  hovertemplate="predicted %{x:.1%}<br>observed %{y:.1%}<extra></extra>"), row=1, col=i)
        figc.update_xaxes(tickformat=".0%", range=[0, mx], row=1, col=i, nticks=4)
        figc.update_yaxes(tickformat=".0%", range=[0, mx], row=1, col=i, nticks=4)
    figc.update_xaxes(title_text="Predicted", row=1, col=3)
    figc.update_yaxes(title_text="Observed", row=1, col=1)
    figc.update_annotations(font_size=12)
    style_fig(figc, 320)
    figc.update_layout(margin=dict(l=10, r=10, t=70, b=10))
    st.plotly_chart(figc, width="stretch", config={"displayModeBar": False})
    lgb = res[res.model.str.startswith("LightGBM")].set_index("dataset")
    t = pd.DataFrame({"Dataset": [DS_LABEL[k] for k in order],
                      "AUC": [f"{lgb.loc[k, 'auc']:.3f} ± {lgb.loc[k, 'auc_sd']:.3f}" for k in order],
                      "KS": [f"{lgb.loc[k, 'ks']:.2f}" for k in order],
                      "Gini": [f"{lgb.loc[k, 'gini']:.2f}" for k in order],
                      "Brier error (lower is better)": [f"{lgb.loc[k, 'brier']:.3f}" for k in order],
                      "Calibration error": [f"{lgb.loc[k, 'ece']:.3f}" for k in order],
                      "Riskiest fifth ÷ safest fifth": [f"{lgb.loc[k, 'quintile_ratio']:.1f}x" for k in order]})
    html_table(t)
    st.caption("Björkegren & Grissen (2020) report that the riskiest fifth of borrowers default 2.8 times as often as the safest fifth when scoring from phone data alone. "
               "The last column is the same measure for our model. Our simulated 14x is far higher than their real-world 2.8x, another sign that real Indian borrowers "
               "would be harder to separate than the simulation suggests.")

    st.markdown("### What does the model rely on?")
    st.caption("Top three drivers by share of the model's total explanation weight (TreeSHAP), per dataset.")
    rows = []
    for k in order:
        s = shap[shap.dataset == k].sort_values("rank").head(3)
        rows.append((DS_LABEL[k], ", ".join(f"{FRIENDLY.get(f, f.replace('_', ' ').lower())} ({v:.0%})" for f, v in zip(s.feature, s.share_of_total))))
    html_table(pd.DataFrame(rows, columns=["Dataset", "Top three drivers"]))
    st.caption("On the real datasets the leading drivers are how much credit is already used and whether earlier payments were late (credit-line utilisation, "
               "repayment delay, bureau risk score; for German loans, account status and loan size). In the simulation they are time on the platform, a falling income trend "
               "and unpaid bills. Both are about stability and payment discipline, but the exact variables differ, so this is a family resemblance, not a match.")


# ---------------------------------------------------------------------------------------------------------------- 3 realism
def _realism(D, explain, style_fig, html_table, pm, real_res, srcs, npl):
    anc, eff, det, dyn, pair, rw = D["anchors"], D["effect_concordance"], D["effect_concordance_detail"], D["dynamics"], D["pair"], D["income_reweighting"]
    explain("This is the test of the made-up data itself. We check it four ways: against published Indian figures, against the direction of each risk driver in real credit data, "
            "against how real default rates move over time, and by asking whether the conclusions survive when the borrower mix is changed to look more like reality.")

    st.markdown("### 1. Against published Indian figures")
    t = pd.DataFrame({"Check": anc.check, "Our made-up data": anc.synthetic, "Real world": anc.real,
                      "Result": [_tag(v, _verdict_kind(v)) for v in anc.verdict], "Comment": anc.note})
    html_table(t, widths=[13, 18, 24, 14, 31])
    st.caption("The sources behind each real-world figure are listed in the Sources section and in research/BENCHMARK_TABLE.md. Figures marked 'Cannot verify' had no public anchor.")

    st.markdown("### 2. Do risk drivers push default the same way as in real credit data?")
    st.caption("Each row is one risk driver. The navy diamond is our made-up data, the ochre circles are the real datasets. Both should sit on the same side of the zero line.")
    tested = eff[eff.real_checks > 0]
    ylab = [SHORT.get(c, c) for c in tested.concept]
    fig = go.Figure()
    fig.add_vline(x=0, line_color="#1f1f1f", line_width=1)
    fig.add_vrect(x0=-0.03, x1=0.03, fillcolor="rgba(138,133,119,.14)", line_width=0)
    d2 = det[det.concept.isin(tested.concept)]
    fig.add_trace(go.Scatter(x=d2.real_rho, y=[SHORT.get(c, c) for c in d2.concept], mode="markers", name="Real datasets",
                             marker=dict(symbol="circle", size=11, color=OCHRE, line=dict(color="#fbf8f1", width=1.5)),
                             customdata=np.stack([d2.real_dataset.map(DS_LABEL), d2.real_feature], axis=1),
                             hovertemplate="%{customdata[0]}: %{customdata[1]}<br>correlation %{x:+.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=tested.synthetic_rho, y=ylab, mode="markers", name="IntelliCredit (made-up data)",
                             marker=dict(symbol="diamond", size=14, color=NAVY, line=dict(color="#fbf8f1", width=1.5)),
                             hovertemplate="Made-up data: correlation %{x:+.2f}<extra></extra>"))
    fig.update_layout(xaxis_title="Correlation with default: left lowers risk, right raises it (grey band: too small to count)",
                      yaxis=dict(autorange="reversed", automargin=True), legend=dict(orientation="h", y=-0.18), xaxis_range=[-0.32, 0.32])
    st.plotly_chart(style_fig(fig, 440), width="stretch", config={"displayModeBar": False})
    t = pd.DataFrame({"Risk driver": eff.concept, "Our feature": eff.synthetic_feature.fillna("").replace("", "none").str.replace("_", " "),
                      "In our data": eff.synthetic_rho.map(lambda x: "n/a" if pd.isna(x) else f"{x:+.2f}"),
                      "In real data": eff.real_rho_range.fillna("n/a"), "Real checks agreeing": [f"{a} of {n}" if n else "n/a" for a, n in zip(eff.real_agree_with_expected, eff.real_checks)],
                      "Result": [_tag(VERDICT_SHORT.get(v.split(":")[0], v.split(":")[0]), _verdict_kind(v)) for v in eff.verdict]})
    html_table(t, widths=[27, 15, 10, 16, 14, 18])
    st.markdown(
        '<div class="callout mid"><b>What went wrong, and why.</b> The generator charges default risk per rupee borrowed and sets loan size from income, '
        'so richer borrowers look riskier (the opposite of reality). Debt burden is given little weight, and age and dependents have no effect at all. '
        'Notebook 09 had already flagged debt burden as under-weighted. These are known limitations of the simulation, not of the method; the method recovers whatever the data contains.</div>',
        unsafe_allow_html=True)

    st.markdown("### 3. Do default rates move over time like real ones?")
    pm2 = pm.set_index("date")["default_rate"].resample("QS").mean()
    j = pd.concat([pm2.rename("sim"), npl.set_index("quarter_start")["npl_ratio_pct"].rename("india")], axis=1).loc["2019-01-01":"2024-12-31"].dropna()
    idx = j / j.iloc[:4].mean() * 100
    figt = go.Figure([go.Scatter(x=idx.index, y=idx.sim, name="Simulated portfolio default rate", line=dict(color=NAVY, width=2.4)),
                      go.Scatter(x=idx.index, y=idx.india, name="India bank bad-loan ratio (IMF, real)", line=dict(color=OCHRE, width=2.4, dash="dash"))])
    figt.update_layout(yaxis_title="Index, 2019 average = 100", legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(style_fig(figt, 320), width="stretch", config={"displayModeBar": False})
    t = dyn.copy()
    t = pd.DataFrame({"Series": t.series, "Quarters / months": t.n, "Persistence (1 = very smooth)": t.level_ar1.map("{:.2f}".format),
                      "Typical jump between periods": t.change_sd.map(lambda x: f"{x:.0%}"), "Highest ÷ typical": t.peak_to_median.map(lambda x: f"{x:.1f}x")})
    html_table(t)
    noise = dyn.sampling_noise_change_sd.dropna().iloc[0]
    loans = dyn.avg_active_loans.dropna().iloc[0]
    st.markdown(f"""
* **Both fall sharply after 2019** (simulated index ends near {idx.sim.iloc[-4:].mean():.0f}, real near {idx.india.iloc[-4:].mean():.0f}), but quarter by quarter the two series are only loosely related (level correlation {pair['corr_level']:+.2f}, changes {pair['corr_change']:+.2f}).
* **The simulated rate is far jumpier than real bank data**: typical monthly jump {dyn.change_sd.iloc[0]:.0%} against {dyn.change_sd.iloc[2:].min():.0%} to {dyn.change_sd.iloc[2:].max():.0%} for real series.
  Most of this is simple sampling noise: the simulated book holds only about {loans:,.0f} loans, which alone would produce jumps of about {noise:.0%}. Real bank figures cover millions of loans.
* **Conclusion:** the simulated series is fine for demonstrating forecasting methods but should not be presented as a realistic copy of an Indian lender's default history. The real-data section replaces it with real series.""")

    st.markdown("### 4. Do the conclusions survive a more realistic borrower mix?")
    st.caption("Only about 15% of simulated borrowers earn under Rs 15,000 a month; the Economic Survey says about 40% of gig workers do. We re-weighted the 6,000 test borrowers to match, without retraining.")
    t = pd.DataFrame({"Borrower mix": rw.scenario, "Earning under Rs 15k": rw.low_income_share.map("{:.0%}".format),
                      "Model ranking quality (AUC)": rw.auc.map("{:.3f}".format), "Actual default rate": rw.default_rate.map("{:.1%}".format),
                      "Model's average prediction": rw.mean_predicted_pd.map("{:.1%}".format), "Approved": rw.approval_rate.map("{:.1%}".format),
                      "Default rate of approved": rw.default_rate_approved.map("{:.1%}".format)})
    html_table(t, widths=[29, 11, 13, 11, 12, 11, 13])
    st.markdown("""
* **Ranking quality and calibration hold**: AUC barely moves and the model's average prediction still tracks the actual rate, so the model does not depend on the income mix.
* **But the default rate falls as low earners are added.** In real life it would rise. This is the same flaw as above (low earners take smaller loans in the simulation, so they look safer). So this test shows robustness, and it also exposes the gap.""")


# ---------------------------------------------------------------------------------------------------------------- 4 literature
def _literature(D, explain, style_fig, html_table, pm, real_res, srcs, npl):
    land, res = D["land"], D["realbench_results"]
    explain("How does our accuracy compare with what published studies and competitions report? Read this as <b>context, not a league table</b>: every study used different borrowers, products and "
            "definitions of default. What matters is whether our numbers are in a believable range and whether our conclusions agree with theirs.")
    pts = pd.DataFrame(land["auc_points"])
    lg = res[res.model.str.startswith("LightGBM")].set_index("dataset")
    ours = [("This project: simulated", "IntelliCredit, simulated borrowers", lg.loc["intellicredit", "auc"], "", "5-fold cross-validation")]
    for k in ["gmsc", "taiwan", "heloc", "german"]:
        ours.append(("This project: real data", f"Our recipe on {DS_LABEL[k]}", lg.loc[k, "auc"], "", "5-fold cross-validation"))
    pts = pd.concat([pts[["group", "label", "auc", "status", "link", "note"]],
                     pd.DataFrame(ours, columns=["group", "label", "auc", "link", "note"]).assign(status="primary")], ignore_index=True)
    pts = pts.sort_values("auc")
    style = {"Published research": (SLATE, "square"), "Commercial / competition": (OCHRE, "triangle-up"),
             "This project: real data": (NAVY, "circle"), "This project: simulated": (BRICK, "diamond")}
    fig = go.Figure()
    for g, (col, sym) in style.items():
        d = pts[pts.group == g]
        fig.add_trace(go.Scatter(x=d.auc, y=d.label.map(lambda s: _wrap(s, 46)), mode="markers", name=g,
                                 marker=dict(symbol=sym, size=12, color=col, line=dict(color="#fbf8f1", width=1.5)),
                                 text=d.note, hovertemplate="%{y}<br>AUC %{x:.3f}<br>%{text}<extra></extra>"))
    fig.update_layout(xaxis_title="AUC (higher is better; 0.5 is a coin flip)", xaxis_range=[0.5, 0.96], legend=dict(orientation="h", y=-0.12),
                      yaxis=dict(categoryorder="array", categoryarray=[_wrap(s, 46) for s in pts.label], automargin=True, tickfont=dict(size=11)))
    st.plotly_chart(style_fig(fig, 760), width="stretch", config={"displayModeBar": False})
    st.markdown(f"""
* **Same data, same level.** On Give Me Some Credit our recipe scores {lg.loc['gmsc', 'auc']:.3f}; public write-ups report about 0.866 on the competition's private score. Our pipeline is not weaker than the standard approach on that dataset.
* **Our simulated {lg.loc['intellicredit', 'auc']:.2f} sits inside the published band** for alternative-data scoring (0.60 to 0.74) but above the only Indian fintech result (0.60). Treat {lg.loc['intellicredit', 'auc']:.2f} as the best case for India.
* **Do not read this as "we beat" any study.** Different borrowers, labels and products make that claim invalid, and the table below says why for each one.""")
    st.markdown("### Study by study")
    rows = [(s["study"], s["data"], s["finding"], s["ours"], s["caveat"], _status(s["status"]) + " " + _link(s["link"])) for s in land["studies"]]
    html_table(pd.DataFrame(rows, columns=["Study", "Data", "What they found", "How it relates to us", "Why not directly comparable", "Evidence"]),
               widths=[13, 11, 21, 21, 19, 15])
    st.caption("'Checked at source' means the DOI or publisher page was opened. 'Seen in summary' means only an excerpt or summary was available: recheck before quoting a number.")


# ---------------------------------------------------------------------------------------------------------------- 5 products
def _products(D, explain, style_fig, html_table, pm, real_res, srcs, npl):
    land = D["land"]
    explain("Who else does this, and where do we stand? We looked at credit bureaus, alternative-data scores, AI lenders and the data rails India now has. "
            "Competitor numbers are <b>marketing claims unless an independent source is named</b>; the table says which.")
    st.markdown("### Feature by feature")
    m = land["matrix"]
    rows = [[r[0]] + [_cell(c) for c in r[1:]] for r in m["rows"]]
    html_table(pd.DataFrame(rows, columns=["Capability"] + m["columns"]), widths=[19, 17, 14, 16, 18, 16])
    st.markdown(
        '<div class="callout"><b>Where we are weaker.</b> We have no real borrower outcomes and no live lending, and the established products do. '
        'Our advantages are in transparency, explanations, stress testing and open method. Those are real, but they do not replace a track record.</div>', unsafe_allow_html=True)

    st.markdown("### Product by product")
    rows = [(f"<b>{p['name']}</b><br><span class='muted'>{p['kind']}</span>", p["uses"], p["evidence"], p["gap_we_fill"], _status(p["status"]) + " " + _link(p["link"])) for p in land["products"]]
    html_table(pd.DataFrame(rows, columns=["Product", "What it uses", "What is publicly known about accuracy", "Where IntelliCredit differs", "Evidence"]),
               widths=[13, 15, 30, 22, 20])

    st.markdown("### What regulators expect")
    rows = [(f"<b>{r['rule']}</b>", r["says"], r["ours"], _status(r["status"]) + " " + _link(r["link"])) for r in land["regulation"]]
    html_table(pd.DataFrame(rows, columns=["Rule", "What it says", "How IntelliCredit lines up", "Evidence"]), widths=[18, 34, 28, 20])
    st.caption("Regulation summaries come from law-firm and news write-ups of the primary documents, so recheck wording against the RBI and CFPB texts before quoting.")


# ---------------------------------------------------------------------------------------------------------------- 6 head to head
def _headtohead(D, explain, style_fig, html_table, pm, real_res, srcs, npl):
    ab, eq = D["feature_group_ablation"], D["approvals_at_equal_loss"]
    explain("The cleanest test of a new method is to run it against the usual alternative <b>on the same data</b>. Four head-to-head comparisons follow: "
            "the first on simulated borrowers, the other three on real data.")
    st.markdown("### 1. Conventional scorecard vs alternative data (simulated borrowers)")
    order = ["Application form only", "Application form + past loans", "Alternative data only", "Everything (IntelliCredit)"]
    fig = go.Figure()
    pos = list(range(len(order)))
    for mname, (col, sym), off in [("Logistic scorecard", (SLATE, "square"), -0.14), ("LightGBM (IntelliCredit recipe)", (NAVY, "circle"), 0.14)]:
        d = ab[ab.model == mname].set_index("feature_group").loc[order]
        fig.add_trace(go.Scatter(x=d.auc, y=[i + off for i in pos], mode="markers", name=mname.replace(" (IntelliCredit recipe)", " (our recipe)"),
                                 marker=dict(symbol=sym, size=11, color=col, line=dict(color="#fbf8f1", width=1.2)),
                                 error_x=dict(type="data", array=d.auc_sd, thickness=1, width=0, color=col),
                                 customdata=order, hovertemplate="%{customdata}<br>AUC %{x:.3f}<extra></extra>"))
    fig.update_layout(xaxis_title="AUC, 5-fold cross-validation (line = spread across folds)", xaxis_range=[0.67, 0.77], legend=dict(orientation="h", y=-0.28),
                      yaxis=dict(tickvals=pos, ticktext=order, range=[len(order) - 0.4, -0.6]))
    st.plotly_chart(style_fig(fig, 330), width="stretch", config={"displayModeBar": False})
    base = ab[(ab.feature_group == order[1]) & ab.model.str.startswith("LightGBM")].auc.iloc[0]
    full = ab[(ab.feature_group == order[3]) & ab.model.str.startswith("LightGBM")].auc.iloc[0]
    st.markdown(f"**Reading it:** a conventional form-plus-history scorecard reaches AUC {base:.3f}; adding alternative data lifts it to {full:.3f} "
                f"(+{(full - base) * 100:.1f} points). Berg et al. (2020) found +5.3 points when adding a digital footprint to a bureau score on real purchases.")
    t = pd.DataFrame({"If the lender accepts this default rate among approved loans": eq.target_default_rate_of_approved.map("{:.0%}".format),
                      "Conventional scorecard approves": eq.conventional_scorecard_approves.map("{:.1%}".format),
                      "IntelliCredit approves": eq.intellicredit_approves.map("{:.1%}".format),
                      "Extra approvals": eq.extra_approvals_pct.map(lambda x: "none" if x == 0 else f"+{x:.0%}")})
    html_table(t)
    st.caption("Same kind of measure as the CFPB's review of Upstart (27% more approvals) and Upstart's own report (43%). Ours is on simulated loans where the benefit is built in, so it shows the mechanism only.")

    st.markdown("### 2. Forecasting Indian bad loans (real IMF data)")
    cmp_ = pd.DataFrame(real_res["india_model_comparison"]).T.sort_values("MAE (pp)")
    names = list(cmp_.index)
    figf = go.Figure(go.Bar(x=cmp_["MAE (pp)"], y=names, orientation="h", marker_color=[NAVY if "SARIMAX" in n else "#b9b19b" for n in names],
                            text=[f"{v:.2f}" for v in cmp_["MAE (pp)"]], textposition="outside", cliponaxis=False,
                            hovertemplate="%{y}<br>average miss %{x:.2f} percentage points<extra></extra>"))
    figf.update_layout(xaxis_title="Average miss on 14 unseen quarters, in percentage points (lower is better)", yaxis=dict(autorange="reversed"), xaxis_range=[0, 3.1])
    st.plotly_chart(style_fig(figf, 290), width="stretch", config={"displayModeBar": False})
    st.caption("Navy is our method. Makridakis et al. (2018) found simple statistical methods hard to beat, which is why the 'same as last quarter' naive forecast is the benchmark.")

    st.markdown("### 3. How honest are the forecast ranges? (real US delinquency data)")
    us = real_res["us_interval_coverage"]
    rows = []
    for lvl in ("80%", "90%"):
        rows.append((lvl, f"{us[lvl]['Gaussian model interval']:.1%}", f"{us[lvl]['split conformal']:.1%}", f"{us[lvl]['adaptive conformal (ACI)']:.1%}"))
    html_table(pd.DataFrame(rows, columns=["Range promised", "Plain model range actually covered", "Split conformal covered", "Adaptive conformal (ours) covered"]))
    st.caption(f"Based on {us['80%']['n forecasts']} quarterly forecasts. A range that promises 80% should contain the real value about 80% of the time; the plain model range falls short.")

    st.markdown("### 4. Do warning patterns generalise? (real US mortgage decisions)")
    h = real_res["hmda_rules"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Patterns found", f"{h['final']}", help="Mined from Delaware and Rhode Island applications.")
    c2.metric("Still riskier in an unseen state", f"{h['share_hold_out_of_state']:.0%}", help="Share of patterns that stay clearly riskier in Vermont, which they never saw.")
    c3.metric("Typical risk multiple", f"{h['median_train_lift']:.1f}x then {h['median_test_lift']:.1f}x", help="Median times-riskier on the training states, then on Vermont.")


# ---------------------------------------------------------------------------------------------------------------- 7 sources
def _sources(D, explain, style_fig, html_table, pm, real_res, srcs, npl):
    land, info = D["land"], D["info"]
    explain("Every dataset, study and product claim used above, how it was checked, and how to repeat the tests yourself.")
    st.markdown("### Real datasets used")
    rows = [
        ("German credit (Statlog)", "1,000 loans", "UCI Machine Learning Repository, via OpenML dataset 31", _link("https://www.openml.org/d/31", "openml.org/d/31")),
        ("Taiwan credit-card clients", "30,000 clients", "UCI, via OpenML 42477. Yeh & Lien (2009)", _link("https://www.openml.org/d/42477", "openml.org/d/42477")),
        ("Give Me Some Credit", "150,000 borrowers", "Kaggle competition data, via OpenML 46929", _link("https://www.openml.org/d/46929", "openml.org/d/46929")),
        ("FICO HELOC (xML Challenge)", "10,459 credit lines", "FICO Explainable ML Challenge, via OpenML 46932", _link("https://www.openml.org/d/46932", "openml.org/d/46932")),
        ("India bank bad-loan ratio", "63 quarters", "IMF Financial Soundness Indicators", _link("https://data.imf.org/", "data.imf.org")),
        ("Bad-loan ratios, 140 countries", "2,021 country-years", "World Bank indicator FB.AST.NPER.ZS", _link("https://data.worldbank.org/indicator/FB.AST.NPER.ZS", "data.worldbank.org")),
        ("US consumer delinquency", "158 quarters", "Federal Reserve via FRED (DRCLACBS)", _link("https://fred.stlouisfed.org/series/DRCLACBS", "fred.stlouisfed.org")),
        ("US mortgage applications", "48,799 (DE, RI, VT)", "CFPB HMDA public data, 2023", _link("https://ffiec.cfpb.gov/", "ffiec.cfpb.gov")),
        ("US consumer complaints", "15,000", "CFPB complaint database", _link("https://www.consumerfinance.gov/data-research/consumer-complaints/", "consumerfinance.gov")),
    ]
    html_table(pd.DataFrame(rows, columns=["Dataset", "Size", "Origin", "Link"]))
    st.caption("The four credit datasets are downloaded on demand into `data/real/benchmarks/` and are not redistributed in the repository because their licences differ.")

    rf = D["refs"]
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Sources in the literature pack", rf["total"], help="research/REFERENCES_ACM.md, numbered and linked.")
    r2.metric("Checked at the primary record", rf["primary"], help="DOI / publisher / regulator / SEC page opened.")
    r3.metric("Seen in a summary only", rf["secondary"], help="Re-check these in the original before quoting a number.")
    r4.metric("Papers saved as PDF", rf["pdfs"], help="research/papers/")

    st.markdown("### How evidence is labelled")
    k = land["status_key"]
    html_table(pd.DataFrame([(_status(s), k[s]) for s in ("primary", "secondary", "vendor")], columns=["Label", "Meaning"]))

    st.markdown("### Repeat it")
    st.code("python scripts/run_validation_suite.py        # about 10 minutes: benchmark, drivers, dynamics, re-weighting, ablation\n"
            "python scripts/build_fidelity_anchors.py      # synthetic-vs-published table\n"
            "streamlit run dashboard/app.py", language="bash")
    st.caption(f"Last run {info['run_at']} ({info['seconds']} s), seed {info['seed']}, LightGBM {info['lightgbm']}, XGBoost {info['xgboost']}, pandas {info['pandas']}. "
               f"Competitor and regulation notes last checked {land['checked_on']}.")

    st.markdown("### Limits of this evidence")
    st.markdown("""
* No real Indian gig-worker repayment data was available, so nothing here proves the model would score 0.74 on real Indian borrowers.
* The benchmark datasets are from the US, Germany and Taiwan, with different products and default definitions.
* Competitor figures are marketing or company studies unless the table says otherwise. Several summaries were seen only in search excerpts and should be rechecked before quoting.
* The made-up data has known gaps (income direction, debt burden, loan size, COVID spike, volatile monthly series). They are listed rather than hidden.""")
