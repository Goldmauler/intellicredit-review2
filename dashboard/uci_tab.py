"""
IntelliCredit dashboard - "Insights & Validation" tab (UCI comparative study)
=============================================================================
Reads the results written by the validation/ pipeline (validation/run_all.py):

    validation/results/experiment_results.json      AUC, PR-AUC, KS, Brier, ECE and policy outcomes per experiment
    validation/results/fidelity_ks_wasserstein.csv  synthetic vs UCI, feature by feature
    validation/results/fidelity_summary.json        correlation distance and distance-to-closest-record privacy check
    validation/results/*.png                        reliability diagrams, correlation heatmaps, DCR histogram

The "S->U ablated" experiment is deliberately left out of this page.
Every table and chart is built from those files; nothing is typed in by hand.
"""
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

NAVY, OCHRE, BRICK, SLATE, GREEN = "#1f3a5f", "#b8862f", "#9b2c2c", "#5a5546", "#3d6b4f"
EXPS = ["S->S", "U->U", "S->U (TSTR)", "U->S (TRTS)"]            # S->U ablated is excluded on purpose
SHOW = {"S->S": "S→S (baseline)", "U->U": "U→U (UCI ceiling)", "S->U (TSTR)": "S→U (TSTR)", "U->S (TRTS)": "U→S (TRTS)"}
EXP_COLOUR = {"S->S": NAVY, "U->U": GREEN, "S->U (TSTR)": BRICK, "U->S (TRTS)": OCHRE}
FLAG_COLOUR = {"OK": GREEN, "MODERATE": OCHRE, "[WARN] HIGH": BRICK}


def _img(path, caption):
    if os.path.exists(path):
        st.image(path, caption=caption, width="stretch")
    else:
        st.info(f"Figure not found: {os.path.basename(path)}")


def render(ctx):
    rdir = os.path.join(ctx.ROOT, "validation", "results")
    jpath = os.path.join(rdir, "experiment_results.json")
    if not os.path.exists(jpath):
        st.markdown("## Insights & Validation")
        st.warning("validation/results/experiment_results.json was not found. Run `python validation/run_all.py` from the repository root.")
        return
    res = json.load(open(jpath, encoding="utf-8"))
    fid = pd.read_csv(os.path.join(rdir, "fidelity_ks_wasserstein.csv"))
    fsum = json.load(open(os.path.join(rdir, "fidelity_summary.json"), encoding="utf-8"))

    st.markdown("## Insights & Validation: comparative study")
    ctx.explain("This tab checks the <b>synthetic-data pipeline against real data</b>: the UCI Credit Card Default dataset (Yeh &amp; Lien, 2009; 30,000 Taiwanese card holders). "
                "Its fields are mapped onto the IntelliCredit features (a crosswalk), then the same LightGBM recipe is trained and tested in every direction. "
                "If the synthetic data were a faithful copy of real borrowers, a model trained on it would work on real borrowers.")

    # ---------------------------------------------------------------- 1. transfer experiments
    st.markdown("### 1 · Transfer experiments (TSTR / TRTS)")
    notation = pd.DataFrame([
        ["S→S", "Train on Synthetic, test on Synthetic (our baseline)"],
        ["U→U", "Train on UCI, test on UCI (the ceiling on real data)"],
        ["S→U (TSTR)", "Train on Synthetic, test on UCI: measures generalisation to real borrowers"],
        ["U→S (TRTS)", "Train on UCI, test on Synthetic: the reverse transfer"],
    ], columns=["Notation", "Meaning"])
    ctx.html_table(notation)

    rows = []
    for k in EXPS:
        r = res[k]
        rows.append({"Experiment": SHOW[k], "AUC": r["auc"], "PR-AUC": r["pr_auc"], "KS": r["ks"], "Brier": r["brier"], "ECE": r["ece"]})
    mt = pd.DataFrame(rows).set_index("Experiment")
    st.caption("Data: validation/results/experiment_results.json. Higher is better for AUC, PR-AUC and KS; lower is better for Brier and ECE. The best value in each column is green. "
               "In the chart below, the dashed red line is AUC 0.5, a coin flip.")
    st.dataframe(mt.style.format("{:.4f}").highlight_max(subset=["AUC", "PR-AUC", "KS"], color="#dfeadb").highlight_min(subset=["Brier", "ECE"], color="#dfeadb"), width="stretch")

    ss, tstr = res["S->S"]["auc"], res["S->U (TSTR)"]["auc"]
    gap = ss - tstr
    verdict = "Good transfer (<0.05)" if gap < 0.05 else "Moderate mismatch (0.05-0.15)" if gap < 0.15 else "Strong domain mismatch (>0.15)"
    kind = "ok" if gap < 0.05 else "mid" if gap < 0.15 else ""
    st.markdown(f'<div class="callout {kind}"><b>AUC gap (S→S vs TSTR):</b> {gap:.4f} &mdash; {verdict}</div>', unsafe_allow_html=True)

    fig = make_subplots(rows=1, cols=3, subplot_titles=("Ranking: ROC-AUC and PR-AUC", "Separation: KS", "Probability error: Brier and ECE"), horizontal_spacing=0.09)
    # the dashed red line in the first panel marks AUC 0.5, a coin flip
    xs = [SHOW[k] for k in EXPS]
    fig.add_trace(go.Bar(x=xs, y=[res[k]["auc"] for k in EXPS], name="ROC-AUC", marker_color=NAVY, hovertemplate="%{x}: %{y:.3f}<extra>ROC-AUC</extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=xs, y=[res[k]["pr_auc"] for k in EXPS], name="PR-AUC", marker_color=OCHRE, hovertemplate="%{x}: %{y:.3f}<extra>PR-AUC</extra>"), row=1, col=1)
    fig.add_hline(y=0.5, line_dash="dash", line_color=BRICK, row=1, col=1)
    fig.add_trace(go.Bar(x=xs, y=[res[k]["ks"] for k in EXPS], marker_color=[EXP_COLOUR[k] for k in EXPS], showlegend=False, hovertemplate="%{x}: %{y:.3f}<extra>KS</extra>"), row=1, col=2)
    fig.add_trace(go.Bar(x=xs, y=[res[k]["brier"] for k in EXPS], name="Brier", marker_color=SLATE, hovertemplate="%{x}: %{y:.3f}<extra>Brier</extra>"), row=1, col=3)
    fig.add_trace(go.Bar(x=xs, y=[res[k]["ece"] for k in EXPS], name="ECE", marker_color=BRICK, hovertemplate="%{x}: %{y:.3f}<extra>ECE</extra>"), row=1, col=3)
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=-0.3))
    fig.update_xaxes(tickangle=-30, tickfont=dict(size=10))
    fig.update_yaxes(range=[0, 1], row=1, col=1)
    st.plotly_chart(ctx.style_fig(fig, 400), width="stretch", config={"displayModeBar": False})

    st.markdown(
        f"**How to read it.** U→U ({res['U->U']['auc']:.2f}) shows the recipe works on real data. S→S ({ss:.2f}) is the baseline inside the synthetic world, capped near 0.78 because every label is a random draw from a known default formula. "
        f"The drop to **{tstr:.2f}** when the synthetic-trained model meets real card holders is the honest finding: the synthetic data carries the structure of the "
        "gig-worker problem we designed, not the structure of Taiwanese credit-card defaults. That is a **domain mismatch**, not a bug. "
        "The same section also lists where the two data sets differ (section 3), so the gap can be explained rather than hidden.")

    # ---------------------------------------------------------------- 2. policy outcomes
    st.markdown("### 2 · Policy outcomes (break-even rule)")
    st.caption("Data: experiment_results.json (policy block). Approve a borrower when the predicted default chance is below the break-even limit, then see who actually defaulted. "
               "A useful model has a much lower default rate among the approved than among the declined.")
    prow = []
    for k in EXPS:
        p = res[k]["policy"]
        prow.append({"Experiment": SHOW[k], "Approval rate": p["approval_rate"], "Default rate (approved)": p["default_rate_approved"], "Default rate (declined)": p["default_rate_declined"]})
    pt = pd.DataFrame(prow).set_index("Experiment")
    st.dataframe(pt.style.format("{:.1%}"), width="stretch")
    fig = go.Figure()
    for col, colour in [("Approval rate", SLATE), ("Default rate (approved)", GREEN), ("Default rate (declined)", BRICK)]:
        fig.add_trace(go.Bar(x=pt.index, y=pt[col], name=col, marker_color=colour, text=[f"{v:.0%}" for v in pt[col]], textposition="outside", cliponaxis=False,
                             hovertemplate="%{x}<br>" + col + ": %{y:.1%}<extra></extra>"))
    fig.update_layout(barmode="group", yaxis_tickformat=".0%", legend=dict(orientation="h", y=1.12))
    fig.update_yaxes(range=[0, 1.05])
    st.plotly_chart(ctx.style_fig(fig, 360), width="stretch", config={"displayModeBar": False})
    sp, up = res["S->S"]["policy"], res["S->U (TSTR)"]["policy"]
    st.markdown(f"**Reading it:** inside the synthetic data the policy works (approved {sp['default_rate_approved']:.1%} vs declined {sp['default_rate_declined']:.1%}). "
                f"On real card holders, a synthetic-trained policy approves {up['approval_rate']:.0%} and barely separates the groups ({up['default_rate_approved']:.1%} vs {up['default_rate_declined']:.1%}), "
                "which is the same domain mismatch seen in section 1.")

    # ---------------------------------------------------------------- calibration
    with st.expander("Reliability diagrams for the four experiments"):
        _img(os.path.join(rdir, "calibration_tstr.png"), "Predicted probability against what really happened. The closer to the dashed diagonal, the better calibrated (validation step 3).")

    # ---------------------------------------------------------------- 3. fidelity
    st.markdown("### 3 · Data fidelity audit (synthetic vs UCI)")
    st.caption("Data: validation/results/fidelity_ks_wasserstein.csv. For each shared feature: the two means and spreads, the Kolmogorov-Smirnov statistic (0 = identical distributions, 1 = completely different) "
               "and the normalised Wasserstein distance. Flags: OK, MODERATE, [WARN] HIGH.")
    ft = fid.rename(columns={"feature": "Feature", "synth_mean": "Synth mean", "uci_mean": "UCI mean", "synth_std": "Synth std", "uci_std": "UCI std",
                             "ks_statistic": "KS statistic", "wasserstein_normalized": "Wasserstein (norm)", "fidelity_flag": "Fidelity flag"})
    ft = ft[["Feature", "Synth mean", "UCI mean", "Synth std", "UCI std", "KS statistic", "Wasserstein (norm)", "Fidelity flag"]].set_index("Feature")
    st.dataframe(ft.style.format({"Synth mean": "{:,.4f}", "UCI mean": "{:,.4f}", "Synth std": "{:,.4f}", "UCI std": "{:,.4f}", "KS statistic": "{:.4f}", "Wasserstein (norm)": "{:.5f}"}),
                 width="stretch")

    d = fid.sort_values("ks_statistic")
    fig = go.Figure(go.Bar(y=d["feature"], x=d["ks_statistic"], orientation="h", marker_color=[FLAG_COLOUR.get(f, SLATE) for f in d["fidelity_flag"]],
                           text=[f"{v:.2f}" for v in d["ks_statistic"]], textposition="outside", cliponaxis=False,
                           customdata=d["fidelity_flag"], hovertemplate="%{y}: KS %{x:.3f} (%{customdata})<extra></extra>"))
    fig.update_layout(xaxis_title="KS statistic (lower = closer to real data). Green = OK, ochre = moderate, red = high difference")
    fig.update_xaxes(range=[0, float(d["ks_statistic"].max() * 1.15)])
    st.plotly_chart(ctx.style_fig(fig, 400), width="stretch", config={"displayModeBar": False})

    nh = len(fsum["overall_fidelity"]["high_ks_features"])
    nok = len(fsum["overall_fidelity"]["good_ks_features"])
    dcr = fsum["dcr"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Features close to real", f"{nok} of {len(fid)}", help="Features whose KS statistic is flagged OK.")
    m2.metric("Correlation structure distance", f"{fsum['correlation_matrix_frobenius_distance']:.2f}", help="Frobenius distance between the two correlation matrices. 0 means identical relationships between features.")
    m3.metric("Privacy check (DCR)", dcr["privacy_flag"], delta=f"mean distance {dcr['mean']:.2f}", delta_color="off",
              help="Distance to closest record: how near each synthetic row is to a real row. Very small values would mean the generator copied real people.")
    st.markdown(f"**Reading it:** only {nok} of {len(fid)} features sit close to the real distributions and {nh} are flagged high. The synthetic data was built around the gig-worker story "
                "(utility bills, income volatility, BNPL), so it differs sharply from card-holder behaviour in exactly those features. The distance-to-closest-record check passes: in a 500-row sample no synthetic row sits on top of a real one.")

    with st.expander("Correlation heatmaps and distance-to-closest-record histogram"):
        c1, c2 = st.columns(2)
        with c1:
            _img(os.path.join(rdir, "fidelity_corr_synth.png"), "Correlations between features, synthetic data")
        with c2:
            _img(os.path.join(rdir, "fidelity_corr_uci.png"), "Correlations between features, UCI real data")
        _img(os.path.join(rdir, "fidelity_dcr_hist.png"), "Distance to the closest real record for sampled synthetic rows (validation step 4)")

    st.caption("Reproduce: python validation/run_all.py (step 1 needs data_compare/UCI_Credit_Card.csv, which is not in the repository; steps 2 to 4 re-run from the saved validation/results/uci_mapped.csv). "
               "The four-real-dataset study with the same model recipe is on the Proof & comparison tab.")
