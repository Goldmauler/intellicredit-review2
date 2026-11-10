"""
IntelliCredit dashboard - "Review 1 models" tab (notebooks 04 and 05)
=====================================================================
    models/model_comparison.csv                  11 Review 1 models on the 6,000 unseen borrowers (notebook 04)
    notebooks/05_M5_evaluation_interpretation    summary table (AUC-PR, Brier) and the saved evaluation figures
    models/underwriting_model_card.json          the final calibrated LightGBM, for the Review 1 -> final comparison
    reports/figures/final/F05..F08               ROC, calibration, SHAP and cost-threshold figures from notebook 05
"""
import json
import os
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

NAVY, OCHRE, BRICK, SLATE, GREEN = "#1f3a5f", "#b8862f", "#9b2c2c", "#5a5546", "#3d6b4f"
STAGE_COLOUR = {"Baseline": SLATE, "Tuned": NAVY, "Ensemble": OCHRE}
FAMILIES = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]


def _stage(name):
    if name.startswith(("Voting", "Stacking")):
        return "Ensemble"
    return "Tuned" if "(Tuned)" in name else "Baseline"


def _notebook05_table(root):
    """Parse the summary table in notebook 05 (markdown) so the numbers come from the file, not from this page."""
    path = os.path.join(root, "notebooks", "05_M5_evaluation_interpretation.ipynb")
    try:
        nb = json.load(open(path, encoding="utf-8"))
        for c in nb["cells"]:
            src = "".join(c["source"])
            if c["cell_type"] == "markdown" and "Model Comparison Summary Table" in src:
                rows = [ln for ln in src.splitlines() if ln.strip().startswith("|")]
                head = [h.strip() for h in rows[0].strip().strip("|").split("|")]
                out = []
                for ln in rows[2:]:
                    cells = [re.sub(r"[*_]", "", x).strip() for x in ln.strip().strip("|").split("|")]
                    if len(cells) == len(head) and "ceiling" not in cells[0].lower():
                        out.append([cells[0]] + [float(x) if x else np.nan for x in cells[1:]])
                return pd.DataFrame(out, columns=head)
    except Exception:
        pass
    return None


def _img(path, caption):
    if os.path.exists(path):
        st.image(path, caption=caption, width="stretch")
    else:
        st.info(f"Figure not found: {os.path.basename(path)}")


def render(ctx):
    st.markdown("## Review 1 models")
    ctx.explain("<b>Review 1</b> (notebooks 01 to 05) built the first default-prediction models on the same 30,000 borrowers. Eleven models were trained and compared on the same "
                "6,000 unseen borrowers: five baselines, four tuned versions, and two ensembles. This tab shows how they compare, what each metric hides, "
                "and how the final model of Review 2 and 3 improved on them.")

    cmp_ = pd.read_csv(os.path.join(ctx.MODELS_DIR, "model_comparison.csv"))
    cmp_["Stage"] = cmp_["Model"].map(_stage)
    cmp_["Family"] = cmp_["Model"].str.replace(r" \(Tuned\)", "", regex=True)
    card = ctx.card
    ceiling = float(card.get("oracle_auc_true_pd", np.nan))
    final_auc = float(card["test_auc"])

    best_auc = cmp_.loc[cmp_["AUC-ROC"].idxmax()]
    best_f1 = cmp_.loc[cmp_["F1-Score"].idxmax()]
    best_rec = cmp_.loc[cmp_["Recall"].idxmax()]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Models compared", f"{len(cmp_)}", help="models/model_comparison.csv, all scored on the same 6,000 unseen borrowers.")
    k2.metric("Best AUC", f"{best_auc['AUC-ROC']:.3f}", delta=best_auc["Model"], delta_color="off", help="AUC is the chance that a random defaulter is ranked riskier than a random non-defaulter. 0.5 is a coin flip.")
    k3.metric("Best F1 at 0.5 cut-off", f"{best_f1['F1-Score']:.3f}", delta=best_f1["Model"], delta_color="off", help="F1 balances precision and recall for the default class at a 0.5 probability cut-off.")
    k4.metric("Best possible AUC", f"{ceiling:.3f}", help="Scoring the generator's own true default probability. No model can beat it, because each label is a random draw from that probability.")

    # ---------------------------------------------------------------- 1. AUC lollipop
    st.markdown("### 1 · Ranking power of every model (AUC)")
    st.caption("Data: models/model_comparison.csv (AUC-ROC column). Each line runs from 0.5 (a coin flip) to the model's AUC. The dotted lines mark the best possible AUC on this data and the final model used in the dashboard.")
    d = cmp_.sort_values("AUC-ROC")
    fig = go.Figure()
    for _, r in d.iterrows():
        fig.add_trace(go.Scatter(x=[0.5, r["AUC-ROC"]], y=[r["Model"]] * 2, mode="lines", line=dict(color="#cfc7b3", width=3), showlegend=False, hoverinfo="skip"))
    for stage, g in d.groupby("Stage"):
        fig.add_trace(go.Scatter(x=g["AUC-ROC"], y=g["Model"], mode="markers+text", name=stage, text=[f"{v:.3f}" for v in g["AUC-ROC"]], textposition="middle right",
                                 marker=dict(size=13, color=STAGE_COLOUR[stage], line=dict(color="#fffdf8", width=1.5)),
                                 hovertemplate="%{y}: AUC %{x:.4f}<extra>" + stage + "</extra>"))
    fig.add_vline(x=ceiling, line_dash="dot", line_color=GREEN, annotation_text=f"Best possible {ceiling:.3f}", annotation_position="top")
    fig.add_vline(x=final_auc, line_dash="dot", line_color=BRICK, annotation_text=f"Final model {final_auc:.3f}", annotation_position="bottom left")
    fig.update_xaxes(range=[0.5, ceiling + 0.025], title_text="AUC-ROC (higher is better)")
    fig.update_layout(legend=dict(orientation="h", y=1.1))
    st.plotly_chart(ctx.style_fig(fig, 470), width="stretch", config={"displayModeBar": False})
    gap = best_auc["AUC-ROC"] - 0.5
    st.markdown(f"**Reading it:** the best Review 1 model ({best_auc['Model']}) captures about **{gap / (ceiling - 0.5):.0%}** of the discrimination that is possible on this data "
                f"(({best_auc['AUC-ROC']:.3f} − 0.5) ÷ ({ceiling:.3f} − 0.5)). The spread between the top models is tiny, so the choice between them was driven by other things: "
                "calibration, explainability and cost.")

    # ---------------------------------------------------------------- 2. accuracy trap
    st.markdown("### 2 · Why accuracy alone misleads")
    st.caption("Data: model_comparison.csv (Accuracy and Recall at the 0.5 cut-off). Only about 8% of borrowers default, so a model that approves almost everyone is 'accurate' but useless to a lender.")
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0.5, x1=1.0, y0=-0.03, y1=0.08, fillcolor=BRICK, opacity=0.10, line_width=0, layer="below")
    fig.add_annotation(x=0.66, y=0.04, text="Catches almost no defaulters", showarrow=False, font=dict(size=12, color=BRICK))
    for stage, g in cmp_.groupby("Stage"):
        fig.add_trace(go.Scatter(x=g["Accuracy"], y=g["Recall"], mode="markers", name=stage, marker=dict(size=13, color=STAGE_COLOUR[stage], line=dict(color="#fffdf8", width=1.5)),
                                 customdata=np.c_[g["Model"], g["Precision"], g["F1-Score"]],
                                 hovertemplate="<b>%{customdata[0]}</b><br>Accuracy %{x:.1%} · Recall %{y:.1%}<br>Precision %{customdata[1]:.1%} · F1 %{customdata[2]:.3f}<extra></extra>"))
    weak_ = cmp_[cmp_["Recall"] < 0.05]
    if len(weak_):
        fig.add_annotation(x=float(weak_["Accuracy"].mean()), y=0.0, ax=-70, ay=-42, showarrow=True, arrowcolor=BRICK, arrowwidth=1.2, xanchor="right",
                           text="<br>".join(m.replace(" Classifier", "").replace(" (Tuned)", " (tuned)") for m in weak_["Model"]), font=dict(size=11, color=BRICK))
    fig.update_xaxes(title_text="Accuracy (share of all borrowers classified correctly)", tickformat=".0%", range=[0.6, 0.97])
    fig.update_yaxes(title_text="Recall (share of real defaulters caught)", tickformat=".0%", range=[-0.03, 0.85])
    fig.update_layout(legend=dict(orientation="h", y=1.1))
    st.plotly_chart(ctx.style_fig(fig, 440), width="stretch", config={"displayModeBar": False})
    weak = cmp_[cmp_["Recall"] < 0.05]
    if len(weak):
        parts = [f"**{r['Model']}**: {r['Accuracy']:.0%} accurate but recall {r['Recall']:.1%}" for _, r in weak.iterrows()]
        st.markdown("**Look at the corner:** " + "; ".join(parts) + ". Both look excellent on accuracy and are useless at finding defaulters, which is why this project judges models on AUC, calibration and money (expected loss), not accuracy.")

    # ---------------------------------------------------------------- 3. tuning
    st.markdown("### 3 · What hyper-parameter tuning bought")
    st.caption("Data: model_comparison.csv. Each family before (grey) and after (blue) RandomizedSearchCV / GridSearchCV tuning in notebook 04.")
    base_ = cmp_[cmp_["Stage"] == "Baseline"].set_index("Family")["AUC-ROC"]
    tuned_ = cmp_[cmp_["Stage"] == "Tuned"].set_index("Family")["AUC-ROC"]
    fams = [f for f in FAMILIES if f in base_.index and f in tuned_.index]
    fig = go.Figure()
    for f in fams:
        fig.add_trace(go.Scatter(x=[base_[f], tuned_[f]], y=[f, f], mode="lines", line=dict(color="#cfc7b3", width=4), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[base_[f] for f in fams], y=fams, mode="markers", name="Baseline", marker=dict(size=14, color=SLATE), hovertemplate="%{y} baseline: %{x:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[tuned_[f] for f in fams], y=fams, mode="markers+text", name="Tuned", marker=dict(size=14, color=NAVY),
                             text=[f"{tuned_[f] - base_[f]:+.3f}" for f in fams], textposition="middle right", hovertemplate="%{y} tuned: %{x:.4f}<extra></extra>"))
    fig.update_xaxes(title_text="AUC-ROC", range=[min(base_.min(), tuned_.min()) - 0.01, max(base_.max(), tuned_.max()) + 0.02])
    fig.update_layout(legend=dict(orientation="h", y=1.15))
    st.plotly_chart(ctx.style_fig(fig, 300), width="stretch", config={"displayModeBar": False})
    big = (tuned_[fams] - base_[fams]).idxmax()
    st.markdown(f"**Reading it:** tuning helped most for **{big}** ({tuned_[big] - base_[big]:+.3f} AUC) and barely moved logistic regression, a transparent scorecard that was already close to the others.")

    # ---------------------------------------------------------------- 4. P / R / F1
    st.markdown("### 4 · Precision, recall and F1 at the 0.5 cut-off")
    st.caption("Data: model_comparison.csv. Precision = of those flagged, how many really defaulted. Recall = of all defaulters, how many were flagged. F1 balances the two. "
               "The baseline random forest flagged only a handful of borrowers (so its precision is 1.0 on very few cases, with recall near zero) and the stacking model flagged none, so its bars are empty.")
    d2 = cmp_.sort_values("F1-Score")
    fig = go.Figure()
    for col, colour in [("Precision", OCHRE), ("Recall", NAVY), ("F1-Score", GREEN)]:
        fig.add_trace(go.Bar(y=d2["Model"], x=d2[col], orientation="h", name=col, marker_color=colour, hovertemplate="%{y}<br>" + col + ": %{x:.3f}<extra></extra>"))
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.07), xaxis_title="Score (0 to 1)")
    st.plotly_chart(ctx.style_fig(fig, 640), width="stretch", config={"displayModeBar": False})

    # ---------------------------------------------------------------- 5. figures
    st.markdown("### 5 · Evaluation figures from notebook 05")
    fdir = os.path.join(ctx.ROOT, "reports", "figures", "final")
    t1, t2, t3, t4 = st.tabs(["ROC curves", "Calibration", "SHAP drivers", "Cost and threshold"])
    with t1:
        _img(os.path.join(fdir, "F05_roc_curves.png"), "ROC curves of the four tuned models (notebook 05, Part D). The closer to the top-left, the better the ranking.")
    with t2:
        _img(os.path.join(fdir, "F06_calibration.png"), "Calibration of the Review 1 models (notebook 05, Part E). Curves below the diagonal mean the predicted default chance is higher than what really happened.")
        st.markdown("This is the weakness the final model fixes: Review 1 trained with SMOTE or class weights, which inflates the predicted probabilities. They rank people well but are not trustworthy as rupee-level default chances.")
    with t3:
        _img(os.path.join(fdir, "F07_shap_summary.png"), "SHAP summary (notebook 05, Part F): which features push the predicted risk up or down.")
    with t4:
        _img(os.path.join(fdir, "F08_cost_threshold.png"), "Cost-benefit analysis (notebook 05, Part G): the best probability cut-off is not 0.5 once the cost of a missed defaulter is counted.")

    # ---------------------------------------------------------------- 6. full table + Review 1 -> final
    st.markdown("### 6 · Full results table")
    st.caption("Data: models/model_comparison.csv. Best value in each column is green.")
    show = cmp_[["Model", "Stage", "Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]].set_index("Model")
    st.dataframe(show.style.format({c: "{:.3f}" for c in ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]})
                 .highlight_max(subset=["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"], color="#dfeadb"), width="stretch")

    st.markdown("### 7 · From Review 1 to the final model")
    nb = _notebook05_table(ctx.ROOT)
    if nb is not None and len(nb):
        final = pd.DataFrame([{"Model": "Final model: LightGBM, calibrated (Review 2/3)", "AUC-ROC": final_auc, "AUC-PR": float(card["test_pr_auc"]), "Brier": float(card["test_brier"])}])
        tbl = pd.concat([nb[["Model", "AUC-ROC", "AUC-PR", "Brier"]], final], ignore_index=True).set_index("Model")
        st.caption("Data: the summary table in notebook 05 (Part H) for the four tuned Review 1 models, and models/underwriting_model_card.json for the final model. Same 6,000 unseen borrowers. Brier score measures probability accuracy, lower is better.")
        st.dataframe(tbl.style.format("{:.3f}").highlight_max(subset=["AUC-ROC", "AUC-PR"], color="#dfeadb").highlight_min(subset=["Brier"], color="#dfeadb"), width="stretch")
        b_old = nb["Brier"].dropna()
        st.markdown(f"**What changed:** ranking power (AUC) is about the same, because it was already near the {ceiling:.2f} ceiling. The big gain is in the probabilities themselves: the Brier score "
                    f"falls from **{b_old.min():.2f} to {b_old.max():.2f}** for the Review 1 models to **{float(card['test_brier']):.3f}** for the final model, because Review 2 stopped re-weighting the classes and calibrated the output. "
                    "A calibrated probability is what lets the dashboard compare a default chance directly with the break-even limit and price the loan in rupees.")
    else:
        st.info("The summary table could not be read from notebook 05.")
