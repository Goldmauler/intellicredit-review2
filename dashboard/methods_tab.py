"""
IntelliCredit dashboard - "Time series & rules" tab (Review 2 Methods 1 and 2, abstract view)
=============================================================================================
One short page for both methods: a plain-language summary, three headline numbers and two charts each.
All charts are drawn from saved files:

    data/portfolio/portfolio_monthly.csv   monthly loan-book history (2019-2024)
    models/ts_forecast_scenarios.csv       SARIMAX forecast for 2025 under three economies
    models/ts_model_comparison.csv         2024 hold-out accuracy of the 7 forecasting methods
    models/association_rules.csv           93 validated rules with training and hold-out measures

The full explorer of the 93 warning patterns is rendered by app.py inside an expander underneath.
"""
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

NAVY, OCHRE, BRICK, SLATE, GREEN = "#1f3a5f", "#b8862f", "#9b2c2c", "#5a5546", "#3d6b4f"
SIZE_COLOURS = {1: GREEN, 2: OCHRE, 3: NAVY, 4: BRICK}
COVID_SPANS = [("2020-03-01", "2020-08-01"), ("2021-04-01", "2021-07-01")]    # same spans as notebook 06
CHART_CFG = {"displayModeBar": False}


def render(ctx):
    st.markdown("## Time series and association rules")
    ctx.explain("The two Review 2 methods in one page. <b>Time series</b> watches the whole loan book month by month and forecasts how defaults move with the economy. "
                "<b>Association rules</b> look for combinations of behaviours that go with many more defaults than average. Each has three headline numbers and two charts.")

    # ================================================================== Method 1
    pm = ctx.pm.sort_values("date").reset_index(drop=True)
    scen = ctx.scen
    cmp_ = pd.read_csv(os.path.join(ctx.MODELS_DIR, "ts_model_comparison.csv"), index_col=0)
    best = cmp_["MAE (pp)"].idxmin()

    st.markdown("### Method 1 · Time series")
    st.caption("Notebook 06. A SARIMAX model with economic drivers (unemployment, GDP, inflation, interest rate, rupee, COVID) forecasts the share of loans newly 90+ days late.")
    k1, k2, k3 = st.columns(3)
    k1.metric("Latest default rate", f"{pm['default_rate'].iloc[-1] * 100:.2f}%", help="Share of active loans newly 90+ days late in the latest month of the simulated loan book.")
    k2.metric("Best method on 2024", best, help="Lowest mean absolute error on the 12 months of 2024 that no model had seen (7 methods compared).")
    k3.metric("Its average error", f"{cmp_.loc[best, 'MAE (pp)']:.3f} pp", help="Mean absolute error in percentage points of the default rate. Lower is better.")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**History and 2025 forecast**")
        names_ = [k for k in ctx.SCEN_COLORS if k in set(scen["scenario"])]
        tail = pm[pm["date"] >= "2019-01-01"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tail["date"], y=tail["default_rate"] * 100, name="History", line=dict(color="#1f1f1f", width=2.2)))
        for name in names_:
            g = scen[scen["scenario"] == name]
            fig.add_trace(go.Scatter(x=g["date"], y=g["forecast"], name=ctx.SCEN_LABELS.get(name, name), line=dict(color=ctx.SCEN_COLORS[name], width=2.6),
                                     hovertemplate="%{x|%b %Y}: %{y:.2f}%<extra>" + ctx.SCEN_LABELS.get(name, name) + "</extra>"))
        for a, b in COVID_SPANS:
            fig.add_vrect(x0=a, x1=b, fillcolor=SLATE, opacity=0.14, line_width=0)
        fig.update_layout(yaxis_title="% of loans newly 90+ days late", hovermode="x unified", legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(ctx.style_fig(fig, 340), width="stretch", config=CHART_CFG)
        st.caption("Data: portfolio_monthly.csv and ts_forecast_scenarios.csv. Grey bands are the COVID months. The three coloured lines are the 2025 economies.")
    with c2:
        st.markdown("**Which method forecast 2024 best?**")
        order = cmp_["MAE (pp)"].sort_values().index.tolist()
        vals = cmp_.loc[order, "MAE (pp)"]
        fig = go.Figure(go.Bar(y=order, x=vals.values, orientation="h", marker_color=[GREEN if m == best else NAVY for m in order],
                               text=[f"{v:.3f}" for v in vals.values], textposition="outside", cliponaxis=False, hovertemplate="%{y}: %{x:.3f} pp<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(range=[0, float(vals.max() * 1.2)], title_text="Error in pp (lower is better)")
        st.plotly_chart(ctx.style_fig(fig, 340), width="stretch", config=CHART_CFG)
        st.caption("Data: ts_model_comparison.csv. Each method was trained on 2019-2023 and asked to forecast 2024.")

    st.markdown("---")

    # ================================================================== Method 2
    rules = ctx.rules.copy()
    rules["pattern"] = rules["antecedent"].map(ctx.pretty)

    st.markdown("### Method 2 · Association rules")
    st.caption("Notebook 07. Rules were mined on 24,000 borrowers (Apriori / FP-Growth), then re-checked on 6,000 borrowers they had never seen.")
    k1, k2, k3 = st.columns(3)
    k1.metric("Rules kept", f"{len(rules)}", help="Rules passing lift of at least 1.5, support of at least 0.5% and a Bonferroni-corrected Fisher exact test.")
    k2.metric("Hold on new data", f"{(rules['test_lift'] > 1).mean():.0%}", help="Share of rules still riskier than average on the 6,000 unseen borrowers.")
    k3.metric("Top lift (unseen)", f"{rules['test_lift'].max():.2f}x", help="The riskiest rule goes with that many times the average default rate on unseen borrowers.")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Do the rules hold on unseen borrowers?**")
        lim = float(max(rules["lift"].max(), rules["test_lift"].max()) * 1.08)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, lim], y=[1, lim], mode="lines", line=dict(color=SLATE, dash="dash"), name="Same on new data", hoverinfo="skip"))
        for n in sorted(rules["n_items"].unique()):
            g = rules[rules["n_items"] == n]
            fig.add_trace(go.Scatter(x=g["lift"], y=g["test_lift"], mode="markers", name=f"{n} sign" + ("" if n == 1 else "s"),
                                     marker=dict(size=9, color=SIZE_COLOURS.get(n, SLATE), line=dict(color="#fffdf8", width=1)), customdata=g["pattern"],
                                     hovertemplate="<b>%{customdata}</b><br>Training %{x:.2f}x, unseen %{y:.2f}x<extra></extra>"))
        fig.update_xaxes(range=[1, lim], title_text="Times riskier on training borrowers")
        fig.update_yaxes(range=[0, lim], title_text="Times riskier on unseen borrowers")
        fig.update_layout(legend=dict(orientation="h", y=1.14))
        st.plotly_chart(ctx.style_fig(fig, 340), width="stretch", config=CHART_CFG)
        st.caption("Data: association_rules.csv. Each dot is a rule. Dots above 1 on the vertical axis still go with more defaults than average on new data.")
    with c2:
        st.markdown("**The 10 riskiest patterns (unseen data)**")
        top = rules.sort_values("test_lift", ascending=False).head(10).iloc[::-1]
        fig = go.Figure(go.Bar(y=top["pattern"].str.slice(0, 46) + np.where(top["pattern"].str.len() > 46, "…", ""), x=top["test_lift"], orientation="h", marker_color=BRICK,
                               text=[f"{c:.0%}" for c in top["test_confidence"]], textposition="outside", cliponaxis=False,
                               customdata=top["pattern"], hovertemplate="<b>%{customdata}</b><br>%{x:.2f}x riskier than average<extra></extra>"))
        fig.update_xaxes(range=[0, float(top["test_lift"].max() * 1.3)], title_text="Times riskier than average")
        st.plotly_chart(ctx.style_fig(fig, 340), width="stretch", config=CHART_CFG)
        st.caption("Data: association_rules.csv. The label is the default rate among matching unseen borrowers (the average borrower defaults at 7.9%).")
