"""
IntelliCredit dashboard - "Scenario lab" tab
============================================
A live what-if tool. Change the economy (and the lender's assumptions) and see, instantly:
  1. the forecast default rate for 2025,   2. expected defaults and expected credit loss,
  3. what each shock contributed,          4. how lending decisions and expected profit change.
The forecasting model is refitted from the files (scenario_core.py) so the Baseline, Adverse and Severe presets
reproduce the Time series tab exactly.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import scenario_core as sc
from validation_tab import _table as table

NAVY, OCHRE, BRICK, SLATE = "#1f3a5f", "#b8862f", "#9b2c2c", "#5a5546"
MONTHS = [d.strftime("%b %Y") for d in sc.FUT]
M2D = dict(zip(MONTHS, sc.FUT))


@st.cache_resource(show_spinner="Fitting the forecasting model from the data files (about 3 seconds, once) ...")
def engine(data_dir):
    pm, macro = sc.load(data_dir)
    res = sc.fit(pm, macro)
    return dict(pm=pm, macro=macro, res=res, base=sc.forecast(res, sc.scenario_macro(macro, sc.PRESETS["Baseline"])),
                neutral=sc.neutral_rate(res, macro, pm), coef=sc.coefficients(res))


def _preset():
    p = sc.PRESETS.get(st.session_state["sc_preset"])
    if p:
        st.session_state.update(sc_unemp=p["unemployment_rate"], sc_gdp=p["gdp_growth"], sc_cpi=p["cpi_inflation_yoy"], sc_call=p["call_money_rate"],
                                sc_inr=p["inr_depreciation_yoy"], sc_lock=[pd.Timestamp(d).strftime("%b %Y") for d in p["lockdown"]], sc_wave=[],
                                sc_start="Mar 2025", sc_ramp=3)


def cr(lakh):
    return f"Rs {lakh / 100:,.2f} cr" if abs(lakh) >= 100 else f"Rs {lakh:,.0f} L"


def money(lakh):
    return f"Rs {lakh / 100:,.2f} crore" if abs(lakh) >= 100 else f"Rs {lakh:,.0f} lakh"


def run_policy(ctx, tb, mult, lgd):
    pit = np.clip(tb["predicted_pd"].values * mult, 0, 1)
    band = np.select([pit < .04, pit < .07, pit < .11, pit < .16], list("ABCD"), "E")
    dec = np.asarray(ctx.decide(pit, tb["break_even_pd"].values, band, tb["rules_hit"].values))
    m = np.maximum(tb["interest_rate_annual"].values / 100 - ctx.COST_OF_FUNDS, 0.001) * tb["loan_tenure_months"].values / 12
    prof = tb["loan_amount_requested"].values * ((1 - pit) * m - pit * lgd)
    return dec, pit, prof


def counts(dec, pit, prof, accept=None):
    acc = np.isin(dec, ["APPROVE", "CONDITIONS"]) if accept is None else accept
    return dict(approve=int((dec == "APPROVE").sum()), cond=int((dec == "CONDITIONS").sum()), decline=int((dec == "DECLINE").sum()),
                pd_acc=float(pit[acc].mean()) if acc.any() else 0.0, profit=float(prof[acc].sum() / 1e5))


def render(ctx):
    st.markdown("## Scenario lab")
    ctx.explain("Ask <b>what if the economy changes?</b> Move the sliders and the forecast, the expected losses and the lending decisions all update at once. "
                "The model behind it is the SARIMAX forecaster, refitted live from the data files, so the three presets match the Time series tab exactly.")
    E = engine(ctx.DATA_DIR)
    res, macro, base, neutral, coef, pm = E["res"], E["macro"], E["base"], E["neutral"], E["coef"], E["pm"]

    for k, v in dict(sc_preset="Baseline", sc_unemp=0.0, sc_gdp=0.0, sc_cpi=0.0, sc_call=0.0, sc_inr=0.0, sc_lock=[], sc_wave=[], sc_start="Mar 2025", sc_ramp=3,
                     sc_lgd=int(round(ctx.LGD * 100)),
                     sc_growth=round(((pm["active_loans"].iloc[-1] / pm["active_loans"].iloc[-13]) ** (1 / 12) - 1) * 100, 2)).items():
        st.session_state.setdefault(k, v)

    left, right = st.columns([1, 1.9], gap="large")
    with left:
        ctx.step(1, "Choose the economy")
        st.radio("Start from a preset", ["Baseline", "Adverse", "Severe", "Custom"], key="sc_preset", on_change=_preset, horizontal=True,
                 help="Baseline: published macro path. Adverse: unemployment +1.5 pp and GDP -3 pp. Severe: unemployment +3 pp, GDP -8 pp and two lockdown months. Custom: move the sliders.")
        st.slider("Unemployment change (percentage points)", -3.0, 8.0, step=0.5, key="sc_unemp", help="Added on top of the baseline unemployment path.")
        st.slider("GDP growth change (pp)", -12.0, 4.0, step=0.5, key="sc_gdp")
        st.slider("Inflation change (pp)", -3.0, 8.0, step=0.5, key="sc_cpi")
        st.slider("Call-money (interest) rate change (pp)", -3.0, 5.0, step=0.25, key="sc_call")
        st.slider("Rupee depreciation change (pp)", -6.0, 15.0, step=0.5, key="sc_inr")
        st.select_slider("Shock begins", MONTHS[:10], key="sc_start", help="First month in which the shock starts to build.")
        st.slider("Months to reach full size", 1, 6, key="sc_ramp", help="The shock builds up gradually, like a real downturn.")
        st.multiselect("Lockdown months (COVID-style)", MONTHS, key="sc_lock", help="Months with a lockdown. Each one multiplies defaults several times, two months later.")
        st.multiselect("Second-wave months", MONTHS, key="sc_wave")
        ctx.step(2, "Lender assumptions")
        st.slider("Loss given default (%)", 30, 90, key="sc_lgd", help="Share of a defaulted loan that is never recovered. The project uses 65%.")
        st.slider("Active-loan growth per month (%)", 0.0, 3.0, step=0.01, key="sc_growth", help="How fast the loan book grows during 2025. The project's forecast uses about 1.4% a month.")

    shocks = dict(unemployment_rate=st.session_state.sc_unemp, gdp_growth=st.session_state.sc_gdp, cpi_inflation_yoy=st.session_state.sc_cpi,
                  call_money_rate=st.session_state.sc_call, inr_depreciation_yoy=st.session_state.sc_inr,
                  lockdown=[M2D[m] for m in st.session_state.sc_lock], wave=[M2D[m] for m in st.session_state.sc_wave])
    start = M2D[st.session_state.sc_start].strftime("%Y-%m-%d")
    fc = sc.forecast(res, sc.scenario_macro(macro, shocks, start=start, ramp_months=st.session_state.sc_ramp))
    lgd = st.session_state.sc_lgd / 100
    g = st.session_state.sc_growth / 100
    active = pm["active_loans"].iloc[-1] * (1 + g) ** np.arange(1, 13)
    imp = ctx.impact
    ead = imp.loc["Baseline", "expected credit loss (INR lakh)"] * 1e5 / (imp.loc["Baseline", "expected defaults 2025"] * ctx.LGD)

    def losses(path):
        d = float((path["forecast"].values / 100 * active).sum())
        return d, d * ead * lgd / 1e5
    d0, ecl0 = losses(base)
    d1, ecl1 = losses(fc)
    mult0, mult1 = base["forecast"].mean() / neutral, fc["forecast"].mean() / neutral

    with right:
        ctx.step(3, "What happens")
        a, b, c, d = st.columns(4)
        a.metric("Default rate", f"{fc['forecast'].mean():.2f}%", delta=f"{fc['forecast'].mean() - base['forecast'].mean():+.2f} pp", delta_color="inverse",
                 help="Average monthly default rate in 2025: the share of active loans that turn 90+ days late each month. The change shown is against the baseline.")
        b.metric("Worst month", f"{fc['forecast'].max():.2f}%", delta=f"{sc.FUT[int(np.argmax(fc['forecast'].values))].strftime('%b %Y')}", delta_color="off")
        c.metric("Defaults", f"{d1:,.0f}", delta=f"{d1 - d0:+,.0f}", delta_color="inverse", help="Expected number of loans that default during 2025. The change shown is against the baseline.")
        d.metric("Credit loss (Rs)", cr(ecl1).replace("Rs ", ""), delta=f"{cr(ecl1 - ecl0) if abs(ecl1 - ecl0) >= 0.5 else 'none'}", delta_color="inverse",
                 help="Expected credit loss = expected defaults x average balance owed at default (about Rs 1.12 lakh) x loss given default. The change shown is against the baseline.")

        hist = pm["default_rate"] * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="History (simulated)", line=dict(color="#1f1f1f", width=2)))
        fig.add_trace(go.Scatter(x=list(fc.index) + list(fc.index[::-1]), y=list(fc["upper"]) + list(fc["lower"][::-1]), fill="toself", fillcolor="rgba(155,44,44,.14)",
                                 line=dict(width=0), name="80% range of this scenario", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=base.index, y=base["forecast"], name="Baseline forecast", line=dict(color=NAVY, width=2.4, dash="dash")))
        fig.add_trace(go.Scatter(x=fc.index, y=fc["forecast"], name="This scenario", line=dict(color=BRICK, width=2.8), mode="lines+markers", marker=dict(size=6, symbol="diamond")))
        fig.update_layout(yaxis_title="Defaults as % of active loans, per month", legend=dict(orientation="h", y=-0.2))
        fig.update_yaxes(type="log" if st.toggle("Log scale (makes small values readable)", value=False, key="sc_log") else "linear")
        st.plotly_chart(ctx.style_fig(fig, 360), width="stretch", config={"displayModeBar": False})

        # ---- what each shock contributed
        ctx.step(4, "Why: what each shock did")
        eff = []
        for dname in ("unemployment_rate", "gdp_growth", "cpi_inflation_yoy", "call_money_rate", "inr_depreciation_yoy"):
            if shocks[dname]:
                eff.append((sc.LABELS[dname] + f" {shocks[dname]:+g}", float(np.exp(coef.loc[dname, "beta"] * shocks[dname]) - 1)))
        if shocks["lockdown"]:
            eff.append((f"{len(shocks['lockdown'])} lockdown month(s)", float(np.exp(coef.loc["covid_lockdown", "beta"]) - 1)))
        if shocks["wave"]:
            eff.append((f"{len(shocks['wave'])} second-wave month(s)", float(np.exp(coef.loc["covid_second_wave", "beta"]) - 1)))
        if not eff:
            st.info("No shock is active, so this scenario equals the baseline. Pick a preset or move a slider.")
        else:
            fe = go.Figure(go.Bar(y=[e[0] for e in eff], x=[e[1] * 100 for e in eff], orientation="h", marker_color=[BRICK if e[1] > 0 else NAVY for e in eff],
                                  text=[f"{e[1] * 100:+.0f}%" for e in eff], textposition="outside", cliponaxis=False))
            top = max(abs(e[1]) for e in eff) * 100
            fe.update_layout(xaxis_title="Change in the default rate once the shock is fully in (%)", yaxis=dict(autorange="reversed"),
                             xaxis_range=[min(0, min(e[1] for e in eff) * 100 * 1.3), max(0, max(e[1] for e in eff) * 100 * 1.3) + top * 0.05])
            st.plotly_chart(ctx.style_fig(fe, 80 + 46 * len(eff)), width="stretch", config={"displayModeBar": False})
            st.caption("Each bar is exp(coefficient x shock) - 1 from the fitted model. Lockdown months act in the single month they apply; the other bars apply for as long as the shock lasts.")

        # ---- plain-language explanation
        lines = []
        first = M2D[st.session_state.sc_start]
        lag_month = (first + pd.DateOffset(months=sc.LAG)).strftime("%b %Y")
        if eff:
            lines.append(f"A shock that begins in <b>{st.session_state.sc_start}</b> reaches the default rate from <b>{lag_month}</b>, because a loan only becomes a 90+ day default "
                         f"about two months after the first missed instalment (30 days late, then 60, then 90). The shock builds over {st.session_state.sc_ramp} month(s).")
            for dname in ("unemployment_rate", "gdp_growth", "cpi_inflation_yoy", "call_money_rate", "inr_depreciation_yoy"):
                if shocks[dname]:
                    bt = coef.loc[dname, "beta"]
                    lines.append(f"<b>{sc.LABELS[dname]} {shocks[dname]:+g}</b> multiplies the default rate by {np.exp(bt * shocks[dname]):.2f} "
                                 f"(the model's coefficient is {bt:+.3f} per point, so each point moves defaults {np.exp(bt) - 1:+.1%}).")
            if shocks["lockdown"]:
                lines.append(f"<b>Lockdown</b> months were the biggest single effect in the real history (coefficient {coef.loc['covid_lockdown', 'beta']:+.2f}, "
                             f"about {np.exp(coef.loc['covid_lockdown', 'beta']):.1f} times the default rate in that month).")
        lines.append(f"Overall the average monthly default rate for 2025 goes from <b>{base['forecast'].mean():.2f}%</b> to <b>{fc['forecast'].mean():.2f}%</b>, "
                     f"so <b>{d1:,.0f}</b> loans are expected to default instead of {d0:,.0f}. With loss given default of {st.session_state.sc_lgd}%, the expected credit loss is "
                     f"<b>{money(ecl1)}</b> against {money(ecl0)} in the baseline.")
        st.markdown('<div class="callout mid"><b>In plain words.</b><br>' + "<br>".join(lines) + "</div>", unsafe_allow_html=True)

    # ---- decisions and profit on the 6,000 unseen borrowers
    st.markdown("### How the lender's decisions change")
    ctx.explain("The same 6,000 borrowers the model never saw. The economy multiplies every applicant's default chance (the multiplier is the scenario's forecast rate divided by the "
                "neutral-economy rate of 0.485%). Then the usual rule decides: approve, approve with conditions or decline. Profit uses the break-even logic: "
                "(1 - default chance) x margin - default chance x loss given default, per rupee lent.")
    tb = ctx.test_borrowers
    dec_b, pit_b, prof_b = run_policy(ctx, tb, mult0, lgd)
    dec_s, pit_s, prof_s = run_policy(ctx, tb, mult1, lgd)
    rows = []
    cb = counts(dec_b, pit_b, prof_b)
    cs = counts(dec_s, pit_s, prof_s)
    acc_old = np.isin(dec_b, ["APPROVE", "CONDITIONS"])
    cd = counts(dec_b, pit_s, prof_s, accept=acc_old)
    for label, cc, mm in (("Baseline economy, rules re-run", cb, mult0), ("This scenario, rules re-run (lender adapts)", cs, mult1),
                          ("This scenario, baseline decisions kept (lender does nothing)", cd, mult1)):
        rows.append((label, f"{mm:.2f}", f"{cc['approve']:,}", f"{cc['cond']:,}", f"{cc['decline']:,}", f"{cc['pd_acc']:.1%}", money(cc["profit"])))
    table(pd.DataFrame(rows, columns=["Case", "Multiplier", "Approve", "With conditions", "Decline", "Avg default chance (accepted)", "Expected profit (accepted)"]),
          widths=[28, 10, 9, 12, 9, 16, 16])
    flips = int(((dec_b == "APPROVE") & (dec_s != "APPROVE")).sum())
    saved = cs["profit"] - cd["profit"]
    st.markdown(f'<div class="callout {"ok" if saved >= 0 else ""}"><b>Reading the table.</b> In this scenario <b>{flips:,}</b> of the 6,000 borrowers who were plain approvals in the baseline are no longer '
                f'plain approvals. If the lender adapts its rules, expected profit on accepted loans is <b>{money(cs["profit"])}</b>; if it keeps the baseline decisions it is '
                f'<b>{money(cd["profit"])}</b>. Adapting is worth <b>{money(saved)}</b>. Expected profit uses the model\'s probabilities, not the real outcomes.</div>', unsafe_allow_html=True)

    with st.expander("How this was computed, and the fitted model"):
        st.markdown("""
**Model.** log(default rate %) at month t = c + sum of (coefficient x driver two months earlier) + an error that follows a seasonal ARMA pattern, SARIMAX(1,0,1)(1,0,0)12.
Because the model works on the log, a shock multiplies the default rate by exp(coefficient x shock). The forecast for the baseline and for any scenario differ by exactly that factor,
so the lab is exact, not an approximation. The model is refitted from `portfolio_monthly.csv` and `macro_india_monthly.csv` each time the server starts (see `dashboard/scenario_core.py`).

**Expected credit loss** = expected defaults x average balance owed at default x loss given default. Expected defaults = monthly default rate x active loans, summed over 2025.
""")
        ct = coef.copy()
        ct["Meaning"] = [f"{np.exp(b) - 1:+.1%} per +1 point" if not n.startswith("covid") else f"x{np.exp(b):.2f} while active" for n, b in zip(ct.index, ct["beta"])]
        ct.insert(0, "Driver (two months earlier)", [sc.LABELS[i] for i in ct.index])
        ct["beta"] = ct["beta"].map("{:+.4f}".format)
        ct["p"] = ct["p"].map("{:.4f}".format)
        table(ct.rename(columns={"beta": "Coefficient", "p": "p-value"}).reset_index(drop=True), widths=[34, 16, 14, 36])
        st.caption(f"Fit on all 72 months: AIC {res.aic:.2f}, BIC {res.bic:.2f}. Neutral-economy rate {neutral:.4f}% a month. "
                   "Presets are checked against models/ts_forecast_scenarios.csv: the largest difference is 0.00000.")
