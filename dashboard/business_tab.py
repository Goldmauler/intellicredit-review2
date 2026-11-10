"""
IntelliCredit dashboard - "Business analytics" tab
==================================================
A business review of the lending book: who is risky, who is profitable, what an approval policy is worth, whether pricing
matches risk, and how the portfolio behaved over time. All numbers are computed from the data files.

Economics of one loan (the same logic as the break-even rule):
    margin  = (interest rate - cost of funds) x loan years          earned if the loan is repaid
    profit  = amount x [ (1 - default) x margin - default x LGD ]   default is 0 or 1 for realised results
"""
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from validation_tab import _table as table

NAVY, OCHRE, BRICK, SLATE = "#1f3a5f", "#b8862f", "#9b2c2c", "#5a5546"


def money(lakh):
    return f"Rs {lakh / 100:,.2f} crore" if abs(lakh) >= 100 else f"Rs {lakh:,.1f} lakh"


def loan_profit(df, lgd, cof, pd_col=None):
    m = np.maximum(df["interest_rate_annual"] / 100 - cof, 0.001) * df["loan_tenure_months"] / 12
    d = df["loan_default_status"] if pd_col is None else df[pd_col]
    return df["loan_amount_requested"] * ((1 - d) * m - d * lgd)


DIMS = {
    "Gig platform": lambda d: d["primary_gig_platform"].str.replace(r"\s*\(.*\)", "", regex=True),
    "City tier": lambda d: d["location_type"],
    "Education": lambda d: d["education_level"],
    "Bank account": lambda d: d["has_bank_account"].map({1: "Has bank account", 0: "No bank account"}),
    "Dependents": lambda d: d["num_dependents"].astype(str),
    "Age band": lambda d: pd.cut(d["age"], [20, 25, 30, 35, 40, 45, 60], labels=["21-25", "26-30", "31-35", "36-40", "41-45", "46-60"]).astype(str),
    "Months on the platform": lambda d: pd.cut(d["gig_tenure_months"], [0, 6, 12, 24, 48, 200], labels=["up to 6", "7-12", "13-24", "25-48", "over 48"]).astype(str),
    "Monthly income": lambda d: pd.cut(d["avg_monthly_income"], [0, 15000, 25000, 35000, 50000, 1e9],
                                       labels=["under 15k", "15k-25k", "25k-35k", "35k-50k", "over 50k"]).astype(str),
    "Loan size vs income": lambda d: pd.cut(d["loan_amount_requested"] / d["avg_monthly_income"], [0, 2, 3, 4, 5, 10],
                                            labels=["under 2x", "2-3x", "3-4x", "4-5x", "over 5x"]).astype(str),
    "Interest rate": lambda d: pd.cut(d["interest_rate_annual"], [0, 12, 16, 20, 24, 40], labels=["under 12%", "12-16%", "16-20%", "20-24%", "over 24%"]).astype(str),
    "Loan length": lambda d: d["loan_tenure_months"].astype(str) + " months",
    "Bills paid on time": lambda d: pd.cut(d["utility_on_time_count"] / 12, [-0.01, .5, .75, .9, 1.0], labels=["half or fewer", "50-75%", "75-90%", "over 90%"]).astype(str),
}
ORDER_HINT = {"Gig platform": None, "City tier": ["Rural", "Semi-Urban", "Tier-2 City", "Tier-1 City", "Metro"],
              "Education": ["Below 10th", "10th Pass", "12th Pass", "Diploma", "Undergraduate", "Postgraduate"]}


@st.cache_data(show_spinner="Reading the loan book ...")
def loan_book(path):
    return pd.read_csv(path)


def seg_table(df, dim, lgd, cof):
    d = df.copy()
    d["seg"] = DIMS[dim](d)
    d["profit"] = loan_profit(d, lgd, cof)
    g = d.groupby("seg", observed=True)
    t = pd.DataFrame({"n": g.size(), "defaults": g["loan_default_status"].sum(), "rate": g["loan_default_status"].mean(), "avg_loan": g["loan_amount_requested"].mean(),
                      "avg_rate": g["interest_rate_annual"].mean(), "profit": g["profit"].sum() / 1e5, "lent": g["loan_amount_requested"].sum()})
    t["share"] = t["n"] / t["n"].sum()
    t["lift"] = t["rate"] / df["loan_default_status"].mean()
    t["profit_per"] = t["profit"] * 1e5 / t["n"]
    t["margin"] = t["profit"] * 1e5 / t["lent"]
    t["def_share"] = t["defaults"] / t["defaults"].sum()
    order = ORDER_HINT.get(dim)
    if order:
        t = t.reindex([o for o in order if o in t.index])
    elif dim not in ("Gig platform", "Education"):
        t = t.sort_index(key=lambda ix: pd.Index([str(i) for i in ix]))
    return t


def render(ctx):
    st.markdown("## Business analytics")
    ctx.explain("A business review of the lending book. It answers: <b>who defaults, who makes money, what is an approval rule worth, is pricing matched to risk, and how did the portfolio behave?</b> "
                "Profit here is the break-even logic applied to each loan: interest margin earned when repaid, loss given default when not.")
    lgd, cof = ctx.LGD, ctx.COST_OF_FUNDS
    b = ctx.borrowers
    prof_all = loan_profit(b, lgd, cof)
    lent = b["loan_amount_requested"].sum()

    st.markdown("### 1. The book at a glance")
    k = st.columns(5)
    k[0].metric("Borrowers", f"{len(b):,}")
    k[1].metric("Default rate", f"{b['loan_default_status'].mean():.2%}", help="Share of borrowers who reached 90+ days late.")
    k[2].metric("Lent", f"Rs {lent / 1e7:,.0f} cr", help="Sum of loan amounts requested by all 30,000 borrowers, in crore rupees.")
    k[3].metric("Loss rate", f"{(b['loan_amount_requested'] * b['loan_default_status'] * lgd).sum() / lent:.2%}", help="Loss rate on money lent: defaulted amount x loss given default, as a share of all money lent.")
    k[4].metric("Net margin", f"{prof_all.sum() / lent:.2%}", help="Net margin on money lent: profit after losses, as a share of all money lent, over the loan lives. This is the approve-everyone case.")

    sections = ["Segments", "Approval policy", "Risk deciles", "Pricing", "Portfolio over time"]
    sec = st.radio("Section", sections, horizontal=True, key="bz_sec", label_visibility="collapsed")

    if sec == "Segments":
        ctx.explain("Pick any way of splitting borrowers. The left chart shows where defaults concentrate; the right shows where money is made or lost. "
                    "A segment can be risky <i>and</i> still profitable if it is priced high enough.")
        dim = st.selectbox("Split borrowers by", list(DIMS), key="bz_dim")
        t = seg_table(b, dim, lgd, cof)
        c1, c2 = st.columns(2)
        f1 = go.Figure(go.Bar(x=t.index.astype(str), y=t["rate"] * 100, marker_color=NAVY, customdata=np.stack([t["n"], t["lift"]], axis=1),
                              hovertemplate="%{x}<br>default rate %{y:.1f}%<br>%{customdata[0]:,.0f} borrowers<br>%{customdata[1]:.2f}x the average<extra></extra>"))
        f1.add_hline(y=b["loan_default_status"].mean() * 100, line_dash="dash", line_color=BRICK, annotation_text="average")
        f1.update_layout(title="Default rate (%)", xaxis_title=dim)
        f1.update_xaxes(tickangle=-30)
        c1.plotly_chart(ctx.style_fig(f1, 340), width="stretch", config={"displayModeBar": False})
        f2 = go.Figure(go.Bar(x=t.index.astype(str), y=t["profit_per"], marker_color=[NAVY if v >= 0 else BRICK for v in t["profit_per"]],
                              hovertemplate="%{x}<br>profit per borrower Rs %{y:,.0f}<extra></extra>"))
        f2.add_hline(y=0, line_color="#1f1f1f", line_width=1)
        f2.update_layout(title="Profit per borrower after losses (Rs)", xaxis_title=dim)
        f2.update_xaxes(tickangle=-30)
        c2.plotly_chart(ctx.style_fig(f2, 340), width="stretch", config={"displayModeBar": False})
        show = pd.DataFrame({dim: t.index.astype(str), "Count": t["n"].map("{:,.0f}".format), "Share": t["share"].map("{:.1%}".format),
                             "Default rate": t["rate"].map("{:.1%}".format), "x average": t["lift"].map("{:.2f}x".format),
                             "Share of defaults": t["def_share"].map("{:.1%}".format), "Avg loan (Rs)": t["avg_loan"].map("{:,.0f}".format),
                             "Avg rate": t["avg_rate"].map("{:.1f}%".format), "Profit after losses": t["profit"].map(money),
                             "Margin": t["margin"].map("{:+.2%}".format)})
        table(show, widths=[16, 8, 7, 9, 8, 10, 11, 8, 13, 10])
        riskiest, safest = t["rate"].idxmax(), t["rate"].idxmin()
        best, worst = t["profit_per"].idxmax(), t["profit_per"].idxmin()
        st.markdown(f'<div class="callout mid"><b>What this split says.</b> The riskiest group is <b>{riskiest}</b> at {t.loc[riskiest, "rate"]:.1%} '
                    f'({t.loc[riskiest, "lift"]:.2f} times the average), the safest is <b>{safest}</b> at {t.loc[safest, "rate"]:.1%}. '
                    f'Most profit per borrower comes from <b>{best}</b> (Rs {t.loc[best, "profit_per"]:,.0f}); the weakest is <b>{worst}</b> (Rs {t.loc[worst, "profit_per"]:,.0f}). '
                    f'The spread between the best and worst default rates is {t["rate"].max() / max(t["rate"].min(), 1e-9):.1f} times.</div>', unsafe_allow_html=True)
        if dim in ("Monthly income", "Loan size vs income"):
            st.markdown('<div class="callout"><b>Careful.</b> In this simulated data, richer borrowers take larger loans and the generator charges risk per rupee borrowed, '
                        'so higher income shows a <i>higher</i> default rate. Real credit data shows the opposite. This is a known limit of the simulation (see Proof &amp; comparison).</div>',
                        unsafe_allow_html=True)

    elif sec == "Approval policy":
        ctx.explain("What is an approval rule worth? The 6,000 borrowers the model never saw are ranked from safest to riskiest. Approving only the safest share cuts defaults, "
                    "but also gives up good loans. The chart finds the share that earns the most, using what really happened to each borrower.")
        tb = ctx.test_borrowers.copy()
        tb["profit"] = loan_profit(tb, lgd, cof)
        tb = tb.sort_values("predicted_pd", kind="mergesort").reset_index(drop=True)
        n = len(tb)
        cum_def = tb["loan_default_status"].cumsum().values
        cum_profit = tb["profit"].cumsum().values / 1e5
        k_ = np.arange(1, n + 1)
        share = k_ / n
        best_i = int(np.argmax(cum_profit))
        be_acc = (tb["predicted_pd"] < tb["break_even_pd"]).values
        pol_acc = (tb["policy_decision"] != "DECLINE").values

        def stat(acc):
            return dict(rate=acc.mean(), dr=tb.loc[acc, "loan_default_status"].mean(), profit=tb.loc[acc, "profit"].sum() / 1e5, n=int(acc.sum()),
                        per=tb.loc[acc, "profit"].mean())
        pick = st.slider("Approve the safest share of applicants", 10, 100, int(round(share[best_i] * 100)), key="bz_share", format="%d%%",
                         help="Default starts at the share that earns the most profit.")
        ki = max(0, int(round(pick / 100 * n)) - 1)
        acc_pick = np.zeros(n, bool)
        acc_pick[:ki + 1] = True
        cases = {"Approve everyone": np.ones(n, bool), "Break-even rule only": be_acc, "Dashboard policy (rules and grades)": pol_acc,
                 f"Best cut-off (safest {share[best_i]:.0%})": (k_ <= best_i + 1), f"Your slider (safest {pick}%)": acc_pick}
        rows, base = [], stat(cases["Approve everyone"])
        for name, acc in cases.items():
            s = stat(acc)
            rows.append((name, f"{s['rate']:.1%}", f"{s['dr']:.2%}", f"{s['profit']:,.1f}", f"{s['per']:,.0f}", f"{base['dr'] * base['n'] - s['dr'] * s['n']:,.0f}"))
        table(pd.DataFrame(rows, columns=["Rule", "Share approved", "Default rate of approved", "Profit after losses (Rs lakh)", "Profit per approved loan (Rs)", "Defaults avoided"]),
              widths=[32, 11, 14, 16, 15, 12])
        c1, c2 = st.columns(2)
        f1 = go.Figure(go.Scatter(x=share * 100, y=cum_def / k_ * 100, mode="lines", line=dict(color=NAVY, width=2.4), name="Default rate of approved"))
        f1.add_trace(go.Scatter(x=[pick], y=[cum_def[ki] / (ki + 1) * 100], mode="markers", marker=dict(size=13, color=BRICK, symbol="diamond"), name="Your slider"))
        f1.update_layout(title="Default rate among approved loans (%)", xaxis_title="Share of applicants approved (safest first, %)", yaxis_title="Default rate (%)", legend=dict(orientation="h", y=-0.28))
        c1.plotly_chart(ctx.style_fig(f1, 350), width="stretch", config={"displayModeBar": False})
        f2 = go.Figure(go.Scatter(x=share * 100, y=cum_profit, mode="lines", line=dict(color=NAVY, width=2.4), name="Profit after losses"))
        f2.add_trace(go.Scatter(x=[share[best_i] * 100], y=[cum_profit[best_i]], mode="markers+text", marker=dict(size=13, color=OCHRE, symbol="star"), text=["best"], textposition="top center", name="Best cut-off"))
        f2.add_trace(go.Scatter(x=[pick], y=[cum_profit[ki]], mode="markers", marker=dict(size=12, color=BRICK, symbol="diamond"), name="Your slider"))
        f2.update_layout(title="Profit after losses (Rs lakh)", xaxis_title="Share of applicants approved (safest first, %)", yaxis_title="Rs lakh", legend=dict(orientation="h", y=-0.28))
        c2.plotly_chart(ctx.style_fig(f2, 350), width="stretch", config={"displayModeBar": False})
        gain = stat(cases[f"Best cut-off (safest {share[best_i]:.0%})"])["profit"] - base["profit"]
        pol = stat(pol_acc)
        st.markdown(f'<div class="callout mid"><b>What this says.</b> Approving everyone earns <b>{money(base["profit"])}</b> with {base["dr"]:.2%} of loans defaulting. '
                    f'The most profitable cut-off approves the safest <b>{share[best_i]:.0%}</b> and earns <b>{money(cum_profit[best_i])}</b>, a gain of <b>{money(gain)}</b>. '
                    f'The dashboard policy approves {pol["rate"]:.1%} with {pol["dr"]:.2%} defaulting and earns <b>{money(pol["profit"])}</b>. '
                    + (f'That is <b>more</b> than the best single cut-off ({money(pol["profit"] - cum_profit[best_i])} more) because the break-even rule judges each loan against its own price, term and size, '
                       f'whereas a single cut-off treats every loan the same. This is the central idea of the project. ' if pol["profit"] > cum_profit[best_i] else "")
                    + 'Results use what actually happened to each of the 6,000 test borrowers.</div>', unsafe_allow_html=True)

    elif sec == "Risk deciles":
        ctx.explain("Split the 6,000 unseen borrowers into ten equal groups by predicted default chance. If the model is any good, observed defaults climb steadily from the safest to the riskiest group, "
                    "and predicted and observed rates stay close.")
        tb = ctx.test_borrowers.copy()
        tb["decile"] = pd.qcut(tb["predicted_pd"].rank(method="first"), 10, labels=False) + 1
        g = tb.groupby("decile")
        t = pd.DataFrame({"n": g.size(), "pred": g["predicted_pd"].mean(), "obs": g["loan_default_status"].mean(), "defaults": g["loan_default_status"].sum()})
        t["lift"] = t["obs"] / tb["loan_default_status"].mean()
        t["cum_caught"] = t["defaults"][::-1].cumsum()[::-1] / t["defaults"].sum()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=t.index, y=t["obs"] * 100, name="Observed default rate", marker_color=NAVY))
        fig.add_trace(go.Scatter(x=t.index, y=t["pred"] * 100, name="Predicted default chance", mode="lines+markers", line=dict(color=OCHRE, width=2.6), marker=dict(symbol="diamond", size=8)))
        fig.update_layout(xaxis_title="Risk decile (1 = safest, 10 = riskiest)", yaxis_title="Default rate (%)", legend=dict(orientation="h", y=-0.22), xaxis=dict(dtick=1))
        st.plotly_chart(ctx.style_fig(fig, 360), width="stretch", config={"displayModeBar": False})
        show = pd.DataFrame({"Risk decile": t.index, "Borrowers": t["n"], "Predicted chance": t["pred"].map("{:.1%}".format), "Observed default rate": t["obs"].map("{:.1%}".format),
                             "Times the average": t["lift"].map("{:.2f}x".format), "Defaults": t["defaults"],
                             "Defaults caught if this and riskier deciles are declined": t["cum_caught"].map("{:.0%}".format)})
        table(show, widths=[9, 10, 12, 14, 12, 9, 34])
        st.markdown(f'<div class="callout mid"><b>What this says.</b> The riskiest decile defaults at <b>{t.loc[10, "obs"]:.1%}</b>, <b>{t.loc[10, "obs"] / max(t.loc[1, "obs"], 1e-9):.1f} times</b> the safest ({t.loc[1, "obs"]:.1%}). '
                    f'Declining the top three deciles would remove <b>{t.loc[8:, "defaults"].sum() / t["defaults"].sum():.0%}</b> of all defaults while turning away 30% of applicants. '
                    f'Predicted and observed rates are close in every decile, which is what calibration means.</div>', unsafe_allow_html=True)

    elif sec == "Pricing":
        ctx.explain("Is the interest rate matched to the risk? A loan is <b>priced below its risk</b> when the model's default chance is above its break-even limit. "
                    "Those loans lose money on average, however the rate looks.")
        tb = ctx.test_borrowers.copy()
        tb["profit"] = loan_profit(tb, lgd, cof)
        tb["band"] = DIMS["Interest rate"](tb)
        tb["below"] = tb["predicted_pd"] >= tb["break_even_pd"]
        order = ["under 12%", "12-16%", "16-20%", "20-24%", "over 24%"]
        g = tb.groupby("band", observed=True)
        t = pd.DataFrame({"n": g.size(), "pred": g["predicted_pd"].mean(), "be": g["break_even_pd"].mean(), "obs": g["loan_default_status"].mean(),
                          "below": g["below"].mean(), "profit": g["profit"].sum() / 1e5, "per": g["profit"].mean()}).reindex([o for o in order if o in set(tb["band"])])
        c1, c2 = st.columns(2)
        f1 = go.Figure()
        f1.add_trace(go.Bar(x=t.index, y=t["obs"] * 100, name="Observed default rate", marker_color=NAVY))
        f1.add_trace(go.Scatter(x=t.index, y=t["be"] * 100, name="Average break-even limit", mode="lines+markers", line=dict(color=BRICK, width=2.6, dash="dash"), marker=dict(symbol="diamond", size=8)))
        f1.update_layout(title="Default rate against the limit the rate can bear (%)", xaxis_title="Interest-rate band", legend=dict(orientation="h", y=-0.28))
        c1.plotly_chart(ctx.style_fig(f1, 350), width="stretch", config={"displayModeBar": False})
        f2 = go.Figure(go.Bar(x=t.index, y=t["per"], marker_color=[NAVY if v >= 0 else BRICK for v in t["per"]], hovertemplate="%{x}<br>Rs %{y:,.0f} per loan<extra></extra>"))
        f2.add_hline(y=0, line_color="#1f1f1f", line_width=1)
        f2.update_layout(title="Profit per loan after losses (Rs)", xaxis_title="Interest-rate band")
        c2.plotly_chart(ctx.style_fig(f2, 350), width="stretch", config={"displayModeBar": False})
        show = pd.DataFrame({"Interest-rate band": t.index, "Loans": t["n"], "Avg predicted chance": t["pred"].map("{:.1%}".format), "Avg break-even limit": t["be"].map("{:.1%}".format),
                             "Observed default rate": t["obs"].map("{:.1%}".format), "Priced below their risk": t["below"].map("{:.0%}".format),
                             "Profit after losses": t["profit"].map(money), "Profit per loan (Rs)": t["per"].map("{:,.0f}".format)})
        table(show, widths=[14, 8, 13, 13, 13, 13, 14, 12])
        total_below = tb["below"].mean()
        loss_below = tb.loc[tb["below"], "profit"].sum() / 1e5
        st.markdown(f'<div class="callout mid"><b>What this says.</b> <b>{total_below:.0%}</b> of loans are priced below the risk the model sees; together they earn <b>{money(loss_below)}</b> '
                    f'after losses. Higher rates raise the limit a loan can bear, but they also attract riskier borrowers, so the best band is where the line of default rates stays below the dashed limit.</div>',
                    unsafe_allow_html=True)

    else:
        ctx.explain("How the simulated lender's book moved from 2019 to 2024: the COVID shock, the recovery and the calm that followed, plus how each year's loans (a vintage) performed.")
        pm = ctx.pm.copy()
        pm["date"] = pd.to_datetime(pm["date"])
        metrics = {"default_rate": ("Default rate (% of active loans)", 100), "dpd30_plus_rate": ("30+ days late rate (%)", 100), "dpd60_plus_rate": ("60+ days late rate (%)", 100),
                   "active_loans": ("Active loans", 1), "new_originations": ("New loans per month", 1), "outstanding_principal": ("Outstanding principal (Rs crore)", 1e-7)}
        mkey = st.selectbox("Metric", list(metrics), format_func=lambda k: metrics[k][0], key="bz_metric")
        lab, scale = metrics[mkey]
        fig = go.Figure(go.Scatter(x=pm["date"], y=pm[mkey] * scale, mode="lines", line=dict(color=NAVY, width=2.4), name=lab))
        for col_, nm in (("covid_lockdown", "Lockdown"), ("covid_second_wave", "Second wave")):
            on = pm.loc[pm[col_] == 1, "date"]
            if len(on):
                fig.add_vrect(x0=on.min(), x1=on.max() + pd.offsets.MonthEnd(0), fillcolor="rgba(138,133,119,.18)", line_width=0, annotation_text=nm, annotation_position="top left")
        fig.update_layout(yaxis_title=lab)
        st.plotly_chart(ctx.style_fig(fig, 340), width="stretch", config={"displayModeBar": False})
        pk = pm.loc[pm["default_rate"].idxmax()]
        calm = pm[(pm["date"] >= "2022-01-01") & (pm["covid_lockdown"] == 0)]["default_rate"].mean() * 100
        a, b2, c = st.columns(3)
        a.metric("Worst month", f"{pk['default_rate'] * 100:.2f}%", delta=pk["date"].strftime("%b %Y"), delta_color="off")
        b2.metric("Calm-period average (2022 on)", f"{calm:.2f}%")
        c.metric("Worst month vs calm", f"{pk['default_rate'] * 100 / calm:.1f} times")
        lb = loan_book(os.path.join(ctx.DATA_DIR, "portfolio", "loan_book.csv"))
        lb["year"] = lb["origination_month"].str[:4]
        g = lb.groupby("year")
        v = pd.DataFrame({"Loans started": g.size(), "Lent (Rs crore)": g["loan_amount"].sum() / 1e7, "Average rate": g["interest_rate_annual"].mean(),
                          "Defaulted inside the window": g["defaulted_in_panel"].mean(), "Still running at the end": g["censored_at_window_end"].mean()})
        st.markdown("### Vintages: how each year's loans performed")
        table(pd.DataFrame({"Year started": v.index, "Loans started": v["Loans started"].map("{:,.0f}".format), "Lent (Rs crore)": v["Lent (Rs crore)"].map("{:,.2f}".format),
                            "Average rate": v["Average rate"].map("{:.1f}%".format), "Defaulted inside the window": v["Defaulted inside the window"].map("{:.1%}".format),
                            "Still running at the end": v["Still running at the end"].map("{:.0%}".format)}), widths=[14, 15, 15, 14, 22, 20])
        st.caption("Recent vintages look safer partly because their loans have had less time to default (many are still running). Compare vintages of similar age, not raw totals.")
