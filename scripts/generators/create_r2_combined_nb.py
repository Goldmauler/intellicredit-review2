"""Builds notebooks/08_R2_combined_insights_dashboard.ipynb (Review 2 - combined insights + dynamic visuals)."""
import os

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


md("""
# Review 2 · Combined Business Insights & Dynamic Visualisation
**Rubric:** Combined business insights and recommendations (2) · Interactive dashboard or dynamic visualisation (1)

The three analytical layers answer different questions, and together they make one underwriting system:

| Layer | Question | Output used here |
|---|---|---|
| Review 1 — PD model | **Who** is likely to default? | calibrated PD per applicant (`models/underwriting_model.joblib`) |
| Review 2 · Method 2 — association rules | **Which combinations** of behaviours signal trouble? | 93 validated warning rules (`models/association_rules.csv`) |
| Review 2 · Method 1 — time-series | **When** does portfolio risk rise, and by how much? | 2025 scenario forecasts & PD multipliers (`models/macro_overlay.json`) |

The interactive version of everything below is the Streamlit **underwriter dashboard** (`streamlit run dashboard/app.py`); this notebook also exports a standalone interactive HTML report to `reports/interactive_dashboard.html`.
""")

code("""
import os, sys, json
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.getcwd())
sys.path.insert(0, ROOT)
from src.config import DATA_DIR, MODELS_DIR, RANDOM_STATE, TEST_SIZE
from src.underwriting_model import build_features, break_even_pd, risk_band
from src.risk_rules import load_rules, flag_items, rules_hit_count

REPORTS_DIR = os.path.join(ROOT, 'reports'); os.makedirs(REPORTS_DIR, exist_ok=True)

borrowers = pd.read_csv(os.path.join(DATA_DIR, 'borrower_profiles.csv'))
pm = pd.read_csv(os.path.join(DATA_DIR, 'portfolio', 'portfolio_monthly.csv'), parse_dates=['date'])
scen = pd.read_csv(os.path.join(MODELS_DIR, 'ts_forecast_scenarios.csv'), parse_dates=['date'])
impact = pd.read_csv(os.path.join(MODELS_DIR, 'ts_scenario_impact.csv'), index_col=0)
with open(os.path.join(MODELS_DIR, 'macro_overlay.json')) as f: overlay = json.load(f)
item_defs, rules = load_rules()
with open(os.path.join(MODELS_DIR, 'underwriting_model_card.json')) as f: card = json.load(f)
model = joblib.load(os.path.join(MODELS_DIR, 'underwriting_model.joblib'))

_, test_idx = train_test_split(borrowers.index, test_size=TEST_SIZE, stratify=borrowers['loan_default_status'], random_state=RANDOM_STATE)
test = borrowers.loc[test_idx].copy()
test['pd_ttc'] = model.predict_proba(build_features(test))[:, 1]
test['break_even_pd'] = break_even_pd(test['interest_rate_annual'], test['loan_tenure_months'])
print(f"underwriting model test AUC {card['test_auc']} | {len(rules)} rules | scenario multipliers {overlay['multipliers_next_12m']}")
""")

md("""
## 1. Flagging every applicant with the warning rules
The item definitions saved by the association-rule notebook are re-applied to the test applicants, so each one gets a **PD** *and* a list of **triggered warning rules**.
""")

code("""
flags = flag_items(test, item_defs)
test['rules_hit'] = rules_hit_count(flags, rules)
test['rule_bucket'] = pd.cut(test['rules_hit'], [-1, 0, 2, 5, 1000], labels=['0', '1-2', '3-5', '6+'])
display(test.groupby('rule_bucket', observed=True).agg(applicants=('pd_ttc', 'size'), mean_PD=('pd_ttc', 'mean'), observed_default_rate=('loan_default_status', 'mean')).style.format({'mean_PD': '{:.1%}', 'observed_default_rate': '{:.1%}'}))
""")

md("""
## 2. Do rules add information beyond the PD?
Within each PD band, compare the observed default rate of applicants with few vs many triggered rules. If rules were redundant with the PD, the rows would be flat.
""")

code("""
test['risk_band'] = test['pd_ttc'].apply(risk_band)
cross = test.pivot_table(index='risk_band', columns='rule_bucket', values='loan_default_status', aggfunc='mean', observed=True)
counts = test.pivot_table(index='risk_band', columns='rule_bucket', values='loan_default_status', aggfunc='size', observed=True)
display(cross.style.format('{:.1%}', na_rep='-').background_gradient(cmap='Reds', axis=None))
display(counts.rename_axis(columns='applicants per cell'))
""")

md("""
## 3. Macro-aware underwriting — what the forecast does to approvals
Point-in-time PD = through-the-cycle PD × scenario multiplier from the time-series model (forecast default rate ÷ the model's neutral-economy rate). Keeping the same break-even rule, the policy **automatically tightens in a downturn and loosens in a benign year**.
""")

code("""
rows = []
for scen_name, mult in [('Neutral economy', 1.0)] + list(overlay['multipliers_next_12m'].items()):
    pit = (test['pd_ttc'] * mult).clip(upper=0.99)
    approve = pit < test['break_even_pd']
    rows.append({'scenario': scen_name, 'PD multiplier': mult, 'approval rate': approve.mean(),
                 'observed default rate of approved': test.loc[approve, 'loan_default_status'].mean(),
                 'expected loss per approved loan (INR)': (pit[approve] * 0.65 * test.loc[approve, 'loan_amount_requested']).mean()})
policy = pd.DataFrame(rows).set_index('scenario')
display(policy.style.format({'PD multiplier': '{:.2f}', 'approval rate': '{:.1%}', 'observed default rate of approved': '{:.2%}', 'expected loss per approved loan (INR)': '{:,.0f}'}))
""")

md("""
## 4. Dynamic visualisations
### 4a. Portfolio default rate — history and 2025 scenarios (toggle scenarios in the legend, zoom with the range slider)
""")

code("""
colors = {'Baseline': '#4c72b0', 'Adverse': '#dd8452', 'Severe': '#c44e52'}
fig_ts = go.Figure()
fig_ts.add_trace(go.Scatter(x=pm['date'], y=pm['default_rate'] * 100, name='Actual default rate', line=dict(color='black', width=2.5)))
fig_ts.add_trace(go.Scatter(x=pm['date'], y=pm['dpd30_plus_rate'] * 100, name='30+ DPD (early warning)', line=dict(color='#937860', width=1.5, dash='dot'), visible='legendonly'))
for name, g in scen.groupby('scenario'):
    fig_ts.add_trace(go.Scatter(x=pd.concat([g['date'], g['date'][::-1]]), y=pd.concat([g['upper80'], g['lower80'][::-1]]), fill='toself',
                                fillcolor=colors[name], opacity=0.15, line=dict(width=0), name=f'{name} 80% band', legendgroup=name, showlegend=False, hoverinfo='skip'))
    fig_ts.add_trace(go.Scatter(x=g['date'], y=g['forecast'], name=f'{name} forecast', line=dict(color=colors[name], width=2.5), legendgroup=name))
for start, end, label in [('2020-03-01', '2020-08-01', 'COVID lockdown'), ('2021-04-01', '2021-07-01', 'Second wave')]:
    fig_ts.add_vrect(x0=start, x1=end, fillcolor='grey', opacity=0.12, line_width=0, annotation_text=label, annotation_position='top left')
fig_ts.update_layout(title='Monthly 90+ DPD default rate: history (simulated book, real macro) and 2025 SARIMAX scenarios', yaxis_title='% of active loans',
                     xaxis=dict(rangeslider=dict(visible=True)), hovermode='x unified', height=520, template='plotly_white')
fig_ts.show()
""")

md("""
### 4b. Warning-rule explorer (hover a point to read the rule)
""")

code("""
fig_rules = px.scatter(rules, x='test_coverage', y='test_confidence', size='support', color='test_lift', hover_name='antecedent',
                       hover_data={'lift': ':.2f', 'test_lift': ':.2f', 'test_coverage': ':.2%', 'test_confidence': ':.1%', 'support': ':.2%'},
                       color_continuous_scale='Reds', labels={'test_coverage': 'share of applicants matching (test)', 'test_confidence': 'default rate when matched (test)'},
                       title='93 validated warning rules - coverage vs default rate on unseen borrowers')
fig_rules.add_hline(y=item_defs['base_default_rate'], line_dash='dash', annotation_text='portfolio average')
fig_rules.update_layout(height=520, template='plotly_white', xaxis_tickformat='.0%', yaxis_tickformat='.0%')
fig_rules.show()
""")

md("""
### 4c. Macro drivers from the APIs (dropdown to switch indicator)
""")

code("""
macro = pd.read_csv(os.path.join(DATA_DIR, 'external', 'macro_india_monthly.csv'), parse_dates=['date'])
macro = macro[macro['date'] >= '2018-01-01']
indicators = {'unemployment_rate': 'Unemployment % (World Bank)', 'gdp_growth': 'Real GDP growth % (World Bank)', 'cpi_inflation_yoy': 'CPI inflation YoY % (FRED)',
              'call_money_rate': 'Call-money rate % (FRED)', 'gsec_10y_yield': '10Y G-sec yield % (FRED)', 'inr_per_usd': 'INR per USD (FRED)'}
fig_macro = go.Figure()
for i, (col, label) in enumerate(indicators.items()):
    fig_macro.add_trace(go.Scatter(x=macro['date'], y=macro[col], name=label, visible=(i == 0), line=dict(width=2.5)))
buttons = [dict(label=label, method='update', args=[{'visible': [j == i for j in range(len(indicators))]}, {'title': label}]) for i, label in enumerate(indicators.values())]
fig_macro.update_layout(updatemenus=[dict(buttons=buttons, x=0, y=1.18, xanchor='left')], title=list(indicators.values())[0], height=430, template='plotly_white')
fig_macro.show()
""")

md("""
### 4d. Approval policy simulator (drag the slider = change the PD multiplier)
""")

code("""
mults = np.round(np.arange(0.4, 2.05, 0.1), 2)
frames, appr, bad = [], [], []
for m in mults:
    pit = (test['pd_ttc'] * m).clip(upper=0.99)
    a = pit < test['break_even_pd']
    appr.append(a.mean()); bad.append(test.loc[a, 'loan_default_status'].mean())
fig_pol = make_subplots(specs=[[{'secondary_y': True}]])
fig_pol.add_trace(go.Scatter(x=mults, y=appr, name='approval rate', mode='lines+markers', line=dict(color='#4c72b0', width=3)), secondary_y=False)
fig_pol.add_trace(go.Scatter(x=mults, y=bad, name='default rate of approved', mode='lines+markers', line=dict(color='#c44e52', width=3)), secondary_y=True)
for name, m in overlay['multipliers_next_12m'].items():
    fig_pol.add_vline(x=m, line_dash='dot', annotation_text=name)
fig_pol.update_layout(title='How the macro multiplier changes approvals and portfolio quality (same break-even rule)', xaxis_title='PD multiplier', height=450, template='plotly_white')
fig_pol.update_yaxes(title_text='approval rate', tickformat='.0%', secondary_y=False)
fig_pol.update_yaxes(title_text='default rate of approved', tickformat='.1%', secondary_y=True)
fig_pol.show()
""")

code("""
# Standalone interactive report (open in any browser, no Python needed)
html_path = os.path.join(REPORTS_DIR, 'interactive_dashboard.html')
with open(html_path, 'w', encoding='utf-8') as f:
    f.write('<html><head><meta charset="utf-8"><title>IntelliCredit - interactive report</title></head><body style="font-family:sans-serif;max-width:1200px;margin:auto">')
    f.write('<h1>IntelliCredit - interactive risk report</h1><p>Portfolio forecast, warning rules, macro drivers and approval-policy simulator. Built by notebooks/08_R2_combined_insights_dashboard.ipynb.</p>')
    for i, fig in enumerate([fig_ts, fig_rules, fig_macro, fig_pol]):
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn' if i == 0 else False))
    f.write('</body></html>')
print('wrote', os.path.relpath(html_path, ROOT), f'({os.path.getsize(html_path) / 1e6:.1f} MB)')
""")

md("""
## 5. Combined business insights & recommendations

### Insights that only appear when the three methods are combined
1. **Rules see risk the PD misses.** Inside the *same* lowest-risk band A, applicants who trigger 6+ warning rules default at **5.6%** vs **1.8%** for those who trigger none — 3× the risk at an identical PD band. In band C the gap is 7.3% → 10.3%. The rules are a second opinion that is cheap to compute and easy to explain to a borrower.
2. **The same behaviours become dangerous in a downturn.** Method 2 found that *volatile income*, *late utility payment* and *new gig worker* add the most extra default risk in stressed months; Method 1 says when those months are coming (the macro stress drivers lead defaults by ~2 months).
3. **A macro-aware policy self-adjusts.** Keeping one break-even rule (approve if PD < margin / (margin + LGD)) and scaling PD by the time-series multiplier:

| Scenario (2025) | PD multiplier | Approval rate | Default rate of approved |
|---|---|---|---|
| Neutral economy | 1.00 | 60.3% | 5.09% |
| Baseline (benign) | 0.57 | 75.1% | 6.10% |
| Adverse | 0.69 | 70.7% | 5.73% |
| Severe (COVID-style) | 1.53 | 47.8% | 3.98% |

   In a benign year the lender can safely serve ~15 percentage points more thin-file applicants; in a severe shock the same rule automatically cuts approvals to under half.
4. **The forecast turns into rupees.** Baseline 2025 expected credit loss ≈ ₹1.6 crore; adverse adds ≈ ₹36 lakh and severe ≈ ₹2.7 crore — sizing the provision overlay and the capital buffer.

### Recommendations for the lender
| # | Action | Evidence |
|---|---|---|
| 1 | **Score → band → rules**: approve automatically only in bands A–B *with* fewer than 3 triggered rules; send A–B with 3+ rules and all of band C to a smaller ticket / weekly-EMI product | Section 2 cross-tab; ARM ≥ 3-rule policy catches 34% of defaults in 14% of applicants |
| 2 | **Size the loan to tenure and income trend** — cap loan-to-income for tenure < 6 months or a falling income trend | top SHAP drivers in Review 1; `LARGE_LOAN_VS_INCOME` in 3 of the top-10 rules |
| 3 | **Apply the point-in-time multiplier every month** from the SARIMAX forecast (re-run `src/macro_api.py` → notebook 06) | Section 3 policy table |
| 4 | **Tighten selectively in a downturn** — on volatile income, late utility payment and new-gig-worker flags, not on low income or rural location | Method 2, Part D (macro-sensitive flags); protects financial inclusion |
| 5 | **Watch 30+ DPD weekly**; above ~4.5% expect defaults to rise within two months | Method 1, nowcast R² 0.76 |
| 6 | **Provision at baseline + adverse overlay (~₹36 lakh)**; stress-test capital on the severe path | Method 1, Part E |

### Limitations
* Borrowers, transactions and the loan book are **synthetic** (the rubric allows synthetic generation); the macro data are **real** (FRED and World Bank APIs).
* The panel's defaults are generated with a macro-linked hazard, so Method 1 recovering the macro effect is a validation of the pipeline, not proof about real Indian borrowers.
* Rules and PD multipliers should be re-estimated as real repayment data accumulate.
""")

nb["cells"] = cells
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
out = os.path.join(os.path.dirname(__file__), "..", "..", "notebooks", "08_R2_combined_insights_dashboard.ipynb")
nbf.write(nb, out)
print("wrote", os.path.abspath(out))
