"""Builds notebooks/06_R2_M1_time_series_forecasting.ipynb (Review 2 - Method 1)."""
import os

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


md("""
# Review 2 · Method 1 — Time-Series Forecasting of Portfolio Default Rate
**Project:** IntelliCredit — FinTech Micro-Lending Risk & Alternative Credit Scoring Engine
**Owner:** goldmauler
**Rubric:** Method 1 — Data preparation & implementation (4) · Evaluation & interpretation (3)

### Business question
*"How many of our micro-loans will turn 90+ days past due each month over the next year, and how much worse does that get if the economy turns?"*

A lender needs this number for three decisions:
1. **Provisioning** — expected-credit-loss (ECL) reserves are booked ahead of defaults.
2. **Capital & funding planning** — how much loss-absorbing capital to hold under an adverse scenario.
3. **Underwriting** — the Review 1 PD model is *through-the-cycle*; a forecast of the portfolio default rate turns it into a *point-in-time* PD (used by the dashboard's macro overlay).

### Data used
| Source | What | Frequency | How collected |
|---|---|---|---|
| `data/portfolio/portfolio_monthly.csv` | default rate, 30+/60+ DPD rates, active loans, originations | monthly, 2019-01 → 2024-12 | `src/portfolio_simulation.py` (30,000 Review 1 borrowers turned into a loan book) |
| `data/external/macro_india_monthly.csv` | CPI inflation, call-money rate, 10Y G-sec yield, INR/USD | monthly | **FRED API** (`src/macro_api.py`) |
| same file | unemployment, real GDP growth, lending rate | annual → monthly | **World Bank API** (`src/macro_api.py`) |

**Target:** `default_rate` = loans that reach 90+ DPD this month ÷ loans active this month (same 90-day definition as the Review 1 target).
""")

code("""
import os, sys, json, itertools, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss, ccf, grangercausalitytests
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 110

ROOT = os.path.dirname(os.getcwd())
sys.path.insert(0, ROOT)
from src.config import DATA_DIR, MODELS_DIR
os.makedirs(MODELS_DIR, exist_ok=True)

COVID_SPANS = [('2020-03-01', '2020-08-01', 'COVID lockdown'), ('2021-04-01', '2021-07-01', 'Second wave')]

def shade_covid(ax):
    for start, end, label in COVID_SPANS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color='grey', alpha=0.15, lw=0)
""")

md("""
## Part A — Data preparation
### A1. Load the portfolio series and the API macro data, and record provenance
""")

code("""
pm = pd.read_csv(os.path.join(DATA_DIR, 'portfolio', 'portfolio_monthly.csv'), parse_dates=['date']).set_index('date').asfreq('MS')
macro = pd.read_csv(os.path.join(DATA_DIR, 'external', 'macro_india_monthly.csv'), parse_dates=['date']).set_index('date').asfreq('MS')
with open(os.path.join(DATA_DIR, 'external', 'macro_sources.json')) as f:
    sources = json.load(f)

prov = pd.DataFrame(sources['columns']).T
display(prov[[c for c in ['api', 'series_id', 'indicator_id', 'origin', 'first_obs', 'last_obs', 'years'] if c in prov.columns]].fillna(''))
print(f"Macro table built {sources['generated_at']} | portfolio months: {len(pm)} ({pm.index.min():%Y-%m} to {pm.index.max():%Y-%m})")
display(pm[['active_loans', 'new_originations', 'new_defaults', 'default_rate', 'dpd30_plus_rate']].describe().round(4))
""")

md("""
### A2. Data-quality checks
* the monthly index must be continuous (no missing months),
* no missing values inside the modelling window,
* macro values that were **carried forward** past the last published observation must not be used as if they were real history.
""")

code("""
expected = pd.date_range(pm.index.min(), pm.index.max(), freq='MS')
print('Missing months in portfolio series :', len(expected.difference(pm.index)))
print('Missing values in portfolio series :', int(pm.isnull().sum().sum()))

window = macro.loc['2018-07-01':'2024-12-01']
flags = [c for c in macro.columns if c.endswith('_is_carried_forward')]
carried = window[flags].sum()
print('Carried-forward macro months inside 2018-07..2024-12:')
print(carried[carried > 0] if carried.sum() else '  none - every modelling month uses a published value')
print('Missing macro values inside the window:', int(window.drop(columns=flags).isnull().sum().sum()))
""")

md("""
### A3. Target transformation
Default rates are small proportions whose variance grows with the level — the COVID months are 4–6× the normal level. Modelling **log(default rate in %)** makes the variance roughly constant and turns macro effects into *percentage* changes (a coefficient β means "×e^β"). Forecasts are transformed back with `exp()`.
""")

code("""
y = (pm['default_rate'] * 100).rename('default_rate_pct')        # in %
log_y = np.log(y).rename('log_default_rate')

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].hist(y, bins=25, color='#4c72b0'); axes[0].set_title(f'Default rate (%)  skew = {stats.skew(y):.2f}')
axes[1].hist(log_y, bins=25, color='#55a868'); axes[1].set_title(f'log default rate  skew = {stats.skew(log_y):.2f}')
plt.tight_layout(); plt.show()
""")

md("""
### A4. Exogenous drivers — lag structure
A loan only becomes a 90+ DPD default **two months after** its first missed instalment (30 → 60 → 90 DPD), so macro conditions should show up in the default rate with a lag. The cross-correlation function (CCF) between each driver and the default rate at lags 0–6 tells us which lag to use.
""")

code("""
drivers = ['unemployment_rate', 'gdp_growth', 'cpi_inflation_yoy', 'call_money_rate', 'inr_depreciation_yoy', 'covid_lockdown', 'covid_second_wave']
lags = range(0, 7)
ccf_tab = pd.DataFrame({d: [log_y.corr(macro[d].shift(L).reindex(log_y.index)) for L in lags] for d in drivers}, index=[f'lag {L}' for L in lags])

plt.figure(figsize=(11, 4.5))
sns.heatmap(ccf_tab.T, annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1)
plt.title('Correlation of log default rate with each macro driver shifted by k months')
plt.tight_layout(); plt.show()
best_lag = ccf_tab.abs().idxmax()
display(pd.DataFrame({'strongest lag': best_lag, 'corr': [ccf_tab.loc[best_lag[d], d] for d in drivers]}))
""")

code("""
LAG = 2   # recognition lag of the 30 -> 60 -> 90 DPD roll, confirmed by the CCF table
exog_all = pd.DataFrame({f'{d}_l{LAG}': macro[d].shift(LAG) for d in drivers}).reindex(pm.index)
print(exog_all.isnull().sum().sum(), 'missing values after lagging (macro history starts in 2017, so none expected)')
display(exog_all.head())
""")

md("""
### A5. Train / test split
Time series are split **by time, never at random**: the model is fitted on 2019-01 → 2023-12 (60 months) and evaluated on the 12 unseen months of 2024. A rolling-origin backtest (Part D) then repeats the exercise from many forecast dates, including the second COVID wave.
""")

code("""
TRAIN_END, TEST_START = '2023-12-01', '2024-01-01'
y_tr, y_te = y[:TRAIN_END], y[TEST_START:]
ly_tr, ly_te = log_y[:TRAIN_END], log_y[TEST_START:]
X_tr, X_te = exog_all[:TRAIN_END], exog_all[TEST_START:]
H = len(y_te)
print(f'train: {len(y_tr)} months ({y_tr.index.min():%Y-%m}..{y_tr.index.max():%Y-%m}) | test: {H} months ({y_te.index.min():%Y-%m}..{y_te.index.max():%Y-%m})')
""")

md("""
## Part B — Exploratory time-series analysis
### B1. The portfolio over time
""")

code("""
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
axes[0].plot(y, color='#c44e52', lw=2); axes[0].set_ylabel('default rate %'); axes[0].set_title('Monthly 90+ DPD default rate')
axes[1].plot(pm['dpd30_plus_rate'] * 100, color='#dd8452', lw=2); axes[1].set_ylabel('30+ DPD %'); axes[1].set_title('Early delinquency (30+ DPD) - the leading indicator')
axes[2].bar(pm.index, pm['new_originations'], width=25, color='#8c8c8c', label='new loans'); axes[2].plot(pm['active_loans'], color='#4c72b0', lw=2, label='active loans')
axes[2].legend(); axes[2].set_title('Book size')
for ax in axes: shade_covid(ax)
plt.tight_layout(); plt.show()
""")

code("""
fig, axes = plt.subplots(2, 3, figsize=(16, 7), sharex=True)
for ax, col, title in zip(axes.ravel(),
        ['unemployment_rate', 'gdp_growth', 'cpi_inflation_yoy', 'call_money_rate', 'gsec_10y_yield', 'inr_per_usd'],
        ['Unemployment % (World Bank)', 'Real GDP growth % (World Bank)', 'CPI inflation YoY % (FRED)', 'Call-money rate % (FRED)', '10Y G-sec yield % (FRED)', 'INR per USD (FRED)']):
    ax.plot(macro.loc['2018-07-01':'2024-12-01', col], lw=2)
    ax.set_title(title); shade_covid(ax)
plt.suptitle('Macro drivers collected from the APIs', y=1.01, fontsize=13)
plt.tight_layout(); plt.show()
""")

md("""
### B2. STL decomposition — trend, seasonality, remainder
STL (Seasonal-Trend decomposition using LOESS) splits the log default rate into a slow **trend**, a repeating 12-month **seasonal** pattern and a **remainder**. *Strength* measures, 0 → 1, say how much of the variation each component explains (Hyndman & Athanasopoulos, FPP3 §4.3).
""")

code("""
stl = STL(log_y, period=12, robust=True).fit()
fig = stl.plot(); fig.set_size_inches(14, 8); plt.tight_layout(); plt.show()

var = np.var
trend_strength = max(0, 1 - var(stl.resid) / var(stl.trend + stl.resid))
seasonal_strength = max(0, 1 - var(stl.resid) / var(stl.seasonal + stl.resid))
print(f'Trend strength    F_T = {trend_strength:.2f}')
print(f'Seasonal strength F_S = {seasonal_strength:.2f}')
""")

code("""
season_df = pd.DataFrame({'month': y.index.month, 'rate': y.values, 'year': y.index.year})
calm = season_df[~season_df['year'].isin([2020, 2021])]
fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
sns.boxplot(data=calm, x='month', y='rate', ax=axes[0], color='#8da0cb'); axes[0].set_title('Default rate by calendar month (excl. 2020-21)')
sns.lineplot(data=season_df, x='month', y='rate', hue='year', palette='viridis', ax=axes[1], marker='o'); axes[1].set_title('Seasonal plot by year')
plt.tight_layout(); plt.show()
print(calm.groupby('month')['rate'].mean().round(3).to_frame('mean default rate % (calm years)').T)
""")

md("""
### B3. Stationarity — ADF and KPSS
ARIMA-type models need a stationary series. The two tests have **opposite null hypotheses**, so they are read together:
* **ADF** H₀: unit root (non-stationary) → want p < 0.05
* **KPSS** H₀: stationary → want p > 0.05
""")

code("""
def stationarity(series, name):
    s = series.dropna()
    adf_p = adfuller(s, autolag='AIC')[1]
    kpss_p = kpss(s, regression='c', nlags='auto')[1]
    verdict = 'stationary' if (adf_p < 0.05 and kpss_p > 0.05) else 'non-stationary' if (adf_p > 0.05 and kpss_p < 0.05) else 'inconclusive'
    return {'series': name, 'ADF p-value': round(adf_p, 4), 'KPSS p-value': round(kpss_p, 4), 'verdict': verdict}

display(pd.DataFrame([
    stationarity(log_y, 'log default rate'),
    stationarity(log_y.diff(), 'Δ log default rate'),
    stationarity(log_y.diff(12), 'Δ12 log default rate'),
]))
""")

md("""
### B4. Autocorrelation (ACF / PACF)
The ACF/PACF of the series suggest candidate AR (p) and MA (q) orders; spikes at lag 12 would point to seasonal terms.
""")

code("""
fig, axes = plt.subplots(2, 2, figsize=(14, 7))
plot_acf(log_y, lags=24, ax=axes[0, 0]); axes[0, 0].set_title('ACF - log default rate')
plot_pacf(log_y, lags=24, ax=axes[0, 1], method='ywm'); axes[0, 1].set_title('PACF - log default rate')
plot_acf(log_y.diff().dropna(), lags=24, ax=axes[1, 0]); axes[1, 0].set_title('ACF - first difference')
plot_pacf(log_y.diff().dropna(), lags=24, ax=axes[1, 1], method='ywm'); axes[1, 1].set_title('PACF - first difference')
plt.tight_layout(); plt.show()
""")

md("""
### B5. Granger causality — do the macro series carry information about *future* defaults?
A driver *Granger-causes* the default rate if its past values improve the prediction of the default rate beyond the default rate's own past. Tested with the F-test at lags 1–4 (H₀: no Granger causality).
""")

code("""
gc_rows = []
for d in ['unemployment_rate', 'gdp_growth', 'cpi_inflation_yoy', 'call_money_rate', 'inr_depreciation_yoy']:
    data = pd.concat([log_y, macro[d].reindex(log_y.index)], axis=1).dropna()
    try:
        res = grangercausalitytests(data, maxlag=4, verbose=False)
    except TypeError:
        res = grangercausalitytests(data, maxlag=4)
    p = {f'lag {k}': res[k][0]['ssr_ftest'][1] for k in res}
    gc_rows.append({'driver': d, **p, 'min p': min(p.values())})
gc = pd.DataFrame(gc_rows).set_index('driver').round(4)
display(gc.style.background_gradient(cmap='RdYlGn_r', subset=['min p'], vmin=0, vmax=0.1).format('{:.4f}'))
""")

md("""
## Part C — Model implementation
Seven forecasting approaches, from naive benchmarks to a macro-driven SARIMAX. All are trained on 2019–2023 and forecast the 12 months of 2024 in one go (a genuine 12-step-ahead forecast). Macro models use the *realised* 2024 macro path — a **conditional** forecast, which is how stress tests are run ("if the economy does X, defaults will be Y").

| # | Model | Idea |
|---|---|---|
| 1 | Naive | next month = last month |
| 2 | Seasonal naive | next month = same month last year |
| 3 | 12-month moving average | long-run level |
| 4 | Holt-Winters (ETS) | exponentially-weighted level + damped trend + seasonality |
| 5 | SARIMA | ARIMA(p,d,q)(P,D,Q)₁₂ on log rate, order chosen by AIC |
| 6 | **SARIMAX + macro** | SARIMA errors + lagged macro drivers & COVID event dummies |
| 7 | Gradient boosting | tree ensemble on own lags + macro, forecast recursively |
""")

code("""
forecasts = {}

# 1-3 benchmarks
forecasts['Naive'] = pd.Series(y_tr.iloc[-1], index=y_te.index)
forecasts['Seasonal naive'] = pd.Series(y_tr.iloc[-12:].values, index=y_te.index)
forecasts['12-month moving average'] = pd.Series(y_tr.iloc[-12:].mean(), index=y_te.index)

# 4 Holt-Winters on the log scale
ets = ExponentialSmoothing(ly_tr, trend='add', damped_trend=True, seasonal='add', seasonal_periods=12).fit(optimized=True)
forecasts['Holt-Winters (ETS)'] = np.exp(ets.forecast(H))
print(f'ETS: alpha={ets.params["smoothing_level"]:.3f}, beta={ets.params["smoothing_trend"]:.3f}, gamma={ets.params["smoothing_seasonal"]:.3f}, phi={ets.params["damping_trend"]:.3f}')
""")

md("""
### C1. SARIMA — order selection by AIC grid search
AIC = 2k − 2 ln L rewards fit but penalises every extra parameter, which matters with only 60 training months.
""")

code("""
def fit_sarimax(endog, order, seasonal_order, exog=None):
    # stationarity / invertibility enforced so short training windows cannot produce explosive forecasts
    return SARIMAX(endog, exog=exog, order=order, seasonal_order=seasonal_order, trend='c' if order[1] == 0 else 'n',
                   enforce_stationarity=True, enforce_invertibility=True).fit(disp=False, maxiter=300)

grid = []
for p, d, q, P, Q in itertools.product([0, 1, 2], [0, 1], [0, 1, 2], [0, 1], [0, 1]):
    try:
        res = fit_sarimax(ly_tr, (p, d, q), (P, 0, Q, 12))
        grid.append({'order': (p, d, q), 'seasonal_order': (P, 0, Q, 12), 'AIC': res.aic, 'BIC': res.bic})
    except Exception:
        pass
grid = pd.DataFrame(grid).sort_values('AIC').reset_index(drop=True)
display(grid.head(8))
best_order, best_sorder = grid.loc[0, 'order'], grid.loc[0, 'seasonal_order']
sarima = fit_sarimax(ly_tr, best_order, best_sorder)
forecasts['SARIMA'] = np.exp(sarima.get_forecast(H).predicted_mean)
print(f'Selected SARIMA{best_order}x{best_sorder}  AIC={sarima.aic:.1f}')
""")

md("""
### C2. SARIMAX — building a macro stress factor and choosing the drivers
Unemployment, GDP growth, inflation, the call-money rate and rupee depreciation move together (correlations up to |0.9|), so putting all five into a 60-month regression makes the individual coefficients unstable — the classic multicollinearity problem from Review 1's VIF analysis. The fix is the same idea as Review 1's PCA: compress them into **one macro stress factor** = the first principal component of the standardised, 2-month-lagged drivers, fitted on the **training months only**.
""")

code("""
macro_drivers = ['unemployment_rate', 'gdp_growth', 'cpi_inflation_yoy', 'call_money_rate', 'inr_depreciation_yoy']
lagged = pd.DataFrame({d: macro[d].shift(LAG) for d in macro_drivers})
mu, sd = lagged.loc[ly_tr.index].mean(), lagged.loc[ly_tr.index].std()

display(lagged.loc[ly_tr.index].corr().round(2))

Z_tr = ((lagged.loc[ly_tr.index] - mu) / sd).to_numpy()
eigval, eigvec = np.linalg.eigh(np.cov(Z_tr.T))
order_ = np.argsort(eigval)[::-1]
eigval, eigvec = eigval[order_], eigvec[:, order_]
loadings = pd.Series(eigvec[:, 0], index=macro_drivers)
if loadings['unemployment_rate'] < 0:          # orient so that higher factor = more stress
    loadings = -loadings
explained = eigval[0] / eigval.sum()

def macro_factor(frame):
    return (((frame[macro_drivers] - mu) / sd) @ loadings).rename('macro_factor_l2')

exog_all['macro_factor_l2'] = macro_factor(lagged).reindex(exog_all.index)
X_tr, X_te = exog_all[:TRAIN_END], exog_all[TEST_START:]
print(f'PC1 explains {explained:.0%} of the variance of the five drivers')
display(loadings.rename('PC1 loading').to_frame().round(3))

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(exog_all['macro_factor_l2'], color='#8172b2', lw=2, label='macro stress factor (lag 2)')
ax2 = ax.twinx(); ax2.plot(log_y, color='#c44e52', lw=1.5, alpha=0.8, label='log default rate')
shade_covid(ax); ax.set_title('Macro stress factor vs log default rate'); ax.legend(loc='upper left'); ax2.legend(loc='upper right')
plt.tight_layout(); plt.show()
""")

md("""
Four candidate driver sets are compared. For each set the ARMA order is re-selected (small grid), and the **set is chosen by BIC**: with only 60 months, BIC's heavier penalty (k·ln n instead of 2k) protects against over-fitting better than AIC. The 2024 test year is *not* used for this choice.
""")

code("""
exog_sets = {
    'Events only': ['covid_lockdown_l2', 'covid_second_wave_l2'],
    'Events + macro factor': ['macro_factor_l2', 'covid_lockdown_l2', 'covid_second_wave_l2'],
    'Events + unemployment + GDP': ['unemployment_rate_l2', 'gdp_growth_l2', 'covid_lockdown_l2', 'covid_second_wave_l2'],
    'Events + all five drivers': [f'{d}_l{LAG}' for d in macro_drivers] + ['covid_lockdown_l2', 'covid_second_wave_l2'],
}
small_grid = list(itertools.product([0, 1, 2], [0], [0, 1, 2], [0, 1], [0, 1]))
exog_cmp, fits = [], {}
for name, cols in exog_sets.items():
    best = None
    for p, d, q, P, Q in small_grid:
        try:
            r = fit_sarimax(ly_tr, (p, d, q), (P, 0, Q, 12), exog=X_tr[cols])
        except Exception:
            continue
        if best is None or r.bic < best[0].bic:
            best = (r, (p, d, q), (P, 0, Q, 12))
    r, o_, so_ = best
    fits[name] = best
    exog_cmp.append({'driver set': name, 'n drivers': len(cols), 'order': o_, 'seasonal_order': so_,
                     'AIC': r.aic, 'BIC': r.bic, 'max |driver t|': np.abs(r.tvalues[cols]).max()})
exog_cmp = pd.DataFrame(exog_cmp).set_index('driver set')
display(exog_cmp.round(2))
best_set = exog_cmp['BIC'].idxmin()
EXOG = exog_sets[best_set]
sarimax, sx_order, sx_sorder = fits[best_set]
print(f'Chosen by BIC: {best_set} -> {EXOG}  with SARIMA{sx_order}x{sx_sorder}')
""")

code("""
fc_obj = sarimax.get_forecast(H, exog=X_te[EXOG])
forecasts['SARIMAX + macro'] = np.exp(fc_obj.predicted_mean)
print(sarimax.summary().tables[1])
""")

md("""
**Reading the coefficients.** Because the target is log(default rate), a coefficient β on a driver means a one-unit rise in that driver (two months earlier) multiplies the default rate by e^β. The table below converts them into business language.
""")

code("""
params = sarimax.params
pvals = sarimax.pvalues
coef = pd.DataFrame({'beta': params[EXOG], 'p-value': pvals[EXOG]})
coef['effect on default rate'] = [f"x{np.exp(b):.2f} while active" if 'covid' in c else
                                  f"{(np.exp(b) - 1) * 100:+.1f}% per +1 SD of macro stress" if c == 'macro_factor_l2' else
                                  f"{(np.exp(b) - 1) * 100:+.1f}% per +1 unit" for c, b in coef['beta'].items()]
display(coef.round(4))
""")

md("""
### C3. Gradient boosting with lag features (machine-learning benchmark)
Features: the default rate's own lags (1, 2, 3, 12 months), calendar month (sin/cos) and the lagged macro drivers. 12-step forecasts are produced **recursively** — each prediction is fed back as the next month's lag.
""")

code("""
def make_features(log_series, exog, idx):
    f = pd.DataFrame(index=idx)
    for L in (1, 2, 3, 12):
        f[f'lag{L}'] = log_series.shift(L).reindex(idx)
    f['month_sin'] = np.sin(2 * np.pi * idx.month / 12)
    f['month_cos'] = np.cos(2 * np.pi * idx.month / 12)
    return f.join(exog.loc[idx])

feat_tr = make_features(ly_tr, exog_all, ly_tr.index).dropna()
gbr = GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=0.03, subsample=0.8, random_state=42)
gbr.fit(feat_tr, ly_tr.loc[feat_tr.index])

history = ly_tr.copy()
preds = []
for t in y_te.index:
    row = make_features(pd.concat([history, pd.Series([np.nan], index=[t])]), exog_all, pd.DatetimeIndex([t]))
    p = gbr.predict(row)[0]
    preds.append(p)
    history.loc[t] = p
forecasts['Gradient boosting'] = pd.Series(np.exp(preds), index=y_te.index)
print(f'GBR trained on {len(feat_tr)} months ({feat_tr.index.min():%Y-%m}..{feat_tr.index.max():%Y-%m})')
""")

md("""
## Part D — Evaluation
### D1. Hold-out accuracy on 2024 (12 months ahead)
* **MAE / RMSE** in percentage points (pp) of default rate; RMSE punishes big misses more.
* **MAPE** — average % error.
* **MASE** — MAE divided by the in-sample MAE of the seasonal-naive method: < 1 means the model beats the seasonal-naive benchmark.
* **Bias** — mean (forecast − actual): positive = over-forecasting (conservative for provisioning).
""")

code("""
scale = np.mean(np.abs(y_tr.values[12:] - y_tr.values[:-12]))   # in-sample seasonal-naive MAE

def score(actual, pred):
    e = pred.values - actual.values
    return {'MAE (pp)': np.mean(np.abs(e)), 'RMSE (pp)': np.sqrt(np.mean(e ** 2)),
            'MAPE %': np.mean(np.abs(e / actual.values)) * 100, 'MASE': np.mean(np.abs(e)) / scale, 'Bias (pp)': e.mean()}

holdout = pd.DataFrame({name: score(y_te, fc) for name, fc in forecasts.items()}).T.sort_values('MAE (pp)')
display(holdout.style.highlight_min(subset=['MAE (pp)', 'RMSE (pp)', 'MAPE %', 'MASE'], color='#c7e9c0').format('{:.4f}'))
""")

code("""
fig, ax = plt.subplots(figsize=(14, 5.5))
ax.plot(y['2022-01-01':], color='black', lw=2.5, label='actual')
for name in holdout.index[:4]:
    ax.plot(forecasts[name], lw=1.8, marker='o', ms=3, label=f'{name} (MAE {holdout.loc[name, "MAE (pp)"]:.3f})')
ax.axvline(pd.Timestamp(TEST_START), color='grey', ls='--')
ax.set_title('2024 hold-out: 12-month-ahead forecasts vs actual (top-4 models)'); ax.set_ylabel('default rate %')
ax.legend(loc='upper left'); plt.tight_layout(); plt.show()
""")

md("""
### D2. Rolling-origin backtest (expanding window)
One hold-out year is a single, calm draw. The backtest re-fits each model at every month-end from **2021-03** to **2024-06** and forecasts 1, 3 and 6 months ahead — so the second COVID wave (mid-2021) and the 2022–23 rate-hike cycle are both inside the evaluation. Drivers that never vary inside a training window (e.g. the second-wave dummy before April 2021) are dropped for that fit. (Simplification: the macro-factor loadings are the ones estimated on 2019–2023.)
""")

code("""
HORIZONS = (1, 3, 6)
origins = pd.date_range('2021-03-01', '2024-06-01', freq='MS')
bt_rows = []
for origin in origins:
    ly_o = log_y[:origin]
    y_o = y[:origin]
    future = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=max(HORIZONS), freq='MS')
    future = future[future <= y.index.max()]
    if len(future) == 0:
        continue
    cand = {}
    cand['Seasonal naive'] = pd.Series(y_o.iloc[-12:].values[:len(future)], index=future)
    try:
        e = ExponentialSmoothing(ly_o, trend='add', damped_trend=True, seasonal='add', seasonal_periods=12).fit()
        cand['Holt-Winters (ETS)'] = np.exp(e.forecast(len(future)))
    except Exception:
        pass
    try:
        cand['SARIMA'] = np.exp(fit_sarimax(ly_o, best_order, best_sorder).get_forecast(len(future)).predicted_mean)
    except Exception:
        pass
    cols = [c for c in EXOG if exog_all.loc[:origin, c].nunique() > 1]
    try:
        r = fit_sarimax(ly_o, sx_order, sx_sorder, exog=exog_all.loc[:origin, cols] if cols else None)
        cand['SARIMAX + macro'] = np.exp(r.get_forecast(len(future), exog=exog_all.loc[future, cols] if cols else None).predicted_mean)
    except Exception:
        pass
    for name, fc in cand.items():
        fc.index = future
        for h in HORIZONS:
            if h <= len(future):
                t = future[h - 1]
                bt_rows.append({'origin': origin, 'model': name, 'h': h, 'target': t, 'forecast': fc.loc[t], 'actual': y.loc[t]})

bt = pd.DataFrame(bt_rows)
bt = bt[np.isfinite(bt['forecast'])]
bt['abs_err'] = (bt['forecast'] - bt['actual']).abs()
bt_summary = bt.pivot_table(index='model', columns='h', values='abs_err', aggfunc='mean')
bt_summary.columns = [f'MAE h={h}' for h in bt_summary.columns]
bt_summary['mean MAE'] = bt_summary.mean(axis=1)
display(bt_summary.sort_values('mean MAE').style.highlight_min(color='#c7e9c0').format('{:.4f}'))
print(f'{bt.origin.nunique()} forecast origins, {len(bt):,} forecasts scored')
""")

code("""
stress_months = bt['target'].dt.strftime('%Y-%m').isin(['2021-05', '2021-06', '2021-07', '2021-08'])
regime = bt.assign(regime=np.where(stress_months, 'second wave (2021-05..08)', 'normal months')).pivot_table(
    index='model', columns='regime', values='abs_err', aggfunc='mean')
display(regime.round(4))

fig, ax = plt.subplots(figsize=(14, 5))
for name in ['Seasonal naive', 'SARIMA', 'SARIMAX + macro']:
    s = bt[(bt.model == name) & (bt.h == 3)].set_index('target')['forecast']
    ax.plot(s, lw=1.6, label=f'{name} (3-month-ahead)')
ax.plot(y['2021-03-01':], color='black', lw=2.5, label='actual')
shade_covid(ax); ax.legend(); ax.set_title('Backtest: what each model predicted 3 months earlier'); ax.set_ylabel('default rate %')
plt.tight_layout(); plt.show()
""")

md("""
### D3. Residual diagnostics of the chosen model
A well-specified model leaves residuals that look like white noise: no autocorrelation (Ljung-Box p > 0.05), mean ≈ 0 and roughly normal.
""")

code("""
resid = sarimax.resid[13:]   # skip the burn-in of the state-space filter
lb = acorr_ljungbox(resid, lags=[6, 12], return_df=True)
display(lb.round(4))
print(f'Residual mean = {resid.mean():.4f}, Jarque-Bera p = {stats.jarque_bera(resid).pvalue:.4f}, ADF p = {adfuller(resid)[1]:.4f}')
print('The log default rate itself trends down with the economy (ADF/KPSS in B3); once the macro drivers explain that trend the residuals are stationary, so no differencing is needed.')

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].plot(resid); axes[0].axhline(0, c='grey'); axes[0].set_title('Residuals over time')
plot_acf(resid, lags=18, ax=axes[1]); axes[1].set_title('Residual ACF')
stats.probplot(resid, dist='norm', plot=axes[2]); axes[2].set_title('Normal Q-Q')
plt.tight_layout(); plt.show()
""")

md("""
### D4. Are the prediction intervals honest?
An 80% interval should contain roughly 80% of the actuals, a 95% interval about 95%.
""")

code("""
coverage = {}
for alpha, label in [(0.20, '80%'), (0.05, '95%')]:
    ci = np.exp(fc_obj.conf_int(alpha=alpha))
    inside = ((y_te.values >= ci.iloc[:, 0].values) & (y_te.values <= ci.iloc[:, 1].values)).mean()
    coverage[label] = inside
print({k: f'{v:.0%}' for k, v in coverage.items()})

ci80, ci95 = np.exp(fc_obj.conf_int(alpha=0.20)), np.exp(fc_obj.conf_int(alpha=0.05))
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(y['2022-07-01':], color='black', lw=2.5, label='actual')
ax.plot(forecasts['SARIMAX + macro'], color='#c44e52', lw=2, label='SARIMAX + macro')
ax.fill_between(y_te.index, ci95.iloc[:, 0], ci95.iloc[:, 1], color='#c44e52', alpha=0.12, label='95% PI')
ax.fill_between(y_te.index, ci80.iloc[:, 0], ci80.iloc[:, 1], color='#c44e52', alpha=0.25, label='80% PI')
ax.legend(loc='upper left'); ax.set_title('SARIMAX hold-out forecast with prediction intervals'); ax.set_ylabel('default rate %')
plt.tight_layout(); plt.show()
""")

md("""
### D5. Early-warning nowcast from 30+ DPD
Every default passes through 30 DPD two months earlier, so this month's 30+ DPD rate is an almost free **2-month-ahead nowcast** that needs no macro forecast at all.
""")

code("""
lead = pd.DataFrame({'log_y': log_y, 'log_dpd30_l2': np.log(pm['dpd30_plus_rate'] * 100).shift(2)}).dropna()
lead_tr, lead_te = lead[:TRAIN_END], lead[TEST_START:]
b1, b0 = np.polyfit(lead_tr['log_dpd30_l2'], lead_tr['log_y'], 1)
nowcast = pd.Series(np.exp(b0 + b1 * lead_te['log_dpd30_l2']), index=lead_te.index)
print(f'log(default rate) = {b0:.3f} + {b1:.3f} x log(30+ DPD rate two months earlier)   R2(train) = {np.corrcoef(lead_tr.log_dpd30_l2, lead_tr.log_y)[0,1]**2:.3f}')
print(f'2024 nowcast MAE = {np.mean(np.abs(nowcast - y_te)):.4f} pp  (vs best 12-step model {holdout["MAE (pp)"].min():.4f} pp)')
""")

md("""
## Part E — 2025 forecast & macro stress scenarios
The chosen SARIMAX is re-fitted on all 72 months and driven by three macro paths for 2025:

| Scenario | Macro path |
|---|---|
| **Baseline** | latest values published by the APIs (rates & FX observed through 2026; CPI through 2025-03; World Bank 2025 estimates) |
| **Adverse** | from 2025-03, unemployment +1.5 pp and GDP growth −3 pp, phased in over 3 months (a sharp slowdown) |
| **Severe** | COVID-style shock: unemployment +3 pp, GDP growth −8 pp and two months of lockdown (2025-04/05) |
""")

code("""
final = fit_sarimax(log_y, sx_order, sx_sorder, exog=exog_all[EXOG])
fut = pd.date_range('2025-01-01', '2025-12-01', freq='MS')
base_macro = macro.copy()

def scenario_macro(kind):
    m = base_macro.copy()
    ramp = pd.Series(0.0, index=m.index)
    ramp.loc['2025-03-01':] = np.minimum(1, (np.arange(len(ramp.loc['2025-03-01':])) + 1) / 3)
    if kind == 'Adverse':
        m['unemployment_rate'] += 1.5 * ramp
        m['gdp_growth'] -= 3.0 * ramp
    if kind == 'Severe':
        m['unemployment_rate'] += 3.0 * ramp
        m['gdp_growth'] -= 8.0 * ramp
        m.loc[['2025-04-01', '2025-05-01'], 'covid_lockdown'] = 1
    if kind == 'Neutral':   # every driver at its 2019-2024 average, no lockdown
        for d in macro_drivers:
            m[d] = base_macro.loc['2019-01-01':'2024-12-01', d].mean()
        m[['covid_lockdown', 'covid_second_wave']] = 0
    x = pd.DataFrame({f'{d}_l{LAG}': m[d].shift(LAG) for d in drivers})
    x['macro_factor_l2'] = macro_factor(pd.DataFrame({d: m[d].shift(LAG) for d in macro_drivers}))
    return x.loc[fut, EXOG]

scen = {}
neutral_rate = np.exp(final.get_forecast(len(fut), exog=scenario_macro('Neutral')).predicted_mean).mean()
print(f'Model-implied through-the-cycle (neutral economy) default rate: {neutral_rate:.3f}% per month')
for kind in ['Baseline', 'Adverse', 'Severe']:
    f = final.get_forecast(len(fut), exog=scenario_macro(kind))
    ci80, ci95 = np.exp(f.conf_int(alpha=0.2)), np.exp(f.conf_int(alpha=0.05))
    scen[kind] = pd.DataFrame({'forecast': np.exp(f.predicted_mean), 'lower80': ci80.iloc[:, 0], 'upper80': ci80.iloc[:, 1],
                               'lower95': ci95.iloc[:, 0], 'upper95': ci95.iloc[:, 1]}, index=fut)

colors = {'Baseline': '#4c72b0', 'Adverse': '#dd8452', 'Severe': '#c44e52'}
fig, ax = plt.subplots(figsize=(15, 5.5))
ax.plot(y, color='black', lw=2, label='history')
for kind, df_ in scen.items():
    ax.plot(df_['forecast'], color=colors[kind], lw=2.2, label=kind)
    ax.fill_between(df_.index, df_['lower80'], df_['upper80'], color=colors[kind], alpha=0.15)
shade_covid(ax); ax.axvline(pd.Timestamp('2025-01-01'), color='grey', ls='--')
ax.set_title('Portfolio default-rate forecast for 2025 under three macro scenarios (80% bands)'); ax.set_ylabel('default rate %'); ax.legend(loc='upper left')
plt.tight_layout(); plt.show()
""")

md("""
### E1. From default rates to rupees — expected defaults and credit loss
* active loans grow at the average monthly growth of 2024,
* exposure at default (EAD) = average outstanding principal of loans in the month they defaulted (from the panel),
* loss given default (LGD) = 65% (unsecured micro-loans; same assumption as Review 1's cost analysis).
""")

code("""
perf = pd.read_csv(os.path.join(DATA_DIR, 'portfolio', 'loan_performance.csv'), usecols=['dpd', 'outstanding_principal'])
EAD = perf.loc[perf['dpd'] >= 90, 'outstanding_principal'].mean()
LGD = 0.65
growth = (pm['active_loans'].iloc[-1] / pm['active_loans'].iloc[-13]) ** (1 / 12) - 1
active_2025 = pm['active_loans'].iloc[-1] * (1 + growth) ** np.arange(1, 13)

impact = []
for kind, df_ in scen.items():
    defaults = (df_['forecast'] / 100 * active_2025).sum()
    impact.append({'scenario': kind, 'avg monthly default rate %': df_['forecast'].mean(), 'peak month rate %': df_['forecast'].max(),
                   'expected defaults 2025': defaults, 'expected credit loss (INR lakh)': defaults * EAD * LGD / 1e5})
impact = pd.DataFrame(impact).set_index('scenario')
impact['extra loss vs baseline (INR lakh)'] = impact['expected credit loss (INR lakh)'] - impact.loc['Baseline', 'expected credit loss (INR lakh)']
print(f'Avg EAD at default INR {EAD:,.0f} | LGD {LGD:.0%} | active-book growth {growth:.2%}/month')
display(impact.round(2))
""")

md("""
## Part F — Save artefacts for the dashboard & combined insights
""")

code("""
scen_out = pd.concat([df_.assign(scenario=k) for k, df_ in scen.items()]).rename_axis('date').reset_index()
scen_out.to_csv(os.path.join(MODELS_DIR, 'ts_forecast_scenarios.csv'), index=False)
holdout.rename_axis('model').to_csv(os.path.join(MODELS_DIR, 'ts_model_comparison.csv'))
bt_summary.rename_axis('model').to_csv(os.path.join(MODELS_DIR, 'ts_backtest_summary.csv'))
impact.to_csv(os.path.join(MODELS_DIR, 'ts_scenario_impact.csv'))

overlay = {
    'description': 'Point-in-time PD multiplier = forecast portfolio default rate / model-implied neutral-economy rate',
    'neutral_default_rate_pct': round(float(neutral_rate), 4),
    'historical_mean_default_rate_pct': round(float(y.mean()), 4),
    'model': f'SARIMAX{sx_order}x{sx_sorder} + {EXOG}',
    'multipliers_next_12m': {k: round(float(df_['forecast'].mean() / neutral_rate), 3) for k, df_ in scen.items()},
    'monthly_forecast_pct': {k: [round(float(v), 4) for v in df_['forecast']] for k, df_ in scen.items()},
}
with open(os.path.join(MODELS_DIR, 'macro_overlay.json'), 'w') as f:
    json.dump(overlay, f, indent=2)
print(json.dumps(overlay, indent=2))
""")

md("""
## Part G — Findings & recommendations

### What the data says
1. **The economy sets the level.** The monthly default rate fell from **~0.85% (2019)** to **~0.33% (2024)** while unemployment (World Bank) fell from ~7.6% to 4.2%. COVID pushed it to **4.9% in June 2020** (two months after the lockdown began) and the second wave to ~1.1% in mid-2021. STL confirms a meaningful trend (F_T = 0.46) but only weak calendar seasonality (F_S = 0.10).
2. **Defaults react with a ~2-month lag** — the time a missed EMI needs to roll 30 → 60 → 90 DPD. The lockdown dummy correlates most strongly at lag 2 (r = 0.70).
3. **The macro series carry real predictive information.** Granger tests: unemployment (p = 0.0003), GDP growth (p = 0.004), the call-money rate (p = 0.010) and rupee depreciation (p = 0.046) all improve forecasts of future defaults; CPI is borderline (p = 0.051).
4. **Stationarity:** the log default rate is non-stationary in levels (ADF p = 0.08, KPSS p = 0.01) because of the macro-driven trend. Once the drivers enter the model the residuals are stationary (ADF p < 0.001), so SARIMAX can model levels without differencing.

### Which model and why
5. **SARIMAX(1,0,1)(1,0,0)₁₂ + five lagged macro drivers + two COVID dummies** was chosen by BIC (37.0 vs 56.7 for the next-best set). Every driver is significant with the economically expected sign:

| Driver (2 months earlier) | Effect on monthly default rate |
|---|---|
| Unemployment +1 pp | **+20.8%** |
| CPI inflation +1 pp | +15.3% |
| Call-money rate +1 pp | +10.4% |
| Rupee depreciation +1 pp | +2.0% |
| Real GDP growth +1 pp | −1.9% |
| Lockdown month | ×5.7 |
| Second-wave month | ×3.0 |

6. **Accuracy (2024 hold-out, 12 months ahead):** MAE **0.054 pp** (MAPE 16%) vs 0.088 pp for univariate SARIMA (−38%) and 0.124 pp for seasonal naive (−56%).
7. **Robustness (rolling backtest, 40 forecast origins, 480 forecasts):** mean MAE **0.124 pp** vs 0.176 (SARIMA), 0.208 (Holt-Winters) and 0.351 (seasonal naive). The seasonal-naive method collapses in the second wave (MAE 2.79 pp) because it copies the 2020 spike.
8. **Diagnostics:** residuals are white noise (Ljung-Box p = 0.12 at lag 6, 0.28 at lag 12) and approximately normal (Jarque-Bera p = 0.31). The prediction intervals are slightly conservative — 92% of 2024 actuals fall inside the 80% band, 100% inside the 95% band.
9. **Limitation, stated honestly:** no model foresaw the *start* of the second wave. Before April 2021 the dummy had never been switched on, so every model under-predicted it (SARIMAX ≈ SARIMA ≈ 0.38 pp error). Unprecedented shocks have to be handled with **scenarios**, not point forecasts — hence Part E.
10. **Early warning:** the 30+ DPD rate two months earlier explains 76% of the variation in the default rate and nowcasts 2024 with MAE 0.038 pp — better than any 12-month model. Operations teams should watch it monthly.

### 2025 outlook (₹, from Part E)
| Scenario | Avg monthly default rate | Expected defaults | Expected credit loss | vs baseline |
|---|---|---|---|---|
| Baseline | 0.28% | ~219 | ₹1.59 crore | — |
| Adverse (unemployment +1.5 pp, GDP −3 pp) | 0.33% | ~268 | ₹1.95 crore | **+₹36 lakh (+22%)** |
| Severe (COVID-style) | 0.74% (peak 2.7%) | ~596 | ₹4.32 crore | **+₹2.74 crore (+172%)** |

### Recommendations
* **Provisioning:** book ECL at the baseline (≈ ₹1.6 crore for 2025) and hold a management overlay of ≈ ₹36 lakh for the adverse case; use the severe case for the capital stress test.
* **Early-warning trigger:** if the portfolio 30+ DPD rate rises above ~4.5% (its 2019–2024 75th percentile), tighten approval cut-offs and shorten tenures for new loans, since defaults will follow within two months.
* **Point-in-time underwriting:** scale Review 1 PDs by the forecast multiplier (baseline **0.57×** of the neutral-economy rate, severe **1.53×**). The dashboard applies this overlay.
* **Re-fit monthly** as FRED / World Bank publish new data (`python -m src.macro_api`), and re-check the driver signs each time.
""")

nb["cells"] = cells
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
out = os.path.join(os.path.dirname(__file__), "..", "..", "notebooks", "06_R2_M1_time_series_forecasting.ipynb")
nbf.write(nb, out)
print("wrote", os.path.abspath(out))
