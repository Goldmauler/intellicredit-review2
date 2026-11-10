"""
Scenario engine for the dashboard (no Streamlit code here, so it can be tested on its own).

It refits the SARIMAX model of notebook 06 on the 72-month portfolio series and the macro file, then forecasts 2025
under any macro path. Reproduced from files: coefficients, AIC/BIC and the stored Baseline, Adverse and Severe paths.

    log(default rate %)_t = c + sum_k beta_k * driver_k,t-2 + u_t,     (1 - phi B)(1 - PHI B^12) u_t = (1 + theta B) e_t
"""
import os
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

DRIVERS = ["unemployment_rate", "gdp_growth", "cpi_inflation_yoy", "call_money_rate", "inr_depreciation_yoy",
           "covid_lockdown", "covid_second_wave"]
LAG = 2
ORDER, SORDER = (1, 0, 1), (1, 0, 0, 12)
FUT = pd.date_range("2025-01-01", "2025-12-01", freq="MS")
LABELS = {"unemployment_rate": "Unemployment (pp)", "gdp_growth": "GDP growth (pp)", "cpi_inflation_yoy": "Inflation (pp)",
          "call_money_rate": "Call-money rate (pp)", "inr_depreciation_yoy": "Rupee depreciation (pp)",
          "covid_lockdown": "Lockdown months", "covid_second_wave": "Second-wave months"}

PRESETS = {  # exactly the shocks used in notebook 06, Part E
    "Baseline": dict(unemployment_rate=0.0, gdp_growth=0.0, cpi_inflation_yoy=0.0, call_money_rate=0.0, inr_depreciation_yoy=0.0,
                     lockdown=[], wave=[]),
    "Adverse": dict(unemployment_rate=1.5, gdp_growth=-3.0, cpi_inflation_yoy=0.0, call_money_rate=0.0, inr_depreciation_yoy=0.0,
                    lockdown=[], wave=[]),
    "Severe": dict(unemployment_rate=3.0, gdp_growth=-8.0, cpi_inflation_yoy=0.0, call_money_rate=0.0, inr_depreciation_yoy=0.0,
                   lockdown=["2025-04-01", "2025-05-01"], wave=[]),
}


def load(data_dir):
    pm = pd.read_csv(os.path.join(data_dir, "portfolio", "portfolio_monthly.csv"), parse_dates=["date"]).set_index("date").asfreq("MS")
    macro = pd.read_csv(os.path.join(data_dir, "external", "macro_india_monthly.csv"), parse_dates=["date"]).set_index("date").asfreq("MS")
    return pm, macro


def _exog(macro, index):
    return pd.DataFrame({f"{d}_l{LAG}": macro[d].shift(LAG) for d in DRIVERS}).reindex(index)


def fit(pm, macro):
    """Final model: all 72 months, as in the notebook (the 2023-trained model is only used for the hold-out test)."""
    ly = np.log(pm["default_rate"] * 100)
    exog = _exog(macro, pm.index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SARIMAX(ly, exog=exog, order=ORDER, seasonal_order=SORDER, trend="c", enforce_stationarity=True,
                       enforce_invertibility=True).fit(disp=False, maxiter=300)


def scenario_macro(macro, shocks, start="2025-03-01", ramp_months=3):
    """Macro path = baseline macro + shock x ramp (reaches the full shock after ramp_months), plus lockdown / wave dummies."""
    m = macro.copy()
    ramp = pd.Series(0.0, index=m.index)
    tail = ramp.loc[start:]
    ramp.loc[start:] = np.minimum(1, (np.arange(len(tail)) + 1) / max(1, ramp_months))
    for d in ("unemployment_rate", "gdp_growth", "cpi_inflation_yoy", "call_money_rate", "inr_depreciation_yoy"):
        m[d] = m[d] + shocks.get(d, 0.0) * ramp
    for key, col in (("lockdown", "covid_lockdown"), ("wave", "covid_second_wave")):
        for dt in shocks.get(key, []):
            m.loc[pd.Timestamp(dt), col] = 1
    return _exog(m, m.index).loc[FUT]


def forecast(res, exog, alpha=0.2):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = res.get_forecast(len(FUT), exog=exog)
    ci = np.exp(f.conf_int(alpha=alpha))
    return pd.DataFrame({"forecast": np.exp(f.predicted_mean), "lower": ci.iloc[:, 0], "upper": ci.iloc[:, 1]}, index=FUT)


def neutral_rate(res, macro, pm):
    """Through-the-cycle rate: every driver at its 2019-2024 average, no COVID flags (notebook 06, Part E)."""
    m = macro.copy()
    for d in DRIVERS[:5]:
        m[d] = macro.loc["2019-01-01":"2024-12-01", d].mean()
    m[["covid_lockdown", "covid_second_wave"]] = 0
    return float(forecast(res, _exog(m, m.index).loc[FUT])["forecast"].mean())


def coefficients(res):
    names = [f"{d}_l{LAG}" for d in DRIVERS]
    t = pd.DataFrame({"beta": res.params[names], "p": res.pvalues[names]})
    t.index = DRIVERS
    return t


if __name__ == "__main__":      # reproduction check against the files written by notebook 06
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root, "data")
    pm, macro = load(data_dir)
    res = fit(pm, macro)
    stored = pd.read_csv(os.path.join(root, "models", "ts_forecast_scenarios.csv"), parse_dates=["date"])
    nr = neutral_rate(res, macro, pm)
    print("AIC/BIC", round(res.aic, 2), round(res.bic, 2), "| neutral rate", round(nr, 4))
    for k, sh in PRESETS.items():
        mine = forecast(res, scenario_macro(macro, sh))["forecast"]
        ref = stored[stored.scenario == k].set_index("date")["forecast"]
        print(f"{k:9s} max abs diff vs stored = {float((mine - ref).abs().max()):.5f} | avg {mine.mean():.4f} | multiplier {mine.mean() / nr:.3f}")
