# Data Dictionary — Review 2 datasets

## `external/macro_india_monthly.csv` — built by `src/macro_api.py`
| Column | Meaning | Source |
|---|---|---|
| `date` | month start (2017-01 … 2025-12) | |
| `cpi_inflation_yoy` | CPI all items, % change vs same month a year earlier | FRED `CPALTT01INM659N` |
| `call_money_rate` | immediate interbank (call money) rate, % p.a. | FRED `IRSTCI01INM156N` |
| `gsec_10y_yield` | 10-year government bond yield, % p.a. | FRED `INDIRLTLT01STM` |
| `inr_per_usd` | rupees per US dollar, monthly average | FRED `EXINUS` |
| `unemployment_rate` | unemployment, % of labour force (annual, interpolated to months) | World Bank `SL.UEM.TOTL.ZS` |
| `gdp_growth` | real GDP growth, annual % (interpolated) | World Bank `NY.GDP.MKTP.KD.ZG` |
| `lending_rate` | commercial lending rate, % (interpolated) | World Bank `FR.INR.LEND` |
| `inr_depreciation_yoy` | 12-month % change of `inr_per_usd` | derived |
| `covid_lockdown`, `covid_second_wave` | 1 in the event months (2020-03…06, 2021-04…06) | public-record event calendar |
| `*_is_carried_forward` | 1 where the value is carried forward beyond the last published observation | derived |

`external/macro_sources.json` records the API, series ID, whether each series was fetched live and its last observation. `external/raw/` holds the untouched API responses.

## `portfolio/loan_book.csv` — one row per loan (built by `src/portfolio_simulation.py`)
| Column | Meaning |
|---|---|
| `borrower_id` | links to `borrower_profiles.csv` |
| `loan_tenure_months`, `loan_amount`, `interest_rate_annual`, `emi` | loan terms (from the application; EMI = standard amortising instalment) |
| `origination_month`, `maturity_month` | calendar months (YYYY-MM) |
| `defaulted_in_panel`, `default_month` | 1 and month if the loan reached 90+ DPD between 2019-01 and 2024-12 |
| `censored_at_window_end` | 1 if the loan was still running at 2024-12 without defaulting |
| `loan_default_status`, `default_probability_true` | Review 1 label and true PD, for reference |

## `portfolio/loan_performance.csv` — one row per loan-month
| Column | Meaning |
|---|---|
| `borrower_id`, `month`, `loan_age` | loan and calendar month; loan age in instalments |
| `dpd` | days past due bucket: 0, 30, 60, 90 (90 = default, the loan leaves the panel) |
| `status` | current / delinquent / default |
| `emi_paid` | 1 if the instalment (and any arrears) was paid that month |
| `outstanding_principal` | balance after this month's payment |

## `portfolio/portfolio_monthly.csv` — one row per month (2019-01 … 2024-12)
| Column | Meaning |
|---|---|
| `active_loans` | loans with an instalment due this month |
| `new_defaults`, `default_rate` | loans reaching 90+ DPD this month, and ÷ active loans (**forecast target**) |
| `dpd30_plus(_rate)`, `dpd60_plus(_rate)` | loans 30+ / 60+ days past due (early-warning indicators) |
| `new_originations`, `outstanding_principal` | book growth and size |
| macro columns, `stress_index` | macro drivers for that month; `stress_index` is the simulator's z-score composite (not used as a model input) |
