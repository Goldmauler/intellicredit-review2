# IntelliCredit: FinTech Micro-Lending Risk & Alternative Credit Scoring Engine
## Final Comprehensive Technical Report — Review 2 (Unit 3: Advanced Analytics)
**Capstone Project in Business Analytics**  
**Total Marks: 20 | Final Panel Presentation: 15 Minutes**

---

## Executive Summary

Traditional financial credit assessment is deeply reliant on conventional credit bureau scores (e.g., CIBIL, FICO) and formal salaried bank statements. Consequently, 1.4 billion adults worldwide are unbanked (World Bank Global Findex 2021), and India's 7.7–12 million gig and platform workers (NITI Aayog 2022; Economic Survey 2025-26) mostly have thin or no credit files (CIBIL/FICO), and are systematically excluded from formal credit markets or forced to borrow from predatory informal channels. 

**IntelliCredit** solves this fundamental business problem through a unified, 3-layer risk intelligence system:
1. **Unit 1 & 2 (Predictive Modelling Layer)**: An interpretable machine learning engine built on high-frequency alternative transaction features, evaluated with TreeSHAP and cost-optimal risk cutoff thresholds.
2. **Unit 3 Method 1 (Macro Time-Series Layer)**: A dynamic portfolio default-rate forecasting engine combining 72 months of portfolio loan performance with live macroeconomic API indicators (FRED & World Bank), evaluated via rolling-origin backtests and stress scenarios.
3. **Unit 3 Method 2 (Behavioral Association Rule Mining Layer)**: Delinquency early-warning pattern mining using Apriori and FP-Growth, yielding 93 non-redundant, statistically significant rules validated on held-out unseen borrowers.
4. **Mandatory Interactive Application**: A production-grade Streamlit web application providing live loan underwriting, TreeSHAP explainability, portfolio stress testing, rule exploration, and multi-filter dataset exploration.

---

## 1. Overall Dataset Architecture & Validation

### 1.1 Data Sources and Collection Strategy
The project avoids static scraped datasets from public repositories, establishing an authentic, reproducible data pipeline:
- **Synthetic Micro-Transaction Engine (`data_generation.py`)**: Generates 30,000 distinct borrower profiles and 360,000 longitudinal monthly transactions using realistic Indian demographic distributions (`Faker en_IN`) and conditional behavioral dependencies (income volatility, utility punctuality, recharge cadence, UPI volume, and BNPL obligations).
- **Live Institutional Macroeconomic APIs (`src/macro_api.py`)**: Directly harvests 108 continuous monthly observations (2017–2025) from the Federal Reserve Bank of St. Louis (FRED) and World Bank Development Indicators. Series include CPI Inflation YoY, Interbank Call-Money Rate, 10-Year G-Sec Yield, USD/INR Exchange Rate, Unemployment Rate, and GDP Growth.
- **Dynamic Portfolio Performance Panel (`src/portfolio_simulation.py`)**: Simulates 30,000 loans originating across a 6-year window (2019–2024), generating 250,191 loan-month records tracking DPD migration ($0 \to 30 \to 60 \to 90+$ default) linked to macroeconomic shocks.

### 1.2 Data Quality Assurance & Validation Summary
- **Data Leakage Elimination**: Stratified random 80/20 train/test split of borrowers (24,000 / 6,000, seed 42) for the PD model and rules; strictly chronological splits for every time-series model. Target encodings, WoE transformations, and SMOTE balancing were constrained strictly to training folds.
- **Continuity & Structure Audits**: Resolved timestamp drift issues by enforcing canonical monthly start frequencies (`pd.date_range freq='MS'`).
- **Missing Value & Outlier Verification**: Validated structural nulls in `loan_book.csv` (28,514 non-default loans) and mathematical lag truncation in `macro_india_monthly.csv` (first 12 months for 12-month YoY differencing). Winsorized extreme outliers at the 1st and 99th percentiles.

### 1.3 Real-World Benchmark Validation (Faculty Feedback Response)
To address the faculty recommendation regarding real-world data validation, the synthetic feature distributions and portfolio defaults were systematically benchmarked against real-world Indian microfinance and labor market datasets:
- **Macroeconomic Reality**: All 7 macroeconomic indicators (CPI inflation, call-money rate, G-Sec yield, INR/USD exchange rate, unemployment, GDP growth, and commercial lending rate) are **100% real institutional data** directly collected from the Federal Reserve Bank of St. Louis (FRED) and the World Bank.
- **Default Rate Benchmarking**: The synthetic default rate of 7.87% is deliberately above MFIN Micrometer PAR 31–180 (2.2% Mar 2024, 6.3% Mar 2025 stress peak, 1.6% Jun 2026), because the segment is thin-file gig workers outside MFI portfolios.
- **Income Grounding**: The Economic Survey 2025-26 reports that about 40% of gig workers earn under ₹15,000 a month; our bottom-quartile income threshold is ₹19,485. (An earlier draft cited a "CGAP India Gig Economy Financial Diary" and a PLFS income band; neither could be verified, so both were removed.)
- **COVID Shock (corrected)**: The simulated 2020 default rate is 2.3× the 2019 level. Real Indian data do **not** show such a spike: the bank NPL ratio fell (IMF FSI; SCB GNPA 8.2% → 7.3% in RBI Trend & Progress 2020-21) because the moratorium and the Supreme Court standstill deferred recognition. The COVID shock is therefore an explicit modelling assumption, and notebook 09 shows the real-data consequence.
- **NEW — real-data track (notebook 09):** Method 1 re-run on India's real NPL ratio (IMF, 63 quarters) and Method 2 on 48,799 real HMDA loan applications; see Section 8.
- *Detailed benchmark tables, sources, and full comparative analysis are documented in Section 7 of data/DATASET_VALIDATION_SUMMARY.md.*

---

## 2. Review 1 Summary (Units 1 & 2): Predictive Scoring & Base Pipeline

- **Exploratory Data Analysis**: Conducted 38 comprehensive visualizations, normality tests, and hypothesis tests (Mann-Whitney U tests for continuous features, Chi-Square tests of independence for categorical variables).
- **Feature Engineering & Selection**: Engineered 14 alternative behavioral ratios (e.g., `income_volatility_index`, `utility_reliability_score`, `savings_buffer_ratio`, `income_trend_slope`). Applied Information Value (IV), Recursive Feature Elimination (RFE), and Variance Inflation Factor (VIF < 5) screening.
- **Predictive Modelling**: Developed and tuned 5 baseline and tree-based models using leak-free Stratified K-Fold CV with embedded SMOTE:
  - *Tuned XGBoost*: **AUC-ROC = 0.738**, **PR-AUC = 0.204**
  - *Tuned LightGBM*: **AUC-ROC = 0.735**, **PR-AUC = 0.201**
  - *Tuned Logistic Regression*: **AUC-ROC = 0.728**, **PR-AUC = 0.197**
  - *Theoretical Ceiling (True PD Oracle)*: **AUC-ROC = 0.781**
  - *Discrimination Capture*: Tuned XGBoost achieves **85.3%** of the maximum possible discrimination on this stochastic data.
- **Cost-Optimal Decision Policy**: Implemented cost-sensitive classification balancing asymmetric financial losses (Missed Default Cost $\approx$ ₹81,900; Wrong Rejection Lost Margin $\approx$ ₹9,700), yielding a 74.7% approval rate with an approved default rate of only 4.4% (vs. 7.87% baseline).

---

## 3. Review 2 — Unit 3 Method 1: Time-Series Analysis & Default Forecasting

### 3.1 Data Preparation & Method Suitability (Rubric: 2 Marks)
- **Target Variable**: Monthly Portfolio Default Rate ($\frac{\text{New 90+ DPD Loans}}{\text{Active Loans}}$), aggregated across 72 consecutive months (2019-01 to 2024-12) from the loan performance panel.
- **Suitability Justification**: While individual borrower default is cross-sectional, loan loss reserves and capital adequacy require predicting portfolio-level cyclical waves driven by macro dynamics.
- **Time-Series Preprocessing**:
  - *Stationarity Testing*: ADF ($p = 0.082$) and KPSS ($p = 0.01$) say the log default rate is non-stationary in levels; its first difference is stationary (ADF $p < 0.001$, KPSS $p = 0.10$). The chosen SARIMAX needs no differencing because the macro drivers explain the trend (residual ADF $p = 0.0004$).
  - *Decomposition*: STL gives trend strength $F_T = 0.46$ and only weak seasonality $F_S = 0.10$; the big movements are the COVID peaks, not the calendar.
  - *Granger Causality* (min p over lags 1–4): unemployment $p = 0.0003$, GDP growth $p = 0.004$, call-money rate $p = 0.010$, rupee depreciation $p = 0.046$ Granger-cause the default rate; CPI inflation is borderline ($p = 0.051$).

### 3.2 Model Implementation & Comparative Evaluation (Rubric: 3 Marks)
Seven competing time-series models were implemented and evaluated across both a 2024 held-out 12-month test window and a 40-origin expanding-window backtest:

| Model Architecture | 2024 Hold-out MAE (pp) | RMSE (pp) | MAPE (%) | MASE | Backtest Mean MAE (40 Origins) |
|---|:---:|:---:|:---:|:---:|:---:|
| **SARIMAX$(1,0,1)\times(1,0,0)_{12}$ + 5 lagged macro drivers + 2 COVID dummies** | **0.054** | **0.068** | **16.36%** | **0.079** | **0.124** |
| Gradient Boosting Regressor (Lag + Macro) | 0.065 | 0.073 | 21.91% | 0.094 | — |
| Holt-Winters Exponential Smoothing (ETS) | 0.075 | 0.102 | 23.86% | 0.108 | 0.208 |
| Univariate SARIMA $(1,1,1)$ (no seasonal terms, chosen by AIC) | 0.088 | 0.102 | 29.54% | 0.127 | 0.176 |
| Naive Random Walk Benchmark | 0.088 | 0.112 | 24.20% | 0.127 | — |
| 12-Month Moving Average | 0.116 | 0.132 | 42.28% | 0.167 | — |
| Seasonal Naive Benchmark | 0.124 | 0.164 | 40.44% | 0.178 | 0.351 |

### 3.3 Diagnostic Verification & Stress Scenarios
- **Residual Diagnostics**: Ljung-Box finds no residual autocorrelation ($p = 0.12$ at lag 6; $p = 0.28$ at lag 12); Jarque-Bera $p = 0.31$ (normality not rejected); residual mean 0.007.
- **Coverage Accuracy**: The 80% interval covers 11 of 12 months of 2024 (92%) and the 95% interval 12 of 12 — slightly too wide on a calm year. Notebook 09 adds **conformal intervals** that keep nominal coverage on real data.
- **2025 Macroeconomic Stress Scenarios**:
  - *Baseline Scenario*: 0.28% avg monthly default rate $\implies$ ₹1.59 Crore Expected Credit Loss (ECL).
  - *Adverse Scenario* (Unemployment +1.5 pp, GDP −3.0 pp): 0.33% avg default rate $\implies$ ₹1.95 Crore ECL (+₹36 Lakhs capital overlay).
  - *Severe Scenario* (COVID-scale disruption): 0.74% avg default rate (2.7% peak) $\implies$ ₹4.32 Crore ECL (+₹2.74 Crore capital buffer).

---

## 4. Review 2 — Unit 3 Method 2: Association Rule Mining of Delinquency Patterns

### 4.1 Data Preparation & Implementation (Rubric: 2 Marks)
- **Feature Discretization**: Formulated 19 discrete binary risk items from borrower financial signals across cash flow, credit history, repayment punctuality, and digital activity:
  - Cash flow: `LOW_INCOME`, `INCOME_VOLATILE`, `DECLINING_INCOME`, `LOW_SAVINGS_BUFFER`, `HIGH_BNPL_BURDEN`
  - Repayment: `LATE_UTILITY_PAYER`, `LONG_UTILITY_DELAYS`, `WEAK_REPAYMENT_HISTORY`, `NO_CREDIT_HISTORY`
  - Digital footprint: `LOW_APP_ENGAGEMENT`, `IRREGULAR_RECHARGE`, `NO_BANK_ACCOUNT`
  - Loan terms: `LARGE_LOAN_VS_INCOME`, `HIGH_INTEREST_LOAN`
  - Profile: `NEW_GIG_WORKER`, `YOUNG_BORROWER`, `MANY_DEPENDENTS`, `METRO_RESIDENT`, `RURAL_RESIDENT`
- **Itemset Mining**: Generated boolean basket representations for all 24,000 training borrowers.
- **Comparative Algorithm Benchmark (Apriori vs. FP-Growth)**:
  - Both algorithms discovered an identical set of 2,852 frequent itemsets at $\text{min\_support} = 0.005$.
  - Apriori executed in **0.62 seconds** vs. FP-Growth in **1.13 seconds**; on moderately-sized dense binary baskets, Apriori's lightweight array operations avoid the recursive tree-construction overhead of FP-Growth.

### 4.2 Statistical Filtering, Pruning & Out-of-Sample Evaluation (Rubric: 3 Marks)
- **Multi-Stage Pruning**:
  1. *Rule generation*: 2,852 frequent itemsets → 21,684 rules with lift ≥ 1 → 242 with consequent `DEFAULT`.
  2. *Significance Filter*: Lift $\ge 1.5$, support $\ge 0.5\%$ and Bonferroni-corrected one-sided Fisher exact $p < 0.05$ → 170 rules.
  3. *Productive-rule filter*: keep a rule only if its confidence beats every sub-rule by ≥ 5% (relative) → **93 non-redundant warning rules**.
- **Held-Out Test Validation (6,000 Unseen Borrowers)**:
  - **100% of the 93 rules generalize out-of-sample**, maintaining a test lift $\ge 1.30\times$.
  - *Top Validated Rule*:  
    $$\text{DECLINING\_INCOME} \land \text{INCOME\_VOLATILE} \land \text{LOW\_SAVINGS\_BUFFER} \implies \text{DEFAULT}$$  
    Train: support 0.60%, confidence 20.2%, lift 2.57. **Unseen test borrowers: coverage 3.85%, default rate 24.7%, lift 3.14** (vs 7.87% base).
- **Early-Warning Policy Threshold**:
  - Applicants triggering $\ge 3$ rules constitute **14%** of the applicant pool but capture **34% of all portfolio defaults**.
  - Applicants with 0 triggered rules exhibit a **4.9%** default rate, whereas applicants with $\ge 6$ triggered rules default at **24.7%** ($5\times$ increase).

---

## 5. Combined Business Insights & Strategic Recommendations (Rubric: 2 Marks)

### 5.1 Integrated Method Synergies
The intersection of the three analytical units reveals critical insights inaccessible to any single model:

1. **Association Rules Detect Risk Missed by the ML PD Score**:
   - Within the *same* lowest-risk PD band (Band A: $\text{PD} < 3\%$), applicants triggering $\ge 6$ warning rules default at **5.6%** compared to only **1.8%** for applicants with 0 rules ($3.1\times$ risk spread).
   - In moderate-risk Band C, the default rate jumps from 7.3% to 10.3%. The association rules function as an indispensable behavioral "second opinion."
2. **Behavioral Risk is Amplified by Macroeconomic Stress**:
   - Conditional rule analysis reveals that under macro stress, *income volatility* adds an extra **+4.5 percentage points** to borrower default rates, whereas demographic factors (*low income*, *rural residence*) show zero incremental default elevation.
3. **Dynamic Point-in-Time (PiT) Underwriting**:
   - Underwriting policies adjust automatically via the time-series multiplier:
     $$\text{PD}_{\text{PiT}} = \text{PD}_{\text{TTC}} \times \text{Scenario Multiplier}$$
   - Under the identical break-even economic rule ($\text{PD} < \text{Margin} / [\text{Margin} + \text{LGD}]$), approval rates adapt safely:
     - **Baseline (Benign)**: Approval = **75.1%** | Approved Default Rate = 6.10%
     - **Neutral**: Approval = **60.3%** | Approved Default Rate = 5.09%
     - **Severe Shock**: Approval = **47.8%** | Approved Default Rate = 3.98%

### 5.2 Actionable Recommendations for Lenders
| # | Policy Action | Operational Rule | Empirical Evidence |
|---|---|---|---|
| **1** | **Two-Tier Decision Engine** | Auto-approve only in Bands A–B with $< 3$ triggered rules; route applicants with $\ge 3$ rules or Band C to structured micro-products (weekly EMIs, reduced tickets). | Section 2 cross-tab; catches 34% of defaults within 14% of applicants. |
| **2** | **Dynamic Loan Sizing** | Cap loan-to-monthly-income ratio at $1.5\times$ whenever gig tenure is $< 6$ months or income trend is negative. | Unit 1/2 SHAP rankings + Rule 3 (`LARGE_LOAN_VS_INCOME`). |
| **3** | **Monthly Macro Repricing** | Ingest live monthly FRED/World Bank data to update the PiT multiplier dynamically. | SARIMAX 2-month leading indicator accuracy. |
| **4** | **Selective Downturn Tightening** | Restrict underwriting on volatile income and late utility payments during downturns; protect approvals for stable low-income applicants. | Preserves financial inclusion without degrading portfolio quality. |
| **5** | **Early Warning Trigger** | Track the 30+ DPD rate monthly as a 2-month-ahead nowcast of defaults (illustrative alert level ~4.5%, to be calibrated). | log-log nowcast $R^2 = 0.76$ on training months; 2024 nowcast MAE 0.038 pp vs 0.054 pp for the best 12-step model. |
| **6** | **Capital Buffer Allocation** | Provision baseline expected loss ₹1.59 Crore + ₹36 Lakhs adverse buffer. | Value-at-Risk scenario projections. |

---

## 6. Interactive Application Architecture (Rubric: 5 Marks)

The project includes an enterprise-grade interactive dashboard deployed via **Streamlit** (`dashboard/app.py`) accompanied by a standalone offline HTML visualization report (`reports/interactive_dashboard.html`).

### 6.1 Interactive Capabilities & Rubric Compliance
- **Final Dataset Explorer**: Interactive data browser enabling search, pagination, and multi-variable filtering (Risk Band, Gig Platform, Location, Default Status) that dynamically recalculates active exposure, average PD, and observed default rates in real time.
- **Dynamic Drilldown**: Direct one-click selection of any applicant from the dataset into the Underwriting Decision Engine.
- **Live Underwriter Engine**: Real-time scoring using calibrated LightGBM, interactive applicant form inputs, instant TreeSHAP feature contribution waterfall charts, and active warning rule badges.
- **Macro Scenario Simulation**: Dynamic radio toggles switching macroeconomic states (Neutral, Baseline, Adverse, Severe), dynamically updating break-even frontiers, loss provisions, and approval flags.
- **Rule Mining Explorer**: Interactive Plotly scatter plot with dynamic sliders for minimum test lift, coverage, and antecedent item multi-selectors.

---

## 7. Limitations & Ethical Considerations

1. **Synthetic Borrower Signals**: Applicant profiles and transactions are generated; macro drivers and the five notebook-09 datasets are real. Notebook 09 found two generator weaknesses to fix — no COVID forbearance regime and under-weighted debt burden.
2. **Temporal Interpolation**: World Bank series (unemployment, GDP growth) are published annually and interpolated to monthly frequency.
3. **Fairness & Inclusion**: The model actively avoids penalizing applicants based on protected demographic attributes (gender, location tier), focusing purely on behavioral cash-flow consistency.

---

## 8. Real-Data Validation Track & Innovations (added after Review 1 feedback)

`src/real_data.py` collects five real credit datasets from official open APIs (cached raw responses, provenance in
`data/real/real_sources.json`); `notebooks/09_R3_real_data_validation.ipynb` re-runs both Unit 3 methods on them.

| Test | Real data | Result |
|---|---|---|
| Method 1 on real Indian defaults | IMF FSI India bank NPL ratio, 2010-Q1 → 2026-Q2 | SARIMAX with RBI regime ramps (AQR recognition +8.0%/qtr, clean-up −5.4%/qtr), chosen by BIC: 14-quarter MAE **0.48 pp** vs ETS 0.91, ARIMA 1.03, naive 2.04 |
| Conformal intervals (innovation) | FRED US consumer-loan delinquency, 94 rolling forecasts 2003–2026 | Adaptive conformal 80% band covers **81%** (90% band: 90%); the Gaussian model band covers only 72% (83%) |
| Is the simulated macro mechanism real? | World Bank panel, 140 countries, 2,021 country-years | Δunemployment +0.28 (p = 0.015), GDP growth −0.11 (p = 0.048), inflation +0.08 (p = 0.002): same signs as the simulation |
| Method 2 on real loan records | CFPB HMDA 2023, 48,799 applications | 82 productive DENIED rules mined on DE + RI; **95% keep lift ≥ 1.3 in Vermont**; rule-count AUC 0.707 |
| Fairness | HMDA | after the 16 risk flags, Black and Asian applicants still have ~1.7× the odds of denial (p < 0.001); female 1.01 (n.s.). Public HMDA has no credit scores, so this is a review signal, not proof |
| Borrower pain points | CFPB, 15,000 payday/personal-loan complaints | top issue "charged fees or interest you didn't expect" (21.7%) — supports RBI's Key Fact Statement / all-in APR disclosure rule |

**What the real data changed in our conclusions**
1. In India's real credit cycle, **regulatory recognition regimes** (AQR 2015–18, IBC clean-up, COVID standstill) moved reported NPLs more than the macro drivers; a production forecaster must carry regime terms.
2. The simulation's COVID spike has no counterpart in real Indian NPL data (forbearance deferred it) → add a forbearance regime to `src/portfolio_simulation.py`.
3. Real lenders' strongest single signal is debt-to-income ≥ 50% (lift 2.55); the generator under-weights loan-to-income (lift 1.12) → recalibrate.

**Dashboard innovations:** (a) *Path to approval* — actionable counterfactual recourse for declined applicants (`src/recourse.py`;
Wachter et al. 2017, Ustun et al. 2019, Mothilal et al. 2020); (b) *Real-Data Reality Check* tab with the India forecasts,
the conformal band, the HMDA rule explorer and the fidelity scorecard.

Literature, benchmark tables and reviewer answers with verified citations: `research/`.
