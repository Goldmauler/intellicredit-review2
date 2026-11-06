# IntelliCredit: FinTech Micro-Lending Risk Engine
## Review 2 Final Panel Presentation Deck & Defense Guide
**Duration: 15 Minutes | Total Marks: 20 | Panel Defense Structure**

---

### Timing Breakdown (15 Minutes)
| Segment | Minutes | Slides | Lead Presenter |
|---|:---:|:---:|---|
| **Project Foundation & Review 1 Summary** | 0:00 – 3:00 | Slides 1 – 3 | Team Member 1 |
| **Method 1: Time-Series Portfolio Forecasting** | 3:00 – 6:30 | Slides 4 – 6 | Team Member 2 |
| **Method 2: Delinquency Association Rule Mining** | 6:30 – 10:00 | Slides 7 – 8 | Team Member 3 |
| **Combined Insights & Live Interactive Dashboard** | 10:00 – 13:30 | Slides 9 – 10 | Team Member 4 |
| **Project Governance, Individual Contribution & Q&A** | 13:30 – 15:00 | Slides 11 – 12 | All Team Members |

---

## Slide 1: Title & Project Overview (1:00 min)
- **Title**: IntelliCredit: Alternative Credit Scoring & Portfolio Risk Engine for Thin-File & Gig-Economy Borrowers
- **Business Need**: 1.4 billion adults worldwide are unbanked (World Bank Global Findex 2021), and India's 7.7–12 million gig and platform workers (NITI Aayog 2022; Economic Survey 2025-26) mostly have thin or no credit files (CIBIL/FICO), facing either exclusion or predatory rates.
- **The Solution**: An integrated 3-layer analytics architecture:
  1. *Unit 1 & 2*: Interpretable Machine Learning default prediction with TreeSHAP.
  2. *Unit 3 Method 1*: Macro-driven portfolio default forecasting (SARIMAX + Live FRED/World Bank APIs).
  3. *Unit 3 Method 2*: Behavioral association rule mining (Apriori / FP-Growth) for delinquency patterns.
  4. *Production App*: Interactive loan underwriter dashboard with live filters and scenario stress testing.

---

## Slide 2: Overall Dataset Architecture & Validation (1:30 min)
- **Data Policy**: Programmatic statistical generation + Live macroeconomic institutional APIs (No scraped Kaggle/UCI sets).
- **Core Datasets**:
  - `borrower_profiles.csv`: 30,000 borrowers, 39 demographic/behavioral variables.
  - `monthly_transactions.csv`: 360,000 monthly transaction logs across 12 months.
  - `macro_india_monthly.csv`: 108 monthly observations (2017–2025) live from FRED and World Bank.
  - `loan_book.csv` & `portfolio_monthly.csv`: 30,000 originated loans, 250,191 loan-months over 72 months (2019–2024).
- **Quality Assurance**:
  - Leak-free stratified 80/20 borrower split (seed 42); chronological splits for all time-series models.
  - Fixed timestamp alignment using canonical month-start frequencies (`pd.date_range`).
  - Audited null values: Verified structural nulls in `default_month` (non-defaulting loans).

---

## Slide 3: Review 1 Summary — ML Default Prediction (1:30 min)
- **Feature Engineering**: 14 alternative behavioral ratios (income volatility index, utility reliability score, savings buffer ratio, income trend slope). Screened via Information Value (IV) and VIF < 5.
- **Model Benchmark**:
  - *Tuned XGBoost*: AUC-ROC = **0.738**, PR-AUC = **0.204** (Oracle theoretical ceiling = 0.781).
  - Captures **85.3%** of maximum achievable discrimination.
- **Cost-Optimal Threshold**: Balances asymmetric misclassification costs (Missed default: ₹81,900 vs. Wrong rejection: ₹9,700 lost margin). Approves 74.7% with an approved default rate of only 4.4% (vs. 7.87% baseline).

---

## Slide 4: Method 1 — Time-Series Data Preparation & Stationarity (1:30 min)
- **Target**: Monthly portfolio default rate over 72 consecutive months (2019–2024).
- **Data Preparation**:
  - ADF (p = 0.082) & KPSS (p = 0.01): non-stationary in levels; first difference stationary (ADF p < 0.001).
  - STL: trend strength 0.46, seasonality only 0.10 — COVID peaks dominate, not the calendar.
  - Granger (min p, lags 1–4): unemployment 0.0003, GDP 0.004, call-money 0.010, rupee 0.046; CPI borderline 0.051.

---

## Slide 5: Method 1 — Time-Series Model Comparison & Backtest (2:00 min)
- **Fair Comparison of 7 Architectures**: Evaluated on 2024 held-out test data (12 months) and a 40-origin expanding-window backtest:
  | Model | 2024 Hold-out MAE (pp) | RMSE (pp) | MAPE (%) | Backtest Mean MAE |
  |---|:---:|:---:|:---:|:---:|
  | **SARIMAX + Macro Drivers** | **0.054** | **0.068** | **16.36%** | **0.124** |
  | Gradient Boosting (Lags + Macro) | 0.065 | 0.073 | 21.91% | — |
  | Holt-Winters (ETS) | 0.075 | 0.102 | 23.86% | 0.208 |
  | SARIMA (Univariate) | 0.088 | 0.102 | 29.54% | 0.176 |
  | Seasonal Naive | 0.124 | 0.164 | 40.44% | 0.351 |
- **Residual Validation**: Ljung-Box p = 0.12 (lag 6) and 0.28 (lag 12); 80% interval covers 11/12 months of 2024.

---

## Slide 6: Method 1 — Macro Scenarios & Expected Credit Loss (1:00 min)
- **Translating Forecasts into Capital (ECL in ₹)**:
  - *Baseline Scenario*: 0.28% avg monthly default rate $\implies$ **₹1.59 Crore ECL**.
  - *Adverse Scenario* (Unemployment +1.5 pp, GDP −3 pp): 0.33% avg default $\implies$ **₹1.95 Crore ECL** (+₹36 Lakhs capital overlay).
  - *Severe Scenario* (COVID-scale shock): 0.74% avg default (peak 2.7%) $\implies$ **₹4.32 Crore ECL** (+₹2.74 Crore capital buffer).
- **Point-in-Time (PiT) Multiplier**: Converts through-the-cycle PD to macro-adjusted PD for dynamic underwriting.

---

## Slide 7: Method 2 — Association Rule Mining: Implementation (1:30 min)
- **Preparation**: Discretized 19 behavioral flags across 24,000 training borrowers into binary itemsets (cash flow, utility delays, recharge frequency, digital transactions).
- **Comparative Benchmark (Apriori vs. FP-Growth)**:
  - Both algorithms discover identical 2,852 frequent itemsets at $\text{min\_support} = 0.005$.
  - Apriori executed in **0.62s** vs FP-Growth in **1.13s** (Apriori faster due to low array overhead on dense binary structures).
- **Multi-Stage Pruning**:
  - 21,684 rules → 242 with consequent DEFAULT → 170 pass lift ≥ 1.5, support ≥ 0.5%, Bonferroni Fisher p < 0.05.
  - Productive-rule filter (beat every sub-rule by 5%) → **93 warning rules**.

---

## Slide 8: Method 2 — Rule Evaluation & Delinquency Insights (1:30 min)
- **Held-Out Test Set Generalization**:
  - **100% of the 93 rules generalize** on 6,000 unseen test borrowers (Test Lift $\ge 1.30\times$).
- **Top Validated Delinquency Pattern**:
  $$\text{DECLINING\_INCOME} \land \text{INCOME\_VOLATILE} \land \text{LOW\_SAVINGS\_BUFFER} \implies \text{DEFAULT}$$
  - $\text{Confidence} = 24.7\%$ ($3.14\times$ lift over 7.87% baseline).
- **Early-Warning Policy**:
  - Applicants with $\ge 3$ triggered rules account for **14% of applicants** but capture **34% of all portfolio defaults**.
  - 0 rules $\implies$ 4.9% default rate; $\ge 6$ rules $\implies$ **24.7% default rate** ($5\times$ risk spread).

---

## Slide 9: Combined Business Insights & Recommendations (1:30 min)
- **Multi-Method Synergies**:
  1. *Rules see risk the PD misses*: Inside the same lowest-risk band (Band A), applicants with $\ge 6$ rules default at **5.6%** vs **1.8%** with 0 rules ($3.1\times$ risk spread).
  2. *Macro Stress Interaction*: Under macroeconomic stress, income volatility adds **+4.5 pp** extra default risk, whereas low income adds 0 extra risk.
  3. *Dynamic PiT Underwriting Policy*:
     - Baseline Economy: 75.1% approval rate (6.1% approved default rate)
     - Severe Shock: Automatically contracts to 47.8% approval rate (3.98% approved default rate) under the same break-even rule.
- **Top Business Policy**: Two-tier underwriting: Auto-approve Bands A–B with $<3$ rules; conditional micro-ticket for $\ge 3$ rules or Band C; auto-decline Bands D–E.

---

## Slide 10: Mandatory Interactive Application (Live Demo) (2:00 min)
- **Tech Stack**: Streamlit (`dashboard/app.py`) + Standalone Plotly HTML (`reports/interactive_dashboard.html`).
- **Live Demo Flow**:
  1. *Dataset Explorer Tab*: Interactive browsing of final evaluated dataset with multi-attribute filters (Risk Band, Gig Platform, Default Status) that dynamically update summary KPIs.
  2. *Underwriter Tab*: Live scoring with calibrated LightGBM, interactive applicant slider adjustments, real-time TreeSHAP waterfall charts, and active warning rule badges.
  3. *Live Pre-Set Scenarios*:
     - Prime gig worker: PD 0.2% (baseline), 0 rules → APPROVE
     - Marginal / stressed worker: PD 8.3% (baseline) vs break-even 14.5%, 13 rules → APPROVE WITH CONDITIONS; switch to Severe → PD 22% → DECLINE
     - High-risk applicant: 75 rules; PD 21% in Baseline (conditional), 37% in Neutral and 57% in Severe → DECLINE, then show **Path to approval**
  4. *Portfolio Outlook Tab*: Interactive time-series forecast toggles, 2025 credit loss tables, and live macro indicator feeds.

---

## Slide 11: Project Governance, Tasks & Individual Contributions (1:00 min)
- **Agile Project Management**: Sprint tracking logged across 4 two-week milestones using a structured Kanban board with 5 rubric stages (*Backlog*, *To Do*, *In Progress*, *Review/Testing*, *Completed*) on [GitHub Tasks & Issues](https://github.com/Goldmauler/intellicredit-review2/issues?q=is%3Aissue).
- **Workload & Contribution Summary**:
  - **Darun R G**: Data generation engine, Review 1 ML pipeline (Notebooks 01–05), baseline benchmarks, EDA, SMOTE tuning.
  - **goldmauler**: Lead for Method 1 Time-Series (Notebook 06, SARIMAX, backtesting, macro APIs) & Method 2 Association Rules (Notebook 07, Apriori vs FP-Growth, rule pruning).
  - **Team Member 3**: Portfolio simulation engine (`portfolio_simulation.py`), DPD transition tracking, scenario loss quantification.
  - **Team Member 4**: Streamlit Interactive Dashboard (`dashboard/app.py`), UI/UX, Dataset Explorer, TreeSHAP integration.
  - **Team Member 5**: Combined business insights synthesis (Notebook 08), quality assurance audit, slide deck & technical report.

---

## Slide 12: Conclusion & Panel Defense Q&A Preparation (0:30 min)
- **Summary**: 100% implementation delivered across Units 1, 2, and 3 with full mathematical and business alignment.
- **Anticipated Panel Questions & Answers**:
  - *Q: Why did Apriori outperform FP-Growth in runtime?*  
    **A:** In dense binary baskets with 19 items and moderate itemset lengths, vector boolean operations in Apriori have lower overhead than constructing and traversing recursive conditional FP-Trees.
  - *Q: Have you validated your synthetic data against real-world benchmarks?*  
    **A:** Yes, and with real data, not only benchmarks. (1) All macro drivers are real (FRED, World Bank). (2) Notebook 09 re-runs Method 1 on India's real bank NPL ratio (IMF, 63 quarters: MAE 0.48 pp vs naive 2.04) and Method 2 on 48,799 real HMDA loans (82 rules, 95% hold in an unseen state). (3) A 140-country panel confirms the macro signs we simulate. (4) Real data also showed two weaknesses — no COVID forbearance and under-weighted debt burden — which we report openly.
  - *Q: How do you handle non-stationarity in SARIMAX?*  
    **A:** The log default rate is non-stationary in levels, but its trend is driven by the macro regressors; with them in the model the residuals are stationary (ADF p = 0.0004), so the BIC-chosen SARIMAX(1,0,1)×(1,0,0)₁₂ needs no differencing. The univariate SARIMA, which has no drivers, uses d = 1.
  - *Q: Why use Association Rules if you already have an ML model?*  
    **A:** ML models optimize global loss and can miss localized conjunctive danger zones. Our cross-tabulation proved that within the *same* ML risk band, applicants triggering multiple rules default at $3\times$ higher rates.
