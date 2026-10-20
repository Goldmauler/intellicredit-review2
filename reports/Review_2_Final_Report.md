# IntelliCredit: FinTech Micro-Lending Risk & Alternative Credit Scoring Engine
## Final Comprehensive Technical Report — Review 2 (Unit 3: Advanced Analytics)
**Capstone Project in Business Analytics**  
**Total Marks: 20 | Final Panel Presentation: 15 Minutes**

---

## Executive Summary

Traditional financial credit assessment is deeply reliant on conventional credit bureau scores (e.g., CIBIL, FICO) and formal salaried bank statements. Consequently, over 1.4 billion gig-economy workers, delivery contractors, and thin-file individuals are systematically excluded from formal credit markets or forced to borrow from predatory informal channels. 

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
- **Data Leakage Elimination**: Strict chronological and stratified train/test partitioning (80% train = 24,000; 20% test = 6,000). Target encodings, WoE transformations, and SMOTE balancing were constrained strictly to training folds.
- **Continuity & Structure Audits**: Resolved timestamp drift issues by enforcing canonical monthly start frequencies (`pd.date_range freq='MS'`).
- **Missing Value & Outlier Verification**: Validated structural nulls in `loan_book.csv` (28,514 non-default loans) and mathematical lag truncation in `macro_india_monthly.csv` (first 12 months for 12-month YoY differencing). Winsorized extreme outliers at the 1st and 99th percentiles.

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
  - *Stationarity Testing*: Augmented Dickey-Fuller (ADF) test ($p = 0.18$) and KPSS test ($p < 0.05$) confirmed non-stationarity in levels. First-order differencing achieved stationary white-noise properties ($p_{\text{ADF}} < 0.001$).
  - *Decomposition*: Seasonal-Trend Decomposition using LOESS (STL) identified a 12-month annual seasonality and cyclical peaks during economic disruptions.
  - *Granger Causality*: Verified that lagged unemployment ($p = 0.008$), call-money rate ($p = 0.019$), and CPI inflation ($p = 0.024$) Granger-cause portfolio defaults with a 2-month lead time.

### 3.2 Model Implementation & Comparative Evaluation (Rubric: 3 Marks)
Seven competing time-series models were implemented and evaluated across both a 2024 held-out 12-month test window and a 40-origin expanding-window backtest:

| Model Architecture | 2024 Hold-out MAE (pp) | RMSE (pp) | MAPE (%) | MASE | Backtest Mean MAE (40 Origins) |
|---|:---:|:---:|:---:|:---:|:---:|
| **SARIMAX + Macro Exogenous Drivers** | **0.054** | **0.068** | **16.36%** | **0.079** | **0.124** |
| Gradient Boosting Regressor (Lag + Macro) | 0.065 | 0.073 | 21.91% | 0.094 | — |
| Holt-Winters Exponential Smoothing (ETS) | 0.075 | 0.102 | 23.86% | 0.108 | 0.208 |
| Univariate SARIMA $(1,1,1)\times(1,0,1)_{12}$ | 0.088 | 0.102 | 29.54% | 0.127 | 0.176 |
| Naive Random Walk Benchmark | 0.088 | 0.112 | 24.20% | 0.127 | — |
| 12-Month Moving Average | 0.116 | 0.132 | 42.28% | 0.167 | — |
| Seasonal Naive Benchmark | 0.124 | 0.164 | 40.44% | 0.178 | 0.351 |

### 3.3 Diagnostic Verification & Stress Scenarios
- **Residual Diagnostics**: Ljung-Box test on SARIMAX residuals confirmed no autocorrelation ($p = 0.12$ at lag 12; $p = 0.28$ at lag 24). Standardized residuals adhere to a Gaussian distribution.
- **Coverage Accuracy**: The 80% prediction interval covers 91.7% of the 2024 actual observed values.
- **2025 Macroeconomic Stress Scenarios**:
  - *Baseline Scenario*: 0.28% avg monthly default rate $\implies$ ₹1.59 Crore Expected Credit Loss (ECL).
  - *Adverse Scenario* (Unemployment +1.5 pp, GDP −3.0 pp): 0.33% avg default rate $\implies$ ₹1.95 Crore ECL (+₹36 Lakhs capital overlay).
  - *Severe Scenario* (COVID-scale disruption): 0.74% avg default rate (2.7% peak) $\implies$ ₹4.32 Crore ECL (+₹2.74 Crore capital buffer).

---

## 4. Review 2 — Unit 3 Method 2: Association Rule Mining of Delinquency Patterns

### 4.1 Data Preparation & Implementation (Rubric: 2 Marks)
- **Feature Discretization**: Formulated 19 discrete binary risk items from borrower financial signals across cash flow, credit history, repayment punctuality, and digital activity:
  - Cash flow: `LOW_INCOME`, `INCOME_VOLATILE`, `DECLINING_INCOME`, `LOW_SAVINGS_BUFFER`
  - Repayment: `LATE_UTILITY_PAYER`, `PERSISTENT_UTILITY_DELINQUENT`, `BNPL_HEAVY_USER`
  - Digital footprint: `LOW_DIGITAL_ENGAGEMENT`, `INFREQUENT_RECHARGE`
  - Loan terms: `LARGE_LOAN_VS_INCOME`, `LONG_TENURE`, `HIGH_INTEREST_RATE`
- **Itemset Mining**: Generated boolean basket representations for all 24,000 training borrowers.
- **Comparative Algorithm Benchmark (Apriori vs. FP-Growth)**:
  - Both algorithms discovered an identical set of 2,852 frequent itemsets at $\text{min\_support} = 0.005$.
  - Apriori executed in **0.62 seconds** vs. FP-Growth in **1.13 seconds**; on moderately-sized dense binary baskets, Apriori's lightweight array operations avoid the recursive tree-construction overhead of FP-Growth.

### 4.2 Statistical Filtering, Pruning & Out-of-Sample Evaluation (Rubric: 3 Marks)
- **Multi-Stage Pruning**:
  1. *Significance Filter*: Retained rules with $\text{Confidence} \ge 15\%$, $\text{Lift} \ge 1.50$, and Bonferroni-corrected Fisher's Exact Test $p < 0.05$.
  2. *Redundancy Pruning*: Eliminated subset rules that provide no additional confidence over simpler constituent patterns.
  3. *Result*: 2,852 raw itemsets compressed into **93 high-conviction, non-redundant warning rules**.
- **Held-Out Test Validation (6,000 Unseen Borrowers)**:
  - **100% of the 93 rules generalize out-of-sample**, maintaining a test lift $\ge 1.30\times$.
  - *Top Validated Rule*:  
    $$\text{DECLINING\_INCOME} \land \text{INCOME\_VOLATILE} \land \text{LOW\_SAVINGS\_BUFFER} \implies \text{DEFAULT}$$  
    $\text{Support} = 1.48\%$, $\text{Confidence} = 24.7\%$, $\text{Lift} = 3.14\times$ (versus 7.87% baseline).
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
| **5** | **Early Warning Trigger** | Track 30+ DPD roll rates weekly; breach of 4.5% triggers automated loss reserving. | High correlation ($R^2 = 0.76$) with 60-day forward defaults. |
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

1. **Synthetic Borrower Signals**: Applicant profiles and transaction streams are programmatically generated via correlated statistical links, whereas macroeconomic drivers are real. While validating model pipelines, real-world credit deployment requires calibration on actual banking data.
2. **Temporal Interpolation**: World Bank series (unemployment, GDP growth) are published annually and interpolated to monthly frequency.
3. **Fairness & Inclusion**: The model actively avoids penalizing applicants based on protected demographic attributes (gender, location tier), focusing purely on behavioral cash-flow consistency.
