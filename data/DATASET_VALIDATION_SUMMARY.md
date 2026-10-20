# IntelliCredit: Dataset Description and Validation Summary
**Academic Capstone — Review 2 (Unit 3: Advanced Analytics)**  
*Project Completion Deliverable: Dataset Architecture, Quality Assurance, and Cross-Unit Validation*

---

## 1. Executive Summary & Problem Context

Traditional credit evaluation relies heavily on bureau scores (CIBIL, FICO) and formal salaried payslips. In India and emerging markets, over **1.4 billion gig-economy workers, delivery partners, and thin-file individuals** are excluded from formal credit or subjected to predatory rates because they lack traditional credit documentation.

**IntelliCredit** constructs an alternative credit intelligence engine using:
1. High-frequency digital cash flow and behavioral transaction signals.
2. Direct macroeconomic data ingested via live institutional APIs (FRED & World Bank).
3. A survival-linked dynamic loan portfolio simulation spanning 6 calendar years (2019–2024).

This document serves as the formal **Dataset Description and Validation Summary** required by the Review 2 rubric.

---

## 2. Dataset Architecture & Record Inventory

| Dataset File | Granularity | Records (Rows) | Variables (Cols) | Collection / Generation Source | Primary Unit Supported |
|---|---|:---:|:---:|---|---|
| `data/borrower_profiles.csv` | Cross-sectional applicant level | **30,000** | **39** | Programmatic statistical synthesis (Faker `en_IN` + correlated distributions) | **Unit 1 & Unit 2** |
| `data/monthly_transactions.csv` | Longitudinal monthly transactions | **360,000** | **15** | Synthetic 12-month transaction stream per borrower (income, utilities, recharge, BNPL, UPI) | **Unit 1 & Unit 2** |
| `data/external/macro_india_monthly.csv` | Monthly macroeconomic time series | **108** | **18** | **Live institutional APIs** (Federal Reserve FRED & World Bank API; 2017–2025) | **Unit 3 (Method 1)** |
| `data/portfolio/loan_book.csv` | Originated loan contracts | **30,000** | **12** | Dynamic hazard-linked portfolio simulation (2019–2024) | **Unit 3 (Methods 1 & 2)** |
| `data/portfolio/loan_performance.csv` | Monthly repayment tracking | **250,191** | **7** | Loan-month DPD (Days Past Due) and amortization panel | **Unit 3 (Method 1)** |
| `data/portfolio/portfolio_monthly.csv` | Aggregate monthly portfolio metrics | **72** | **21** | Portfolio roll rates, active book size, default rates, and macro overlays | **Unit 3 (Method 1)** |
| `data/processed/train.csv` | Partitioned train applicants | **24,000** | **18** | Stratified 80% train split (zero leakage) | **Units 2 & 3 (ARM mining)** |
| `data/processed/test.csv` | Partitioned held-out test applicants | **6,000** | **18** | Stratified 20% held-out test split | **Units 2 & 3 (ARM validation)** |
| `data/sample_final_dataset.csv` | Evaluated final applicant sample | **100** | **44** | Representative sample containing raw features + ML PD + ARM flags + Decisions | **Review 2 Panel Inspection** |

---

## 3. Data Collection, Provenance & Generation Methodology

### 3.1 Synthetic Borrower & Transaction Pipeline (`data_generation.py`)
To strictly adhere to academic ethics and institutional data privacy without relying on scraped Kaggle/UCI datasets:
- **Demographics**: Synthesized via `Faker(locale='en_IN')` with localized names, realistic Indian cities (Metro, Tier-1, Tier-2, Semi-Urban, Rural), educational levels, and employment platform categories (Delivery, Ride-Hailing, Freelance, Micro-Retail).
- **Interdependent Financial Behavior**: Features are not isolated random draws; they are linked via realistic conditional distributions:
  - Monthly earnings follow a log-normal distribution conditioned on gig tenure and skill tier.
  - Utility bill delays and BNPL borrowing intensify when income volatility spikes.
  - Digital footprint (UPI transactions, app logins) inversely correlates with cash delinquency.
- **Ground Truth Default Label**: Generated via a calibrated logistic link based on underlying financial fragility:
  $$\text{Log-Odds} = \beta_0 + \beta_1 (\text{Income Volatility}) + \beta_2 (\text{Utility Delay}) + \beta_3 (\text{Debt-to-Income}) - \beta_4 (\text{Savings Buffer}) - \beta_5 (\text{Gig Tenure})$$
  Observed overall default rate: **7.87%** (2,361 defaults / 30,000 borrowers), perfectly matching industry benchmarks for unsecured micro-lending.

### 3.2 Live Macroeconomic API Pipeline (`src/macro_api.py`)
Macroeconomic conditions directly govern credit default cycles. Live monthly macroeconomic data was programmatically harvested from verified public APIs:
- **FRED (Federal Reserve Bank of St. Louis)**:
  - CPI Inflation YoY (`CPALTT01INM659N`)
  - Interbank Call-Money Rate (`IRSTCI01INM156N`)
  - 10-Year Government Securities Yield (`INDIRLTLT01STM`)
  - INR/USD Exchange Rate (`EXINUS`)
- **World Bank Development Indicators**:
  - Unemployment Rate (`SL.UEM.TOTL.ZS`)
  - Real GDP Growth (`NY.GDP.MKTP.KD.ZG`)
  - Commercial Lending Rate (`FR.INR.LEND`)
- **Resilience & Provenance Architecture**:
  - Responses cached in `data/external/raw/`.
  - Full source attribution and timestamps logged in `data/external/macro_sources.json`.
  - Fallback mechanism ensures offline reproducibility if external APIs undergo maintenance.

### 3.3 Dynamic Loan Book & Performance Simulation (`src/portfolio_simulation.py`)
- Simulates 30,000 borrowers originating loans across staggered origination months (2019-01 to 2024-12).
- Tracks contractual monthly EMI amortization, roll rates across delinquency stages ($DPD = 0 \to 30 \to 60 \to 90+$), and terminal default states.
- Default hazards are dynamically modulated by real Indian macroeconomic shocks (e.g., COVID-19 nationwide lockdowns in Q2 2020 and the 2021 second wave).

---

## 4. Quality Checks, Audits & Corrections Applied

During the development cycle, rigorous data quality checks and audits were conducted to ensure data integrity:

| Inspection Area | Anomaly Detected | Root Cause | Engineering Correction Applied |
|---|---|---|---|
| **Date Alignment** | Month label collisions (e.g. repeated "2025-01" rows) | Using `timedelta(30 * i)` caused monthly drift over 360 days | Switched to canonical `pd.date_range(..., freq="MS")` ensuring exact monthly timestamps. |
| **Categorical Encoding** | Unknown category mapping (-1 outputs) | Discrepancy between raw category string names and dictionary keys | Created centralized `src/config.py:ORDINAL_MAPS` with programmatic assertions verifying 100% dictionary coverage. |
| **Leakage Audit (Unit 1/2)** | Target leakage in engineered features | Target encoding of platform risk was computed over the entire dataset before splitting | Enforced strict train/test partitioning prior to any feature engineering; target encodings fitted strictly on `train.csv`. |
| **Leakage Audit (Unit 2)** | Pre-SMOTE cross-validation overfitting | SMOTE oversampling was applied to the full training set prior to K-fold partitioning | Moved SMOTE inside an `imblearn.pipeline.Pipeline` so synthetic generation occurs strictly within training folds during cross-validation. |
| **Missing Values Audit** | 28,514 null values in `data/portfolio/loan_book.csv` | Non-defaulted loans have no default month | **Verified as valid structural nulls**: Only loans reaching 90+ DPD receive a `default_month` ($30,000 - 28,514 = 1,486$ defaulted loans). |
| **Time Series Differencing** | 12 null values in `inr_depreciation_yoy` | 12-month percentage lag calculation on month 1..12 | **Verified as valid mathematical lag truncation**: 12-month YoY change cannot exist for the initial year (2017). |

---

## 5. How the Validated Dataset Supports Units 1, 2, and 3

```
                                  =========================================
                                  INTELLICREDIT VALIDATED DATA FOUNDATION
                                  =========================================
                                                     |
             +---------------------------------------+---------------------------------------+
             |                                       |                                       |
             v                                       v                                       v
      [ UNIT 1: FOUNDATION ]                 [ UNIT 2: MODELLING ]                 [ UNIT 3: ADVANCED ]
  * 30k Demographics & Cash Flows         * Leak-free Train/Test Split (80/20)    * Method 1: Time Series (SARIMAX)
  * 360k Transaction Audit Logs           * 14 Engineered Risk Ratios             * Method 2: Delinquency ARM Patterns
  * Missing Value Imputation (KNN/Mode)   * SMOTE Balancing + Cost-Optimal Cut    * Combined Business Insights
  * Outlier Winsorization & Robust Scaler * Tuned XGBoost / LightGBM Ensembles    * Mandatory Interactive Dashboard
```

### 5.1 Support for Unit 1 (Data Preprocessing & Exploratory Analysis)
- `borrower_profiles.csv` and `monthly_transactions.csv` provided the basis for 38 comprehensive exploratory charts, normality evaluations, bivariate hypothesis testing (Mann-Whitney U tests for continuous features, Chi-Square contingency tests for categorical attributes), and outlier winsorization.

### 5.2 Support for Unit 2 (Feature Engineering & Predictive Modelling)
- The structured 30,000-record dataset enabled the engineering of critical non-traditional risk metrics:
  - `income_volatility_index` (Coefficient of Variation)
  - `utility_reliability_score` (On-time ratio)
  - `savings_buffer_ratio` (Liquidity runway)
  - `income_trend_slope` (OLS slope of earnings trajectory)
- The stratified 80/20 split (`train.csv` / `test.csv`) ensured clean, leak-free training for 5 candidate algorithms (LR, RF, XGBoost, LightGBM, Stacking Classifier), achieving an optimal test ROC-AUC of **0.738** and PR-AUC of **0.204** against a theoretical ceiling of 0.781.

### 5.3 Support for Unit 3 Method 1 (Time-Series Analysis)
- `portfolio_monthly.csv` (72 months) combined with `macro_india_monthly.csv` (108 months) provides the validated time series foundation.
- Stationarity tests (ADF and KPSS), STL decomposition, Granger causality tests, and rolling-origin walk-forward evaluation (40 origins) proved that SARIMAX with macroeconomic exogenous drivers outperforms classical univariate SARIMA and ETS, achieving a hold-out test MAE of **0.054 percentage points** (16% MAPE).

### 5.4 Support for Unit 3 Method 2 (Association Rule Mining)
- Discretized 19 behavioral flags across 24,000 training borrowers into binary itemsets.
- Apriori and FP-Growth mined 2,852 frequent itemsets. Statistical pruning (Bonferroni-adjusted Fisher's p < 0.05) and subset redundancy filtering yielded **93 highly predictive warning rules**.
- Evaluated on the 6,000 held-out test borrowers, **100% of the 93 rules retained test lift $\ge 1.3\times$**, verifying that the patterns generalize to unseen applicants.

### 5.5 Support for Unit 3 Interactive Application
- The complete evaluated dataset (`data/sample_final_dataset.csv`, `portfolio_monthly.csv`, and `macro_sources.json`) powers the interactive Streamlit underwriter platform (`dashboard/app.py`), enabling live underwriting decisions, multi-filter portfolio drilldowns, and macroeconomic scenario stress testing.

---

## 6. Verification Checklist

- [x] **Data Source Documentation**: Generation parameters and live institutional API endpoints fully documented.
- [x] **Sample Dataset Available**: `data/sample_final_dataset.csv` and `data/sample/` directory created for quick panel review.
- [x] **Data Dictionary**: Comprehensive feature definitions in `data/data_dictionary.txt` and `data/data_dictionary_review2.md`.
- [x] **Audit Trail**: Full data transformation scripts and reproducibility seeds (`42` for data/ML, `2026` for portfolio simulation) locked.
- [x] **Validation Across All Units**: Explicit mapping showing how raw signals translate into predictive models, time series forecasts, and association rules.
