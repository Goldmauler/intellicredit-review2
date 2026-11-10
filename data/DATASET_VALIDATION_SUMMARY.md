# IntelliCredit: Dataset Description and Validation Summary
**Academic Capstone — Review 2 (Unit 3: Advanced Analytics)**  
*Project Completion Deliverable: Dataset Architecture, Quality Assurance, and Cross-Unit Validation*

---

## 1. Executive Summary & Problem Context

Traditional credit evaluation relies heavily on bureau scores (CIBIL, FICO) and formal salaried payslips. Worldwide **1.4 billion adults are unbanked** (World Bank Findex 2021), and India's **7.7–12 million gig and platform workers** (NITI Aayog 2022; Economic Survey 2025-26) mostly have thin credit files; they are excluded from formal credit or subjected to predatory rates because they lack traditional credit documentation.

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
  - Each borrower's base income is drawn from Uniform(₹8,000, ₹55,000) and scaled by location and education multipliers; monthly earnings then vary around that base with Normal noise whose size is larger for newer gig workers. The result is not log-normal (median ₹31,532, skewness 0.67).
  - Utility bill delays and BNPL borrowing intensify when income volatility spikes.
  - Digital footprint (UPI transactions, app logins) inversely correlates with cash delinquency.
- **Ground Truth Default Label**: Each borrower's default is a coin flip whose probability comes from a logistic link of ten behavioural signals plus Normal(0, 0.5) noise: income volatility (+3.0), utility on-time share (−2.5), BNPL debt-to-inflow (+2.0), savings buffer (−1.5), gig tenure (−0.02 per month), recharge gap above 35 days (+0.5), app logins (−0.03), income trend below −500 (+0.8), loan amount (+0.000005 per rupee) and prior repayment ratio (−0.5), with intercept −1.2 (`data_generation.py`, step 4).
  Observed overall default rate: **7.87%** (2,360 defaults / 30,000 borrowers). This is higher than microfinance portfolio-at-risk benchmarks by design, because the project targets the thin-file segment those portfolios exclude (Section 7).

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
- Simulates 30,000 borrowers originating loans across staggered origination months (2018-07 to 2024-12; 1,249 loans originate in 2018 H2, and the monthly portfolio panel is tracked from 2019-01).
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
- The stratified 80/20 split (`train.csv` / `test.csv`) ensured clean, leak-free training for 5 candidate algorithms (LR, RF, XGBoost, LightGBM, Stacking Classifier), On the 6,000-borrower test set the Review 1 models reach ROC-AUC **0.738** (tuned XGBoost, PR-AUC **0.204**) and **0.739** (tuned stacking ensemble), against a theoretical ceiling of 0.781. The calibrated LightGBM used in the dashboard reaches ROC-AUC **0.741**, PR-AUC **0.199** and Brier score 0.068 (recomputed 2026-11-09 from the saved model).

### 5.3 Support for Unit 3 Method 1 (Time-Series Analysis)
- `portfolio_monthly.csv` (72 months) combined with `macro_india_monthly.csv` (108 months) provides the validated time series foundation.
- Stationarity tests (ADF and KPSS), STL decomposition, Granger causality tests, and rolling-origin walk-forward evaluation (40 origins) proved that SARIMAX with macroeconomic exogenous drivers outperforms classical univariate SARIMA and ETS, achieving a hold-out test MAE of **0.054 percentage points** (16% MAPE).

### 5.4 Support for Unit 3 Method 2 (Association Rule Mining)
- Discretized 19 behavioral flags across 24,000 training borrowers into binary itemsets.
- Apriori and FP-Growth mined 2,852 frequent itemsets. Statistical pruning (Bonferroni-adjusted Fisher's p < 0.05) and subset redundancy filtering yielded **93 highly predictive warning rules**.
- Evaluated on the 6,000 held-out test borrowers, **100% of the 93 rules retained test lift $\ge 1.3\times$**, verifying that the patterns generalize to unseen applicants (minimum test lift 1.35; median lift fell from 2.03 on train to 1.89 on test, the expected selection shrinkage).

### 5.5 Support for Unit 3 Interactive Application
- The complete evaluated dataset (`data/sample_final_dataset.csv`, `portfolio_monthly.csv`, and `macro_sources.json`) powers the interactive Streamlit underwriter platform (`dashboard/app.py`), enabling live underwriting decisions, multi-filter portfolio drilldowns, and macroeconomic scenario stress testing.

---

## 6. Verification Checklist

- [x] **Data Source Documentation**: Generation parameters and live institutional API endpoints fully documented.
- [x] **Sample Dataset Available**: `data/sample_final_dataset.csv` and `data/sample/` directory created for quick panel review.
- [x] **Data Dictionary**: Comprehensive feature definitions in `data/data_dictionary.txt` and `data/data_dictionary_review2.md`.
- [x] **Audit Trail**: Full data transformation scripts and reproducibility seeds (`42` for data/ML, `2026` for portfolio simulation) locked.
- [x] **Validation Across All Units**: Explicit mapping showing how raw signals translate into predictive models, time series forecasts, and association rules.
- [x] **Real-Data Validation**: five real credit datasets collected from IMF, World Bank, FRED and CFPB APIs; both Unit 3 methods re-run on them (notebook 09); synthetic statistics compared with verified MFIN / RBI / Economic Survey / Findex figures only, and the places where they do not match are stated (Section 7).

---

## 7. Real-World Benchmark Validation (Faculty Feedback Response)

> **Faculty Feedback:** "If you have any real-life data to validate the synthetic generated data, that would be better."

### 7.1 What is Already Real in This Project

The project already integrates **two categories of real-world data**:

| Layer | Data | Source | Real? |
|---|---|---|:---:|
| Macroeconomic indicators (7 series, 108 months) | CPI, Call-money rate, G-Sec yield, INR/USD, Unemployment, GDP growth, Lending rate | FRED & World Bank live APIs | ✅ **Real** |
| COVID lockdown event flags | Q2 2020 and Q2 2021 disruption months | Public RBI/GOI records | ✅ **Real** |
| Borrower profiles & transactions (30k / 360k records) | Demographics, income, utility, UPI, BNPL | Synthetic (Faker en_IN + correlated distributions) | Synthetic |
| Loan portfolio performance (250k loan-months) | DPD migration, defaults | Macro-linked simulation | Simulated |

### 7.2 Synthetic Data Compared with Real-World Anchors (recomputed 2026-11-09)

Every synthetic value below was recomputed from the files in `data/`. Real-world anchors are only those that were verified in the source (see `research/BENCHMARK_TABLE.md`). Where no verified anchor exists the row says so instead of claiming a match.

| Metric | Synthetic value (recomputed) | Verified real-world anchor | Source | Assessment |
|---|---|---|---|:---:|
| **Default rate** | Borrower label (90+ DPD) **7.87%** (2,360 of 30,000). In the simulated loan book, **4.95%** of loans (1,486) reach 90+ DPD inside the window | MFIN Micrometer PAR 31–180: 2.2% (Mar 2024), 6.3% (Mar 2025), 1.6% (Jun 2026) | MFIN Micrometer | Higher by design: thin-file segment outside MFI portfolios. The 2025 MFI stress shows real delinquency can triple within a year |
| **COVID default spike** | Calendar 2020 is **2.3×** the 2019 level; the peak month (Jun 2020) is 4.9% against a 0.82% average in 2019 (**6×**) | Real bank NPLs *fell*: SCB GNPA 8.2% (Mar 2020) to 7.3% (Mar 2021) because of the moratorium and write-offs | RBI Trend & Progress 2020-21 | **Does not match.** The spike is an explicit modelling assumption, not a benchmark; notebook 09 shows the real-data consequence |
| **Gig-worker income** | Median ₹31,532; 5th–95th percentile ₹9,942–₹68,833; only **15.2%** earn under ₹15,000 | About 40% of gig workers earn under ₹15,000 a month | Economic Survey 2025-26 | **Optimistic.** The synthetic data has far fewer low earners than reported |
| **Loan size requested** | Median ₹107,848; 5th–95th percentile ₹29,128–₹299,801 (generated as 1.5–6× monthly income, capped at ₹300,000) | No verified anchor | — | **Not validated.** Larger than typical micro-loans; the earlier "₹40,000–₹80,000" figure did not match the data |
| **Income volatility (CoV of monthly income)** | Mean 0.28; 5th–95th percentile 0.15–0.42 | No verified anchor | — | **Not validated.** The earlier "30%–55% consistent with field evidence" claim could not be verified and does not match the data |
| **Utility bill punctuality** | Mean 8.3 of 12 bills on time; 96% of borrowers have some late bills; 31% average at least 5 days late | No verified anchor | — | **Not validated.** The earlier "~18% with delays" figure was wrong |
| **UPI activity** | Every borrower has UPI activity (median 13.6 transactions a month) | No verified anchor | — | Holds by construction, so it is not evidence of realism |
| **Thin-file population** | 30,000 synthetic borrowers | 1.4 billion unbanked adults worldwide | World Bank Global Findex 2021 | Context for the problem statement, not a validation of the data |

### 7.3 Verified Sources Used

| Source | Used for |
|---|---|
| **MFIN Micrometer** (quarterly MFI portfolio statistics) | Portfolio-at-risk anchor for the default rate |
| **RBI Report on Trend & Progress of Banking 2020-21** | Real bank NPL behaviour during COVID |
| **Economic Survey 2025-26** | Gig-worker income anchor |
| **World Bank Global Findex 2021** | Financial-inclusion context |

An earlier draft also cited a CGAP gig-economy financial diary, PLFS income bands, a CEEW utility survey and NPCI UPI penetration. None of these could be verified, so they were removed (the same decision is recorded in `reports/Review_2_Final_Report.md` Section 1.3).

### 7.4 Conclusion

The synthetic borrower data is **not** claimed to be statistically consistent with real Indian microfinance data. Two verified anchors disagree with it (gig income is more favourable than reported, and the COVID default spike has no real counterpart), and for loan size, income volatility and bill punctuality no verified anchor exists. The 7.87% default rate is higher than MFI benchmarks on purpose, because the project models the thin-file segment those portfolios exclude.

What the project does claim is narrower and checkable: the synthetic data proves the pipeline works end to end against a known ceiling (AUC 0.781), the macroeconomic series are real (FRED and World Bank), and every mechanism the synthetic data relies on was re-tested on five real datasets in notebook 09 (69,662 records). Two of the four checks held and two did not (Table 8 of the final report).

---

## 8. Validation Audit Log (2026-11-09)

All claims in this document were recomputed from the saved files, models and notebooks. Results:

| Claim | Result |
|---|---|
| Record and column counts of all nine datasets (Section 2) | Correct |
| Default rate 7.87% | Correct; the count is 2,360 (corrected from 2,361) |
| Loan-book structural nulls (28,514) and defaulted loans (1,486) | Correct |
| Calibrated LightGBM: test AUC 0.7406, PR-AUC 0.1989, Brier 0.0681, mean PD 7.94% vs 7.87% observed | Correct, reproduced exactly |
| Break-even policy: 60.3% approved, 5.09% approved default rate, 12.09% declined default rate | Correct, reproduced exactly |
| 93 rules: every stored lift, coverage and test lift | Reproduced exactly; all 93 keep test lift of at least 1.3 (minimum 1.345) |
| Time-series hold-out MAE 0.0545 pp, MAPE 16.4%; scenario losses 158.9 / 194.6 / 432.5 lakh | Correct (read from saved tables) |
| Real-data track: 69,662 records, India NPL MAE 0.48 vs 2.04 naive, ACI coverage 80.9% vs Gaussian 72.3%, 82 HMDA rules with 95.1% holding out-of-state | Reproduced exactly |
| Section 7.2 benchmark table (previous version) | Several rows were wrong and have been replaced above |
| Income "log-normal", five-term default formula, origination from 2019-01, "optimal 0.738" | Corrected in Sections 3 and 5 |
| Rules add information inside PD bands | True in bands A, C and D (for example band A: 1.8% with no rules, 5.6% with six or more); **not** in band B (7.0% with up to two rules, 4.7% with six or more) |

---

## 9. Review 3 Validation Suite (2026-11-10)

Reproducible evidence, recomputed from files by `scripts/run_validation_suite.py` and `scripts/build_fidelity_anchors.py` (outputs in `models/validation/`, shown in the dashboard tab **Proof & comparison**). The four public real datasets (German credit, Taiwan cards, Give Me Some Credit, FICO HELOC) are downloaded on demand from OpenML into `data/real/benchmarks/` (git-ignored; licences differ).

| Test | Result |
|---|---|
| Same four model types, 5-fold CV, on real data (calibrated-LightGBM recipe) | AUC German 0.793, Taiwan 0.786, Give Me Some Credit 0.866, HELOC 0.794; IntelliCredit (simulated) 0.748 |
| Model-ranking agreement, simulated vs real | Spearman +1.00 (German, HELOC), +0.80 (GMSC), +0.40 (Taiwan); boosted trees best or tied everywhere |
| Risk-driver direction, simulated vs real (9 testable drivers) | 5 same direction (late payments, on-time share, loan size, track record, savings); 3 negligible (age, dependents, debt burden); 1 opposite (income level) |
| Published Indian anchors | Default rate, interest rate, recharge cycle roughly match; gig income (15.2% under Rs 15k vs about 40%), loan size (2.2x MFIN ticket) and the COVID spike do not |
| Simulated monthly default series vs real NPL series | Far jumpier (typical jump 43% vs 4-9%); about two thirds of the variance is sampling noise from a book of about 3,475 loans |
| Income re-weighting (40% earning under Rs 15k) | AUC 0.747, predictions stay calibrated; default rate falls (7.9% to 7.0%), which is the income/loan-size coupling flaw again |
| Alternative data (simulated) | Form + past loans AUC 0.701, all features 0.748 (+4.8 points; Berg et al. +5.3 on real data); +13% approvals at a 5% default rate versus a conventional scorecard (built in by construction) |

**Known generator gaps this exposed:** default risk is charged per rupee borrowed and loan size scales with income, so higher income looks riskier (real data says the opposite); debt burden, age and dependents carry almost no effect. The competitor and literature comparison used by the dashboard is `research/competitive_landscape.json` (each entry labelled checked-at-source, seen-in-summary or company claim).
