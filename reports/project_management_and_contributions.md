# IntelliCredit: Project Management, Task History & Individual Contributions
**Capstone Project in Business Analytics — Review 2 (Unit 3: Advanced Analytics)**  
*Official Submission Deliverable: Project Board Link, Sprint Task History & Individual Contribution Matrix*

---

## 1. Project Governance & Agile Kanban Board

- **Repository**: [https://github.com/Darun1017/credit-scoring-engine](https://github.com/Darun1017/credit-scoring-engine)  
- **Review 2 Repository**: [https://github.com/Goldmauler/intellicredit-review2](https://github.com/Goldmauler/intellicredit-review2)  
- **Live GitHub Project Board**: `@Darun1017's credit-scoring-engine` ([View Issues & Tasks](https://github.com/Darun1017/credit-scoring-engine/issues?q=is%3Aissue))  
- **Project Tracking Tool**: GitHub Projects (Table & Kanban View)  
- **Structure**: 14 core project tasks tracked across Review 1 (Tasks #3–#12) and Review 2 (Tasks #13–#16).

---

## 2. 14-Task Project Inventory & Sprint History

All 14 tasks are mapped directly to the GitHub Project board with assignees, status, and supporting deliverables:

| Row | Task ID & Title | GitHub Issue | Assigned Team Member(s) | Status | Deliverable & Supporting Evidence |
|:---:|---|:---:|---|:---:|---|
| **1** | `[R1-M0-01] Rubric and project plan reading` | [#3](https://github.com/Darun1017/credit-scoring-engine/issues/3) | `@adii11001`, `@Darun11`, `@abhinav-juturu`, `@Goldmauler`, `@gowthamreddy285` | **Done** | Review 1 & 2 rubric analysis, scoping, sprint planning |
| **2** | `[R1-M0-02] Repository setup, folder structure, README and config` | [#4](https://github.com/Darun1017/credit-scoring-engine/issues/4) | `@Darun11` | **Done** | Git repository architecture, `src/config.py`, requirements, folder layout |
| **3** | `[R1-M0-03] Synthetic data generation (data_generation.py) and data dictionary` | [#5](https://github.com/Darun1017/credit-scoring-engine/issues/5) | `@Darun11` | **Done** | `data_generation.py` (30k borrowers, 360k txns), `data/data_dictionary.txt` |
| **4** | `[R1-M1-01] Data cleaning and preprocessing notebook` | [#6](https://github.com/Darun1017/credit-scoring-engine/issues/6) | `@gowthamreddy285` | **Done** | `notebooks/01_M1_data_cleaning_preprocessing.ipynb` (KNN imputation, winsorization) |
| **5** | `[R1-M2-01] Exploratory data analysis and visualization notebook` | [#7](https://github.com/Darun1017/credit-scoring-engine/issues/7) | `@abhinav-juturu` | **Done** | `notebooks/02_M2_exploratory_data_analysis.ipynb` (38 charts, Mann-Whitney & Chi-Square tests) |
| **6** | `[R1-M3-01] Feature engineering, selection and SMOTE notebook` | [#8](https://github.com/Darun1017/credit-scoring-engine/issues/8) | `@Darun11` | **Done** | `notebooks/03_M3_feature_engineering.ipynb` (14 risk ratios, VIF, WoE/IV, train/test split) |
| **7** | `[R1-M4-01] Predictive modelling, tuning and ensemble notebook` | [#9](https://github.com/Darun1017/credit-scoring-engine/issues/9) | `@Goldmauler` | **Done** | `notebooks/04_M4_predictive_modeling.ipynb` (5 algorithms, leak-free CV + SMOTE pipeline) |
| **8** | `[R1-M5-01] Evaluation, SHAP explainability and risk cut-off notebook` | [#10](https://github.com/Darun1017/credit-scoring-engine/issues/10) | `@adii11001` | **Done** | `notebooks/05_M5_evaluation_interpretation.ipynb` (ROC 0.738, TreeSHAP, INR cost cut-off) |
| **9** | `[R1-MX-01] Pipeline audit and correctness fixes (leakage, SMOTE, paths)` | [#11](https://github.com/Darun1017/credit-scoring-engine/issues/11) | `@adii11001`, `@Darun11`, `@abhinav-juturu`, `@Goldmauler`, `@gowthamreddy285` | **Done** | Leakage elimination, ordinal encoding alignment, reproducible random seeds |
| **10** | `[R1-MX-02] Review 1 presentation and individual contribution summary` | [#12](https://github.com/Darun1017/credit-scoring-engine/issues/12) | `@adii11001`, `@Darun11`, `@abhinav-juturu`, `@Goldmauler`, `@gowthamreddy285` | **Done** | Review 1 presentation slides and milestone defense |
| **11** | `[R2-MX-01] Time-series data preparation (monthly default trends, macro reshape)` | [#13](https://github.com/Darun1017/credit-scoring-engine/issues/13) | `@abhinav-juturu`, `@Goldmauler` | **Done** | `src/macro_api.py` (FRED/World Bank live APIs), `src/portfolio_simulation.py` (250k panel) |
| **12** | `[R2-MX-02] Time-series forecasting and evaluation (ARIMA/SARIMA, plots, MAE/RMSE)` | [#14](https://github.com/Darun1017/credit-scoring-engine/issues/14) | `@Goldmauler` | **Done** | `notebooks/06_R2_M1_time_series_forecasting.ipynb` (SARIMAX, 7 models, 40-origin backtest) |
| **13** | `[R2-MX-01] Association rule mining data preparation (discretize borrower features)` | [#15](https://github.com/Darun1017/credit-scoring-engine/issues/15) | `@Darun11`, `@Goldmauler` | **Done** | 19 behavioral flags discretized on 24k training borrowers (`notebooks/07_R2_M2`) |
| **14** | `[R2-MX-02] Association rule mining implementation and interpretation (Apriori, support/confidence/lift)` | [#16](https://github.com/Darun1017/credit-scoring-engine/issues/16) | `@Goldmauler` | **Done** | `notebooks/07_R2_M2`, 93 rules, Fisher test pruning, 100% test-lift retention on 6k unseen |

---

## 3. Individual Contribution Summary across the 14 Tasks

| Team Member | GitHub Username | Assigned Core Tasks | Primary Deliverable Scope |
|---|---|:---:|---|
| **Darun R G** | `@Darun11` | Tasks #2, #3, #6, #13 (+ #1, #9, #10) | Data architecture, synthetic data generator, feature engineering, ARM data prep, dataset validation summary |
| **goldmauler** | `@Goldmauler` | Tasks #7, #11, #12, #13, #14 (+ #1, #9, #10) | Lead — Predictive modeling tuning, Method 1 SARIMAX forecasting, Method 2 Association Rule Mining |
| **Abhinav Juturu** | `@abhinav-juturu` | Tasks #5, #11 (+ #1, #9, #10) | Exploratory data analysis, statistical hypothesis testing, time-series data preparation & portfolio panel |
| **Gowtham Reddy** | `@gowthamreddy285` | Tasks #4 (+ #1, #9, #10) | Data cleaning, preprocessing pipelines, Streamlit interactive dashboard implementation |
| **Adithya** | `@adii11001` | Tasks #8 (+ #1, #9, #10) | Model evaluation & TreeSHAP explainability, combined insights synthesis, technical reporting & presentation |

---

## 4. Verification Evidence

1. **GitHub Issues & History**: All 14 tasks exist as individual issues in [`Darun1017/credit-scoring-engine`](https://github.com/Darun1017/credit-scoring-engine/issues?q=is%3Aissue).
2. **Board Table View**: Visible directly on `@Darun1017's credit-scoring-engine` GitHub Projects view.
3. **Commit History**: Verifiable via Git commit logs across Review 1 (`Darun1017/credit-scoring-engine`) and Review 2 (`Goldmauler/intellicredit-review2`).
