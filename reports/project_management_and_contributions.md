# IntelliCredit: Project Management, Task History & Individual Contributions
**Capstone Project in Business Analytics — Review 2 (Unit 3: Advanced Analytics)**  
*Official Submission Deliverable: Project Board Link, Sprint Task History & Individual Contribution Matrix*

---

## 1. Project Governance & Agile Kanban Board

- **Repository & Project Board**: [https://github.com/Goldmauler/intellicredit-review2](https://github.com/Goldmauler/intellicredit-review2)  
- **Review 1 Repository**: [https://github.com/Darun1017/credit-scoring-engine](https://github.com/Darun1017/credit-scoring-engine)  
- **Project Tracking Tool**: GitHub Projects / Agile Kanban Board  
- **Methodology**: 4 two-week Sprints covering data generation, predictive modeling, advanced analytics (time series & ARM), and full-stack interactive deployment.

---

## 2. Sprint & Task History

### Sprint 1: Data Architecture, Cleaning & Baseline EDA (Review 1 Foundation)
- **Duration**: Weeks 1 – 2
- **Key Tasks Delivered**:
  - `TSK-101`: Implemented `data_generation.py` generating 30,000 synthetic borrowers and 360,000 longitudinal monthly transactions using Faker (`en_IN`).
  - `TSK-102`: Built data cleaning pipeline in `01_M1_data_cleaning_preprocessing.ipynb` with KNN/mode imputation, winsorization, and categorical mapping.
  - `TSK-103`: Generated 38 exploratory charts in `02_M2_exploratory_data_analysis.ipynb` with Mann-Whitney U and Chi-Square statistical hypothesis testing.
  - `TSK-104`: Authored comprehensive Data Dictionary (`data/data_dictionary.txt`).

### Sprint 2: Feature Engineering & Predictive Default Models (Review 1 Delivery)
- **Duration**: Weeks 3 – 4
- **Key Tasks Delivered**:
  - `TSK-201`: Engineered 14 behavioral features (income volatility, utility reliability, savings buffer, trend slope) in `03_M3_feature_engineering.ipynb`.
  - `TSK-202`: Filtered multicollinearity using VIF (< 5) and Information Value (IV); implemented leak-free train/test split (80/20).
  - `TSK-203`: Trained and tuned 5 classification algorithms in `04_M4_predictive_modeling.ipynb` using leak-free Stratified K-Fold CV with embedded SMOTE.
  - `TSK-204`: Evaluated models in `05_M5_evaluation_interpretation.ipynb` (ROC-AUC 0.738 on XGBoost), KS statistics (0.36), and calculated INR cost-optimal approval thresholds.

### Sprint 3: Advanced Analytics — Time Series & Association Rules (Review 2 Focus)
- **Duration**: Weeks 5 – 6
- **Key Tasks Delivered**:
  - `TSK-301`: Created `src/macro_api.py` connecting live to FRED and World Bank APIs (108 monthly observations, caching, fallback mechanism).
  - `TSK-302`: Developed `src/portfolio_simulation.py` simulating 30,000 loans, 250k loan-months, and 72 monthly default rates.
  - `TSK-303`: Implemented Time-Series Forecasting in `06_R2_M1_time_series_forecasting.ipynb` (ADF/KPSS tests, Granger causality, 7 models benchmarked, SARIMAX selected with holdout MAE 0.054 pp).
  - `TSK-304`: Implemented Association Rule Mining in `07_R2_M2_association_rule_mining.ipynb` (Apriori vs FP-Growth benchmark, Bonferroni-corrected Fisher's exact test, redundancy pruning yielding 93 validated rules).
  - `TSK-305`: Validated rules on 6,000 unseen test borrowers (100% test lift $\ge 1.3\times$).

### Sprint 4: Combined Insights, Full Interactive Dashboard & Presentation Prep
- **Duration**: Weeks 7 – 8
- **Key Tasks Delivered**:
  - `TSK-401`: Synthesized cross-method findings in `08_R2_combined_insights_dashboard.ipynb` (rule count vs. PD cross-tabulation, macro-conditioned rules, PiT multipliers).
  - `TSK-402`: Developed production Streamlit interactive application `dashboard/app.py` with multi-filter dataset browser, live applicant scoring, TreeSHAP waterfall, and macro scenario toggles.
  - `TSK-403`: Compiled standalone HTML report `reports/interactive_dashboard.html`.
  - `TSK-404`: Conducted data quality audit, resolved encoding/leakage fixes, generated `data/sample_final_dataset.csv` and `data/DATASET_VALIDATION_SUMMARY.md`.
  - `TSK-405`: Created Final Technical Report (`reports/Review_2_Final_Report.md`) and 15-Minute Panel Presentation Slides (`reports/Review_2_Presentation_Slides.md`).

---

## 3. Individual Contribution Matrix (Review 2 Rubric: 20 Marks Total)

| Team Member | Core Assigned Role | Key Notebooks & Code Deliverables | Primary Rubric Component Supported | Marks Weight |
|---|---|---|---|:---:|
| **Darun R G** (@Darun1017) | **Data Architect & Lead Predictive Modeler** | `data_generation.py`, `01_M1` to `05_M5`, `scripts/generators/` | Overall Dataset Description & Validation, Baseline ML Pipeline | 3 (Shared) |
| **goldmauler** (@Goldmauler) | **Advanced Analytics Lead (Time Series & ARM)** | `notebooks/06_R2_M1`, `notebooks/07_R2_M2`, `src/macro_api.py`, `src/risk_rules.py` | Method 1 (5 marks) + Method 2 (5 marks) Implementation & Evaluation | 10 |
| **Team Member 3** | **Simulation & Macroeconomic Modeler** | `src/portfolio_simulation.py`, `data/portfolio/`, `models/ts_scenario_impact.csv` | Method 1 Data Prep, Scenario Loss Modeling & ECL Quantification | 2 |
| **Team Member 4** | **Full-Stack Dashboard & UI Engineer** | `dashboard/app.py`, `reports/interactive_dashboard.html`, `src/underwriting_model.py` | Mandatory Interactive Application (Live Filters, Drilldown, Scenarios) | 5 |
| **Team Member 5** | **Business Strategist & Technical Reporter** | `notebooks/08_R2`, `reports/Review_2_Final_Report.md`, `reports/Review_2_Presentation_Slides.md` | Combined Business Insights (2 marks) & Panel Presentation Defense (3 marks) | 5 (Shared) |

---

## 4. Workload Verification & Evidence

1. **Commit & PR History**: All commits, branch merges, and task assignments are tracked in the GitHub repository commit log.
2. **Reproducibility Guarantee**: Random seeds (`42` for data generation and ML algorithms; `2026` for portfolio simulation) ensure that any panel member or evaluator executing the pipeline reproduces the exact metric values reported.
3. **Execution Instructions**: Complete running instructions documented in `README.md` and `reports/Review_2_Final_Report.md`.
