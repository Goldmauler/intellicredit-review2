# IntelliCredit: GitHub Project Tracking & Kanban Board Setup
**Capstone Project in Business Analytics — Review 2 (Unit 3: Advanced Analytics)**  
*Compliance Guide: 100% Alignment with Capstone Project Rubric Tracking Requirements*

---

## 1. Compliance with Capstone Project Rubric

The updated Capstone Project Rubric defines the following strict criteria for project tracking:

| Rubric Requirement | How IntelliCredit Satisfies It | Evidence Location |
|---|---|---|
| **Project-tracking tool** | Hosted on **GitHub Projects** and integrated with the GitHub repository | [Goldmauler/intellicredit-review2](https://github.com/Goldmauler/intellicredit-review2) |
| **5 Mandatory Board Stages** | **Backlog**, **To Do**, **In Progress**, **Review/Testing**, **Completed** | Status labels (`status: Backlog` ... `status: Completed`) and ProjectV2 single-select fields |
| **Clearly defined tasks** | 18 modular engineering tasks covering Sprints 1 to 4 (`TSK-101` to `TSK-405`) | GitHub Issues #1 to #18 & `create_github_board.py` |
| **Assigned to team member with due date** | Every task has an assigned owner (`Darun R G`, `goldmauler`, `Team Member 3`, `Team Member 4`, `Team Member 5`) and sprint due date | Task issue metadata & Milestone deadlines (Sprint 1: W2, Sprint 2: W4, Sprint 3: W6, Sprint 4: W8) |
| **Supporting evidence on each task** | Code files, notebooks, datasets, dashboard tabs, report sections explicitly cited | Documented in issue body, `README.md`, and `reports/project_management_and_contributions.md` |
| **Activity history & contribution verification** | Full Git commit history + closed issues + milestones | Verified via GitHub Repository Insights, Milestones, and Issue history |

---

## 2. Sprint Milestones & Task Inventory

### Sprint 1: Data Architecture & Baseline EDA (Review 1 Foundation · Due: Week 2)
* **Milestone:** `Sprint 1: Data Architecture & Baseline EDA` (Closed / 100% Complete)
* **TSK-101:** Synthetic data generation (30k borrowers, 360k transactions) — *Darun R G*  
  *Evidence:* `data_generation.py`, `data/borrower_profiles.csv`, `data/monthly_transactions.csv`
* **TSK-102:** Data cleaning & preprocessing pipeline (KNN/mode, winsorization, encoding) — *Darun R G*  
  *Evidence:* `notebooks/01_M1_data_cleaning_preprocessing.ipynb`
* **TSK-103:** Exploratory data analysis (38 visualizations, hypothesis tests) — *Darun R G*  
  *Evidence:* `notebooks/02_M2_exploratory_data_analysis.ipynb`
* **TSK-104:** Data dictionary and schema documentation — *Darun R G*  
  *Evidence:* `data/data_dictionary.txt`, `data/data_dictionary_review2.md`

### Sprint 2: Feature Engineering & Predictive Default Models (Review 1 Delivery · Due: Week 4)
* **Milestone:** `Sprint 2: Feature Engineering & Predictive Default Models` (Closed / 100% Complete)
* **TSK-201:** Feature engineering (14 behavioral risk ratios, VIF, WoE/IV screening) — *Darun R G*  
  *Evidence:* `notebooks/03_M3_feature_engineering.ipynb`
* **TSK-202:** Leak-free stratified 80/20 train/test split — *Darun R G*  
  *Evidence:* `data/processed/train.csv`, `data/processed/test.csv`
* **TSK-203:** Predictive default model development (5 models + leak-free SMOTE CV) — *goldmauler*  
  *Evidence:* `notebooks/04_M4_predictive_modeling.ipynb`
* **TSK-204:** Comprehensive model evaluation (ROC-AUC 0.738, KS 0.36, cost-optimal INR threshold, TreeSHAP) — *goldmauler*  
  *Evidence:* `notebooks/05_M5_evaluation_interpretation.ipynb`, `models/model_comparison.csv`

### Sprint 3: Advanced Analytics — Time Series & Association Rules (Review 2 · Due: Week 6)
* **Milestone:** `Sprint 3: Advanced Analytics — Time Series & ARM` (Closed / 100% Complete)
* **TSK-301:** Live macroeconomic API collector (FRED + World Bank, 108 months, caching) — *goldmauler*  
  *Evidence:* `src/macro_api.py`, `data/external/macro_india_monthly.csv`, `data/external/macro_sources.json`
* **TSK-302:** Portfolio loan performance simulation (30k loans, 250k loan-months, macro hazard link) — *Team Member 3*  
  *Evidence:* `src/portfolio_simulation.py`, `data/portfolio/`
* **TSK-303:** Portfolio default-rate time-series forecasting (SARIMAX, 7 models, 40-origin backtest, 3 stress scenarios) — *goldmauler*  
  *Evidence:* `notebooks/06_R2_M1_time_series_forecasting.ipynb`, `models/ts_model_comparison.csv`
* **TSK-304:** Delinquency warning association rule mining (Apriori vs FP-Growth, 93 validated rules) — *goldmauler*  
  *Evidence:* `notebooks/07_R2_M2_association_rule_mining.ipynb`, `models/arm_model_comparison.csv`
* **TSK-305:** Out-of-sample rule generalization on 6,000 held-out test applicants (100% test lift ≥ 1.3×) — *goldmauler*  
  *Evidence:* `models/association_rules.csv`

### Sprint 4: Combined Insights, Full Interactive Dashboard & Presentation (Review 2 · Due: Week 8)
* **Milestone:** `Sprint 4: Combined Insights, Full Interactive Dashboard & Presentation` (Closed / 100% Complete)
* **TSK-401:** Cross-method risk synthesis (cross-tab heatmap, macro multipliers, credit loss overlay) — *Team Member 5*  
  *Evidence:* `notebooks/08_R2_combined_insights_dashboard.ipynb`
* **TSK-402:** Production Streamlit loan underwriter & portfolio risk application (5 interactive tabs) — *Team Member 4*  
  *Evidence:* `dashboard/app.py`
* **TSK-403:** Standalone browser-executable interactive HTML dashboard — *Team Member 4*  
  *Evidence:* `reports/interactive_dashboard.html`
* **TSK-404:** Data quality audit, sample dataset (100 rows × 44 cols), and real-world benchmark validation (MFIN/PLFS) — *Darun R G*  
  *Evidence:* `data/DATASET_VALIDATION_SUMMARY.md`, `data/sample_final_dataset.csv`
* **TSK-405:** Comprehensive final technical report and 15-minute panel slide deck — *Team Member 5*  
  *Evidence:* `reports/Review_2_Final_Report.md`, `reports/Review_2_Presentation_Slides.md`

---

## 3. Viewing the Project Tracking on GitHub

### Option 1: Direct GitHub Repository Issues & Milestones View
All 18 tasks are published directly to the repository:
- **All Issues (Closed/Completed Tasks with Evidence):**  
  `https://github.com/Goldmauler/intellicredit-review2/issues?q=is%3Aissue`
- **Sprint Milestones with Deadlines:**  
  `https://github.com/Goldmauler/intellicredit-review2/milestones`

### Option 2: Live GitHub Projects (v2) Kanban Board View

To display all issues in a 5-column Kanban board on GitHub in under 1 minute:

1. Log into GitHub and go to the repository or your GitHub profile:
   - Repo: [https://github.com/Goldmauler/intellicredit-review2](https://github.com/Goldmauler/intellicredit-review2)
   - Projects tab: Click **Projects** in the top navigation bar.
2. Click **New project** (green button).
3. Select the **Board** template and click **Create**.
4. Set the project title to:
   `IntelliCredit — Sprint Board (Review 1 & 2)`
5. Verify the 5 columns match the rubric:
   - `Backlog`
   - `To Do`
   - `In Progress`
   - `Review/Testing`
   - `Completed`
6. Click **Add items** (or press `+` at the bottom of the `Completed` column), type `#`, and select the repository `Goldmauler/intellicredit-review2` to add all 18 created task issues.
7. Set the project visibility to **Public** so panel evaluators can access it without login credentials.
8. Copy your project board URL (e.g. `https://github.com/users/Goldmauler/projects/1` or `https://github.com/users/Darun11/projects/1`).

### Option 3: Automated Board Creation via Script (`create_github_board.py`)

If you have a GitHub Personal Access Token (PAT) with `project` scope:
```bash
# 1. Install requests
pip install requests

# 2. Run the automated setup script
python create_github_board.py --token <YOUR_GITHUB_PAT>
```
The script will programmatically create the ProjectV2 board, establish the 5 rubric stages, populate all 18 tasks with evidence descriptions and member assignments, and output the direct URL.

---

## 4. Verification for Faculty Panel

During the Review 2 presentation defense:
1. Open the Project Board link or the Repository Milestones tab.
2. Demonstrate that:
   - Every task has an owner, due date, and verified deliverable.
   - All 4 sprints progressed systematically from initial data collection to final interactive deployment.
   - Individual contribution weights strictly mirror the team workload breakdown submitted in `reports/project_management_and_contributions.md`.
