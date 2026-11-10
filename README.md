# IntelliCredit: Alternative Credit Scoring & Portfolio Risk Engine


> **Capstone Project — Business Analytics**  
> An end-to-end data science and machine learning framework designed to evaluate creditworthiness for underserved borrowers using alternative financial and behavioral indicators.

---

## Executive Summary & Problem Statement

Traditional credit scoring models rely heavily on formal bureau history (e.g., CIBIL/FICO scores, legacy banking records), excluding millions of creditworthy thin-file individuals and micro-entrepreneurs. 

**IntelliCredit** bridges this financial inclusion gap by:
1. Synthesizing holistic financial footprints across micro-lending transactions, mobile repayment behavior, and macroeconomic shifts.
2. Developing modular, explainable machine learning scoring models to predict 90-day delinquency risk.
3. Providing clear model interpretability (SHAP values) and risk cut-off analysis to balance approval rates against expected default losses.

---

## Rubric & Academic Policy Compliance

In accordance with course project rubrics:
* **No Pre-Built Datasets Policy:** Pre-built datasets from Kaggle, UCI, or GitHub repositories were strictly prohibited. All data used in this study was programmatically generated using `data_generation.py`.
* **Reproducibility:** A fixed random seed (`seed=42`) is utilized across synthetic data generation and train-test splits.
* **Review 1 Deliverables (20 Marks):** Complete implementation of Data Cleaning, Preprocessing, EDA, Feature Engineering (WOE/IV), Benchmark Modeling, and Model Interpretability.

---

## Repository Structure

```text
credit-scoring-engine/
│
├── README.md                           # Project documentation & overview
├── requirements.txt                    # Project dependencies
├── .gitignore                          # Git tracking rules for large data/models
├── data_generation.py                  # Vectorized synthetic dataset generator
│
├── data/
│   ├── data_dictionary.txt             # Variable definitions and schema documentation
│   ├── macroeconomic_indicators.csv    # 36 months of macroeconomic trend indicators
│   ├── borrower_profiles.csv           # Raw profile records (30,000 borrowers)
│   ├── monthly_transactions.csv        # Detailed transactions (360,000 records)
│   └── processed/
│       └── feature_columns.txt         # Final engineered features used for training
│
├── notebooks/                          # Graded Review 1 Deliverables
│   ├── 01_M1_data_cleaning_preprocessing.ipynb    # Data cleaning & type conversion
│   ├── 02_M2_exploratory_data_analysis.ipynb      # Univariate, bivariate & macro EDA
│   ├── 03_M3_feature_engineering.ipynb            # WoE/IV, VIF selection & SMOTE
│   ├── 04_M4_predictive_modeling.ipynb            # Classifiers & hyperparameter tuning
│   └── 05_M5_evaluation_interpretation.ipynb      # ROC, KS-statistic, SHAP & Cut-offs
│
├── src/                                # Modular configuration
│   ├── __init__.py
│   └── config.py                       # Project paths, features & random state
│
├── models/
│   └── model_comparison.csv            # Summary metrics across all trained models
│
└── scripts/
    └── generators/                     # Helper notebook scaffolding scripts
```

---

## Project Workflow & Modules

The project is structured into 5 sequential review modules:

### Module 1: Data Cleaning & Preprocessing (`01_M1`)
* Ingestion of multi-table synthetic lending data (borrower demographics, transaction logs, macro indicators).
* Imputation of missing values and median replacement for skew-tolerant variables.
* Detection and handling of statistical outliers using interquartile range (IQR) bounds.

### Module 2: Exploratory Data Analysis (`02_M2`)
* Target imbalance inspection (approx. 92% non-default vs. 8% default rate).
* Cross-sectional distributions across monthly income, debt-to-income ratio (DTI), and credit utilization.
* Correlation matrices and macroeconomic shock impact analysis on repayment rates.

### Module 3: Feature Engineering & Selection (`03_M3`)
* **Weight of Evidence (WoE)** and **Information Value (IV)** screening to retain high-predictive power features ($IV > 0.1$).
* Multicollinearity filtering using **Variance Inflation Factor (VIF)**.
* Class imbalance mitigation using **SMOTE** (Synthetic Minority Over-sampling Technique) on training sets only (preventing test set leakage).

### Module 4: Predictive Modeling & Ensembling (`04_M4`)
* Training baseline and tuned configurations for:
  * **Logistic Regression** (traditional regulatory scorecard baseline)
  * **Random Forest** & **Gradient Boosting**
  * **XGBoost** & **LightGBM**
  * **Voting Classifier** & **Stacking Ensemble**
* Grid/Randomized hyperparameter search optimizing for ROC-AUC.

### Module 5: Evaluation & Interpretability (`05_M5`)
* **Performance Metrics:** ROC-AUC, PR-AUC curve, Confusion Matrices, and Kolmogorov-Smirnov (**KS-statistic**).
* **Explainable AI (XAI):** SHAP summary plots, beeswarm distributions, and individual applicant waterfall plots.
* **Risk Policy Cut-offs:** Evaluation of credit approval rates vs. expected default rates across decision thresholds.

---

## Model Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Tuned)** | **67.18%** | **0.1504** | **68.22%** | **0.2465** | **0.7409** |
| **Stacking Classifier (Tuned)** | 92.13% | 0.0000 | 0.00% | 0.0000 | 0.7398 |
| **Voting Classifier (Tuned)** | 87.65% | 0.2388 | 26.06% | 0.2492 | 0.7386 |
| **LightGBM (Tuned)** | 78.17% | 0.1864 | 52.75% | 0.2754 | 0.7343 |
| **Logistic Regression (Tuned)** | 68.90% | 0.1487 | 62.50% | 0.2402 | 0.7261 |

*Note: For credit default prediction, tuned models prioritized recall and AUC-ROC to capture high-risk default applicants while optimizing threshold trade-offs.*

---

## Getting Started & Execution Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Darun1017/credit-scoring-engine.git
cd credit-scoring-engine
```

### 2. Set Up Python Environment
Ensure Python 3.9+ is installed:
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Generate the Dataset
To regenerate the compliant synthetic dataset:
```bash
python data_generation.py
```
*(Produces `borrower_profiles.csv`, `monthly_transactions.csv`, and `macroeconomic_indicators.csv` inside `data/`)*

### 4. Run the Pipeline
Open Jupyter Notebook or JupyterLab:
```bash
jupyter lab
```
Execute the notebooks in sequence inside the `notebooks/` directory:
1. `01_M1_data_cleaning_preprocessing.ipynb`
2. `02_M2_exploratory_data_analysis.ipynb`
3. `03_M3_feature_engineering.ipynb`
4. `04_M4_predictive_modeling.ipynb`
5. `05_M5_evaluation_interpretation.ipynb`

---
