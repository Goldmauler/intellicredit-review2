#!/usr/bin/env python3
"""
IntelliCredit - GitHub Projects v2 Board Creator
=================================================
Creates a full Kanban project board on GitHub with all sprint tasks.

Usage:
    python create_github_board.py --token YOUR_GITHUB_PAT

Requirements:
    pip install requests

The PAT needs the following scopes:
    - repo (for linking project to the repository)
    - project  (or read:project + write:project for classic PATs)
    - For fine-grained PATs: "Projects" read & write permission
"""

import argparse
import json
import sys
import time
import requests

OWNER = "Goldmauler"
REPO  = "intellicredit-review2"

GRAPHQL_URL = "https://api.github.com/graphql"

# ─── Sprint Tasks ──────────────────────────────────────────────────────────────
TASKS = [
    # Sprint 1
    {
        "id": "TSK-101",
        "title": "TSK-101: Synthetic data generation (30k borrowers, 360k transactions)",
        "body": "Implemented programmatic data generation using Faker (en_IN) and correlated NumPy distributions.\n\n**Assigned to:** Darun R G\n**Evidence:** `data_generation.py`, `data/borrower_profiles.csv`, `data/monthly_transactions.csv`\n**Sprint:** Sprint 1 — Data Architecture\n**Due:** Week 2",
        "status": "Completed",
    },
    {
        "id": "TSK-102",
        "title": "TSK-102: Data cleaning pipeline (KNN/mode imputation, winsorization, encoding)",
        "body": "Built preprocessing pipeline with KNN imputation for continuous features, mode for categorical, IQR winsorisation at 1st/99th percentile, and centralized ordinal encoding via src/config.py.\n\n**Assigned to:** Darun R G\n**Evidence:** `notebooks/01_M1_data_cleaning_preprocessing.ipynb`\n**Sprint:** Sprint 1 — Data Architecture\n**Due:** Week 2",
        "status": "Completed",
    },
    {
        "id": "TSK-103",
        "title": "TSK-103: Exploratory data analysis (38 charts, Mann-Whitney U, Chi-Square)",
        "body": "Generated 38 comprehensive visualizations, normality evaluations, bivariate hypothesis tests (Mann-Whitney U for continuous, Chi-Square for categorical) and outlier analysis.\n\n**Assigned to:** Darun R G\n**Evidence:** `notebooks/02_M2_exploratory_data_analysis.ipynb`\n**Sprint:** Sprint 1 — Data Architecture\n**Due:** Week 2",
        "status": "Completed",
    },
    {
        "id": "TSK-104",
        "title": "TSK-104: Data dictionary and documentation authoring",
        "body": "Authored comprehensive data dictionary covering all 39 borrower features and 15 transaction features with types, ranges, and business meaning.\n\n**Assigned to:** Darun R G\n**Evidence:** `data/data_dictionary.txt`, `data/data_dictionary_review2.md`\n**Sprint:** Sprint 1 — Data Architecture\n**Due:** Week 2",
        "status": "Completed",
    },
    # Sprint 2
    {
        "id": "TSK-201",
        "title": "TSK-201: Feature engineering (14 behavioral features, VIF, IV, WoE screening)",
        "body": "Engineered 14 alternative risk ratios including income_volatility_index, utility_reliability_score, savings_buffer_ratio, income_trend_slope. Applied VIF (<5), Information Value (IV), and WoE screening for feature selection.\n\n**Assigned to:** Darun R G\n**Evidence:** `notebooks/03_M3_feature_engineering.ipynb`\n**Sprint:** Sprint 2 — Predictive Modelling\n**Due:** Week 4",
        "status": "Completed",
    },
    {
        "id": "TSK-202",
        "title": "TSK-202: Leak-free stratified 80/20 train-test split",
        "body": "Implemented strict chronological and stratified train/test partitioning (80% = 24,000 train; 20% = 6,000 test). All target encodings and SMOTE constrained to training folds only.\n\n**Assigned to:** Darun R G\n**Evidence:** `data/processed/train.csv`, `data/processed/test.csv`\n**Sprint:** Sprint 2 — Predictive Modelling\n**Due:** Week 3",
        "status": "Completed",
    },
    {
        "id": "TSK-203",
        "title": "TSK-203: Predictive model development (LR, RF, XGB, LGBM, Stacking + SMOTE pipeline)",
        "body": "Trained and tuned 5 classification algorithms with leak-free Stratified K-Fold CV. SMOTE moved inside imblearn Pipeline to prevent oversampling leakage. Best: XGBoost AUC-ROC 0.738.\n\n**Assigned to:** goldmauler\n**Evidence:** `notebooks/04_M4_predictive_modeling.ipynb`\n**Sprint:** Sprint 2 — Predictive Modelling\n**Due:** Week 4",
        "status": "Completed",
    },
    {
        "id": "TSK-204",
        "title": "TSK-204: Model evaluation (ROC-AUC, KS=0.36, cost-optimal INR threshold, SHAP)",
        "body": "Evaluated models with ROC/PR curves, KS statistic (0.36), lift/gains charts, calibration plots, TreeSHAP feature importance. Computed cost-optimal threshold: 74.7% approval rate, 4.4% approved default rate.\n\n**Assigned to:** goldmauler\n**Evidence:** `notebooks/05_M5_evaluation_interpretation.ipynb`, `models/model_comparison.csv`\n**Sprint:** Sprint 2 — Predictive Modelling\n**Due:** Week 4",
        "status": "Completed",
    },
    # Sprint 3
    {
        "id": "TSK-301",
        "title": "TSK-301: Live macroeconomic API (FRED + World Bank, 108 months, caching, fallback)",
        "body": "Built macro data collector connecting to FRED API (CPI, call-money rate, G-Sec yield, INR/USD) and World Bank API (unemployment, GDP growth, lending rate). Includes raw-file caching, offline fallback, and full provenance JSON.\n\n**Assigned to:** goldmauler\n**Evidence:** `src/macro_api.py`, `data/external/macro_india_monthly.csv`, `data/external/macro_sources.json`\n**Sprint:** Sprint 3 — Advanced Analytics\n**Due:** Week 6",
        "status": "Completed",
    },
    {
        "id": "TSK-302",
        "title": "TSK-302: Portfolio simulation (30k loans, 250k loan-months, macro-linked DPD migration)",
        "body": "Simulated 30,000 loan contracts originating 2019-2024. Monthly DPD migration (0→30→60→90+ DPD) driven by real macro shocks including COVID lockdown (×5.7 default spike). Generates loan_book.csv, loan_performance.csv, portfolio_monthly.csv.\n\n**Assigned to:** Team Member 3\n**Evidence:** `src/portfolio_simulation.py`, `data/portfolio/`\n**Sprint:** Sprint 3 — Advanced Analytics\n**Due:** Week 5",
        "status": "Completed",
    },
    {
        "id": "TSK-303",
        "title": "TSK-303: Time-series forecasting (SARIMAX, 7 models, 40-origin backtest, 3 stress scenarios)",
        "body": "Implemented full time-series analysis: ADF/KPSS stationarity tests, STL decomposition, Granger causality (unemployment p=0.008, CPI p=0.024). Benchmarked 7 models; SARIMAX with macro drivers wins (hold-out MAE 0.054 pp, MAPE 16%). Rolling-origin backtest across 40 origins. Three 2025 stress scenarios with ECL quantification.\n\n**Assigned to:** goldmauler\n**Evidence:** `notebooks/06_R2_M1_time_series_forecasting.ipynb`, `models/ts_model_comparison.csv`, `models/ts_backtest_summary.csv`\n**Sprint:** Sprint 3 — Advanced Analytics\n**Due:** Week 6",
        "status": "Completed",
    },
    {
        "id": "TSK-304",
        "title": "TSK-304: Association rule mining (Apriori vs FP-Growth benchmark, 93 validated rules)",
        "body": "Discretized 19 behavioral flags on 24,000 training borrowers. Benchmarked Apriori vs FP-Growth (same rules, FP-Growth 2.3× faster). Bonferroni-corrected Fisher's exact test + subset redundancy pruning → 93 non-redundant, statistically significant warning rules.\n\n**Assigned to:** goldmauler\n**Evidence:** `notebooks/07_R2_M2_association_rule_mining.ipynb`, `models/arm_model_comparison.csv`\n**Sprint:** Sprint 3 — Advanced Analytics\n**Due:** Week 6",
        "status": "Completed",
    },
    {
        "id": "TSK-305",
        "title": "TSK-305: ARM validation on 6,000 held-out test borrowers (100% test-lift ≥ 1.3×)",
        "body": "Applied all 93 mined rules to the unseen 6,000 test borrowers. 100% of rules retained test-lift ≥ 1.3×, confirming generalization. Top rule: declining+volatile income + low savings → 24.7% default rate (3.1× lift on test).\n\n**Assigned to:** goldmauler\n**Evidence:** `models/association_rules.csv`\n**Sprint:** Sprint 3 — Advanced Analytics\n**Due:** Week 6",
        "status": "Completed",
    },
    # Sprint 4
    {
        "id": "TSK-401",
        "title": "TSK-401: Combined insights notebook (cross-method synthesis, PiT PD, ECL, recommendations)",
        "body": "Synthesized findings from all 3 analytical layers: rules×PD cross-tabulation, point-in-time PD multiplier, macro-conditioned risk, approval rate simulation across 4 scenarios. Includes 6 evidence-based business recommendations.\n\n**Assigned to:** Team Member 5\n**Evidence:** `notebooks/08_R2_combined_insights_dashboard.ipynb`\n**Sprint:** Sprint 4 — Dashboard & Reporting\n**Due:** Week 8",
        "status": "Completed",
    },
    {
        "id": "TSK-402",
        "title": "TSK-402: Streamlit interactive dashboard (5-tab, SHAP, macro scenarios, multi-filter drilldown)",
        "body": "Production Streamlit application with 5 tabs: (1) Credit Decision Engine with real-time scoring + TreeSHAP waterfall, (2) Dataset Explorer with multi-attribute filters + dynamic KPIs, (3) Portfolio Outlook with scenario toggling, (4) Delinquency Warning Rule Explorer, (5) Model Card & Data Provenance.\n\n**Assigned to:** Team Member 4\n**Evidence:** `dashboard/app.py`\n**Sprint:** Sprint 4 — Dashboard & Reporting\n**Due:** Week 8",
        "status": "Completed",
    },
    {
        "id": "TSK-403",
        "title": "TSK-403: Standalone interactive HTML report (no Python required, browser-ready)",
        "body": "Exported all Plotly charts from notebook 08 into a single standalone HTML file accessible without a Python environment. Includes portfolio forecast, rule explorer, macro driver dashboard, and approval policy simulator.\n\n**Assigned to:** Team Member 4\n**Evidence:** `reports/interactive_dashboard.html`\n**Sprint:** Sprint 4 — Dashboard & Reporting\n**Due:** Week 8",
        "status": "Completed",
    },
    {
        "id": "TSK-404",
        "title": "TSK-404: Dataset quality audit, DATASET_VALIDATION_SUMMARY.md, sample_final_dataset.csv, real-world benchmarks",
        "body": "Conducted full data quality audit resolving 7 anomalies (date drift, encoding bugs, leakage, SMOTE placement). Generated sample_final_dataset.csv (100 rows × 44 cols) with PD scores, risk bands, ARM flags, and underwriting decisions. Added Section 7: Real-World Benchmark Validation against MFIN Micrometer, PLFS 2022-23, CGAP India data.\n\n**Assigned to:** Darun R G\n**Evidence:** `data/DATASET_VALIDATION_SUMMARY.md`, `data/sample_final_dataset.csv`\n**Sprint:** Sprint 4 — Dashboard & Reporting\n**Due:** Week 8",
        "status": "Completed",
    },
    {
        "id": "TSK-405",
        "title": "TSK-405: Final technical report and 15-minute panel presentation slides",
        "body": "Authored comprehensive 15,000-word technical report covering all 3 units + Review 2 advanced analytics. Created 15-minute panel presentation slides with speaker notes for each team member's module.\n\n**Assigned to:** Team Member 5\n**Evidence:** `reports/Review_2_Final_Report.md`, `reports/Review_2_Presentation_Slides.md`\n**Sprint:** Sprint 4 — Dashboard & Reporting\n**Due:** Week 8",
        "status": "Completed",
    },
]

STATUS_OPTIONS = ["Backlog", "To Do", "In Progress", "Review/Testing", "Completed"]


def gql(token: str, query: str, variables: dict = None):
    """Execute a GraphQL query against the GitHub API."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/vnd.github+json",
    }
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    resp = requests.post(GRAPHQL_URL, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(f"GraphQL errors: {json.dumps(data['errors'], indent=2)}")
    return data["data"]


def get_owner_and_repo_ids(token: str):
    """Return (owner_node_id, repo_node_id)."""
    query = """
    query($owner: String!, $repo: String!) {
      repositoryOwner(login: $owner) { id login }
      repository(owner: $owner, name: $repo) { id name }
    }
    """
    data = gql(token, query, {"owner": OWNER, "repo": REPO})
    return data["repositoryOwner"]["id"], data["repository"]["id"]


def create_project(token: str, owner_id: str, title: str):
    """Create a ProjectV2 under the owner and return its node id."""
    mutation = """
    mutation($ownerId: ID!, $title: String!) {
      createProjectV2(input: {ownerId: $ownerId, title: $title}) {
        projectV2 { id number url }
      }
    }
    """
    data = gql(token, mutation, {"ownerId": owner_id, "title": title})
    proj = data["createProjectV2"]["projectV2"]
    print(f"  Created project #{proj['number']}: {proj['url']}")
    return proj["id"], proj["url"]


def link_project_to_repo(token: str, project_id: str, repo_id: str):
    """Link the project to the repository."""
    mutation = """
    mutation($projectId: ID!, $contentId: ID!) {
      linkProjectV2ToRepository(input: {projectId: $projectId, repositoryId: $contentId}) {
        repository { name }
      }
    }
    """
    try:
        gql(token, mutation, {"projectId": project_id, "contentId": repo_id})
        print("  Linked project to repository.")
    except Exception as e:
        print(f"  Warning: Could not link to repo (may already be linked): {e}")


def get_status_field_id(token: str, project_id: str):
    """Find the Status single-select field and return (field_id, {name: option_id})."""
    query = """
    query($projectId: ID!) {
      node(id: $projectId) {
        ... on ProjectV2 {
          fields(first: 20) {
            nodes {
              ... on ProjectV2SingleSelectField {
                id
                name
                options { id name }
              }
            }
          }
        }
      }
    }
    """
    data = gql(token, query, {"projectId": project_id})
    for node in data["node"]["fields"]["nodes"]:
        if node.get("name") == "Status":
            opts = {opt["name"]: opt["id"] for opt in node["options"]}
            return node["id"], opts
    raise RuntimeError("No 'Status' field found on the project.")


def update_status_options(token: str, project_id: str, field_id: str, current_opts: dict):
    """Ensure all required status columns exist; add missing ones."""
    existing = set(current_opts.keys())
    for status in STATUS_OPTIONS:
        if status not in existing:
            mutation = """
            mutation($projectId: ID!, $fieldId: ID!, $name: String!, $color: ProjectV2SingleSelectFieldOptionColor!) {
              createProjectV2FieldOption(input: {
                projectId: $projectId,
                fieldId: $fieldId,
                name: $name,
                color: $color
              }) {
                projectV2FieldOption { id name }
              }
            }
            """
            color_map = {
                "Backlog": "GRAY",
                "To Do": "BLUE",
                "In Progress": "YELLOW",
                "Review/Testing": "ORANGE",
                "Completed": "GREEN",
            }
            try:
                result = gql(token, mutation, {
                    "projectId": project_id,
                    "fieldId": field_id,
                    "name": status,
                    "color": color_map.get(status, "GRAY"),
                })
                new_id = result["createProjectV2FieldOption"]["projectV2FieldOption"]["id"]
                current_opts[status] = new_id
                print(f"  Added status option: {status}")
            except Exception as e:
                print(f"  Warning: Could not add status '{status}': {e}")
    return current_opts


def add_draft_item(token: str, project_id: str, title: str, body: str):
    """Add a draft issue item to the project and return its item id."""
    mutation = """
    mutation($projectId: ID!, $title: String!, $body: String!) {
      addProjectV2DraftIssue(input: {projectId: $projectId, title: $title, body: $body}) {
        projectItem { id }
      }
    }
    """
    data = gql(token, mutation, {"projectId": project_id, "title": title, "body": body})
    return data["addProjectV2DraftIssue"]["projectItem"]["id"]


def set_item_status(token: str, project_id: str, item_id: str, field_id: str, option_id: str):
    """Set the Status field for a project item."""
    mutation = """
    mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $optionId: String!) {
      updateProjectV2ItemFieldValue(input: {
        projectId: $projectId,
        itemId: $itemId,
        fieldId: $fieldId,
        value: { singleSelectOptionId: $optionId }
      }) {
        projectV2Item { id }
      }
    }
    """
    gql(token, mutation, {
        "projectId": project_id,
        "itemId": item_id,
        "fieldId": field_id,
        "optionId": option_id,
    })


def main():
    parser = argparse.ArgumentParser(description="Create IntelliCredit GitHub Projects board")
    parser.add_argument("--token", required=True, help="GitHub Personal Access Token (needs 'project' + 'repo' scopes)")
    parser.add_argument("--title", default="IntelliCredit — Sprint Board (Review 1 & 2)", help="Project title")
    args = parser.parse_args()

    token = args.token
    print(f"\n🚀 Creating GitHub Project: '{args.title}'")
    print(f"   Repo: {OWNER}/{REPO}\n")

    print("1. Fetching owner and repo node IDs...")
    owner_id, repo_id = get_owner_and_repo_ids(token)
    print(f"   Owner ID: {owner_id}")
    print(f"   Repo ID:  {repo_id}")

    print("\n2. Creating ProjectV2...")
    project_id, project_url = create_project(token, owner_id, args.title)

    print("\n3. Linking project to repository...")
    link_project_to_repo(token, project_id, repo_id)

    print("\n4. Fetching status field and options...")
    field_id, status_opts = get_status_field_id(token, project_id)
    print(f"   Status field ID: {field_id}")
    print(f"   Current options: {list(status_opts.keys())}")

    print("\n5. Ensuring all 5 Kanban columns exist...")
    status_opts = update_status_options(token, project_id, field_id, status_opts)

    print(f"\n6. Adding {len(TASKS)} tasks to the board...")
    for i, task in enumerate(TASKS, 1):
        print(f"   [{i:02d}/{len(TASKS)}] {task['id']}: {task['title'][:60]}...")
        item_id = add_draft_item(token, project_id, task["title"], task["body"])
        target_status = task["status"]
        if target_status in status_opts:
            set_item_status(token, project_id, item_id, field_id, status_opts[target_status])
        else:
            print(f"         Warning: status '{target_status}' not found in options")
        time.sleep(0.3)  # be gentle with the API rate limit

    print(f"\n✅ Done! Project board created successfully.")
    print(f"\n📋 Board URL: {project_url}")
    print(f"\n⚠️  Next steps:")
    print(f"   1. Open the URL above and verify all {len(TASKS)} tasks are visible")
    print(f"   2. Update README.md line:")
    print(f'      * **Project Board (Kanban & Sprint Tracking):** [{args.title}]({project_url})')
    print(f"   3. Update reports/project_management_and_contributions.md with the same URL")

    return project_url


if __name__ == "__main__":
    main()
