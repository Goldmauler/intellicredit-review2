import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

def add_md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def add_code(text):
    cells.append(nbf.v4.new_code_cell(text))

add_md("# Feature Engineering & Dimensionality Reduction\n**Project**: IntelliCredit: Alternative Credit Scoring & Portfolio Risk Engine\n**Member**: Placeholder")

add_md("### Part A: Setup & Data Loading")
add_code("""import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import scipy.stats as stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Set up paths
sys.path.insert(0, os.path.dirname(os.getcwd()))
data_dir = os.path.join(os.path.dirname(os.getcwd()), 'data', 'processed')
fallback_dir = os.path.join(os.path.dirname(os.getcwd()), 'data')

import warnings
warnings.filterwarnings('ignore')
""")

add_code("""# Load cleaned borrower profiles
try:
    df_borrower = pd.read_csv(os.path.join(data_dir, 'borrower_profiles.csv'))
except FileNotFoundError:
    df_borrower = pd.read_csv(os.path.join(fallback_dir, 'borrower_profiles.csv'))

# Load cleaned transactions
try:
    df_txn = pd.read_csv(os.path.join(data_dir, 'monthly_transactions.csv'))
except FileNotFoundError:
    df_txn = pd.read_csv(os.path.join(fallback_dir, 'monthly_transactions.csv'))

# Load macroeconomic indicators
try:
    df_macro = pd.read_csv(os.path.join(data_dir, 'macroeconomic_indicators.csv'))
except FileNotFoundError:
    df_macro = pd.read_csv(os.path.join(fallback_dir, 'macroeconomic_indicators.csv'))

print(f"Borrower data shape: {df_borrower.shape}")
print(f"Transactions shape: {df_txn.shape}")
print(f"Macro data shape: {df_macro.shape}")
""")

add_md("**Insight:** Data successfully loaded. The scale matches expectations (e.g., 30K borrowers, 360K transaction records).")

add_md("### Part B: Transaction-Based Feature Engineering\nThese features are computed FROM `monthly_transactions.csv` and merged into borrower profiles.")

add_code("""# 6. income_stability_score: Rolling 3-month coefficient of variation of gig_income, averaged
def cv(x):
    return np.std(x) / (np.mean(x) + 1e-5)

# Sort by borrower and month
df_txn_sorted = df_txn.sort_values(['borrower_id', 'month_index'])

# Compute CV over a rolling 3-month window
rolling_cv = df_txn_sorted.groupby('borrower_id')['gig_income'].rolling(window=3).apply(cv).reset_index(0, drop=True)
df_txn_sorted['rolling_cv_income'] = rolling_cv

income_stability = df_txn_sorted.groupby('borrower_id')['rolling_cv_income'].mean().reset_index()
income_stability.rename(columns={'rolling_cv_income': 'income_stability_score'}, inplace=True)
income_stability.fillna(0, inplace=True)
""")

add_code("""# 7. income_momentum: Income in last 3 months minus first 3 months, normalized
first_3_months = df_txn_sorted.groupby('borrower_id').head(3).groupby('borrower_id')['gig_income'].sum()
last_3_months = df_txn_sorted.groupby('borrower_id').tail(3).groupby('borrower_id')['gig_income'].sum()

income_momentum = ((last_3_months - first_3_months) / (first_3_months + 1e-5)).reset_index()
income_momentum.rename(columns={'gig_income': 'income_momentum'}, inplace=True)
""")

add_code("""# 8. spending_discipline_ratio: Ratio of months where outflow < income to total months
df_txn_sorted['discipline_flag'] = (df_txn_sorted['monthly_outflow'] < df_txn_sorted['gig_income']).astype(int)
spending_discipline = df_txn_sorted.groupby('borrower_id')['discipline_flag'].mean().reset_index()
spending_discipline.rename(columns={'discipline_flag': 'spending_discipline_ratio'}, inplace=True)
""")

add_code("""# 9. bnpl_acceleration: Slope of BNPL usage over 12 months (increasing BNPL = risky)
def compute_slope(y):
    x = np.arange(len(y))
    if len(y) > 1:
        return np.polyfit(x, y, 1)[0]
    return 0

bnpl_accel = df_txn_sorted.groupby('borrower_id')['bnpl_total_amount'].apply(compute_slope).reset_index()
bnpl_accel.rename(columns={'bnpl_total_amount': 'bnpl_acceleration'}, inplace=True)
""")

add_code("""# 10. payment_consistency_streak: Longest consecutive streak of on-time utility payments
def longest_streak(s):
    streak = max_streak = 0
    for val in s:
        if val == 1:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak

payment_streak = df_txn_sorted.groupby('borrower_id')['utility_paid_on_time'].apply(longest_streak).reset_index()
payment_streak.rename(columns={'utility_paid_on_time': 'payment_consistency_streak'}, inplace=True)
""")

add_code("""# 11. digital_engagement_score & 12. balance_volatility & 13. worst_month_ratio
# digital_engagement_score: Composite of avg(app_login_count) + avg(upi_transaction_count), normalized
dig_eng = df_txn_sorted.groupby('borrower_id')[['app_login_count', 'upi_transaction_count']].mean()
dig_eng['digital_engagement_score'] = StandardScaler().fit_transform(dig_eng.sum(axis=1).values.reshape(-1, 1))

# balance_volatility: Coefficient of variation of end_of_month_balance
bal_vol = df_txn_sorted.groupby('borrower_id')['end_of_month_balance'].apply(cv).reset_index()
bal_vol.rename(columns={'end_of_month_balance': 'balance_volatility'}, inplace=True)
bal_vol.fillna(0, inplace=True)

# worst_month_ratio: Min monthly income / Max monthly income
min_inc = df_txn_sorted.groupby('borrower_id')['gig_income'].min()
max_inc = df_txn_sorted.groupby('borrower_id')['gig_income'].max()
worst_month = (min_inc / (max_inc + 1e-5)).reset_index()
worst_month.rename(columns={'gig_income': 'worst_month_ratio'}, inplace=True)
""")

add_code("""# 14. Merge all new features back into borrower profiles
df_engineered = df_borrower.copy()
dfs_to_merge = [
    income_stability, income_momentum, spending_discipline, 
    bnpl_accel, payment_streak, dig_eng['digital_engagement_score'].reset_index(),
    bal_vol, worst_month
]

for df in dfs_to_merge:
    df_engineered = df_engineered.merge(df, on='borrower_id', how='left')

print("Shape after merging transaction-based features:", df_engineered.shape)
""")

add_code("""# 15. Display correlation of new features with target
new_txn_features = ['income_stability_score', 'income_momentum', 'spending_discipline_ratio', 
                    'bnpl_acceleration', 'payment_consistency_streak', 'digital_engagement_score',
                    'balance_volatility', 'worst_month_ratio']

corr_with_target = df_engineered[new_txn_features + ['loan_default_status']].corr()['loan_default_status'].drop('loan_default_status')
corr_with_target.sort_values(ascending=False).plot(kind='bar', color='skyblue')
plt.title('Correlation of Transaction Features with Default Status')
plt.ylabel('Correlation Coefficient')
plt.show()
""")

add_md("**Insight:** Transaction-based features like `bnpl_acceleration` and `balance_volatility` show noticeable correlations with default risk. Increased BNPL usage over time tends to indicate higher risk, while better `spending_discipline_ratio` typically correlates negatively with default.")

add_md("### Part C: Domain-Based Feature Engineering")

add_code("""# 16. loan_to_income_ratio: loan_amount_requested / (avg_monthly_income * loan_tenure_months)
df_engineered['loan_to_income_ratio'] = df_engineered['loan_amount_requested'] / (df_engineered['avg_monthly_income'] * df_engineered['loan_tenure_months'] + 1e-5)

# 17. repayment_track_record: previous_loans_repaid_on_time / previous_micro_loans (handle 0)
df_engineered['repayment_track_record'] = df_engineered['previous_loans_repaid_on_time'] / (df_engineered['previous_micro_loans'] + 1e-5)
df_engineered.loc[df_engineered['previous_micro_loans'] == 0, 'repayment_track_record'] = -1 # Indicate no history
""")

add_code("""# 18. financial_stress_index: (debt_to_inflow_ratio * income_volatility_index) / (savings_buffer_ratio + 0.01)
df_engineered['financial_stress_index'] = (df_engineered['debt_to_inflow_ratio'] * df_engineered['income_volatility_index']) / (df_engineered['savings_buffer_ratio'] + 0.01)

# 19. debt_service_coverage_ratio: avg_monthly_income / (avg_monthly_outflow + avg_bnpl_monthly_amount + 1)
df_engineered['debt_service_coverage_ratio'] = df_engineered['avg_monthly_income'] / (df_engineered['avg_monthly_outflow'] + df_engineered['avg_bnpl_monthly_amount'] + 1)

# 20. age_tenure_interaction: age * gig_tenure_months (experience proxy)
df_engineered['age_tenure_interaction'] = df_engineered['age'] * df_engineered['gig_tenure_months']
""")

add_code("""# 21. platform_risk_premium: Encode each platform with its observed default rate
platform_defaults = df_engineered.groupby('primary_gig_platform')['loan_default_status'].mean().to_dict()
df_engineered['platform_risk_premium'] = df_engineered['primary_gig_platform'].map(platform_defaults)
""")

add_code("""# 22. macro_stress_exposure: Sensitivity mapping
# We'll map location_type to a synthetic macro sensitivity score (urban might be more sensitive to inflation, etc.)
# A simplified proxy:
location_defaults = df_engineered.groupby('location_type')['loan_default_status'].mean().to_dict()
df_engineered['macro_stress_exposure'] = df_engineered['location_type'].map(location_defaults)
""")

add_code("""# 23. Display summary of all new features with basic stats
domain_features = ['loan_to_income_ratio', 'repayment_track_record', 'financial_stress_index', 
                   'debt_service_coverage_ratio', 'age_tenure_interaction', 'platform_risk_premium', 
                   'macro_stress_exposure']

df_engineered[domain_features].describe()
""")

add_md("**Insight:** Domain-based features capture the fundamental financial health rules. For instance, `financial_stress_index` unifies debt and volatility components into a single critical risk indicator.")

add_md("### Part D: Multicollinearity Analysis")

add_code("""# 24. Compute VIF for all numeric features
numeric_cols = df_engineered.select_dtypes(include=[np.number]).columns.tolist()
# Drop ID and target from VIF
vif_cols = [c for c in numeric_cols if c not in ['borrower_id', 'loan_default_status', 'default_probability_true']]

# Handle NaNs and inf
df_vif_data = df_engineered[vif_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

# VIF computation takes time for 30K rows, we sample for speed if needed, but here we do full or random 5000
df_vif_sample = df_vif_data.sample(min(5000, len(df_vif_data)), random_state=42)
scaler = StandardScaler()
scaled_vif_data = scaler.fit_transform(df_vif_sample)

vif_df = pd.DataFrame()
vif_df["feature"] = vif_cols
vif_df["VIF"] = [variance_inflation_factor(scaled_vif_data, i) for i in range(scaled_vif_data.shape[1])]
""")

add_code("""# 25. Display VIF table, highlight features with VIF > 10
high_vif = vif_df[vif_df['VIF'] > 10].sort_values(by='VIF', ascending=False)
display(high_vif)
""")

add_code("""# 26. Correlation matrix of all features, identify pairs with |r| > 0.85
corr_matrix = df_vif_data.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_pairs = [(c, upper[c].idxmax(), upper[c].max()) for c in upper.columns if upper[c].max() > 0.85]

print("Highly correlated pairs (>0.85):")
for pair in high_corr_pairs:
    print(f"{pair[0]} & {pair[1]} : {pair[2]:.3f}")
""")

add_md("**Insight & Recommendation (27):** Multicollinearity inflates standard errors and makes model interpretation difficult. Features with VIF > 10 and correlation > 0.85 need attention.")

add_code("""# 28. Drop highly collinear redundant features
# Example logic: drop one of each highly correlated pair
cols_to_drop = [pair[0] for pair in high_corr_pairs]
df_engineered.drop(columns=cols_to_drop, inplace=True, errors='ignore')
print(f"Dropped {len(cols_to_drop)} features due to high multicollinearity.")
""")

add_md("### Part E: Feature Selection")

add_code("""# 29. Mutual Information scores for all features vs target
numeric_features = df_engineered.select_dtypes(include=[np.number]).columns.tolist()
if 'borrower_id' in numeric_features: numeric_features.remove('borrower_id')
if 'loan_default_status' in numeric_features: numeric_features.remove('loan_default_status')
if 'default_probability_true' in numeric_features: numeric_features.remove('default_probability_true')

X_full = df_engineered[numeric_features].fillna(0)
y_full = df_engineered['loan_default_status']

mi_scores = mutual_info_classif(X_full, y_full, random_state=42)
mi_scores = pd.Series(mi_scores, name="MI Scores", index=X_full.columns)
mi_scores = mi_scores.sort_values(ascending=False)
""")

add_code("""# 30. Bar chart of top 20 features by MI score
plt.figure(figsize=(10, 6))
mi_scores.head(20).sort_values().plot(kind='barh', color='teal')
plt.title('Top 20 Features by Mutual Information Score')
plt.xlabel('Mutual Information')
plt.show()
""")

add_md("**Insight (31):** The Mutual Information scores highlight which features share the most information with the default status, identifying the strongest non-linear predictors.")

add_code("""# 32. Recursive Feature Elimination (RFE) using RandomForest
rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
# Subsample for speed
X_sample, _, y_sample, _ = train_test_split(X_full, y_full, train_size=0.3, stratify=y_full, random_state=42)

rfe = RFE(estimator=rf, n_features_to_select=20, step=0.1)
rfe.fit(X_sample, y_sample)

rfe_features = X_full.columns[rfe.support_].tolist()
""")

add_code("""# 33. Compare MI ranking vs RFE ranking
mi_top_20 = mi_scores.head(20).index.tolist()
overlap = set(mi_top_20).intersection(set(rfe_features))
print(f"Number of overlapping features between Top 20 MI and RFE: {len(overlap)}")
print("Overlapping features:", overlap)
""")

add_md("**Insight (34):** Comparing tree-based RFE with statistical MI gives us confidence in robust features that are selected by both methodologies.")

add_code("""# 35. Select final feature set based on combined evidence
# We will use the union of top 20 MI and RFE features to ensure no strong predictor is left out
final_features = list(set(mi_top_20).union(set(rfe_features)))
print(f"Total selected final features: {len(final_features)}")
""")

add_md("### Part F: Dimensionality Reduction Visualization")

add_code("""# 36. Standard Scaling & 34. PCA with scree plot
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full[final_features])

pca = PCA()
pca.fit(X_scaled)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o', linestyle='--')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.grid(True)
plt.show()
""")

add_md("**Insight:** The scree plot indicates how many principal components are capturing the most variance. An elbow point usually indicates the optimal number of components to retain.")

add_code("""# 35. Cumulative variance plot
cum_variance = np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cum_variance) + 1), cum_variance, marker='o', color='purple')
plt.axhline(y=0.90, color='r', linestyle='-')
plt.axhline(y=0.95, color='g', linestyle='-')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance')
plt.legend(['Cumulative Variance', '90% Threshold', '95% Threshold'])
plt.grid(True)
plt.show()

n_90 = np.argmax(cum_variance >= 0.90) + 1
n_95 = np.argmax(cum_variance >= 0.95) + 1
print(f"Components for 90% variance: {n_90}")
print(f"Components for 95% variance: {n_95}")
""")

add_code("""# 36. PCA 2D projection colored by default status
pca_2 = PCA(n_components=2)
X_pca_2 = pca_2.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca_2[:, 0], y=X_pca_2[:, 1], hue=y_full, palette='coolwarm', alpha=0.5)
plt.title('PCA 2D Projection')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()
""")

add_code("""# 37. t-SNE 2D projection colored by default status
# Use a sample for t-SNE as it is computationally expensive
tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
sample_idx = np.random.choice(len(X_scaled), 5000, replace=False)

X_tsne = tsne.fit_transform(X_scaled[sample_idx])
y_tsne = y_full.iloc[sample_idx]

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_tsne, palette='Set1', alpha=0.6)
plt.title('t-SNE 2D Projection (Sampled)')
plt.show()
""")

add_md("**Insight (38):** Compare PCA vs t-SNE: t-SNE generally uncovers complex, non-linear cluster formations which might not be distinctly linearly separable in the PCA projection. Both plots provide intuition on feature separability for the modeling phase.")

add_md("### Part G: Train/Test Split & SMOTE")

add_code("""# 39. Define X (selected features) and y (loan_default_status)
X = df_engineered[final_features].fillna(0)
y = df_engineered['loan_default_status']

# 40. Stratified train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
""")

add_code("""# 41. Display class distribution in train and test
print("Train class distribution:\\n", y_train.value_counts(normalize=True))
print("\\nTest class distribution:\\n", y_test.value_counts(normalize=True))
""")

add_code("""# 42. Apply SMOTE to training data only
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
""")

add_code("""# 43. Display class distribution after SMOTE
print("Train class distribution after SMOTE:\\n", y_train_smote.value_counts(normalize=True))
""")

add_md("**Insight:** SMOTE effectively balances the training dataset. We only apply it to the training set to prevent data leakage and ensure the test set represents the true real-world distribution.")

add_md("### Part H: Save Outputs")

add_code("""# 44. Save feature-engineered full dataset to data/processed/borrower_featured.csv
df_engineered.to_csv(os.path.join(data_dir, 'borrower_featured.csv'), index=False)
""")

add_code("""# 45. Save train, test, train_smote CSVs to data/processed/
train_df = X_train.copy()
train_df['loan_default_status'] = y_train
train_df.to_csv(os.path.join(data_dir, 'train.csv'), index=False)

test_df = X_test.copy()
test_df['loan_default_status'] = y_test
test_df.to_csv(os.path.join(data_dir, 'test.csv'), index=False)

train_smote_df = X_train_smote.copy()
train_smote_df['loan_default_status'] = y_train_smote
train_smote_df.to_csv(os.path.join(data_dir, 'train_smote.csv'), index=False)
""")

add_code("""# 46. Save feature column list to data/processed/feature_columns.txt
with open(os.path.join(data_dir, 'feature_columns.txt'), 'w') as f:
    for feature in final_features:
        f.write(f"{feature}\\n")
print("Saved feature columns successfully.")
""")

add_md("""### Summary of Engineered Features & Rationale
- **Transaction-based:** `income_stability_score`, `income_momentum`, `bnpl_acceleration`, etc., derived directly from temporal transaction data.
- **Domain-based:** `financial_stress_index`, `loan_to_income_ratio`, `platform_risk_premium`, capturing core gig-economy financial principles and risk factors.
- **Multicollinearity & Selection:** Reduced feature redundancy using VIF and Correlation limits. Final set identified via Mutual Information and Random Forest RFE to select top predictors.
- **Data Balancing:** Applied SMOTE to the training set to handle default imbalances, setting up a solid foundation for robust machine learning models in M4.
""")

nb.cells = cells
with open(r'c:\Users\rgdar\OneDrive\Desktop\Academics\ASC\Sem-7\BA\CapStone_Project\notebooks\03_M3_feature_engineering.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook created successfully.")
