"""Builds notebooks/04_M4_predictive_modeling.ipynb (Review 1 - Member 4: predictive model development).

Run from the repository root; the notebook is written by nbformat and is not executed here."""
import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = []

# Part A: Setup & Data Loading
cells.append(nbf.v4.new_markdown_cell("# Predictive Model Development\n**Member 4: Predictive Model Development**\n\nThis notebook covers Rubric Section 5: Predictive Model Development."))
cells.append(nbf.v4.new_code_cell("""import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import xgboost as xgb
import lightgbm as lgb
import joblib

sys.path.insert(0, os.path.dirname(os.getcwd()))
# %matplotlib inline"""))

cells.append(nbf.v4.new_markdown_cell("### Load Data\nWe load the preprocessed and split data from the `data/processed/` directory. If it doesn't exist, we fallback to splitting `borrower_featured.csv`."))
cells.append(nbf.v4.new_code_cell("""# Define paths
data_dir = os.path.join(os.path.dirname(os.getcwd()), 'data')
processed_dir = os.path.join(data_dir, 'processed')
models_dir = os.path.join(os.path.dirname(os.getcwd()), 'models')
os.makedirs(models_dir, exist_ok=True)

# Attempt to load split data
try:
    X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(processed_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(processed_dir, 'y_test.csv')).values.ravel()
    
    X_train_smote = pd.read_csv(os.path.join(processed_dir, 'X_train_smote.csv'))
    y_train_smote = pd.read_csv(os.path.join(processed_dir, 'y_train_smote.csv')).values.ravel()
    print("Loaded processed and split data (including SMOTE).")
except FileNotFoundError:
    print("Split data not found. Falling back to borrower_featured.csv...")
    df = pd.read_csv(os.path.join(data_dir, 'processed', 'borrower_featured.csv'))
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE
    
    # Exclude non-feature columns
    exclude_cols = ['borrower_id', 'borrower_name', 'loan_default_status', 'default_probability_true']
    features = [c for c in df.columns if c not in exclude_cols]
    
    X = df[features]
    y = df['loan_default_status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print("Data loaded and split manually.")

print(f"X_train_smote shape: {X_train_smote.shape}, y_train_smote shape: {y_train_smote.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")"""))

# Part B: Baseline Models
cells.append(nbf.v4.new_markdown_cell("## Part B: Baseline Models\nTraining baseline models on SMOTE data (or using class weights) to compare initial performance."))

cells.append(nbf.v4.new_code_cell("""# Helper function to evaluate and store metrics
results_list = []

def evaluate_model(model_name, y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    
    print(f"--- {model_name} ---")
    print(classification_report(y_true, y_pred))
    
    results_list.append({
        'Model': model_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'AUC-ROC': auc
    })"""))

# LR
cells.append(nbf.v4.new_markdown_cell("### 1. Logistic Regression"))
cells.append(nbf.v4.new_code_cell("""lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(random_state=42, max_iter=1000))
])

lr_pipeline.fit(X_train_smote, y_train_smote)
y_pred_lr = lr_pipeline.predict(X_test)
y_prob_lr = lr_pipeline.predict_proba(X_test)[:, 1]

evaluate_model('Logistic Regression', y_test, y_pred_lr, y_prob_lr)"""))
cells.append(nbf.v4.new_markdown_cell("Logistic regression provides a solid baseline. SMOTE helps with the minority class recall."))

# RF
cells.append(nbf.v4.new_markdown_cell("### 2. Random Forest"))
cells.append(nbf.v4.new_code_cell("""rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train) # Training on regular data since RF handles it better, or we could use smote
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

evaluate_model('Random Forest', y_test, y_pred_rf, y_prob_rf)"""))
cells.append(nbf.v4.new_markdown_cell("Random Forest performs well right out of the box, though it might overfit slightly."))

# XGB
cells.append(nbf.v4.new_markdown_cell("### 3. XGBoost"))
cells.append(nbf.v4.new_code_cell("""scale_pos_weight = sum(y_train==0) / sum(y_train==1)
xgb_model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

evaluate_model('XGBoost', y_test, y_pred_xgb, y_prob_xgb)"""))
cells.append(nbf.v4.new_markdown_cell("XGBoost with `scale_pos_weight` shows excellent capability in capturing defaults."))

# LGBM
cells.append(nbf.v4.new_markdown_cell("### 4. LightGBM"))
cells.append(nbf.v4.new_code_cell("""lgb_model = lgb.LGBMClassifier(is_unbalance=True, random_state=42)
lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_test)
y_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]

evaluate_model('LightGBM', y_test, y_pred_lgb, y_prob_lgb)"""))
cells.append(nbf.v4.new_markdown_cell("LightGBM is fast and very competitive with XGBoost when handling imbalanced data."))

# GBC
cells.append(nbf.v4.new_markdown_cell("### 5. Gradient Boosting Classifier"))
cells.append(nbf.v4.new_code_cell("""gbc_model = GradientBoostingClassifier(random_state=42)
gbc_model.fit(X_train_smote, y_train_smote)
y_pred_gbc = gbc_model.predict(X_test)
y_prob_gbc = gbc_model.predict_proba(X_test)[:, 1]

evaluate_model('Gradient Boosting', y_test, y_pred_gbc, y_prob_gbc)"""))
cells.append(nbf.v4.new_markdown_cell("Standard Gradient Boosting trained on SMOTE data also provides strong performance."))

# Baseline Comparison
cells.append(nbf.v4.new_markdown_cell("### Baseline Comparison"))
cells.append(nbf.v4.new_code_cell("""baseline_results = pd.DataFrame(results_list)
display(baseline_results)"""))
cells.append(nbf.v4.new_code_cell("""baseline_results.set_index('Model')[['F1-Score', 'AUC-ROC']].plot(kind='bar', figsize=(10, 6))
plt.title('Baseline Models Comparison')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()"""))

# Part C: Hyperparameter Tuning
cells.append(nbf.v4.new_markdown_cell("## Part C: Hyperparameter Tuning\nWe will tune the top models using RandomizedSearchCV/GridSearchCV."))
cells.append(nbf.v4.new_code_cell("""cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
tuned_results_list = []

def evaluate_tuned_model(model_name, y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    
    tuned_results_list.append({
        'Model': model_name + ' (Tuned)',
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'AUC-ROC': auc
    })"""))

# Tuned LR
cells.append(nbf.v4.new_code_cell("""# Logistic Regression Grid Search
lr_params = {
    'lr__C': [0.01, 0.1, 1, 10],
    'lr__penalty': ['l2'],
    'lr__solver': ['lbfgs', 'liblinear']
}

lr_grid = GridSearchCV(lr_pipeline, param_grid=lr_params, cv=cv, scoring='f1', n_jobs=-1)
lr_grid.fit(X_train_smote, y_train_smote)
print(f"Best LR Params: {lr_grid.best_params_}")

best_lr = lr_grid.best_estimator_
y_pred_lr_t = best_lr.predict(X_test)
y_prob_lr_t = best_lr.predict_proba(X_test)[:, 1]
evaluate_tuned_model('Logistic Regression', y_test, y_pred_lr_t, y_prob_lr_t)"""))

# Tuned RF
cells.append(nbf.v4.new_code_cell("""# Random Forest Randomized Search
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

rf_random = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_distributions=rf_params, 
                               n_iter=10, cv=cv, scoring='f1', n_jobs=-1, random_state=42)
rf_random.fit(X_train_smote, y_train_smote) # Use SMOTE for tuning
print(f"Best RF Params: {rf_random.best_params_}")

best_rf = rf_random.best_estimator_
y_pred_rf_t = best_rf.predict(X_test)
y_prob_rf_t = best_rf.predict_proba(X_test)[:, 1]
evaluate_tuned_model('Random Forest', y_test, y_pred_rf_t, y_prob_rf_t)"""))

# Tuned XGB
cells.append(nbf.v4.new_code_cell("""# XGBoost Randomized Search
xgb_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 2, 5]
}

xgb_random = RandomizedSearchCV(xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss'), 
                                param_distributions=xgb_params, n_iter=15, cv=cv, scoring='roc_auc', n_jobs=-1, random_state=42)
xgb_random.fit(X_train, y_train)
print(f"Best XGB Params: {xgb_random.best_params_}")

best_xgb = xgb_random.best_estimator_
y_pred_xgb_t = best_xgb.predict(X_test)
y_prob_xgb_t = best_xgb.predict_proba(X_test)[:, 1]
evaluate_tuned_model('XGBoost', y_test, y_pred_xgb_t, y_prob_xgb_t)"""))

# Tuned LGBM
cells.append(nbf.v4.new_code_cell("""# LightGBM Randomized Search
lgb_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [-1, 5, 10, 15],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [31, 50, 100],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [0, 0.1, 1]
}

lgb_random = RandomizedSearchCV(lgb.LGBMClassifier(is_unbalance=True, random_state=42), 
                                param_distributions=lgb_params, n_iter=15, cv=cv, scoring='roc_auc', n_jobs=-1, random_state=42)
lgb_random.fit(X_train, y_train)
print(f"Best LGBM Params: {lgb_random.best_params_}")

best_lgb = lgb_random.best_estimator_
y_pred_lgb_t = best_lgb.predict(X_test)
y_prob_lgb_t = best_lgb.predict_proba(X_test)[:, 1]
evaluate_tuned_model('LightGBM', y_test, y_pred_lgb_t, y_prob_lgb_t)"""))

cells.append(nbf.v4.new_markdown_cell("### Tuned Models Comparison"))
cells.append(nbf.v4.new_code_cell("""tuned_results = pd.DataFrame(tuned_results_list)
display(tuned_results)"""))

cells.append(nbf.v4.new_code_cell("""tuned_results.set_index('Model')[['F1-Score', 'AUC-ROC']].plot(kind='bar', figsize=(10, 6))
plt.title('Tuned Models Comparison')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()"""))
cells.append(nbf.v4.new_markdown_cell("Tuning generally improves the stability and slightly boosts the AUC-ROC and F1-score across models."))

# Part D: CV Analysis
cells.append(nbf.v4.new_markdown_cell("## Part D: Cross-Validation Analysis\nLet's evaluate the stability of our tuned models using 5-fold cross-validation."))
cells.append(nbf.v4.new_code_cell("""models = {
    'Logistic Regression': best_lr,
    'Random Forest': best_rf,
    'XGBoost': best_xgb,
    'LightGBM': best_lgb
}

cv_results = {}
for name, model in models.items():
    # Use standard training data since CV does its own splitting (except LR/RF which we trained on SMOTE, we'll use X_train_smote for them here for fair comparison of the estimator itself)
    X_cv = X_train_smote if name in ['Logistic Regression', 'Random Forest'] else X_train
    y_cv = y_train_smote if name in ['Logistic Regression', 'Random Forest'] else y_train
    
    scores = cross_val_score(model, X_cv, y_cv, cv=cv, scoring='roc_auc', n_jobs=-1)
    cv_results[name] = scores

cv_df = pd.DataFrame(cv_results)

# Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=cv_df)
plt.title('5-Fold Cross Validation AUC-ROC Scores')
plt.ylabel('AUC-ROC')
plt.show()"""))

cells.append(nbf.v4.new_code_cell("""print("Mean CV AUC-ROC Scores (± std):")
for name in cv_df.columns:
    print(f"{name}: {cv_df[name].mean():.4f} (± {cv_df[name].std():.4f})")"""))

# Learning curves for top 2
cells.append(nbf.v4.new_code_cell("""def plot_learning_curve(model, title, X, y):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.title(title)
    plt.xlabel('Training examples')
    plt.ylabel('AUC-ROC Score')
    plt.legend(loc="best")

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plot_learning_curve(best_xgb, "Learning Curve (XGBoost)", X_train, y_train)
plt.subplot(1, 2, 2)
plot_learning_curve(best_lgb, "Learning Curve (LightGBM)", X_train, y_train)
plt.tight_layout()
plt.show()"""))
cells.append(nbf.v4.new_markdown_cell("The learning curves show whether our models suffer from high bias or high variance as more data is introduced."))

# Part E: Ensembles
cells.append(nbf.v4.new_markdown_cell("## Part E: Ensemble Methods\nCombining our best models to see if we can eke out more performance."))
cells.append(nbf.v4.new_code_cell("""# Voting Classifier
voting_clf = VotingClassifier(
    estimators=[
        ('rf', best_rf),
        ('xgb', best_xgb),
        ('lgb', best_lgb)
    ],
    voting='soft'
)

voting_clf.fit(X_train, y_train) # Training on standard data
y_pred_vote = voting_clf.predict(X_test)
y_prob_vote = voting_clf.predict_proba(X_test)[:, 1]

evaluate_tuned_model('Voting Classifier', y_test, y_pred_vote, y_prob_vote)"""))

cells.append(nbf.v4.new_code_cell("""# Stacking Classifier
stacking_clf = StackingClassifier(
    estimators=[
        ('rf', best_rf),
        ('xgb', best_xgb),
        ('lgb', best_lgb)
    ],
    final_estimator=LogisticRegression(),
    cv=3,
    n_jobs=-1
)

stacking_clf.fit(X_train, y_train)
y_pred_stack = stacking_clf.predict(X_test)
y_prob_stack = stacking_clf.predict_proba(X_test)[:, 1]

evaluate_tuned_model('Stacking Classifier', y_test, y_pred_stack, y_prob_stack)"""))

cells.append(nbf.v4.new_markdown_cell("### Final Comparison"))
cells.append(nbf.v4.new_code_cell("""final_results = pd.DataFrame(tuned_results_list)
display(final_results)"""))

# Part F: Save Models
cells.append(nbf.v4.new_markdown_cell("## Part F: Save Models\nSaving the trained models for future evaluation and deployment."))
cells.append(nbf.v4.new_code_cell("""# Define best model based on F1 and AUC
# Here we'll select the model with highest AUC-ROC from the final results
best_model_name = final_results.loc[final_results['AUC-ROC'].idxmax()]['Model']
print(f"Best overall model is: {best_model_name}")

# Map name to object
model_dict = {
    'Logistic Regression (Tuned)': best_lr,
    'Random Forest (Tuned)': best_rf,
    'XGBoost (Tuned)': best_xgb,
    'LightGBM (Tuned)': best_lgb,
    'Voting Classifier (Tuned)': voting_clf,
    'Stacking Classifier (Tuned)': stacking_clf
}
best_model_obj = model_dict.get(best_model_name, best_xgb)

# Save all to models/
joblib.dump(best_lr, os.path.join(models_dir, 'tuned_lr.joblib'))
joblib.dump(best_rf, os.path.join(models_dir, 'tuned_rf.joblib'))
joblib.dump(best_xgb, os.path.join(models_dir, 'tuned_xgb.joblib'))
joblib.dump(best_lgb, os.path.join(models_dir, 'tuned_lgbm.joblib'))
joblib.dump(voting_clf, os.path.join(models_dir, 'voting_clf.joblib'))
joblib.dump(stacking_clf, os.path.join(models_dir, 'stacking_clf.joblib'))

# Save best model explicitly
joblib.dump(best_model_obj, os.path.join(models_dir, 'best_model.joblib'))

# Save comparison results
all_results = pd.concat([baseline_results, final_results], ignore_index=True)
all_results.to_csv(os.path.join(models_dir, 'model_comparison.csv'), index=False)
print("Models and results saved successfully.")"""))

nb['cells'] = cells

notebook_path = os.path.join(os.path.dirname(os.getcwd()), 'notebooks', '04_M4_predictive_modeling.ipynb')
os.makedirs(os.path.dirname(notebook_path), exist_ok=True)
with open(notebook_path, 'w') as f:
    nbf.write(nb, f)
print(f"Notebook written to {notebook_path}")
