import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, roc_curve, classification_report)

# 1. LOAD AND PREPARE DATA
import os

# Build path relative to this file or project root
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
csv_path = os.path.join(root_path, 'classData.csv')

df = pd.read_csv(csv_path)

# Defining the "Blackout Risk" target
# A '1' in any of G, C, B, or A columns indicates a fault/blackout condition.
df['Target'] = (df[['G', 'C', 'B', 'A']].sum(axis=1) > 0).astype(int)
# Features and Target
X = df[['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']]
y = df['Target']
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.close()
# Feature Distribution (Ia example)
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='Ia', hue='Target', kde=True, bins=30)
plt.title('Current (Ia) Distribution by Target Class')
plt.savefig('feature_dist_ia.png')
plt.close()
# 3. PREPROCESSING
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Scaling is crucial for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# 4. MODELING
# a. Baseline: Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
# b. Main Model: Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]
# c. Optional Comparison: Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
y_prob_gb = gb.predict_proba(X_test)[:, 1]
# 5. DEEP EVALUATION
models_stats = []
def evaluate_model(y_true, y_pred, y_prob, name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    models_stats.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': auc
    })
    return acc, prec, rec, f1, auc

evaluate_model(y_test, y_pred_lr, y_prob_lr, "Logistic Regression (Baseline)")
evaluate_model(y_test, y_pred_rf, y_prob_rf, "Random Forest (Main)")
evaluate_model(y_test, y_pred_gb, y_prob_gb, "Gradient Boosting (Comparison)")

comparison_df = pd.DataFrame(models_stats)
comparison_df.to_csv('model_comparison_results.csv', index=False)
# 6. VISUALIZATION OF RESULTS
# ROC Curve comparison
plt.figure(figsize=(10, 8))
for name, prob in [("LR", y_prob_lr), ("RF", y_prob_rf), ("GB", y_prob_gb)]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, prob):.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend()
plt.savefig('roc_curves.png')
plt.close()
# Feature Importance (Random Forest)
plt.figure(figsize=(10, 6))
feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
feat_importances.sort_values().plot(kind='barh', color='skyblue')
plt.title('Random Forest Feature Importance')
plt.savefig('rf_feature_importance.png')
plt.close()
# 7. EXPLAINABILITY / ACTIONABLE INSIGHTS
# Logic: If a fault is predicted, find the feature that deviated most from the mean
means = X_train.mean()
stds = X_train.std()
def explain_prediction(features_row):
    # Z-score to see which sensor is most 'out of normal'
    z_scores = (features_row - means) / stds
    extreme_feature = z_scores.abs().idxmax()
    extreme_val = z_scores[extreme_feature]
    explanation = f"High Risk Detected. Primary indicator: Abnormal reading in {extreme_feature} "
    explanation += f"({extreme_val:.2f} standard deviations from normal)."

    if 'I' in extreme_feature:
        action = "Action: Immediate check for Short-Circuit / Overcurrent on this phase."
    else:
        action = "Action: Investigate Voltage Sag / Swell or aging insulation."

    return explanation, action
# Get a sample fault from test set
fault_sample_idx = y_test[y_test == 1].index[0]
sample_features = X_test.loc[fault_sample_idx]
exp, act = explain_prediction(sample_features)
print("--- Sample Insight for Hackathon Presentation ---")
print(f"Prediction: BLACKOUT RISK")
print(f"Reasoning: {exp}")
print(f"Recommendation: {act}")
print("\n--- Model Comparison Summary ---")
print(comparison_df)