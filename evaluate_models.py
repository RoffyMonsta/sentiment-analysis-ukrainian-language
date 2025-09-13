import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.model_selection import train_test_split, cross_val_score
from scipy import stats

# === 1. Config ===
DATASET_PATH = 'features_dataset.csv'
MODELS_DIR = 'models'
REPORTS_DIR = 'reports'

os.makedirs(REPORTS_DIR, exist_ok=True)

# === 2. Load data ===
print(f"🔍 Loading dataset: {DATASET_PATH}")
df = pd.read_csv(DATASET_PATH)
X = df.drop(columns=['label'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"✅ Train size: {X_train.shape}, Test size: {X_test.shape}")

# === 3. Models to evaluate ===
models = {
    "Random Forest": os.path.join(MODELS_DIR, "random_forest_model.pkl"),
    "SVM": os.path.join(MODELS_DIR, "svm_model.pkl"),
    "KNN": os.path.join(MODELS_DIR, "knn_model.pkl"),
    "Decision Tree": os.path.join(MODELS_DIR, "decision_tree_model.pkl"),
}

results = []

# === 4. Evaluate each model ===
for name, path in models.items():
    print(f"\n🔍 Evaluating: {name}")

    model = joblib.load(path)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(classification_report(y_test, y_pred, digits=4))
    cm = confusion_matrix(y_test, y_pred)

    # Append results
    # Mathematical calculations
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    # Cross-validation for confidence intervals
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores, ddof=1)
    
    # 95% Confidence Interval: CI = x̄ ± t₀.₀₂₅ × (s/√n)
    n = len(cv_scores)
    t_val = stats.t.ppf(0.975, n-1)
    margin_error = t_val * (cv_std / np.sqrt(n))
    
    print(f"📊 Mathematical Analysis:")
    print(f"   F1 CV: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"   95% CI: [{cv_mean - margin_error:.4f}, {cv_mean + margin_error:.4f}]")
    print(f"   Precision: {precision:.4f}, Recall: {recall:.4f}")
    
    results.append({'Model': name, 'Metric': 'Accuracy', 'Score': acc})
    results.append({'Model': name, 'Metric': 'F1-score', 'Score': f1})
    results.append({'Model': name, 'Metric': 'Precision', 'Score': precision})
    results.append({'Model': name, 'Metric': 'Recall', 'Score': recall})
    results.append({'Model': name, 'Metric': 'F1-CV', 'Score': cv_mean})

    # Confusion matrix plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=model.classes_,
                yticklabels=model.classes_)
    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, f"{name}_confusion_matrix.png"))
    plt.close()

print("\n✅ All confusion matrices have been saved!")

# === 5. Enhanced Comparison Charts ===

results_df = pd.DataFrame(results)
print("\n📊 Overall model metrics:")
print(results_df)

# Create enhanced visualization with mathematical insights
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Comprehensive metrics comparison
sns.barplot(data=results_df, x='Model', y='Score', hue='Metric', ax=axes[0,0])
axes[0,0].set_title('🔬 Comprehensive Model Metrics')
axes[0,0].set_ylabel('Score')
axes[0,0].set_ylim(0, 1)
axes[0,0].tick_params(axis='x', rotation=45)
axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Add mathematical formula
axes[0,0].text(0.02, 0.98, 'F1 = 2×(P×R)/(P+R)', 
              transform=axes[0,0].transAxes, fontsize=10,
              bbox=dict(boxstyle='round', facecolor='wheat'))

# 2. Precision vs Recall scatter
model_names = results_df[results_df['Metric'] == 'Precision']['Model'].values
precisions = results_df[results_df['Metric'] == 'Precision']['Score'].values
recalls = results_df[results_df['Metric'] == 'Recall']['Score'].values

axes[0,1].scatter(precisions, recalls, s=100, alpha=0.7)
for i, model in enumerate(model_names):
    axes[0,1].annotate(model, (precisions[i], recalls[i]), 
                      xytext=(5, 5), textcoords='offset points')
axes[0,1].set_xlabel('Precision')
axes[0,1].set_ylabel('Recall')
axes[0,1].set_title('Precision vs Recall')
axes[0,1].grid(True, alpha=0.3)

# 3. F1-Score distribution (if we had CV data per model)
f1_scores = results_df[results_df['Metric'] == 'F1-score']['Score'].values
model_names_f1 = results_df[results_df['Metric'] == 'F1-score']['Model'].values

axes[1,0].bar(model_names_f1, f1_scores, alpha=0.7, color='skyblue')
axes[1,0].set_title('F1-Score Comparison')
axes[1,0].set_ylabel('F1-Score')
axes[1,0].tick_params(axis='x', rotation=45)
axes[1,0].grid(True, alpha=0.3)

# 4. Model ranking heatmap
metrics_pivot = results_df.pivot(index='Model', columns='Metric', values='Score')
sns.heatmap(metrics_pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[1,1])
axes[1,1].set_title('Model Performance Heatmap')

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "enhanced_model_comparison.png"), dpi=300, bbox_inches='tight')
plt.show()

# Mathematical summary
print("\n🧮 MATHEMATICAL SUMMARY")
print("=" * 40)
print("Formulas used:")
print("• F1-Score: F1 = 2 × (Precision × Recall) / (Precision + Recall)")
print("• Confidence Interval: CI = x̄ ± t₀.₀₂₅ × (s/√n)")
print("• Accuracy: (TP + TN) / (TP + TN + FP + FN)")

print(f"\n🎉 Enhanced reports with mathematical analysis saved in '{REPORTS_DIR}'")
print(f"📈 Generated: enhanced_model_comparison.png with 4 analytical views")
