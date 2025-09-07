import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)
from sklearn.model_selection import train_test_split

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
    results.append({
        'Model': name,
        'Metric': 'Accuracy',
        'Score': acc
    })
    results.append({
        'Model': name,
        'Metric': 'F1-score',
        'Score': f1
    })

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

# === 5. Comparison chart ===

# Results to DataFrame
results_df = pd.DataFrame(results)
print("\n📊 Overall model metrics:")
print(results_df)

# Grouped barplot for clearer comparison
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x='Model', y='Score', hue='Metric')
plt.title('🔬 Comparison of Models: Accuracy vs F1-score')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.legend(title='Metric')
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "model_comparison_grouped.png"))
plt.show()

print(f"\n🎉 Reports have been saved in '{REPORTS_DIR}'")
