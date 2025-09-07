import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# === 1. Параметри ===
DATASET_PATH = 'features_dataset.csv'
OUTPUT_DIR = 'models'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 2. Завантажуємо дані ===
print(f"🔍 Завантажую датасет: {DATASET_PATH}")
df = pd.read_csv(DATASET_PATH)
X = df.drop(columns=['label'])
y = df['label']

print(f"🗂️ X shape: {X.shape}, y distribution:\n{y.value_counts()}")

# === 3. Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}")

# === 4. Навчання моделей ===

def train_and_save_model(model, name):
    print(f"\n🚀 Training: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    model_path = os.path.join(OUTPUT_DIR, f"{name}_model.pkl")
    joblib.dump(model, model_path)
    print(f"✅ Збережено: {model_path}")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
train_and_save_model(rf, "random_forest")

# SVM
svm = SVC(kernel='rbf', probability=True, random_state=42)
train_and_save_model(svm, "svm")

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
train_and_save_model(knn, "knn")

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
train_and_save_model(dt, "decision_tree")

print("\n🎉 Всі моделі навчені та збережені в папці 'models'.")
