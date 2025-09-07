import pandas as pd
import joblib
import os
from custom_sentiment import calculate_sentiment

# === 1. Config ===
MODELS_DIR = 'models'
MODELS = {
    "Random Forest": os.path.join(MODELS_DIR, "random_forest_model.pkl"),
    "SVM": os.path.join(MODELS_DIR, "svm_model.pkl"),
    "KNN": os.path.join(MODELS_DIR, "knn_model.pkl"),
    "Decision Tree": os.path.join(MODELS_DIR, "decision_tree_model.pkl"),
}

# === 2. Твій приклад тексту ===
sample_text = "Це було прекрасно, але трохи сумно 😢"

print(f"🔍 Input text: {sample_text}")

# === 3. Викликаємо кастомний аналайзер для фіч ===
result = calculate_sentiment(sample_text)
print("\n✅ Generated features from rule-based pipeline:")
print(result)

# === 4. Готуємо рядок фіч як DataFrame ===
features = pd.DataFrame([{
    'positive': result.get('positive', 0),
    'negative': result.get('negative', 0),
    'neutral': result.get('neutral', 0),
    'compound': result.get('compound', 0),
    'emoji_score': result.get('emoji_score', 0),
    'num_boosters': result.get('num_boosters', 0),
    'has_negation': result.get('has_negation', 0),
    'num_tokens': result.get('num_tokens', 0),
}])

print("\n🔍 Features prepared for ML models:")
print(features)

# === 5. Проганяємо через всі моделі ===
print("\n🔬 Predictions:")

for name, path in MODELS.items():
    model = joblib.load(path)
    pred = model.predict(features)[0]
    probas = None

    # Якщо модель підтримує predict_proba
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(features)
        class_probs = dict(zip(model.classes_, probas[0]))
        probas_str = ', '.join(f"{cls}: {p:.2f}" for cls, p in class_probs.items())
    else:
        probas_str = "(probabilities not available)"

    print(f"🗂️ {name}: {pred} | Probabilities: {probas_str}")
