import pandas as pd
import joblib
import os
import numpy as np
from scipy import stats
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
sample_text = "Це було прекрасно, але трохи сумно"

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

# === 5. Enhanced Mathematical Analysis ===
print("\n🧮 Mathematical Sentiment Analysis:")
print(f"Formula: {result.get('formula', 'S = Σ(w_i × s_i × p_i × b_i)')}")
print(f"Confidence: {result.get('confidence', 0.5):.4f}")
print(f"Entropy: {result.get('entropy', 1.0):.4f}")
print(f"Score Variance: {result.get('score_variance', 0.0):.4f}")
print(f"Compound Score: {result['compound']:.4f}")

# === 6. Enhanced Model Predictions with Statistical Analysis ===
print("\n🔬 Model Predictions with Statistical Analysis:")

predictions = []
confidences = []

for name, path in MODELS.items():
    try:
        model = joblib.load(path)
        pred = model.predict(features)[0]
        predictions.append(pred)
        
        # Calculate prediction confidence
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(features)[0]
            max_prob = np.max(probas)
            confidences.append(max_prob)
            
            class_probs = dict(zip(model.classes_, probas))
            probas_str = ', '.join(f"{cls}: {p:.3f}" for cls, p in class_probs.items())
            
            # Entropy-based uncertainty: H = -Σ(p_i * log(p_i))
            entropy = -np.sum(probas * np.log(probas + 1e-10))
            uncertainty = entropy / np.log(len(probas))  # Normalized
            
            print(f"🗂️ {name}:")
            print(f"   Prediction: {pred}")
            print(f"   Confidence: {max_prob:.3f}")
            print(f"   Uncertainty: {uncertainty:.3f}")
            print(f"   Probabilities: {probas_str}")
        else:
            print(f"🗂️ {name}: {pred} (probabilities not available)")
            confidences.append(0.5)  # Default confidence
    except FileNotFoundError:
        print(f"⚠️ Model {name} not found")
        continue

# === 7. Ensemble Analysis ===
if predictions:
    print("\n📊 Ensemble Statistical Analysis:")
    
    # Mode (most frequent prediction)
    from collections import Counter
    pred_counts = Counter(predictions)
    ensemble_pred = pred_counts.most_common(1)[0][0]
    consensus_ratio = pred_counts[ensemble_pred] / len(predictions)
    
    # Average confidence
    avg_confidence = np.mean(confidences) if confidences else 0
    confidence_std = np.std(confidences) if len(confidences) > 1 else 0
    
    print(f"Ensemble Prediction: {ensemble_pred}")
    print(f"Consensus Ratio: {consensus_ratio:.3f}")
    print(f"Average Confidence: {avg_confidence:.3f} ± {confidence_std:.3f}")
    
    # Agreement analysis
    agreement = len(set(predictions)) == 1
    print(f"Model Agreement: {'✅ Full' if agreement else '⚠️ Partial'}")
