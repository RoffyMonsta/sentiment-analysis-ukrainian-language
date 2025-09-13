import pandas as pd
import joblib
import numpy as np
from custom_sentiment import calculate_sentiment

# Your sentence
text = "Це було прекрасно, але трохи сумно"

print(f"🔍 Analyzing: '{text}'")

# 1. Custom sentiment analysis
result = calculate_sentiment(text)
print(f"\n📊 Custom Analyzer Results:")
print(f"   Compound: {result['compound']:.4f}")
print(f"   Positive: {result['positive']:.4f}")
print(f"   Negative: {result['negative']:.4f}")
print(f"   Confidence: {result.get('confidence', 0.5):.4f}")

# 2. Prepare features for ML models
features = pd.DataFrame([{
    'positive': result['positive'],
    'negative': result['negative'], 
    'neutral': result['neutral'],
    'compound': result['compound'],
    'emoji_score': result['emoji_score'],
    'num_boosters': result['num_boosters'],
    'has_negation': result['has_negation'],
    'num_tokens': result['num_tokens']
}])

# 3. Load and test models
models = {
    "Random Forest": "models/random_forest_model.pkl",
    "SVM": "models/svm_model.pkl", 
    "KNN": "models/knn_model.pkl",
    "Decision Tree": "models/decision_tree_model.pkl"
}

print(f"\n🤖 ML Model Predictions:")
for name, path in models.items():
    try:
        model = joblib.load(path)
        pred = model.predict(features)[0]
        
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(features)[0]
            max_prob = np.max(probas)
            class_probs = dict(zip(model.classes_, probas))
            
            print(f"   {name}: {pred} (confidence: {max_prob:.3f})")
            for cls, prob in class_probs.items():
                print(f"      {cls}: {prob:.3f}")
        else:
            print(f"   {name}: {pred}")
    except FileNotFoundError:
        print(f"   {name}: Model not found")

print(f"\n💡 To test different sentences, edit the 'text' variable in this file")