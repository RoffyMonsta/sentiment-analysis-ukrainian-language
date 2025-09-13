#!/bin/bash

echo "🚀 Starting Enhanced Sentiment Analysis Pipeline"
echo "================================================"

# Activate venv
source venv/bin/activate

echo "✅ Virtual environment activated"

# 1. Generate features with neutral data
echo "📊 Step 1: Generating features with neutral data..."
python generate_features_csv.py

# 2. Train models
echo "🤖 Step 2: Training models..."
python train_models.py

# 3. Enhanced evaluation with mathematical analysis
echo "📈 Step 3: Enhanced evaluation with mathematical analysis..."
python evaluate_models.py

# 4. Enhanced prediction analysis
echo "🔍 Step 4: Enhanced prediction analysis..."
python predict_compare_real_text.py

echo "🎉 Enhanced pipeline completed!"
echo "📁 Check the 'reports' folder for enhanced visualizations"