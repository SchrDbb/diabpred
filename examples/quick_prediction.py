#!/usr/bin/env python3
"""
examples/quick_prediction.py
-----------------------------
Minimal working example: train a model and predict for a new patient.

Run from the repo root:
    python examples/quick_prediction.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from diabpred.data import load_dataset, preprocess
from diabpred.models import train_model
from diabpred.predict import predict, predict_proba
from diabpred.models import train_all_models

# ── 1. Load and preprocess ────────────────────────────────────────────────────
print("Loading dataset...")
df = load_dataset()
X_train, X_test, y_train, y_test, scaler = preprocess(df)

# ── 2. Train a single model ───────────────────────────────────────────────────
print("Training Random Forest...")
model = train_model("Random Forest", X_train, y_train)

# ── 3. Predict for a new patient ──────────────────────────────────────────────
patient = {
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50,
}

result = predict(patient, model, scaler)

print("\n── Single Model Prediction ──────────────────────────")
print(f"  Prediction  : {result['label']}")
print(f"  Probability : {result['probability']:.1%}")
print(f"  Risk Level  : {result['risk_level']}")

# ── 4. Ensemble probabilities from all models ─────────────────────────────────
print("\nTraining all models for ensemble view...")
all_models = train_all_models(X_train, y_train, verbose=False)
probs = predict_proba(patient, all_models, scaler)

print("\n── Ensemble Probabilities ───────────────────────────")
for model_name, prob in probs.items():
    bar = "█" * int(prob * 30)
    print(f"  {model_name:<28} {prob:.1%}  {bar}")
