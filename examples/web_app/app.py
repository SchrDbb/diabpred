"""
examples/web_app/app.py
=======================
Minimal Flask web interface for DiabPred.

Lets non-technical users enter patient measurements in a browser form
and instantly see a prediction.

Usage
-----
    cd examples/web_app
    pip install flask
    python app.py
    # Open: http://localhost:5000

This is an OPTIONAL component. The core diabpred package does not depend
on Flask. Install it separately: pip install flask
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow importing diabpred from repo root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from flask import Flask, jsonify, render_template_string, request
except ImportError:
    print("Flask is not installed. Run: pip install flask")
    sys.exit(1)

from diabpred.data import load_dataset, preprocess
from diabpred.models import load_model, save_model, train_model
from diabpred.predict import predict, predict_proba
from diabpred.models import train_all_models

app = Flask(__name__)

# ── Bootstrap model on startup ────────────────────────────────────────────────
_MODELS_DIR = Path("models")
_MODEL_NAME = "Random Forest"

def _ensure_model():
    """Train and save the default model if not already saved."""
    try:
        model = load_model(_MODEL_NAME, _MODELS_DIR)
        print(f"[startup] Loaded saved model: {_MODEL_NAME}")
        return model
    except FileNotFoundError:
        print(f"[startup] Training {_MODEL_NAME}...")
        df = load_dataset()
        X_train, _, y_train, _, _ = preprocess(df)
        model = train_model(_MODEL_NAME, X_train, y_train)
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        save_model(model, _MODEL_NAME, _MODELS_DIR)
        return model

def _get_scaler():
    df = load_dataset()
    _, _, _, _, scaler = preprocess(df)
    return scaler

_model = _ensure_model()
_scaler = _get_scaler()


# ── HTML template (self-contained, no external assets) ───────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DiabPred – Diabetes Risk Predictor</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #f5f5f4; color: #1c1917; min-height: 100vh; padding: 2rem 1rem; }
  .container { max-width: 640px; margin: 0 auto; }
  h1 { font-size: 1.6rem; font-weight: 600; margin-bottom: 0.25rem; }
  .subtitle { color: #78716c; font-size: 0.9rem; margin-bottom: 2rem; }
  .card { background: #fff; border-radius: 12px; border: 1px solid #e7e5e4;
          padding: 1.5rem; margin-bottom: 1.25rem; }
  .card h2 { font-size: 1rem; font-weight: 600; margin-bottom: 1rem; color: #44403c; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  label { display: block; font-size: 0.8rem; color: #78716c;
          font-weight: 500; margin-bottom: 4px; }
  input[type=number] { width: 100%; padding: 8px 10px; border: 1px solid #d6d3d1;
    border-radius: 8px; font-size: 0.9rem; background: #fafaf9;
    transition: border-color 0.15s; }
  input[type=number]:focus { outline: none; border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.15); }
  .hint { font-size: 0.7rem; color: #a8a29e; margin-top: 2px; }
  button { width: 100%; padding: 12px; background: #3b82f6; color: #fff;
    border: none; border-radius: 10px; font-size: 1rem; font-weight: 600;
    cursor: pointer; transition: background 0.15s; }
  button:hover { background: #2563eb; }
  button:active { transform: scale(0.99); }
  #result { display: none; }
  .result-header { display: flex; align-items: center; gap: 12px; margin-bottom: 1rem; }
  .result-label { font-size: 1.4rem; font-weight: 700; }
  .result-label.diabetic { color: #dc2626; }
  .result-label.healthy  { color: #16a34a; }
  .badge { padding: 4px 12px; border-radius: 99px; font-size: 0.8rem; font-weight: 600; }
  .badge.high     { background: #fee2e2; color: #991b1b; }
  .badge.moderate { background: #fef3c7; color: #92400e; }
  .badge.low      { background: #dcfce7; color: #166534; }
  .prob-bar-wrap { margin: 1rem 0; }
  .prob-label { font-size: 0.85rem; color: #78716c; margin-bottom: 6px; }
  .prob-bar { height: 10px; background: #f5f5f4; border-radius: 99px; overflow: hidden; }
  .prob-fill { height: 100%; border-radius: 99px; transition: width 0.5s ease;
    background: linear-gradient(to right, #3b82f6, #dc2626); }
  .model-row { display: flex; justify-content: space-between; align-items: center;
    padding: 6px 0; border-bottom: 1px solid #f5f5f4; font-size: 0.85rem; }
  .model-row:last-child { border-bottom: none; }
  .model-name { color: #44403c; }
  .model-prob { font-weight: 600; color: #1c1917; }
  .mini-bar { height: 6px; background: #f0fdf4; border-radius: 99px;
    flex: 1; margin: 0 10px; overflow: hidden; }
  .mini-fill { height: 100%; background: #22c55e; border-radius: 99px; }
  .error { color: #dc2626; font-size: 0.85rem; margin-top: 8px; }
  .loading { color: #3b82f6; font-size: 0.85rem; text-align: center; padding: 1rem; }
</style>
</head>
<body>
<div class="container">
  <h1>🩺 DiabPred</h1>
  <p class="subtitle">Enter patient measurements below to assess diabetes risk.</p>

  <div class="card">
    <h2>Patient measurements</h2>
    <div class="grid">
      <div>
        <label for="pregnancies">Pregnancies</label>
        <input type="number" id="pregnancies" min="0" max="20" step="1" value="2">
        <div class="hint">Number of times pregnant</div>
      </div>
      <div>
        <label for="glucose">Glucose (mg/dL)</label>
        <input type="number" id="glucose" min="44" max="199" step="1" value="120">
        <div class="hint">2-hour plasma glucose</div>
      </div>
      <div>
        <label for="bp">Blood Pressure (mmHg)</label>
        <input type="number" id="bp" min="24" max="122" step="1" value="70">
        <div class="hint">Diastolic blood pressure</div>
      </div>
      <div>
        <label for="skin">Skin Thickness (mm)</label>
        <input type="number" id="skin" min="0" max="99" step="1" value="20">
        <div class="hint">Triceps skinfold thickness</div>
      </div>
      <div>
        <label for="insulin">Insulin (μU/mL)</label>
        <input type="number" id="insulin" min="0" max="846" step="1" value="80">
        <div class="hint">2-hour serum insulin</div>
      </div>
      <div>
        <label for="bmi">BMI (kg/m²)</label>
        <input type="number" id="bmi" min="15" max="67" step="0.1" value="28.0">
        <div class="hint">Body mass index</div>
      </div>
      <div>
        <label for="dpf">Diabetes Pedigree</label>
        <input type="number" id="dpf" min="0.078" max="2.42" step="0.001" value="0.300">
        <div class="hint">Diabetes pedigree function</div>
      </div>
      <div>
        <label for="age">Age (years)</label>
        <input type="number" id="age" min="21" max="81" step="1" value="35">
        <div class="hint">Patient age</div>
      </div>
    </div>
    <div style="margin-top:1.25rem;">
      <button onclick="runPrediction()">Predict Diabetes Risk</button>
      <div id="error" class="error"></div>
    </div>
  </div>

  <div class="card" id="result">
    <div class="result-header">
      <div class="result-label" id="result-label">—</div>
      <div class="badge" id="risk-badge">—</div>
    </div>
    <div class="prob-bar-wrap">
      <div class="prob-label">Probability of diabetes: <strong id="prob-pct">—</strong></div>
      <div class="prob-bar"><div class="prob-fill" id="prob-fill" style="width:0%"></div></div>
    </div>
    <h2 style="margin-top:1rem;">All models</h2>
    <div id="model-list"></div>
  </div>
</div>

<script>
async function runPrediction() {
  document.getElementById('error').textContent = '';
  document.getElementById('result').style.display = 'none';

  const payload = {
    pregnancies: +document.getElementById('pregnancies').value,
    glucose:     +document.getElementById('glucose').value,
    bp:          +document.getElementById('bp').value,
    skin:        +document.getElementById('skin').value,
    insulin:     +document.getElementById('insulin').value,
    bmi:         +document.getElementById('bmi').value,
    dpf:         +document.getElementById('dpf').value,
    age:         +document.getElementById('age').value,
  };

  try {
    const resp = await fetch('/api/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    const data = await resp.json();
    if (!resp.ok) { document.getElementById('error').textContent = data.error; return; }

    // Main result
    const lbl = document.getElementById('result-label');
    lbl.textContent = data.label;
    lbl.className = 'result-label ' + (data.prediction === 1 ? 'diabetic' : 'healthy');

    const badge = document.getElementById('risk-badge');
    badge.textContent = data.risk_level + ' risk';
    badge.className = 'badge ' + data.risk_level.toLowerCase();

    document.getElementById('prob-pct').textContent = (data.probability * 100).toFixed(1) + '%';
    document.getElementById('prob-fill').style.width = (data.probability * 100) + '%';

    // Model list
    const list = document.getElementById('model-list');
    list.innerHTML = '';
    const ensemble = data.all_probs.ensemble_mean;
    Object.entries(data.all_probs).forEach(([name, prob]) => {
      if (name === 'ensemble_mean') return;
      const row = document.createElement('div');
      row.className = 'model-row';
      row.innerHTML = `
        <span class="model-name">${name}</span>
        <div class="mini-bar"><div class="mini-fill" style="width:${prob*100}%"></div></div>
        <span class="model-prob">${(prob*100).toFixed(1)}%</span>`;
      list.appendChild(row);
    });
    const ens = document.createElement('div');
    ens.className = 'model-row';
    ens.style.fontWeight = '600';
    ens.innerHTML = `<span class="model-name">Ensemble mean</span>
      <div class="mini-bar"><div class="mini-fill" style="width:${ensemble*100}%;background:#3b82f6"></div></div>
      <span class="model-prob">${(ensemble*100).toFixed(1)}%</span>`;
    list.appendChild(ens);

    document.getElementById('result').style.display = 'block';
  } catch(e) {
    document.getElementById('error').textContent = 'Prediction failed: ' + e.message;
  }
}
</script>
</body>
</html>
"""


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """REST endpoint: POST JSON → prediction + all model probabilities."""
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON body received"}), 400

    patient = {
        "Pregnancies":              float(data.get("pregnancies", 0)),
        "Glucose":                  float(data.get("glucose", 0)),
        "BloodPressure":            float(data.get("bp", 0)),
        "SkinThickness":            float(data.get("skin", 0)),
        "Insulin":                  float(data.get("insulin", 0)),
        "BMI":                      float(data.get("bmi", 0)),
        "DiabetesPedigreeFunction": float(data.get("dpf", 0)),
        "Age":                      float(data.get("age", 0)),
    }

    try:
        result = predict(patient, _model, _scaler)
        all_probs = predict_proba(patient, _get_all_models(), _scaler)
        result["all_probs"] = all_probs
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422
    except Exception as exc:
        return jsonify({"error": f"Prediction error: {exc}"}), 500


def _get_all_models():
    """Lazily train all models on first ensemble call."""
    if not hasattr(_get_all_models, "_cache"):
        df = load_dataset()
        X_train, _, y_train, _, _ = preprocess(df)
        _get_all_models._cache = train_all_models(X_train, y_train, verbose=False)
    return _get_all_models._cache


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "version": "1.0.0"})


if __name__ == "__main__":
    print("\n  DiabPred Web App")
    print("  Open: http://localhost:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
