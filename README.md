# DiabPred 🩺

[![CI](https://github.com/SchrDbb/diabpred/actions/workflows/ci.yml/badge.svg)](https://github.com/SchrDbb/diabpred/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)


**DiabPred** is an open-source Python toolkit for reproducible diabetes risk
prediction. It benchmarks seven classical machine learning classifiers on the
Pima Indians Diabetes Dataset and provides a clean API for predicting risk
from new patient measurements.

> **One command** reproduces the entire experiment, all figures, and all tables.

---

## Features

- 📊 **Unified benchmark** of 7 classifiers under identical conditions
- 🔬 **Reproducible experiments** (fixed seeds, `Dockerfile`, single-script runner)
- 🧩 **Clean prediction API** – input a patient dict, get a risk score
- 📈 **Automated figures** – ROC curves, PR curves, confusion matrices, feature importance
- ✅ **Full test suite** with pytest (45+ tests)
- 🐳 **Docker support** for containerised reproducibility
- 📝 **JOSS paper** included (`paper/paper.md`)

---

## Quick Start

### Installation

```bash
git clone https://github.com/SchrDbb/diabpred.git
cd diabpred
pip install -r requirements.txt
pip install -e .
```

### Reproduce the full experiment

```bash
python scripts/run_experiment.py
```

Outputs land in `outputs/`:
```
outputs/
├── models/          ← 7 trained .pkl files
├── figures/         ← ROC curves, PR curves, confusion matrix, feature importance
├── tables/          ← benchmark.csv, benchmark.tex
└── experiment_meta.json
```

### Predict for a new patient

```python
from diabpred.data import load_dataset, preprocess
from diabpred.models import train_model
from diabpred.predict import predict

df = load_dataset()
X_train, X_test, y_train, y_test, scaler = preprocess(df)
model = train_model("Random Forest", X_train, y_train)

result = predict(
    {
        "Pregnancies": 6,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50,
    },
    model,
    scaler,
)

print(result)
# {'prediction': 1, 'label': 'Diabetic', 'probability': 0.72, 'risk_level': 'High', 'threshold': 0.5}
```

See `examples/quick_prediction.py` for a complete runnable example.

---

## Benchmark Results

| Model | Accuracy | F1 | ROC-AUC | CV AUC (mean ± std) |
|---|---|---|---|---|
| Random Forest | 0.7922 | 0.7143 | 0.8412 | 0.836 ± 0.021 |
| Gradient Boosting | 0.7857 | 0.7042 | 0.8389 | 0.831 ± 0.019 |
| Logistic Regression | 0.7792 | 0.6923 | 0.8301 | 0.826 ± 0.022 |
| SVM | 0.7727 | 0.6800 | 0.8265 | 0.820 ± 0.024 |
| Naive Bayes | 0.7468 | 0.6452 | 0.8134 | 0.807 ± 0.025 |
| K-Nearest Neighbors | 0.7338 | 0.6296 | 0.7921 | 0.786 ± 0.029 |
| Decision Tree | 0.7208 | 0.6190 | 0.7201 | 0.718 ± 0.033 |

*Results with `random_state=42`, 80/20 split, 5-fold CV. Run `scripts/run_experiment.py` to reproduce.*

---

## API Reference

### `diabpred.data`
| Function | Description |
|---|---|
| `load_dataset(path=None)` | Load the Pima dataset (auto-downloads if needed) |
| `preprocess(df, test_size, random_state, ...)` | Impute → split → scale |
| `get_feature_names()` | Return ordered list of feature names |
| `describe_dataset(df)` | Summary statistics including zero counts |

### `diabpred.models`
| Function | Description |
|---|---|
| `list_available_models()` | List all 7 registered classifiers |
| `train_model(name, X, y)` | Fit a single model |
| `train_all_models(X, y, names, verbose)` | Fit all (or a subset of) models |
| `save_model(model, name, directory)` | Persist model to disk |
| `load_model(name, directory)` | Reload a saved model |

### `diabpred.predict`
| Function | Description |
|---|---|
| `predict(features, model, scaler, threshold)` | Single-patient prediction |
| `predict_proba(features, models, scaler)` | Probabilities from all models |
| `batch_predict(records, model, scaler)` | Predict for a list of patients |

### `diabpred.evaluate`
| Function | Description |
|---|---|
| `evaluate_model(model, X_test, y_test)` | Full metrics dict |
| `cross_validate_model(model, X, y, cv)` | Cross-validated metrics |
| `full_benchmark(models, X_train, X_test, y_train, y_test)` | Complete benchmark table |

### `diabpred.visualize`
| Function | Description |
|---|---|
| `plot_roc_curves(models, X_test, y_test)` | ROC curve overlay |
| `plot_confusion_matrix(model, X_test, y_test)` | Confusion matrix heatmap |
| `plot_feature_importance(model, names, X_test, y_test)` | Permutation importance |
| `plot_benchmark_comparison(df, metric)` | Bar chart comparison |
| `generate_all_figures(...)` | Generate and save all figures at once |

---

## Docker

```bash
docker build -t diabpred .
docker run --rm -v $(pwd)/outputs:/app/outputs diabpred
```

---

## Running Tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=diabpred --cov-report=term-missing
```

---

## Project Structure

```
diabpred/
├── diabpred/               ← Python package
│   ├── __init__.py
│   ├── data.py             ← Data loading and preprocessing
│   ├── models.py           ← Model registry and training
│   ├── evaluate.py         ← Evaluation metrics and benchmarking
│   ├── predict.py          ← Patient-facing prediction API
│   └── visualize.py        ← Figure generation pipeline
├── tests/
│   ├── conftest.py
│   └── test_diabpred.py    ← 45+ unit tests
├── scripts/
│   └── run_experiment.py   ← One-command experiment runner
├── examples/
│   └── quick_prediction.py ← Minimal usage example
├── paper/
│   ├── paper.md            ← JOSS manuscript
│   └── paper.bib           ← References
├── .github/workflows/
│   └── ci.yml              ← GitHub Actions CI
├── Dockerfile
├── requirements.txt
├── setup.py
├── CITATION.cff
├── CONTRIBUTING.md
└── LICENSE
```

---

## Citation

If you use DiabPred in your research, please cite:

```bibtex
@article{LastYear,
  title   = {DiabPred: An open-source machine learning toolkit for reproducible diabetes risk prediction},
  author  = {Last, First},
  journal = {Journal of Open Source Software},
  year    = {2025},
  doi     = {10.21105/joss.XXXXX}
}
```

Or use the `CITATION.cff` file:

```bash
cffconvert --format bibtex
```

---

## License

MIT — see [LICENSE](LICENSE).
