# Installation Guide

## Requirements

- Python 3.9 or higher
- pip
- No GPU required
- ~200 MB disk space (including model files and figures)
- Internet access on first run only (to download the dataset)

---

## Option 1 — Standard install (recommended)

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/diabpred.git
cd diabpred

# 2. Create a virtual environment (strongly recommended)
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows PowerShell

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the package in editable mode
pip install -e .

# 5. Verify the installation
python -c "import diabpred; print(diabpred.__version__)"
```

Expected output: `1.0.0`

---

## Option 2 — Docker (fully isolated, no Python setup needed)

```bash
# Build and run the benchmark experiment
docker build -t diabpred .
docker run --rm -v $(pwd)/outputs:/app/outputs diabpred

# Start the web interface
docker compose up webapp
# Then open: http://localhost:5000
```

---

## Option 3 — Google Colab

Open `notebooks/01_full_walkthrough.ipynb` and uncomment the first cell:

```python
!git clone https://github.com/yourusername/diabpred.git
%cd diabpred
!pip install -e . -q
```

---

## Providing the dataset

The Pima Indians Diabetes Dataset is **downloaded automatically** from
OpenML on the first run. If you are offline, download it manually:

1. Go to https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
2. Download `diabetes.csv`
3. Place it in the `data/` directory at the project root

```
diabpred/
└── data/
    └── diabetes.csv   ← place it here
```

---

## Running tests

```bash
# All tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=diabpred --cov-report=term-missing

# Single test class
pytest tests/test_diabpred.py::TestPredict -v
```

---

## Reproducing the paper results

```bash
python scripts/run_experiment.py
```

All outputs appear in `outputs/`:

```
outputs/
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── ...
├── figures/
│   ├── roc_curves.png
│   ├── pr_curves.png
│   ├── benchmark_comparison.png
│   ├── confusion_matrix_random_forest.png
│   └── feature_importance_random_forest.png
├── tables/
│   ├── benchmark.csv
│   └── benchmark.tex
└── experiment_meta.json
```

---

## Using the CLI

```bash
# Predict for a patient
python -m diabpred.cli predict \
    --pregnancies 6 --glucose 148 --bp 72 --skin 35 \
    --insulin 0 --bmi 33.6 --dpf 0.627 --age 50

# Show all available models and features
python -m diabpred.cli info

# Run benchmark via CLI
python -m diabpred.cli benchmark --cv 10
```

---

## Using the web interface

```bash
pip install flask
python examples/web_app/app.py
# Open: http://localhost:5000
```

---

## Common installation problems

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'diabpred'` | Run `pip install -e .` from the repo root |
| `sklearn.datasets.fetch_openml` timeout | Download the CSV manually (see above) |
| Matplotlib `_tkinter` error | Already handled: Agg backend is set automatically |
| `pytest: command not found` | Run `pip install pytest pytest-cov` |
| Windows path issues | Use `\` instead of `/` in paths, or use WSL |
