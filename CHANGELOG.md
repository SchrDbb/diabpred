# Changelog

All notable changes to DiabPred are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [1.0.0] – 2025-01-01

### Added
- Initial public release
- `diabpred.data` — dataset loading (OpenML auto-download + local CSV), zero imputation, train/test split, StandardScaler pipeline
- `diabpred.models` — registry of 7 scikit-learn classifiers (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, k-NN, Naive Bayes); save/load via pickle
- `diabpred.evaluate` — 13-metric evaluation per model; 5-fold stratified cross-validation; `full_benchmark()` producing the paper's Table 1
- `diabpred.predict` — `predict()`, `predict_proba()`, `batch_predict()` with input validation and risk-level mapping
- `diabpred.visualize` — 5 auto-saved publication figures (ROC curves, PR curves, benchmark bar chart, confusion matrix, permutation feature importance)
- `diabpred.cli` — command-line interface with `predict`, `benchmark`, and `info` sub-commands
- `scripts/run_experiment.py` — single-command full experiment reproducer
- `scripts/tune_hyperparameters.py` — optional grid-search tuning
- `examples/quick_prediction.py` — minimal usage example
- `examples/web_app/app.py` — lightweight Flask prediction web interface
- `notebooks/01_full_walkthrough.ipynb` — Jupyter walkthrough notebook
- Full pytest test suite (31 tests across 5 test classes)
- GitHub Actions CI (Python 3.9–3.12, lint, experiment smoke test)
- Docker + docker-compose support
- JOSS paper (`paper/paper.md` + `paper/paper.bib`)
- `CITATION.cff`, `CONTRIBUTING.md`, `LICENSE` (MIT)
