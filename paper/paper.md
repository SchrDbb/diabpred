---
title: 'DiabPred: An Open-Source Machine Learning Toolkit for Reproducible Diabetes Risk Prediction'
tags:
  - Python
  - diabetes
  - machine learning
  - clinical decision support
  - scikit-learn
  - reproducible research
authors:
  - name: Dassi Bopda Blondel Christian
    orcid: https://orcid.org/0009-0002-6694-7611
    affiliation: 1, 2, 3
affiliations:
  - name: DirimSi Institute, Cameroon
    index: 1
- name: University of Buea, Cameroon
index: 2
- name: University of Cheikh Anta Diop, Senegal
index: 3
date: 3 January 2026
bibliography: paper.bib
---

# Summary

Diabetes mellitus is a chronic metabolic condition affecting over 500 million
people worldwide and represents a leading cause of cardiovascular disease,
renal failure, and premature mortality [@WHO2023]. Early identification of
at-risk individuals enables preventive interventions that significantly reduce
disease progression and associated healthcare costs. Machine learning (ML)
approaches have demonstrated strong performance in predicting diabetes onset
from routinely collected clinical measurements, yet existing published
implementations are fragmented, difficult to reproduce, and lack standardised
evaluation protocols.

`DiabPred` is an open-source Python toolkit that provides a clean,
well-documented, and fully reproducible benchmark of seven classical ML
classifiers for diabetes risk prediction. It is designed to serve both
researchers who wish to reproduce published results and practitioners who
need a reliable prediction API for integration into clinical workflows.
The entire benchmark — from data loading through model training, evaluation,
and figure generation — is reproducible with a single command. Experiments
were conducted on the Pima Indians Diabetes Dataset (768 patients, 8 features),
with Gradient Boosting achieving the best held-out ROC-AUC of 0.8304 and
Random Forest achieving the best cross-validated ROC-AUC of 0.8249 ± 0.0223.

# Statement of Need

Several studies have applied ML to diabetes prediction [@Sneha2019;
@Zou2018; @Kavakiotis2017], but their implementations are rarely released
as reusable software. When code is released, it typically exists as a
Jupyter notebook without a formal package structure, automated tests, or
installation instructions, making independent verification and downstream
adoption difficult. This reproducibility gap is well-documented in
computational science and represents a significant barrier to cumulative
progress [@Stodden2016].

`DiabPred` addresses this gap by providing:

- **A unified benchmark** comparing seven classifiers (Logistic Regression,
  Decision Tree, Random Forest, Gradient Boosting, SVM, k-NN, Naive Bayes)
  under strictly identical experimental conditions — same random seed (42),
  same 80/20 stratified split, same preprocessing pipeline, same evaluation
  metrics.
- **A clean prediction API** (`predict`, `predict_proba`, `batch_predict`)
  that accepts a dictionary of patient measurements and returns a risk label,
  probability score, and a three-tier risk level (Low / Moderate / High).
- **Automated evaluation figures** including overlaid ROC curves,
  precision-recall curves, confusion matrices, and permutation feature
  importance plots, all generated and saved by a single script call.
- **Full reproducibility** via fixed random seeds, a `Dockerfile`,
  a `docker-compose.yml`, and a `run_experiment.py` entry point that
  regenerates every result from scratch on any platform.
- **An installation-tested Python package** with 31 unit tests across
  five test classes, covering all public API functions, and a GitHub
  Actions CI pipeline tested on Python 3.9 through 3.12.

The software targets three primary audiences: (1) clinical informatics
researchers who need a reproducible baseline for benchmarking new diabetes
prediction algorithms; (2) medical data scientists who need a validated
prediction API for integration into clinical decision-support prototypes;
and (3) educators teaching ML in healthcare contexts who need a
well-documented, beginner-friendly reference implementation.

# Methods

## Dataset

`DiabPred` uses the Pima Indians Diabetes Dataset [@Smith1988], a widely
cited benchmark containing 768 female patients of Pima Indian heritage,
each described by eight clinical features: number of pregnancies,
plasma glucose concentration (2-hour oral glucose tolerance test), diastolic
blood pressure (mmHg), triceps skinfold thickness (mm), 2-hour serum insulin
(μU/mL), body mass index (BMI, kg/m²), diabetes pedigree function, and age
(years). The binary outcome indicates whether the patient was diagnosed
with diabetes within five years. The dataset exhibits moderate class
imbalance, with 268 positive cases (34.9%) and 500 negative cases (65.1%).

The dataset is automatically downloaded from OpenML (dataset ID 37)
[@Vanschoren2014] on first use and cached locally at `data/diabetes.csv`.

## Preprocessing

Five features — Glucose, BloodPressure, SkinThickness, Insulin, and BMI —
contain biologically impossible zero values that encode missing measurements
(together accounting for 652 zero entries across the dataset).
Following standard practice [@Naz2020], these are replaced with the
per-column median computed from non-zero observations prior to splitting.
Features are then standardised using z-score normalisation (zero mean,
unit variance) via `sklearn.preprocessing.StandardScaler` [@Pedregosa2011].
The scaler is fitted exclusively on the training set and applied to the
test set, preventing data leakage. The dataset is partitioned into 80%
training (614 samples) and 20% test (154 samples) sets using stratified
random sampling (`random_state=42`) to preserve the class imbalance ratio
in both splits.

## Classifiers

Seven classifiers are evaluated using default scikit-learn hyperparameters
[@Pedregosa2011], which represent reproducible baselines without
dataset-specific tuning:

| Classifier | Key hyperparameters |
|---|---|
| Logistic Regression | `max_iter=1000`, `solver=lbfgs` |
| Decision Tree | `random_state=42` |
| Random Forest | `n_estimators=100`, `random_state=42` |
| Gradient Boosting | `n_estimators=100`, `random_state=42` |
| Support Vector Machine | `kernel=rbf`, `probability=True`, `random_state=42` |
| k-Nearest Neighbours | `n_neighbors=5` |
| Gaussian Naive Bayes | default |

## Evaluation

Each model is evaluated on the held-out test set (154 samples) using
nine metrics: accuracy, precision, recall (sensitivity), F1-score,
specificity, negative predictive value (NPV), ROC-AUC, average precision
(AP-AUC), and Matthews Correlation Coefficient (MCC). Confusion matrix
components (TP, TN, FP, FN) are also recorded.

In addition, 5-fold stratified cross-validation is performed on the full
dataset (768 samples) to report mean ± standard deviation for accuracy,
F1-score, and ROC-AUC, providing performance estimates less sensitive to
the particular train-test split.

Feature importances are computed using permutation importance
[@Breiman2001] with ROC-AUC as the scoring function (10 repeats,
`random_state=42`), providing a model-agnostic comparison applicable
to all seven classifiers.

# Results

## Benchmark performance

Table 1 presents the complete benchmark results on the held-out test set
(n = 154) together with 5-fold cross-validated ROC-AUC. All results were
obtained with `random_state=42` and are fully reproducible by running
`python scripts/run_experiment.py`.

**Table 1.** Classification performance of all seven models on the held-out
test set (n = 154) and 5-fold cross-validation (CV) on the full dataset
(n = 768). Models are ranked by test-set ROC-AUC.

| Model | Accuracy | Precision | Recall | F1 | Specificity | ROC-AUC | AP-AUC | MCC | CV AUC (mean ± std) |
|---|---|---|---|---|---|---|---|---|---|
| Gradient Boosting | 0.7597 | 0.6889 | 0.5741 | 0.6263 | 0.8600 | **0.8304** | 0.7199 | 0.4555 | 0.8244 ± 0.0198 |
| Random Forest | 0.7792 | 0.7174 | 0.6111 | 0.6600 | 0.8700 | 0.8179 | 0.6939 | 0.5016 | 0.8249 ± 0.0223 |
| Logistic Regression | 0.7078 | 0.6000 | 0.5000 | 0.5455 | 0.8200 | 0.8130 | 0.6733 | 0.3358 | **0.8356 ± 0.0248** |
| SVM | 0.7403 | 0.6522 | 0.5556 | 0.6000 | 0.8400 | 0.7964 | 0.6615 | 0.4124 | 0.8250 ± 0.0197 |
| K-Nearest Neighbors | 0.7532 | 0.6600 | 0.6111 | 0.6346 | 0.8300 | 0.7886 | 0.6393 | 0.4495 | 0.7787 ± 0.0203 |
| Naive Bayes | 0.7013 | 0.5667 | 0.6296 | 0.5965 | 0.7400 | 0.7646 | 0.5961 | 0.3617 | 0.8145 ± 0.0269 |
| Decision Tree | 0.6818 | 0.5532 | 0.4815 | 0.5149 | 0.7900 | 0.6357 | 0.4482 | 0.2813 | 0.6684 ± 0.0271 |

Gradient Boosting achieved the highest test-set ROC-AUC (0.8304), while
Logistic Regression achieved the highest cross-validated ROC-AUC
(0.8356 ± 0.0248), indicating that the simpler linear model generalises
most consistently across folds despite being outperformed on the single
test split. Random Forest achieved the best accuracy (0.7792) and MCC
(0.5016), indicating the most balanced overall performance across both
classes. The Decision Tree showed the weakest performance (ROC-AUC =
0.6357), consistent with its susceptibility to overfitting without
ensemble averaging.

## Confusion matrix analysis

On the test set of 154 patients (100 non-diabetic, 54 diabetic), the
best-performing model (Gradient Boosting) produced 86 true negatives,
14 false positives, 23 false negatives, and 31 true positives. Random
Forest produced 87 true negatives, 13 false positives, 21 false negatives,
and 33 true positives — a slightly lower false-negative count, which is
clinically preferable as it reduces missed diagnoses.

## Feature importance

Permutation importance analysis on the best-performing model identified
plasma glucose concentration as the most informative feature for diabetes
prediction, followed by BMI and age. This is consistent with established
clinical knowledge that hyperglycaemia is the defining characteristic of
diabetes, and with findings reported by @Zou2018 and @Sneha2019 on the
same dataset.

## Figures

The following figures are generated automatically by `run_experiment.py`
and saved to `outputs/figures/`.

![ROC curves for all seven classifiers on the held-out test set. Gradient
Boosting achieves the highest AUC of 0.830.](../outputs/figures/roc_curves.png)

![Precision-recall curves for all seven classifiers. Gradient Boosting
achieves the highest average precision of 0.720.](../outputs/figures/pr_curves.png)

![Model comparison bar chart ranked by ROC-AUC.](../outputs/figures/benchmark_comparison.png)

![Confusion matrix for Gradient Boosting (best model by test-set
ROC-AUC).](../outputs/figures/confusion_matrix_gradient_boosting.png)

![Permutation feature importance for Gradient Boosting. Glucose and BMI
are the two most informative
features.](../outputs/figures/feature_importance_gradient_boosting.png)

## Reproducibility

The complete experiment is reproducible with a single command:

```bash
python scripts/run_experiment.py
```

All outputs — trained model files (`.pkl`), evaluation figures (`.png`),
benchmark table (`.csv` and `.tex`), and experiment metadata (`.json`) —
are saved to the `outputs/` directory. The random seed is fixed at 42
throughout; alternative seeds can be passed via `--seed`. A Docker
container is provided for platform-independent reproducibility:

```bash
docker build -t diabpred . && docker run --rm \
  -v $(pwd)/outputs:/app/outputs diabpred
```

# Acknowledgements

The authors thank the UCI Machine Learning Repository and OpenML for
providing free access to the Pima Indians Diabetes Dataset, and the
scikit-learn development team for the machine learning infrastructure
on which `DiabPred` is built.

# References
