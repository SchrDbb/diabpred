"""
diabpred.evaluate
=================
Evaluation pipeline: per-model metrics and cross-validated benchmark table.

All metrics are standard clinical ML metrics. The full benchmark is what
powers the results table in the JOSS paper.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """Compute a comprehensive set of binary classification metrics.

    Parameters
    ----------
    model : fitted sklearn estimator
    X_test : np.ndarray
    y_test : np.ndarray

    Returns
    -------
    dict
        Keys: accuracy, precision, recall, f1, roc_auc, avg_precision, mcc,
              specificity, npv, tn, fp, fn, tp.

    Examples
    --------
    >>> from diabpred.data import load_dataset, preprocess
    >>> from diabpred.models import train_model
    >>> from diabpred.evaluate import evaluate_model
    >>> df = load_dataset()
    >>> X_train, X_test, y_train, y_test, scaler = preprocess(df)
    >>> model = train_model("Logistic Regression", X_train, y_train)
    >>> metrics = evaluate_model(model, X_test, y_test)
    >>> 0.0 <= metrics["roc_auc"] <= 1.0
    True
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "specificity": round(specificity, 4),
        "npv": round(npv, 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "avg_precision": round(average_precision_score(y_test, y_prob), 4),
        "mcc": round(matthews_corrcoef(y_test, y_pred), 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def cross_validate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    random_state: int = 42,
) -> Dict[str, float]:
    """Run stratified k-fold cross-validation and return mean ± std metrics.

    Parameters
    ----------
    model : unfitted sklearn estimator
    X : np.ndarray
    y : np.ndarray
    cv : int, optional
        Number of folds (default 5).
    random_state : int, optional

    Returns
    -------
    dict
        Keys: cv_accuracy_mean, cv_accuracy_std, cv_f1_mean, cv_f1_std,
              cv_roc_auc_mean, cv_roc_auc_std.
    """
    kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scoring = ["accuracy", "f1", "roc_auc"]

    results = cross_validate(model, X, y, cv=kfold, scoring=scoring, n_jobs=-1)

    return {
        "cv_accuracy_mean": round(results["test_accuracy"].mean(), 4),
        "cv_accuracy_std": round(results["test_accuracy"].std(), 4),
        "cv_f1_mean": round(results["test_f1"].mean(), 4),
        "cv_f1_std": round(results["test_f1"].std(), 4),
        "cv_roc_auc_mean": round(results["test_roc_auc"].mean(), 4),
        "cv_roc_auc_std": round(results["test_roc_auc"].std(), 4),
    }


def full_benchmark(
    trained_models: Dict[str, Any],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    cv: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """Produce the complete benchmark table for all trained models.

    Combines hold-out test metrics with cross-validation results into a
    single DataFrame suitable for publication.

    Parameters
    ----------
    trained_models : dict
        Output of :func:`~diabpred.models.train_all_models`.
    X_train, X_test, y_train, y_test : np.ndarray
    cv : int, optional
        Number of CV folds (default 5).
    verbose : bool, optional

    Returns
    -------
    pd.DataFrame
        One row per model, sorted by ROC-AUC descending.
    """
    from diabpred.models import get_model

    rows = []
    for name, model in trained_models.items():
        if verbose:
            print(f"  Evaluating {name}...", end=" ", flush=True)

        test_metrics = evaluate_model(model, X_test, y_test)

        # Cross-validate on the FULL dataset for the paper table
        X_full = np.vstack([X_train, X_test])
        y_full = np.concatenate([y_train, y_test])
        cv_metrics = cross_validate_model(
            get_model(name), X_full, y_full, cv=cv
        )

        row = {"Model": name, **test_metrics, **cv_metrics}
        rows.append(row)

        if verbose:
            print(f"AUC={test_metrics['roc_auc']:.4f}")

    df = pd.DataFrame(rows).set_index("Model")
    return df.sort_values("roc_auc", ascending=False)


def print_benchmark(df: pd.DataFrame) -> None:
    """Pretty-print the benchmark table to stdout."""
    display_cols = [
        "accuracy", "precision", "recall", "f1", "specificity",
        "roc_auc", "avg_precision", "mcc",
        "cv_roc_auc_mean", "cv_roc_auc_std",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    print(df[display_cols].to_string())
