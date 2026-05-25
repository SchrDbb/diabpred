#!/usr/bin/env python3
"""
scripts/tune_hyperparameters.py
================================
Grid search hyperparameter tuning for all DiabPred classifiers.

This script is OPTIONAL — it shows how to go beyond default hyperparameters.
Results can be compared against the default benchmark in the paper.

Usage
-----
    python scripts/tune_hyperparameters.py
    python scripts/tune_hyperparameters.py --model "Random Forest"
    python scripts/tune_hyperparameters.py --cv 10 --output-dir tuned/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from diabpred.data import load_dataset, preprocess
from diabpred.evaluate import evaluate_model

# ── Parameter grids ────────────────────────────────────────────────────────────
PARAM_GRIDS = {
    "Logistic Regression": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "solver": ["lbfgs", "liblinear"],
        "max_iter": [500, 1000],
    },
    "Decision Tree": {
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
        "criterion": ["gini", "entropy"],
    },
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
    },
    "Gradient Boosting": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [3, 5],
    },
    "SVM": {
        "C": [0.1, 1.0, 10.0],
        "gamma": ["scale", "auto"],
        "kernel": ["rbf", "linear"],
    },
    "K-Nearest Neighbors": {
        "n_neighbors": [3, 5, 7, 11],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    },
    "Naive Bayes": {
        "var_smoothing": [1e-9, 1e-8, 1e-7],
    },
}


def tune_model(name: str, X_train, y_train, X_test, y_test, cv: int = 5) -> dict:
    """Run grid search for a single model and return results."""
    from diabpred.models import get_model

    model = get_model(name)
    grid = PARAM_GRIDS.get(name, {})
    if not grid:
        print(f"  No param grid for {name}, skipping.")
        return {}

    kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    search = GridSearchCV(
        model, grid, cv=kfold,
        scoring="roc_auc", n_jobs=-1, verbose=0
    )
    t0 = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - t0

    best = search.best_estimator_
    metrics = evaluate_model(best, X_test, y_test)

    return {
        "model": name,
        "best_params": search.best_params_,
        "cv_roc_auc": round(search.best_score_, 4),
        "test_roc_auc": metrics["roc_auc"],
        "test_f1": metrics["f1"],
        "test_accuracy": metrics["accuracy"],
        "n_combinations": len(search.cv_results_["params"]),
        "elapsed_seconds": round(elapsed, 2),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Tune DiabPred hyperparameters with grid search.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=None,
                        help="Tune a single model by name (default: all)")
    parser.add_argument("--cv", type=int, default=5,
                        help="Cross-validation folds")
    parser.add_argument("--output-dir", default="outputs/tuning", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("DiabPred – Hyperparameter Tuning")
    print("=" * 55)

    df = load_dataset()
    X_train, X_test, y_train, y_test, scaler = preprocess(df, random_state=args.seed)

    names = [args.model] if args.model else list(PARAM_GRIDS.keys())
    results = []

    for name in names:
        print(f"\n[tuning] {name}...")
        r = tune_model(name, X_train, y_train, X_test, y_test, cv=args.cv)
        if r:
            results.append(r)
            print(f"  Best params  : {r['best_params']}")
            print(f"  CV AUC       : {r['cv_roc_auc']:.4f}")
            print(f"  Test AUC     : {r['test_roc_auc']:.4f}")
            print(f"  Time         : {r['elapsed_seconds']}s  ({r['n_combinations']} combos)")

    # Save results
    out_path = args.output_dir / "tuning_results.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
