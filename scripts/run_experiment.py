#!/usr/bin/env python3
"""
Run the complete DiabPred experiment pipeline.

This single script:
1. Loads and preprocesses the dataset
2. Trains all 7 classifiers
3. Evaluates on a held-out test set + 5-fold CV
4. Saves trained models to disk
5. Generates all evaluation figures
6. Exports the benchmark table as CSV and LaTeX

Usage
-----
    python scripts/run_experiment.py
    python scripts/run_experiment.py --output-dir results/ --cv 10
    python scripts/run_experiment.py --no-figures

Running this script produces a fully reproducible experiment.
Set the DIABPRED_SEED environment variable to change the random seed.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from diabpred.data import get_feature_names, load_dataset, preprocess
from diabpred.evaluate import full_benchmark, print_benchmark
from diabpred.models import list_available_models, save_model, train_all_models
from diabpred.visualize import generate_all_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the DiabPred benchmark experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-path", default=None,
        help="Path to diabetes CSV. Auto-downloads if not supplied."
    )
    parser.add_argument(
        "--output-dir", default="outputs", type=Path,
        help="Root directory for models, figures, and tables."
    )
    parser.add_argument(
        "--test-size", default=0.2, type=float,
        help="Fraction of data held out for testing."
    )
    parser.add_argument(
        "--cv", default=5, type=int,
        help="Number of cross-validation folds."
    )
    parser.add_argument(
        "--seed", default=int(os.environ.get("DIABPRED_SEED", 42)), type=int,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--no-figures", action="store_true",
        help="Skip figure generation (useful in headless CI)."
    )
    parser.add_argument(
        "--models-dir", default=None, type=Path,
        help="Where to save trained models (default: <output-dir>/models)."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir: Path = args.output_dir
    models_dir: Path = args.models_dir or output_dir / "models"
    figures_dir: Path = output_dir / "figures"
    tables_dir: Path = output_dir / "tables"

    for d in [output_dir, models_dir, figures_dir, tables_dir]:
        d.mkdir(parents=True, exist_ok=True)

    separator = "=" * 60

    print(separator)
    print("DiabPred – Diabetes Prediction Benchmark")
    print(separator)
    print(f"  Seed        : {args.seed}")
    print(f"  Test size   : {args.test_size}")
    print(f"  CV folds    : {args.cv}")
    print(f"  Output dir  : {output_dir}")
    print(separator)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("\n[1/5] Loading and preprocessing data...")
    t0 = time.time()
    df = load_dataset(args.data_path)
    X_train, X_test, y_train, y_test, scaler = preprocess(
        df, test_size=args.test_size, random_state=args.seed
    )
    print(f"  Dataset     : {df.shape[0]} samples, {df.shape[1] - 1} features")
    print(f"  Train / Test: {len(X_train)} / {len(X_test)}")
    print(f"  Class balance (train): {y_train.mean():.1%} positive")
    print(f"  Done in {time.time() - t0:.1f}s")

    # ── 2. Training ───────────────────────────────────────────────────────────
    print(f"\n[2/5] Training {len(list_available_models())} models...")
    t0 = time.time()
    trained_models = train_all_models(X_train, y_train, verbose=True)
    print(f"  Done in {time.time() - t0:.1f}s")

    # ── 3. Save models ────────────────────────────────────────────────────────
    print("\n[3/5] Saving trained models...")
    for name, model in trained_models.items():
        path = save_model(model, name, models_dir)
        print(f"  Saved: {path}")

    # ── 4. Benchmark ──────────────────────────────────────────────────────────
    print(f"\n[4/5] Benchmarking (test set + {args.cv}-fold CV)...")
    t0 = time.time()
    bench_df = full_benchmark(
        trained_models, X_train, X_test, y_train, y_test,
        cv=args.cv, verbose=True
    )
    print(f"  Done in {time.time() - t0:.1f}s\n")
    print_benchmark(bench_df)

    # Save tables
    csv_path = tables_dir / "benchmark.csv"
    bench_df.to_csv(csv_path)
    print(f"\n  Table saved: {csv_path}")

    try:
        latex_path = tables_dir / "benchmark.tex"
        latex_cols = ["accuracy", "f1", "roc_auc", "cv_roc_auc_mean", "cv_roc_auc_std"]
        latex_cols = [c for c in latex_cols if c in bench_df.columns]
        bench_df[latex_cols].to_latex(latex_path, float_format="%.4f")
        print(f"  LaTeX table: {latex_path}")
    except Exception:
        pass

    # Save run metadata
    meta = {
        "seed": args.seed,
        "test_size": args.test_size,
        "cv_folds": args.cv,
        "n_samples": int(df.shape[0]),
        "n_features": int(df.shape[1] - 1),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "best_model": str(bench_df["roc_auc"].idxmax()),
        "best_roc_auc": float(bench_df["roc_auc"].max()),
    }
    meta_path = output_dir / "experiment_meta.json"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"  Metadata   : {meta_path}")

    # ── 5. Figures ────────────────────────────────────────────────────────────
    if not args.no_figures:
        print("\n[5/5] Generating figures...")
        saved_figs = generate_all_figures(
            trained_models, X_train, X_test, y_train, y_test,
            get_feature_names(), bench_df, figures_dir, verbose=True
        )
        for label, path in saved_figs.items():
            print(f"  {label}: {path}")
    else:
        print("\n[5/5] Skipping figures (--no-figures)")

    print(f"\n{separator}")
    print("Experiment complete.")
    print(f"Best model : {meta['best_model']} (ROC-AUC = {meta['best_roc_auc']:.4f})")
    print(f"All outputs: {output_dir.resolve()}")
    print(separator)


if __name__ == "__main__":
    main()
