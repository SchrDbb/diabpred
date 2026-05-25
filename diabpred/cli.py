#!/usr/bin/env python3
"""
diabpred/cli.py
===============
Command-line interface for DiabPred.

Provides three sub-commands:
  diabpred predict   -- predict for a single patient from CLI flags
  diabpred benchmark -- run the full benchmark experiment
  diabpred info      -- show available models and features

Usage examples
--------------
# Predict for a patient
python -m diabpred.cli predict \
    --pregnancies 6 --glucose 148 --bp 72 --skin 35 \
    --insulin 0 --bmi 33.6 --dpf 0.627 --age 50

# Run the full benchmark
python -m diabpred.cli benchmark --output-dir outputs/ --cv 5

# Show available models
python -m diabpred.cli info
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Colour helpers ────────────────────────────────────────────────────────────

def _green(s: str) -> str:
    return f"\033[92m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[91m{s}\033[0m"


def _yellow(s: str) -> str:
    return f"\033[93m{s}\033[0m"


def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m"


# ── Sub-command: predict ──────────────────────────────────────────────────────

def cmd_predict(args: argparse.Namespace) -> None:
    """Predict diabetes risk for a single patient."""
    from diabpred.data import load_dataset, preprocess
    from diabpred.models import load_model, save_model, train_model
    from diabpred.predict import predict, predict_proba

    patient = {
        "Pregnancies": args.pregnancies,
        "Glucose": args.glucose,
        "BloodPressure": args.bp,
        "SkinThickness": args.skin,
        "Insulin": args.insulin,
        "BMI": args.bmi,
        "DiabetesPedigreeFunction": args.dpf,
        "Age": args.age,
    }

    model_dir = Path(args.model_dir)
    model_name = args.model

    try:
        model = load_model(model_name, model_dir)
        print(f"  Loaded saved model: {model_name}")
    except FileNotFoundError:
        print(f"  No saved model found — training {model_name} now...")
        df = load_dataset()
        X_train, _, y_train, _, _ = preprocess(df)
        model = train_model(model_name, X_train, y_train)
        model_dir.mkdir(parents=True, exist_ok=True)
        save_model(model, model_name, model_dir)

    df = load_dataset()
    _, _, _, _, scaler = preprocess(df)

    result = predict(patient, model, scaler, threshold=args.threshold)

    print()
    print(_bold("=" * 42))
    print(_bold("  DiabPred - Patient Risk Assessment"))
    print(_bold("=" * 42))
    print(f"  Model     : {model_name}")
    print(f"  Threshold : {args.threshold}")
    print()

    label_str = _red("  DIABETIC") if result["prediction"] == 1 else _green("  NON-DIABETIC")
    print(f"  Prediction: {label_str}")
    print(f"  Probability: {result['probability']:.1%}")

    risk = result["risk_level"]
    if risk == "High":
        risk_str = _red(f"  Risk level: {risk}")
    elif risk == "Moderate":
        risk_str = _yellow(f"  Risk level: {risk}")
    else:
        risk_str = _green(f"  Risk level: {risk}")
    print(risk_str)

    if args.ensemble:
        print()
        print(_bold("  Ensemble view (all models):"))
        from diabpred.models import train_all_models
        all_models = train_all_models(
            preprocess(df)[0],
            preprocess(df)[2],
            verbose=False,
        )
        probs = predict_proba(patient, all_models, scaler)
        for name, prob in probs.items():
            bar = "X" * int(prob * 25)
            marker = " <- mean" if name == "ensemble_mean" else ""
            print(f"  {name:<28} {prob:.1%}  {bar}{marker}")

    if args.json:
        print()
        print(json.dumps(result, indent=2))

    print(_bold("=" * 42))


# ── Sub-command: benchmark ────────────────────────────────────────────────────

def cmd_benchmark(args: argparse.Namespace) -> None:
    """Run the full benchmark experiment."""
    import subprocess
    script = Path(__file__).parent.parent / "scripts" / "run_experiment.py"
    cmd = [
        sys.executable, str(script),
        "--output-dir", args.output_dir,
        "--cv", str(args.cv),
        "--seed", str(args.seed),
        "--test-size", str(args.test_size),
    ]
    if args.no_figures:
        cmd.append("--no-figures")
    subprocess.run(cmd, check=True)


# ── Sub-command: info ─────────────────────────────────────────────────────────

def cmd_info(_args: argparse.Namespace) -> None:
    """Print package info and available models."""
    import diabpred
    from diabpred.data import FEATURE_NAMES
    from diabpred.models import list_available_models

    print(_bold("\nDiabPred - Package Info"))
    print(f"  Version  : {diabpred.__version__}")
    print(f"  License  : {diabpred.__license__}")
    print()
    print(_bold("  Available models:"))
    for name in list_available_models():
        print(f"    - {name}")
    print()
    print(_bold("  Required input features:"))
    for i, name in enumerate(FEATURE_NAMES, 1):
        print(f"    {i}. {name}")
    print()


# ── Argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="diabpred",
        description="DiabPred - diabetes risk prediction toolkit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version="%(prog)s 1.0.0")
    sub = parser.add_subparsers(dest="command", required=True)

    # predict
    p_pred = sub.add_parser(
        "predict",
        help="Predict diabetes risk for a single patient",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_pred.add_argument("--pregnancies", type=float, required=True,
                        help="Number of pregnancies")
    p_pred.add_argument("--glucose", type=float, required=True,
                        help="Plasma glucose (mg/dL)")
    p_pred.add_argument("--bp", type=float, required=True,
                        help="Diastolic blood pressure (mmHg)")
    p_pred.add_argument("--skin", type=float, required=True,
                        help="Triceps skinfold thickness (mm)")
    p_pred.add_argument("--insulin", type=float, required=True,
                        help="2-hour serum insulin (uU/mL)")
    p_pred.add_argument("--bmi", type=float, required=True,
                        help="BMI (kg/m2)")
    p_pred.add_argument("--dpf", type=float, required=True,
                        help="Diabetes pedigree function")
    p_pred.add_argument("--age", type=float, required=True,
                        help="Age (years)")
    p_pred.add_argument("--model", default="Random Forest",
                        help="Model to use for prediction")
    p_pred.add_argument("--model-dir", default="models/",
                        help="Directory containing saved .pkl models")
    p_pred.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold (0-1)")
    p_pred.add_argument("--ensemble", action="store_true",
                        help="Show probabilities from all models")
    p_pred.add_argument("--json", action="store_true",
                        help="Also print result as JSON")
    p_pred.set_defaults(func=cmd_predict)

    # benchmark
    p_bench = sub.add_parser(
        "benchmark",
        help="Run the full benchmark experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_bench.add_argument("--output-dir", default="outputs/")
    p_bench.add_argument("--cv", type=int, default=5)
    p_bench.add_argument("--seed", type=int, default=42)
    p_bench.add_argument("--test-size", type=float, default=0.2)
    p_bench.add_argument("--no-figures", action="store_true")
    p_bench.set_defaults(func=cmd_benchmark)

    # info
    p_info = sub.add_parser("info", help="Show package info and available models")
    p_info.set_defaults(func=cmd_info)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
