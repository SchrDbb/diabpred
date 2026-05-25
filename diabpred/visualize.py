"""
diabpred.visualize
==================
Automated evaluation visualization pipeline.

All plot functions save to an output directory and return the figure path.
This makes the entire figure generation step reproducible by running a
single script (``scripts/run_experiment.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CI use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    auc,
    roc_curve,
)

DEFAULT_OUTPUT_DIR = Path("outputs") / "figures"
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2",
]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_roc_curves(
    trained_models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    filename: str = "roc_curves.png",
) -> Path:
    """Plot overlaid ROC curves for all models.

    Parameters
    ----------
    trained_models : dict
    X_test, y_test : np.ndarray
    output_dir : Path or str
    filename : str

    Returns
    -------
    Path
        Saved figure path.
    """
    output_dir = _ensure_dir(Path(output_dir))
    fig, ax = plt.subplots(figsize=(8, 6))

    for (name, model), color in zip(trained_models.items(), PALETTE):
        prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        area = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {area:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves – All Models", fontsize=14)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = output_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusion_matrix(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    filename: Optional[str] = None,
) -> Path:
    """Save a confusion matrix heatmap for a single model.

    Parameters
    ----------
    model : fitted sklearn estimator
    X_test, y_test : np.ndarray
    model_name : str
    output_dir : Path or str
    filename : str, optional

    Returns
    -------
    Path
    """
    output_dir = _ensure_dir(Path(output_dir))
    if filename is None:
        safe = model_name.lower().replace(" ", "_")
        filename = f"confusion_matrix_{safe}.png"

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels=["Non-diabetic", "Diabetic"],
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title(f"Confusion Matrix – {model_name}", fontsize=13)
    fig.tight_layout()

    path = output_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_feature_importance(
    model,
    feature_names: List[str],
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    filename: Optional[str] = None,
    n_repeats: int = 10,
    random_state: int = 42,
) -> Path:
    """Compute and plot permutation feature importances.

    Uses permutation importance (model-agnostic) so all models are
    treated consistently regardless of internal feature weights.

    Parameters
    ----------
    model : fitted sklearn estimator
    feature_names : list of str
    X_test, y_test : np.ndarray
    model_name : str
    output_dir : Path or str
    filename : str, optional
    n_repeats : int
    random_state : int

    Returns
    -------
    Path
    """
    output_dir = _ensure_dir(Path(output_dir))
    if filename is None:
        safe = model_name.lower().replace(" ", "_")
        filename = f"feature_importance_{safe}.png"

    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="roc_auc",
        n_jobs=-1,
    )

    idx = result.importances_mean.argsort()[::-1]
    sorted_names = [feature_names[i] for i in idx]
    sorted_means = result.importances_mean[idx]
    sorted_stds = result.importances_std[idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(
        range(len(sorted_names)),
        sorted_means,
        xerr=sorted_stds,
        color=PALETTE[0],
        alpha=0.85,
        ecolor="gray",
        capsize=3,
    )
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Mean decrease in ROC-AUC", fontsize=11)
    ax.set_title(f"Permutation Feature Importance – {model_name}", fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    path = output_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_benchmark_comparison(
    benchmark_df: pd.DataFrame,
    metric: str = "roc_auc",
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    filename: str = "benchmark_comparison.png",
) -> Path:
    """Bar chart comparing all models on a single metric.

    Parameters
    ----------
    benchmark_df : pd.DataFrame
        Output of :func:`~diabpred.evaluate.full_benchmark`.
    metric : str
        Column name to plot (default ``roc_auc``).
    output_dir : Path or str
    filename : str

    Returns
    -------
    Path
    """
    output_dir = _ensure_dir(Path(output_dir))
    df = benchmark_df.sort_values(metric, ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(df))]
    ax.barh(df.index, df[metric], color=colors, alpha=0.85)
    ax.set_xlabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(f"Model Comparison – {metric.replace('_', ' ').title()}", fontsize=14)
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.3)

    for i, (val, name) in enumerate(zip(df[metric], df.index)):
        ax.text(val + 0.005, i, f"{val:.4f}", va="center", fontsize=9)

    fig.tight_layout()
    path = output_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_precision_recall_curves(
    trained_models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    filename: str = "pr_curves.png",
) -> Path:
    """Plot precision-recall curves for all models.

    Parameters
    ----------
    trained_models, X_test, y_test, output_dir, filename : see :func:`plot_roc_curves`

    Returns
    -------
    Path
    """
    from sklearn.metrics import average_precision_score, precision_recall_curve

    output_dir = _ensure_dir(Path(output_dir))
    fig, ax = plt.subplots(figsize=(8, 6))

    for (name, model), color in zip(trained_models.items(), PALETTE):
        prob = model.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, prob)
        ap = average_precision_score(y_test, prob)
        ax.plot(rec, prec, color=color, lw=2, label=f"{name} (AP = {ap:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves – All Models", fontsize=14)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = output_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_all_figures(
    trained_models: Dict[str, Any],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    benchmark_df: pd.DataFrame,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    verbose: bool = True,
) -> Dict[str, Path]:
    """Generate and save all evaluation figures in one call.

    Parameters
    ----------
    trained_models : dict
    X_train, X_test, y_train, y_test : np.ndarray
    feature_names : list of str
    benchmark_df : pd.DataFrame
    output_dir : Path or str
    verbose : bool

    Returns
    -------
    dict
        Figure name → saved path.
    """
    output_dir = Path(output_dir)
    saved: Dict[str, Path] = {}

    def _log(msg: str) -> None:
        if verbose:
            print(f"  {msg}")

    _log("Plotting ROC curves...")
    saved["roc_curves"] = plot_roc_curves(trained_models, X_test, y_test, output_dir)

    _log("Plotting PR curves...")
    saved["pr_curves"] = plot_precision_recall_curves(trained_models, X_test, y_test, output_dir)

    _log("Plotting benchmark comparison...")
    saved["benchmark"] = plot_benchmark_comparison(benchmark_df, output_dir=output_dir)

    # Best model for detailed plots
    best_name = benchmark_df["roc_auc"].idxmax()
    best_model = trained_models[best_name]

    _log(f"Plotting confusion matrix for best model ({best_name})...")
    saved["confusion_matrix"] = plot_confusion_matrix(
        best_model, X_test, y_test, best_name, output_dir
    )

    _log(f"Plotting feature importance for best model ({best_name})...")
    saved["feature_importance"] = plot_feature_importance(
        best_model, feature_names, X_test, y_test, best_name, output_dir
    )

    return saved
