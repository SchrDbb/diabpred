"""
diabpred.models
===============
Model registry and training pipeline.

Provides a unified interface for training, saving, and loading all supported
classifiers. Adding a new algorithm requires only a single dict entry.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ── Model registry ────────────────────────────────────────────────────────────
# Each entry maps a human-readable name → (class, constructor kwargs).
# All models use probability calibration where supported (probability=True).
MODEL_REGISTRY: Dict[str, Any] = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42, solver="lbfgs"
    ),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, random_state=42
    ),
    "SVM": SVC(probability=True, random_state=42, kernel="rbf"),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
}

DEFAULT_MODELS_DIR = Path("models")


def get_model(name: str):
    """Return a fresh (unfitted) model instance by name.

    Parameters
    ----------
    name : str
        One of the keys in :data:`MODEL_REGISTRY`.

    Returns
    -------
    sklearn estimator

    Raises
    ------
    ValueError
        If *name* is not in the registry.
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    # Return a fresh clone to avoid state pollution between experiments
    from sklearn.base import clone

    return clone(MODEL_REGISTRY[name])


def train_model(
    name: str, X_train: np.ndarray, y_train: np.ndarray
):
    """Fit a single model and return it.

    Parameters
    ----------
    name : str
        Model name from the registry.
    X_train : np.ndarray
    y_train : np.ndarray

    Returns
    -------
    fitted sklearn estimator
    """
    model = get_model(name)
    model.fit(X_train, y_train)
    return model


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    names: Optional[list] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train every registered model and return a name → fitted model dict.

    Parameters
    ----------
    X_train : np.ndarray
    y_train : np.ndarray
    names : list of str, optional
        Subset of model names to train. Defaults to all registry entries.
    verbose : bool, optional
        Print progress (default True).

    Returns
    -------
    dict
        Mapping model name → fitted estimator.

    Examples
    --------
    >>> from diabpred.data import load_dataset, preprocess
    >>> from diabpred.models import train_all_models
    >>> df = load_dataset()
    >>> X_train, X_test, y_train, y_test, scaler = preprocess(df)
    >>> models = train_all_models(X_train, y_train)
    >>> len(models)
    7
    """
    names = names or list(MODEL_REGISTRY.keys())
    trained: Dict[str, Any] = {}

    for name in names:
        if verbose:
            print(f"  Training {name}...", end=" ", flush=True)
        trained[name] = train_model(name, X_train, y_train)
        if verbose:
            print("done")

    return trained


def save_model(model, name: str, directory: Path | str = DEFAULT_MODELS_DIR) -> Path:
    """Persist a fitted model to disk with pickle.

    Parameters
    ----------
    model : fitted sklearn estimator
    name : str
        Logical name used to derive the filename.
    directory : Path or str
        Target directory (created if absent).

    Returns
    -------
    Path
        Full path of the saved file.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    safe_name = name.lower().replace(" ", "_")
    path = directory / f"{safe_name}.pkl"
    with open(path, "wb") as fh:
        pickle.dump(model, fh)
    return path


def load_model(name: str, directory: Path | str = DEFAULT_MODELS_DIR):
    """Load a previously saved model from disk.

    Parameters
    ----------
    name : str
        Logical name used when the model was saved.
    directory : Path or str

    Returns
    -------
    fitted sklearn estimator

    Raises
    ------
    FileNotFoundError
        If no saved file is found.
    """
    directory = Path(directory)
    safe_name = name.lower().replace(" ", "_")
    path = directory / f"{safe_name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No saved model found at {path}")
    with open(path, "rb") as fh:
        return pickle.load(fh)


def list_available_models() -> list:
    """Return the names of all models in the registry.

    Returns
    -------
    list of str
    """
    return list(MODEL_REGISTRY.keys())
