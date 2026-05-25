"""
diabpred.data
=============
Data loading and preprocessing pipeline for the Pima Indians Diabetes Dataset.

The dataset is fetched automatically from the UCI ML Repository via OpenML.
All preprocessing steps are deterministic and reproducible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Feature names for the Pima Indians Diabetes Dataset
FEATURE_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

TARGET_NAME = "Outcome"

# Biologically impossible zero values that must be treated as missing
ZERO_IS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Default data path (cached after first download)
_DEFAULT_DATA_PATH = Path(__file__).parent.parent / "data" / "diabetes.csv"


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load the Pima Indians Diabetes Dataset.

    Attempts to load from a local CSV file first. If not found, downloads
    automatically from OpenML (dataset ID 37).

    Parameters
    ----------
    path : str or Path, optional
        Path to a local CSV file. If None, uses the bundled data path or
        downloads from OpenML.

    Returns
    -------
    pd.DataFrame
        Raw dataset with shape (768, 9): 8 features + 1 target column.

    Examples
    --------
    >>> from diabpred.data import load_dataset
    >>> df = load_dataset()
    >>> df.shape
    (768, 9)
    """
    if path is not None:
        return pd.read_csv(path)

    if _DEFAULT_DATA_PATH.exists():
        return pd.read_csv(_DEFAULT_DATA_PATH)

    return _download_dataset()


def _download_dataset() -> pd.DataFrame:
    """Download dataset from OpenML and cache it locally."""
    try:
        from sklearn.datasets import fetch_openml

        _DEFAULT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = fetch_openml(data_id=37, as_frame=True, parser="auto")
        df = data.frame.copy()

        # Standardise column names
        df.columns = FEATURE_NAMES + [TARGET_NAME]
        df[TARGET_NAME] = df[TARGET_NAME].astype(int)

        df.to_csv(_DEFAULT_DATA_PATH, index=False)
        return df

    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Could not download the dataset. Please download it manually from "
            "https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database "
            "and place it at data/diabetes.csv"
        ) from exc


def preprocess(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    impute_zeros: bool = True,
    scale: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Full preprocessing pipeline: imputation → split → scaling.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset as returned by :func:`load_dataset`.
    test_size : float, optional
        Fraction of data held out for testing (default 0.2).
    random_state : int, optional
        Random seed for reproducibility (default 42).
    impute_zeros : bool, optional
        Replace biologically impossible zeros with column median (default True).
    scale : bool, optional
        Apply StandardScaler to features (default True).

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
        Train/test arrays ready for model fitting.
    scaler : StandardScaler
        Fitted scaler (needed to transform new inputs at prediction time).

    Examples
    --------
    >>> from diabpred.data import load_dataset, preprocess
    >>> df = load_dataset()
    >>> X_train, X_test, y_train, y_test, scaler = preprocess(df)
    >>> X_train.shape[1]
    8
    """
    df = df.copy()

    # ── 1. Impute biologically impossible zeros ─────────────────────────────
    if impute_zeros:
        for col in ZERO_IS_MISSING:
            if col in df.columns:
                median = df.loc[df[col] != 0, col].median()
                df[col] = df[col].replace(0, median)

    # ── 2. Separate features / target ───────────────────────────────────────
    X = df[FEATURE_NAMES].values
    y = df[TARGET_NAME].values

    # ── 3. Train / test split ────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # ── 4. Feature scaling ───────────────────────────────────────────────────
    scaler = StandardScaler()
    if scale:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        scaler.fit(X_train)  # fitted but not applied (kept for API consistency)

    return X_train, X_test, y_train, y_test, scaler


def get_feature_names() -> list[str]:
    """Return the ordered list of feature names.

    Returns
    -------
    list of str
    """
    return list(FEATURE_NAMES)


def describe_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary statistics table including zero-value counts.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Descriptive statistics with an additional ``zeros`` row.
    """
    stats = df.describe()
    zero_counts = (df[FEATURE_NAMES] == 0).sum().rename("zeros")
    return pd.concat([stats, zero_counts.to_frame().T])
