"""
diabpred.predict
================
High-level prediction API for new patient data.

This is the primary user-facing interface. Given a dict of patient
measurements, it returns a risk label and probability score.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from diabpred.data import FEATURE_NAMES


def _validate_input(features: Dict[str, float]) -> np.ndarray:
    """Validate and order a feature dict into a model-ready array.

    Parameters
    ----------
    features : dict
        Keys must match :data:`~diabpred.data.FEATURE_NAMES`.

    Returns
    -------
    np.ndarray
        Shape (1, 8) ordered array.

    Raises
    ------
    ValueError
        If any required feature is missing.
    """
    missing = [f for f in FEATURE_NAMES if f not in features]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    return np.array([[features[f] for f in FEATURE_NAMES]], dtype=float)


def predict(
    features: Dict[str, float],
    model,
    scaler=None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Predict diabetes risk for a single patient.

    Parameters
    ----------
    features : dict
        Patient measurements. Required keys:
        Pregnancies, Glucose, BloodPressure, SkinThickness,
        Insulin, BMI, DiabetesPedigreeFunction, Age.
    model : fitted sklearn estimator
    scaler : fitted StandardScaler, optional
        If provided, scales the input before prediction.
    threshold : float, optional
        Decision threshold (default 0.5).

    Returns
    -------
    dict with keys:
        - ``prediction`` (int): 1 = diabetic, 0 = non-diabetic
        - ``label`` (str): human-readable label
        - ``probability`` (float): probability of being diabetic
        - ``risk_level`` (str): Low / Moderate / High
        - ``threshold`` (float): threshold used

    Examples
    --------
    >>> from diabpred.data import load_dataset, preprocess
    >>> from diabpred.models import train_model
    >>> from diabpred.predict import predict
    >>> df = load_dataset()
    >>> X_train, X_test, y_train, y_test, scaler = preprocess(df)
    >>> model = train_model("Random Forest", X_train, y_train)
    >>> result = predict(
    ...     {"Pregnancies": 2, "Glucose": 120, "BloodPressure": 70,
    ...      "SkinThickness": 20, "Insulin": 80, "BMI": 28.0,
    ...      "DiabetesPedigreeFunction": 0.3, "Age": 35},
    ...     model, scaler
    ... )
    >>> result["prediction"] in (0, 1)
    True
    """
    X = _validate_input(features)

    if scaler is not None:
        X = scaler.transform(X)

    prob = float(model.predict_proba(X)[0, 1])
    pred = int(prob >= threshold)

    if prob < 0.3:
        risk_level = "Low"
    elif prob < 0.6:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    return {
        "prediction": pred,
        "label": "Diabetic" if pred == 1 else "Non-diabetic",
        "probability": round(prob, 4),
        "risk_level": risk_level,
        "threshold": threshold,
    }


def predict_proba(
    features: Dict[str, float],
    models: Dict[str, Any],
    scaler=None,
) -> Dict[str, float]:
    """Return diabetes probability from every model as an ensemble view.

    Parameters
    ----------
    features : dict
        Patient measurements (same as :func:`predict`).
    models : dict
        Mapping model name → fitted estimator (output of
        :func:`~diabpred.models.train_all_models`).
    scaler : fitted StandardScaler, optional

    Returns
    -------
    dict
        Model name → probability of diabetes, plus ``ensemble_mean``.

    Examples
    --------
    >>> from diabpred.data import load_dataset, preprocess
    >>> from diabpred.models import train_all_models
    >>> from diabpred.predict import predict_proba
    >>> df = load_dataset()
    >>> X_train, X_test, y_train, y_test, scaler = preprocess(df)
    >>> all_models = train_all_models(X_train, y_train, verbose=False)
    >>> probs = predict_proba(
    ...     {"Pregnancies": 5, "Glucose": 165, "BloodPressure": 72,
    ...      "SkinThickness": 35, "Insulin": 0, "BMI": 33.6,
    ...      "DiabetesPedigreeFunction": 0.627, "Age": 50},
    ...     all_models, scaler
    ... )
    >>> 0.0 <= probs["ensemble_mean"] <= 1.0
    True
    """
    X = _validate_input(features)
    if scaler is not None:
        X = scaler.transform(X)

    probs: Dict[str, float] = {}
    for name, model in models.items():
        probs[name] = round(float(model.predict_proba(X)[0, 1]), 4)

    probs["ensemble_mean"] = round(float(np.mean(list(probs.values()))), 4)
    return probs


def batch_predict(
    records: List[Dict[str, float]],
    model,
    scaler=None,
    threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """Predict diabetes risk for a list of patients.

    Parameters
    ----------
    records : list of dict
        Each dict must contain all required features.
    model : fitted sklearn estimator
    scaler : fitted StandardScaler, optional
    threshold : float, optional

    Returns
    -------
    list of dict
        One result dict per input record (same format as :func:`predict`).
    """
    return [predict(rec, model, scaler, threshold) for rec in records]
