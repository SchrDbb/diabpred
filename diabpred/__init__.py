"""
DiabPred: An open-source machine learning toolkit for diabetes risk prediction.

A reproducible, benchmarked comparison of classical ML algorithms on the
Pima Indians Diabetes Dataset, with a clean prediction API and automated
evaluation pipeline.
"""

__version__ = "1.0.0"
__author__ = "Dassi Bopda Blondel Christian"
__email__ = "dassibopdablondel@gmail.com"
__license__ = "MIT"

from diabpred.data import load_dataset, preprocess
from diabpred.models import train_all_models, load_model, save_model
from diabpred.evaluate import evaluate_model, full_benchmark
from diabpred.predict import predict, predict_proba
from diabpred.visualize import plot_roc_curves, plot_confusion_matrix, plot_feature_importance

__all__ = [
    "load_dataset",
    "preprocess",
    "train_all_models",
    "load_model",
    "save_model",
    "evaluate_model",
    "full_benchmark",
    "predict",
    "predict_proba",
    "plot_roc_curves",
    "plot_confusion_matrix",
    "plot_feature_importance",
]
