"""
Unit tests for DiabPred.

Run with:
    pytest tests/ -v
    pytest tests/ -v --cov=diabpred --cov-report=term-missing
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_df():
    """A tiny synthetic dataset that mirrors the Pima structure."""
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "Pregnancies": rng.integers(0, 15, n).astype(float),
        "Glucose": rng.uniform(60, 200, n),
        "BloodPressure": rng.uniform(40, 110, n),
        "SkinThickness": rng.uniform(0, 60, n),
        "Insulin": rng.uniform(0, 300, n),
        "BMI": rng.uniform(15, 55, n),
        "DiabetesPedigreeFunction": rng.uniform(0.05, 2.5, n),
        "Age": rng.integers(21, 80, n).astype(float),
        "Outcome": rng.integers(0, 2, n),
    })
    return df


@pytest.fixture(scope="module")
def processed_data(synthetic_df):
    from diabpred.data import preprocess
    return preprocess(synthetic_df, random_state=42)


@pytest.fixture(scope="module")
def trained_lr(processed_data):
    from diabpred.models import train_model
    X_train, X_test, y_train, y_test, scaler = processed_data
    return train_model("Logistic Regression", X_train, y_train)


# ── diabpred.data ─────────────────────────────────────────────────────────────

class TestData:
    def test_preprocess_shapes(self, processed_data):
        X_train, X_test, y_train, y_test, scaler = processed_data
        assert X_train.shape[1] == 8
        assert X_test.shape[1] == 8
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)

    def test_preprocess_split_ratio(self, synthetic_df):
        from diabpred.data import preprocess
        X_train, X_test, y_train, y_test, scaler = preprocess(synthetic_df, test_size=0.2)
        total = len(X_train) + len(X_test)
        assert total == len(synthetic_df)
        assert abs(len(X_test) / total - 0.2) < 0.05

    def test_preprocess_reproducible(self, synthetic_df):
        from diabpred.data import preprocess
        r1 = preprocess(synthetic_df, random_state=0)
        r2 = preprocess(synthetic_df, random_state=0)
        np.testing.assert_array_equal(r1[0], r2[0])

    def test_preprocess_different_seeds(self, synthetic_df):
        from diabpred.data import preprocess
        r1 = preprocess(synthetic_df, random_state=0)
        r2 = preprocess(synthetic_df, random_state=1)
        # Different seeds → different splits (with very high probability)
        assert not np.array_equal(r1[0], r2[0])

    def test_scaler_fitted(self, processed_data):
        X_train, X_test, y_train, y_test, scaler = processed_data
        # StandardScaler should have computed mean/std
        assert hasattr(scaler, "mean_")
        assert scaler.mean_.shape == (8,)

    def test_zero_imputation(self, synthetic_df):
        from diabpred.data import preprocess, ZERO_IS_MISSING
        # Introduce artificial zeros
        df = synthetic_df.copy()
        df.loc[0, "Glucose"] = 0
        X_train, X_test, y_train, y_test, scaler = preprocess(df, impute_zeros=True)
        # After scaling, we can't check exact values, but shape is correct
        assert X_train.shape[1] == 8

    def test_get_feature_names(self):
        from diabpred.data import get_feature_names
        names = get_feature_names()
        assert len(names) == 8
        assert "Glucose" in names
        assert "BMI" in names

    def test_describe_dataset(self, synthetic_df):
        from diabpred.data import describe_dataset
        stats = describe_dataset(synthetic_df)
        assert "Glucose" in stats.columns
        assert "zeros" in stats.index


# ── diabpred.models ───────────────────────────────────────────────────────────

class TestModels:
    def test_list_available_models(self):
        from diabpred.models import list_available_models
        names = list_available_models()
        assert len(names) == 7
        assert "Random Forest" in names
        assert "Logistic Regression" in names

    def test_get_model_valid(self):
        from diabpred.models import get_model
        model = get_model("Random Forest")
        assert model is not None

    def test_get_model_invalid(self):
        from diabpred.models import get_model
        with pytest.raises(ValueError, match="Unknown model"):
            get_model("Nonexistent Model")

    def test_train_model_fits(self, processed_data):
        from diabpred.models import train_model
        X_train, X_test, y_train, y_test, scaler = processed_data
        model = train_model("Naive Bayes", X_train, y_train)
        assert hasattr(model, "predict_proba")

    def test_train_all_models_count(self, processed_data):
        from diabpred.models import train_all_models
        X_train, X_test, y_train, y_test, scaler = processed_data
        models = train_all_models(X_train, y_train, verbose=False)
        assert len(models) == 7

    def test_train_subset_of_models(self, processed_data):
        from diabpred.models import train_all_models
        X_train, X_test, y_train, y_test, scaler = processed_data
        models = train_all_models(
            X_train, y_train,
            names=["Logistic Regression", "Naive Bayes"],
            verbose=False
        )
        assert len(models) == 2

    def test_save_and_load_model(self, tmp_path, processed_data):
        from diabpred.models import load_model, save_model, train_model
        X_train, X_test, y_train, y_test, scaler = processed_data
        model = train_model("Logistic Regression", X_train, y_train)

        path = save_model(model, "Logistic Regression", tmp_path)
        assert path.exists()

        loaded = load_model("Logistic Regression", tmp_path)
        preds_original = model.predict(X_test)
        preds_loaded = loaded.predict(X_test)
        np.testing.assert_array_equal(preds_original, preds_loaded)

    def test_load_model_not_found(self, tmp_path):
        from diabpred.models import load_model
        with pytest.raises(FileNotFoundError):
            load_model("Ghost Model", tmp_path)


# ── diabpred.evaluate ─────────────────────────────────────────────────────────

class TestEvaluate:
    def test_evaluate_model_keys(self, processed_data, trained_lr):
        from diabpred.evaluate import evaluate_model
        X_train, X_test, y_train, y_test, scaler = processed_data
        metrics = evaluate_model(trained_lr, X_test, y_test)
        expected_keys = {"accuracy", "precision", "recall", "f1", "roc_auc",
                         "specificity", "npv", "avg_precision", "mcc",
                         "tn", "fp", "fn", "tp"}
        assert expected_keys.issubset(set(metrics.keys()))

    def test_evaluate_model_ranges(self, processed_data, trained_lr):
        from diabpred.evaluate import evaluate_model
        X_train, X_test, y_train, y_test, scaler = processed_data
        metrics = evaluate_model(trained_lr, X_test, y_test)
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            assert 0.0 <= metrics[key] <= 1.0, f"{key} out of range: {metrics[key]}"

    def test_confusion_matrix_consistency(self, processed_data, trained_lr):
        from diabpred.evaluate import evaluate_model
        X_train, X_test, y_train, y_test, scaler = processed_data
        m = evaluate_model(trained_lr, X_test, y_test)
        assert m["tn"] + m["fp"] + m["fn"] + m["tp"] == len(y_test)

    def test_cross_validate_model(self, processed_data):
        from diabpred.evaluate import cross_validate_model
        from diabpred.models import get_model
        X_train, X_test, y_train, y_test, scaler = processed_data
        X_full = np.vstack([X_train, X_test])
        y_full = np.concatenate([y_train, y_test])
        cv_metrics = cross_validate_model(get_model("Logistic Regression"), X_full, y_full, cv=3)
        assert "cv_roc_auc_mean" in cv_metrics
        assert 0.0 <= cv_metrics["cv_roc_auc_mean"] <= 1.0


# ── diabpred.predict ──────────────────────────────────────────────────────────

SAMPLE_PATIENT = {
    "Pregnancies": 2,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 20,
    "Insulin": 80,
    "BMI": 28.0,
    "DiabetesPedigreeFunction": 0.3,
    "Age": 35,
}

HIGH_RISK_PATIENT = {
    "Pregnancies": 6,
    "Glucose": 180,
    "BloodPressure": 80,
    "SkinThickness": 40,
    "Insulin": 250,
    "BMI": 40.0,
    "DiabetesPedigreeFunction": 1.2,
    "Age": 55,
}


class TestPredict:
    def test_predict_keys(self, processed_data, trained_lr):
        from diabpred.predict import predict
        X_train, X_test, y_train, y_test, scaler = processed_data
        result = predict(SAMPLE_PATIENT, trained_lr, scaler)
        assert "prediction" in result
        assert "label" in result
        assert "probability" in result
        assert "risk_level" in result

    def test_predict_binary_output(self, processed_data, trained_lr):
        from diabpred.predict import predict
        X_train, X_test, y_train, y_test, scaler = processed_data
        result = predict(SAMPLE_PATIENT, trained_lr, scaler)
        assert result["prediction"] in (0, 1)

    def test_predict_probability_range(self, processed_data, trained_lr):
        from diabpred.predict import predict
        X_train, X_test, y_train, y_test, scaler = processed_data
        result = predict(SAMPLE_PATIENT, trained_lr, scaler)
        assert 0.0 <= result["probability"] <= 1.0

    def test_predict_risk_levels(self, processed_data, trained_lr):
        from diabpred.predict import predict
        X_train, X_test, y_train, y_test, scaler = processed_data
        result = predict(SAMPLE_PATIENT, trained_lr, scaler)
        assert result["risk_level"] in ("Low", "Moderate", "High")

    def test_predict_missing_feature_raises(self, processed_data, trained_lr):
        from diabpred.predict import predict
        X_train, X_test, y_train, y_test, scaler = processed_data
        bad_patient = {k: v for k, v in SAMPLE_PATIENT.items() if k != "Glucose"}
        with pytest.raises(ValueError, match="Missing features"):
            predict(bad_patient, trained_lr, scaler)

    def test_predict_proba_all_models(self, processed_data):
        from diabpred.models import train_all_models
        from diabpred.predict import predict_proba
        X_train, X_test, y_train, y_test, scaler = processed_data
        models = train_all_models(X_train, y_train, verbose=False)
        probs = predict_proba(SAMPLE_PATIENT, models, scaler)
        assert "ensemble_mean" in probs
        assert 0.0 <= probs["ensemble_mean"] <= 1.0

    def test_batch_predict(self, processed_data, trained_lr):
        from diabpred.predict import batch_predict
        X_train, X_test, y_train, y_test, scaler = processed_data
        records = [SAMPLE_PATIENT, HIGH_RISK_PATIENT]
        results = batch_predict(records, trained_lr, scaler)
        assert len(results) == 2
        for r in results:
            assert r["prediction"] in (0, 1)

    def test_custom_threshold(self, processed_data, trained_lr):
        from diabpred.predict import predict
        X_train, X_test, y_train, y_test, scaler = processed_data
        # Threshold 0.0 → always predict positive
        result = predict(SAMPLE_PATIENT, trained_lr, scaler, threshold=0.0)
        assert result["prediction"] == 1
        # Threshold 1.0 → always predict negative
        result = predict(SAMPLE_PATIENT, trained_lr, scaler, threshold=1.0)
        assert result["prediction"] == 0


# ── diabpred.visualize ────────────────────────────────────────────────────────

class TestVisualize:
    def test_plot_roc_curves(self, tmp_path, processed_data):
        from diabpred.models import train_all_models
        from diabpred.visualize import plot_roc_curves
        X_train, X_test, y_train, y_test, scaler = processed_data
        models = train_all_models(X_train, y_train, verbose=False)
        path = plot_roc_curves(models, X_test, y_test, output_dir=tmp_path)
        assert path.exists()
        assert path.suffix == ".png"

    def test_plot_confusion_matrix(self, tmp_path, processed_data, trained_lr):
        from diabpred.visualize import plot_confusion_matrix
        X_train, X_test, y_train, y_test, scaler = processed_data
        path = plot_confusion_matrix(trained_lr, X_test, y_test, output_dir=tmp_path)
        assert path.exists()

    def test_plot_benchmark_comparison(self, tmp_path, processed_data):
        from diabpred.evaluate import full_benchmark
        from diabpred.models import train_all_models
        from diabpred.visualize import plot_benchmark_comparison
        X_train, X_test, y_train, y_test, scaler = processed_data
        models = train_all_models(X_train, y_train, verbose=False)
        bench = full_benchmark(models, X_train, X_test, y_train, y_test, cv=3, verbose=False)
        path = plot_benchmark_comparison(bench, output_dir=tmp_path)
        assert path.exists()
