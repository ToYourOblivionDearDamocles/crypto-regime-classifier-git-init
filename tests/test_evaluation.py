import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from crypto_regime.evaluation.evaluate import run_evaluation
from crypto_regime.evaluation.metrics import (
    compute_binary_classification_metrics,
    compute_metrics,
    compute_multiclass_classification_metrics,
    compute_regression_metrics,
    precision_at_top_fraction,
    recall_at_top_fraction,
)


def make_split_dataset(n: int = 180) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2025-01-01 00:00:00",
        periods=n,
        freq="5min",
        tz="UTC",
    )

    x = np.arange(n)

    split = np.array(["train"] * n, dtype=object)
    split[int(0.70 * n): int(0.85 * n)] = "validation"
    split[int(0.85 * n):] = "test"

    feature_a = np.sin(x / 8.0)
    feature_b = np.cos(x / 15.0)
    feature_c = x / n

    score = feature_a + 0.5 * feature_b + 0.2 * feature_c
    threshold = np.quantile(score[: int(0.70 * n)], 0.8)
    y = (score > threshold).astype(int)

    return pd.DataFrame(
        {
            "symbol": "BTCUSDT",
            "timestamp": timestamps,
            "close": 100.0 + x * 0.1,
            "feature_a": feature_a,
            "feature_b": feature_b,
            "feature_c": feature_c,
            "future_rv_12": np.abs(score),
            "y_high_vol_12_split_safe": y,
            "split": split,
        }
    )


def test_precision_and_recall_at_top_fraction():
    y = np.array([0, 1, 0, 1, 1])
    score = np.array([0.1, 0.9, 0.2, 0.8, 0.7])

    assert precision_at_top_fraction(y, score, 0.4) == 1.0
    assert recall_at_top_fraction(y, score, 0.4) == 2 / 3


def test_binary_classification_metrics():
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = compute_binary_classification_metrics(
        y_true=y,
        y_score=score,
        thresholds=[0.5],
        top_k_fractions=[0.5],
    )

    assert metrics["task_type"] == "binary_classification"
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["threshold_metrics"]["0.5"]["accuracy"] == 1.0


def test_multiclass_metrics():
    y_true = np.array([0, 1, 2, 1, 0])
    y_pred = np.array([0, 1, 2, 0, 0])

    metrics = compute_multiclass_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
    )

    assert metrics["task_type"] == "multiclass_classification"
    assert "macro_f1" in metrics
    assert "per_class" in metrics


def test_regression_metrics():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])

    metrics = compute_regression_metrics(y_true, y_pred)

    assert metrics["task_type"] == "regression"
    assert metrics["mae"] is not None
    assert metrics["rmse"] is not None
    assert metrics["r2"] is not None


def test_metric_dispatcher_for_regression():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.1, 2.9])

    metrics = compute_metrics(
        task_type="regression",
        y_true=y_true,
        y_pred=y_pred,
    )

    assert metrics["task_type"] == "regression"


def test_run_evaluation_end_to_end(tmp_path: Path):
    df = make_split_dataset()

    split_path = tmp_path / "splits.parquet"
    schema_path = tmp_path / "feature_schema.json"
    artifact_root = tmp_path / "models"
    evaluation_output_dir = tmp_path / "evaluation"
    prediction_output_dir = evaluation_output_dir / "predictions"
    report_path = evaluation_output_dir / "evaluation_report.md"
    summary_path = evaluation_output_dir / "evaluation_summary.json"

    df.to_parquet(split_path, index=False)

    schema_path.write_text(
        json.dumps(
            {
                "feature_columns": ["feature_a", "feature_b", "feature_c"],
                "num_features": 3,
            }
        ),
        encoding="utf-8",
    )

    X_train = df[df["split"] == "train"][["feature_a", "feature_b", "feature_c"]]
    y_train = df[df["split"] == "train"]["y_high_vol_12_split_safe"].astype(int)

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)

    logistic = LogisticRegression(max_iter=500)
    logistic.fit(X_train, y_train)

    for name, model, estimator in [
        ("majority_baseline", dummy, "dummy_classifier"),
        ("logistic_regression", logistic, "logistic_regression"),
    ]:
        model_dir = artifact_root / name
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / "model.joblib"
        metadata_path = model_dir / "metadata.json"

        joblib.dump(model, model_path)

        metadata_path.write_text(
            json.dumps(
                {
                    "name": name,
                    "backend": "sklearn",
                    "estimator": estimator,
                    "artifact_format": "joblib",
                    "model_path": str(model_path),
                }
            ),
            encoding="utf-8",
        )

    config = {
        "task_type": "binary_classification",
        "model_version": "test_binary_v1",
        "split_input_path": str(split_path),
        "feature_schema_path": str(schema_path),
        "model_artifact_dir": str(artifact_root),
        "evaluation_output_dir": str(evaluation_output_dir),
        "prediction_output_dir": str(prediction_output_dir),
        "evaluation_report_path": str(report_path),
        "evaluation_summary_path": str(summary_path),
        "target_column": "y_high_vol_12_split_safe",
        "split_column": "split",
        "evaluation_splits": ["validation", "test"],
        "positive_label": 1,
        "thresholds": [0.5],
        "top_k_fractions": [0.05, 0.10],
        "models": "discover",
    }

    result = run_evaluation(config)

    assert Path(result["evaluation_summary_path"]).exists()
    assert Path(result["evaluation_report_path"]).exists()
    assert Path(result["metrics_table_path"]).exists()

    assert (prediction_output_dir / "majority_baseline" / "validation_predictions.parquet").exists()
    assert (prediction_output_dir / "logistic_regression" / "test_predictions.parquet").exists()
