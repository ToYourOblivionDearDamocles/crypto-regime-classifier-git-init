import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crypto_regime.models.data import (
    infer_feature_columns,
    make_modeling_dataset,
    resolve_feature_columns,
)
from crypto_regime.models.registry import build_model_adapter
from crypto_regime.models.train import compute_binary_metrics, run_training


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
    y = (score > np.quantile(score[: int(0.70 * n)], 0.8)).astype(int)

    return pd.DataFrame(
        {
            "symbol": "BTCUSDT",
            "timestamp": timestamps,
            "close": 100.0 + x * 0.1,
            "feature_a": feature_a,
            "feature_b": feature_b,
            "feature_c": feature_c,
            "future_rv_12": np.abs(score),
            "y_high_vol_12": y,
            "y_high_vol_12_split_safe": y,
            "split": split,
        }
    )


def test_infer_feature_columns_excludes_targets_and_metadata():
    df = make_split_dataset()

    feature_cols = infer_feature_columns(df)

    assert "feature_a" in feature_cols
    assert "feature_b" in feature_cols
    assert "feature_c" in feature_cols
    assert "future_rv_12" not in feature_cols
    assert "y_high_vol_12" not in feature_cols
    assert "y_high_vol_12_split_safe" not in feature_cols
    assert "split" not in feature_cols
    assert "timestamp" not in feature_cols


def test_resolve_feature_columns_from_schema():
    df = make_split_dataset()
    schema = {"feature_columns": ["feature_a", "feature_b"]}

    feature_cols = resolve_feature_columns(df, schema)

    assert feature_cols == ["feature_a", "feature_b"]


def test_resolve_feature_columns_rejects_forbidden_schema_column():
    df = make_split_dataset()
    schema = {"feature_columns": ["feature_a", "future_rv_12"]}

    with pytest.raises(ValueError):
        resolve_feature_columns(df, schema)


def test_make_modeling_dataset():
    df = make_split_dataset()
    feature_cols = ["feature_a", "feature_b", "feature_c"]

    dataset = make_modeling_dataset(
        df=df,
        feature_columns=feature_cols,
        target_column="y_high_vol_12_split_safe",
        split_column="split",
    )

    assert len(dataset.X_train) > 0
    assert len(dataset.X_validation) > 0
    assert len(dataset.X_test) > 0
    assert list(dataset.X_train.columns) == feature_cols


def test_sklearn_dummy_adapter_fit_predict():
    df = make_split_dataset()
    feature_cols = ["feature_a", "feature_b", "feature_c"]

    dataset = make_modeling_dataset(
        df=df,
        feature_columns=feature_cols,
        target_column="y_high_vol_12_split_safe",
        split_column="split",
    )

    model_config = {
        "name": "majority_baseline",
        "backend": "sklearn",
        "estimator": "dummy_classifier",
        "preprocessing": "none",
        "params": {"strategy": "most_frequent"},
    }

    adapter = build_model_adapter(model_config)
    adapter.fit(dataset.X_train, dataset.y_train)

    proba = adapter.predict_proba(dataset.X_validation)

    assert proba.shape == (len(dataset.X_validation), 2)


def test_sklearn_logistic_adapter_fit_predict():
    df = make_split_dataset()
    feature_cols = ["feature_a", "feature_b", "feature_c"]

    dataset = make_modeling_dataset(
        df=df,
        feature_columns=feature_cols,
        target_column="y_high_vol_12_split_safe",
        split_column="split",
    )

    model_config = {
        "name": "logistic_regression",
        "backend": "sklearn",
        "estimator": "logistic_regression",
        "preprocessing": "standard_scaler",
        "params": {"max_iter": 500, "solver": "lbfgs"},
    }

    adapter = build_model_adapter(model_config)
    adapter.fit(dataset.X_train, dataset.y_train)

    proba = adapter.predict_proba(dataset.X_validation)

    assert proba.shape == (len(dataset.X_validation), 2)
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_unknown_backend_raises_not_implemented():
    model_config = {
        "name": "future_torch_model",
        "backend": "pytorch",
        "estimator": "mlp",
        "params": {},
    }

    with pytest.raises(NotImplementedError):
        build_model_adapter(model_config)


def test_compute_binary_metrics():
    y_true = pd.Series([0, 0, 1, 1])
    proba = np.array([0.1, 0.3, 0.7, 0.9])

    metrics = compute_binary_metrics(y_true, proba, threshold=0.5)

    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0


def test_run_training_writes_artifacts(tmp_path: Path):
    df = make_split_dataset()
    split_path = tmp_path / "splits.parquet"
    schema_path = tmp_path / "feature_schema.json"
    artifact_dir = tmp_path / "models"
    report_path = tmp_path / "model_training_report.md"

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

    config = {
        "task_type": "binary_classification",
        "model_version": "test_binary_v1",
        "split_input_path": str(split_path),
        "feature_schema_path": str(schema_path),
        "target_column": "y_high_vol_12_split_safe",
        "split_column": "split",
        "model_artifact_dir": str(artifact_dir),
        "training_report_path": str(report_path),
        "model_splits": {
            "train": "train",
            "validation": "validation",
            "test": "test",
        },
        "models": [
            {
                "name": "majority_baseline",
                "backend": "sklearn",
                "estimator": "dummy_classifier",
                "preprocessing": "none",
                "params": {"strategy": "most_frequent"},
            },
            {
                "name": "logistic_regression",
                "backend": "sklearn",
                "estimator": "logistic_regression",
                "preprocessing": "standard_scaler",
                "params": {"max_iter": 500, "solver": "lbfgs"},
            },
        ],
    }

    result = run_training(config)

    assert Path(result["summary_path"]).exists()
    assert report_path.exists()
    assert (artifact_dir / "majority_baseline" / "model.joblib").exists()
    assert (artifact_dir / "logistic_regression" / "model.joblib").exists()
