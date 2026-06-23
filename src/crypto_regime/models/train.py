from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from crypto_regime.models.data import (
    load_feature_schema,
    load_split_dataset,
    make_modeling_dataset,
    resolve_feature_columns,
)
from crypto_regime.models.registry import build_model_adapter


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML config."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Config file must contain a YAML mapping.")

    return config


def compute_binary_metrics(
    y_true: pd.Series,
    probability_positive: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Compute lightweight binary metrics for training diagnostics.

    Teaching pipeline 7 will handle full evaluation and calibration.
    """
    y_true_array = np.asarray(y_true).astype(int)
    proba = np.asarray(probability_positive).astype(float)

    y_pred = (proba >= threshold).astype(int)

    metrics: dict[str, Any] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true_array, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_array, y_pred)),
        "precision": float(precision_score(y_true_array, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_array, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_array, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true_array, proba)),
        "positive_rate_true": float(np.mean(y_true_array)),
        "positive_rate_predicted": float(np.mean(y_pred)),
        "mean_probability_positive": float(np.mean(proba)),
    }

    if len(np.unique(y_true_array)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true_array, proba))
        metrics["pr_auc"] = float(average_precision_score(y_true_array, proba))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    return metrics


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Save JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def dataframe_to_markdown_safe(df: pd.DataFrame, max_rows: int = 100) -> str:
    """Convert dataframe to markdown with fallback."""
    if df is None or df.empty:
        return "_None._"

    small = df.head(max_rows)

    try:
        return small.to_markdown(index=False)
    except Exception:
        return "```text\n" + small.to_string(index=False) + "\n```"


def write_training_report(
    path: str | Path,
    config: dict[str, Any],
    model_results: list[dict[str, Any]],
    feature_columns: list[str],
) -> None:
    """Write human-readable training report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for result in model_results:
        validation_metrics = result["validation_metrics"]

        rows.append(
            {
                "model_name": result["model_name"],
                "backend": result["backend"],
                "estimator": result["estimator"],
                "accuracy": validation_metrics.get("accuracy"),
                "balanced_accuracy": validation_metrics.get("balanced_accuracy"),
                "precision": validation_metrics.get("precision"),
                "recall": validation_metrics.get("recall"),
                "f1": validation_metrics.get("f1"),
                "roc_auc": validation_metrics.get("roc_auc"),
                "pr_auc": validation_metrics.get("pr_auc"),
                "brier_score": validation_metrics.get("brier_score"),
            }
        )

    metrics_df = pd.DataFrame(rows)

    lines: list[str] = []

    lines.append("# Model Training Report")
    lines.append("")
    lines.append("## Context")
    lines.append("")
    lines.append(f"- Task type: `{config.get('task_type')}`")
    lines.append(f"- Model version: `{config.get('model_version')}`")
    lines.append(f"- Split input path: `{config.get('split_input_path')}`")
    lines.append(f"- Target column: `{config.get('target_column')}`")
    lines.append(f"- Model artifact directory: `{config.get('model_artifact_dir')}`")
    lines.append("")
    lines.append("## Models Trained")
    lines.append("")
    lines.append(dataframe_to_markdown_safe(metrics_df, max_rows=100))
    lines.append("")
    lines.append("## Feature Columns")
    lines.append("")
    lines.append(f"- Number of features: `{len(feature_columns)}`")
    lines.append("")
    lines.append("```text")
    for col in feature_columns:
        lines.append(col)
    lines.append("```")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "This pipeline trains baseline and first-pass binary classifiers. "
        "It is not the final evaluation pipeline. Full error analysis, threshold tuning, "
        "calibration, and final test evaluation belong to later teaching pipelines."
    )
    lines.append("")
    lines.append(
        "The majority baseline is included intentionally. More complex models must be "
        "compared against it to demonstrate actual modeling value."
    )
    lines.append("")
    lines.append(
        "The modeling interface is adapter-based. New backends such as PyTorch, JAX, "
        "LightGBM, or XGBoost can be added by implementing the same adapter interface."
    )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def train_one_model(
    model_config: dict[str, Any],
    dataset,
    artifact_root: Path,
    model_version: str,
) -> dict[str, Any]:
    """Train one configured model and save artifacts."""
    adapter = build_model_adapter(model_config)

    logging.info("Training model: %s", adapter.name)

    adapter.fit(dataset.X_train, dataset.y_train)

    validation_proba = adapter.predict_proba(dataset.X_validation)[:, 1]

    validation_metrics = compute_binary_metrics(
        y_true=dataset.y_validation,
        probability_positive=validation_proba,
        threshold=0.5,
    )

    model_dir = artifact_root / adapter.name
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.joblib"
    metadata_path = model_dir / "metadata.json"
    validation_metrics_path = model_dir / "validation_metrics.json"
    feature_schema_path = model_dir / "feature_schema.json"

    adapter.save(model_path)

    metadata = {
        **adapter.metadata(),
        "model_version": model_version,
        "model_path": str(model_path),
        "target_column": dataset.target_column,
    }

    save_json(metadata_path, metadata)
    save_json(validation_metrics_path, validation_metrics)

    save_json(
        feature_schema_path,
        {
            "feature_columns": dataset.feature_columns,
            "num_features": len(dataset.feature_columns),
            "target_column": dataset.target_column,
        },
    )

    result = {
        "model_name": adapter.name,
        "backend": metadata["backend"],
        "estimator": metadata.get("estimator"),
        "model_dir": str(model_dir),
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "validation_metrics_path": str(validation_metrics_path),
        "validation_metrics": validation_metrics,
    }

    logging.info("Finished model: %s", adapter.name)
    logging.info("Validation metrics: %s", validation_metrics)

    return result


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    """Main entry point for Teaching pipeline 6: Modeling."""
    if config.get("task_type") != "binary_classification":
        raise ValueError("This training script currently supports binary_classification only.")

    split_input_path = Path(config["split_input_path"])
    feature_schema_path = Path(config["feature_schema_path"])
    artifact_root = Path(config["model_artifact_dir"])
    training_report_path = Path(config["training_report_path"])

    target_column = str(config["target_column"])
    split_column = str(config.get("split_column", "split"))

    split_names = config.get("model_splits", {})
    train_split = split_names.get("train", "train")
    validation_split = split_names.get("validation", "validation")
    test_split = split_names.get("test", "test")

    df = load_split_dataset(split_input_path)
    feature_schema = load_feature_schema(feature_schema_path)

    feature_columns = resolve_feature_columns(
        df=df,
        feature_schema=feature_schema,
    )

    dataset = make_modeling_dataset(
        df=df,
        feature_columns=feature_columns,
        target_column=target_column,
        split_column=split_column,
        train_split=train_split,
        validation_split=validation_split,
        test_split=test_split,
    )

    artifact_root.mkdir(parents=True, exist_ok=True)

    model_results = []

    for model_config in config["models"]:
        result = train_one_model(
            model_config=model_config,
            dataset=dataset,
            artifact_root=artifact_root,
            model_version=str(config["model_version"]),
        )
        model_results.append(result)

    write_training_report(
        path=training_report_path,
        config=config,
        model_results=model_results,
        feature_columns=feature_columns,
    )

    summary_path = artifact_root / "training_summary.json"

    summary = {
        "model_version": config["model_version"],
        "task_type": config["task_type"],
        "split_input_path": str(split_input_path),
        "feature_schema_path": str(feature_schema_path),
        "target_column": target_column,
        "num_features": len(feature_columns),
        "num_train_rows": int(len(dataset.X_train)),
        "num_validation_rows": int(len(dataset.X_validation)),
        "num_test_rows": int(len(dataset.X_test)),
        "models": model_results,
        "training_report_path": str(training_report_path),
    }

    save_json(summary_path, summary)

    return {
        "artifact_root": str(artifact_root),
        "summary_path": str(summary_path),
        "training_report_path": str(training_report_path),
        "num_models": len(model_results),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train binary classification models for crypto regime classification."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to model config YAML, for example configs/model.yaml.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    args = parse_args()
    config = load_config(args.config)

    result = run_training(config)

    logging.info("Model training complete.")
    logging.info("Result: %s", result)


if __name__ == "__main__":
    main()
