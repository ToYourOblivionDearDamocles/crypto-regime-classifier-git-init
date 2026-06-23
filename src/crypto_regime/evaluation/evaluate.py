from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from crypto_regime.evaluation.metrics import compute_metrics
from crypto_regime.models.data import (
    load_feature_schema,
    load_split_dataset,
    resolve_feature_columns,
)


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


def load_json_if_exists(path: str | Path) -> dict[str, Any]:
    """Load JSON if it exists, otherwise return empty dict."""
    path = Path(path)

    if not path.exists():
        return {}

    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Save JSON payload."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def discover_model_artifacts(model_artifact_dir: str | Path) -> list[dict[str, Any]]:
    """
    Discover trained model artifacts from Teaching pipeline 6.

    Expected structure:

        models/saved/crypto_binary_v1/
          logistic_regression/
            model.joblib
            metadata.json
            validation_metrics.json
    """
    root = Path(model_artifact_dir)

    if not root.exists():
        raise FileNotFoundError(f"Model artifact directory does not exist: {root}")

    artifacts: list[dict[str, Any]] = []

    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue

        metadata_path = child / "metadata.json"
        metadata = load_json_if_exists(metadata_path)

        model_path = Path(metadata.get("model_path", child / "model.joblib"))

        if not model_path.exists():
            fallback = child / "model.joblib"
            if fallback.exists():
                model_path = fallback
            else:
                continue

        artifact_format = metadata.get("artifact_format", model_path.suffix.replace(".", ""))

        artifacts.append(
            {
                "name": metadata.get("name", child.name),
                "backend": metadata.get("backend", "unknown"),
                "estimator": metadata.get("estimator", "unknown"),
                "artifact_format": artifact_format,
                "model_path": str(model_path),
                "metadata_path": str(metadata_path) if metadata_path.exists() else None,
                "metadata": metadata,
            }
        )

    if not artifacts:
        raise FileNotFoundError(f"No model artifacts discovered under: {root}")

    return artifacts


def resolve_model_artifacts(config: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Resolve model artifacts from config.

    Current modes:
        models: discover
        models:
          - name: ...
            model_path: ...
            artifact_format: joblib
    """
    models_config = config.get("models", "discover")

    if models_config == "discover":
        return discover_model_artifacts(config["model_artifact_dir"])

    if isinstance(models_config, list):
        artifacts = []

        for item in models_config:
            model_path = Path(item["model_path"])

            if not model_path.exists():
                raise FileNotFoundError(f"Configured model path does not exist: {model_path}")

            artifacts.append(
                {
                    "name": item["name"],
                    "backend": item.get("backend", "unknown"),
                    "estimator": item.get("estimator", "unknown"),
                    "artifact_format": item.get("artifact_format", "joblib"),
                    "model_path": str(model_path),
                    "metadata_path": item.get("metadata_path"),
                    "metadata": load_json_if_exists(item["metadata_path"])
                    if item.get("metadata_path")
                    else {},
                }
            )

        return artifacts

    raise ValueError("models must be either 'discover' or a list of model artifact configs.")


def load_model_artifact(artifact: dict[str, Any]) -> Any:
    """
    Load model artifact.

    Current supported artifact format:
        joblib

    Future PyTorch/JAX support should be added here or through backend-specific loaders.
    """
    artifact_format = str(artifact.get("artifact_format", "joblib")).lower()
    model_path = Path(artifact["model_path"])

    if artifact_format in {"joblib", "pkl", "pickle"}:
        return joblib.load(model_path)

    raise NotImplementedError(
        f"Artifact format '{artifact_format}' is not supported yet. "
        "Add a backend-specific loader for PyTorch/JAX/LightGBM/XGBoost artifacts."
    )


def get_model_classes(model: Any) -> np.ndarray | None:
    """
    Get classifier class labels from sklearn model or sklearn Pipeline.
    """
    classes = getattr(model, "classes_", None)

    if classes is not None:
        return np.asarray(classes)

    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("estimator")
        if estimator is not None and hasattr(estimator, "classes_"):
            return np.asarray(estimator.classes_)

    return None


def predict_for_task(
    model: Any,
    X: pd.DataFrame,
    task_type: str,
    positive_label: int = 1,
) -> dict[str, Any]:
    """
    Generate predictions in a task-aware way.

    Returns a dictionary that may contain:
        y_pred
        y_score
        y_proba
        class_labels
    """
    if task_type == "binary_classification":
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            classes = get_model_classes(model)

            if proba.ndim != 2:
                raise ValueError("predict_proba output must be 2-dimensional.")

            if proba.shape[1] == 1:
                score = np.ones(len(X)) if int(classes[0]) == positive_label else np.zeros(len(X))
            else:
                if classes is not None and positive_label in classes:
                    positive_index = int(np.where(classes == positive_label)[0][0])
                else:
                    positive_index = 1

                score = proba[:, positive_index]

            y_pred = (score >= 0.5).astype(int)

            return {
                "y_pred": y_pred,
                "y_score": score,
                "y_proba": proba,
                "class_labels": classes.tolist() if classes is not None else [0, 1],
            }

        if hasattr(model, "predict"):
            y_pred = model.predict(X)
            return {
                "y_pred": y_pred,
                "y_score": np.asarray(y_pred).astype(float),
                "y_proba": None,
                "class_labels": [0, 1],
            }

        raise TypeError("Binary classification model must implement predict_proba or predict.")

    if task_type == "multiclass_classification":
        y_pred = model.predict(X) if hasattr(model, "predict") else None
        y_proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
        classes = get_model_classes(model)

        if y_pred is None and y_proba is None:
            raise TypeError("Multiclass model must implement predict or predict_proba.")

        return {
            "y_pred": y_pred,
            "y_score": None,
            "y_proba": y_proba,
            "class_labels": classes.tolist() if classes is not None else None,
        }

    if task_type == "regression":
        if not hasattr(model, "predict"):
            raise TypeError("Regression model must implement predict.")

        y_pred = model.predict(X)

        return {
            "y_pred": np.asarray(y_pred).astype(float),
            "y_score": None,
            "y_proba": None,
            "class_labels": None,
        }

    raise ValueError(f"Unsupported task_type: {task_type}")


def build_prediction_frame(
    df_split: pd.DataFrame,
    y_true: pd.Series,
    prediction: dict[str, Any],
    task_type: str,
) -> pd.DataFrame:
    """
    Build a prediction dataframe for saving/debugging.
    """
    pred_df = df_split[["symbol", "timestamp", "split"]].copy()
    pred_df["y_true"] = np.asarray(y_true)

    if task_type == "binary_classification":
        pred_df["y_score"] = prediction["y_score"]
        pred_df["y_pred_at_0_5"] = prediction["y_pred"]

    elif task_type == "multiclass_classification":
        if prediction.get("y_pred") is not None:
            pred_df["y_pred"] = prediction["y_pred"]

        y_proba = prediction.get("y_proba")
        class_labels = prediction.get("class_labels")

        if y_proba is not None:
            for idx in range(y_proba.shape[1]):
                label = class_labels[idx] if class_labels is not None else idx
                pred_df[f"proba_class_{label}"] = y_proba[:, idx]

    elif task_type == "regression":
        pred_df["y_pred"] = prediction["y_pred"]
        pred_df["residual"] = pred_df["y_true"] - pred_df["y_pred"]

    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    return pred_df


def flatten_metrics(
    metrics: dict[str, Any],
    prefix: str = "",
) -> dict[str, Any]:
    """
    Flatten nested metric dictionary for summary tables.

    Large objects such as confusion matrices are kept as JSON strings.
    """
    flat: dict[str, Any] = {}

    for key, value in metrics.items():
        new_key = f"{prefix}{key}" if prefix == "" else f"{prefix}.{key}"

        if isinstance(value, dict):
            flat.update(flatten_metrics(value, new_key))
        elif isinstance(value, list):
            flat[new_key] = json.dumps(value)
        else:
            flat[new_key] = value

    return flat


def evaluate_one_model_on_split(
    model: Any,
    artifact: dict[str, Any],
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    split_column: str,
    split_name: str,
    task_type: str,
    thresholds: list[float],
    top_k_fractions: list[float],
    positive_label: int,
    prediction_output_dir: Path,
) -> dict[str, Any]:
    """
    Evaluate one model on one split.
    """
    df_split = df[df[split_column] == split_name].copy()

    if df_split.empty:
        raise ValueError(f"Split is empty: {split_name}")

    X = df_split[feature_columns].copy()
    y_true = df_split[target_column].copy()

    prediction = predict_for_task(
        model=model,
        X=X,
        task_type=task_type,
        positive_label=positive_label,
    )

    metrics = compute_metrics(
        task_type=task_type,
        y_true=y_true,
        y_pred=prediction.get("y_pred"),
        y_score=prediction.get("y_score"),
        y_proba=prediction.get("y_proba"),
        thresholds=thresholds,
        top_k_fractions=top_k_fractions,
        positive_label=positive_label,
        class_labels=prediction.get("class_labels"),
    )

    prediction_df = build_prediction_frame(
        df_split=df_split,
        y_true=y_true,
        prediction=prediction,
        task_type=task_type,
    )

    model_prediction_dir = prediction_output_dir / artifact["name"]
    model_prediction_dir.mkdir(parents=True, exist_ok=True)

    prediction_path = model_prediction_dir / f"{split_name}_predictions.parquet"
    prediction_df.to_parquet(prediction_path, index=False)

    return {
        "model_name": artifact["name"],
        "backend": artifact.get("backend"),
        "estimator": artifact.get("estimator"),
        "split": split_name,
        "num_rows": int(len(df_split)),
        "prediction_path": str(prediction_path),
        "metrics": metrics,
    }


def dataframe_to_markdown_safe(df: pd.DataFrame, max_rows: int = 100) -> str:
    """Convert dataframe to markdown with fallback."""
    if df is None or df.empty:
        return "_None._"

    small = df.head(max_rows)

    try:
        return small.to_markdown(index=False)
    except Exception:
        return "```text\n" + small.to_string(index=False) + "\n```"


def build_evaluation_report(
    config: dict[str, Any],
    summary_df: pd.DataFrame,
    feature_columns: list[str],
) -> str:
    """
    Build human-readable evaluation report.
    """
    lines: list[str] = []

    lines.append("# Evaluation Report")
    lines.append("")
    lines.append("## Context")
    lines.append("")
    lines.append(f"- Task type: `{config.get('task_type')}`")
    lines.append(f"- Model version: `{config.get('model_version')}`")
    lines.append(f"- Split input path: `{config.get('split_input_path')}`")
    lines.append(f"- Target column: `{config.get('target_column')}`")
    lines.append(f"- Evaluation splits: `{config.get('evaluation_splits')}`")
    lines.append("")
    lines.append("## Metric Summary")
    lines.append("")
    lines.append(dataframe_to_markdown_safe(summary_df, max_rows=200))
    lines.append("")
    lines.append("## Feature Columns")
    lines.append("")
    lines.append(f"- Number of features: `{len(feature_columns)}`")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "This evaluation pipeline is task-aware. It supports binary classification, "
        "multiclass classification, and regression metrics. The current active project "
        "uses binary classification."
    )
    lines.append("")
    lines.append(
        "For binary high-volatility classification, accuracy alone is not enough. "
        "PR-AUC, recall, precision@top-k, Brier score, and calibration-related metrics "
        "are more informative because high-volatility events are minority events."
    )
    lines.append("")
    lines.append(
        "This pipeline evaluates existing trained model artifacts. It does not fit models "
        "or tune hyperparameters."
    )
    lines.append("")

    return "\n".join(lines)


def run_evaluation(config: dict[str, Any]) -> dict[str, Any]:
    """
    Main entry point for Teaching pipeline 7: Evaluation.
    """
    task_type = str(config["task_type"])

    split_input_path = Path(config["split_input_path"])
    feature_schema_path = Path(config["feature_schema_path"])
    evaluation_output_dir = Path(config["evaluation_output_dir"])
    prediction_output_dir = Path(config["prediction_output_dir"])
    evaluation_report_path = Path(config["evaluation_report_path"])
    evaluation_summary_path = Path(config["evaluation_summary_path"])

    target_column = str(config["target_column"])
    split_column = str(config.get("split_column", "split"))

    evaluation_splits = list(config.get("evaluation_splits", ["validation"]))
    thresholds = [float(x) for x in config.get("thresholds", [0.5])]
    top_k_fractions = [float(x) for x in config.get("top_k_fractions", [0.05, 0.10])]
    positive_label = int(config.get("positive_label", 1))

    evaluation_output_dir.mkdir(parents=True, exist_ok=True)
    prediction_output_dir.mkdir(parents=True, exist_ok=True)

    df = load_split_dataset(split_input_path)
    feature_schema = load_feature_schema(feature_schema_path)

    feature_columns = resolve_feature_columns(
        df=df,
        feature_schema=feature_schema,
    )

    missing_target = target_column not in df.columns
    if missing_target:
        raise ValueError(f"Target column missing from split dataset: {target_column}")

    if df[feature_columns].isna().any().any():
        raise ValueError("Feature matrix contains NaNs. Fix earlier pipeline first.")

    if df[target_column].isna().any():
        raise ValueError(f"Target column contains NaNs: {target_column}")

    artifacts = resolve_model_artifacts(config)

    results: list[dict[str, Any]] = []

    for artifact in artifacts:
        logging.info("Evaluating model: %s", artifact["name"])

        model = load_model_artifact(artifact)

        for split_name in evaluation_splits:
            result = evaluate_one_model_on_split(
                model=model,
                artifact=artifact,
                df=df,
                feature_columns=feature_columns,
                target_column=target_column,
                split_column=split_column,
                split_name=split_name,
                task_type=task_type,
                thresholds=thresholds,
                top_k_fractions=top_k_fractions,
                positive_label=positive_label,
                prediction_output_dir=prediction_output_dir,
            )
            results.append(result)

    summary_rows = []

    for result in results:
        flat = flatten_metrics(result["metrics"])

        summary_rows.append(
            {
                "model_name": result["model_name"],
                "backend": result["backend"],
                "estimator": result["estimator"],
                "split": result["split"],
                "num_rows": result["num_rows"],
                **flat,
                "prediction_path": result["prediction_path"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    summary_payload = {
        "task_type": task_type,
        "model_version": config.get("model_version"),
        "split_input_path": str(split_input_path),
        "feature_schema_path": str(feature_schema_path),
        "target_column": target_column,
        "evaluation_splits": evaluation_splits,
        "num_models": len(artifacts),
        "num_results": len(results),
        "results": results,
    }

    save_json(evaluation_summary_path, summary_payload)

    metrics_table_path = evaluation_output_dir / "metrics_table.csv"
    summary_df.to_csv(metrics_table_path, index=False)

    report_text = build_evaluation_report(
        config=config,
        summary_df=summary_df,
        feature_columns=feature_columns,
    )
    evaluation_report_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_report_path.write_text(report_text, encoding="utf-8")

    return {
        "evaluation_summary_path": str(evaluation_summary_path),
        "evaluation_report_path": str(evaluation_report_path),
        "metrics_table_path": str(metrics_table_path),
        "num_models": len(artifacts),
        "num_results": len(results),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained models with task-aware metrics."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to evaluation config YAML, for example configs/evaluation.yaml.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    args = parse_args()
    config = load_config(args.config)

    result = run_evaluation(config)

    logging.info("Evaluation complete.")
    logging.info("Result: %s", result)


if __name__ == "__main__":
    main()
