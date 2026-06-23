from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSON file does not exist: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def load_json_if_exists(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        return {}

    return json.loads(path.read_text(encoding="utf-8"))


def discover_model_names(model_artifact_dir: str | Path) -> list[str]:
    root = Path(model_artifact_dir)

    if not root.exists():
        raise FileNotFoundError(f"Model artifact directory does not exist: {root}")

    return sorted([child.name for child in root.iterdir() if child.is_dir()])


def resolve_model_dir(model_artifact_dir: str | Path, model_name: str) -> Path:
    model_dir = Path(model_artifact_dir) / model_name

    if not model_dir.exists():
        available = discover_model_names(model_artifact_dir)
        raise FileNotFoundError(
            f"Model directory does not exist: {model_dir}. "
            f"Available models: {available}"
        )

    return model_dir


def load_model_artifact(model_dir: str | Path) -> Any:
    """
    Load a saved model artifact.

    Current supported artifact:
        model.joblib

    Future extension:
        PyTorch/JAX/LightGBM/XGBoost loaders can be added here while keeping
        the Predictor interface unchanged.
    """
    model_dir = Path(model_dir)
    metadata = load_json_if_exists(model_dir / "metadata.json")

    model_path = Path(metadata.get("model_path", model_dir / "model.joblib"))

    if not model_path.exists():
        fallback = model_dir / "model.joblib"
        if fallback.exists():
            model_path = fallback
        else:
            raise FileNotFoundError(f"No model artifact found in {model_dir}")

    artifact_format = metadata.get("artifact_format", model_path.suffix.replace(".", ""))

    if artifact_format in {"joblib", "pkl", "pickle"}:
        return joblib.load(model_path)

    raise NotImplementedError(
        f"Unsupported artifact format: {artifact_format}. "
        "Add a backend-specific model loader."
    )


def load_feature_schema_for_model(
    model_dir: str | Path,
    fallback_feature_schema_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Load feature schema.

    Priority:
        1. model_dir/feature_schema.json
        2. fallback_feature_schema_path
    """
    model_dir = Path(model_dir)
    model_schema = model_dir / "feature_schema.json"

    if model_schema.exists():
        return load_json(model_schema)

    if fallback_feature_schema_path is not None:
        return load_json(fallback_feature_schema_path)

    raise FileNotFoundError(
        f"No feature schema found at {model_schema}, "
        "and no fallback_feature_schema_path was provided."
    )


def get_model_classes(model: Any) -> list[Any] | None:
    """
    Get class labels from sklearn estimator or sklearn Pipeline.
    """
    classes = getattr(model, "classes_", None)

    if classes is not None:
        return list(classes)

    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("estimator")
        if estimator is not None and hasattr(estimator, "classes_"):
            return list(estimator.classes_)

    return None
