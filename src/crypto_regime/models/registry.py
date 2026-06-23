from __future__ import annotations

from typing import Any

from crypto_regime.models.base import BinaryModelAdapter
from crypto_regime.models.sklearn_models import build_sklearn_binary_classifier


def build_model_adapter(model_config: dict[str, Any]) -> BinaryModelAdapter:
    """
    Build a model adapter from config.

    Current implemented backend:
        sklearn

    Future backend pattern:
        backend: pytorch
        backend: jax
        backend: lightgbm
        backend: xgboost

    Each future backend should implement the BinaryModelAdapter interface.
    """
    backend = str(model_config["backend"])

    if backend == "sklearn":
        return build_sklearn_binary_classifier(model_config)

    if backend in {"pytorch", "jax", "lightgbm", "xgboost"}:
        raise NotImplementedError(
            f"Backend '{backend}' is not implemented yet. "
            "Add an adapter that implements BinaryModelAdapter, then register it here."
        )

    raise ValueError(f"Unknown model backend: {backend}")
