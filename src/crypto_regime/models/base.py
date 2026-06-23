from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd


class BinaryModelAdapter(Protocol):
    """
    Framework-agnostic interface for binary classification models.

    Any backend can implement this interface:
        - scikit-learn
        - PyTorch
        - JAX
        - LightGBM
        - XGBoost
        - custom neural network

    The training orchestration should not need to know the framework details.
    """

    name: str
    backend: str
    task_type: str

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BinaryModelAdapter":
        """Fit the model."""
        ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return class probabilities with shape:

            (n_samples, 2)

        Column 0 = probability of class 0.
        Column 1 = probability of class 1.
        """
        ...

    def save(self, path: str | Path) -> None:
        """Save model artifact."""
        ...

    def metadata(self) -> dict[str, Any]:
        """Return serializable model metadata."""
        ...
