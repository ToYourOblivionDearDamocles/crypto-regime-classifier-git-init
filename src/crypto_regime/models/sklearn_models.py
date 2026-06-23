from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class SklearnBinaryClassifierAdapter:
    """
    Binary classifier adapter for scikit-learn estimators.

    This class satisfies the BinaryModelAdapter protocol.
    """

    def __init__(
        self,
        name: str,
        estimator_name: str,
        estimator: Any,
        preprocessing: str = "none",
        params: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.backend = "sklearn"
        self.task_type = "binary_classification"
        self.estimator_name = estimator_name
        self.preprocessing = preprocessing
        self.params = params or {}

        if preprocessing == "standard_scaler":
            self.model = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("estimator", estimator),
                ]
            )
        elif preprocessing == "none":
            self.model = estimator
        else:
            raise ValueError(f"Unsupported preprocessing option: {preprocessing}")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SklearnBinaryClassifierAdapter":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self.model, "predict_proba"):
            raise TypeError(f"Model {self.name} does not support predict_proba.")

        proba = self.model.predict_proba(X)

        if proba.shape[1] == 1:
            # Defensive handling for rare degenerate cases.
            only_class = int(getattr(self.model, "classes_", [0])[0])
            out = np.zeros((len(X), 2), dtype=float)
            out[:, only_class] = 1.0
            return out

        return proba

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "backend": self.backend,
            "task_type": self.task_type,
            "estimator": self.estimator_name,
            "preprocessing": self.preprocessing,
            "params": self.params,
            "artifact_format": "joblib",
        }


def build_sklearn_binary_classifier(
    model_config: dict[str, Any],
) -> SklearnBinaryClassifierAdapter:
    """
    Build a scikit-learn binary classifier from config.
    """
    name = str(model_config["name"])
    estimator_name = str(model_config["estimator"])
    preprocessing = str(model_config.get("preprocessing", "none"))
    params = dict(model_config.get("params", {}))

    if estimator_name == "dummy_classifier":
        estimator = DummyClassifier(**params)

    elif estimator_name == "logistic_regression":
        estimator = LogisticRegression(**params)

    elif estimator_name == "hist_gradient_boosting_classifier":
        estimator = HistGradientBoostingClassifier(**params)

    else:
        raise ValueError(f"Unsupported sklearn estimator: {estimator_name}")

    return SklearnBinaryClassifierAdapter(
        name=name,
        estimator_name=estimator_name,
        estimator=estimator,
        preprocessing=preprocessing,
        params=params,
    )
