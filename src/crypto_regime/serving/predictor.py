from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crypto_regime.features.build_features import build_feature_dataframe
from crypto_regime.serving.model_loader import (
    get_model_classes,
    load_feature_schema_for_model,
    load_json_if_exists,
    load_model_artifact,
    resolve_model_dir,
)


@dataclass
class PredictionResult:
    symbol: str
    feature_timestamp: str
    task_type: str
    model_name: str
    model_version: str
    prediction: Any
    predicted_class: Any | None
    probability_high_volatility: float | None
    class_probabilities: dict[str, float] | None
    decision_threshold: float | None
    warnings: list[str]


class Predictor:
    """
    Framework-aware serving wrapper.

    The Predictor does not assume the model is logistic regression or a tree.
    It only assumes the loaded object exposes a prediction interface:

        binary/multiclass:
            predict_proba or predict

        regression:
            predict

    Future PyTorch/JAX models can be supported by loading an adapter object with
    the same methods.
    """

    def __init__(
        self,
        *,
        task_type: str,
        model_version: str,
        model_artifact_dir: str | Path,
        model_name: str,
        fallback_feature_schema_path: str | Path | None,
        rolling_windows: list[int],
        min_periods_policy: str,
        positive_label: int = 1,
        decision_threshold: float = 0.5,
    ) -> None:
        self.task_type = task_type
        self.model_version = model_version
        self.model_artifact_dir = Path(model_artifact_dir)
        self.model_name = model_name
        self.rolling_windows = rolling_windows
        self.min_periods_policy = min_periods_policy
        self.positive_label = positive_label
        self.decision_threshold = decision_threshold

        self.model_dir = resolve_model_dir(self.model_artifact_dir, model_name)
        self.model = load_model_artifact(self.model_dir)
        self.metadata = load_json_if_exists(self.model_dir / "metadata.json")

        self.feature_schema = load_feature_schema_for_model(
            self.model_dir,
            fallback_feature_schema_path=fallback_feature_schema_path,
        )

        self.feature_columns = list(self.feature_schema["feature_columns"])

        if not self.feature_columns:
            raise ValueError("Loaded feature schema has no feature columns.")

    def model_info(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "model_version": self.model_version,
            "model_name": self.model_name,
            "model_dir": str(self.model_dir),
            "metadata": self.metadata,
            "num_features": len(self.feature_columns),
            "feature_columns": self.feature_columns,
            "rolling_windows": self.rolling_windows,
            "min_periods_policy": self.min_periods_policy,
            "decision_threshold": self.decision_threshold,
        }

    def _prepare_candles(self, candles: list[dict[str, Any]], symbol: str) -> pd.DataFrame:
        if not candles:
            raise ValueError("candles must not be empty.")

        df = pd.DataFrame(candles).copy()

        required = ["timestamp", "open", "high", "low", "close", "volume"]
        missing = [col for col in required if col not in df.columns]

        if missing:
            raise ValueError(f"Missing required candle fields: {missing}")

        df["symbol"] = symbol
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

        if df["timestamp"].isna().any():
            raise ValueError("At least one candle timestamp is invalid.")

        numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume", "num_trades"]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        core_numeric = ["open", "high", "low", "close", "volume"]

        if df[core_numeric].isna().any().any():
            raise ValueError("At least one core OHLCV field is null or non-numeric.")

        if (df[["open", "high", "low", "close"]] <= 0).any().any():
            raise ValueError("OHLC prices must be positive.")

        if (df["volume"] < 0).any():
            raise ValueError("volume must be non-negative.")

        if df["timestamp"].duplicated().any():
            raise ValueError("Duplicate candle timestamps are not allowed.")

        df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

        return df

    def _build_latest_feature_row(self, candles_df: pd.DataFrame) -> tuple[pd.DataFrame, str, list[str]]:
        warnings: list[str] = []

        max_window = max(self.rolling_windows) if self.rolling_windows else 1
        minimum_recommended_rows = max_window + 1

        if len(candles_df) < minimum_recommended_rows:
            warnings.append(
                f"Only {len(candles_df)} candles were provided. "
                f"At least {minimum_recommended_rows} are recommended for full-window features."
            )

        features = build_feature_dataframe(
            candles_df,
            windows=self.rolling_windows,
            min_periods_policy=self.min_periods_policy,
        )

        missing_features = [col for col in self.feature_columns if col not in features.columns]

        if missing_features:
            raise ValueError(
                "Feature generation did not produce required model features: "
                f"{missing_features}"
            )

        latest = features.sort_values(["symbol", "timestamp"]).iloc[[-1]].copy()

        if latest[self.feature_columns].isna().any().any():
            bad_cols = latest[self.feature_columns].columns[
                latest[self.feature_columns].isna().iloc[0]
            ].tolist()

            raise ValueError(
                "Latest feature row contains NaNs. Usually this means not enough "
                f"historical candles were provided. Bad columns: {bad_cols[:20]}"
            )

        feature_timestamp = str(latest["timestamp"].iloc[0])

        return latest[self.feature_columns], feature_timestamp, warnings

    def _predict_binary(self, X: pd.DataFrame) -> tuple[int, float, dict[str, float] | None]:
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)
            classes = get_model_classes(self.model)

            if proba.ndim != 2:
                raise ValueError("predict_proba output must be 2-dimensional.")

            if proba.shape[1] == 1:
                only_class = classes[0] if classes else 0
                positive_probability = 1.0 if int(only_class) == self.positive_label else 0.0
            else:
                if classes is not None and self.positive_label in classes:
                    positive_index = classes.index(self.positive_label)
                else:
                    positive_index = 1

                positive_probability = float(proba[0, positive_index])

            predicted_class = int(positive_probability >= self.decision_threshold)

            class_probabilities = {
                "0": float(1.0 - positive_probability),
                "1": positive_probability,
            }

            return predicted_class, positive_probability, class_probabilities

        if hasattr(self.model, "predict"):
            pred = int(self.model.predict(X)[0])
            return pred, None, None

        raise TypeError("Binary model must implement predict_proba or predict.")

    def _predict_multiclass(self, X: pd.DataFrame) -> tuple[Any, dict[str, float] | None]:
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)
            classes = get_model_classes(self.model)

            predicted_index = int(np.argmax(proba[0]))

            if classes is not None:
                predicted_class = classes[predicted_index]
                class_probabilities = {
                    str(classes[i]): float(proba[0, i])
                    for i in range(proba.shape[1])
                }
            else:
                predicted_class = predicted_index
                class_probabilities = {
                    str(i): float(proba[0, i])
                    for i in range(proba.shape[1])
                }

            return predicted_class, class_probabilities

        if hasattr(self.model, "predict"):
            return self.model.predict(X)[0], None

        raise TypeError("Multiclass model must implement predict_proba or predict.")

    def _predict_regression(self, X: pd.DataFrame) -> float:
        if not hasattr(self.model, "predict"):
            raise TypeError("Regression model must implement predict.")

        return float(self.model.predict(X)[0])

    def predict_one(self, *, symbol: str, candles: list[dict[str, Any]]) -> PredictionResult:
        candles_df = self._prepare_candles(candles, symbol=symbol)
        X_latest, feature_timestamp, warnings = self._build_latest_feature_row(candles_df)

        if self.task_type == "binary_classification":
            predicted_class, probability, class_probabilities = self._predict_binary(X_latest)

            return PredictionResult(
                symbol=symbol,
                feature_timestamp=feature_timestamp,
                task_type=self.task_type,
                model_name=self.model_name,
                model_version=self.model_version,
                prediction=predicted_class,
                predicted_class=predicted_class,
                probability_high_volatility=probability,
                class_probabilities=class_probabilities,
                decision_threshold=self.decision_threshold,
                warnings=warnings,
            )

        if self.task_type == "multiclass_classification":
            predicted_class, class_probabilities = self._predict_multiclass(X_latest)

            return PredictionResult(
                symbol=symbol,
                feature_timestamp=feature_timestamp,
                task_type=self.task_type,
                model_name=self.model_name,
                model_version=self.model_version,
                prediction=predicted_class,
                predicted_class=predicted_class,
                probability_high_volatility=None,
                class_probabilities=class_probabilities,
                decision_threshold=None,
                warnings=warnings,
            )

        if self.task_type == "regression":
            prediction = self._predict_regression(X_latest)

            return PredictionResult(
                symbol=symbol,
                feature_timestamp=feature_timestamp,
                task_type=self.task_type,
                model_name=self.model_name,
                model_version=self.model_version,
                prediction=prediction,
                predicted_class=None,
                probability_high_volatility=None,
                class_probabilities=None,
                decision_threshold=None,
                warnings=warnings,
            )

        raise ValueError(f"Unsupported task_type: {self.task_type}")
