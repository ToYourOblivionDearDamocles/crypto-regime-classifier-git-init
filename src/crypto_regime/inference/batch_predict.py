from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from crypto_regime.features.build_features import build_feature_dataframe
from crypto_regime.serving.predictor import Predictor


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Config must contain a YAML mapping.")

    return config


def load_candles(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Input candle file does not exist: {path}")

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported input file type: {path.suffix}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    return df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def predict_matrix_for_task(
    predictor: Predictor,
    X: pd.DataFrame,
) -> dict[str, Any]:
    """
    Predict for an already-built feature matrix.
    """
    if predictor.task_type == "binary_classification":
        if hasattr(predictor.model, "predict_proba"):
            proba = predictor.model.predict_proba(X)

            classes = None
            if hasattr(predictor.model, "classes_"):
                classes = list(predictor.model.classes_)
            elif hasattr(predictor.model, "named_steps"):
                estimator = predictor.model.named_steps.get("estimator")
                if estimator is not None and hasattr(estimator, "classes_"):
                    classes = list(estimator.classes_)

            if proba.shape[1] == 1:
                only_class = classes[0] if classes else 0
                score = (
                    pd.Series(1.0, index=X.index)
                    if int(only_class) == predictor.positive_label
                    else pd.Series(0.0, index=X.index)
                )
            else:
                if classes is not None and predictor.positive_label in classes:
                    positive_index = classes.index(predictor.positive_label)
                else:
                    positive_index = 1

                score = pd.Series(proba[:, positive_index], index=X.index)

            pred = (score >= predictor.decision_threshold).astype(int)

            return {
                "prediction": pred,
                "y_score": score,
            }

        pred = pd.Series(predictor.model.predict(X), index=X.index)
        return {
            "prediction": pred,
            "y_score": pred.astype(float),
        }

    if predictor.task_type == "multiclass_classification":
        pred = pd.Series(predictor.model.predict(X), index=X.index)
        return {"prediction": pred}

    if predictor.task_type == "regression":
        pred = pd.Series(predictor.model.predict(X), index=X.index)
        return {"prediction": pred.astype(float)}

    raise ValueError(f"Unsupported task_type: {predictor.task_type}")


def run_batch_prediction(
    config: dict[str, Any],
    input_path: str | Path,
    output_path: str | Path,
    model_name: str | None = None,
) -> dict[str, Any]:
    """
    Run offline batch inference.

    Input should be a candle table with:
        symbol, timestamp, open, high, low, close, volume, ...
    """
    selected_model_name = model_name or str(config["default_model_name"])

    predictor = Predictor(
        task_type=str(config["task_type"]),
        model_version=str(config["model_version"]),
        model_artifact_dir=config["model_artifact_dir"],
        model_name=selected_model_name,
        fallback_feature_schema_path=config.get("fallback_feature_schema_path"),
        rolling_windows=[int(x) for x in config["rolling_windows"]],
        min_periods_policy=str(config.get("min_periods_policy", "full_window")),
        positive_label=int(config.get("positive_label", 1)),
        decision_threshold=float(config.get("decision_threshold", 0.5)),
    )

    candles = load_candles(input_path)

    features = build_feature_dataframe(
        candles,
        windows=predictor.rolling_windows,
        min_periods_policy=predictor.min_periods_policy,
    )

    missing_features = [
        col for col in predictor.feature_columns if col not in features.columns
    ]

    if missing_features:
        raise ValueError(f"Missing required model features: {missing_features}")

    feature_ready = features.dropna(subset=predictor.feature_columns).copy()

    if feature_ready.empty:
        raise ValueError("No rows remain after dropping missing feature values.")

    X = feature_ready[predictor.feature_columns].copy()
    predictions = predict_matrix_for_task(predictor, X)

    output = feature_ready[["symbol", "timestamp", "close"]].copy()
    output["model_name"] = predictor.model_name
    output["model_version"] = predictor.model_version
    output["task_type"] = predictor.task_type
    output["prediction"] = predictions["prediction"].values

    if "y_score" in predictions:
        output["y_score"] = predictions["y_score"].values

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".parquet":
        output.to_parquet(output_path, index=False)
    elif output_path.suffix == ".csv":
        output.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported output file type: {output_path.suffix}")

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "model_name": predictor.model_name,
        "num_input_rows": int(len(candles)),
        "num_prediction_rows": int(len(output)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline batch prediction.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    result = run_batch_prediction(
        config=config,
        input_path=args.input,
        output_path=args.output,
        model_name=args.model_name,
    )

    print(result)


if __name__ == "__main__":
    main()
