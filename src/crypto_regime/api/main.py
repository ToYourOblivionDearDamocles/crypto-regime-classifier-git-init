from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException

from crypto_regime.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from crypto_regime.serving.predictor import Predictor


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Serving config does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Serving config must contain a YAML mapping.")

    return config


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    config_path = os.environ.get("CRYPTO_REGIME_SERVING_CONFIG", "configs/serving.yaml")
    return load_yaml(config_path)


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    config = get_config()

    return Predictor(
        task_type=str(config["task_type"]),
        model_version=str(config["model_version"]),
        model_artifact_dir=config["model_artifact_dir"],
        model_name=str(config["default_model_name"]),
        fallback_feature_schema_path=config.get("fallback_feature_schema_path"),
        rolling_windows=[int(x) for x in config["rolling_windows"]],
        min_periods_policy=str(config.get("min_periods_policy", "full_window")),
        positive_label=int(config.get("positive_label", 1)),
        decision_threshold=float(config.get("decision_threshold", 0.5)),
    )


config = get_config()
api_config = config.get("api", {})

app = FastAPI(
    title=api_config.get("title", "Crypto Regime Classifier API"),
    version=api_config.get("version", "0.1.0"),
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    predictor = get_predictor()

    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_name=predictor.model_name,
        model_version=predictor.model_version,
        task_type=predictor.task_type,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    predictor = get_predictor()
    info = predictor.model_info()

    return ModelInfoResponse(
        task_type=info["task_type"],
        model_version=info["model_version"],
        model_name=info["model_name"],
        model_dir=info["model_dir"],
        num_features=info["num_features"],
        feature_columns=info["feature_columns"],
        rolling_windows=info["rolling_windows"],
        min_periods_policy=info["min_periods_policy"],
        decision_threshold=info.get("decision_threshold"),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    predictor = get_predictor()

    try:
        result = predictor.predict_one(
            symbol=request.symbol,
            candles=[candle.model_dump() for candle in request.candles],
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PredictionResponse(**result.__dict__)


@app.post("/batch-predict", response_model=BatchPredictionResponse)
def batch_predict(request: BatchPredictionRequest) -> BatchPredictionResponse:
    predictor = get_predictor()
    responses: list[PredictionResponse] = []

    for item in request.requests:
        try:
            result = predictor.predict_one(
                symbol=item.symbol,
                candles=[candle.model_dump() for candle in item.candles],
            )
            responses.append(PredictionResponse(**result.__dict__))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return BatchPredictionResponse(predictions=responses)
