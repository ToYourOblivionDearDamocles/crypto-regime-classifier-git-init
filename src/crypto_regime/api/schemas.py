from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Candle(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float | None = None
    num_trades: int | None = None


class PredictionRequest(BaseModel):
    symbol: str = Field(default="BTCUSDT")
    candles: list[Candle]


class BatchPredictionRequest(BaseModel):
    requests: list[PredictionRequest]


class PredictionResponse(BaseModel):
    symbol: str
    feature_timestamp: str
    task_type: str
    model_name: str
    model_version: str
    prediction: Any
    predicted_class: Any | None = None
    probability_high_volatility: float | None = None
    class_probabilities: dict[str, float] | None = None
    decision_threshold: float | None = None
    warnings: list[str] = Field(default_factory=list)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    model_version: str
    task_type: str


class ModelInfoResponse(BaseModel):
    task_type: str
    model_version: str
    model_name: str
    model_dir: str
    num_features: int
    feature_columns: list[str]
    rolling_windows: list[int]
    min_periods_policy: str
    decision_threshold: float | None = None
