from crypto_regime.api.schemas import (
    Candle,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)


def test_prediction_request_defaults_symbol():
    request = PredictionRequest(
        candles=[
            Candle(
                timestamp="2025-01-01T00:00:00Z",
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=12.0,
            )
        ]
    )

    assert request.symbol == "BTCUSDT"
    assert request.candles[0].close == 100.5


def test_health_response_requires_model_context():
    response = HealthResponse(
        status="ok",
        model_loaded=True,
        model_name="logistic_regression",
        model_version="crypto_binary_v1",
        task_type="binary_classification",
    )

    assert response.status == "ok"
    assert response.model_loaded is True


def test_prediction_response_carries_binary_probability():
    response = PredictionResponse(
        symbol="BTCUSDT",
        feature_timestamp="2025-01-01T00:00:00Z",
        task_type="binary_classification",
        model_name="logistic_regression",
        model_version="crypto_binary_v1",
        prediction=1,
        predicted_class=1,
        probability_high_volatility=0.72,
        decision_threshold=0.5,
    )

    assert response.probability_high_volatility == 0.72
    assert response.warnings == []
