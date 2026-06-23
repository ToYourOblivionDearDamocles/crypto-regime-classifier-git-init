import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from crypto_regime.serving.predictor import Predictor


def make_candles(n: int = 320) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2025-01-01 00:00:00",
        periods=n,
        freq="5min",
        tz="UTC",
    )

    x = np.arange(n)
    close = 100.0 + 0.1 * x + np.sin(x / 10.0)
    open_ = close + 0.05 * np.cos(x)
    high = np.maximum(open_, close) + 0.2
    low = np.minimum(open_, close) - 0.2
    volume = 1000.0 + 20.0 * np.sin(x / 7.0)

    return pd.DataFrame(
        {
            "symbol": "BTCUSDT",
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "quote_volume": volume * close,
            "num_trades": x + 100,
        }
    )


def test_predictor_binary_prediction(tmp_path: Path):
    feature_columns = [
        "log_return_1",
        "open_close_return",
        "high_low_range_pct",
        "candle_body_pct",
        "upper_wick_pct",
        "lower_wick_pct",
        "volume_change",
        "quote_volume_change",
        "num_trades_change",
        "rolling_return_mean_3",
        "rolling_return_std_3",
        "rolling_abs_return_mean_3",
        "rolling_realized_vol_3",
        "rolling_volume_mean_3",
        "rolling_volume_std_3",
        "volume_zscore_3",
        "rolling_high_low_range_mean_3",
        "rolling_drawdown_3",
        "distance_from_rolling_high_3",
        "distance_from_rolling_low_3",
    ]

    candles = make_candles(80)

    from crypto_regime.features.build_features import build_feature_dataframe

    features = build_feature_dataframe(
        candles,
        windows=[3],
        min_periods_policy="full_window",
    ).dropna(subset=feature_columns)

    X = features[feature_columns]
    y = (features["rolling_realized_vol_3"] > features["rolling_realized_vol_3"].median()).astype(int)

    model = LogisticRegression(max_iter=500)
    model.fit(X, y)

    model_root = tmp_path / "models"
    model_dir = model_root / "logistic_regression"
    model_dir.mkdir(parents=True)

    model_path = model_dir / "model.joblib"
    joblib.dump(model, model_path)

    (model_dir / "metadata.json").write_text(
        json.dumps(
            {
                "name": "logistic_regression",
                "backend": "sklearn",
                "estimator": "logistic_regression",
                "artifact_format": "joblib",
                "model_path": str(model_path),
            }
        ),
        encoding="utf-8",
    )

    (model_dir / "feature_schema.json").write_text(
        json.dumps(
            {
                "feature_columns": feature_columns,
                "num_features": len(feature_columns),
            }
        ),
        encoding="utf-8",
    )

    predictor = Predictor(
        task_type="binary_classification",
        model_version="test_v1",
        model_artifact_dir=model_root,
        model_name="logistic_regression",
        fallback_feature_schema_path=None,
        rolling_windows=[3],
        min_periods_policy="full_window",
        positive_label=1,
        decision_threshold=0.5,
    )

    request_candles = candles.tail(20).drop(columns=["symbol"]).to_dict(orient="records")

    result = predictor.predict_one(
        symbol="BTCUSDT",
        candles=request_candles,
    )

    assert result.task_type == "binary_classification"
    assert result.model_name == "logistic_regression"
    assert result.predicted_class in {0, 1}
    assert result.probability_high_volatility is not None
    assert 0.0 <= result.probability_high_volatility <= 1.0
