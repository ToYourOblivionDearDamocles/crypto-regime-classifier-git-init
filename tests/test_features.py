import numpy as np
import pandas as pd
import pytest

from crypto_regime.features.build_features import (
    add_basic_candle_features,
    add_rolling_features,
    audit_feature_missingness,
    build_feature_dataframe,
    check_past_only_features,
    get_feature_columns,
    make_model_feature_table,
)


def make_test_candles(n: int = 400) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2025-01-01 00:00:00",
        periods=n,
        freq="5min",
        tz="UTC",
    )

    base = 100.0 + np.linspace(0, 10, n)
    wave = 0.5 * np.sin(np.arange(n) / 10.0)
    close = base + wave
    open_ = close + 0.05 * np.cos(np.arange(n))
    high = np.maximum(open_, close) + 0.2
    low = np.minimum(open_, close) - 0.2
    volume = 1000.0 + 20.0 * np.sin(np.arange(n) / 7.0)

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
            "num_trades": np.arange(n) + 100,
        }
    )


def test_basic_features_are_created():
    candles = make_test_candles(20)

    features = add_basic_candle_features(candles)

    expected = [
        "log_return_1",
        "open_close_return",
        "high_low_range_pct",
        "candle_body_pct",
        "upper_wick_pct",
        "lower_wick_pct",
        "volume_change",
        "quote_volume_change",
        "num_trades_change",
    ]

    for col in expected:
        assert col in features.columns


def test_rolling_features_are_created():
    candles = make_test_candles(50)
    basic = add_basic_candle_features(candles)

    features = add_rolling_features(
        basic,
        windows=[3, 12],
        min_periods_policy="full_window",
    )

    expected = [
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
        "rolling_realized_vol_12",
    ]

    for col in expected:
        assert col in features.columns


def test_feature_columns_exclude_raw_and_helper_columns():
    candles = make_test_candles(50)

    features = build_feature_dataframe(
        candles,
        windows=[3, 12],
        min_periods_policy="full_window",
    )

    feature_cols = get_feature_columns(features)

    forbidden = {
        "symbol",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "num_trades",
        "log_close",
    }

    assert not any(col in forbidden for col in feature_cols)
    assert "log_return_1" in feature_cols
    assert "rolling_realized_vol_12" in feature_cols


def test_rolling_features_have_expected_warmup_nans():
    candles = make_test_candles(50)

    features = build_feature_dataframe(
        candles,
        windows=[12],
        min_periods_policy="full_window",
    )

    assert features["rolling_realized_vol_12"].isna().sum() > 0
    assert features["rolling_realized_vol_12"].notna().sum() > 0


def test_make_model_feature_table_drops_warmup_rows():
    candles = make_test_candles(50)

    features = build_feature_dataframe(
        candles,
        windows=[12],
        min_periods_policy="full_window",
    )

    feature_cols = get_feature_columns(features)

    table = make_model_feature_table(
        features,
        feature_cols=feature_cols,
        drop_missing_features=True,
    )

    assert len(table) < len(features)
    assert table[feature_cols].isna().sum().sum() == 0


def test_audit_feature_missingness_marks_warmup_as_expected():
    candles = make_test_candles(50)

    features = build_feature_dataframe(
        candles,
        windows=[12],
        min_periods_policy="full_window",
    )

    feature_cols = get_feature_columns(features)

    audit = audit_feature_missingness(
        features,
        feature_cols=feature_cols,
        windows=[12],
    )

    row = audit[audit["feature"] == "rolling_realized_vol_12"].iloc[0]

    assert row["missing_count"] > 0
    assert row["interpretation"] == "expected_rolling_warmup"


def test_past_only_feature_check_passes():
    candles = make_test_candles(400)

    check_past_only_features(
        candles=candles,
        windows=[3, 12, 48],
        min_periods_policy="full_window",
    )


def test_no_future_or_label_columns_in_features():
    candles = make_test_candles(50)

    features = build_feature_dataframe(
        candles,
        windows=[3, 12],
        min_periods_policy="full_window",
    )

    features["future_return"] = 1.0
    features["label"] = 0
    features["target"] = 0
    features["y_binary"] = 0

    feature_cols = get_feature_columns(features)

    assert "future_return" not in feature_cols
    assert "label" not in feature_cols
    assert "target" not in feature_cols
    assert "y_binary" not in feature_cols


def test_invalid_min_periods_policy_raises_error():
    candles = make_test_candles(50)
    basic = add_basic_candle_features(candles)

    with pytest.raises(ValueError):
        add_rolling_features(
            basic,
            windows=[12],
            min_periods_policy="bad_policy",
        )
