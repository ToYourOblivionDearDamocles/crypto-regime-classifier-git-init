import numpy as np
import pandas as pd
import pytest

from crypto_regime.labels.binary import (
    add_binary_high_vol_label,
    add_future_label_values,
    audit_future_label_missingness,
    compute_train_only_threshold,
    future_window_sum_by_symbol,
    make_labeled_model_table,
    make_threshold_train_mask,
    summarize_quantile_sensitivity,
)


def make_feature_table(n: int = 100) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2025-01-01 00:00:00",
        periods=n,
        freq="5min",
        tz="UTC",
    )

    # Smooth positive price path with nonconstant returns.
    close = 100.0 * np.exp(
        0.001 * np.arange(n) + 0.01 * np.sin(np.arange(n) / 5.0)
    )

    return pd.DataFrame(
        {
            "symbol": "BTCUSDT",
            "timestamp": timestamps,
            "close": close,
            "feature_a": np.linspace(0.0, 1.0, n),
            "feature_b": np.cos(np.arange(n) / 10.0),
        }
    )


def test_future_window_sum_by_symbol():
    df = pd.DataFrame(
        {
            "symbol": ["A"] * 5,
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    result = future_window_sum_by_symbol(df, value_col="value", horizon=2)

    expected = pd.Series([5.0, 7.0, 9.0, np.nan, np.nan])

    pd.testing.assert_series_equal(
        result.reset_index(drop=True),
        expected,
        check_names=False,
    )


def test_add_future_label_values_creates_expected_columns():
    df = make_feature_table(30)
    out = add_future_label_values(df, horizon=3)

    assert "future_return_3" in out.columns
    assert "future_rv_3" in out.columns
    assert out["future_return_3"].isna().sum() == 3
    assert out["future_rv_3"].isna().sum() == 3


def test_future_return_alignment():
    df = make_feature_table(30)
    out = add_future_label_values(df, horizon=3)

    log_close = np.log(out["close"])

    manual = log_close.iloc[3] - log_close.iloc[0]
    computed = out.loc[0, "future_return_3"]

    assert abs(manual - computed) < 1e-12


def test_future_rv_alignment():
    df = make_feature_table(30)
    out = add_future_label_values(df, horizon=3)

    log_close = np.log(out["close"])

    returns = [
        log_close.iloc[1] - log_close.iloc[0],
        log_close.iloc[2] - log_close.iloc[1],
        log_close.iloc[3] - log_close.iloc[2],
    ]

    manual = np.sqrt(np.sum(np.square(returns)))
    computed = out.loc[0, "future_rv_3"]

    assert abs(manual - computed) < 1e-12


def test_threshold_uses_train_mask_only():
    df = make_feature_table(100)
    out = add_future_label_values(df, horizon=3)

    mask = make_threshold_train_mask(
        out,
        horizon=3,
        train_fraction=0.5,
    )

    threshold = compute_train_only_threshold(
        out,
        horizon=3,
        train_mask=mask,
        quantile=0.8,
    )

    expected = out.loc[mask, "future_rv_3"].dropna().quantile(0.8)

    assert threshold == expected


def test_add_binary_label_creates_nullable_int_label():
    df = make_feature_table(100)
    out = add_future_label_values(df, horizon=3)

    mask = make_threshold_train_mask(out, horizon=3, train_fraction=0.7)

    threshold = compute_train_only_threshold(
        out,
        horizon=3,
        train_mask=mask,
        quantile=0.8,
    )

    labeled = add_binary_high_vol_label(
        out,
        horizon=3,
        volatility_threshold=threshold,
        threshold_train_mask=mask,
    )

    assert "y_high_vol_3" in labeled.columns
    assert str(labeled["y_high_vol_3"].dtype) == "Int64"
    assert "used_for_label_threshold" in labeled.columns
    assert labeled["y_high_vol_3"].isna().sum() == 3


def test_make_labeled_model_table_drops_missing_labels():
    df = make_feature_table(100)
    out = add_future_label_values(df, horizon=5)

    mask = make_threshold_train_mask(out, horizon=5, train_fraction=0.7)

    threshold = compute_train_only_threshold(
        out,
        horizon=5,
        train_mask=mask,
        quantile=0.8,
    )

    labeled = add_binary_high_vol_label(
        out,
        horizon=5,
        volatility_threshold=threshold,
        threshold_train_mask=mask,
    )

    table = make_labeled_model_table(
        labeled,
        horizon=5,
        drop_missing_labels=True,
    )

    assert len(table) == len(labeled) - 5
    assert table["y_high_vol_5"].isna().sum() == 0


def test_audit_future_label_missingness():
    df = make_feature_table(50)
    out = add_future_label_values(df, horizon=4)

    audit = audit_future_label_missingness(out, horizon=4)

    assert audit.loc[0, "missing_future_return"] == 4
    assert audit.loc[0, "missing_future_rv"] == 4


def test_summarize_quantile_sensitivity():
    df = make_feature_table(100)
    out = add_future_label_values(df, horizon=3)

    mask = make_threshold_train_mask(out, horizon=3, train_fraction=0.7)

    sensitivity = summarize_quantile_sensitivity(
        out,
        horizon=3,
        train_mask=mask,
        quantiles=[0.7, 0.8, 0.9],
    )

    assert list(sensitivity["quantile"]) == [0.7, 0.8, 0.9]
    assert "train_positive_rate" in sensitivity.columns
    assert "overall_positive_rate" in sensitivity.columns


def test_invalid_quantile_raises_error():
    df = make_feature_table(100)
    out = add_future_label_values(df, horizon=3)

    mask = make_threshold_train_mask(out, horizon=3, train_fraction=0.7)

    with pytest.raises(ValueError):
        compute_train_only_threshold(
            out,
            horizon=3,
            train_mask=mask,
            quantile=1.5,
        )
