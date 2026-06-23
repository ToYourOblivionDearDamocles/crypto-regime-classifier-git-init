import pandas as pd

from crypto_regime.data.validate import (
    CORE_REQUIRED_COLUMNS,
    interval_to_timedelta,
    prepare_candles_for_validation,
    validate_ohlcv_values,
    validate_schema,
)


def test_core_required_columns_document_expected_schema():
    assert CORE_REQUIRED_COLUMNS == [
        "symbol",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]


def test_interval_to_timedelta_converts_supported_intervals():
    assert interval_to_timedelta("5m") == pd.Timedelta(minutes=5)
    assert interval_to_timedelta("1h") == pd.Timedelta(hours=1)
    assert interval_to_timedelta("1d") == pd.Timedelta(days=1)


def test_validate_schema_reports_missing_core_columns():
    issues = validate_schema(pd.DataFrame(columns=["timestamp", "open", "close"]))
    required_issue = next(issue for issue in issues if issue["check_name"] == "required_columns_exist")

    assert required_issue["passed"] is False
    assert required_issue["count"] == 4


def test_prepare_candles_for_validation_parses_timestamp_and_numeric_columns():
    frame = pd.DataFrame(
        {
            "timestamp": ["2025-01-01 00:00:00"],
            "open": ["100.0"],
            "high": ["101.0"],
            "low": ["99.0"],
            "close": ["100.5"],
            "volume": ["12.5"],
        }
    )

    prepared = prepare_candles_for_validation(frame)

    assert str(prepared["timestamp"].dt.tz) == "UTC"
    assert prepared["open"].dtype == "float64"
    assert prepared["volume"].dtype == "float64"


def test_validate_ohlcv_values_flags_invalid_prices():
    frame = pd.DataFrame(
        {
            "open": [100.0],
            "high": [99.0],
            "low": [98.0],
            "close": [101.0],
            "volume": [1.0],
        }
    )

    issues, failed_rows = validate_ohlcv_values(frame)
    failed_checks = [issue["check_name"] for issue in issues if not issue["passed"]]

    assert "high_greater_equal_open" in failed_checks
    assert "high_greater_equal_close" in failed_checks
    assert not failed_rows.empty
