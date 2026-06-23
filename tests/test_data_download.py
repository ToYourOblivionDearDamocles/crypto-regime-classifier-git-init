import io
import zipfile

import pandas as pd
import pytest

from crypto_regime.data.download import (
    BINANCE_KLINE_COLUMNS,
    CANONICAL_COLUMNS,
    build_binance_monthly_zip_url,
    canonicalize_binance_klines,
    infer_binance_timestamp_unit,
    interval_to_milliseconds,
    month_periods_between,
    read_binance_kline_zip_bytes,
    to_utc_timestamp,
)


def make_raw_kline_df() -> pd.DataFrame:
    rows = [
        [
            1735689600000,
            "100.0",
            "110.0",
            "90.0",
            "105.0",
            "12.5",
            1735689899999,
            "1250.0",
            100,
            "6.0",
            "600.0",
            "0",
        ],
        [
            1735689900000,
            "105.0",
            "115.0",
            "95.0",
            "108.0",
            "10.0",
            1735690199999,
            "1080.0",
            120,
            "5.0",
            "540.0",
            "0",
        ],
    ]

    return pd.DataFrame(rows, columns=BINANCE_KLINE_COLUMNS)


def test_to_utc_timestamp_localizes_naive_timestamp():
    ts = to_utc_timestamp("2025-01-01 00:00:00")

    assert str(ts.tz) == "UTC"
    assert ts.year == 2025
    assert ts.month == 1
    assert ts.day == 1


def test_interval_to_milliseconds():
    assert interval_to_milliseconds("5m") == 300_000
    assert interval_to_milliseconds("1h") == 3_600_000
    assert interval_to_milliseconds("1d") == 86_400_000


def test_interval_to_milliseconds_rejects_bad_interval():
    with pytest.raises(ValueError):
        interval_to_milliseconds("7x")


def test_month_periods_between_single_month():
    periods = month_periods_between(
        "2025-01-01 00:00:00",
        "2025-02-01 00:00:00",
    )

    assert [str(p) for p in periods] == ["2025-01"]


def test_month_periods_between_multiple_months():
    periods = month_periods_between(
        "2025-01-15 00:00:00",
        "2025-03-02 00:00:00",
    )

    assert [str(p) for p in periods] == ["2025-01", "2025-02", "2025-03"]


def test_infer_binance_timestamp_unit_ms():
    values = pd.Series([1735689600000, 1735689900000])

    assert infer_binance_timestamp_unit(values) == "ms"


def test_infer_binance_timestamp_unit_us():
    values = pd.Series([1735689600000000, 1735689900000000])

    assert infer_binance_timestamp_unit(values) == "us"


def test_build_binance_monthly_zip_url():
    url = build_binance_monthly_zip_url(
        symbol="BTCUSDT",
        interval="5m",
        year=2025,
        month=1,
        base_url="https://data.binance.vision",
    )

    assert url == (
        "https://data.binance.vision/data/spot/monthly/klines/"
        "BTCUSDT/5m/BTCUSDT-5m-2025-01.zip"
    )


def test_read_binance_kline_zip_bytes():
    raw_df = make_raw_kline_df()

    csv_bytes = raw_df.to_csv(index=False, header=False).encode("utf-8")

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, mode="w") as zf:
        zf.writestr("BTCUSDT-5m-2025-01.csv", csv_bytes)

    parsed = read_binance_kline_zip_bytes(zip_buffer.getvalue())

    assert list(parsed.columns) == BINANCE_KLINE_COLUMNS
    assert parsed.shape == (2, len(BINANCE_KLINE_COLUMNS))


def test_canonicalize_binance_klines_schema_and_utc_timestamp():
    raw_df = make_raw_kline_df()

    processed = canonicalize_binance_klines(raw_df, default_symbol="BTCUSDT")

    assert list(processed.columns) == CANONICAL_COLUMNS
    assert processed["symbol"].iloc[0] == "BTCUSDT"
    assert str(processed["timestamp"].dt.tz) == "UTC"
    assert processed["timestamp"].is_monotonic_increasing
    assert processed["open"].dtype == "float64"
    assert processed["close"].dtype == "float64"


def test_canonicalize_binance_klines_drops_duplicate_timestamps():
    raw_df = make_raw_kline_df()
    duplicated = pd.concat([raw_df, raw_df.iloc[[0]]], ignore_index=True)

    processed = canonicalize_binance_klines(duplicated, default_symbol="BTCUSDT")

    assert len(processed) == 2
