from __future__ import annotations

import argparse
import io
import logging
import time
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml


BINANCE_KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]


CANONICAL_COLUMNS = [
    "symbol",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "num_trades",
]


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")

    return config


def to_utc_timestamp(ts: str | pd.Timestamp) -> pd.Timestamp:
    """
    Convert a timestamp-like object to timezone-aware UTC pandas Timestamp.

    This is important because time-series splits, missing-candle checks,
    and later label generation all depend on consistent timestamp semantics.
    """
    t = pd.Timestamp(ts)

    if t.tzinfo is None:
        return t.tz_localize("UTC")

    return t.tz_convert("UTC")


def to_milliseconds_utc(ts: str | pd.Timestamp) -> int:
    """Convert timestamp to Unix milliseconds in UTC."""
    return int(to_utc_timestamp(ts).timestamp() * 1000)


def interval_to_milliseconds(interval: str) -> int:
    """
    Convert Binance interval string to milliseconds.

    Examples:
        "5m" -> 300000
        "1h" -> 3600000
        "1d" -> 86400000
    """
    if len(interval) < 2:
        raise ValueError(f"Invalid interval: {interval}")

    unit = interval[-1]
    value = int(interval[:-1])

    if value <= 0:
        raise ValueError(f"Interval value must be positive: {interval}")

    if unit == "m":
        return value * 60_000

    if unit == "h":
        return value * 60 * 60_000

    if unit == "d":
        return value * 24 * 60 * 60_000

    raise ValueError(f"Unsupported interval: {interval}")


def month_periods_between(
    start_utc: str | pd.Timestamp,
    end_utc: str | pd.Timestamp,
) -> pd.PeriodIndex:
    """
    Return all monthly periods touched by [start_utc, end_utc).

    Example:
        start = 2025-01-15
        end   = 2025-03-02
        returns: 2025-01, 2025-02, 2025-03
    """
    start = to_utc_timestamp(start_utc)
    end = to_utc_timestamp(end_utc)

    if end <= start:
        raise ValueError("end_utc must be after start_utc.")

    end_inclusive = end - pd.Timedelta(nanoseconds=1)

    # Convert to timezone-naive before to_period to avoid pandas timezone warning.
    start_month = start.tz_convert(None).to_period("M")
    end_month = end_inclusive.tz_convert(None).to_period("M")

    return pd.period_range(start=start_month, end=end_month, freq="M")


def infer_binance_timestamp_unit(values: pd.Series) -> str:
    """
    Infer whether Binance timestamps are milliseconds or microseconds.

    Heuristic:
        around 1e12 -> milliseconds
        around 1e15 -> microseconds
    """
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna()

    if numeric.empty:
        raise ValueError("Cannot infer timestamp unit from empty/non-numeric values.")

    median_value = numeric.median()

    if median_value > 1e14:
        return "us"

    return "ms"


def build_binance_monthly_zip_url(
    symbol: str,
    interval: str,
    year: int,
    month: int,
    base_url: str,
) -> str:
    """Build Binance public-data monthly kline ZIP URL."""
    yyyy_mm = f"{year}-{month:02d}"

    return (
        f"{base_url}/data/spot/monthly/klines/"
        f"{symbol}/{interval}/{symbol}-{interval}-{yyyy_mm}.zip"
    )


def read_binance_kline_zip_bytes(content: bytes) -> pd.DataFrame:
    """
    Read one Binance monthly ZIP file from bytes.

    Returns the raw Binance kline schema.
    """
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        csv_names = [name for name in zf.namelist() if name.endswith(".csv")]

        if not csv_names:
            raise ValueError("No CSV file found inside Binance ZIP.")

        with zf.open(csv_names[0]) as f:
            df = pd.read_csv(f, header=None)

    # Some files may include a header row. Handle both header/no-header cases.
    if not df.empty and str(df.iloc[0, 0]).lower() == "open_time":
        df = df.iloc[1:].reset_index(drop=True)

    if df.shape[1] != len(BINANCE_KLINE_COLUMNS):
        raise ValueError(
            f"Expected {len(BINANCE_KLINE_COLUMNS)} columns, got {df.shape[1]}."
        )

    df.columns = BINANCE_KLINE_COLUMNS
    return df


def download_binance_monthly_klines(
    symbol: str,
    interval: str,
    start_utc: str,
    end_utc: str,
    base_url: str,
) -> pd.DataFrame:
    """
    Download Binance public historical monthly kline ZIP files.

    This is the primary source for historical ingestion.
    """
    months = month_periods_between(start_utc, end_utc)
    frames: list[pd.DataFrame] = []

    for period in months:
        url = build_binance_monthly_zip_url(
            symbol=symbol,
            interval=interval,
            year=period.year,
            month=period.month,
            base_url=base_url,
        )

        logging.info("Downloading Binance monthly kline ZIP: %s", url)
        response = requests.get(url, timeout=60)

        if response.status_code != 200:
            raise RuntimeError(
                "Monthly ZIP download failed. "
                f"status={response.status_code}, url={url}, "
                f"body_preview={response.text[:300]!r}"
            )

        month_df = read_binance_kline_zip_bytes(response.content)
        month_df.insert(0, "symbol", symbol)
        frames.append(month_df)

    if not frames:
        raise RuntimeError("No Binance monthly kline data downloaded.")

    raw_df = pd.concat(frames, ignore_index=True)

    open_time_unit = infer_binance_timestamp_unit(raw_df["open_time"])

    timestamps = pd.to_datetime(
        pd.to_numeric(raw_df["open_time"], errors="coerce"),
        unit=open_time_unit,
        utc=True,
    )

    start = to_utc_timestamp(start_utc)
    end = to_utc_timestamp(end_utc)

    mask = (timestamps >= start) & (timestamps < end)
    raw_df = raw_df.loc[mask].reset_index(drop=True)

    if raw_df.empty:
        raise RuntimeError(
            "Downloaded monthly files, but no rows remained after date filtering."
        )

    return raw_df


def download_binance_rest_klines(
    symbol: str,
    interval: str,
    start_utc: str,
    end_utc: str,
    base_url: str,
    limit: int = 1000,
    sleep_seconds: float = 0.2,
) -> pd.DataFrame:
    """
    Backup Binance REST kline downloader.

    This is useful when monthly files are unavailable or the requested period
    is too recent for the historical archive.
    """
    endpoint = f"{base_url}/api/v3/klines"

    start_ms = to_milliseconds_utc(start_utc)
    end_ms = to_milliseconds_utc(end_utc)
    interval_ms = interval_to_milliseconds(interval)

    all_rows: list[list[Any]] = []
    current_start = start_ms

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": limit,
        }

        response = requests.get(endpoint, params=params, timeout=30)

        if response.status_code != 200:
            raise RuntimeError(
                "Binance REST request failed. "
                f"status={response.status_code}, url={response.url}, "
                f"body_preview={response.text[:300]!r}"
            )

        rows = response.json()

        if not rows:
            break

        all_rows.extend(rows)

        last_open_time = int(rows[-1][0])
        next_start = last_open_time + interval_ms

        if next_start <= current_start:
            raise RuntimeError("REST pagination failed: next_start did not advance.")

        current_start = next_start
        time.sleep(sleep_seconds)

    if not all_rows:
        raise RuntimeError("Binance REST API returned no kline rows.")

    raw_df = pd.DataFrame(all_rows, columns=BINANCE_KLINE_COLUMNS)
    raw_df.insert(0, "symbol", symbol)

    return raw_df


def canonicalize_binance_klines(
    raw_df: pd.DataFrame,
    default_symbol: str,
) -> pd.DataFrame:
    """
    Convert raw Binance kline data into the project's canonical processed schema.

    Output schema:
        symbol
        timestamp
        open
        high
        low
        close
        volume
        quote_volume
        num_trades
    """
    df = raw_df.copy()

    if "symbol" not in df.columns:
        df.insert(0, "symbol", default_symbol)

    missing = [col for col in BINANCE_KLINE_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(f"Missing Binance raw kline columns: {missing}")

    numeric_columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    open_time_unit = infer_binance_timestamp_unit(df["open_time"])

    processed = pd.DataFrame(
        {
            "symbol": df["symbol"].astype(str),
            "timestamp": pd.to_datetime(df["open_time"], unit=open_time_unit, utc=True),
            "open": df["open"].astype(float),
            "high": df["high"].astype(float),
            "low": df["low"].astype(float),
            "close": df["close"].astype(float),
            "volume": df["volume"].astype(float),
            "quote_volume": df["quote_asset_volume"].astype(float),
            "num_trades": df["number_of_trades"].astype("Int64"),
        }
    )

    processed = (
        processed.sort_values(["symbol", "timestamp"])
        .drop_duplicates(subset=["symbol", "timestamp"])
        .reset_index(drop=True)
    )

    return processed[CANONICAL_COLUMNS]


def run_minimal_ingestion_checks(processed_df: pd.DataFrame) -> None:
    """
    Minimal Stage 1 checks.

    Full data validation belongs in Stage 2. These checks only confirm that
    ingestion produced a usable canonical dataset.
    """
    missing = [col for col in CANONICAL_COLUMNS if col not in processed_df.columns]

    if missing:
        raise ValueError(f"Processed data is missing canonical columns: {missing}")

    if processed_df.empty:
        raise ValueError("Processed dataframe is empty.")

    if processed_df["timestamp"].dt.tz is None:
        raise ValueError("timestamp column must be timezone-aware UTC.")

    if not processed_df["timestamp"].is_monotonic_increasing:
        raise ValueError("timestamp column must be sorted.")

    core_numeric = ["open", "high", "low", "close", "volume"]

    if processed_df[core_numeric].isna().any().any():
        raise ValueError("NaN detected in core OHLCV columns.")

    if (processed_df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Non-positive OHLC price detected.")

    if (processed_df["volume"] < 0).any():
        raise ValueError("Negative volume detected.")


def run_ingestion(config: dict[str, Any]) -> dict[str, Any]:
    """
    Run Stage 1 data ingestion.

    Steps:
        1. Try Binance public monthly ZIP files.
        2. If enabled and necessary, fallback to Binance REST API.
        3. Save raw CSV.
        4. Canonicalize and save processed Parquet.
        5. Return a small metadata summary.
    """
    symbol = str(config["symbol"])
    interval = str(config["interval"])
    start_utc = str(config["start_utc"])
    end_utc = str(config["end_utc"])

    raw_dir = Path(config["raw_dir"])
    processed_dir = Path(config["processed_dir"])

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_df: pd.DataFrame | None = None
    data_source: str | None = None

    try:
        raw_df = download_binance_monthly_klines(
            symbol=symbol,
            interval=interval,
            start_utc=start_utc,
            end_utc=end_utc,
            base_url=str(config["binance_public_data_base_url"]),
        )
        data_source = "binance_public_monthly_zip"

    except Exception as exc:
        logging.warning("Primary monthly ZIP source failed: %s", repr(exc))

    use_rest_backup = bool(config.get("use_rest_backup", True))

    if raw_df is None and use_rest_backup:
        raw_df = download_binance_rest_klines(
            symbol=symbol,
            interval=interval,
            start_utc=start_utc,
            end_utc=end_utc,
            base_url=str(config["binance_rest_base_url"]),
        )
        data_source = "binance_rest_api"

    if raw_df is None:
        raise RuntimeError(
            "Ingestion failed: monthly ZIP failed and REST backup was disabled or failed."
        )

    safe_start = start_utc[:10]
    safe_end = end_utc[:10]

    raw_path = raw_dir / f"{symbol}_{interval}_{safe_start}_{safe_end}_raw.csv"
    processed_path = (
        processed_dir / f"{symbol}_{interval}_{safe_start}_{safe_end}_processed.parquet"
    )

    raw_df.to_csv(raw_path, index=False)
    logging.info("Saved raw data: %s", raw_path)

    processed_df = canonicalize_binance_klines(raw_df, default_symbol=symbol)
    run_minimal_ingestion_checks(processed_df)

    processed_df.to_parquet(processed_path, index=False)
    logging.info("Saved processed data: %s", processed_path)

    result = {
        "data_source": data_source,
        "symbol": symbol,
        "interval": interval,
        "raw_path": str(raw_path),
        "processed_path": str(processed_path),
        "num_rows": int(len(processed_df)),
        "start_timestamp": str(processed_df["timestamp"].min()),
        "end_timestamp": str(processed_df["timestamp"].max()),
        "columns": list(processed_df.columns),
    }

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1 data ingestion for crypto regime classifier."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file, for example configs/data.yaml.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    args = parse_args()
    config = load_config(args.config)

    result = run_ingestion(config)

    logging.info("Stage 1 ingestion complete.")
    logging.info("Result: %s", result)


if __name__ == "__main__":
    main()
