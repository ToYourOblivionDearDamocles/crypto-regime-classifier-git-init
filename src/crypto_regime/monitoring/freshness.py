from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_table(path: str | Path) -> pd.DataFrame:
    """Load Parquet or CSV table."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Table does not exist: {path}")

    if path.suffix == ".parquet":
        return pd.read_parquet(path)

    if path.suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported file type: {path.suffix}")


def interval_to_timedelta(interval: str) -> pd.Timedelta:
    """Convert interval string like '5m', '1h', '1d' to pandas Timedelta."""
    if len(interval) < 2:
        raise ValueError(f"Invalid interval: {interval}")

    value = int(interval[:-1])
    unit = interval[-1]

    if value <= 0:
        raise ValueError(f"Interval value must be positive: {interval}")

    if unit == "m":
        return pd.Timedelta(minutes=value)

    if unit == "h":
        return pd.Timedelta(hours=value)

    if unit == "d":
        return pd.Timedelta(days=value)

    raise ValueError(f"Unsupported interval: {interval}")


def parse_reference_time(reference_time_utc: str | None) -> pd.Timestamp:
    """
    Parse reference time.

    If None, use current UTC time.
    """
    if reference_time_utc is None:
        return pd.Timestamp.now(tz="UTC")

    return pd.Timestamp(reference_time_utc).tz_convert("UTC") if pd.Timestamp(reference_time_utc).tzinfo else pd.Timestamp(reference_time_utc).tz_localize("UTC")


def compute_missing_intervals(
    timestamps: pd.Series,
    expected_interval: pd.Timedelta,
) -> pd.DatetimeIndex:
    """Compute missing timestamps inside observed min/max range."""
    times = (
        pd.to_datetime(timestamps, utc=True, errors="coerce")
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    if len(times) <= 1:
        return pd.DatetimeIndex([], tz="UTC")

    expected = pd.date_range(
        start=times.min(),
        end=times.max(),
        freq=expected_interval,
        tz="UTC",
    )

    return expected.difference(pd.DatetimeIndex(times))


def compute_freshness_report(
    df: pd.DataFrame,
    *,
    timestamp_column: str,
    symbol_column: str | None,
    expected_interval: str,
    mode: str = "historical",
    max_allowed_delay_minutes: int = 15,
    reference_time_utc: str | None = None,
) -> dict[str, Any]:
    """
    Compute data freshness report.

    mode:
        historical:
            Report dataset coverage and missing intervals.
            Do not fail because the dataset is old.

        live:
            Compare latest timestamp against current/reference time.
    """
    if timestamp_column not in df.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_column}")

    working = df.copy()
    working[timestamp_column] = pd.to_datetime(
        working[timestamp_column],
        utc=True,
        errors="coerce",
    )

    if working[timestamp_column].isna().any():
        raise ValueError("Timestamp column contains null/unparseable values.")

    expected_delta = interval_to_timedelta(expected_interval)

    if symbol_column is not None and symbol_column in working.columns:
        group_items = working.groupby(symbol_column, sort=False)
    else:
        group_items = [("ALL", working)]

    symbol_reports: list[dict[str, Any]] = []

    for symbol, group in group_items:
        timestamps = group[timestamp_column].sort_values()
        missing = compute_missing_intervals(timestamps, expected_delta)

        latest_timestamp = timestamps.max()
        earliest_timestamp = timestamps.min()

        if mode == "live":
            reference_time = parse_reference_time(reference_time_utc)
            delay_minutes = (reference_time - latest_timestamp).total_seconds() / 60.0
            is_fresh = delay_minutes <= max_allowed_delay_minutes
        elif mode == "historical":
            reference_time = None
            delay_minutes = None
            is_fresh = True
        else:
            raise ValueError("mode must be either 'historical' or 'live'.")

        symbol_reports.append(
            {
                "symbol": str(symbol),
                "num_rows": int(len(group)),
                "earliest_timestamp": str(earliest_timestamp),
                "latest_timestamp": str(latest_timestamp),
                "expected_interval": str(expected_delta),
                "missing_interval_count": int(len(missing)),
                "first_missing_timestamps": [str(x) for x in missing[:10]],
                "mode": mode,
                "reference_time_utc": str(reference_time) if reference_time is not None else None,
                "delay_minutes": float(delay_minutes) if delay_minutes is not None else None,
                "max_allowed_delay_minutes": int(max_allowed_delay_minutes),
                "is_fresh": bool(is_fresh),
            }
        )

    all_fresh = all(item["is_fresh"] for item in symbol_reports)
    total_missing = sum(item["missing_interval_count"] for item in symbol_reports)

    return {
        "check_name": "data_freshness",
        "mode": mode,
        "passed": bool(all_fresh),
        "total_missing_interval_count": int(total_missing),
        "symbols": symbol_reports,
    }


def run_freshness_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Run freshness check from config section."""
    df = load_table(config["data_path"])

    return compute_freshness_report(
        df,
        timestamp_column=str(config.get("timestamp_column", "timestamp")),
        symbol_column=config.get("symbol_column", "symbol"),
        expected_interval=str(config["expected_interval"]),
        mode=str(config.get("mode", "historical")),
        max_allowed_delay_minutes=int(config.get("max_allowed_delay_minutes", 15)),
        reference_time_utc=config.get("reference_time_utc"),
    )
