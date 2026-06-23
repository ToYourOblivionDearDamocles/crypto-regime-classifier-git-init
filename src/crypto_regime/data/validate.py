from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


CORE_REQUIRED_COLUMNS = [
    "symbol",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

OPTIONAL_CANONICAL_COLUMNS = [
    "quote_volume",
    "num_trades",
]

CANONICAL_COLUMNS = CORE_REQUIRED_COLUMNS + OPTIONAL_CANONICAL_COLUMNS

OHLC_COLUMNS = ["open", "high", "low", "close"]
NUMERIC_COLUMNS = ["open", "high", "low", "close", "volume", "quote_volume", "num_trades"]


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML config."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Config file must contain a YAML mapping.")

    return config


def resolve_processed_path(config: dict[str, Any]) -> Path:
    """
    Resolve processed candle path.

    Priority:
        1. config["processed_path"], if explicitly provided.
        2. Infer from symbol / interval / start_utc / end_utc / processed_dir.
    """
    if "processed_path" in config:
        return Path(config["processed_path"])

    required_keys = ["symbol", "interval", "start_utc", "end_utc", "processed_dir"]
    missing = [key for key in required_keys if key not in config]

    if missing:
        raise KeyError(
            "Cannot infer processed_path because config is missing keys: "
            f"{missing}. Either add processed_path or provide all required keys."
        )

    symbol = str(config["symbol"])
    interval = str(config["interval"])
    start_utc = str(config["start_utc"])
    end_utc = str(config["end_utc"])
    processed_dir = Path(config["processed_dir"])

    safe_start = start_utc[:10]
    safe_end = end_utc[:10]

    return processed_dir / f"{symbol}_{interval}_{safe_start}_{safe_end}_processed.parquet"


def resolve_report_path(config: dict[str, Any]) -> Path:
    """Resolve data-quality report path."""
    return Path(config.get("data_quality_report_path", "reports/data_quality_report.md"))


def interval_to_timedelta(interval: str) -> pd.Timedelta:
    """Convert interval string like '5m', '1h', '1d' to pandas Timedelta."""
    if len(interval) < 2:
        raise ValueError(f"Invalid interval: {interval}")

    unit = interval[-1]
    value = int(interval[:-1])

    if value <= 0:
        raise ValueError(f"Interval value must be positive: {interval}")

    if unit == "m":
        return pd.Timedelta(minutes=value)

    if unit == "h":
        return pd.Timedelta(hours=value)

    if unit == "d":
        return pd.Timedelta(days=value)

    raise ValueError(f"Unsupported interval: {interval}")


def load_processed_candles(path: str | Path) -> pd.DataFrame:
    """Load processed candles from Parquet or CSV."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Processed data file does not exist: {path}")

    if path.suffix == ".parquet":
        return pd.read_parquet(path)

    if path.suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported processed data file type: {path.suffix}")


def make_issue(
    check_name: str,
    severity: str,
    passed: bool,
    message: str,
    count: int | None = None,
) -> dict[str, Any]:
    """
    Standard validation issue record.

    severity:
        ERROR   = should block downstream ML
        WARNING = should be inspected
        INFO    = descriptive
    """
    return {
        "check_name": check_name,
        "severity": severity,
        "passed": bool(passed),
        "count": count,
        "message": message,
    }


def prepare_candles_for_validation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data types for validation.

    This function does not clean or delete rows.
    It only parses timestamps and numeric columns so checks are consistent.
    """
    prepared = df.copy()

    if "timestamp" in prepared.columns:
        prepared["timestamp"] = pd.to_datetime(
            prepared["timestamp"],
            utc=True,
            errors="coerce",
        )

    for col in NUMERIC_COLUMNS:
        if col in prepared.columns:
            prepared[col] = pd.to_numeric(prepared[col], errors="coerce")

    return prepared


def validate_schema(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Check required and optional canonical columns."""
    issues: list[dict[str, Any]] = []

    missing_core = [col for col in CORE_REQUIRED_COLUMNS if col not in df.columns]
    missing_optional = [col for col in OPTIONAL_CANONICAL_COLUMNS if col not in df.columns]
    extra_columns = [col for col in df.columns if col not in CANONICAL_COLUMNS]

    issues.append(
        make_issue(
            check_name="required_columns_exist",
            severity="ERROR",
            passed=len(missing_core) == 0,
            count=len(missing_core),
            message=f"Missing core required columns: {missing_core}",
        )
    )

    issues.append(
        make_issue(
            check_name="optional_columns_present",
            severity="WARNING",
            passed=len(missing_optional) == 0,
            count=len(missing_optional),
            message=f"Missing optional canonical columns: {missing_optional}",
        )
    )

    issues.append(
        make_issue(
            check_name="extra_columns_detected",
            severity="INFO",
            passed=True,
            count=len(extra_columns),
            message=f"Extra columns present: {extra_columns}",
        )
    )

    return issues


def validate_nulls(df: pd.DataFrame) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Count null values after timestamp and numeric parsing."""
    issues: list[dict[str, Any]] = []

    null_counts = df.isna().sum().sort_values(ascending=False)
    null_counts_df = null_counts.reset_index()
    null_counts_df.columns = ["column", "null_count"]

    core_existing = [col for col in CORE_REQUIRED_COLUMNS if col in df.columns]
    core_null_count = int(df[core_existing].isna().sum().sum()) if core_existing else 0
    total_null_count = int(null_counts.sum())

    issues.append(
        make_issue(
            check_name="core_null_values",
            severity="ERROR",
            passed=core_null_count == 0,
            count=core_null_count,
            message=f"Null values in core required columns: {core_null_count}",
        )
    )

    issues.append(
        make_issue(
            check_name="total_null_values_counted",
            severity="INFO",
            passed=True,
            count=total_null_count,
            message=f"Total null values across all columns: {total_null_count}",
        )
    )

    return issues, null_counts_df


def validate_timestamps(
    df: pd.DataFrame,
    interval: str,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """
    Validate timestamp parsing, sorting, uniqueness, interval consistency,
    and missing candles.
    """
    issues: list[dict[str, Any]] = []

    if "timestamp" not in df.columns:
        issues.append(
            make_issue(
                check_name="timestamp_column_exists",
                severity="ERROR",
                passed=False,
                count=1,
                message="timestamp column is missing.",
            )
        )
        return issues, pd.DataFrame(columns=["symbol", "missing_timestamp"])

    if "symbol" not in df.columns:
        issues.append(
            make_issue(
                check_name="symbol_column_exists",
                severity="ERROR",
                passed=False,
                count=1,
                message="symbol column is missing.",
            )
        )
        return issues, pd.DataFrame(columns=["symbol", "missing_timestamp"])

    timestamp_null_count = int(df["timestamp"].isna().sum())

    issues.append(
        make_issue(
            check_name="timestamps_parse_as_utc",
            severity="ERROR",
            passed=timestamp_null_count == 0,
            count=timestamp_null_count,
            message=f"Unparseable or null timestamps: {timestamp_null_count}",
        )
    )

    if timestamp_null_count > 0:
        return issues, pd.DataFrame(columns=["symbol", "missing_timestamp"])

    duplicate_mask = df.duplicated(subset=["symbol", "timestamp"], keep=False)
    duplicate_count = int(duplicate_mask.sum())

    issues.append(
        make_issue(
            check_name="timestamps_unique_within_symbol",
            severity="ERROR",
            passed=duplicate_count == 0,
            count=duplicate_count,
            message=f"Duplicate rows by symbol/timestamp: {duplicate_count}",
        )
    )

    unsorted_symbols: list[str] = []

    for symbol, group in df.groupby("symbol", sort=False):
        if not group["timestamp"].is_monotonic_increasing:
            unsorted_symbols.append(str(symbol))

    issues.append(
        make_issue(
            check_name="timestamps_sorted_within_symbol",
            severity="ERROR",
            passed=len(unsorted_symbols) == 0,
            count=len(unsorted_symbols),
            message=f"Symbols with unsorted timestamps: {unsorted_symbols}",
        )
    )

    expected_delta = interval_to_timedelta(interval)

    irregular_gap_count = 0
    missing_records: list[dict[str, Any]] = []

    for symbol, group in df.groupby("symbol", sort=False):
        unique_times = (
            group["timestamp"]
            .dropna()
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        )

        if len(unique_times) <= 1:
            continue

        diffs = unique_times.diff().dropna()
        irregular_gap_count += int((diffs != expected_delta).sum())

        expected_range = pd.date_range(
            start=unique_times.min(),
            end=unique_times.max(),
            freq=expected_delta,
            tz="UTC",
        )

        actual_index = pd.DatetimeIndex(unique_times)
        missing_times = expected_range.difference(actual_index)

        for missing_ts in missing_times:
            missing_records.append(
                {
                    "symbol": symbol,
                    "missing_timestamp": missing_ts,
                }
            )

    missing_intervals_df = pd.DataFrame(
        missing_records,
        columns=["symbol", "missing_timestamp"],
    )

    issues.append(
        make_issue(
            check_name="expected_interval_consistent",
            severity="WARNING",
            passed=irregular_gap_count == 0,
            count=irregular_gap_count,
            message=(
                f"Expected interval is {expected_delta}. "
                f"Irregular timestamp gaps: {irregular_gap_count}"
            ),
        )
    )

    issues.append(
        make_issue(
            check_name="missing_intervals_counted",
            severity="WARNING",
            passed=len(missing_intervals_df) == 0,
            count=int(len(missing_intervals_df)),
            message=f"Missing expected candles: {len(missing_intervals_df)}",
        )
    )

    return issues, missing_intervals_df


def validate_duplicate_rows(df: pd.DataFrame) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Check exact duplicate rows."""
    duplicate_mask = df.duplicated(keep=False)
    duplicate_count = int(duplicate_mask.sum())

    issues = [
        make_issue(
            check_name="no_duplicate_rows",
            severity="ERROR",
            passed=duplicate_count == 0,
            count=duplicate_count,
            message=f"Fully duplicated rows: {duplicate_count}",
        )
    ]

    duplicate_rows_df = df.loc[duplicate_mask].head(20).copy()

    return issues, duplicate_rows_df


def validate_ohlcv_values(df: pd.DataFrame) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """
    Validate OHLCV constraints.

    Checks:
        open/high/low/close > 0
        volume >= 0
        high >= open
        high >= close
        low <= open
        low <= close
        high >= low
    """
    issues: list[dict[str, Any]] = []

    required = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        issues.append(
            make_issue(
                check_name="ohlcv_columns_exist",
                severity="ERROR",
                passed=False,
                count=len(missing),
                message=f"Missing OHLCV columns: {missing}",
            )
        )
        return issues, pd.DataFrame()

    failed_rows: list[pd.DataFrame] = []

    checks = [
        (
            "ohlc_prices_positive",
            (df[OHLC_COLUMNS] <= 0).any(axis=1),
            "Rows where at least one OHLC price is non-positive.",
        ),
        (
            "volume_non_negative",
            df["volume"] < 0,
            "Rows where volume is negative.",
        ),
        (
            "high_greater_equal_open",
            df["high"] < df["open"],
            "Rows where high < open.",
        ),
        (
            "high_greater_equal_close",
            df["high"] < df["close"],
            "Rows where high < close.",
        ),
        (
            "low_less_equal_open",
            df["low"] > df["open"],
            "Rows where low > open.",
        ),
        (
            "low_less_equal_close",
            df["low"] > df["close"],
            "Rows where low > close.",
        ),
        (
            "high_greater_equal_low",
            df["high"] < df["low"],
            "Rows where high < low.",
        ),
    ]

    for check_name, mask, message in checks:
        count = int(mask.sum())

        issues.append(
            make_issue(
                check_name=check_name,
                severity="ERROR",
                passed=count == 0,
                count=count,
                message=message,
            )
        )

        if count > 0:
            preview = df.loc[mask].head(20).copy()
            preview.insert(0, "failed_check", check_name)
            failed_rows.append(preview)

    failed_ohlcv_df = (
        pd.concat(failed_rows, ignore_index=True)
        if failed_rows
        else pd.DataFrame()
    )

    return issues, failed_ohlcv_df


def validate_extreme_returns(
    df: pd.DataFrame,
    threshold: float,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """
    Flag extreme one-candle log returns.

    This is a warning, not an automatic deletion rule.
    Crypto can have real jumps, so suspicious returns should be inspected.
    """
    issues: list[dict[str, Any]] = []

    required = ["symbol", "timestamp", "close"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        issues.append(
            make_issue(
                check_name="extreme_return_required_columns_exist",
                severity="ERROR",
                passed=False,
                count=len(missing),
                message=f"Missing columns for extreme-return check: {missing}",
            )
        )
        return issues, pd.DataFrame()

    working = df[["symbol", "timestamp", "close"]].copy()
    working = working.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    working["log_close"] = np.log(working["close"])
    working["log_return_1"] = working.groupby("symbol")["log_close"].diff()
    working["abs_log_return_1"] = working["log_return_1"].abs()

    extreme_mask = working["abs_log_return_1"] > threshold
    extreme_count = int(extreme_mask.sum())

    issues.append(
        make_issue(
            check_name="extreme_returns_flagged",
            severity="WARNING",
            passed=extreme_count == 0,
            count=extreme_count,
            message=(
                f"Rows with abs(one-candle log return) > {threshold}: "
                f"{extreme_count}"
            ),
        )
    )

    extreme_returns_df = (
        working.loc[
            extreme_mask,
            ["symbol", "timestamp", "close", "log_return_1", "abs_log_return_1"],
        ]
        .sort_values("abs_log_return_1", ascending=False)
        .head(50)
        .copy()
    )

    return issues, extreme_returns_df


def summarize_validation(issues_df: pd.DataFrame) -> dict[str, Any]:
    """Summarize validation outcome."""
    failed = issues_df[issues_df["passed"] == False]
    failed_errors = failed[failed["severity"] == "ERROR"]
    failed_warnings = failed[failed["severity"] == "WARNING"]

    return {
        "total_checks": int(len(issues_df)),
        "failed_checks": int(len(failed)),
        "failed_error_checks": int(len(failed_errors)),
        "failed_warning_checks": int(len(failed_warnings)),
        "can_continue": bool(len(failed_errors) == 0),
    }


def dataframe_to_markdown_safe(df: pd.DataFrame, max_rows: int = 50) -> str:
    """Convert DataFrame to Markdown with fallback if tabulate is unavailable."""
    if df is None or df.empty:
        return "_None._"

    small = df.head(max_rows)

    try:
        return small.to_markdown(index=False)
    except Exception:
        return "```text\n" + small.to_string(index=False) + "\n```"


def build_data_quality_report(
    *,
    config: dict[str, Any],
    processed_path: Path,
    issues_df: pd.DataFrame,
    summary: dict[str, Any],
    null_counts_df: pd.DataFrame,
    missing_intervals_df: pd.DataFrame,
    duplicate_rows_df: pd.DataFrame,
    failed_ohlcv_df: pd.DataFrame,
    extreme_returns_df: pd.DataFrame,
) -> str:
    """Build Markdown data-quality report."""
    failed_df = issues_df[issues_df["passed"] == False]

    lines: list[str] = []

    lines.append("# Data Quality Report")
    lines.append("")
    lines.append("## Context")
    lines.append("")
    lines.append(f"- Symbol: `{config.get('symbol', 'UNKNOWN')}`")
    lines.append(f"- Interval: `{config.get('interval', 'UNKNOWN')}`")
    lines.append(f"- Processed path: `{processed_path}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total checks: `{summary['total_checks']}`")
    lines.append(f"- Failed checks: `{summary['failed_checks']}`")
    lines.append(f"- Failed ERROR checks: `{summary['failed_error_checks']}`")
    lines.append(f"- Failed WARNING checks: `{summary['failed_warning_checks']}`")
    lines.append(f"- Can continue to downstream ML pipelines: `{summary['can_continue']}`")
    lines.append("")
    lines.append("## All Checks")
    lines.append("")
    lines.append(dataframe_to_markdown_safe(issues_df, max_rows=100))
    lines.append("")
    lines.append("## Failed Checks")
    lines.append("")
    lines.append(dataframe_to_markdown_safe(failed_df, max_rows=100))
    lines.append("")
    lines.append("## Null Counts")
    lines.append("")
    lines.append(dataframe_to_markdown_safe(null_counts_df, max_rows=100))
    lines.append("")
    lines.append("## Missing Interval Examples")
    lines.append("")
    lines.append(dataframe_to_markdown_safe(missing_intervals_df, max_rows=50))
    lines.append("")
    lines.append("## Duplicate Row Examples")
    lines.append("")
    lines.append(dataframe_to_markdown_safe(duplicate_rows_df, max_rows=50))
    lines.append("")
    lines.append("## Failed OHLCV Row Examples")
    lines.append("")
    lines.append(dataframe_to_markdown_safe(failed_ohlcv_df, max_rows=50))
    lines.append("")
    lines.append("## Extreme Return Examples")
    lines.append("")
    lines.append(dataframe_to_markdown_safe(extreme_returns_df, max_rows=50))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- `ERROR` checks should block feature engineering, label generation, and model training."
    )
    lines.append(
        "- `WARNING` checks should be inspected, but they do not automatically mean the data is unusable."
    )
    lines.append(
        "- This validation step does not modify or delete data. It only reports quality issues."
    )
    lines.append("")

    return "\n".join(lines)


def run_validation(config: dict[str, Any]) -> dict[str, Any]:
    """
    Run data validation and write a Markdown report.

    This function is the main project entry point for Teaching pipeline 2.
    """
    processed_path = resolve_processed_path(config)
    report_path = resolve_report_path(config)

    interval = str(config["interval"])
    extreme_threshold = float(config.get("extreme_abs_log_return_threshold", 0.10))

    raw_df = load_processed_candles(processed_path)
    df = prepare_candles_for_validation(raw_df)

    all_issues: list[dict[str, Any]] = []

    all_issues.extend(validate_schema(df))

    null_issues, null_counts_df = validate_nulls(df)
    all_issues.extend(null_issues)

    timestamp_issues, missing_intervals_df = validate_timestamps(df, interval)
    all_issues.extend(timestamp_issues)

    duplicate_issues, duplicate_rows_df = validate_duplicate_rows(df)
    all_issues.extend(duplicate_issues)

    ohlcv_issues, failed_ohlcv_df = validate_ohlcv_values(df)
    all_issues.extend(ohlcv_issues)

    return_issues, extreme_returns_df = validate_extreme_returns(
        df,
        threshold=extreme_threshold,
    )
    all_issues.extend(return_issues)

    issues_df = pd.DataFrame(all_issues)
    summary = summarize_validation(issues_df)

    report_text = build_data_quality_report(
        config=config,
        processed_path=processed_path,
        issues_df=issues_df,
        summary=summary,
        null_counts_df=null_counts_df,
        missing_intervals_df=missing_intervals_df,
        duplicate_rows_df=duplicate_rows_df,
        failed_ohlcv_df=failed_ohlcv_df,
        extreme_returns_df=extreme_returns_df,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")

    result = {
        "processed_path": str(processed_path),
        "report_path": str(report_path),
        "num_rows": int(len(df)),
        "summary": summary,
    }

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate processed OHLCV candle data."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file, for example configs/data.yaml.",
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Exit with failure if WARNING checks fail.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    args = parse_args()
    config = load_config(args.config)

    result = run_validation(config)

    logging.info("Data validation complete.")
    logging.info("Processed path: %s", result["processed_path"])
    logging.info("Report path: %s", result["report_path"])
    logging.info("Rows checked: %s", result["num_rows"])
    logging.info("Summary: %s", result["summary"])

    summary = result["summary"]

    if not summary["can_continue"]:
        raise SystemExit(1)

    if args.fail_on_warning and summary["failed_warning_checks"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
