from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def is_missing_path(path: str | Path | None) -> bool:
    """Return True when an optional artifact path is blank."""
    return path is None or (isinstance(path, str) and path.strip() == "")


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load YAML config."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"YAML config does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"YAML config must contain a mapping: {path}")

    return config


def load_json_if_exists(path: str | Path | None) -> dict[str, Any]:
    """Load JSON if it exists, otherwise return empty dict."""
    if is_missing_path(path):
        return {}

    path = Path(path)

    if not path.exists() or path.is_dir():
        return {}

    return json.loads(path.read_text(encoding="utf-8"))


def load_text_if_exists(path: str | Path | None) -> str:
    """Load text if it exists, otherwise return an empty string."""
    if is_missing_path(path):
        return ""

    path = Path(path)

    if not path.exists() or path.is_dir():
        return ""

    return path.read_text(encoding="utf-8")


def load_table_if_exists(path: str | Path | None) -> pd.DataFrame:
    """Load Parquet or CSV if it exists, otherwise return empty DataFrame."""
    if is_missing_path(path):
        return pd.DataFrame()

    path = Path(path)

    if not path.exists() or path.is_dir():
        return pd.DataFrame()

    if path.suffix == ".parquet":
        return pd.read_parquet(path)

    if path.suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported table file type: {path.suffix}")


def normalize_timestamp_column(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Parse timestamp column as UTC if present."""
    if df.empty or timestamp_col not in df.columns:
        return df

    out = df.copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col], utc=True, errors="coerce")
    return out


def filter_by_symbol(
    df: pd.DataFrame,
    symbol: str,
    symbol_col: str = "symbol",
) -> pd.DataFrame:
    """Filter dataframe by symbol if symbol column exists."""
    if df.empty or symbol_col not in df.columns:
        return df.copy()

    return df[df[symbol_col] == symbol].copy()


def filter_by_date_range(
    df: pd.DataFrame,
    start,
    end,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Filter dataframe by timestamp range."""
    if df.empty or timestamp_col not in df.columns:
        return df.copy()

    out = df.copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col], utc=True, errors="coerce")

    if start is not None:
        start_ts = (
            pd.Timestamp(start).tz_localize("UTC")
            if pd.Timestamp(start).tzinfo is None
            else pd.Timestamp(start).tz_convert("UTC")
        )
        out = out[out[timestamp_col] >= start_ts]

    if end is not None:
        end_ts = (
            pd.Timestamp(end).tz_localize("UTC")
            if pd.Timestamp(end).tzinfo is None
            else pd.Timestamp(end).tz_convert("UTC")
        )
        out = out[out[timestamp_col] <= end_ts]

    return out.copy()


def join_predictions_to_split_data(
    split_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
) -> pd.DataFrame:
    """
    Join prediction records onto split/labeled data.

    Expected join keys:
        symbol
        timestamp

    If prediction_df is empty, return split_df unchanged.
    """
    if split_df.empty:
        return split_df.copy()

    left = normalize_timestamp_column(split_df, timestamp_col)

    if prediction_df.empty:
        return left.copy()

    right = normalize_timestamp_column(prediction_df, timestamp_col)

    required = [symbol_col, timestamp_col]
    missing_left = [col for col in required if col not in left.columns]
    missing_right = [col for col in required if col not in right.columns]

    if missing_left or missing_right:
        return left.copy()

    duplicate_prediction_cols = [
        col for col in right.columns
        if col in left.columns and col not in required
    ]

    right = right.drop(columns=duplicate_prediction_cols, errors="ignore")

    merged = left.merge(
        right,
        on=required,
        how="left",
        validate="one_to_one",
    )

    return merged


def summarize_available_artifacts(config: dict[str, Any]) -> pd.DataFrame:
    """Return a table showing whether dashboard input artifacts exist."""
    paths = config.get("paths", {})

    rows = []

    for name, path_value in paths.items():
        path = Path(path_value)

        rows.append(
            {
                "artifact": name,
                "path": str(path),
                "exists": path.exists(),
            }
        )

    return pd.DataFrame(rows)


def load_dashboard_artifacts(config: dict[str, Any]) -> dict[str, Any]:
    """Load all dashboard artifacts in a tolerant way."""
    paths = config.get("paths", {})
    cols = config.get("columns", {})

    timestamp_col = cols.get("timestamp", "timestamp")

    split_df = load_table_if_exists(paths.get("split_data_path", ""))
    split_df = normalize_timestamp_column(split_df, timestamp_col)

    prediction_df = load_table_if_exists(paths.get("prediction_path", ""))
    prediction_df = normalize_timestamp_column(prediction_df, timestamp_col)

    joined_df = join_predictions_to_split_data(
        split_df,
        prediction_df,
        timestamp_col=timestamp_col,
        symbol_col=cols.get("symbol", "symbol"),
    )

    metrics_df = load_table_if_exists(paths.get("evaluation_metrics_table_path", ""))

    return {
        "split_df": split_df,
        "prediction_df": prediction_df,
        "timeline_df": joined_df,
        "metrics_df": metrics_df,
        "monitoring_summary": load_json_if_exists(paths.get("monitoring_summary_path", "")),
        "monitoring_report": load_text_if_exists(paths.get("monitoring_report_path", "")),
        "feature_schema": load_json_if_exists(paths.get("feature_schema_path", "")),
        "label_config": load_json_if_exists(paths.get("label_config_path", "")),
        "split_manifest": load_json_if_exists(paths.get("split_manifest_path", "")),
        "training_summary": load_json_if_exists(paths.get("training_summary_path", "")),
        "artifact_status": summarize_available_artifacts(config),
    }
