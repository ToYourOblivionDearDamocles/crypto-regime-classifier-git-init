from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml


REQUIRED_INPUT_COLUMNS = [
    "symbol",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

OPTIONAL_INPUT_COLUMNS = [
    "quote_volume",
    "num_trades",
]

RAW_COLUMNS = {
    "symbol",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "num_trades",
}

HELPER_COLUMNS = {
    "log_close",
}

FORBIDDEN_FEATURE_PREFIXES = (
    "future_",
    "target",
    "label",
    "y_",
)


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


def load_processed_candles(path: str | Path) -> pd.DataFrame:
    """Load processed OHLCV candles from Parquet or CSV."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Processed candle file does not exist: {path}")

    if path.suffix == ".parquet":
        return pd.read_parquet(path)

    if path.suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported file type: {path.suffix}")


def validate_feature_input(df: pd.DataFrame) -> None:
    """
    Minimal input validation for feature engineering.

    Full data validation belongs to Teaching pipeline 2.
    This function only ensures this module receives the expected candle schema.
    """
    missing = [col for col in REQUIRED_INPUT_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required input columns: {missing}")

    if df.empty:
        raise ValueError("Input candle dataframe is empty.")


def prepare_candles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse timestamps and numeric columns, then sort by symbol/time.

    This does not clean or impute data. It only prepares types for feature logic.
    """
    validate_feature_input(df)

    out = df.copy()

    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "num_trades",
    ]

    for col in numeric_columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    return out


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two series and replace infinite values with NaN."""
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan)


def pct_change_by_symbol(df: pd.DataFrame, column: str) -> pd.Series:
    """Compute percentage change within each symbol."""
    if column not in df.columns:
        raise ValueError(f"Column does not exist: {column}")

    result = df.groupby("symbol", sort=False)[column].pct_change()
    return result.replace([np.inf, -np.inf], np.nan)


def rolling_by_symbol(
    df: pd.DataFrame,
    column: str,
    window: int,
    agg: str,
    min_periods: int,
) -> pd.Series:
    """
    Compute a backward-looking rolling aggregation within each symbol.

    At row t, the rolling window uses only rows:
        t, t-1, ..., t-window+1
    """
    if column not in df.columns:
        raise ValueError(f"Column does not exist: {column}")

    return df.groupby("symbol", sort=False)[column].transform(
        lambda s: s.rolling(window=window, min_periods=min_periods).agg(agg)
    )


def rolling_realized_vol_by_symbol(
    df: pd.DataFrame,
    return_col: str,
    window: int,
    min_periods: int,
) -> pd.Series:
    """
    Backward-looking realized volatility:

        RV_w(t) = sqrt(sum_{k=0}^{w-1} r_{t-k}^2)

    This is a feature, not the future-volatility label.
    """
    if return_col not in df.columns:
        raise ValueError(f"Return column does not exist: {return_col}")

    temp = df[["symbol"]].copy()
    temp["squared_return"] = df[return_col] ** 2

    return temp.groupby("symbol", sort=False)["squared_return"].transform(
        lambda s: np.sqrt(s.rolling(window=window, min_periods=min_periods).sum())
    )


def resolve_min_periods(window: int, policy: str) -> int:
    """
    Resolve min_periods for rolling features.

    full_window:
        Require a complete window. Produces warmup NaNs.

    allow_partial:
        Allow partial windows. Fewer NaNs, but early features are less comparable.
    """
    if policy == "full_window":
        return window

    if policy == "allow_partial":
        return 1

    raise ValueError(f"Unknown min_periods_policy: {policy}")


def add_basic_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add non-rolling candle features.

    Features:
        log_return_1
        open_close_return
        high_low_range_pct
        candle_body_pct
        upper_wick_pct
        lower_wick_pct
        volume_change
        quote_volume_change, if quote_volume exists
        num_trades_change, if num_trades exists
    """
    out = prepare_candles(df)

    out["log_close"] = np.log(out["close"])
    out["log_return_1"] = out.groupby("symbol", sort=False)["log_close"].diff()

    out["open_close_return"] = safe_divide(out["close"] - out["open"], out["open"])

    out["high_low_range_pct"] = safe_divide(out["high"] - out["low"], out["open"])

    out["candle_body_pct"] = safe_divide(
        (out["close"] - out["open"]).abs(),
        out["open"],
    )

    candle_top = out[["open", "close"]].max(axis=1)
    candle_bottom = out[["open", "close"]].min(axis=1)

    out["upper_wick_pct"] = safe_divide(out["high"] - candle_top, out["open"])

    out["lower_wick_pct"] = safe_divide(candle_bottom - out["low"], out["open"])

    out["volume_change"] = pct_change_by_symbol(out, "volume")

    if "quote_volume" in out.columns:
        out["quote_volume_change"] = pct_change_by_symbol(out, "quote_volume")

    if "num_trades" in out.columns:
        out["num_trades_change"] = pct_change_by_symbol(out, "num_trades")

    return out


def add_rolling_features(
    df: pd.DataFrame,
    windows: Sequence[int],
    min_periods_policy: str = "full_window",
) -> pd.DataFrame:
    """
    Add backward-looking rolling-window features.

    For a window w, each feature at time t uses only information from:
        t, t-1, ..., t-w+1

    Expected warmup behavior:
        With full_window policy, the first w-ish rows have NaNs because there
        is not enough historical context yet.
    """
    out = prepare_candles(df)

    if "log_return_1" not in out.columns:
        raise ValueError("log_return_1 is missing. Run add_basic_candle_features first.")

    if "high_low_range_pct" not in out.columns:
        raise ValueError(
            "high_low_range_pct is missing. Run add_basic_candle_features first."
        )

    for window in windows:
        if window <= 0:
            raise ValueError(f"Rolling window must be positive: {window}")

        min_periods = resolve_min_periods(window, min_periods_policy)

        abs_return_col = f"_abs_log_return_1_for_window_{window}"
        out[abs_return_col] = out["log_return_1"].abs()

        out[f"rolling_return_mean_{window}"] = rolling_by_symbol(
            out,
            column="log_return_1",
            window=window,
            agg="mean",
            min_periods=min_periods,
        )

        out[f"rolling_return_std_{window}"] = rolling_by_symbol(
            out,
            column="log_return_1",
            window=window,
            agg="std",
            min_periods=min_periods,
        )

        out[f"rolling_abs_return_mean_{window}"] = rolling_by_symbol(
            out,
            column=abs_return_col,
            window=window,
            agg="mean",
            min_periods=min_periods,
        )

        out[f"rolling_realized_vol_{window}"] = rolling_realized_vol_by_symbol(
            out,
            return_col="log_return_1",
            window=window,
            min_periods=min_periods,
        )

        out[f"rolling_volume_mean_{window}"] = rolling_by_symbol(
            out,
            column="volume",
            window=window,
            agg="mean",
            min_periods=min_periods,
        )

        out[f"rolling_volume_std_{window}"] = rolling_by_symbol(
            out,
            column="volume",
            window=window,
            agg="std",
            min_periods=min_periods,
        )

        out[f"volume_zscore_{window}"] = safe_divide(
            out["volume"] - out[f"rolling_volume_mean_{window}"],
            out[f"rolling_volume_std_{window}"],
        )

        out[f"rolling_high_low_range_mean_{window}"] = rolling_by_symbol(
            out,
            column="high_low_range_pct",
            window=window,
            agg="mean",
            min_periods=min_periods,
        )

        rolling_high = rolling_by_symbol(
            out,
            column="high",
            window=window,
            agg="max",
            min_periods=min_periods,
        )

        rolling_low = rolling_by_symbol(
            out,
            column="low",
            window=window,
            agg="min",
            min_periods=min_periods,
        )

        rolling_close_max = rolling_by_symbol(
            out,
            column="close",
            window=window,
            agg="max",
            min_periods=min_periods,
        )

        out[f"rolling_drawdown_{window}"] = safe_divide(
            out["close"],
            rolling_close_max,
        ) - 1.0

        out[f"distance_from_rolling_high_{window}"] = safe_divide(
            out["close"],
            rolling_high,
        ) - 1.0

        out[f"distance_from_rolling_low_{window}"] = safe_divide(
            out["close"],
            rolling_low,
        ) - 1.0

        out = out.drop(columns=[abs_return_col])

    return out


def build_feature_dataframe(
    candles: pd.DataFrame,
    windows: Sequence[int],
    min_periods_policy: str = "full_window",
) -> pd.DataFrame:
    """
    Full feature-building pipeline.

    This function should be reused by training and inference code to avoid
    training-serving skew.
    """
    basic = add_basic_candle_features(candles)

    features = add_rolling_features(
        basic,
        windows=windows,
        min_periods_policy=min_periods_policy,
    )

    return features


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Select model-intended engineered feature columns.

    Excludes:
        identifiers
        timestamps
        raw OHLCV columns
        helper columns
        future-looking or label-like columns
    """
    feature_cols: list[str] = []

    for col in df.columns:
        if col in RAW_COLUMNS:
            continue

        if col in HELPER_COLUMNS:
            continue

        if col.startswith("_"):
            continue

        if col.startswith(FORBIDDEN_FEATURE_PREFIXES):
            continue

        feature_cols.append(col)

    return feature_cols


def audit_feature_missingness(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    windows: Sequence[int],
) -> pd.DataFrame:
    """
    Summarize NaNs in feature columns.

    Many rolling features should have warmup NaNs at the beginning.
    Entirely NaN columns are suspicious.
    """
    rows: list[dict[str, Any]] = []
    n_rows = len(df)
    max_window = max(windows) if windows else 0

    for col in feature_cols:
        missing_count = int(df[col].isna().sum())
        missing_pct = float(df[col].isna().mean())

        if missing_count == 0:
            interpretation = "no_missing_values"
        elif missing_count >= n_rows:
            interpretation = "suspicious_all_values_missing"
        elif missing_count <= max_window + 5:
            interpretation = "expected_rolling_warmup"
        else:
            interpretation = "inspect_more_missing_than_expected"

        rows.append(
            {
                "feature": col,
                "missing_count": missing_count,
                "missing_pct": missing_pct,
                "interpretation": interpretation,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["missing_count", "feature"], ascending=[False, True])
        .reset_index(drop=True)
    )


def make_model_feature_table(
    features: pd.DataFrame,
    feature_cols: Sequence[str],
    drop_missing_features: bool = True,
) -> pd.DataFrame:
    """
    Create the model-ready feature table.

    Keeps:
        symbol
        timestamp
        close
        engineered feature columns

    The close column is kept for later label alignment and plots, but is not
    included in feature_cols.
    """
    keep_cols = ["symbol", "timestamp", "close"] + list(feature_cols)

    missing_keep_cols = [col for col in keep_cols if col not in features.columns]

    if missing_keep_cols:
        raise ValueError(f"Missing expected columns in feature dataframe: {missing_keep_cols}")

    table = features[keep_cols].copy()

    before = len(table)

    if drop_missing_features:
        table = table.dropna(subset=feature_cols).reset_index(drop=True)

    after = len(table)

    if after == 0:
        raise ValueError(
            "Feature table is empty after dropping missing features. "
            "Use more input history, reduce rolling windows, or set "
            "drop_missing_features=false for debugging."
        )

    logging.info("Feature rows before drop: %s", before)
    logging.info("Feature rows after drop: %s", after)

    return table


def check_past_only_features(
    candles: pd.DataFrame,
    windows: Sequence[int],
    min_periods_policy: str,
    atol: float = 1e-12,
) -> None:
    """
    Check that features at a chosen cutoff row do not depend on future rows.

    Build features from:
        1. the full dataset
        2. a prefix ending at cutoff

    Feature values at cutoff should match.
    """
    sorted_candles = prepare_candles(candles)

    if len(sorted_candles) < max(windows) + 20:
        logging.warning("Skipping leakage check: not enough rows.")
        return

    cutoff_index = max(windows) + 10

    full_features = build_feature_dataframe(
        sorted_candles,
        windows=windows,
        min_periods_policy=min_periods_policy,
    )
    full_feature_cols = get_feature_columns(full_features)

    prefix_candles = sorted_candles.iloc[: cutoff_index + 1].copy()

    prefix_features = build_feature_dataframe(
        prefix_candles,
        windows=windows,
        min_periods_policy=min_periods_policy,
    )

    full_row = full_features.iloc[cutoff_index][full_feature_cols]
    prefix_row = prefix_features.iloc[-1][full_feature_cols]

    full_values = pd.to_numeric(full_row, errors="coerce")
    prefix_values = pd.to_numeric(prefix_row, errors="coerce")

    diff = (full_values - prefix_values).abs()
    max_diff = diff.max()

    if pd.isna(max_diff):
        raise AssertionError("Past-only feature check produced only NaN differences.")

    if max_diff > atol:
        worst = diff.sort_values(ascending=False).head(10)
        raise AssertionError(
            "Past-only feature check failed. "
            f"Max difference: {max_diff}. Worst features: {worst.to_dict()}"
        )

    logging.info("Past-only feature check passed. Max diff: %s", max_diff)


def dataframe_to_markdown_safe(df: pd.DataFrame, max_rows: int = 80) -> str:
    """Convert dataframe to markdown with fallback."""
    if df is None or df.empty:
        return "_None._"

    small = df.head(max_rows)

    try:
        return small.to_markdown(index=False)
    except Exception:
        return "```text\n" + small.to_string(index=False) + "\n```"


def write_feature_schema(
    path: str | Path,
    config: dict[str, Any],
    feature_cols: Sequence[str],
) -> None:
    """Save feature schema for training/inference consistency."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    schema = {
        "symbol": config.get("symbol"),
        "interval": config.get("interval"),
        "processed_path": config.get("processed_path"),
        "features_output_path": config.get("features_output_path"),
        "feature_columns": list(feature_cols),
        "num_features": len(feature_cols),
        "rolling_windows": list(config.get("rolling_windows", [])),
        "min_periods_policy": config.get("min_periods_policy", "full_window"),
        "notes": [
            "Features use current completed candle and historical candles only.",
            "This file contains no labels.",
            "The close column is retained for alignment and visualization but is not a model feature.",
        ],
    }

    path.write_text(json.dumps(schema, indent=2), encoding="utf-8")


def write_feature_report(
    path: str | Path,
    config: dict[str, Any],
    feature_table: pd.DataFrame,
    feature_cols: Sequence[str],
    missingness: pd.DataFrame,
) -> None:
    """Write a concise Markdown report for feature engineering."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    suspicious_missing = missingness[
        missingness["interpretation"].isin(
            [
                "suspicious_all_values_missing",
                "inspect_more_missing_than_expected",
            ]
        )
    ]

    lines: list[str] = []

    lines.append("# Feature Engineering Report")
    lines.append("")
    lines.append("## Context")
    lines.append("")
    lines.append(f"- Symbol: `{config.get('symbol')}`")
    lines.append(f"- Interval: `{config.get('interval')}`")
    lines.append(f"- Input path: `{config.get('processed_path')}`")
    lines.append(f"- Output path: `{config.get('features_output_path')}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Feature rows: `{len(feature_table)}`")
    lines.append(f"- Number of engineered features: `{len(feature_cols)}`")
    lines.append(f"- Rolling windows: `{config.get('rolling_windows')}`")
    lines.append(f"- Min periods policy: `{config.get('min_periods_policy')}`")
    lines.append("")
    lines.append("## Feature Columns")
    lines.append("")
    lines.append("```text")
    for col in feature_cols:
        lines.append(col)
    lines.append("```")
    lines.append("")
    lines.append("## Missingness Audit")
    lines.append("")
    lines.append(dataframe_to_markdown_safe(missingness, max_rows=120))
    lines.append("")
    lines.append("## Suspicious Missingness")
    lines.append("")
    lines.append(dataframe_to_markdown_safe(suspicious_missing, max_rows=80))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "Rolling-window features naturally create warmup NaNs at the beginning of "
        "the dataset. Those rows are dropped when `drop_missing_features` is true."
    )
    lines.append("")
    lines.append(
        "This pipeline creates only backward-looking features. It does not create "
        "future returns, future volatility, labels, train/test splits, or models."
    )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def run_feature_engineering(config: dict[str, Any]) -> dict[str, Any]:
    """Main entry point for Teaching pipeline 3: Feature Engineering."""
    processed_path = Path(config["processed_path"])
    features_output_path = Path(config["features_output_path"])
    feature_schema_path = Path(config["feature_schema_path"])
    feature_report_path = Path(config["feature_report_path"])

    windows = [int(w) for w in config["rolling_windows"]]
    min_periods_policy = str(config.get("min_periods_policy", "full_window"))
    drop_missing_features = bool(config.get("drop_missing_features", True))
    run_leakage_check = bool(config.get("run_leakage_check", True))

    candles = load_processed_candles(processed_path)
    candles = prepare_candles(candles)

    if run_leakage_check:
        check_past_only_features(
            candles=candles,
            windows=windows,
            min_periods_policy=min_periods_policy,
        )

    features = build_feature_dataframe(
        candles,
        windows=windows,
        min_periods_policy=min_periods_policy,
    )

    feature_cols = get_feature_columns(features)

    if not feature_cols:
        raise ValueError("No engineered feature columns were created.")

    missingness = audit_feature_missingness(
        features,
        feature_cols=feature_cols,
        windows=windows,
    )

    suspicious = missingness[
        missingness["interpretation"].isin(
            [
                "suspicious_all_values_missing",
                "inspect_more_missing_than_expected",
            ]
        )
    ]

    if not suspicious.empty:
        logging.warning(
            "Suspicious feature missingness detected:\n%s",
            suspicious.to_string(index=False),
        )

    feature_table = make_model_feature_table(
        features,
        feature_cols=feature_cols,
        drop_missing_features=drop_missing_features,
    )

    features_output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_table.to_parquet(features_output_path, index=False)

    write_feature_schema(
        path=feature_schema_path,
        config=config,
        feature_cols=feature_cols,
    )

    write_feature_report(
        path=feature_report_path,
        config=config,
        feature_table=feature_table,
        feature_cols=feature_cols,
        missingness=missingness,
    )

    result = {
        "processed_path": str(processed_path),
        "features_output_path": str(features_output_path),
        "feature_schema_path": str(feature_schema_path),
        "feature_report_path": str(feature_report_path),
        "num_rows": int(len(feature_table)),
        "num_features": int(len(feature_cols)),
    }

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build backward-looking OHLCV features for crypto regime classification."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to feature config YAML, for example configs/features.yaml.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    args = parse_args()
    config = load_config(args.config)

    result = run_feature_engineering(config)

    logging.info("Feature engineering complete.")
    logging.info("Result: %s", result)


if __name__ == "__main__":
    main()
