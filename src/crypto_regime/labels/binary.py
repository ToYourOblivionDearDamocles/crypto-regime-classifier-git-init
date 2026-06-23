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
    "close",
]

TARGET_PREFIXES = (
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


def load_feature_table(path: str | Path) -> pd.DataFrame:
    """Load feature table from Parquet or CSV."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Feature table does not exist: {path}")

    if path.suffix == ".parquet":
        return pd.read_parquet(path)

    if path.suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported feature table file type: {path.suffix}")


def interval_to_minutes(interval: str) -> int:
    """Convert interval string like '5m', '1h', or '1d' to minutes."""
    if len(interval) < 2:
        raise ValueError(f"Invalid interval: {interval}")

    unit = interval[-1]
    value = int(interval[:-1])

    if value <= 0:
        raise ValueError(f"Interval value must be positive: {interval}")

    if unit == "m":
        return value

    if unit == "h":
        return value * 60

    if unit == "d":
        return value * 24 * 60

    raise ValueError(f"Unsupported interval: {interval}")


def validate_label_input(df: pd.DataFrame) -> None:
    """
    Minimal input validation for label generation.

    Full data validation belongs to Teaching pipeline 2.
    This function only ensures label generation receives the required columns.
    """
    missing = [col for col in REQUIRED_INPUT_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns for label generation: {missing}")

    if df.empty:
        raise ValueError("Feature table is empty.")


def prepare_label_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare feature table for label generation.

    This function:
        - parses timestamp as UTC
        - converts close to numeric
        - sorts by symbol/timestamp

    It does not modify features or create labels.
    """
    validate_label_input(df)

    out = df.copy()

    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")

    if out["timestamp"].isna().any():
        raise ValueError("timestamp contains null/unparseable values.")

    if out["close"].isna().any():
        raise ValueError("close contains null/non-numeric values.")

    if (out["close"] <= 0).any():
        raise ValueError("close must be positive to compute log returns.")

    out = out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    return out


def future_window_sum_by_symbol(
    df: pd.DataFrame,
    value_col: str,
    horizon: int,
) -> pd.Series:
    """
    For each row t, compute future window sum within each symbol:

        value_{t+1} + value_{t+2} + ... + value_{t+horizon}

    This returns a Series aligned with the original dataframe index.

    If fewer than `horizon` future rows exist, the result is NaN.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive.")

    if value_col not in df.columns:
        raise ValueError(f"Column does not exist: {value_col}")

    def future_sum_one_symbol(s: pd.Series) -> pd.Series:
        shifted_terms = [s.shift(-i) for i in range(1, horizon + 1)]

        return pd.concat(shifted_terms, axis=1).sum(
            axis=1,
            min_count=horizon,
        )

    return df.groupby("symbol", sort=False)[value_col].transform(
        future_sum_one_symbol
    )


def add_future_label_values(
    df: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """
    Add continuous future target quantities.

    Definitions:

        r_t = log(close_t) - log(close_{t-1})

        future_return_h(t)
            = log(close_{t+h}) - log(close_t)

        future_rv_h(t)
            = sqrt(sum_{i=1}^{h} r_{t+i}^2)

    These future columns are label-construction columns.
    They must not be used as model features.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive.")

    out = prepare_label_input(df)

    future_return_col = f"future_return_{horizon}"
    future_rv_col = f"future_rv_{horizon}"

    out["_label_log_close"] = np.log(out["close"])

    out["_label_log_return_1"] = (
        out.groupby("symbol", sort=False)["_label_log_close"].diff()
    )

    out[future_return_col] = (
        out.groupby("symbol", sort=False)["_label_log_close"].shift(-horizon)
        - out["_label_log_close"]
    )

    out["_label_squared_log_return_1"] = out["_label_log_return_1"] ** 2

    future_sum_squared_returns = future_window_sum_by_symbol(
        out,
        value_col="_label_squared_log_return_1",
        horizon=horizon,
    )

    out[future_rv_col] = np.sqrt(future_sum_squared_returns)

    out = out.drop(
        columns=[
            "_label_log_close",
            "_label_log_return_1",
            "_label_squared_log_return_1",
        ],
        errors="ignore",
    )

    return out


def make_threshold_train_mask(
    df: pd.DataFrame,
    horizon: int,
    train_fraction: float,
) -> pd.Series:
    """
    Create a chronological mask for rows used to estimate the label threshold.

    This is not the final train/validation/test split.
    Teaching pipeline 5 will formalize splitting.

    Here we only ensure the label threshold is estimated from an early
    chronological period, not from the full dataset.
    """
    if not (0 < train_fraction < 1):
        raise ValueError("train_fraction must be between 0 and 1.")

    future_rv_col = f"future_rv_{horizon}"

    if future_rv_col not in df.columns:
        raise ValueError(f"Missing column: {future_rv_col}")

    valid_mask = df[future_rv_col].notna()
    mask = pd.Series(False, index=df.index)

    for symbol, group in df.loc[valid_mask].groupby("symbol", sort=False):
        group = group.sort_values("timestamp")

        n_train = int(len(group) * train_fraction)

        if n_train <= 0:
            raise ValueError(f"Not enough valid rows for threshold training: {symbol}")

        mask.loc[group.index[:n_train]] = True

    return mask


def compute_train_only_threshold(
    df: pd.DataFrame,
    horizon: int,
    train_mask: pd.Series,
    quantile: float,
) -> float:
    """
    Compute high-volatility threshold from training-period rows only.

    This is the main leakage-control rule in binary label generation.
    """
    if not (0 < quantile < 1):
        raise ValueError("quantile must be between 0 and 1.")

    future_rv_col = f"future_rv_{horizon}"

    if future_rv_col not in df.columns:
        raise ValueError(f"Missing column: {future_rv_col}")

    train_values = df.loc[train_mask, future_rv_col].dropna()

    if train_values.empty:
        raise ValueError("No training-period future_rv values available.")

    return float(train_values.quantile(quantile))


def add_binary_high_vol_label(
    df: pd.DataFrame,
    horizon: int,
    volatility_threshold: float,
    threshold_train_mask: pd.Series,
) -> pd.DataFrame:
    """
    Add binary high-volatility label.

    y_high_vol_h = 1 if future_rv_h > volatility_threshold
                 = 0 otherwise

    Rows without enough future candles receive missing labels.
    """
    out = df.copy()

    future_rv_col = f"future_rv_{horizon}"
    label_col = f"y_high_vol_{horizon}"

    if future_rv_col not in out.columns:
        raise ValueError(f"Missing column: {future_rv_col}")

    out[label_col] = pd.NA

    valid_mask = out[future_rv_col].notna()

    out.loc[valid_mask, label_col] = (
        out.loc[valid_mask, future_rv_col] > volatility_threshold
    ).astype(int)

    out[label_col] = out[label_col].astype("Int64")
    out["used_for_label_threshold"] = threshold_train_mask.astype(bool)

    return out


def make_labeled_model_table(
    df: pd.DataFrame,
    horizon: int,
    drop_missing_labels: bool = True,
) -> pd.DataFrame:
    """
    Create labeled table for later splitting/modeling.

    The output keeps:
        - all engineered feature columns
        - close for alignment/plots
        - future_return_h
        - future_rv_h
        - y_high_vol_h
        - used_for_label_threshold
    """
    label_col = f"y_high_vol_{horizon}"

    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    out = df.copy()

    before = len(out)

    if drop_missing_labels:
        out = out[out[label_col].notna()].reset_index(drop=True)

    after = len(out)

    if after == 0:
        raise ValueError(
            "Labeled table is empty after dropping missing labels. "
            "Use more data or reduce horizon."
        )

    logging.info("Rows before label drop: %s", before)
    logging.info("Rows after label drop: %s", after)

    return out


def summarize_binary_labels(
    df: pd.DataFrame,
    horizon: int,
    volatility_threshold: float,
) -> dict[str, Any]:
    """Summarize label distribution and future volatility statistics."""
    label_col = f"y_high_vol_{horizon}"
    future_rv_col = f"future_rv_{horizon}"

    valid = df[df[label_col].notna()].copy()

    if valid.empty:
        raise ValueError("No valid labeled rows available for summary.")

    class_counts = valid[label_col].value_counts().sort_index()
    class_rates = valid[label_col].value_counts(normalize=True).sort_index()

    threshold_train_rows = valid[valid["used_for_label_threshold"]]

    train_class_counts = threshold_train_rows[label_col].value_counts().sort_index()
    train_class_rates = (
        threshold_train_rows[label_col].value_counts(normalize=True).sort_index()
    )

    return {
        "horizon": int(horizon),
        "total_rows_before_label_drop": int(len(df)),
        "valid_labeled_rows": int(len(valid)),
        "missing_label_rows": int(df[label_col].isna().sum()),
        "volatility_threshold": float(volatility_threshold),
        "overall_class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "overall_class_rates": {str(k): float(v) for k, v in class_rates.items()},
        "threshold_train_class_counts": {
            str(k): int(v) for k, v in train_class_counts.items()
        },
        "threshold_train_class_rates": {
            str(k): float(v) for k, v in train_class_rates.items()
        },
        "future_rv_min": float(valid[future_rv_col].min()),
        "future_rv_median": float(valid[future_rv_col].median()),
        "future_rv_mean": float(valid[future_rv_col].mean()),
        "future_rv_max": float(valid[future_rv_col].max()),
    }


def summarize_quantile_sensitivity(
    df: pd.DataFrame,
    horizon: int,
    train_mask: pd.Series,
    quantiles: Sequence[float],
) -> pd.DataFrame:
    """
    Summarize alternative label thresholds.

    This does not change the active label. It documents how class balance would
    change under q70/q80/q90/q95-style definitions.
    """
    future_rv_col = f"future_rv_{horizon}"

    valid = df[df[future_rv_col].notna()].copy()
    train_values = df.loc[train_mask, future_rv_col].dropna()

    rows: list[dict[str, Any]] = []

    for q in quantiles:
        if not (0 < float(q) < 1):
            raise ValueError(f"Invalid sensitivity quantile: {q}")

        threshold = float(train_values.quantile(float(q)))

        train_positive_rate = float((train_values > threshold).mean())
        overall_positive_rate = float((valid[future_rv_col] > threshold).mean())

        rows.append(
            {
                "quantile": float(q),
                "threshold": threshold,
                "train_positive_rate": train_positive_rate,
                "overall_positive_rate": overall_positive_rate,
            }
        )

    return pd.DataFrame(rows)


def audit_future_label_missingness(
    df: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """
    Count missing future target values per symbol.

    The final `horizon` rows per symbol normally cannot be labeled.
    """
    future_return_col = f"future_return_{horizon}"
    future_rv_col = f"future_rv_{horizon}"

    rows: list[dict[str, Any]] = []

    for symbol, group in df.groupby("symbol", sort=False):
        rows.append(
            {
                "symbol": symbol,
                "rows": int(len(group)),
                "missing_future_return": int(group[future_return_col].isna().sum()),
                "missing_future_rv": int(group[future_rv_col].isna().sum()),
                "expected_missing_approximately": int(horizon),
            }
        )

    return pd.DataFrame(rows)


def check_future_rv_alignment(
    df: pd.DataFrame,
    horizon: int,
    row_position: int = 10,
    atol: float = 1e-12,
) -> None:
    """
    Manual off-by-one check for future realized volatility.

    Verifies:

        future_rv_t = sqrt(r_{t+1}^2 + ... + r_{t+horizon}^2)
    """
    future_rv_col = f"future_rv_{horizon}"

    if future_rv_col not in df.columns:
        raise ValueError(f"Missing column: {future_rv_col}")

    working = prepare_label_input(df)

    symbol = working["symbol"].iloc[0]

    group = (
        working[working["symbol"] == symbol]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if row_position < 0 or row_position + horizon >= len(group):
        logging.warning("Skipping future_rv alignment check: not enough future rows.")
        return

    log_close = np.log(group["close"])

    future_returns = [
        log_close.iloc[row_position + i] - log_close.iloc[row_position + i - 1]
        for i in range(1, horizon + 1)
    ]

    manual_future_rv = float(np.sqrt(np.sum(np.square(future_returns))))
    computed_future_rv = float(group.loc[row_position, future_rv_col])

    diff = abs(manual_future_rv - computed_future_rv)

    if diff > atol:
        raise AssertionError(
            "future_rv alignment check failed. "
            f"manual={manual_future_rv}, computed={computed_future_rv}, diff={diff}"
        )

    logging.info("future_rv alignment check passed. diff=%s", diff)


def dataframe_to_markdown_safe(df: pd.DataFrame, max_rows: int = 80) -> str:
    """Convert dataframe to Markdown with fallback."""
    if df is None or df.empty:
        return "_None._"

    small = df.head(max_rows)

    try:
        return small.to_markdown(index=False)
    except Exception:
        return "```text\n" + small.to_string(index=False) + "\n```"


def write_label_config(
    path: str | Path,
    config: dict[str, Any],
    label_summary: dict[str, Any],
    volatility_threshold: float,
) -> None:
    """Write machine-readable label metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    horizon = int(config["horizon_candles"])
    interval_minutes = interval_to_minutes(str(config["interval"]))

    metadata = {
        "symbol": config.get("symbol"),
        "interval": config.get("interval"),
        "features_input_path": config.get("features_input_path"),
        "labeled_output_path": config.get("labeled_output_path"),
        "label_type": "binary_high_volatility",
        "horizon_candles": horizon,
        "horizon_minutes": horizon * interval_minutes,
        "future_return_column": f"future_return_{horizon}",
        "future_rv_column": f"future_rv_{horizon}",
        "label_column": f"y_high_vol_{horizon}",
        "future_rv_definition": (
            "sqrt(sum_{i=1}^{horizon} r_{t+i}^2), "
            "where r_t = log(close_t) - log(close_{t-1})"
        ),
        "volatility_quantile": float(config["volatility_quantile"]),
        "volatility_threshold": float(volatility_threshold),
        "threshold_source": (
            "chronological threshold-training portion only; "
            "not computed from the full dataset"
        ),
        "train_fraction_for_threshold": float(config["train_fraction_for_threshold"]),
        "summary": label_summary,
    }

    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def write_label_report(
    path: str | Path,
    config: dict[str, Any],
    label_summary: dict[str, Any],
    volatility_threshold: float,
    missingness: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> None:
    """Write human-readable binary label generation report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    horizon = int(config["horizon_candles"])
    interval_minutes = interval_to_minutes(str(config["interval"]))

    lines: list[str] = []

    lines.append("# Binary Label Generation Report")
    lines.append("")
    lines.append("## Context")
    lines.append("")
    lines.append(f"- Symbol: `{config.get('symbol')}`")
    lines.append(f"- Interval: `{config.get('interval')}`")
    lines.append(f"- Feature input path: `{config.get('features_input_path')}`")
    lines.append(f"- Labeled output path: `{config.get('labeled_output_path')}`")
    lines.append("")
    lines.append("## Active Binary Label")
    lines.append("")
    lines.append(f"- Horizon candles: `{horizon}`")
    lines.append(f"- Horizon minutes: `{horizon * interval_minutes}`")
    lines.append(f"- Volatility quantile: `{config['volatility_quantile']}`")
    lines.append(f"- Train-only volatility threshold: `{volatility_threshold}`")
    lines.append("")
    lines.append("```text")
    lines.append("r_t = log(close_t) - log(close_{t-1})")
    lines.append("future_rv_t = sqrt(sum_{i=1}^{horizon} r_{t+i}^2)")
    lines.append("y_t = 1 if future_rv_t > threshold")
    lines.append("y_t = 0 otherwise")
    lines.append("```")
    lines.append("")
    lines.append("## Leakage Rule")
    lines.append("")
    lines.append(
        "The volatility threshold is computed only from the chronological "
        "threshold-training portion, not from the full dataset."
    )
    lines.append("")
    lines.append("## Label Summary")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(label_summary, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Missing Future Label Audit")
    lines.append("")
    lines.append(dataframe_to_markdown_safe(missingness, max_rows=80))
    lines.append("")
    lines.append("## Quantile Sensitivity")
    lines.append("")
    lines.append(
        "This table documents how class balance would change under alternative "
        "threshold quantiles. The active implementation still uses the configured "
        "binary threshold above."
    )
    lines.append("")
    lines.append(dataframe_to_markdown_safe(sensitivity, max_rows=80))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "A positive label at timestamp t means the future horizon after t has "
        "high realized volatility. It does not mean the current candle itself "
        "is high-volatility."
    )
    lines.append("")
    lines.append(
        "The 80th percentile threshold is an operational event definition, not "
        "a theoretically sacred constant. The continuous future_rv column is kept "
        "so regression and quantile-regression variants can be added later."
    )
    lines.append("")
    lines.append(
        "This pipeline creates labels only. It does not train or evaluate a model."
    )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def run_binary_label_generation(config: dict[str, Any]) -> dict[str, Any]:
    """Main entry point for Teaching pipeline 4: binary label generation."""
    features_input_path = Path(config["features_input_path"])
    labeled_output_path = Path(config["labeled_output_path"])
    label_config_path = Path(config["label_config_path"])
    label_report_path = Path(config["label_report_path"])

    horizon = int(config["horizon_candles"])
    volatility_quantile = float(config["volatility_quantile"])
    train_fraction = float(config["train_fraction_for_threshold"])
    drop_missing_labels = bool(config.get("drop_missing_labels", True))

    sensitivity_quantiles = config.get(
        "sensitivity_quantiles",
        [0.70, 0.80, 0.90, 0.95],
    )

    feature_table = load_feature_table(features_input_path)

    labeled_base = add_future_label_values(
        feature_table,
        horizon=horizon,
    )

    check_future_rv_alignment(
        labeled_base,
        horizon=horizon,
    )

    threshold_train_mask = make_threshold_train_mask(
        labeled_base,
        horizon=horizon,
        train_fraction=train_fraction,
    )

    volatility_threshold = compute_train_only_threshold(
        labeled_base,
        horizon=horizon,
        train_mask=threshold_train_mask,
        quantile=volatility_quantile,
    )

    labeled_with_target = add_binary_high_vol_label(
        labeled_base,
        horizon=horizon,
        volatility_threshold=volatility_threshold,
        threshold_train_mask=threshold_train_mask,
    )

    label_summary = summarize_binary_labels(
        labeled_with_target,
        horizon=horizon,
        volatility_threshold=volatility_threshold,
    )

    missingness = audit_future_label_missingness(
        labeled_with_target,
        horizon=horizon,
    )

    sensitivity = summarize_quantile_sensitivity(
        labeled_with_target,
        horizon=horizon,
        train_mask=threshold_train_mask,
        quantiles=sensitivity_quantiles,
    )

    labeled_model_table = make_labeled_model_table(
        labeled_with_target,
        horizon=horizon,
        drop_missing_labels=drop_missing_labels,
    )

    labeled_output_path.parent.mkdir(parents=True, exist_ok=True)
    labeled_model_table.to_parquet(labeled_output_path, index=False)

    write_label_config(
        path=label_config_path,
        config=config,
        label_summary=label_summary,
        volatility_threshold=volatility_threshold,
    )

    write_label_report(
        path=label_report_path,
        config=config,
        label_summary=label_summary,
        volatility_threshold=volatility_threshold,
        missingness=missingness,
        sensitivity=sensitivity,
    )

    result = {
        "features_input_path": str(features_input_path),
        "labeled_output_path": str(labeled_output_path),
        "label_config_path": str(label_config_path),
        "label_report_path": str(label_report_path),
        "num_rows": int(len(labeled_model_table)),
        "horizon_candles": horizon,
        "volatility_quantile": volatility_quantile,
        "volatility_threshold": float(volatility_threshold),
        "label_column": f"y_high_vol_{horizon}",
    }

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate binary high-volatility labels."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to binary label config YAML, for example configs/label_binary.yaml.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    args = parse_args()
    config = load_config(args.config)

    result = run_binary_label_generation(config)

    logging.info("Binary label generation complete.")
    logging.info("Result: %s", result)


if __name__ == "__main__":
    main()
