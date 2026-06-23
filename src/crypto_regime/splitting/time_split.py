from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


SPLIT_TRAIN = "train"
SPLIT_VALIDATION = "validation"
SPLIT_TEST = "test"
SPLIT_PURGE_TRAIN_VALIDATION = "purge_train_validation"
SPLIT_PURGE_VALIDATION_TEST = "purge_validation_test"

MODEL_SPLITS = [SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST]

ALL_SPLITS = [
    SPLIT_TRAIN,
    SPLIT_PURGE_TRAIN_VALIDATION,
    SPLIT_VALIDATION,
    SPLIT_PURGE_VALIDATION_TEST,
    SPLIT_TEST,
]


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


def load_labeled_table(path: str | Path) -> pd.DataFrame:
    """Load labeled binary dataset from Parquet or CSV."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Labeled file does not exist: {path}")

    if path.suffix == ".parquet":
        return pd.read_parquet(path)

    if path.suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported labeled file type: {path.suffix}")


def interval_to_timedelta(interval: str) -> pd.Timedelta:
    """Convert interval string like '5m', '1h', or '1d' to pandas Timedelta."""
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


def validate_split_input(df: pd.DataFrame, horizon: int) -> None:
    """
    Validate minimal input requirements for splitting.

    Full data validation belongs to Teaching pipeline 2.
    Label generation belongs to Teaching pipeline 4.
    """
    future_rv_col = f"future_rv_{horizon}"
    label_col = f"y_high_vol_{horizon}"

    required = [
        "symbol",
        "timestamp",
        "close",
        future_rv_col,
        label_col,
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns for splitting: {missing}")

    if df.empty:
        raise ValueError("Input dataframe is empty.")


def prepare_split_input(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Prepare labeled table for chronological splitting.

    This function:
        - parses timestamp as UTC
        - sorts by symbol/timestamp
        - checks required target columns
    """
    validate_split_input(df, horizon=horizon)

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")

    if out["timestamp"].isna().any():
        raise ValueError("timestamp contains null or unparseable values.")

    out = out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    return out


def assign_chronological_splits_with_purge(
    df: pd.DataFrame,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    purge_gap_candles: int,
) -> pd.DataFrame:
    """
    Assign chronological train/validation/test splits with purge gaps.

    Split is done independently per symbol.

    Output split labels:
        train
        purge_train_validation
        validation
        purge_validation_test
        test

    Purged rows are not used for model fitting or final evaluation.
    """
    total = train_fraction + validation_fraction + test_fraction

    if not np.isclose(total, 1.0):
        raise ValueError("train_fraction + validation_fraction + test_fraction must sum to 1.")

    if min(train_fraction, validation_fraction, test_fraction) <= 0:
        raise ValueError("All split fractions must be positive.")

    if purge_gap_candles < 0:
        raise ValueError("purge_gap_candles must be non-negative.")

    out = df.copy().sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    out["split"] = "unassigned"

    for symbol, group in out.groupby("symbol", sort=False):
        idx = group.index.to_list()
        n = len(idx)

        if n < 10:
            raise ValueError(f"Not enough rows to split symbol {symbol}: {n}")

        train_boundary = int(n * train_fraction)
        validation_boundary = int(n * (train_fraction + validation_fraction))

        train_end = max(0, train_boundary - purge_gap_candles)
        validation_start = train_boundary
        validation_end = max(validation_start, validation_boundary - purge_gap_candles)
        test_start = validation_boundary

        if train_end <= 0:
            raise ValueError(
                f"Train split is empty after purge for symbol {symbol}. "
                "Use more data or reduce purge_gap_candles."
            )

        if validation_end <= validation_start:
            raise ValueError(
                f"Validation split is empty after purge for symbol {symbol}. "
                "Use more data or reduce purge_gap_candles."
            )

        if test_start >= n:
            raise ValueError(f"Test split is empty for symbol {symbol}.")

        train_idx = idx[:train_end]
        purge_train_val_idx = idx[train_end:validation_start]
        validation_idx = idx[validation_start:validation_end]
        purge_val_test_idx = idx[validation_end:test_start]
        test_idx = idx[test_start:]

        out.loc[train_idx, "split"] = SPLIT_TRAIN
        out.loc[purge_train_val_idx, "split"] = SPLIT_PURGE_TRAIN_VALIDATION
        out.loc[validation_idx, "split"] = SPLIT_VALIDATION
        out.loc[purge_val_test_idx, "split"] = SPLIT_PURGE_VALIDATION_TEST
        out.loc[test_idx, "split"] = SPLIT_TEST

    if (out["split"] == "unassigned").any():
        raise RuntimeError("Some rows were not assigned a split.")

    return out


def compute_split_train_threshold(
    df: pd.DataFrame,
    horizon: int,
    quantile: float,
) -> float:
    """
    Compute the binary high-volatility threshold from split == train only.

    This is stricter than the preliminary threshold used in label generation.
    """
    if not (0 < quantile < 1):
        raise ValueError("quantile must be between 0 and 1.")

    future_rv_col = f"future_rv_{horizon}"

    if future_rv_col not in df.columns:
        raise ValueError(f"Missing column: {future_rv_col}")

    if "split" not in df.columns:
        raise ValueError("Missing split column.")

    train_values = df.loc[df["split"] == SPLIT_TRAIN, future_rv_col].dropna()

    if train_values.empty:
        raise ValueError("No train future_rv values available.")

    return float(train_values.quantile(quantile))


def add_split_safe_binary_label(
    df: pd.DataFrame,
    horizon: int,
    threshold: float,
) -> pd.DataFrame:
    """
    Add split-safe binary label.

    y_high_vol_h_split_safe = 1 if future_rv_h > train-only threshold
                            = 0 otherwise
    """
    out = df.copy()

    future_rv_col = f"future_rv_{horizon}"
    target_col = f"y_high_vol_{horizon}_split_safe"

    if future_rv_col not in out.columns:
        raise ValueError(f"Missing column: {future_rv_col}")

    out[target_col] = pd.NA

    valid_mask = out[future_rv_col].notna()

    out.loc[valid_mask, target_col] = (
        out.loc[valid_mask, future_rv_col] > threshold
    ).astype(int)

    out[target_col] = out[target_col].astype("Int64")

    return out


def compare_existing_and_split_safe_labels(
    df: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """
    Compare Teaching pipeline 4 label with split-safe label.

    Differences are not automatically bugs. They can occur because the
    split-safe threshold is computed after purge-aware splitting.
    """
    old_label_col = f"y_high_vol_{horizon}"
    split_safe_label_col = f"y_high_vol_{horizon}_split_safe"

    if old_label_col not in df.columns:
        raise ValueError(f"Missing old label column: {old_label_col}")

    if split_safe_label_col not in df.columns:
        raise ValueError(f"Missing split-safe label column: {split_safe_label_col}")

    valid = df[
        df[old_label_col].notna()
        & df[split_safe_label_col].notna()
        & df["split"].isin(MODEL_SPLITS)
    ].copy()

    if valid.empty:
        return pd.DataFrame(
            columns=["split", "rows_compared", "num_changed", "changed_rate"]
        )

    valid["label_changed"] = valid[old_label_col] != valid[split_safe_label_col]

    summary = (
        valid.groupby("split")["label_changed"]
        .agg(["count", "sum", "mean"])
        .reset_index()
        .rename(
            columns={
                "count": "rows_compared",
                "sum": "num_changed",
                "mean": "changed_rate",
            }
        )
    )

    return summary


def summarize_splits(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Summarize split sizes, time ranges, and class balance."""
    target_col = f"y_high_vol_{horizon}_split_safe"

    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    rows: list[dict[str, Any]] = []

    for split_name in ALL_SPLITS:
        group = df[df["split"] == split_name]

        if group.empty:
            rows.append(
                {
                    "split": split_name,
                    "num_rows": 0,
                    "start_timestamp": None,
                    "end_timestamp": None,
                    "labeled_rows": 0,
                    "positive_count": 0,
                    "negative_count": 0,
                    "positive_rate": np.nan,
                }
            )
            continue

        labeled = group[group[target_col].notna()]
        positives = int((labeled[target_col] == 1).sum())
        negatives = int((labeled[target_col] == 0).sum())

        positive_rate = float(labeled[target_col].mean()) if len(labeled) > 0 else np.nan

        rows.append(
            {
                "split": split_name,
                "num_rows": int(len(group)),
                "start_timestamp": str(group["timestamp"].min()),
                "end_timestamp": str(group["timestamp"].max()),
                "labeled_rows": int(len(labeled)),
                "positive_count": positives,
                "negative_count": negatives,
                "positive_rate": positive_rate,
            }
        )

    return pd.DataFrame(rows)


def validate_split_order_and_overlap(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Validate:
        - train, validation, test are non-empty
        - train < validation < test chronologically
        - no timestamp overlap between model splits
    """
    issues: list[dict[str, Any]] = []

    used = df[df["split"].isin(MODEL_SPLITS)].copy()

    for symbol, group in used.groupby("symbol", sort=False):
        train = group[group["split"] == SPLIT_TRAIN]
        validation = group[group["split"] == SPLIT_VALIDATION]
        test = group[group["split"] == SPLIT_TEST]

        non_empty = not train.empty and not validation.empty and not test.empty

        issues.append(
            {
                "symbol": symbol,
                "check": "non_empty_model_splits",
                "passed": bool(non_empty),
                "message": (
                    f"train={len(train)}, validation={len(validation)}, test={len(test)}"
                ),
            }
        )

        if not non_empty:
            continue

        train_max = train["timestamp"].max()
        val_min = validation["timestamp"].min()
        val_max = validation["timestamp"].max()
        test_min = test["timestamp"].min()

        order_passed = train_max < val_min and val_max < test_min

        issues.append(
            {
                "symbol": symbol,
                "check": "chronological_order",
                "passed": bool(order_passed),
                "message": (
                    f"train_max={train_max}, val_min={val_min}, "
                    f"val_max={val_max}, test_min={test_min}"
                ),
            }
        )

        train_times = set(train["timestamp"])
        val_times = set(validation["timestamp"])
        test_times = set(test["timestamp"])

        overlap_count = (
            len(train_times & val_times)
            + len(train_times & test_times)
            + len(val_times & test_times)
        )

        issues.append(
            {
                "symbol": symbol,
                "check": "no_timestamp_overlap",
                "passed": overlap_count == 0,
                "message": f"timestamp_overlap_count={overlap_count}",
            }
        )

    return issues


def validate_purge_gap(
    df: pd.DataFrame,
    horizon: int,
    interval_delta: pd.Timedelta,
) -> list[dict[str, Any]]:
    """
    Validate purge gap around train/validation and validation/test boundaries.

    A row at time t has a label window ending around:

        t + horizon * interval_delta

    Require:
        max_train_timestamp + horizon * interval_delta < min_validation_timestamp
        max_validation_timestamp + horizon * interval_delta < min_test_timestamp
    """
    issues: list[dict[str, Any]] = []

    used = df[df["split"].isin(MODEL_SPLITS)].copy()

    for symbol, group in used.groupby("symbol", sort=False):
        train = group[group["split"] == SPLIT_TRAIN]
        validation = group[group["split"] == SPLIT_VALIDATION]
        test = group[group["split"] == SPLIT_TEST]

        if train.empty or validation.empty or test.empty:
            continue

        train_label_end = train["timestamp"].max() + horizon * interval_delta
        validation_start = validation["timestamp"].min()

        validation_label_end = validation["timestamp"].max() + horizon * interval_delta
        test_start = test["timestamp"].min()

        train_val_passed = train_label_end < validation_start
        val_test_passed = validation_label_end < test_start

        issues.append(
            {
                "symbol": symbol,
                "check": "train_label_window_before_validation",
                "passed": bool(train_val_passed),
                "message": (
                    f"train_label_end={train_label_end}, "
                    f"validation_start={validation_start}"
                ),
            }
        )

        issues.append(
            {
                "symbol": symbol,
                "check": "validation_label_window_before_test",
                "passed": bool(val_test_passed),
                "message": (
                    f"validation_label_end={validation_label_end}, "
                    f"test_start={test_start}"
                ),
            }
        )

    return issues


def raise_if_split_validation_fails(*issue_lists: list[dict[str, Any]]) -> None:
    """Raise RuntimeError if any split validation issue failed."""
    all_issues: list[dict[str, Any]] = []

    for issues in issue_lists:
        all_issues.extend(issues)

    if not all_issues:
        raise RuntimeError("No split validation issues were produced.")

    issues_df = pd.DataFrame(all_issues)
    failed = issues_df[issues_df["passed"] == False]

    if not failed.empty:
        raise RuntimeError(
            "Split validation failed:\n" + failed.to_string(index=False)
        )


def make_walk_forward_folds(
    df: pd.DataFrame,
    min_train_fraction: float,
    validation_window_fraction: float,
    step_fraction: float,
    purge_gap_candles: int,
) -> pd.DataFrame:
    """
    Create walk-forward fold metadata.

    This does not train models. It only records fold boundaries for later use.
    """
    rows: list[dict[str, Any]] = []

    if not (0 < min_train_fraction < 1):
        raise ValueError("min_train_fraction must be between 0 and 1.")

    if not (0 < validation_window_fraction < 1):
        raise ValueError("validation_window_fraction must be between 0 and 1.")

    if not (0 < step_fraction < 1):
        raise ValueError("step_fraction must be between 0 and 1.")

    for symbol, group in df.sort_values(["symbol", "timestamp"]).groupby(
        "symbol",
        sort=False,
    ):
        n = len(group)

        min_train_size = int(n * min_train_fraction)
        validation_size = int(n * validation_window_fraction)
        step_size = max(1, int(n * step_fraction))

        fold_id = 0
        train_end = min_train_size

        while train_end + purge_gap_candles + validation_size <= n:
            val_start = train_end + purge_gap_candles
            val_end = val_start + validation_size

            train_group = group.iloc[:train_end]
            val_group = group.iloc[val_start:val_end]

            rows.append(
                {
                    "fold_id": int(fold_id),
                    "symbol": symbol,
                    "train_start": str(train_group["timestamp"].min()),
                    "train_end": str(train_group["timestamp"].max()),
                    "validation_start": str(val_group["timestamp"].min()),
                    "validation_end": str(val_group["timestamp"].max()),
                    "train_rows": int(len(train_group)),
                    "validation_rows": int(len(val_group)),
                    "purge_gap_candles": int(purge_gap_candles),
                }
            )

            fold_id += 1
            train_end += step_size

    return pd.DataFrame(rows)


def dataframe_to_markdown_safe(df: pd.DataFrame, max_rows: int = 100) -> str:
    """Convert DataFrame to Markdown with fallback."""
    if df is None or df.empty:
        return "_None._"

    small = df.head(max_rows)

    try:
        return small.to_markdown(index=False)
    except Exception:
        return "```text\n" + small.to_string(index=False) + "\n```"


def write_split_manifest(
    path: str | Path,
    config: dict[str, Any],
    split_df: pd.DataFrame,
    split_safe_threshold: float,
) -> None:
    """Write machine-readable split metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    horizon = int(config["horizon_candles"])

    manifest = {
        "symbol": config.get("symbol"),
        "interval": config.get("interval"),
        "labeled_input_path": config.get("labeled_input_path"),
        "split_output_path": config.get("split_output_path"),
        "target_column": f"y_high_vol_{horizon}_split_safe",
        "future_rv_column": f"future_rv_{horizon}",
        "train_fraction": float(config["train_fraction"]),
        "validation_fraction": float(config["validation_fraction"]),
        "test_fraction": float(config["test_fraction"]),
        "horizon_candles": horizon,
        "purge_gap_candles": int(config["purge_gap_candles"]),
        "volatility_quantile": float(config["volatility_quantile"]),
        "split_safe_volatility_threshold": float(split_safe_threshold),
        "split_counts": {
            str(k): int(v)
            for k, v in split_df["split"].value_counts().to_dict().items()
        },
    }

    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def write_split_report(
    path: str | Path,
    config: dict[str, Any],
    split_summary: pd.DataFrame,
    label_comparison: pd.DataFrame,
    order_overlap_issues: list[dict[str, Any]],
    purge_issues: list[dict[str, Any]],
    walk_forward_folds: pd.DataFrame,
    split_safe_threshold: float,
) -> None:
    """Write human-readable split report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    horizon = int(config["horizon_candles"])

    lines: list[str] = []

    lines.append("# Split Report")
    lines.append("")
    lines.append("## Context")
    lines.append("")
    lines.append(f"- Symbol: `{config.get('symbol')}`")
    lines.append(f"- Interval: `{config.get('interval')}`")
    lines.append(f"- Labeled input path: `{config.get('labeled_input_path')}`")
    lines.append(f"- Split output path: `{config.get('split_output_path')}`")
    lines.append("")
    lines.append("## Split Configuration")
    lines.append("")
    lines.append(f"- Train fraction: `{config['train_fraction']}`")
    lines.append(f"- Validation fraction: `{config['validation_fraction']}`")
    lines.append(f"- Test fraction: `{config['test_fraction']}`")
    lines.append(f"- Horizon candles: `{horizon}`")
    lines.append(f"- Purge gap candles: `{config['purge_gap_candles']}`")
    lines.append(f"- Split-safe volatility threshold: `{split_safe_threshold}`")
    lines.append(f"- Preferred target column: `y_high_vol_{horizon}_split_safe`")
    lines.append("")
    lines.append("## Split Summary")
    lines.append("")
    lines.append(dataframe_to_markdown_safe(split_summary, max_rows=100))
    lines.append("")
    lines.append("## Original Label vs Split-Safe Label")
    lines.append("")
    lines.append(dataframe_to_markdown_safe(label_comparison, max_rows=100))
    lines.append("")
    lines.append("## Order and Overlap Checks")
    lines.append("")
    lines.append(dataframe_to_markdown_safe(pd.DataFrame(order_overlap_issues), max_rows=100))
    lines.append("")
    lines.append("## Purge Gap Checks")
    lines.append("")
    lines.append(dataframe_to_markdown_safe(pd.DataFrame(purge_issues), max_rows=100))
    lines.append("")
    lines.append("## Walk-Forward Fold Preview")
    lines.append("")
    lines.append(dataframe_to_markdown_safe(walk_forward_folds, max_rows=100))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "Random splitting is not used. Splits are chronological because financial "
        "time series are non-iid and future market regimes must not influence earlier training."
    )
    lines.append("")
    lines.append(
        "Rows near train/validation and validation/test boundaries are purged because "
        "the label uses future candles. Purged rows are excluded from model fitting "
        "and model evaluation."
    )
    lines.append("")
    lines.append(
        "From this point onward, modeling should use the split-safe target column, "
        "because its threshold is computed from split == train only."
    )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def run_time_split(config: dict[str, Any]) -> dict[str, Any]:
    """Main entry point for Teaching pipeline 5: splitting and validation."""
    labeled_input_path = Path(config["labeled_input_path"])
    split_output_path = Path(config["split_output_path"])
    split_manifest_path = Path(config["split_manifest_path"])
    split_report_path = Path(config["split_report_path"])

    horizon = int(config["horizon_candles"])
    interval = str(config["interval"])
    interval_delta = interval_to_timedelta(interval)

    train_fraction = float(config["train_fraction"])
    validation_fraction = float(config["validation_fraction"])
    test_fraction = float(config["test_fraction"])
    purge_gap_candles = int(config["purge_gap_candles"])
    volatility_quantile = float(config["volatility_quantile"])

    raw_df = load_labeled_table(labeled_input_path)
    labeled_df = prepare_split_input(raw_df, horizon=horizon)

    split_df = assign_chronological_splits_with_purge(
        labeled_df,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        purge_gap_candles=purge_gap_candles,
    )

    split_safe_threshold = compute_split_train_threshold(
        split_df,
        horizon=horizon,
        quantile=volatility_quantile,
    )

    split_df = add_split_safe_binary_label(
        split_df,
        horizon=horizon,
        threshold=split_safe_threshold,
    )

    label_comparison = compare_existing_and_split_safe_labels(
        split_df,
        horizon=horizon,
    )

    split_summary = summarize_splits(split_df, horizon=horizon)

    order_overlap_issues = validate_split_order_and_overlap(split_df)

    purge_issues = validate_purge_gap(
        split_df,
        horizon=horizon,
        interval_delta=interval_delta,
    )

    raise_if_split_validation_fails(order_overlap_issues, purge_issues)

    walk_forward_config = config.get("walk_forward", {})
    walk_forward_enabled = bool(walk_forward_config.get("enabled", True))

    if walk_forward_enabled:
        walk_forward_folds = make_walk_forward_folds(
            split_df,
            min_train_fraction=float(walk_forward_config.get("min_train_fraction", 0.50)),
            validation_window_fraction=float(
                walk_forward_config.get("validation_window_fraction", 0.10)
            ),
            step_fraction=float(walk_forward_config.get("step_fraction", 0.10)),
            purge_gap_candles=purge_gap_candles,
        )
    else:
        walk_forward_folds = pd.DataFrame()

    split_output_path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_parquet(split_output_path, index=False)

    write_split_manifest(
        path=split_manifest_path,
        config=config,
        split_df=split_df,
        split_safe_threshold=split_safe_threshold,
    )

    write_split_report(
        path=split_report_path,
        config=config,
        split_summary=split_summary,
        label_comparison=label_comparison,
        order_overlap_issues=order_overlap_issues,
        purge_issues=purge_issues,
        walk_forward_folds=walk_forward_folds,
        split_safe_threshold=split_safe_threshold,
    )

    result = {
        "labeled_input_path": str(labeled_input_path),
        "split_output_path": str(split_output_path),
        "split_manifest_path": str(split_manifest_path),
        "split_report_path": str(split_report_path),
        "target_column": f"y_high_vol_{horizon}_split_safe",
        "split_safe_threshold": float(split_safe_threshold),
        "num_rows": int(len(split_df)),
        "split_counts": {
            str(k): int(v)
            for k, v in split_df["split"].value_counts().to_dict().items()
        },
    }

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create chronological train/validation/test splits with purge gaps."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to splitting config YAML, for example configs/splitting.yaml.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    args = parse_args()
    config = load_config(args.config)

    result = run_time_split(config)

    logging.info("Time split complete.")
    logging.info("Result: %s", result)


if __name__ == "__main__":
    main()
