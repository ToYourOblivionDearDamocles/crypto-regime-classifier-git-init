from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


FORBIDDEN_FEATURE_PREFIXES = (
    "future_",
    "target",
    "label",
    "y_",
)

DEFAULT_NON_FEATURE_COLUMNS = {
    "symbol",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "num_trades",
    "split",
    "used_for_label_threshold",
}


@dataclass(frozen=True)
class ModelingDataset:
    """Container for train/validation/test modeling data."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_validation: pd.DataFrame
    y_validation: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    feature_columns: list[str]
    target_column: str


def load_split_dataset(path: str | Path) -> pd.DataFrame:
    """Load split dataset from Parquet or CSV."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Split dataset does not exist: {path}")

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported split dataset file type: {path.suffix}")

    df = df.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    return df


def load_feature_schema(path: str | Path) -> dict[str, Any]:
    """Load feature schema created by Teaching pipeline 3."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Feature schema does not exist: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def infer_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Infer feature columns if feature_schema.json is unavailable.

    Prefer using the schema. This fallback exists for testing/debugging.
    """
    feature_cols: list[str] = []

    for col in df.columns:
        if col in DEFAULT_NON_FEATURE_COLUMNS:
            continue

        if col.startswith(FORBIDDEN_FEATURE_PREFIXES):
            continue

        if col.startswith("_"):
            continue

        feature_cols.append(col)

    return feature_cols


def resolve_feature_columns(
    df: pd.DataFrame,
    feature_schema: dict[str, Any] | None,
) -> list[str]:
    """
    Resolve feature columns.

    Priority:
        1. feature_schema["feature_columns"]
        2. infer from dataframe with leakage-safe exclusions
    """
    if feature_schema is not None and "feature_columns" in feature_schema:
        feature_cols = list(feature_schema["feature_columns"])
    else:
        feature_cols = infer_feature_columns(df)

    missing = [col for col in feature_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Feature columns missing from dataset: {missing}")

    forbidden = [
        col
        for col in feature_cols
        if col.startswith(FORBIDDEN_FEATURE_PREFIXES)
        or col in DEFAULT_NON_FEATURE_COLUMNS
    ]

    if forbidden:
        raise ValueError(f"Forbidden leakage-prone columns selected as features: {forbidden}")

    if not feature_cols:
        raise ValueError("No feature columns resolved.")

    return feature_cols


def validate_modeling_input(
    df: pd.DataFrame,
    target_column: str,
    split_column: str,
    feature_columns: Sequence[str],
) -> None:
    """Validate modeling dataset before fitting models."""
    required = ["symbol", "timestamp", split_column, target_column] + list(feature_columns)
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required modeling columns: {missing}")

    if df.empty:
        raise ValueError("Modeling dataframe is empty.")

    if df[target_column].isna().any():
        raise ValueError(f"Target column contains NaNs: {target_column}")

    if df[list(feature_columns)].isna().any().any():
        raise ValueError("Feature matrix contains NaNs. Fix feature pipeline first.")


def make_modeling_dataset(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    split_column: str = "split",
    train_split: str = "train",
    validation_split: str = "validation",
    test_split: str = "test",
) -> ModelingDataset:
    """
    Create train/validation/test matrices.

    Purged rows are ignored because only explicit train/validation/test split
    names are selected.
    """
    validate_modeling_input(
        df=df,
        target_column=target_column,
        split_column=split_column,
        feature_columns=feature_columns,
    )

    train = df[df[split_column] == train_split].copy()
    validation = df[df[split_column] == validation_split].copy()
    test = df[df[split_column] == test_split].copy()

    if train.empty:
        raise ValueError("Train split is empty.")

    if validation.empty:
        raise ValueError("Validation split is empty.")

    if test.empty:
        raise ValueError("Test split is empty.")

    X_train = train[list(feature_columns)].copy()
    y_train = train[target_column].astype(int).copy()

    X_validation = validation[list(feature_columns)].copy()
    y_validation = validation[target_column].astype(int).copy()

    X_test = test[list(feature_columns)].copy()
    y_test = test[target_column].astype(int).copy()

    return ModelingDataset(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        X_test=X_test,
        y_test=y_test,
        feature_columns=list(feature_columns),
        target_column=target_column,
    )
