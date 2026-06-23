from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
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


def load_feature_schema(path: str | Path) -> dict[str, Any]:
    """Load feature schema JSON."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Feature schema does not exist: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def filter_by_split_values(
    df: pd.DataFrame,
    split_column: str | None,
    split_values: Sequence[str] | None,
) -> pd.DataFrame:
    """
    Filter dataframe by split values if requested.
    """
    if split_column is None or split_values is None:
        return df.copy()

    if split_column not in df.columns:
        raise ValueError(f"Missing split column: {split_column}")

    return df[df[split_column].isin(split_values)].copy()


def make_psi_bins(
    reference: pd.Series,
    bins: int,
) -> np.ndarray:
    """
    Create PSI bin edges from reference quantiles.

    Duplicate quantile edges are removed.
    """
    ref = pd.to_numeric(reference, errors="coerce").dropna()

    if ref.empty:
        raise ValueError("Reference series is empty after numeric conversion.")

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(ref, quantiles)
    edges = np.unique(edges)

    if len(edges) < 2:
        # Constant feature: create a tiny artificial interval.
        value = float(ref.iloc[0])
        edges = np.array([value - 1e-12, value + 1e-12])

    edges[0] = -np.inf
    edges[-1] = np.inf

    return edges


def population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """
    Compute Population Stability Index.

    PSI = sum((current_pct - reference_pct) * log(current_pct / reference_pct))

    Interpretation convention:
        < 0.10  small drift
        0.10-0.25 moderate drift
        > 0.25 large drift
    """
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()

    if ref.empty or cur.empty:
        return float("nan")

    edges = make_psi_bins(ref, bins=bins)

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)

    ref_pct = np.clip(ref_pct, epsilon, None)
    cur_pct = np.clip(cur_pct, epsilon, None)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))

    return float(psi)


def classify_psi(
    psi: float,
    warning_threshold: float,
    alert_threshold: float,
) -> str:
    """Classify PSI severity."""
    if np.isnan(psi):
        return "unknown"

    if psi >= alert_threshold:
        return "alert"

    if psi >= warning_threshold:
        return "warning"

    return "ok"


def compute_numeric_summary(series: pd.Series) -> dict[str, Any]:
    """Compute compact numeric summary."""
    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
        }

    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p25": float(values.quantile(0.25)),
        "median": float(values.median()),
        "p75": float(values.quantile(0.75)),
        "max": float(values.max()),
    }


def compute_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    psi_bins: int,
    psi_warning_threshold: float,
    psi_alert_threshold: float,
) -> pd.DataFrame:
    """
    Compute feature drift table for numeric features.
    """
    rows: list[dict[str, Any]] = []

    for feature in feature_columns:
        if feature not in reference_df.columns:
            raise ValueError(f"Feature missing from reference dataframe: {feature}")

        if feature not in current_df.columns:
            raise ValueError(f"Feature missing from current dataframe: {feature}")

        ref = reference_df[feature]
        cur = current_df[feature]

        psi = population_stability_index(ref, cur, bins=psi_bins)
        severity = classify_psi(
            psi,
            warning_threshold=psi_warning_threshold,
            alert_threshold=psi_alert_threshold,
        )

        ref_summary = compute_numeric_summary(ref)
        cur_summary = compute_numeric_summary(cur)

        rows.append(
            {
                "feature": feature,
                "psi": psi,
                "severity": severity,
                "reference_count": ref_summary["count"],
                "current_count": cur_summary["count"],
                "reference_mean": ref_summary["mean"],
                "current_mean": cur_summary["mean"],
                "reference_std": ref_summary["std"],
                "current_std": cur_summary["std"],
                "reference_median": ref_summary["median"],
                "current_median": cur_summary["median"],
                "reference_min": ref_summary["min"],
                "current_min": cur_summary["min"],
                "reference_max": ref_summary["max"],
                "current_max": cur_summary["max"],
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    return result.sort_values(["severity", "psi"], ascending=[True, False]).reset_index(drop=True)


def run_feature_drift_from_config(
    config: dict[str, Any],
    feature_schema_path: str | Path,
) -> dict[str, Any]:
    """Run feature drift from config section."""
    schema = load_feature_schema(feature_schema_path)
    feature_columns = list(schema["feature_columns"])

    reference = load_table(config["reference_data_path"])
    current = load_table(config["current_data_path"])

    reference = filter_by_split_values(
        reference,
        config.get("reference_split_column"),
        config.get("reference_split_values"),
    )

    current = filter_by_split_values(
        current,
        config.get("current_split_column"),
        config.get("current_split_values"),
    )

    if reference.empty:
        raise ValueError("Reference feature dataframe is empty after filtering.")

    if current.empty:
        raise ValueError("Current feature dataframe is empty after filtering.")

    drift_df = compute_feature_drift(
        reference,
        current,
        feature_columns=feature_columns,
        psi_bins=int(config.get("psi_bins", 10)),
        psi_warning_threshold=float(config.get("psi_warning_threshold", 0.10)),
        psi_alert_threshold=float(config.get("psi_alert_threshold", 0.25)),
    )

    counts = drift_df["severity"].value_counts().to_dict() if not drift_df.empty else {}

    return {
        "check_name": "feature_drift",
        "num_features": int(len(feature_columns)),
        "severity_counts": {str(k): int(v) for k, v in counts.items()},
        "drift_table": drift_df,
    }


def compute_prediction_drift(
    reference_predictions: pd.DataFrame,
    current_predictions: pd.DataFrame,
    *,
    score_column: str,
    decision_threshold: float,
    psi_bins: int,
    psi_warning_threshold: float,
    psi_alert_threshold: float,
) -> dict[str, Any]:
    """
    Compute prediction score/probability drift.
    """
    if score_column not in reference_predictions.columns:
        raise ValueError(f"Missing score column in reference predictions: {score_column}")

    if score_column not in current_predictions.columns:
        raise ValueError(f"Missing score column in current predictions: {score_column}")

    ref = pd.to_numeric(reference_predictions[score_column], errors="coerce").dropna()
    cur = pd.to_numeric(current_predictions[score_column], errors="coerce").dropna()

    if ref.empty:
        raise ValueError("Reference prediction scores are empty.")

    if cur.empty:
        raise ValueError("Current prediction scores are empty.")

    psi = population_stability_index(ref, cur, bins=psi_bins)
    severity = classify_psi(
        psi,
        warning_threshold=psi_warning_threshold,
        alert_threshold=psi_alert_threshold,
    )

    return {
        "check_name": "prediction_drift",
        "score_column": score_column,
        "psi": psi,
        "severity": severity,
        "reference_summary": compute_numeric_summary(ref),
        "current_summary": compute_numeric_summary(cur),
        "reference_predicted_positive_rate": float((ref >= decision_threshold).mean()),
        "current_predicted_positive_rate": float((cur >= decision_threshold).mean()),
        "decision_threshold": float(decision_threshold),
    }


def run_prediction_drift_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Run prediction drift from config section."""
    reference = load_table(config["reference_prediction_path"])
    current = load_table(config["current_prediction_path"])

    return compute_prediction_drift(
        reference,
        current,
        score_column=str(config.get("score_column", "y_score")),
        decision_threshold=float(config.get("decision_threshold", 0.5)),
        psi_bins=int(config.get("psi_bins", 10)),
        psi_warning_threshold=float(config.get("psi_warning_threshold", 0.10)),
        psi_alert_threshold=float(config.get("psi_alert_threshold", 0.25)),
    )
