from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def _to_float_or_none(value: Any) -> float | None:
    """Convert metric values to JSON-safe float or None."""
    if value is None:
        return None

    try:
        value_float = float(value)
    except Exception:
        return None

    if np.isnan(value_float) or np.isinf(value_float):
        return None

    return value_float


def _safe_metric(fn, *args, **kwargs) -> float | None:
    """Run metric function and return None if metric is undefined."""
    try:
        return _to_float_or_none(fn(*args, **kwargs))
    except Exception:
        return None


def precision_at_top_fraction(
    y_true_binary: Sequence[int],
    score: Sequence[float],
    top_fraction: float,
) -> float | None:
    """
    Precision among the top-scored fraction of samples.

    Example:
        top_fraction = 0.05 means:
        among the top 5% highest-risk predictions, what fraction were true positives?
    """
    if not (0 < top_fraction <= 1):
        raise ValueError("top_fraction must be in (0, 1].")

    y = np.asarray(y_true_binary).astype(int)
    s = np.asarray(score).astype(float)

    if len(y) == 0:
        return None

    n_top = max(1, int(np.ceil(len(y) * top_fraction)))
    order = np.argsort(-s)
    top_idx = order[:n_top]

    return float(np.mean(y[top_idx]))


def recall_at_top_fraction(
    y_true_binary: Sequence[int],
    score: Sequence[float],
    top_fraction: float,
) -> float | None:
    """
    Recall captured by the top-scored fraction of samples.
    """
    if not (0 < top_fraction <= 1):
        raise ValueError("top_fraction must be in (0, 1].")

    y = np.asarray(y_true_binary).astype(int)
    s = np.asarray(score).astype(float)

    total_positive = int(y.sum())

    if total_positive == 0:
        return None

    n_top = max(1, int(np.ceil(len(y) * top_fraction)))
    order = np.argsort(-s)
    top_idx = order[:n_top]

    return float(y[top_idx].sum() / total_positive)


def compute_binary_classification_metrics(
    y_true: Sequence[int],
    y_score: Sequence[float],
    thresholds: Sequence[float] = (0.5,),
    top_k_fractions: Sequence[float] = (0.05, 0.10),
    positive_label: int = 1,
) -> dict[str, Any]:
    """
    Compute binary classification metrics.

    y_score should be the predicted probability or score for the positive class.
    """
    y_raw = np.asarray(y_true)
    y = (y_raw == positive_label).astype(int)
    score = np.asarray(y_score).astype(float)

    if len(y) != len(score):
        raise ValueError("y_true and y_score must have the same length.")

    if len(y) == 0:
        raise ValueError("Cannot compute metrics on empty arrays.")

    proba_matrix = np.column_stack([1.0 - score, score])

    metrics: dict[str, Any] = {
        "task_type": "binary_classification",
        "num_samples": int(len(y)),
        "positive_count": int(y.sum()),
        "negative_count": int((1 - y).sum()),
        "positive_rate": float(np.mean(y)),
        "mean_score": float(np.mean(score)),
        "brier_score": _safe_metric(brier_score_loss, y, score),
        "log_loss": _safe_metric(log_loss, y, proba_matrix, labels=[0, 1]),
        "roc_auc": None,
        "pr_auc": None,
        "threshold_metrics": {},
        "top_k_metrics": {},
    }

    if len(np.unique(y)) == 2:
        metrics["roc_auc"] = _safe_metric(roc_auc_score, y, score)
        metrics["pr_auc"] = _safe_metric(average_precision_score, y, score)

    for threshold in thresholds:
        threshold_float = float(threshold)
        y_pred = (score >= threshold_float).astype(int)

        cm = confusion_matrix(y, y_pred, labels=[0, 1])

        metrics["threshold_metrics"][str(threshold_float)] = {
            "accuracy": _safe_metric(accuracy_score, y, y_pred),
            "balanced_accuracy": _safe_metric(balanced_accuracy_score, y, y_pred),
            "precision": _safe_metric(
                precision_score,
                y,
                y_pred,
                zero_division=0,
            ),
            "recall": _safe_metric(
                recall_score,
                y,
                y_pred,
                zero_division=0,
            ),
            "f1": _safe_metric(
                f1_score,
                y,
                y_pred,
                zero_division=0,
            ),
            "predicted_positive_rate": float(np.mean(y_pred)),
            "confusion_matrix": cm.tolist(),
        }

    for frac in top_k_fractions:
        frac_float = float(frac)

        metrics["top_k_metrics"][str(frac_float)] = {
            "precision_at_top_k": precision_at_top_fraction(y, score, frac_float),
            "recall_at_top_k": recall_at_top_fraction(y, score, frac_float),
        }

    return metrics


def compute_multiclass_classification_metrics(
    y_true: Sequence[int | str],
    y_pred: Sequence[int | str] | None = None,
    y_proba: np.ndarray | None = None,
    class_labels: Sequence[int | str] | None = None,
) -> dict[str, Any]:
    """
    Compute multiclass classification metrics.

    If y_proba is provided and y_pred is None, prediction is argmax over probabilities.
    """
    y = np.asarray(y_true)

    if len(y) == 0:
        raise ValueError("Cannot compute multiclass metrics on empty arrays.")

    if class_labels is None:
        labels = np.unique(y)
    else:
        labels = np.asarray(class_labels)

    if y_pred is None:
        if y_proba is None:
            raise ValueError("Either y_pred or y_proba must be provided.")

        y_pred_array = labels[np.argmax(y_proba, axis=1)]
    else:
        y_pred_array = np.asarray(y_pred)

    metrics: dict[str, Any] = {
        "task_type": "multiclass_classification",
        "num_samples": int(len(y)),
        "class_labels": [str(label) for label in labels],
        "accuracy": _safe_metric(accuracy_score, y, y_pred_array),
        "balanced_accuracy": _safe_metric(balanced_accuracy_score, y, y_pred_array),
        "macro_precision": _safe_metric(
            precision_score,
            y,
            y_pred_array,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": _safe_metric(
            recall_score,
            y,
            y_pred_array,
            average="macro",
            zero_division=0,
        ),
        "macro_f1": _safe_metric(
            f1_score,
            y,
            y_pred_array,
            average="macro",
            zero_division=0,
        ),
        "weighted_precision": _safe_metric(
            precision_score,
            y,
            y_pred_array,
            average="weighted",
            zero_division=0,
        ),
        "weighted_recall": _safe_metric(
            recall_score,
            y,
            y_pred_array,
            average="weighted",
            zero_division=0,
        ),
        "weighted_f1": _safe_metric(
            f1_score,
            y,
            y_pred_array,
            average="weighted",
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y, y_pred_array, labels=labels).tolist(),
        "per_class": {},
        "log_loss": None,
        "roc_auc_ovr": None,
    }

    for label in labels:
        y_binary = (y == label).astype(int)
        pred_binary = (y_pred_array == label).astype(int)

        metrics["per_class"][str(label)] = {
            "precision": _safe_metric(
                precision_score,
                y_binary,
                pred_binary,
                zero_division=0,
            ),
            "recall": _safe_metric(
                recall_score,
                y_binary,
                pred_binary,
                zero_division=0,
            ),
            "f1": _safe_metric(
                f1_score,
                y_binary,
                pred_binary,
                zero_division=0,
            ),
            "support": int(y_binary.sum()),
        }

    if y_proba is not None and len(np.unique(y)) > 1:
        metrics["log_loss"] = _safe_metric(log_loss, y, y_proba, labels=labels)

        if y_proba.shape[1] == len(labels):
            metrics["roc_auc_ovr"] = _safe_metric(
                roc_auc_score,
                y,
                y_proba,
                multi_class="ovr",
                labels=labels,
            )

    return metrics


def compute_regression_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> dict[str, Any]:
    """
    Compute regression metrics.

    Useful for later variants that predict continuous future realized volatility.
    """
    y = np.asarray(y_true).astype(float)
    pred = np.asarray(y_pred).astype(float)

    if len(y) != len(pred):
        raise ValueError("y_true and y_pred must have the same length.")

    if len(y) == 0:
        raise ValueError("Cannot compute regression metrics on empty arrays.")

    mse = _safe_metric(mean_squared_error, y, pred)

    metrics = {
        "task_type": "regression",
        "num_samples": int(len(y)),
        "target_mean": float(np.mean(y)),
        "prediction_mean": float(np.mean(pred)),
        "mae": _safe_metric(mean_absolute_error, y, pred),
        "mse": mse,
        "rmse": float(np.sqrt(mse)) if mse is not None else None,
        "median_absolute_error": _safe_metric(median_absolute_error, y, pred),
        "r2": _safe_metric(r2_score, y, pred),
        "explained_variance": _safe_metric(explained_variance_score, y, pred),
        "mape": None,
    }

    if not np.any(y == 0):
        metrics["mape"] = _safe_metric(mean_absolute_percentage_error, y, pred)

    return metrics


def compute_metrics(
    task_type: str,
    y_true: Sequence[Any],
    *,
    y_score: Sequence[float] | None = None,
    y_pred: Sequence[Any] | None = None,
    y_proba: np.ndarray | None = None,
    thresholds: Sequence[float] = (0.5,),
    top_k_fractions: Sequence[float] = (0.05, 0.10),
    positive_label: int = 1,
    class_labels: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """
    Task-aware metric dispatcher.

    Supported:
        binary_classification
        multiclass_classification
        regression
    """
    if task_type == "binary_classification":
        if y_score is None:
            if y_proba is not None:
                y_score = y_proba[:, 1]
            elif y_pred is not None:
                y_score = np.asarray(y_pred).astype(float)
            else:
                raise ValueError("Binary classification requires y_score, y_proba, or y_pred.")

        return compute_binary_classification_metrics(
            y_true=y_true,
            y_score=y_score,
            thresholds=thresholds,
            top_k_fractions=top_k_fractions,
            positive_label=positive_label,
        )

    if task_type == "multiclass_classification":
        return compute_multiclass_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            class_labels=class_labels,
        )

    if task_type == "regression":
        if y_pred is None:
            raise ValueError("Regression requires y_pred.")

        return compute_regression_metrics(
            y_true=y_true,
            y_pred=y_pred,
        )

    raise ValueError(f"Unsupported task_type: {task_type}")
