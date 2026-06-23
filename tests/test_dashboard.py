import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from crypto_regime.dashboard.loaders import (
    filter_by_date_range,
    join_predictions_to_split_data,
    load_dashboard_artifacts,
)
from crypto_regime.dashboard.plots import (
    flatten_monitoring_summary,
    make_market_timeline_figure,
    make_metric_bar_figure,
)


def make_split_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="5min", tz="UTC"),
            "close": [100.0, 101.0, 102.0],
            "split": ["train", "validation", "test"],
            "future_rv_12": [0.01, 0.02, 0.03],
            "y_high_vol_12_split_safe": [0, 1, 0],
        }
    )


def make_prediction_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "timestamp": pd.date_range("2025-01-01 00:05:00", periods=2, freq="5min", tz="UTC"),
            "y_score": [0.7, 0.2],
            "y_pred_at_0_5": [1, 0],
        }
    )


def test_join_predictions_to_split_data():
    split_df = make_split_df()
    pred_df = make_prediction_df()

    joined = join_predictions_to_split_data(split_df, pred_df)

    assert "y_score" in joined.columns
    assert joined["y_score"].notna().sum() == 2


def test_filter_by_date_range():
    df = make_split_df()

    filtered = filter_by_date_range(
        df,
        start=pd.Timestamp("2025-01-01 00:05:00", tz="UTC"),
        end=pd.Timestamp("2025-01-01 00:10:00", tz="UTC"),
    )

    assert len(filtered) == 2


def test_make_market_timeline_figure_returns_plotly_figure():
    df = join_predictions_to_split_data(make_split_df(), make_prediction_df())

    fig = make_market_timeline_figure(
        df,
        timestamp_col="timestamp",
        close_col="close",
        target_col="y_high_vol_12_split_safe",
        score_col="y_score",
        threshold=0.5,
    )

    assert isinstance(fig, go.Figure)


def test_make_metric_bar_figure_returns_plotly_figure():
    metrics = pd.DataFrame(
        {
            "model_name": ["baseline", "logistic"],
            "split": ["validation", "validation"],
            "pr_auc": [0.2, 0.7],
        }
    )

    fig = make_metric_bar_figure(metrics, metric_col="pr_auc")

    assert isinstance(fig, go.Figure)


def test_flatten_monitoring_summary():
    summary = {
        "freshness": {
            "passed": True,
            "total_missing_interval_count": 0,
        },
        "feature_drift": {
            "severity_counts": {"ok": 10},
        },
        "prediction_drift": {
            "severity": "ok",
            "psi": 0.01,
        },
    }

    flat = flatten_monitoring_summary(summary)

    assert not flat.empty
    assert set(flat["check"]) == {
        "data_freshness",
        "feature_drift",
        "prediction_drift",
    }


def test_load_dashboard_artifacts(tmp_path: Path):
    split_path = tmp_path / "splits.parquet"
    prediction_path = tmp_path / "predictions.parquet"
    metrics_path = tmp_path / "metrics.csv"
    monitoring_path = tmp_path / "monitoring.json"
    feature_schema_path = tmp_path / "feature_schema.json"

    make_split_df().to_parquet(split_path, index=False)
    make_prediction_df().to_parquet(prediction_path, index=False)

    pd.DataFrame(
        {
            "model_name": ["logistic"],
            "split": ["test"],
            "pr_auc": [0.7],
        }
    ).to_csv(metrics_path, index=False)

    monitoring_path.write_text(
        json.dumps({"freshness": {"passed": True, "total_missing_interval_count": 0}}),
        encoding="utf-8",
    )

    feature_schema_path.write_text(
        json.dumps({"feature_columns": ["a", "b"], "num_features": 2}),
        encoding="utf-8",
    )

    config = {
        "paths": {
            "split_data_path": str(split_path),
            "prediction_path": str(prediction_path),
            "evaluation_metrics_table_path": str(metrics_path),
            "monitoring_summary_path": str(monitoring_path),
            "feature_schema_path": str(feature_schema_path),
        },
        "columns": {
            "timestamp": "timestamp",
            "symbol": "symbol",
        },
    }

    artifacts = load_dashboard_artifacts(config)

    assert not artifacts["timeline_df"].empty
    assert not artifacts["metrics_df"].empty
    assert artifacts["monitoring_summary"]
    assert artifacts["feature_schema"]["num_features"] == 2
