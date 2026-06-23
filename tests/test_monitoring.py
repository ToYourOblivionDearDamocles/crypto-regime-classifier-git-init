import json
from pathlib import Path

import numpy as np
import pandas as pd

from crypto_regime.monitoring.drift import (
    compute_feature_drift,
    compute_prediction_drift,
    population_stability_index,
)
from crypto_regime.monitoring.freshness import (
    compute_freshness_report,
    interval_to_timedelta,
)
from crypto_regime.monitoring.run_monitoring import run_monitoring


def make_feature_data(n: int = 100, shift: float = 0.0) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2025-01-01 00:00:00",
        periods=n,
        freq="5min",
        tz="UTC",
    )

    return pd.DataFrame(
        {
            "symbol": "BTCUSDT",
            "timestamp": timestamps,
            "split": ["train"] * n,
            "feature_a": np.linspace(0.0, 1.0, n) + shift,
            "feature_b": np.sin(np.arange(n) / 10.0) + shift,
        }
    )


def test_interval_to_timedelta():
    assert interval_to_timedelta("5m") == pd.Timedelta(minutes=5)
    assert interval_to_timedelta("1h") == pd.Timedelta(hours=1)
    assert interval_to_timedelta("1d") == pd.Timedelta(days=1)


def test_freshness_historical_passes():
    df = make_feature_data(20)

    result = compute_freshness_report(
        df,
        timestamp_column="timestamp",
        symbol_column="symbol",
        expected_interval="5m",
        mode="historical",
    )

    assert result["check_name"] == "data_freshness"
    assert result["passed"] is True
    assert result["total_missing_interval_count"] == 0


def test_freshness_detects_missing_interval():
    df = make_feature_data(20).drop(index=[5]).reset_index(drop=True)

    result = compute_freshness_report(
        df,
        timestamp_column="timestamp",
        symbol_column="symbol",
        expected_interval="5m",
        mode="historical",
    )

    assert result["total_missing_interval_count"] == 1


def test_population_stability_index_small_for_same_distribution():
    x = pd.Series(np.linspace(0.0, 1.0, 100))

    psi = population_stability_index(x, x, bins=10)

    assert psi < 1e-8


def test_population_stability_index_larger_for_shifted_distribution():
    reference = pd.Series(np.linspace(0.0, 1.0, 100))
    current = pd.Series(np.linspace(1.0, 2.0, 100))

    psi = population_stability_index(reference, current, bins=10)

    assert psi > 0.1


def test_compute_feature_drift():
    reference = make_feature_data(100, shift=0.0)
    current = make_feature_data(100, shift=0.5)

    drift = compute_feature_drift(
        reference,
        current,
        feature_columns=["feature_a", "feature_b"],
        psi_bins=10,
        psi_warning_threshold=0.10,
        psi_alert_threshold=0.25,
    )

    assert set(drift["feature"]) == {"feature_a", "feature_b"}
    assert "psi" in drift.columns
    assert "severity" in drift.columns


def test_compute_prediction_drift():
    reference = pd.DataFrame({"y_score": np.linspace(0.0, 0.5, 100)})
    current = pd.DataFrame({"y_score": np.linspace(0.5, 1.0, 100)})

    result = compute_prediction_drift(
        reference,
        current,
        score_column="y_score",
        decision_threshold=0.5,
        psi_bins=10,
        psi_warning_threshold=0.10,
        psi_alert_threshold=0.25,
    )

    assert result["check_name"] == "prediction_drift"
    assert result["psi"] > 0.1
    assert result["current_predicted_positive_rate"] >= 0.5


def test_run_monitoring_end_to_end(tmp_path: Path):
    reference = make_feature_data(100, shift=0.0)
    current = make_feature_data(100, shift=0.2)
    current["split"] = "validation"

    processed_path = tmp_path / "processed.parquet"
    reference_path = tmp_path / "reference.parquet"
    current_path = tmp_path / "current.parquet"
    schema_path = tmp_path / "feature_schema.json"

    prediction_reference_path = tmp_path / "pred_ref.parquet"
    prediction_current_path = tmp_path / "pred_cur.parquet"

    reference.to_parquet(processed_path, index=False)
    reference.to_parquet(reference_path, index=False)
    current.to_parquet(current_path, index=False)

    schema_path.write_text(
        json.dumps(
            {
                "feature_columns": ["feature_a", "feature_b"],
                "num_features": 2,
            }
        ),
        encoding="utf-8",
    )

    pd.DataFrame({"y_score": np.linspace(0.0, 0.5, 100)}).to_parquet(
        prediction_reference_path,
        index=False,
    )
    pd.DataFrame({"y_score": np.linspace(0.3, 0.8, 100)}).to_parquet(
        prediction_current_path,
        index=False,
    )

    output_dir = tmp_path / "monitoring"

    config = {
        "monitoring_version": "test_monitoring_v1",
        "symbol": "BTCUSDT",
        "interval": "5m",
        "output_dir": str(output_dir),
        "monitoring_summary_path": str(output_dir / "monitoring_summary.json"),
        "monitoring_report_path": str(output_dir / "monitoring_report.md"),
        "feature_schema_path": str(schema_path),
        "freshness": {
            "enabled": True,
            "data_path": str(processed_path),
            "timestamp_column": "timestamp",
            "symbol_column": "symbol",
            "mode": "historical",
            "expected_interval": "5m",
            "max_allowed_delay_minutes": 15,
            "reference_time_utc": None,
        },
        "feature_drift": {
            "enabled": True,
            "reference_data_path": str(reference_path),
            "reference_split_column": "split",
            "reference_split_values": ["train"],
            "current_data_path": str(current_path),
            "current_split_column": "split",
            "current_split_values": ["validation"],
            "psi_bins": 10,
            "psi_warning_threshold": 0.10,
            "psi_alert_threshold": 0.25,
            "max_features_in_report": 50,
        },
        "prediction_drift": {
            "enabled": True,
            "reference_prediction_path": str(prediction_reference_path),
            "current_prediction_path": str(prediction_current_path),
            "score_column": "y_score",
            "decision_threshold": 0.5,
            "psi_bins": 10,
            "psi_warning_threshold": 0.10,
            "psi_alert_threshold": 0.25,
        },
    }

    result = run_monitoring(config)

    assert Path(result["monitoring_summary_path"]).exists()
    assert Path(result["monitoring_report_path"]).exists()
    assert (output_dir / "feature_drift_table.csv").exists()
