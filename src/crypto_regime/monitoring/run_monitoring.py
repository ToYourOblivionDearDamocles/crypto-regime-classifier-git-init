from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from crypto_regime.monitoring.drift import (
    run_feature_drift_from_config,
    run_prediction_drift_from_config,
)
from crypto_regime.monitoring.freshness import run_freshness_from_config


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


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Save JSON payload."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def dataframe_to_markdown_safe(df: pd.DataFrame, max_rows: int = 80) -> str:
    """Convert dataframe to Markdown with fallback."""
    if df is None or df.empty:
        return "_None._"

    small = df.head(max_rows)

    try:
        return small.to_markdown(index=False)
    except Exception:
        return "```text\n" + small.to_string(index=False) + "\n```"


def make_json_safe(value: Any) -> Any:
    """
    Convert objects to JSON-safe form.
    DataFrames are excluded from JSON and should be saved separately.
    """
    if isinstance(value, pd.DataFrame):
        return None

    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items() if not isinstance(v, pd.DataFrame)}

    if isinstance(value, list):
        return [make_json_safe(v) for v in value]

    return value


def build_monitoring_report(
    config: dict[str, Any],
    freshness_result: dict[str, Any] | None,
    feature_drift_result: dict[str, Any] | None,
    prediction_drift_result: dict[str, Any] | None,
) -> str:
    """Build human-readable monitoring report."""
    lines: list[str] = []

    lines.append("# Monitoring Report")
    lines.append("")
    lines.append("## Context")
    lines.append("")
    lines.append(f"- Monitoring version: `{config.get('monitoring_version')}`")
    lines.append(f"- Symbol: `{config.get('symbol')}`")
    lines.append(f"- Interval: `{config.get('interval')}`")
    lines.append("")
    lines.append("## Data Freshness")
    lines.append("")

    if freshness_result is None:
        lines.append("_Freshness check disabled._")
    else:
        lines.append("```json")
        lines.append(json.dumps(make_json_safe(freshness_result), indent=2))
        lines.append("```")

    lines.append("")
    lines.append("## Feature Drift")
    lines.append("")

    if feature_drift_result is None:
        lines.append("_Feature drift check disabled._")
    else:
        lines.append("### Summary")
        lines.append("")
        lines.append("```json")
        lines.append(
            json.dumps(
                {
                    "num_features": feature_drift_result["num_features"],
                    "severity_counts": feature_drift_result["severity_counts"],
                },
                indent=2,
            )
        )
        lines.append("```")
        lines.append("")
        lines.append("### Feature Drift Table")
        lines.append("")
        lines.append(
            dataframe_to_markdown_safe(
                feature_drift_result["drift_table"],
                max_rows=int(config.get("feature_drift", {}).get("max_features_in_report", 50)),
            )
        )

    lines.append("")
    lines.append("## Prediction Drift")
    lines.append("")

    if prediction_drift_result is None:
        lines.append("_Prediction drift check disabled._")
    else:
        lines.append("```json")
        lines.append(json.dumps(make_json_safe(prediction_drift_result), indent=2))
        lines.append("```")

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "Monitoring does not retrain the model. It checks whether incoming data, "
        "features, or prediction distributions have changed enough to require inspection."
    )
    lines.append("")
    lines.append(
        "Data freshness checks whether candles are delayed or internally missing. "
        "Feature drift compares current feature distributions against the training reference. "
        "Prediction drift checks whether model score distributions changed."
    )
    lines.append("")
    lines.append(
        "PSI thresholds are heuristics. They should trigger investigation, not automatic model replacement."
    )
    lines.append("")

    return "\n".join(lines)


def run_monitoring(config: dict[str, Any]) -> dict[str, Any]:
    """Main entry point for Teaching pipeline 9: Monitoring."""
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    freshness_result = None
    feature_drift_result = None
    prediction_drift_result = None

    if config.get("freshness", {}).get("enabled", False):
        freshness_result = run_freshness_from_config(config["freshness"])

    if config.get("feature_drift", {}).get("enabled", False):
        feature_drift_result = run_feature_drift_from_config(
            config["feature_drift"],
            feature_schema_path=config["feature_schema_path"],
        )

        feature_drift_path = output_dir / "feature_drift_table.csv"
        feature_drift_result["drift_table"].to_csv(feature_drift_path, index=False)

    if config.get("prediction_drift", {}).get("enabled", False):
        prediction_drift_result = run_prediction_drift_from_config(
            config["prediction_drift"]
        )

    summary = {
        "monitoring_version": config.get("monitoring_version"),
        "symbol": config.get("symbol"),
        "interval": config.get("interval"),
        "freshness": make_json_safe(freshness_result),
        "feature_drift": make_json_safe(feature_drift_result),
        "prediction_drift": make_json_safe(prediction_drift_result),
    }

    save_json(config["monitoring_summary_path"], summary)

    report_text = build_monitoring_report(
        config=config,
        freshness_result=freshness_result,
        feature_drift_result=feature_drift_result,
        prediction_drift_result=prediction_drift_result,
    )

    report_path = Path(config["monitoring_report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")

    return {
        "monitoring_summary_path": str(config["monitoring_summary_path"]),
        "monitoring_report_path": str(config["monitoring_report_path"]),
        "output_dir": str(output_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run data freshness, feature drift, and prediction drift monitoring."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to monitoring config YAML, for example configs/monitoring.yaml.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    args = parse_args()
    config = load_config(args.config)

    result = run_monitoring(config)

    logging.info("Monitoring complete.")
    logging.info("Result: %s", result)


if __name__ == "__main__":
    main()
