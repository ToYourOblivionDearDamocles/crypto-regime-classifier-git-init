from __future__ import annotations

from typing import Any

import pandas as pd
import requests
import streamlit as st

from crypto_regime.dashboard.loaders import (
    filter_by_date_range,
    filter_by_symbol,
    load_dashboard_artifacts,
    load_yaml,
)
from crypto_regime.dashboard.plots import (
    flatten_monitoring_summary,
    make_future_rv_figure,
    make_market_timeline_figure,
    make_metric_bar_figure,
    make_split_counts_figure,
)


DEFAULT_CONFIG_PATH = "configs/dashboard.yaml"


@st.cache_data(show_spinner=False)
def cached_load_config(config_path: str) -> dict[str, Any]:
    return load_yaml(config_path)


@st.cache_data(show_spinner=False)
def cached_load_artifacts(config: dict[str, Any]) -> dict[str, Any]:
    return load_dashboard_artifacts(config)


def render_sidebar(config: dict[str, Any], artifacts: dict[str, Any]) -> tuple[str, Any, Any]:
    st.sidebar.title("Dashboard Controls")

    symbol_default = str(config.get("symbol", "BTCUSDT"))
    symbol = st.sidebar.text_input("Symbol", value=symbol_default)

    timeline_df = artifacts["timeline_df"]
    timestamp_col = config.get("columns", {}).get("timestamp", "timestamp")

    start_date = None
    end_date = None

    if not timeline_df.empty and timestamp_col in timeline_df.columns:
        min_ts = pd.to_datetime(timeline_df[timestamp_col]).min()
        max_ts = pd.to_datetime(timeline_df[timestamp_col]).max()

        start_date = st.sidebar.date_input("Start date", value=min_ts.date())
        end_date = st.sidebar.date_input("End date", value=max_ts.date())

    return symbol, start_date, end_date


def page_project_overview(config: dict[str, Any], artifacts: dict[str, Any]) -> None:
    st.header("Project Overview")

    st.markdown(
        """
        This project is an end-to-end machine learning engineering system for
        crypto market-regime classification.

        The current milestone is **binary high-volatility classification**:
        predict whether BTCUSDT will enter a high-volatility regime over the next hour.

        This is a **risk signal service**, not a trading bot and not a PnL backtest.
        """
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Task", config.get("task_type", "unknown"))
    col2.metric("Model version", config.get("model_version", "unknown"))
    col3.metric("Interval", config.get("interval", "unknown"))

    st.subheader("Pipeline")
    st.code(
        """
Data ingestion
  -> Data validation
  -> Feature engineering
  -> Label generation
  -> Chronological splitting
  -> Modeling
  -> Evaluation
  -> Serving
  -> Monitoring
  -> Dashboard
        """.strip()
    )

    st.subheader("Artifact Status")
    st.dataframe(artifacts["artifact_status"], width="stretch")

    feature_schema = artifacts.get("feature_schema", {})
    if feature_schema:
        st.subheader("Feature Schema")
        st.write(f"Number of features: {feature_schema.get('num_features')}")
        with st.expander("Feature columns"):
            st.write(feature_schema.get("feature_columns", []))

    label_config = artifacts.get("label_config", {})
    if label_config:
        st.subheader("Label Definition")
        st.json(
            {
                "label_type": label_config.get("label_type"),
                "horizon_candles": label_config.get("horizon_candles"),
                "horizon_minutes": label_config.get("horizon_minutes"),
                "volatility_quantile": label_config.get("volatility_quantile"),
                "label_column": label_config.get("label_column"),
            }
        )


def page_market_timeline(
    config: dict[str, Any],
    artifacts: dict[str, Any],
    symbol: str,
    start_date,
    end_date,
) -> None:
    st.header("Market Regime Timeline")

    cols = config.get("columns", {})
    timestamp_col = cols.get("timestamp", "timestamp")
    symbol_col = cols.get("symbol", "symbol")
    close_col = cols.get("close", "close")
    target_col = cols.get("target", "y_high_vol_12_split_safe")
    score_col = cols.get("prediction_score", "y_score")
    future_rv_col = cols.get("future_rv", "future_rv_12")

    df = artifacts["timeline_df"]

    if df.empty:
        st.warning("No timeline data found. Run earlier pipelines and check dashboard config paths.")
        return

    df = filter_by_symbol(df, symbol, symbol_col=symbol_col)
    df = filter_by_date_range(df, start_date, end_date, timestamp_col=timestamp_col)

    if df.empty:
        st.warning("No rows remain after symbol/date filtering.")
        return

    threshold = float(config.get("decision_threshold", 0.5))

    st.plotly_chart(
        make_market_timeline_figure(
            df,
            timestamp_col=timestamp_col,
            close_col=close_col,
            target_col=target_col,
            score_col=score_col,
            threshold=threshold,
        ),
        width="stretch",
    )

    st.plotly_chart(
        make_future_rv_figure(
            df,
            timestamp_col=timestamp_col,
            future_rv_col=future_rv_col,
        ),
        width="stretch",
    )

    st.plotly_chart(
        make_split_counts_figure(df, split_col=cols.get("split", "split")),
        width="stretch",
    )

    st.subheader("Filtered Data Preview")
    st.dataframe(df.head(100), width="stretch")


def page_model_performance(config: dict[str, Any], artifacts: dict[str, Any]) -> None:
    st.header("Model Performance")

    metrics_df = artifacts["metrics_df"]

    if metrics_df.empty:
        st.warning("No evaluation metrics found. Run Teaching pipeline 7 first.")
        return

    st.subheader("Metrics Table")
    st.dataframe(metrics_df, width="stretch")

    candidate_metrics = [
        "roc_auc",
        "pr_auc",
        "brier_score",
        "threshold_metrics.0.5.balanced_accuracy",
        "threshold_metrics.0.5.precision",
        "threshold_metrics.0.5.recall",
        "threshold_metrics.0.5.f1",
        "top_k_metrics.0.05.precision_at_top_k",
        "top_k_metrics.0.1.precision_at_top_k",
    ]

    available = [metric for metric in candidate_metrics if metric in metrics_df.columns]

    if not available:
        st.info("No known metric columns available for plotting.")
        return

    metric_col = st.selectbox("Metric to plot", available)

    st.plotly_chart(
        make_metric_bar_figure(metrics_df, metric_col=metric_col),
        width="stretch",
    )

    st.markdown(
        """
        Accuracy alone is not enough for this project. High-volatility periods are
        minority events, so PR-AUC, recall, precision@top-k, and calibration-related
        metrics are more meaningful.
        """
    )


def page_monitoring(config: dict[str, Any], artifacts: dict[str, Any]) -> None:
    st.header("Monitoring")

    summary = artifacts.get("monitoring_summary", {})

    if not summary:
        st.warning("No monitoring summary found. Run Teaching pipeline 9 first.")
        return

    st.subheader("Monitoring Summary")
    flat = flatten_monitoring_summary(summary)

    if not flat.empty:
        st.dataframe(flat, width="stretch")

    with st.expander("Raw monitoring summary JSON"):
        st.json(summary)

    report_text = artifacts.get("monitoring_report", "")
    if report_text:
        st.subheader("Monitoring Report")
        st.markdown(report_text)


def page_api_demo(config: dict[str, Any]) -> None:
    st.header("API Demo")

    api_config = config.get("api", {})
    base_url = api_config.get("base_url", "http://127.0.0.1:8000")

    st.markdown("This page checks whether the FastAPI serving layer is running.")

    st.code("uvicorn crypto_regime.api.main:app --reload")

    col1, col2 = st.columns(2)

    if col1.button("Check /health"):
        url = base_url + api_config.get("health_endpoint", "/health")

        try:
            response = requests.get(url, timeout=5)
            st.write("Status code:", response.status_code)
            st.json(response.json())
        except Exception as exc:
            st.error(f"Request failed: {exc}")

    if col2.button("Check /model-info"):
        url = base_url + api_config.get("model_info_endpoint", "/model-info")

        try:
            response = requests.get(url, timeout=5)
            st.write("Status code:", response.status_code)
            st.json(response.json())
        except Exception as exc:
            st.error(f"Request failed: {exc}")

    st.subheader("Example Prediction Request")
    st.code(
        """
{
  "symbol": "BTCUSDT",
  "candles": [
    {
      "timestamp": "2025-01-01T00:00:00Z",
      "open": 100000.0,
      "high": 100500.0,
      "low": 99500.0,
      "close": 100200.0,
      "volume": 123.45
    }
  ]
}
        """.strip(),
        language="json",
    )

    st.info(
        "For real prediction, send enough historical candles to compute the largest rolling window."
    )


def main() -> None:
    st.set_page_config(
        page_title="Crypto Regime Classifier",
        layout="wide",
    )

    st.title("Crypto Market Regime Classifier")

    config_path = st.sidebar.text_input("Dashboard config path", value=DEFAULT_CONFIG_PATH)

    try:
        config = cached_load_config(config_path)
        artifacts = cached_load_artifacts(config)
    except Exception as exc:
        st.error(f"Failed to load dashboard config/artifacts: {exc}")
        return

    symbol, start_date, end_date = render_sidebar(config, artifacts)

    pages = [
        "Project Overview",
        "Market Regime Timeline",
        "Model Performance",
        "Monitoring",
        "API Demo",
    ]

    page = st.sidebar.radio("Page", pages)

    if page == "Project Overview":
        page_project_overview(config, artifacts)

    elif page == "Market Regime Timeline":
        page_market_timeline(config, artifacts, symbol, start_date, end_date)

    elif page == "Model Performance":
        page_model_performance(config, artifacts)

    elif page == "Monitoring":
        page_monitoring(config, artifacts)

    elif page == "API Demo":
        page_api_demo(config)


if __name__ == "__main__":
    main()
