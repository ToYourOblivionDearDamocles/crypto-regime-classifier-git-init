from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go


def empty_figure(message: str) -> go.Figure:
    """Return a simple empty Plotly figure with a message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
    )
    fig.update_layout(height=350)
    return fig


def make_market_timeline_figure(
    df: pd.DataFrame,
    *,
    timestamp_col: str,
    close_col: str,
    target_col: str | None = None,
    score_col: str | None = None,
    threshold: float | None = None,
) -> go.Figure:
    """
    Market-regime timeline figure.

    Shows:
        close price
        true high-volatility labels
        predicted probability / score if available
    """
    if df.empty:
        return empty_figure("No timeline data available.")

    if timestamp_col not in df.columns or close_col not in df.columns:
        return empty_figure("Timeline data is missing timestamp or close column.")

    plot_df = df.copy().sort_values(timestamp_col)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=plot_df[timestamp_col],
            y=plot_df[close_col],
            mode="lines",
            name="Close",
            yaxis="y1",
        )
    )

    if target_col and target_col in plot_df.columns:
        high_vol = plot_df[plot_df[target_col] == 1]

        if not high_vol.empty:
            fig.add_trace(
                go.Scatter(
                    x=high_vol[timestamp_col],
                    y=high_vol[close_col],
                    mode="markers",
                    name="True high-vol label",
                    yaxis="y1",
                    marker={"size": 7},
                )
            )

    if score_col and score_col in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df[timestamp_col],
                y=plot_df[score_col],
                mode="lines",
                name="Predicted probability / score",
                yaxis="y2",
            )
        )

        if threshold is not None:
            fig.add_trace(
                go.Scatter(
                    x=plot_df[timestamp_col],
                    y=[threshold] * len(plot_df),
                    mode="lines",
                    name=f"Decision threshold={threshold}",
                    yaxis="y2",
                    line={"dash": "dash"},
                )
            )

    fig.update_layout(
        title="Market Regime Timeline",
        xaxis_title="Timestamp",
        yaxis={"title": "Close"},
        yaxis2={
            "title": "Probability / Score",
            "overlaying": "y",
            "side": "right",
            "range": [0, 1],
        },
        legend={"orientation": "h"},
        height=500,
    )

    return fig


def make_future_rv_figure(
    df: pd.DataFrame,
    *,
    timestamp_col: str,
    future_rv_col: str,
) -> go.Figure:
    """Plot future realized volatility if available."""
    if df.empty:
        return empty_figure("No future volatility data available.")

    if timestamp_col not in df.columns or future_rv_col not in df.columns:
        return empty_figure("Future RV column is unavailable.")

    plot_df = df.copy().sort_values(timestamp_col)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df[timestamp_col],
            y=plot_df[future_rv_col],
            mode="lines",
            name=future_rv_col,
        )
    )

    fig.update_layout(
        title="Future Realized Volatility",
        xaxis_title="Timestamp",
        yaxis_title=future_rv_col,
        height=350,
    )

    return fig


def make_split_counts_figure(df: pd.DataFrame, split_col: str = "split") -> go.Figure:
    """Bar chart for split counts."""
    if df.empty or split_col not in df.columns:
        return empty_figure("No split data available.")

    counts = df[split_col].value_counts().reset_index()
    counts.columns = ["split", "count"]

    fig = go.Figure(
        data=[
            go.Bar(
                x=counts["split"],
                y=counts["count"],
                name="Rows",
            )
        ]
    )

    fig.update_layout(
        title="Split Counts",
        xaxis_title="Split",
        yaxis_title="Rows",
        height=350,
    )

    return fig


def make_metric_bar_figure(
    metrics_df: pd.DataFrame,
    metric_col: str,
    model_col: str = "model_name",
    split_col: str = "split",
) -> go.Figure:
    """Compare models by one metric."""
    if metrics_df.empty:
        return empty_figure("No metric table available.")

    if metric_col not in metrics_df.columns:
        return empty_figure(f"Metric column not found: {metric_col}")

    if model_col not in metrics_df.columns or split_col not in metrics_df.columns:
        return empty_figure("Metric table missing model or split column.")

    plot_df = metrics_df[[model_col, split_col, metric_col]].copy()
    plot_df[metric_col] = pd.to_numeric(plot_df[metric_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[metric_col])

    if plot_df.empty:
        return empty_figure(f"No numeric values for metric: {metric_col}")

    fig = go.Figure()

    for split_name, group in plot_df.groupby(split_col, sort=False):
        fig.add_trace(
            go.Bar(
                x=group[model_col],
                y=group[metric_col],
                name=str(split_name),
            )
        )

    fig.update_layout(
        title=f"Model Comparison: {metric_col}",
        xaxis_title="Model",
        yaxis_title=metric_col,
        barmode="group",
        height=400,
    )

    return fig


def flatten_monitoring_summary(summary: dict[str, Any]) -> pd.DataFrame:
    """Flatten high-level monitoring summary into a small table."""
    if not summary:
        return pd.DataFrame()

    rows = []

    freshness = summary.get("freshness")
    if freshness:
        rows.append(
            {
                "check": "data_freshness",
                "status": "passed" if freshness.get("passed") else "failed",
                "details": f"missing_intervals={freshness.get('total_missing_interval_count')}",
            }
        )

    feature_drift = summary.get("feature_drift")
    if feature_drift:
        rows.append(
            {
                "check": "feature_drift",
                "status": "available",
                "details": str(feature_drift.get("severity_counts")),
            }
        )

    prediction_drift = summary.get("prediction_drift")
    if prediction_drift:
        rows.append(
            {
                "check": "prediction_drift",
                "status": prediction_drift.get("severity", "unknown"),
                "details": f"psi={prediction_drift.get('psi')}",
            }
        )

    return pd.DataFrame(rows)
