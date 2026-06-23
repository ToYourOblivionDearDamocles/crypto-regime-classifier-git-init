import numpy as np
import pandas as pd
import pytest

from crypto_regime.splitting.time_split import (
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    add_split_safe_binary_label,
    assign_chronological_splits_with_purge,
    compute_split_train_threshold,
    compare_existing_and_split_safe_labels,
    interval_to_timedelta,
    make_walk_forward_folds,
    summarize_splits,
    validate_purge_gap,
    validate_split_order_and_overlap,
)


def make_labeled_table(n: int = 160, horizon: int = 5) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2025-01-01 00:00:00",
        periods=n,
        freq="5min",
        tz="UTC",
    )

    x = np.arange(n)

    close = 100.0 + 0.1 * x + np.sin(x / 10.0)

    # Deterministic future_rv-like values.
    future_rv = 0.01 + 0.005 * (np.sin(x / 7.0) + 1.0)

    label = (future_rv > np.quantile(future_rv[: int(0.7 * n)], 0.8)).astype(int)

    return pd.DataFrame(
        {
            "symbol": "BTCUSDT",
            "timestamp": timestamps,
            "close": close,
            f"future_rv_{horizon}": future_rv,
            f"y_high_vol_{horizon}": pd.Series(label, dtype="Int64"),
            "feature_a": np.linspace(0.0, 1.0, n),
        }
    )


def test_interval_to_timedelta():
    assert interval_to_timedelta("5m") == pd.Timedelta(minutes=5)
    assert interval_to_timedelta("1h") == pd.Timedelta(hours=1)
    assert interval_to_timedelta("1d") == pd.Timedelta(days=1)


def test_assign_chronological_splits_with_purge_creates_expected_splits():
    df = make_labeled_table(n=160, horizon=5)

    split_df = assign_chronological_splits_with_purge(
        df,
        train_fraction=0.70,
        validation_fraction=0.15,
        test_fraction=0.15,
        purge_gap_candles=5,
    )

    assert "split" in split_df.columns
    assert set(split_df["split"].unique()) == {
        "train",
        "purge_train_validation",
        "validation",
        "purge_validation_test",
        "test",
    }

    assert len(split_df[split_df["split"] == "train"]) > 0
    assert len(split_df[split_df["split"] == "validation"]) > 0
    assert len(split_df[split_df["split"] == "test"]) > 0


def test_compute_split_train_threshold_uses_train_only():
    horizon = 5
    df = make_labeled_table(n=160, horizon=horizon)

    split_df = assign_chronological_splits_with_purge(
        df,
        train_fraction=0.70,
        validation_fraction=0.15,
        test_fraction=0.15,
        purge_gap_candles=5,
    )

    threshold = compute_split_train_threshold(
        split_df,
        horizon=horizon,
        quantile=0.8,
    )

    expected = split_df.loc[
        split_df["split"] == "train",
        f"future_rv_{horizon}",
    ].quantile(0.8)

    assert threshold == expected


def test_add_split_safe_binary_label():
    horizon = 5
    df = make_labeled_table(n=160, horizon=horizon)

    split_df = assign_chronological_splits_with_purge(
        df,
        train_fraction=0.70,
        validation_fraction=0.15,
        test_fraction=0.15,
        purge_gap_candles=5,
    )

    threshold = compute_split_train_threshold(
        split_df,
        horizon=horizon,
        quantile=0.8,
    )

    labeled = add_split_safe_binary_label(
        split_df,
        horizon=horizon,
        threshold=threshold,
    )

    target_col = f"y_high_vol_{horizon}_split_safe"

    assert target_col in labeled.columns
    assert str(labeled[target_col].dtype) == "Int64"
    assert labeled[target_col].isna().sum() == 0


def test_validate_split_order_and_overlap_passes():
    horizon = 5
    df = make_labeled_table(n=160, horizon=horizon)

    split_df = assign_chronological_splits_with_purge(
        df,
        train_fraction=0.70,
        validation_fraction=0.15,
        test_fraction=0.15,
        purge_gap_candles=horizon,
    )

    issues = validate_split_order_and_overlap(split_df)

    assert all(issue["passed"] for issue in issues)


def test_validate_purge_gap_passes_with_sufficient_purge():
    horizon = 5
    df = make_labeled_table(n=160, horizon=horizon)

    split_df = assign_chronological_splits_with_purge(
        df,
        train_fraction=0.70,
        validation_fraction=0.15,
        test_fraction=0.15,
        purge_gap_candles=horizon,
    )

    issues = validate_purge_gap(
        split_df,
        horizon=horizon,
        interval_delta=pd.Timedelta(minutes=5),
    )

    assert all(issue["passed"] for issue in issues)


def test_validate_purge_gap_fails_without_purge():
    horizon = 5
    df = make_labeled_table(n=160, horizon=horizon)

    split_df = assign_chronological_splits_with_purge(
        df,
        train_fraction=0.70,
        validation_fraction=0.15,
        test_fraction=0.15,
        purge_gap_candles=0,
    )

    issues = validate_purge_gap(
        split_df,
        horizon=horizon,
        interval_delta=pd.Timedelta(minutes=5),
    )

    assert any(not issue["passed"] for issue in issues)


def test_compare_existing_and_split_safe_labels():
    horizon = 5
    df = make_labeled_table(n=160, horizon=horizon)

    split_df = assign_chronological_splits_with_purge(
        df,
        train_fraction=0.70,
        validation_fraction=0.15,
        test_fraction=0.15,
        purge_gap_candles=horizon,
    )

    threshold = compute_split_train_threshold(
        split_df,
        horizon=horizon,
        quantile=0.8,
    )

    split_df = add_split_safe_binary_label(
        split_df,
        horizon=horizon,
        threshold=threshold,
    )

    comparison = compare_existing_and_split_safe_labels(
        split_df,
        horizon=horizon,
    )

    assert set(comparison.columns) == {
        "split",
        "rows_compared",
        "num_changed",
        "changed_rate",
    }


def test_summarize_splits():
    horizon = 5
    df = make_labeled_table(n=160, horizon=horizon)

    split_df = assign_chronological_splits_with_purge(
        df,
        train_fraction=0.70,
        validation_fraction=0.15,
        test_fraction=0.15,
        purge_gap_candles=horizon,
    )

    threshold = compute_split_train_threshold(
        split_df,
        horizon=horizon,
        quantile=0.8,
    )

    split_df = add_split_safe_binary_label(
        split_df,
        horizon=horizon,
        threshold=threshold,
    )

    summary = summarize_splits(split_df, horizon=horizon)

    assert "split" in summary.columns
    assert "positive_rate" in summary.columns
    assert set(["train", "validation", "test"]).issubset(set(summary["split"]))


def test_make_walk_forward_folds():
    df = make_labeled_table(n=160, horizon=5)

    folds = make_walk_forward_folds(
        df,
        min_train_fraction=0.50,
        validation_window_fraction=0.10,
        step_fraction=0.10,
        purge_gap_candles=5,
    )

    assert not folds.empty
    assert "train_start" in folds.columns
    assert "validation_start" in folds.columns


def test_bad_split_fraction_raises():
    df = make_labeled_table(n=160, horizon=5)

    with pytest.raises(ValueError):
        assign_chronological_splits_with_purge(
            df,
            train_fraction=0.60,
            validation_fraction=0.20,
            test_fraction=0.30,
            purge_gap_candles=5,
        )
