import pandas as pd

from crypto_regime.labels.binary import (
    add_binary_high_vol_label,
    add_future_label_values,
    check_future_rv_alignment,
    compute_train_only_threshold,
    make_labeled_model_table,
    make_threshold_train_mask,
)
from crypto_regime.labels.multiclass import make_multiclass_labels


def make_label_frame(n_rows: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["BTCUSDT"] * n_rows,
            "timestamp": pd.date_range(
                "2025-01-01",
                periods=n_rows,
                freq="5min",
                tz="UTC",
            ),
            "close": [100.0 + i for i in range(n_rows)],
            "feature_example": [float(i) for i in range(n_rows)],
        }
    )


def test_add_future_label_values_adds_return_and_realized_volatility():
    result = add_future_label_values(make_label_frame(), horizon=3)

    assert "future_return_3" in result.columns
    assert "future_rv_3" in result.columns
    assert result["future_return_3"].notna().sum() > 0
    assert result["future_rv_3"].notna().sum() > 0


def test_compute_train_only_threshold_uses_masked_rows():
    labeled = add_future_label_values(make_label_frame(), horizon=2)
    mask = make_threshold_train_mask(labeled, horizon=2, train_fraction=0.5)
    threshold = compute_train_only_threshold(
        labeled,
        horizon=2,
        train_mask=mask,
        quantile=0.8,
    )

    assert threshold == labeled.loc[mask, "future_rv_2"].quantile(0.8)


def test_add_binary_high_vol_label_adds_nullable_target():
    labeled = add_future_label_values(make_label_frame(), horizon=2)
    mask = make_threshold_train_mask(labeled, horizon=2, train_fraction=0.5)
    threshold = compute_train_only_threshold(labeled, horizon=2, train_mask=mask, quantile=0.8)
    result = add_binary_high_vol_label(
        labeled,
        horizon=2,
        volatility_threshold=threshold,
        threshold_train_mask=mask,
    )

    assert "y_high_vol_2" in result.columns
    assert "used_for_label_threshold" in result.columns
    assert str(result["y_high_vol_2"].dtype) == "Int64"


def test_make_labeled_model_table_drops_missing_future_rows():
    labeled = add_future_label_values(make_label_frame(), horizon=2)
    mask = make_threshold_train_mask(labeled, horizon=2, train_fraction=0.5)
    threshold = compute_train_only_threshold(labeled, horizon=2, train_mask=mask, quantile=0.8)
    with_target = add_binary_high_vol_label(
        labeled,
        horizon=2,
        volatility_threshold=threshold,
        threshold_train_mask=mask,
    )
    table = make_labeled_model_table(with_target, horizon=2, drop_missing_labels=True)

    assert len(table) == len(with_target) - 2
    assert table["y_high_vol_2"].notna().all()


def test_future_rv_alignment_check_passes():
    labeled = add_future_label_values(make_label_frame(), horizon=3)

    check_future_rv_alignment(labeled, horizon=3, row_position=4)


def test_make_multiclass_labels_adds_target_column():
    frame = pd.DataFrame({"close": [100, 102, 101]})
    result = make_multiclass_labels(frame, horizon=1)

    assert "regime_multiclass" in result.columns
