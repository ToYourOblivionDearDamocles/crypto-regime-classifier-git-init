# Model Training Report

## Context

- Task type: `binary_classification`
- Model version: `crypto_binary_v1`
- Split input path: `data/interim/BTCUSDT_5m_2025-01-01_2025-02-01_labeled_binary_splits.parquet`
- Target column: `y_high_vol_12_split_safe`
- Model artifact directory: `models/saved/crypto_binary_v1`

## Models Trained

```text
            model_name backend                         estimator  accuracy  balanced_accuracy  precision   recall       f1  roc_auc   pr_auc  brier_score
     majority_baseline sklearn                  dummy_classifier  0.819033           0.500000   0.000000 0.000000 0.000000 0.500000 0.180967     0.180967
   logistic_regression sklearn               logistic_regression  0.843994           0.797307   0.552632 0.724138 0.626866 0.884778 0.656842     0.122743
hist_gradient_boosting sklearn hist_gradient_boosting_classifier  0.846334           0.714787   0.587065 0.508621 0.545035 0.831634 0.586141     0.119601
```

## Feature Columns

- Number of features: `64`

```text
log_return_1
open_close_return
high_low_range_pct
candle_body_pct
upper_wick_pct
lower_wick_pct
volume_change
quote_volume_change
num_trades_change
rolling_return_mean_3
rolling_return_std_3
rolling_abs_return_mean_3
rolling_realized_vol_3
rolling_volume_mean_3
rolling_volume_std_3
volume_zscore_3
rolling_high_low_range_mean_3
rolling_drawdown_3
distance_from_rolling_high_3
distance_from_rolling_low_3
rolling_return_mean_6
rolling_return_std_6
rolling_abs_return_mean_6
rolling_realized_vol_6
rolling_volume_mean_6
rolling_volume_std_6
volume_zscore_6
rolling_high_low_range_mean_6
rolling_drawdown_6
distance_from_rolling_high_6
distance_from_rolling_low_6
rolling_return_mean_12
rolling_return_std_12
rolling_abs_return_mean_12
rolling_realized_vol_12
rolling_volume_mean_12
rolling_volume_std_12
volume_zscore_12
rolling_high_low_range_mean_12
rolling_drawdown_12
distance_from_rolling_high_12
distance_from_rolling_low_12
rolling_return_mean_48
rolling_return_std_48
rolling_abs_return_mean_48
rolling_realized_vol_48
rolling_volume_mean_48
rolling_volume_std_48
volume_zscore_48
rolling_high_low_range_mean_48
rolling_drawdown_48
distance_from_rolling_high_48
distance_from_rolling_low_48
rolling_return_mean_288
rolling_return_std_288
rolling_abs_return_mean_288
rolling_realized_vol_288
rolling_volume_mean_288
rolling_volume_std_288
volume_zscore_288
rolling_high_low_range_mean_288
rolling_drawdown_288
distance_from_rolling_high_288
distance_from_rolling_low_288
```

## Interpretation

This pipeline trains baseline and first-pass binary classifiers. It is not the final evaluation pipeline. Full error analysis, threshold tuning, calibration, and final test evaluation belong to later teaching pipelines.

The majority baseline is included intentionally. More complex models must be compared against it to demonstrate actual modeling value.

The modeling interface is adapter-based. New backends such as PyTorch, JAX, LightGBM, or XGBoost can be added by implementing the same adapter interface.
