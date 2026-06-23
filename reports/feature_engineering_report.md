# Feature Engineering Report

## Context

- Symbol: `BTCUSDT`
- Interval: `5m`
- Input path: `data/processed/BTCUSDT_5m_2025-01-01_2025-02-01_processed.parquet`
- Output path: `data/interim/BTCUSDT_5m_2025-01-01_2025-02-01_features.parquet`

## Summary

- Feature rows: `8640`
- Number of engineered features: `64`
- Rolling windows: `[3, 6, 12, 48, 288]`
- Min periods policy: `full_window`

## Feature Columns

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

## Missingness Audit

```text
                        feature  missing_count  missing_pct          interpretation
    rolling_abs_return_mean_288            288     0.032258 expected_rolling_warmup
       rolling_realized_vol_288            288     0.032258 expected_rolling_warmup
        rolling_return_mean_288            288     0.032258 expected_rolling_warmup
         rolling_return_std_288            288     0.032258 expected_rolling_warmup
 distance_from_rolling_high_288            287     0.032146 expected_rolling_warmup
  distance_from_rolling_low_288            287     0.032146 expected_rolling_warmup
           rolling_drawdown_288            287     0.032146 expected_rolling_warmup
rolling_high_low_range_mean_288            287     0.032146 expected_rolling_warmup
        rolling_volume_mean_288            287     0.032146 expected_rolling_warmup
         rolling_volume_std_288            287     0.032146 expected_rolling_warmup
              volume_zscore_288            287     0.032146 expected_rolling_warmup
     rolling_abs_return_mean_48             48     0.005376 expected_rolling_warmup
        rolling_realized_vol_48             48     0.005376 expected_rolling_warmup
         rolling_return_mean_48             48     0.005376 expected_rolling_warmup
          rolling_return_std_48             48     0.005376 expected_rolling_warmup
  distance_from_rolling_high_48             47     0.005264 expected_rolling_warmup
   distance_from_rolling_low_48             47     0.005264 expected_rolling_warmup
            rolling_drawdown_48             47     0.005264 expected_rolling_warmup
 rolling_high_low_range_mean_48             47     0.005264 expected_rolling_warmup
         rolling_volume_mean_48             47     0.005264 expected_rolling_warmup
          rolling_volume_std_48             47     0.005264 expected_rolling_warmup
               volume_zscore_48             47     0.005264 expected_rolling_warmup
     rolling_abs_return_mean_12             12     0.001344 expected_rolling_warmup
        rolling_realized_vol_12             12     0.001344 expected_rolling_warmup
         rolling_return_mean_12             12     0.001344 expected_rolling_warmup
          rolling_return_std_12             12     0.001344 expected_rolling_warmup
  distance_from_rolling_high_12             11     0.001232 expected_rolling_warmup
   distance_from_rolling_low_12             11     0.001232 expected_rolling_warmup
            rolling_drawdown_12             11     0.001232 expected_rolling_warmup
 rolling_high_low_range_mean_12             11     0.001232 expected_rolling_warmup
         rolling_volume_mean_12             11     0.001232 expected_rolling_warmup
          rolling_volume_std_12             11     0.001232 expected_rolling_warmup
               volume_zscore_12             11     0.001232 expected_rolling_warmup
      rolling_abs_return_mean_6              6     0.000672 expected_rolling_warmup
         rolling_realized_vol_6              6     0.000672 expected_rolling_warmup
          rolling_return_mean_6              6     0.000672 expected_rolling_warmup
           rolling_return_std_6              6     0.000672 expected_rolling_warmup
   distance_from_rolling_high_6              5     0.000560 expected_rolling_warmup
    distance_from_rolling_low_6              5     0.000560 expected_rolling_warmup
             rolling_drawdown_6              5     0.000560 expected_rolling_warmup
  rolling_high_low_range_mean_6              5     0.000560 expected_rolling_warmup
          rolling_volume_mean_6              5     0.000560 expected_rolling_warmup
           rolling_volume_std_6              5     0.000560 expected_rolling_warmup
                volume_zscore_6              5     0.000560 expected_rolling_warmup
      rolling_abs_return_mean_3              3     0.000336 expected_rolling_warmup
         rolling_realized_vol_3              3     0.000336 expected_rolling_warmup
          rolling_return_mean_3              3     0.000336 expected_rolling_warmup
           rolling_return_std_3              3     0.000336 expected_rolling_warmup
   distance_from_rolling_high_3              2     0.000224 expected_rolling_warmup
    distance_from_rolling_low_3              2     0.000224 expected_rolling_warmup
             rolling_drawdown_3              2     0.000224 expected_rolling_warmup
  rolling_high_low_range_mean_3              2     0.000224 expected_rolling_warmup
          rolling_volume_mean_3              2     0.000224 expected_rolling_warmup
           rolling_volume_std_3              2     0.000224 expected_rolling_warmup
                volume_zscore_3              2     0.000224 expected_rolling_warmup
                   log_return_1              1     0.000112 expected_rolling_warmup
              num_trades_change              1     0.000112 expected_rolling_warmup
            quote_volume_change              1     0.000112 expected_rolling_warmup
                  volume_change              1     0.000112 expected_rolling_warmup
                candle_body_pct              0     0.000000       no_missing_values
             high_low_range_pct              0     0.000000       no_missing_values
                 lower_wick_pct              0     0.000000       no_missing_values
              open_close_return              0     0.000000       no_missing_values
                 upper_wick_pct              0     0.000000       no_missing_values
```

## Suspicious Missingness

_None._

## Interpretation

Rolling-window features naturally create warmup NaNs at the beginning of the dataset. Those rows are dropped when `drop_missing_features` is true.

This pipeline creates only backward-looking features. It does not create future returns, future volatility, labels, train/test splits, or models.
