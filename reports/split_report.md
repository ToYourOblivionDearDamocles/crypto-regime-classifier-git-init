# Split Report

## Context

- Symbol: `BTCUSDT`
- Interval: `5m`
- Labeled input path: `data/interim/BTCUSDT_5m_2025-01-01_2025-02-01_labeled_binary.parquet`
- Split output path: `data/interim/BTCUSDT_5m_2025-01-01_2025-02-01_labeled_binary_splits.parquet`

## Split Configuration

- Train fraction: `0.7`
- Validation fraction: `0.15`
- Test fraction: `0.15`
- Horizon candles: `12`
- Purge gap candles: `12`
- Split-safe volatility threshold: `0.006680095405165883`
- Preferred target column: `y_high_vol_12_split_safe`

## Split Summary

```text
                 split  num_rows           start_timestamp             end_timestamp  labeled_rows  positive_count  negative_count  positive_rate
                 train      6027 2025-01-02 00:00:00+00:00 2025-01-22 22:10:00+00:00          6027            1206            4821       0.200100
purge_train_validation        12 2025-01-22 22:15:00+00:00 2025-01-22 23:10:00+00:00            12               0              12       0.000000
            validation      1282 2025-01-22 23:15:00+00:00 2025-01-27 10:00:00+00:00          1282             232            1050       0.180967
 purge_validation_test        12 2025-01-27 10:05:00+00:00 2025-01-27 11:00:00+00:00            12               9               3       0.750000
                  test      1295 2025-01-27 11:05:00+00:00 2025-01-31 22:55:00+00:00          1295             230            1065       0.177606
```

## Original Label vs Split-Safe Label

```text
     split  rows_compared  num_changed  changed_rate
      test           1295            2      0.001544
     train           6027            2      0.000332
validation           1282            1       0.00078
```

## Order and Overlap Checks

```text
 symbol                  check  passed                                                                                                                                       message
BTCUSDT non_empty_model_splits    True                                                                                                        train=6027, validation=1282, test=1295
BTCUSDT    chronological_order    True train_max=2025-01-22 22:10:00+00:00, val_min=2025-01-22 23:15:00+00:00, val_max=2025-01-27 10:00:00+00:00, test_min=2025-01-27 11:05:00+00:00
BTCUSDT   no_timestamp_overlap    True                                                                                                                     timestamp_overlap_count=0
```

## Purge Gap Checks

```text
 symbol                                check  passed                                                                               message
BTCUSDT train_label_window_before_validation    True train_label_end=2025-01-22 23:10:00+00:00, validation_start=2025-01-22 23:15:00+00:00
BTCUSDT  validation_label_window_before_test    True  validation_label_end=2025-01-27 11:00:00+00:00, test_start=2025-01-27 11:05:00+00:00
```

## Walk-Forward Fold Preview

```text
 fold_id  symbol               train_start                 train_end          validation_start            validation_end  train_rows  validation_rows  purge_gap_candles
       0 BTCUSDT 2025-01-02 00:00:00+00:00 2025-01-16 23:25:00+00:00 2025-01-17 00:30:00+00:00 2025-01-20 00:15:00+00:00        4314              862                 12
       1 BTCUSDT 2025-01-02 00:00:00+00:00 2025-01-19 23:15:00+00:00 2025-01-20 00:20:00+00:00 2025-01-23 00:05:00+00:00        5176              862                 12
       2 BTCUSDT 2025-01-02 00:00:00+00:00 2025-01-22 23:05:00+00:00 2025-01-23 00:10:00+00:00 2025-01-25 23:55:00+00:00        6038              862                 12
       3 BTCUSDT 2025-01-02 00:00:00+00:00 2025-01-25 22:55:00+00:00 2025-01-26 00:00:00+00:00 2025-01-28 23:45:00+00:00        6900              862                 12
```

## Interpretation

Random splitting is not used. Splits are chronological because financial time series are non-iid and future market regimes must not influence earlier training.

Rows near train/validation and validation/test boundaries are purged because the label uses future candles. Purged rows are excluded from model fitting and model evaluation.

From this point onward, modeling should use the split-safe target column, because its threshold is computed from split == train only.
