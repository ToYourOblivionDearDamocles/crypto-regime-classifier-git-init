# Data Quality Report

## Context

- Symbol: `BTCUSDT`
- Interval: `5m`
- Processed path: `data/processed/BTCUSDT_5m_2025-01-01_2025-02-01_processed.parquet`

## Summary

- Total checks: `19`
- Failed checks: `0`
- Failed ERROR checks: `0`
- Failed WARNING checks: `0`
- Can continue to downstream ML pipelines: `True`

## All Checks

```text
                     check_name severity  passed  count                                                           message
         required_columns_exist    ERROR    True      0                                 Missing core required columns: []
       optional_columns_present  WARNING    True      0                            Missing optional canonical columns: []
         extra_columns_detected     INFO    True      0                                         Extra columns present: []
               core_null_values    ERROR    True      0                           Null values in core required columns: 0
      total_null_values_counted     INFO    True      0                           Total null values across all columns: 0
        timestamps_parse_as_utc    ERROR    True      0                                 Unparseable or null timestamps: 0
timestamps_unique_within_symbol    ERROR    True      0                             Duplicate rows by symbol/timestamp: 0
timestamps_sorted_within_symbol    ERROR    True      0                              Symbols with unsorted timestamps: []
   expected_interval_consistent  WARNING    True      0 Expected interval is 0 days 00:05:00. Irregular timestamp gaps: 0
      missing_intervals_counted  WARNING    True      0                                       Missing expected candles: 0
              no_duplicate_rows    ERROR    True      0                                          Fully duplicated rows: 0
           ohlc_prices_positive    ERROR    True      0               Rows where at least one OHLC price is non-positive.
            volume_non_negative    ERROR    True      0                                    Rows where volume is negative.
        high_greater_equal_open    ERROR    True      0                                           Rows where high < open.
       high_greater_equal_close    ERROR    True      0                                          Rows where high < close.
            low_less_equal_open    ERROR    True      0                                            Rows where low > open.
           low_less_equal_close    ERROR    True      0                                           Rows where low > close.
         high_greater_equal_low    ERROR    True      0                                            Rows where high < low.
        extreme_returns_flagged  WARNING    True      0                     Rows with abs(one-candle log return) > 0.1: 0
```

## Failed Checks

_None._

## Null Counts

```text
      column  null_count
      symbol           0
   timestamp           0
        open           0
        high           0
         low           0
       close           0
      volume           0
quote_volume           0
  num_trades           0
```

## Missing Interval Examples

_None._

## Duplicate Row Examples

_None._

## Failed OHLCV Row Examples

_None._

## Extreme Return Examples

_None._

## Interpretation

- `ERROR` checks should block feature engineering, label generation, and model training.
- `WARNING` checks should be inspected, but they do not automatically mean the data is unusable.
- This validation step does not modify or delete data. It only reports quality issues.
