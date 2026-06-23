# Binary Label Generation Report

## Context

- Symbol: `BTCUSDT`
- Interval: `5m`
- Feature input path: `data/interim/BTCUSDT_5m_2025-01-01_2025-02-01_features.parquet`
- Labeled output path: `data/interim/BTCUSDT_5m_2025-01-01_2025-02-01_labeled_binary.parquet`

## Active Binary Label

- Horizon candles: `12`
- Horizon minutes: `60`
- Volatility quantile: `0.8`
- Train-only volatility threshold: `0.006677282643354623`

```text
r_t = log(close_t) - log(close_{t-1})
future_rv_t = sqrt(sum_{i=1}^{horizon} r_{t+i}^2)
y_t = 1 if future_rv_t > threshold
y_t = 0 otherwise
```

## Leakage Rule

The volatility threshold is computed only from the chronological threshold-training portion, not from the full dataset.

## Label Summary

```json
{
  "horizon": 12,
  "total_rows_before_label_drop": 8640,
  "valid_labeled_rows": 8628,
  "missing_label_rows": 12,
  "volatility_threshold": 0.006677282643354623,
  "overall_class_counts": {
    "0": 6946,
    "1": 1682
  },
  "overall_class_rates": {
    "0": 0.8050533147890588,
    "1": 0.19494668521094113
  },
  "threshold_train_class_counts": {
    "0": 4831,
    "1": 1208
  },
  "threshold_train_class_rates": {
    "0": 0.799966881934095,
    "1": 0.20003311806590496
  },
  "future_rv_min": 0.0007673065738977039,
  "future_rv_median": 0.003860900649108866,
  "future_rv_mean": 0.004871017688529035,
  "future_rv_max": 0.047084491037938984
}
```

## Missing Future Label Audit

```text
 symbol  rows  missing_future_return  missing_future_rv  expected_missing_approximately
BTCUSDT  8640                     12                 12                              12
```

## Quantile Sensitivity

This table documents how class balance would change under alternative threshold quantiles. The active implementation still uses the configured binary threshold above.

```text
 quantile  threshold  train_positive_rate  overall_positive_rate
     0.70   0.005456             0.300050               0.296708
     0.80   0.006677             0.200033               0.194947
     0.90   0.008686             0.100017               0.094576
     0.95   0.011213             0.050008               0.046940
```

## Interpretation

A positive label at timestamp t means the future horizon after t has high realized volatility. It does not mean the current candle itself is high-volatility.

The 80th percentile threshold is an operational event definition, not a theoretically sacred constant. The continuous future_rv column is kept so regression and quantile-regression variants can be added later.

This pipeline creates labels only. It does not train or evaluate a model.
