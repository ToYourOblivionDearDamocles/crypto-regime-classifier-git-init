# Scope and Variants

## 1. Current Scope

This project currently focuses on the first implementation target:

> Binary classification for future crypto high-volatility detection.

The system uses public OHLCV candle data and predicts whether the market will enter a high-volatility regime over the next 60 minutes.

For 5-minute candles, the prediction horizon is:

```text
60 minutes = 12 future candles
```

The current binary label is defined by future realized volatility:

```text
r_t = log(close_t) - log(close_{t-1})

future_rv_t = sqrt(sum_{i=1}^{12} r_{t+i}^2)

y_t = 1 if future_rv_t > threshold
y_t = 0 otherwise
```

The default threshold is the 80th percentile of `future_rv_t` computed from the chronological training period only.

This means the positive class represents approximately the top 20% highest future-volatility windows in the training period.

## 2. Why Binary Classification First?

The first project milestone is intentionally framed as a risk-alert system rather than a price-prediction system.

The product question is:

> Will the next hour be high-volatility enough to flag?

This is naturally a binary classification task.

Binary classification is useful for the first version because it gives a clear HR-visible product story:

> The model watches recent market data and flags future high-volatility periods.

This is easier to explain and visualize than a raw regression output. It also supports later system components such as probability calibration, alert thresholds, API serving, monitoring, and dashboards.

## 3. Why the 80th Percentile Threshold?

The 80th percentile threshold is an operational event definition, not a theoretically optimal constant.

It means:

> Treat the top quintile of future realized-volatility windows as high-volatility events.

This threshold is useful for a first version because it creates a positive class that is uncommon but not too rare. A 95th-percentile threshold would define a more extreme event, but it would also make the positive class much smaller and harder to model. A 60th-percentile threshold would create a more balanced dataset, but the event would no longer clearly mean high volatility.

The 80th percentile should therefore be understood as a configurable design choice.

It should not be described as an objective law of the market.

## 4. Leakage Rule

The volatility threshold must be computed from the chronological training period only.

Incorrect:

```text
threshold = 80th percentile of future_rv over the full dataset
```

Correct:

```text
threshold = 80th percentile of future_rv over the training period only
```

Using the full dataset would leak validation/test distribution information into label construction.

This is especially dangerous in financial time series because future market regimes may differ from earlier regimes.

## 5. Two Different Thresholds

This project has two different thresholds.

### 5.1 Label Threshold

The label threshold defines the event:

```text
future_rv_t > train_period_80th_percentile(future_rv)
```

This answers:

> What do we mean by high volatility?

### 5.2 Prediction Decision Threshold

The prediction threshold converts model probabilities into hard alerts:

```text
P(y = 1 | features_t) > tau
```

This answers:

> When should the system raise an alert?

These thresholds serve different purposes. The label threshold defines the target. The prediction threshold defines the operating policy.

The prediction threshold should be tuned later using validation data and the intended product tradeoff between false positives and false negatives.

## 6. Class Imbalance

The default 80th-percentile label creates an imbalanced dataset:

```text
positive class ≈ 20%
negative class ≈ 80%
```

This is expected because high-volatility periods should be less common than normal periods.

However, this also means accuracy alone is not a reliable metric. A naive model that always predicts the negative class can appear strong under accuracy.

The project should emphasize metrics such as:

```text
PR-AUC
ROC-AUC
balanced accuracy
precision
recall
F1
Brier score
calibration curve
precision@top-k
```

For the binary risk-alert system, PR-AUC, recall, calibration, and precision@top-k are especially important.

## 7. Why Not Regression First?

A regression formulation is mathematically natural.

Instead of predicting a binary event, a regression model would predict:

```text
future_rv_12
```

That avoids throwing away information. Binary classification compresses all future volatility values into two classes:

```text
normal
high volatility
```

This loses severity information. A window barely above the 80th percentile and a window above the 99th percentile both receive label `1`.

However, binary classification is still appropriate for the first version because the product is an alerting system. The first milestone is not to estimate the exact magnitude of future volatility; it is to build a clean end-to-end ML engineering pipeline for a risk signal.

The project should still keep the continuous `future_rv_12` column as metadata so regression variants can be added later.

## 8. Current Modeling Decision

The current implementation decision is:

```text
Primary first version:
    binary classification

Target:
    y_high_vol_12

Event definition:
    future_rv_12 > train-period q80(future_rv_12)

Prediction output:
    calibrated probability of high volatility

Main product interpretation:
    probability that the next hour enters a high-volatility regime
```

This is the active scope.

The project should not yet implement regression, quantile regression, or multi-class classification until the binary pipeline is complete.

## 9. Variants for Later

The following variants are intentionally postponed.

### 9.1 Regression Variant

Target:

```text
future_rv_12
```

Goal:

> Predict the magnitude of future realized volatility directly.

Possible metrics:

```text
MAE
RMSE
R²
Spearman rank correlation
top-k realized volatility capture
```

This variant is more mathematically direct but may be harder to explain as a simple alerting product.

### 9.2 Quantile Regression Variant

Targets:

```text
Q_0.50(future_rv_12 | features)
Q_0.80(future_rv_12 | features)
Q_0.90(future_rv_12 | features)
Q_0.95(future_rv_12 | features)
```

Goal:

> Estimate conditional risk quantiles of future volatility.

This is highly relevant for risk modeling because it asks how bad volatility could become under current conditions.

This should be considered after the binary classifier is working.

### 9.3 Threshold Sensitivity Variant

Instead of using only the 80th percentile, compare multiple label definitions:

```text
q70
q80
q90
q95
```

For each threshold, report:

```text
positive class rate
PR-AUC
ROC-AUC
recall
precision
precision@top-k
calibration
```

This tests whether the model is robust to the chosen event definition.

This is a strong later improvement because it directly addresses the arbitrariness of the 80th-percentile threshold.

### 9.4 Ranking Variant

Instead of focusing on hard classification, evaluate whether the highest model scores correspond to the highest-risk periods.

Useful metrics:

```text
precision@top_5_percent
precision@top_10_percent
mean future_rv in top-scored windows
lift over baseline
```

This is appropriate for alert systems where only a limited number of high-risk warnings can be reviewed.

### 9.5 Multi-Class Variant

The later multi-class version may classify the future regime into:

```text
normal
upside breakout
downside turbulence
high-volatility uncertain
```

This should not be started until the binary classification pipeline has data ingestion, validation, feature engineering, label generation, time-series splitting, training, evaluation, calibration, API serving, dashboard, tests, and documentation.

## 10. Non-Goals

This project is not currently trying to build:

```text
a profitable trading strategy
a trading execution engine
a PnL backtest
a reinforcement learning trading agent
a market-making system
a price-direction prediction system
```

The correct claim is:

> This project demonstrates an end-to-end machine learning engineering workflow for a financial time-series risk classification system.

## 11. How to Explain This in an Interview

A concise explanation:

> I framed the first version as a binary risk-alert system. The label is whether future realized volatility over the next 60 minutes exceeds the 80th percentile of training-period future volatility. The 80th percentile is not theoretically sacred; it is a configurable operational event definition. I keep the continuous future volatility target so that regression, quantile regression, and threshold-sensitivity variants can be added later. I also separate the event-definition threshold from the model decision threshold, which is tuned later on validation data.

## 12. Current Implementation Priority

Focus only on the binary classification path first:

```text
1. data ingestion
2. data validation
3. feature engineering
4. binary label generation
5. chronological train/validation/test split
6. baseline model
7. logistic regression
8. gradient boosting
9. evaluation
10. calibration
11. artifact saving
12. API and dashboard
```

Only after this is complete should the project add regression, quantile regression, threshold sensitivity, or multi-class classification.
