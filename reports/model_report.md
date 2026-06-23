# Model Report

## Summary

Stage 1 target: compare high-volatility classifiers using chronological and walk-forward evaluation.

## Candidate Models

- Majority class baseline
- Logistic Regression
- HistGradientBoostingClassifier or LightGBM/XGBoost
- Calibrated classifier

## Metrics

Primary metrics: PR-AUC, ROC-AUC, balanced accuracy, precision, recall, F1, Brier score, precision@top-k, and confusion matrix.
