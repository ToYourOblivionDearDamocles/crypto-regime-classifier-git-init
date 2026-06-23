# Teaching Guide

## Learning Objectives

By the end of this project, a reviewer should be able to see how the system:

- Turns raw crypto market data into validated datasets
- Builds time-series features without future leakage
- Defines binary and multiclass market regime labels
- Evaluates models with time-aware splits
- Serves predictions through an API
- Documents model risks and monitoring requirements

## Suggested Build Order

1. Implement data loading and validation.
2. Add resampling and feature generation.
3. Implement label construction.
4. Add chronological and walk-forward splits.
5. Train a baseline model.
6. Evaluate metrics and calibration.
7. Add API and Streamlit demo.
8. Write reports, model card, and data card.

## Interview Talking Points

- Why random splits are risky for time-series data
- How regime labels can encode business assumptions
- Why calibration matters when probabilities are consumed by users
- How monitoring differs for data freshness, feature drift, and prediction drift
- Which shortcuts were avoided to keep the project honest
