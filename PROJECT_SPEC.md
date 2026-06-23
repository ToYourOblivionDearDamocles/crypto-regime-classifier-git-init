# Project Spec

## Goal

Build a cryptocurrency market regime classifier that predicts whether market conditions are bullish, neutral, bearish, or risk-off using historical OHLCV data and derived features.

## Target Users

- Hiring managers reviewing an end-to-end machine learning project
- Data scientists evaluating time-series modeling choices
- Engineers looking for a clean, reproducible ML service structure

## Core Questions

- Can recent price, volume, volatility, and momentum behavior identify market regimes?
- How does model performance change across walk-forward time splits?
- Are predicted probabilities calibrated enough for downstream decision support?
- What data freshness and drift checks are needed for reliable monitoring?

## Deliverables

- Clean source package under `src/crypto_regime`
- Config-driven pipeline
- Time-aware train, validation, and test splits
- Binary and multiclass labeling strategies
- Model evaluation and calibration reports
- FastAPI prediction endpoint
- Streamlit demo application
- Model card and data card

## Non-Goals

- Live trading execution
- Financial advice
- Exchange account integration
- High-frequency market making

## Success Criteria

- Reproducible local setup
- Meaningful baseline model with documented metrics
- No random train/test leakage across time
- Clear explanation of assumptions, limitations, and monitoring needs
