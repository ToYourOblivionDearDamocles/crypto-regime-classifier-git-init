# Crypto Market Regime Classifier

End-to-end ML production pipeline for crypto volatility and market-regime classification using public OHLCV data.

The project is organized as a production-style workflow:

- Binance BTCUSDT OHLCV ingestion, validation, and resampling
- Leakage-safe feature engineering for 5-minute market candles
- Binary high-volatility labeling and later multiclass regime labeling
- Time-aware model splitting and walk-forward validation
- Model training, evaluation, calibration, and registry utilities
- FastAPI serving, Streamlit exploration, and monitoring reports

## Project Milestones

### Milestone 1 — Binary High-Volatility Classifier

Predict whether BTCUSDT enters a high-volatility regime over the next 60 minutes using only information available up to the prediction timestamp.

### Milestone 2 — Multiclass Market Regime Classifier

Extend the target into four future regimes: normal, upside breakout, downside turbulence, and high-volatility uncertain.

## Project Layout

```text
crypto-regime-classifier/
  configs/           YAML configuration files
  data/              Raw, interim, and processed datasets
  models/            Saved model artifacts
  reports/           Project reports and figures
  src/crypto_regime/ Python package source code
  app/               Streamlit application
  tests/             Unit and integration tests
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Pipeline Step 1 — Data Ingestion

This step downloads BTCUSDT 5-minute OHLCV candles, stores the raw Binance-format data, converts it into the project canonical schema, normalizes timestamps to UTC, and saves the processed candles as Parquet.

Run:

```bash
python -m crypto_regime.data.download --config configs/data.yaml
```

## Teaching Pipeline 2 — Data Validation

This pipeline validates the processed BTCUSDT candles before feature engineering and model training. It checks schema, UTC timestamps, sorted and unique candles, missing intervals, duplicate rows, OHLCV consistency, null values, and extreme one-candle returns.

Run:

```bash
python -m crypto_regime.data.validate --config configs/data.yaml
```

## Teaching Pipeline 3 — Feature Engineering

This pipeline builds backward-looking OHLCV features for BTCUSDT 5-minute candles. It creates candle-shape, return, volume, realized-volatility, drawdown, and rolling-window features while checking that feature values do not depend on future rows.

Run:

```bash
python -m crypto_regime.features.build_features --config configs/features.yaml
```

## Teaching Pipeline 5 — Splitting and Validation

This pipeline creates chronological train/validation/test splits with purge gaps around split boundaries.

Run:

```bash
python -m crypto_regime.splitting.time_split --config configs/splitting.yaml
```

## Teaching Pipeline 6 — Modeling

This pipeline trains the first binary classification models using the split-safe target from Teaching pipeline 5.

Run:

```bash
python -m crypto_regime.models.train --config configs/model.yaml
```

## Teaching Pipeline 7 — Evaluation

This pipeline evaluates trained model artifacts using task-aware metrics.

Run:

```bash
python -m crypto_regime.evaluation.evaluate --config configs/evaluation.yaml
```

## Teaching Pipeline 8 — Serving and Inference

This pipeline serves trained model artifacts through FastAPI and supports offline batch inference.

Run the API:

```bash
uvicorn crypto_regime.api.main:app --reload
```

## Teaching Pipeline 9 — Monitoring

This pipeline monitors data freshness, feature drift, and prediction drift.

Run:

```bash
python -m crypto_regime.monitoring.run_monitoring --config configs/monitoring.yaml
```

## Teaching Pipeline 10 — Dashboard and Communication

This pipeline builds a Streamlit dashboard for communicating the ML system.

Run:

```bash
streamlit run app/streamlit_app.py
```

## Current Build

The current build is Milestone 1. The pipeline starts with data ingestion, then adds validation, feature engineering, leakage-safe binary labels, time-based evaluation, calibration, FastAPI serving, Docker packaging, and Streamlit dashboards.
