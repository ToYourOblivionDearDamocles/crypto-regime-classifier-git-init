# Crypto Market Regime Classifier

This is a work-in-progress machine learning project for classifying crypto market regimes using public OHLCV data.

The current focus is a binary classifier for BTCUSDT high-volatility periods. The project is still under development, so some modules, reports, and dashboard views may change as the pipeline improves.

Main components:

- Data ingestion and validation
- Feature engineering
- Binary label generation
- Time-based train/validation/test splitting
- Model training and evaluation
- FastAPI serving
- Streamlit dashboard
- Basic monitoring reports

## Project Milestones

### Milestone 1 — Binary High-Volatility Classifier

Predict whether BTCUSDT enters a high-volatility period over the next 60 minutes.

### Milestone 2 — Multiclass Market Regime Classifier

Planned future extension: classify several market-regime types instead of only high-volatility vs. not high-volatility.

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

Downloads BTCUSDT 5-minute OHLCV candles and saves raw and processed data.

Run:

```bash
python -m crypto_regime.data.download --config configs/data.yaml
```

## Teaching Pipeline 2 — Data Validation

Validates the processed candle data before feature engineering.

Run:

```bash
python -m crypto_regime.data.validate --config configs/data.yaml
```

## Teaching Pipeline 3 — Feature Engineering

Builds backward-looking candle, return, volume, volatility, and rolling-window features.

Run:

```bash
python -m crypto_regime.features.build_features --config configs/features.yaml
```

## Teaching Pipeline 5 — Splitting and Validation

Creates chronological train/validation/test splits.

Run:

```bash
python -m crypto_regime.splitting.time_split --config configs/splitting.yaml
```

## Teaching Pipeline 6 — Modeling

Trains binary classification models.

Run:

```bash
python -m crypto_regime.models.train --config configs/model.yaml
```

## Teaching Pipeline 7 — Evaluation

Evaluates trained model artifacts.

Run:

```bash
python -m crypto_regime.evaluation.evaluate --config configs/evaluation.yaml
```

## Teaching Pipeline 8 — Serving and Inference

Serves trained model artifacts through FastAPI.

Run the API:

```bash
uvicorn crypto_regime.api.main:app --reload
```

## Teaching Pipeline 9 — Monitoring

Creates basic monitoring outputs for freshness and drift.

Run:

```bash
python -m crypto_regime.monitoring.run_monitoring --config configs/monitoring.yaml
```

## Teaching Pipeline 10 — Dashboard and Communication

Runs the Streamlit dashboard.

Run:

```bash
streamlit run app/streamlit_app.py
```

## Current Build

This project is still under development. The current build supports the Milestone 1 binary high-volatility workflow. The multiclass regime workflow and some production polish are planned but not complete yet.
