# Model Card

## Model Details

Stage 1 model family: binary high-volatility classifier for BTCUSDT 5-minute candles with versioned artifacts under `models/saved`.

## Intended Use

Educational risk-signal service that estimates the probability of a high-volatility regime in the next 60 minutes.

## Out-of-Scope Use

This model should not be used as financial advice or as an automated trading system.

## Limitations

The model does not execute trades, predict guaranteed direction, or provide financial advice. Performance may decay during exchange outages, liquidity shocks, macro events, or market regimes absent from training data.

## Ethical and Risk Considerations

Cryptocurrency markets are volatile, speculative, and sensitive to external events that may not be reflected in historical price data.
