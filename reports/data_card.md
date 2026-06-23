# Data Card

## Dataset

BTCUSDT 5-minute OHLCV candles from Binance public historical klines.

## Data Source

Primary source: Binance public historical market data for spot klines. Backup sources: Binance REST klines or Kraken OHLCVT CSV files.

## Fields

- timestamp
- open
- high
- low
- close
- volume
- quote_volume
- num_trades

## Known Limitations

Exchange-specific data can contain missing candles, outlier moves, symbol-specific liquidity effects, and historical coverage differences.
