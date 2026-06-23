"""Resample OHLCV market data."""


def resample_ohlcv(frame, rule: str = "1D", timestamp_column: str = "timestamp"):
    """Resample OHLCV data to a target frequency."""
    aggregations = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    indexed = frame.set_index(timestamp_column).sort_index()
    return indexed.resample(rule).agg(aggregations).dropna().reset_index()
