"""Multiclass market regime labels."""


def make_multiclass_labels(
    frame,
    close_column: str = "close",
    horizon: int = 7,
    lower_threshold: float = -0.02,
    upper_threshold: float = 0.02,
    output_column: str = "regime_multiclass",
):
    """Create bearish, neutral, and bullish labels from forward returns."""
    result = frame.copy()
    forward_return = result[close_column].shift(-horizon) / result[close_column] - 1
    result[output_column] = 0
    result.loc[forward_return < lower_threshold, output_column] = -1
    result.loc[forward_return > upper_threshold, output_column] = 1
    return result
