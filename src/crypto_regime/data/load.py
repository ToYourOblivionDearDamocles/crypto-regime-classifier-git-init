"""Load crypto market datasets from disk."""

from pathlib import Path
from typing import Any


def load_price_data(path: str | Path, **read_csv_kwargs: Any):
    """Load a CSV price dataset.

    Pandas is imported lazily so lightweight package imports do not require data dependencies.
    """
    import pandas as pd

    return pd.read_csv(path, **read_csv_kwargs)
