"""Walk-forward validation split generation."""

from collections.abc import Iterator


def walk_forward_splits(
    n_samples: int,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
) -> Iterator[tuple[slice, slice]]:
    """Yield train and test slices for walk-forward validation."""
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive")
    step = step_size or test_size
    start = 0
    while start + train_size + test_size <= n_samples:
        train_slice = slice(start, start + train_size)
        test_slice = slice(start + train_size, start + train_size + test_size)
        yield train_slice, test_slice
        start += step
