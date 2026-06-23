"""Logging setup helpers."""

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a module logger."""
    return logging.getLogger(name)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure basic logging for command-line workflows."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
