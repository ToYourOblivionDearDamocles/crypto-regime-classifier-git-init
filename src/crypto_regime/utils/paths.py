"""Path helpers for project directories."""

from pathlib import Path


def project_root() -> Path:
    """Return the repository root based on the current file location."""
    return Path(__file__).resolve().parents[3]


def data_dir() -> Path:
    """Return the data directory."""
    return project_root() / "data"


def models_dir() -> Path:
    """Return the saved models directory."""
    return project_root() / "models" / "saved"
