"""Shared utility helpers."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logger(name: str = "nebulaquant", level: int = logging.INFO) -> logging.Logger:
    """Return a configured stdout logger (idempotent)."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists and return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
