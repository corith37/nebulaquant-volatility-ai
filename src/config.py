"""Project configuration loader.

Loads `config.yaml` from the project root and exposes it as a nested dict
plus a set of resolved absolute paths for the standard project folders.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(path: Path | str | None = None) -> Dict[str, Any]:
    """Load YAML config from disk."""
    path = Path(path) if path else CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def get_paths() -> Dict[str, Path]:
    """Resolve canonical project paths."""
    return {
        "root": PROJECT_ROOT,
        "data_raw": PROJECT_ROOT / "data" / "raw",
        "data_processed": PROJECT_ROOT / "data" / "processed",
        "data_predictions": PROJECT_ROOT / "data" / "predictions",
        "data_paper": PROJECT_ROOT / "data" / "paper",
        "models": PROJECT_ROOT / "models" / "saved",
        "reports_metrics": PROJECT_ROOT / "reports" / "metrics",
        "reports_backtests": PROJECT_ROOT / "reports" / "backtests",
        "reports_charts": PROJECT_ROOT / "reports" / "charts",
    }


def ensure_paths() -> Dict[str, Path]:
    """Create all standard folders if missing and return them."""
    paths = get_paths()
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths
