"""Download historical OHLCV data for the configured ticker universe."""
from __future__ import annotations

import sys
from pathlib import Path

# Make `src` importable when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, ensure_paths
from src.data_loader import download_universe
from src.utils import setup_logger


def main() -> int:
    logger = setup_logger("nebulaquant.download")
    cfg = load_config()
    paths = ensure_paths()

    data_cfg = cfg.get("data", {})
    tickers = list(data_cfg.get("tickers", []))
    context = list(data_cfg.get("context_tickers", []))
    start = data_cfg.get("start_date", "2015-01-01")
    end = data_cfg.get("end_date")

    if not tickers:
        logger.error("No tickers configured under data.tickers in config.yaml")
        return 1

    # Download the tradeable universe plus any context-only symbols (e.g. ^VIX),
    # de-duplicated while preserving order.
    all_symbols = list(dict.fromkeys(tickers + context))
    logger.info(
        f"Downloading {len(all_symbols)} symbols "
        f"({len(tickers)} tradeable + {len(context)} context) "
        f"from {start} to {end or 'today'}"
    )
    success = download_universe(all_symbols, start=start, end=end, out_dir=paths["data_raw"])

    if not success:
        logger.error("No tickers downloaded successfully.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
