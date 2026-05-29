"""Build the combined model dataset from raw OHLCV CSVs."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, ensure_paths
from src.dataset import build_dataset
from src.utils import setup_logger


def main() -> int:
    logger = setup_logger("nebulaquant.build_dataset")
    cfg = load_config()
    paths = ensure_paths()

    tickers = cfg["data"]["tickers"]
    market_ticker = cfg["data"].get("market_ticker", "SPY")
    qqq_ticker = cfg["data"].get("qqq_ticker", "QQQ")
    vix_ticker = cfg["data"].get("vix_ticker", "^VIX")
    tb_cfg = cfg.get("triple_barrier", {})

    logger.info(f"Building dataset for {len(tickers)} tickers")
    df = build_dataset(
        tickers=tickers,
        raw_dir=paths["data_raw"],
        processed_dir=paths["data_processed"],
        tb_cfg=tb_cfg,
        market_ticker=market_ticker,
        qqq_ticker=qqq_ticker,
        vix_ticker=vix_ticker,
    )
    logger.info(f"Final dataset shape: {df.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
