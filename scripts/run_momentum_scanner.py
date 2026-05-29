"""Rules-based momentum scanner: today's ranked long candidates.

Ranks the universe by 60-day cross-sectional momentum and flags the top-decile
names that also pass the regime/trend gate (SPY > 200d SMA, benign VIX, name
> 20d & 200d SMA). This is the live counterpart to the backtest in
scripts/run_momentum_backtest.py (the pivoted strategy that beats SPY).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ensure_paths, load_config
from src.scanner import latest_per_ticker, scan_momentum
from src.utils import setup_logger


def main() -> int:
    logger = setup_logger("nebulaquant.momentum_scanner")
    cfg = load_config()
    paths = ensure_paths()
    mom = cfg.get("momentum", {})
    bt = cfg.get("backtest", {})

    tickers = cfg["data"]["tickers"]
    latest = latest_per_ticker(paths["data_processed"], tickers)
    if latest.empty:
        logger.error("No latest features available. Run build_dataset first.")
        return 1

    out_path = paths["data_predictions"] / "latest_momentum.csv"
    result = scan_momentum(
        latest,
        top_pctile=float(mom.get("top_pctile", 0.90)),
        max_vix_pctile=float(bt.get("max_vix_pctile", 0.85)),
        out_path=out_path,
    )
    if result.empty:
        logger.error("Scanner produced no rows (missing cross-sectional columns?).")
        return 2

    longs = result[result["suggested_action"] == "Long"]
    logger.info(f"Scanned {len(result)} tickers; {len(longs)} flagged Long.")
    cols = ["ticker", "close", "momentum_rank", "ret_60d", "rs_vs_spy",
            "market_regime_ok", "suggested_action"]
    logger.info("Top setups:\n%s", result.head(15)[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
