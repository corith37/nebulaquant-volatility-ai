"""Scan the latest data and produce a ranked predictions CSV."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, ensure_paths
from src.scanner import latest_per_ticker, scan
from src.train_model import load_model
from src.utils import setup_logger


def main() -> int:
    logger = setup_logger("nebulaquant.scanner_script")
    cfg = load_config()
    paths = ensure_paths()

    try:
        model, feat_cols = load_model(paths["models"])
    except FileNotFoundError:
        logger.error("Trained model not found. Run train first.")
        return 1

    threshold = cfg.get("model", {}).get("probability_threshold", 0.55)

    tickers = cfg["data"]["tickers"]
    latest = latest_per_ticker(paths["data_processed"], tickers)
    if latest.empty:
        logger.error("No latest features available. Run build_dataset first.")
        return 2

    out_path = paths["data_predictions"] / "latest_predictions.csv"
    result = scan(model, latest, threshold=threshold, out_path=out_path)
    logger.info(f"Scanned {len(result)} tickers")
    if not result.empty:
        top = result.head(10)[["ticker", "final_score", "expansion_probability", "long_probability", "suggested_action"]]
        logger.info("Top setups:\n%s", top.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
