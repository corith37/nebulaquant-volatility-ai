"""Combine per-ticker features + labels into a single training dataset.

Pipeline per ticker:
    raw OHLCV -> features -> market context (SPY/QQQ/VIX)
              -> expansion label (Stage A) + triple-barrier label (Stage B)
              -> uniqueness sample weights
Then all tickers are concatenated and rows with missing values are dropped.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from src.data_loader import load_raw
from src.features import add_cross_sectional, add_features, add_market_context
from src.labels import (
    add_expansion_label,
    add_triple_barrier_label,
    add_uniqueness_weight,
)
from src.utils import setup_logger

logger = setup_logger("nebulaquant.dataset")


def _load_context(ticker: Optional[str], raw_dir: Path, with_features: bool) -> Optional[pd.DataFrame]:
    if not ticker:
        return None
    try:
        raw = load_raw(ticker, raw_dir)
    except FileNotFoundError:
        logger.warning(f"Context ticker {ticker} not found; skipping")
        return None
    return add_features(raw) if with_features else raw


def build_ticker_frame(
    ticker: str,
    raw_dir: Path,
    spy_ctx: Optional[pd.DataFrame],
    qqq_ctx: Optional[pd.DataFrame],
    vix_ctx: Optional[pd.DataFrame],
    tb_cfg: dict,
) -> Optional[pd.DataFrame]:
    """Load a single ticker, build features + labels + weights."""
    try:
        raw = load_raw(ticker, raw_dir)
    except FileNotFoundError:
        logger.warning(f"[{ticker}] raw file missing, skipping")
        return None
    if raw.empty:
        logger.warning(f"[{ticker}] empty raw file, skipping")
        return None

    feats = add_features(raw)
    if spy_ctx is not None:
        feats = add_market_context(feats, spy_ctx, qqq=qqq_ctx, vix=vix_ctx)

    feats = add_expansion_label(
        feats,
        horizon=int(tb_cfg.get("expansion_horizon", 10)),
        atr_mult=float(tb_cfg.get("expansion_atr_mult", 1.5)),
    )
    feats = add_triple_barrier_label(
        feats,
        horizon=int(tb_cfg.get("horizon", 10)),
        tp_atr_mult=float(tb_cfg.get("tp_atr_mult", 2.0)),
        sl_atr_mult=float(tb_cfg.get("sl_atr_mult", 1.0)),
    )
    feats = add_uniqueness_weight(feats)
    feats["Ticker"] = ticker
    return feats


def build_dataset(
    tickers: Iterable[str],
    raw_dir: Path,
    processed_dir: Path,
    tb_cfg: dict,
    market_ticker: str = "SPY",
    qqq_ticker: Optional[str] = "QQQ",
    vix_ticker: Optional[str] = "^VIX",
) -> pd.DataFrame:
    """Build and persist the combined model dataset."""
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    spy_ctx = _load_context(market_ticker, raw_dir, with_features=True)
    qqq_ctx = _load_context(qqq_ticker, raw_dir, with_features=False)
    vix_ctx = _load_context(vix_ticker, raw_dir, with_features=False)
    if spy_ctx is not None:
        logger.info(f"Market context loaded (SPY{' +QQQ' if qqq_ctx is not None else ''}"
                    f"{' +VIX' if vix_ctx is not None else ''})")

    # Don't model the VIX index itself as a tradeable ticker.
    skip = {vix_ticker} if vix_ticker else set()
    frames: List[pd.DataFrame] = []
    for t in tickers:
        if t in skip:
            continue
        f = build_ticker_frame(t, raw_dir, spy_ctx, qqq_ctx, vix_ctx, tb_cfg)
        if f is None or f.empty:
            continue
        from src.data_loader import safe_filename
        out_path = processed_dir / f"{safe_filename(t)}_features.csv"
        f.to_csv(out_path, index=False)
        logger.info(f"[{t}] processed {len(f)} rows -> {out_path.name}")
        frames.append(f)

    if not frames:
        raise RuntimeError("No ticker frames produced - check raw data.")

    combined = pd.concat(frames, ignore_index=True)
    # Cross-sectional (per-date) relative-strength / low-vol rank features.
    # Must run on the full multi-ticker frame so each date ranks the universe.
    combined = add_cross_sectional(combined)
    combined = combined.replace([np.inf, -np.inf], np.nan)
    before = len(combined)
    combined = combined.dropna()
    logger.info(f"Combined dataset: {before} -> {len(combined)} rows after dropna")

    out_path = processed_dir / "model_dataset.csv"
    combined.to_csv(out_path, index=False)
    logger.info(f"Saved combined dataset -> {out_path}")
    return combined


def chronological_split(
    df: pd.DataFrame,
    test_size: float = 0.15,
    validation_size: float = 0.15,
):
    """Split a time-sorted dataframe into train/val/test chronologically (no shuffle)."""
    df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    n = len(df)
    n_test = int(n * test_size)
    n_val = int(n * validation_size)
    n_train = n - n_val - n_test
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train:n_train + n_val].copy()
    test = df.iloc[n_train + n_val:].copy()
    return train, val, test
