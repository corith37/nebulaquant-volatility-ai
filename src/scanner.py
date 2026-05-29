"""Scanner: rank today's setups with the calibrated two-stage model.

For each ticker's most recent bar we emit the expansion probability (Stage A),
the long-hit probability (Stage B), and the combined ``final_score =
expansion_prob * long_prob``. A suggested action is derived from the calibrated
score plus simple confirmation filters (trend, squeeze, relative volume).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from src.features import add_cross_sectional
from src.train_model import TwoStageModel
from src.utils import setup_logger

logger = setup_logger("nebulaquant.scanner")


def suggest_action(row: pd.Series, threshold: float) -> str:
    """Map a calibrated score + confirmations onto a watch label."""
    score = row.get("final_score", 0.0)
    long_p = row.get("long_probability", 0.0)
    above20 = row.get("price_above_sma20", 0)
    rel_vol = row.get("relative_volume", 0) or 0
    squeeze = row.get("squeeze_on", 0)

    if score < threshold or above20 != 1:
        return "No Trade"
    strong = score >= threshold + 0.10 and long_p >= 0.55
    if strong and (squeeze == 1 or rel_vol >= 1.5):
        return "Strong Long Watch"
    return "Long Watch"


def latest_per_ticker(processed_dir: Path, tickers: List[str]) -> pd.DataFrame:
    """Take the last fully-populated row of each per-ticker processed CSV."""
    processed_dir = Path(processed_dir)
    frames = []
    for t in tickers:
        path = processed_dir / f"{t}_features.csv"
        if not path.exists():
            logger.warning(f"[{t}] features file missing: {path}")
            continue
        df = pd.read_csv(path, parse_dates=["Date"])
        if df.empty:
            continue
        last = df.dropna().tail(1)
        if last.empty:
            continue
        frames.append(last)
    if not frames:
        return pd.DataFrame()
    snapshot = pd.concat(frames, ignore_index=True)
    # Per-ticker CSVs lack the cross-sectional rank features (those are computed
    # on the full universe). Recompute them across today's snapshot so the
    # latest bars are ranked against each other, matching training.
    snapshot = add_cross_sectional(snapshot)
    return snapshot


def scan_momentum(
    latest_df: pd.DataFrame,
    top_pctile: float = 0.90,
    max_vix_pctile: float = 0.85,
    out_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Rules-based scan: rank today's names by the proven momentum signal.

    Implements the pivoted strategy the backtest validated (beats SPY on return
    and Sharpe): among names in a broad-market uptrend (SPY > 200d SMA, benign
    VIX) that are themselves trending (> 20d & 200d SMA), rank by 60-day
    cross-sectional momentum and flag the top decile as longs. No ML model.

    ``latest_df`` must carry the cross-sectional rank columns (produced by
    ``latest_per_ticker`` -> ``add_cross_sectional`` on today's snapshot).
    """
    if latest_df.empty or "xs_ret60_rank" not in latest_df.columns:
        return pd.DataFrame()

    df = latest_df.copy()

    def _col(name, default=np.nan):
        return df.get(name, pd.Series([default] * len(df))).to_numpy()

    spy_up = _col("spy_above_sma200", 1)
    vix_pct = _col("vix_pctile_252", 0.0)
    above20 = _col("price_above_sma20", 0)
    above200 = _col("price_above_sma200", 0)
    mom = df["xs_ret60_rank"].to_numpy()

    market_ok = (np.nan_to_num(spy_up, nan=1.0) == 1) & (np.nan_to_num(vix_pct, nan=0.0) <= max_vix_pctile)
    trend_ok = (np.nan_to_num(above20, nan=0.0) == 1) & (np.nan_to_num(above200, nan=0.0) == 1)
    eligible = market_ok & trend_ok

    action = np.where(
        eligible & (mom >= top_pctile), "Long",
        np.where(eligible & (mom >= top_pctile - 0.10), "Watch", "No Trade"),
    )

    out = pd.DataFrame({
        "ticker": df["Ticker"].to_numpy(),
        "date": pd.to_datetime(df["Date"]).dt.date,
        "close": np.round(df["Close"].to_numpy(), 4),
        "momentum_rank": np.round(mom, 4),
        "ret_60d": np.round(_col("ret_60d"), 4),
        "ret_20d": np.round(_col("ret_20d"), 4),
        "rs_vs_spy": np.round(_col("xs_rs_vs_spy"), 4),
        "atr_percent": np.round(_col("atr_pct"), 4),
        "price_above_sma20": above20,
        "price_above_sma200": above200,
        "market_regime_ok": market_ok.astype(int),
        "suggested_action": action,
    })
    out = out.sort_values("momentum_rank", ascending=False).reset_index(drop=True)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        logger.info(f"Saved momentum scanner output -> {out_path}")
    return out


def scan(
    model: TwoStageModel,
    latest_df: pd.DataFrame,
    threshold: float = 0.55,
    out_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Score the latest rows with the two-stage model and rank them."""
    if latest_df.empty:
        return pd.DataFrame()

    missing = [c for c in model.feat_cols if c not in latest_df.columns]
    if missing:
        logger.warning(f"Latest data missing {len(missing)} feature columns: {missing[:8]}...")
        for c in missing:
            latest_df[c] = 0.0

    scores = model.predict(latest_df)

    def _col(name, default=np.nan):
        return latest_df.get(name, pd.Series([default] * len(latest_df))).to_numpy()

    out = pd.DataFrame({
        "ticker": latest_df["Ticker"].to_numpy(),
        "date": pd.to_datetime(latest_df["Date"]).dt.date,
        "close": np.round(latest_df["Close"].to_numpy(), 4),
        "expansion_probability": np.round(scores["expansion_prob"].to_numpy(), 4),
        "long_probability": np.round(scores["long_prob"].to_numpy(), 4),
        "final_score": np.round(scores["score"].to_numpy(), 4),
        "atr_percent": _col("atr_pct"),
        "relative_volume": _col("rel_volume"),
        "rsi": _col("rsi_14"),
        "bb_width_pctile": _col("bb_width_pctile_126"),
        "squeeze_on": _col("ttm_squeeze_on", 0),
        "atr_ratio_5_20": _col("atr_ratio_5_20"),
        "price_above_sma20": _col("price_above_sma20", 0),
        "price_above_sma50": _col("price_above_sma50", 0),
    })
    out["suggested_action"] = out.apply(lambda r: suggest_action(r, threshold), axis=1)
    out = out.sort_values("final_score", ascending=False).reset_index(drop=True)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        logger.info(f"Saved scanner output -> {out_path}")

    return out
