"""Tests for feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import add_features, feature_columns


def _synthetic_ohlcv(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    base = 100 + np.cumsum(rng.normal(0, 1, n))
    high = base + rng.uniform(0.2, 1.5, n)
    low = base - rng.uniform(0.2, 1.5, n)
    openp = base + rng.normal(0, 0.3, n)
    close = base + rng.normal(0, 0.3, n)
    volume = rng.integers(500_000, 5_000_000, n)
    return pd.DataFrame({
        "Date": dates, "Open": openp, "High": high, "Low": low,
        "Close": close, "Adj Close": close, "Volume": volume,
    })


def test_add_features_returns_required_columns():
    df = add_features(_synthetic_ohlcv())
    required = [
        "atr_14", "atr_pct", "rsi_14", "macd", "macd_signal", "macd_hist",
        "sma_20", "sma_50", "sma_200", "bb_width_20",
        "rel_volume", "vol_5d", "vol_10d", "vol_20d",
    ]
    for col in required:
        assert col in df.columns, f"missing feature: {col}"


def test_features_no_inf_after_dropna():
    df = add_features(_synthetic_ohlcv()).dropna()
    numeric = df.select_dtypes(include=[np.number])
    assert not np.isinf(numeric.values).any(), "found inf in features"


def test_feature_columns_excludes_meta():
    df = add_features(_synthetic_ohlcv())
    cols = feature_columns(df)
    for forbidden in ("Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"):
        assert forbidden not in cols
    # Label/meta columns must never leak into the feature set.
    for forbidden in ("tb_label", "tb_return", "tb_exit_idx", "expansion_label", "sample_weight"):
        assert forbidden not in cols


def test_compression_features_present_and_bounded():
    df = add_features(_synthetic_ohlcv())
    squeeze_cols = [
        "bb_width_pctile_126", "ttm_squeeze_on", "atr_ratio_5_20",
        "atr_pctile_126", "nr7", "nr4", "consolidation_days",
        "donchian_width_20", "donchian_pos_20",
    ]
    for col in squeeze_cols:
        assert col in df.columns, f"missing compression feature: {col}"

    clean = df.dropna()
    # Flags are binary.
    for flag in ("ttm_squeeze_on", "nr7", "nr4"):
        assert set(clean[flag].unique()).issubset({0, 1}), f"{flag} not binary"
    # Percentile-rank features live in [0, 1] by construction.
    for pcol in ("bb_width_pctile_126", "atr_pctile_126"):
        vals = clean[pcol]
        assert vals.min() >= -1e-9 and vals.max() <= 1 + 1e-9, f"{pcol} out of [0,1]"
