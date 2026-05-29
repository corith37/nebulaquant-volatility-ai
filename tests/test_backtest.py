"""Tests for backtest metrics and risk/sizing math (long-only, vol-targeted)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import compute_metrics
from src.risk import (
    RiskConfig,
    confidence_fraction,
    shares_for_fraction,
    stop_loss_price,
    take_profit_price,
)


# ---- Sizing / barrier-price helpers ----

def test_shares_for_fraction_floor():
    # $10,000 * 0.1 / $99 -> 10 shares (floored).
    assert shares_for_fraction(10_000, 0.10, 99) == 10


def test_shares_for_fraction_guards():
    assert shares_for_fraction(10_000, 0.0, 100) == 0
    assert shares_for_fraction(10_000, 0.1, 0) == 0


def test_atr_scaled_barriers():
    cfg = RiskConfig(tp_atr_mult=2.0, sl_atr_mult=1.0)
    assert take_profit_price(100, 2.0, cfg) == 104.0   # 100 + 2*2
    assert stop_loss_price(100, 2.0, cfg) == 98.0      # 100 - 1*2


def test_confidence_fraction_zero_below_threshold():
    cfg = RiskConfig(position_size_pct=0.10, vol_target=False)
    assert confidence_fraction(0.50, 0.55, 0.02, cfg, ref_atr_pct=0.02) == 0.0


def test_confidence_fraction_scales_and_caps():
    cfg = RiskConfig(position_size_pct=0.10, max_position_pct=0.20, vol_target=False)
    # High conviction (well above threshold) -> full 1.5x tilt of base size.
    frac = confidence_fraction(1.0, 0.55, 0.02, cfg, ref_atr_pct=0.02)
    assert abs(frac - 0.15) < 1e-9        # 0.10 * (0.5 + 1.0)
    # Just above the threshold -> minimum 0.5x tilt of base size.
    frac_low = confidence_fraction(0.551, 0.55, 0.02, cfg, ref_atr_pct=0.02)
    assert abs(frac_low - 0.05) < 1e-3    # ~0.10 * 0.5


def test_confidence_fraction_vol_targeting_downsizes_high_vol():
    cfg = RiskConfig(position_size_pct=0.10, max_position_pct=0.20, vol_target=True)
    # Twice as volatile as the reference -> ~half the size.
    high_vol = confidence_fraction(1.0, 0.55, 0.04, cfg, ref_atr_pct=0.02)
    low_vol = confidence_fraction(1.0, 0.55, 0.02, cfg, ref_atr_pct=0.02)
    assert high_vol < low_vol
    assert abs(high_vol - 0.075) < 1e-9   # 0.15 * (0.02/0.04)


# ---- Portfolio metrics ----

def test_compute_metrics_empty():
    m = compute_metrics(pd.DataFrame(), pd.DataFrame(columns=["date", "equity"]), initial_capital=10_000)
    assert m["n_trades"] == 0
    assert m["final_equity"] == 10_000


def test_compute_metrics_basic():
    trades = pd.DataFrame({
        "pnl": [100.0, -50.0],
        "return_pct": [0.01, -0.005],
        "hold_days": [3, 2],
    })
    eq = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-10", "2024-01-20"]),
        "equity": [10_000.0, 10_100.0, 10_050.0],
    })
    m = compute_metrics(trades, eq, initial_capital=10_000)
    assert m["n_trades"] == 2
    assert m["win_rate"] == 0.5
    assert m["best_trade"] == 100.0
    assert m["worst_trade"] == -50.0
    assert m["final_equity"] == 10_050.0
    assert abs(m["profit_factor"] - 2.0) < 1e-6      # 100 / 50
    assert abs(m["expectancy"] - 25.0) < 1e-6        # (100 - 50) / 2
