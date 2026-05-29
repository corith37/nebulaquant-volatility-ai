"""Tests for triple-barrier, expansion, and uniqueness labels."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.labels import (
    add_expansion_label,
    add_triple_barrier_label,
    add_uniqueness_weight,
)


def _frame(close, high, low, atr=1.0):
    n = len(close)
    return pd.DataFrame({
        "Date": pd.bdate_range("2024-01-01", periods=n),
        "Close": np.asarray(close, float),
        "High": np.asarray(high, float),
        "Low": np.asarray(low, float),
        "atr_14": np.full(n, float(atr)),
    })


def test_triple_barrier_take_profit_first():
    # Entry = close[1] = 100, atr = 1, tp = 102, sl = 99.
    # Bar 3 touches the upper barrier (high 102.5) before any stop.
    close = [100, 100, 101, 103, 103, 103, 103]
    high =  [100, 100.5, 101.5, 102.5, 103, 103, 103]
    low =   [100, 99.5, 100.5, 101.0, 102, 102, 102]
    df = add_triple_barrier_label(_frame(close, high, low), horizon=5, tp_atr_mult=2.0, sl_atr_mult=1.0)
    assert df.loc[0, "tb_label"] == 1
    assert abs(df.loc[0, "tb_return"] - 0.02) < 1e-9   # 102/100 - 1
    assert df.loc[0, "tb_exit_idx"] == 3


def test_triple_barrier_stop_loss_first():
    # Entry = 100, tp = 102, sl = 99. Bar 2 low 98.5 hits the stop first.
    close = [100, 100, 100, 100, 100, 100, 100]
    high =  [100, 100.5, 100.5, 100.5, 100.5, 100.5, 100.5]
    low =   [100, 99.5, 98.5, 99.0, 99.0, 99.0, 99.0]
    df = add_triple_barrier_label(_frame(close, high, low), horizon=5, tp_atr_mult=2.0, sl_atr_mult=1.0)
    assert df.loc[0, "tb_label"] == 0
    assert abs(df.loc[0, "tb_return"] - (-0.01)) < 1e-9   # 99/100 - 1
    assert df.loc[0, "tb_exit_idx"] == 2


def test_triple_barrier_both_touch_assumes_stop():
    # Bar 2 spans both barriers (low 98 <= sl, high 103 >= tp) -> stop assumed.
    close = [100, 100, 100, 100, 100, 100, 100]
    high =  [100, 100.5, 103.0, 100.5, 100.5, 100.5, 100.5]
    low =   [100, 99.5, 98.0, 99.5, 99.5, 99.5, 99.5]
    df = add_triple_barrier_label(_frame(close, high, low), horizon=5, tp_atr_mult=2.0, sl_atr_mult=1.0)
    assert df.loc[0, "tb_label"] == 0


def test_triple_barrier_time_barrier_exit():
    # Never touches either barrier; exits at the time barrier on close.
    close = [100, 100, 100.5, 100.5, 100.5, 101.0, 100]
    high =  [100, 100.4, 100.9, 100.9, 100.9, 101.4, 100]
    low =   [100, 99.6, 100.1, 100.1, 100.1, 100.6, 100]
    df = add_triple_barrier_label(_frame(close, high, low), horizon=4, tp_atr_mult=2.0, sl_atr_mult=1.0)
    # exit at entry_idx+horizon = 1+4 = 5, close 101 vs entry 100 -> positive.
    assert df.loc[0, "tb_exit_idx"] == 5
    assert df.loc[0, "tb_label"] == 1
    assert abs(df.loc[0, "tb_return"] - 0.01) < 1e-9


def test_expansion_label_detects_sustained_expansion():
    # Forward bars carry a wide ~2.0 true range vs current ATR 1.0 -> the
    # forward average TR (2.0) exceeds 1.3*ATR, so volatility expands.
    close = [100] * 12
    high = [100] + [101] * 11      # forward bars span 99..101 -> TR 2.0
    low = [100] + [99] * 11
    df = add_expansion_label(_frame(close, high, low, atr=1.0), horizon=10, atr_mult=1.3)
    assert df.loc[0, "expansion_label"] == 1


def test_expansion_label_quiet_is_zero():
    # Forward average TR (~0.6) stays below 1.3*ATR -> no expansion.
    close = [100] * 12
    high = [100.3] * 12
    low = [99.7] * 12
    df = add_expansion_label(_frame(close, high, low, atr=1.0), horizon=10, atr_mult=1.3)
    assert df.loc[0, "expansion_label"] == 0


def test_expansion_label_single_spike_is_not_expansion():
    # A lone spike does NOT count as sustained expansion (forward avg TR low).
    close = [100] * 12
    high = [100.2] * 5 + [105] + [100.2] * 6     # one +5 spike only
    low = [99.8] * 12
    df = add_expansion_label(_frame(close, high, low, atr=1.0), horizon=10, atr_mult=1.3)
    assert df.loc[0, "expansion_label"] == 0


def test_uniqueness_weight_in_unit_range():
    close = list(range(100, 130))
    high = [c + 1 for c in close]
    low = [c - 1 for c in close]
    df = add_triple_barrier_label(_frame(close, high, low), horizon=5)
    df = add_uniqueness_weight(df)
    w = df["sample_weight"].dropna()
    assert len(w) > 0
    assert (w > 0).all() and (w <= 1.0 + 1e-9).all()
