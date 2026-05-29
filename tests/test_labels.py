"""Tests for label generation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.labels import add_labels


def _frame(prices):
    return pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=len(prices), freq="B"),
        "Close": prices,
    })


def test_upside_label():
    # Price jumps >4% over 5 bars.
    prices = [100, 100, 100, 100, 100, 110, 110, 110, 110, 110]
    df = add_labels(_frame(prices), horizon_days=5, upside_threshold=0.04, downside_threshold=-0.04)
    assert df.loc[0, "label"] == 1


def test_downside_label():
    prices = [100, 100, 100, 100, 100, 90, 90, 90, 90, 90]
    df = add_labels(_frame(prices), horizon_days=5, upside_threshold=0.04, downside_threshold=-0.04)
    assert df.loc[0, "label"] == 2


def test_sideways_label():
    prices = [100] * 10
    df = add_labels(_frame(prices), horizon_days=5, upside_threshold=0.04, downside_threshold=-0.04)
    assert df.loc[0, "label"] == 0


def test_tail_rows_have_nan_label():
    prices = list(range(100, 110))
    df = add_labels(_frame(prices), horizon_days=5)
    # Last `horizon_days` rows have undefined future return -> NaN label.
    assert df["label"].iloc[-1] != df["label"].iloc[-1]  # NaN check
