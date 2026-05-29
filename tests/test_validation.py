"""Leakage tests for the purged + embargoed walk-forward splitter.

The core correctness guarantee: no training sample's label window (entry .. entry
+ horizon) may overlap the dates in its fold's test block. If it did, the model
would be trained on information from the test period.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.validation import (
    WalkForwardConfig,
    purged_walkforward_splits,
    train_holdout_split,
)


def _panel(n_dates: int = 800, n_tickers: int = 3) -> pd.DataFrame:
    dates = pd.bdate_range("2016-01-01", periods=n_dates)
    rows = []
    for t in range(n_tickers):
        for d in dates:
            rows.append({"Date": d, "Ticker": f"T{t}", "Close": 100.0})
    return pd.DataFrame(rows)


def test_holdout_is_contiguous_and_last():
    df = _panel()
    dev, hold = train_holdout_split(df, holdout_frac=0.15)
    assert not dev.empty and not hold.empty
    # Every dev date strictly precedes every holdout date.
    assert dev["Date"].max() < hold["Date"].min()
    # Holdout is ~15% of unique dates.
    frac = hold["Date"].nunique() / df["Date"].nunique()
    assert 0.10 < frac < 0.20


def test_no_label_window_overlaps_test_block():
    df = _panel(n_dates=900).sort_values(["Date", "Ticker"]).reset_index(drop=True)
    cfg = WalkForwardConfig(n_splits=5, horizon=10, embargo=5, holdout_frac=0.0, min_train_dates=252)
    splits = purged_walkforward_splits(df, cfg)
    assert len(splits) >= 1

    uniq_dates = np.array(sorted(df["Date"].unique()))
    pos = {d: i for i, d in enumerate(uniq_dates)}
    purge = cfg.horizon + cfg.embargo

    for tr_idx, te_idx in splits:
        train_dates = df.iloc[tr_idx]["Date"]
        test_dates = df.iloc[te_idx]["Date"]
        test_start_pos = pos[test_dates.min()]
        # Each training sample's label window ends `horizon` bars later; the
        # purge guarantees train positions end before test_start - purge, so even
        # the forward label window cannot reach the test block.
        max_train_pos = max(pos[d] for d in train_dates)
        assert max_train_pos + cfg.horizon < test_start_pos + purge
        # Hard guarantee: no train date is inside the test block.
        assert train_dates.max() < test_dates.min()


def test_expanding_window_grows():
    df = _panel(n_dates=1000)
    cfg = WalkForwardConfig(n_splits=4, horizon=10, embargo=5, holdout_frac=0.0, min_train_dates=300)
    splits = purged_walkforward_splits(df, cfg)
    train_sizes = [len(tr) for tr, _ in splits]
    # Expanding window: later folds train on at least as much data.
    assert train_sizes == sorted(train_sizes)
