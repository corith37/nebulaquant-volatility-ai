"""Leakage-free, walk-forward validation for time-series labels.

Because labels look forward up to `horizon` bars, a training sample taken too
close to the test block would have a label window that overlaps the test
period - that is leakage. `purged_walkforward_splits` removes those samples
(a "purge" zone of `horizon` trading days plus an `embargo` gap) so every
train/test pair is genuinely out-of-sample.

All splits operate on a dataframe that is sorted ascending by Date. Multiple
tickers may share a date (one row each) - splitting is done on the set of
unique trading dates so a whole cross-section moves together.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class WalkForwardConfig:
    n_splits: int = 5
    horizon: int = 10          # label look-ahead in trading days (purge size)
    embargo: int = 5           # extra gap (trading days) between train and test
    holdout_frac: float = 0.15  # final untouched holdout fraction of dates
    min_train_dates: int = 252  # require at least ~1y of training dates


def _unique_date_positions(dates: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Return (sorted unique dates, position index assigned to each row)."""
    uniq = np.array(sorted(pd.unique(dates)))
    pos_map = {d: i for i, d in enumerate(uniq)}
    row_pos = dates.map(pos_map).to_numpy()
    return uniq, row_pos


def train_holdout_split(
    df: pd.DataFrame, holdout_frac: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split off a final, contiguous, never-optimized holdout by date."""
    df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    uniq, _ = _unique_date_positions(df["Date"])
    if len(uniq) == 0:
        return df, df.iloc[0:0]
    cut_pos = int(len(uniq) * (1 - holdout_frac))
    cut_date = uniq[min(cut_pos, len(uniq) - 1)]
    dev = df[df["Date"] < cut_date].copy()
    holdout = df[df["Date"] >= cut_date].copy()
    return dev, holdout


def purged_walkforward_splits(
    df: pd.DataFrame,
    cfg: WalkForwardConfig,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) positional index arrays for expanding-window
    walk-forward with purge + embargo.

    The development dates (after removing the holdout) are divided into
    `n_splits` contiguous test blocks. For block k, the training set is every
    row whose date precedes the block start by more than (horizon + embargo)
    trading days.
    """
    df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    uniq, row_pos = _unique_date_positions(df["Date"])
    n_dates = len(uniq)
    if n_dates < cfg.min_train_dates + cfg.n_splits:
        # Not enough data for the requested folds; fall back to one split.
        n_splits = 1
    else:
        n_splits = cfg.n_splits

    # Test blocks span the dates AFTER the minimum training window.
    test_region_start = cfg.min_train_dates
    test_positions = np.arange(test_region_start, n_dates)
    if len(test_positions) == 0:
        return []
    blocks = np.array_split(test_positions, n_splits)

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    purge = cfg.horizon + cfg.embargo
    for block in blocks:
        if len(block) == 0:
            continue
        test_start = block[0]
        test_end = block[-1]
        train_cutoff = test_start - purge  # last allowed train date position
        if train_cutoff <= 0:
            continue
        train_mask = row_pos < train_cutoff
        test_mask = (row_pos >= test_start) & (row_pos <= test_end)
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        splits.append((train_idx, test_idx))
    return splits


def describe_splits(df: pd.DataFrame, splits) -> pd.DataFrame:
    """Human-readable summary of each fold's date ranges and sizes."""
    rows = []
    for k, (tr, te) in enumerate(splits):
        rows.append({
            "fold": k,
            "train_rows": len(tr),
            "test_rows": len(te),
            "train_start": df.iloc[tr]["Date"].min(),
            "train_end": df.iloc[tr]["Date"].max(),
            "test_start": df.iloc[te]["Date"].min(),
            "test_end": df.iloc[te]["Date"].max(),
        })
    return pd.DataFrame(rows)
