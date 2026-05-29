"""Labels for the two-stage volatility model.

Three label products are produced (all leakage-free; they look forward only to
assign the target, never to build features):

1. `add_expansion_label`  - Stage A binary target: will volatility/range expand
   meaningfully over the next `horizon` bars?
2. `add_triple_barrier_label` - Stage B binary target: for a long entered at the
   next bar, does an ATR-scaled profit target get hit before the stop (within a
   vertical time barrier)? This is exactly the outcome the backtest rewards.
3. `add_uniqueness_weight` - sample weights that down-weight overlapping label
   windows (concurrent labels share information; treating them as independent
   over-counts evidence).

The legacy 3-class `add_labels` is kept for backward compatibility.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Legacy 3-class label (kept for compatibility / comparison)
# ----------------------------------------------------------------------------

def add_labels(
    df: pd.DataFrame,
    horizon_days: int = 5,
    upside_threshold: float = 0.04,
    downside_threshold: float = -0.04,
) -> pd.DataFrame:
    """Legacy fixed-threshold 3-class label (0 sideways / 1 up / 2 down)."""
    df = df.copy()
    close = df["Close"]
    df["future_return"] = (close.shift(-horizon_days) / close) - 1
    conditions = [
        df["future_return"] > upside_threshold,
        df["future_return"] < downside_threshold,
    ]
    df["label"] = np.select(conditions, [1, 2], default=0).astype(float)
    df.loc[df["future_return"].isna(), "label"] = np.nan
    return df


# ----------------------------------------------------------------------------
# Stage A: volatility-expansion label
# ----------------------------------------------------------------------------

def add_expansion_label(
    df: pd.DataFrame,
    horizon: int = 10,
    atr_mult: float = 1.3,
) -> pd.DataFrame:
    """Binary label: 1 if *sustained* volatility expands over the next `horizon`
    bars, else 0.

    Expansion is measured as the forward **average true range** over the next
    `horizon` bars relative to the current ATR known at the signal bar:

        label = 1  iff  mean(TR over bars i+1..i+horizon) >= atr_mult * ATR_i

    This is the true "squeeze -> expansion" definition. The earlier label used
    the *max* forward move vs ``1.5 * ATR``, which fires on ~85-95% of bars (over
    10 bars almost any name travels >1.5 daily ATRs), so Stage A learned nothing.
    Comparing the forward *average* per-bar range to the current per-bar ATR is
    naturally balanced and rewards genuinely sustained range expansion, not a
    single spike. Uses only information known at bar i for the threshold (no
    leakage into the feature set).
    """
    df = df.copy()
    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()
    close = df["Close"].to_numpy()
    atr = df["atr_14"].to_numpy()
    n = len(df)

    # True range per bar (max of HL, |H-prevC|, |L-prevC|).
    prev_close = np.empty(n)
    prev_close[0] = np.nan
    prev_close[1:] = close[:-1]
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])

    labels = np.full(n, np.nan)
    for i in range(n):
        if not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        end = i + horizon
        if end >= n:
            break
        fwd_tr = tr[i + 1:end + 1]
        if fwd_tr.size == 0 or not np.isfinite(fwd_tr).all():
            continue
        fwd_atr = float(fwd_tr.mean())
        labels[i] = 1.0 if fwd_atr >= atr_mult * atr[i] else 0.0

    df["expansion_label"] = labels
    return df


# ----------------------------------------------------------------------------
# Stage B: triple-barrier label (long-only meta-label)
# ----------------------------------------------------------------------------

def add_triple_barrier_label(
    df: pd.DataFrame,
    horizon: int = 10,
    tp_atr_mult: float = 2.0,
    sl_atr_mult: float = 1.0,
) -> pd.DataFrame:
    """ATR-scaled triple-barrier label for a LONG entered at the next bar's close.

    For each bar i, entry is the close of bar i+1. Upper barrier =
    entry + tp_atr_mult*ATR, lower barrier = entry - sl_atr_mult*ATR, vertical
    barrier = `horizon` bars. Label = 1 if the upper barrier is touched strictly
    before the lower one; 0 otherwise (lower or time barrier first). Intrabar
    touches use high/low; if both barriers fall inside one bar, the stop is
    assumed hit first (conservative).

    Adds: tb_label (0/1), tb_return (realized fractional PnL of the barrier
    trade), tb_exit_idx (row offset of the exit, used for uniqueness weighting).
    """
    df = df.copy()
    close = df["Close"].to_numpy()
    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()
    atr = df["atr_14"].to_numpy()
    n = len(df)

    tb_label = np.full(n, np.nan)
    tb_return = np.full(n, np.nan)
    tb_exit = np.full(n, np.nan)

    for i in range(n):
        if not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        entry_idx = i + 1
        if entry_idx >= n:
            break
        entry = close[entry_idx]
        if not np.isfinite(entry) or entry <= 0:
            continue
        tp = entry + tp_atr_mult * atr[i]
        sl = entry - sl_atr_mult * atr[i]

        label = 0
        ret = 0.0
        exit_idx = min(entry_idx + horizon, n - 1)
        for j in range(entry_idx, min(entry_idx + horizon, n - 1) + 1):
            hit_sl = low[j] <= sl
            hit_tp = high[j] >= tp
            if hit_sl and hit_tp:
                # Both inside one bar -> assume stop first (conservative).
                label, ret, exit_idx = 0, (sl / entry - 1), j
                break
            if hit_sl:
                label, ret, exit_idx = 0, (sl / entry - 1), j
                break
            if hit_tp:
                label, ret, exit_idx = 1, (tp / entry - 1), j
                break
        else:
            # Time barrier: exit at last bar's close.
            ret = close[exit_idx] / entry - 1
            label = 1 if ret > 0 else 0

        tb_label[i] = label
        tb_return[i] = ret
        tb_exit[i] = exit_idx

    df["tb_label"] = tb_label
    df["tb_return"] = tb_return
    df["tb_exit_idx"] = tb_exit
    return df


# ----------------------------------------------------------------------------
# Sample-uniqueness weights (López de Prado, simplified)
# ----------------------------------------------------------------------------

def add_uniqueness_weight(df: pd.DataFrame) -> pd.DataFrame:
    """Weight each sample by the inverse of its average label concurrency.

    Two samples are "concurrent" if their [entry, exit] index spans overlap.
    Highly overlapping samples carry redundant information, so they get less
    weight. Requires `tb_exit_idx` from `add_triple_barrier_label`. Operates on
    a single ticker's frame (positional index = bar order).
    """
    df = df.copy().reset_index(drop=True)
    n = len(df)
    exit_idx = df["tb_exit_idx"].to_numpy()

    # concurrency[t] = number of labels whose span covers bar t.
    concurrency = np.zeros(n)
    spans = []
    for i in range(n):
        if not np.isfinite(exit_idx[i]):
            spans.append(None)
            continue
        start = i + 1
        end = int(exit_idx[i])
        if end < start:
            spans.append(None)
            continue
        spans.append((start, end))
        concurrency[start:end + 1] += 1

    concurrency[concurrency == 0] = 1.0  # avoid divide-by-zero
    weights = np.full(n, np.nan)
    for i, span in enumerate(spans):
        if span is None:
            continue
        start, end = span
        # Average uniqueness = mean(1 / concurrency) over the label's span.
        weights[i] = np.mean(1.0 / concurrency[start:end + 1])

    df["sample_weight"] = weights
    return df
