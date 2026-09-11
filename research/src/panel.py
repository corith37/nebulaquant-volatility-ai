"""Build a clean, point-in-time daily panel from raw per-ticker OHLCV files.

Design notes
------------
* yfinance-style raw files carry BOTH ``Close`` (split-adjusted only) and
  ``Adj Close`` (split + dividend adjusted). Total returns must come from
  ``Adj Close``; intraday shape (Open/High/Low) is split-adjusted only, so we
  rescale OHL onto the Adj Close basis with the per-bar dividend factor. Skipping
  this step injects a spurious ~-0.5% "overnight gap" on every ex-dividend date,
  which is fatal for an overnight/intraday decomposition.
* Everything here is backward-looking. No column may be computed with
  information dated after the row's own ``date``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_COLS = ["Date", "Adj Close", "Close", "High", "Low", "Open", "Volume", "Ticker"]


def load_ticker(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in RAW_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Dividend factor: adj_close = close * cumprod(div factors). The ratio is a
    # smooth, non-decreasing-in-time multiplier, so applying it to O/H/L puts
    # every price on one consistent total-return basis.
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = df["Adj Close"] / df["Close"]
    factor = factor.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    out = pd.DataFrame({
        "date": df["Date"],
        "ticker": df["Ticker"].astype(str).str.replace("^", "", regex=False),
        "open": df["Open"] * factor,
        "high": df["High"] * factor,
        "low": df["Low"] * factor,
        "close": df["Adj Close"],
        "raw_close": df["Close"],
        "volume": df["Volume"].astype(float),
    })
    return out


def build_panel(raw_dir: Path, exclude: tuple[str, ...] = ("VIX",)) -> pd.DataFrame:
    frames = []
    for path in sorted(Path(raw_dir).glob("*.csv")):
        stem = path.stem.lstrip("_")
        if stem in exclude:
            continue
        try:
            frames.append(load_ticker(path))
        except Exception as exc:  # noqa: BLE001
            print(f"  ! skipped {path.name}: {exc}")
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    return panel


def load_vix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return pd.DataFrame({"date": df["Date"], "vix": df["Close"]}).sort_values("date")


# ---------------------------------------------------------------------------
# Quality assurance
# ---------------------------------------------------------------------------

def _oscillation_flags(close: pd.Series, ret: pd.Series) -> tuple[int, int]:
    """Detect a price series that flips between two levels (bad split adjustment).

    A genuine earnings gap moves the level once and stays there. A misapplied
    split makes the series oscillate: MNST in the 503-name S&P pull alternated
    between ~47 and ~93 for weeks, which the single-bar `suspect_split` check
    below counts but does not distinguish from a real +40% M&A day. Two tests:

    * `n_oscillate` - bars more than 30% away from their own CENTRED 5-day
      median. A one-off jump barely registers (the median follows it); a
      flip-flop puts every other bar far from the local centre.
    * `round_trips` - a >35% move reversed by a >30% opposite move within three
      bars. Real events do not round-trip like that.
    """
    med = close.rolling(5, center=True, min_periods=3).median()
    n_osc = int(((close / med - 1).abs() > 0.30).sum())

    r = ret.to_numpy()
    rt = 0
    for i in np.where(np.abs(r) > 0.35)[0]:
        w = r[i + 1: i + 4]
        if len(w) and np.any(np.sign(w) * np.sign(r[i]) < 0) and np.any(np.abs(w) > 0.30):
            rt += 1
    return n_osc, rt


def qa_report(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker data-integrity checks. Anything flagged here would silently
    corrupt a backtest, so we surface it before a single feature is built."""
    rows = []
    for tkr, g in panel.groupby("ticker", sort=True):
        g = g.sort_values("date")
        ret = g["close"].pct_change()
        # An unadjusted split shows up as a huge single-bar return that is not
        # echoed by a comparable move in the high/low range.
        big = ret.abs() > 0.35
        rng = (g["high"] - g["low"]) / g["close"]
        suspect_split = int((big & (rng < 0.15)).sum())
        n_osc, round_trips = _oscillation_flags(g["close"], ret.fillna(0.0))
        rows.append({
            "ticker": tkr,
            "n": len(g),
            "start": g["date"].min().date(),
            "end": g["date"].max().date(),
            "nan_close": int(g["close"].isna().sum()),
            "nonpos_close": int((g["close"] <= 0).sum()),
            "zero_vol_days": int((g["volume"] <= 0).sum()),
            "dup_dates": int(g["date"].duplicated().sum()),
            "ohlc_violations": int(
                ((g["high"] < g["low"])
                 | (g["high"] < g["close"] - 1e-9)
                 | (g["low"] > g["close"] + 1e-9)).sum()
            ),
            "abs_ret_gt_35pct": int(big.sum()),
            "suspect_unadj_split": suspect_split,
            "n_oscillate": n_osc,
            "round_trips": round_trips,
            "corrupt": int(n_osc >= 3 or round_trips >= 2),
            "max_abs_ret": float(ret.abs().max()),
        })
    return pd.DataFrame(rows)
