"""Data downloading via yfinance.

Downloads daily OHLCV bars for a list of tickers and saves one CSV per ticker
to `data/raw/`. Failures on individual tickers are logged and skipped, not raised.
"""
from __future__ import annotations

from datetime import datetime, time
from pathlib import Path
from typing import Iterable, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from src.utils import setup_logger

logger = setup_logger("nebulaquant.data_loader")

_EASTERN = ZoneInfo("America/New_York")
# US equities close at 16:00 ET; allow a short buffer for the final print to settle.
_SETTLED_AFTER = time(16, 15)


def safe_filename(ticker: str) -> str:
    """Map a ticker symbol to a filesystem-safe base name (e.g. ^VIX -> _VIX)."""
    return ticker.replace("^", "_").replace("/", "_").replace("\\", "_")


def drop_partial_bar(df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    """Drop a still-forming bar for the current session.

    Run intraday, yfinance returns a row for today whose Close is the *live* price
    and whose Volume is only the session-to-date total. Feeding that into daily
    features makes signals flicker through the day and depresses relative-volume,
    so we keep only settled sessions. After 16:15 ET today's bar is final and kept.
    """
    if df.empty or "Date" not in df.columns:
        return df

    now_et = datetime.now(_EASTERN)
    if now_et.time() >= _SETTLED_AFTER:
        return df

    today_et = pd.Timestamp(now_et.date())
    last = pd.to_datetime(df["Date"].iloc[-1])
    if last.normalize() == today_et:
        logger.info(f"[{ticker}] dropping unsettled {today_et.date()} bar (market still open)")
        return df.iloc[:-1]
    return df


def download_ticker(
    ticker: str,
    start: str,
    end: Optional[str] = None,
    out_dir: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    """Download a single ticker. Returns the dataframe (and writes CSV if out_dir set)."""
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
            group_by="column",
        )
    except Exception as e:
        logger.error(f"[{ticker}] download failed: {e}")
        return None

    if df is None or df.empty:
        logger.warning(f"[{ticker}] no data returned")
        return None

    # yfinance sometimes returns a MultiIndex column even for a single ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Clear MultiIndex level names (e.g., "Price") so they don't leak into CSV.
    df.columns.name = None
    # Force the date index to be named "Date" - newer yfinance may leave it
    # unnamed after the MultiIndex flatten, which makes reset_index() emit a
    # column literally named "index".
    df.index.name = "Date"

    df = df.reset_index()
    df = drop_partial_bar(df, ticker)
    df["Ticker"] = ticker

    # Standardize column names if yfinance returns "Adj Close" etc.
    expected = {"Date", "Open", "High", "Low", "Close", "Volume", "Ticker"}
    missing = expected - set(df.columns)
    if missing:
        logger.warning(f"[{ticker}] missing columns: {missing}")

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{safe_filename(ticker)}.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"[{ticker}] saved {len(df)} rows -> {out_path}")

    return df


def download_universe(
    tickers: Iterable[str],
    start: str,
    end: Optional[str] = None,
    out_dir: Optional[Path] = None,
) -> List[str]:
    """Download a list of tickers. Returns the list of successful tickers."""
    successes: List[str] = []
    for t in tickers:
        df = download_ticker(t, start=start, end=end, out_dir=out_dir)
        if df is not None and not df.empty:
            successes.append(t)
    logger.info(f"Downloaded {len(successes)}/{len(list(tickers))} tickers successfully")
    return successes


def load_raw(ticker: str, raw_dir: Path) -> pd.DataFrame:
    """Load a raw ticker CSV from disk.

    Tolerant of legacy CSVs where the date column was saved under a different
    name (``index``, ``Datetime``, ``Price``, etc.) due to yfinance MultiIndex
    quirks.
    """
    path = Path(raw_dir) / f"{safe_filename(ticker)}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found for {ticker}: {path}")

    df = pd.read_csv(path)
    if "Date" not in df.columns:
        for cand in ("index", "Datetime", "Price", "Unnamed: 0"):
            if cand in df.columns:
                df = df.rename(columns={cand: "Date"})
                break
    if "Date" not in df.columns:
        raise ValueError(
            f"[{ticker}] no Date column found in {path}; columns: {list(df.columns)}"
        )
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    return df.sort_values("Date").reset_index(drop=True)
