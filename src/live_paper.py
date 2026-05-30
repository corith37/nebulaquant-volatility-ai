"""Forward paper-trading ledger for live momentum picks (fake cash, no orders).

This is distinct from ``src.paper``, which *replays* the validated strategy over
historical bars to produce an instant equity curve. This module supports the
other half of paper trading: tracking the picks you take **today** going forward.

Each pick is recorded with its entry price and a target exit date (entry + N
trading days, matching the deployed ~20-day hold). On later visits the open
positions are marked to market against the latest close so you can watch the
trade play out in real time, and any position past its target exit date is
flagged "due" (the strategy would sell it on its time exit).

NOTHING here places a real order or contacts a broker. It is a local CSV ledger.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

LEDGER_NAME = "live_picks.csv"
LEDGER_COLS = [
    "logged_at", "ticker", "entry_date", "entry_price", "shares", "alloc",
    "hold_days", "target_exit_date", "status",
    "exit_date", "exit_price", "realized_pnl",
]
# Text columns must stay object dtype: when the ledger has no closed rows yet,
# CSV round-trips an all-empty column to float64, and pandas 2.x then refuses to
# write a date/string into it. Forcing object on load keeps assignment valid.
_TEXT_COLS = ["logged_at", "ticker", "entry_date", "target_exit_date", "status", "exit_date"]


def ledger_path(paper_dir) -> Path:
    return Path(paper_dir) / LEDGER_NAME


def load_ledger(paper_dir) -> pd.DataFrame:
    """Load the live-picks ledger, returning an empty (typed) frame if absent."""
    p = ledger_path(paper_dir)
    if not p.exists():
        return pd.DataFrame(columns=LEDGER_COLS)
    df = pd.read_csv(p)
    for c in LEDGER_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[LEDGER_COLS].copy()
    for c in _TEXT_COLS:
        df[c] = df[c].astype("object")
    return df


def save_ledger(paper_dir, df: pd.DataFrame) -> None:
    p = ledger_path(paper_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


def add_picks(paper_dir, picks: List[dict]) -> tuple[int, int]:
    """Append new open picks. A ticker already open is skipped (no double-buy).

    Returns ``(added, skipped)``.
    """
    df = load_ledger(paper_dir)
    open_tickers = set(df.loc[df["status"] == "open", "ticker"].astype(str))
    rows = []
    skipped = 0
    for p in picks:
        tk = str(p["ticker"])
        if tk in open_tickers:
            skipped += 1
            continue
        rows.append({
            "logged_at": date.today().isoformat(),
            "ticker": tk,
            "entry_date": str(p["entry_date"]),
            "entry_price": round(float(p["entry_price"]), 4),
            "shares": round(float(p["shares"]), 6),  # float: supports fractional/notional fills
            "alloc": round(float(p.get("alloc", 0.0)), 2),
            "hold_days": int(p["hold_days"]),
            "target_exit_date": str(p["target_exit_date"]),
            "status": "open",
            "exit_date": pd.NA,
            "exit_price": pd.NA,
            "realized_pnl": pd.NA,
        })
        open_tickers.add(tk)
    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        save_ledger(paper_dir, df)
    return len(rows), skipped


def _latest_close(processed_dir, ticker) -> Optional[tuple[float, str]]:
    """Return (last close, last date str) from the ticker's processed features."""
    p = Path(processed_dir) / f"{ticker}_features.csv"
    if not p.exists():
        return None
    try:
        d = pd.read_csv(p, usecols=["Date", "Close"]).dropna()
    except (ValueError, KeyError):
        return None
    if d.empty:
        return None
    return float(d["Close"].iloc[-1]), str(pd.to_datetime(d["Date"].iloc[-1]).date())


def mark_to_market(ledger: pd.DataFrame, processed_dir, today=None) -> pd.DataFrame:
    """Mark open positions to the latest close; add P&L and days-to-target."""
    today = pd.to_datetime(today) if today is not None else pd.Timestamp(date.today())
    openpos = ledger[ledger["status"] == "open"].copy()
    if openpos.empty:
        return openpos
    last_close, last_date = [], []
    for t in openpos["ticker"]:
        info = _latest_close(processed_dir, t)
        last_close.append(info[0] if info else float("nan"))
        last_date.append(info[1] if info else None)
    openpos["entry_price"] = openpos["entry_price"].astype(float)
    openpos["shares"] = openpos["shares"].astype(float)
    openpos["last_close"] = last_close
    openpos["last_date"] = last_date
    openpos["market_value"] = openpos["last_close"] * openpos["shares"]
    openpos["unrealized_pnl"] = (openpos["last_close"] - openpos["entry_price"]) * openpos["shares"]
    openpos["unrealized_pct"] = (openpos["last_close"] / openpos["entry_price"]) - 1.0
    tex = pd.to_datetime(openpos["target_exit_date"], errors="coerce")
    openpos["days_to_target"] = (tex - today).dt.days
    openpos["due"] = openpos["days_to_target"] <= 0
    return openpos.reset_index(drop=True)


def close_positions(paper_dir, tickers: Iterable[str], processed_dir, today=None) -> int:
    """Close the given open tickers at the latest close; record realized P&L."""
    today = today or date.today()
    want = set(map(str, tickers))
    df = load_ledger(paper_dir)
    n = 0
    for i, row in df.iterrows():
        if row["status"] == "open" and str(row["ticker"]) in want:
            info = _latest_close(processed_dir, row["ticker"])
            px = info[0] if info else float(row["entry_price"])
            df.at[i, "status"] = "closed"
            df.at[i, "exit_date"] = pd.to_datetime(today).date().isoformat()
            df.at[i, "exit_price"] = round(px, 4)
            df.at[i, "realized_pnl"] = round((px - float(row["entry_price"])) * float(row["shares"]), 2)
            n += 1
    if n:
        save_ledger(paper_dir, df)
    return n
