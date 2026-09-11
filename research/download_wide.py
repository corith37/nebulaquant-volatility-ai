"""Download a wide (400-500 name) universe into research/data_raw_wide/.

HANDOFF.md ranks "widen the universe to 400-500 names" as the single
highest-value next step: IR ~ IC * sqrt(breadth) is the one relationship in the
study with theory behind it, and the 482-name validation run produced less than
half the volatility for the same IC.

Writes one CSV per ticker in the exact yfinance layout src/panel.py expects
(Date, Adj Close, Close, High, Low, Open, Volume, Ticker), into a SEPARATE
directory so ../data/raw -- which the deployed momentum model reads -- is never
touched.

    python download_wide.py                      # all symbols in sp500_symbols.txt
    python download_wide.py --start 2015-01-01
    python download_wide.py --limit 50           # smoke test

NOTE ON BIAS: the symbol list is *today's* index membership, so this buys
breadth, not freedom from survivorship bias. Only a point-in-time constituent
file with delisted names fixes that (HANDOFF next step #2).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

HERE = Path(__file__).resolve().parent
OUT_COLS = ["Date", "Adj Close", "Close", "High", "Low", "Open", "Volume", "Ticker"]


def _one_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """Normalise a single ticker's yfinance frame to the on-disk layout."""
    if raw is None or raw.empty:
        return None
    df = raw.copy()
    df.columns.name = None
    df.index.name = "Date"
    df = df.reset_index()
    df["Ticker"] = ticker
    if "Adj Close" not in df.columns:
        return None
    df = df[[c for c in OUT_COLS if c in df.columns]].dropna(subset=["Adj Close", "Close"])
    return df if len(df) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=str(HERE / "sp500_symbols.txt"))
    ap.add_argument("--out", default=str(HERE / "data_raw_wide"))
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    syms = [s.strip() for s in Path(args.symbols).read_text().split() if s.strip()]
    if args.limit:
        syms = syms[: args.limit]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    ok, bad = [], []
    for i in range(0, len(syms), args.batch):
        chunk = syms[i: i + args.batch]
        try:
            data = yf.download(chunk, start=args.start, end=args.end, progress=False,
                               auto_adjust=False, group_by="ticker", threads=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! batch {i//args.batch} failed: {exc}", flush=True)
            bad.extend(chunk)
            continue
        for t in chunk:
            try:
                raw = data[t] if isinstance(data.columns, pd.MultiIndex) else data
                df = _one_frame(raw, t)
            except (KeyError, IndexError):
                df = None
            if df is None:
                bad.append(t)
                continue
            df.to_csv(out_dir / f"{t}.csv", index=False)
            ok.append(t)
        print(f"  batch {i//args.batch + 1}/{(len(syms)-1)//args.batch + 1}: "
              f"{len(ok)} ok, {len(bad)} failed  [{time.time()-t0:.0f}s]", flush=True)

    # The regime block needs VIX; reuse the copy the primary project already has.
    vix_src = HERE.parent / "data" / "raw" / "_VIX.csv"
    if vix_src.exists():
        pd.read_csv(vix_src).to_csv(out_dir / "_VIX.csv", index=False)
        print("copied _VIX.csv from ../data/raw")

    print(f"\nDownloaded {len(ok)}/{len(syms)} tickers -> {out_dir}  [{time.time()-t0:.0f}s]")
    if bad:
        print(f"failed ({len(bad)}): {', '.join(sorted(bad))}")
    Path(out_dir / "_failed.txt").write_text("\n".join(sorted(bad)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
