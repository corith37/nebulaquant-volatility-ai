"""Stage 0: raw per-ticker CSVs -> data_proc/panel.parquet + vix.parquet.

This step was referenced by HANDOFF.md ("build_dataset.py reads ../data/raw/*.csv")
but no script actually produced the parquet intermediates that build_dataset.py
opens, so the pipeline could not be run from a clean checkout. This restores it.

Reads ../data/raw/*.csv (yfinance-style: Date, Adj Close, Close, High, Low, Open,
Volume, Ticker), rescales OHL onto the dividend-adjusted basis, runs the QA
report, and writes the two parquet files every later stage expects.

    python build_panel.py                 # default universe (../data/raw)
    python build_panel.py --raw path/to   # alternate raw directory
    python build_panel.py --exclude VIX SPY QQQ
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, '.')

import pandas as pd

from src.panel import build_panel, load_vix, qa_report

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default=str(HERE.parent / 'data' / 'raw'),
                    help='directory of per-ticker OHLCV CSVs')
    ap.add_argument('--vix', default=None,
                    help='VIX csv (default: <raw>/_VIX.csv)')
    ap.add_argument('--out', default='data_proc', help='output directory')
    ap.add_argument('--exclude', nargs='*', default=['VIX'],
                    help='ticker stems to drop from the tradeable panel')
    ap.add_argument('--qa', action='store_true', help='print the full QA table')
    args = ap.parse_args()

    t0 = time.time()
    raw_dir = Path(args.raw)
    if not raw_dir.exists():
        print(f"! raw directory not found: {raw_dir}")
        return 1
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = build_panel(raw_dir, exclude=tuple(args.exclude))
    print(f"panel {panel.shape}  {panel.ticker.nunique()} tickers  "
          f"{panel.date.min().date()}..{panel.date.max().date()}  [{time.time()-t0:.0f}s]")

    qa = qa_report(panel)

    # Hard failures: a price series that oscillates between two levels is a
    # misapplied split adjustment, not a market event. It must be excluded, not
    # merely noted -- its returns corrupt the market factor for every name.
    corrupt = qa[qa.corrupt == 1]
    if not corrupt.empty:
        print("!! CORRUPT PRICE SERIES -- exclude these before building features:")
        print(corrupt[['ticker', 'n', 'n_oscillate', 'round_trips',
                       'max_abs_ret']].to_string(index=False))
        print(f"   re-run with:  --exclude {' '.join(corrupt.ticker)}\n"
              f"   and pass the same list to build_features.py")

    flags = qa[(qa.nan_close > 0) | (qa.nonpos_close > 0) | (qa.dup_dates > 0)
               | (qa.ohlc_violations > 0) | (qa.suspect_unadj_split > 0)]
    flags = flags[flags.corrupt == 0]
    if flags.empty and corrupt.empty:
        print("QA clean: no NaN/nonpositive closes, no duplicate dates, "
              "no OHLC violations, no suspected unadjusted splits, no oscillation")
    elif not flags.empty:
        print("QA FLAGS -- inspect before trusting any downstream number:")
        print(flags[['ticker', 'n', 'nan_close', 'dup_dates', 'ohlc_violations',
                     'suspect_unadj_split', 'max_abs_ret']].to_string(index=False))
    big = qa[qa.abs_ret_gt_35pct > 0][['ticker', 'abs_ret_gt_35pct', 'max_abs_ret']]
    if not big.empty:
        print(f"single-day moves >35% (verify these are real earnings reactions):\n"
              f"{big.to_string(index=False)}")
    if args.qa:
        print(qa.to_string(index=False))

    vix_path = Path(args.vix) if args.vix else raw_dir / '_VIX.csv'
    if not vix_path.exists():
        print(f"! VIX file not found: {vix_path} (regime features need it)")
        return 1
    vix = load_vix(vix_path)
    print(f"vix {vix.shape}  {vix.date.min().date()}..{vix.date.max().date()}")

    panel.to_parquet(out_dir / 'panel.parquet')
    vix.to_parquet(out_dir / 'vix.parquet')
    qa.to_csv(out_dir / 'qa_report.csv', index=False)
    print(f"saved {out_dir}/panel.parquet, vix.parquet, qa_report.csv  [{time.time()-t0:.0f}s]")
    return 0


if __name__ == '__main__':
    sys.exit(main())
