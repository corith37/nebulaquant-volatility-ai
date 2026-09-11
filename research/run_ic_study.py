"""Univariate IC study: every raw signal against forward residual return.

Restores the producer for `ic_full.csv`. HANDOFF section 6 cites that table for
42 signals, and export_charts.py reads it, but no script in the folder actually
wrote it -- so the section could not be regenerated from a clean checkout.

Reports, per signal, the daily cross-sectional Spearman IC against the forward
5-day RESIDUAL return, Newey-West corrected (overlapping 5-day labels make the
daily IC series strongly autocorrelated; a naive t-stat overstates significance
by roughly sqrt(h)).

Also runs the two qualifications HANDOFF section 6 rests on:
  * IC against forward RAW returns -- almost nothing survives, which is the
    observation that reframed the project around factor-neutral construction.
  * Reversal's IC split on the direction of the forward market return, showing
    it carries residual market exposure that momentum does not.

    python run_ic_study.py                       # primary universe
    python run_ic_study.py --data data_proc_wide/dataset.parquet --out out/ic_wide.csv
"""
from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, '.')

import numpy as np
import pandas as pd

from src.features import STOCK_FEATURES
from src.stats import daily_ic, ic_summary, newey_west_tstat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data_proc/dataset.parquet')
    ap.add_argument('--out', default='out/ic_full.csv')
    ap.add_argument('--lags', type=int, default=10)
    args = ap.parse_args()

    t0 = time.time()
    df = pd.read_parquet(args.data)
    print(f"{df.ticker.nunique()} names, {len(df):,} rows, "
          f"{df.date.min().date()}..{df.date.max().date()}")

    rows = []
    for c in STOCK_FEATURES:
        z = f"z_{c}"
        if z not in df.columns:
            continue
        s = ic_summary(df, z, tgt='y', lags=args.lags)
        s['signal'] = c
        rows.append(s)
    out = pd.DataFrame(rows).sort_values('ic_t_nw').reset_index(drop=True)
    out.to_csv(args.out, index=False)
    print(f"\n=== UNIVARIATE IC vs forward 5d RESIDUAL (NW lag {args.lags}) "
          f"[{time.time()-t0:.0f}s] ===")
    show = out[['signal', 'ic_mean', 'ic_t_nw', 'ic_ir', 'hit_rate']]
    print(pd.concat([show.head(8), show.tail(8)]).to_string(index=False))

    # --- qualification 1: raw returns ---------------------------------------
    df['y_raw'] = df.groupby('date')['fwd_ret_5'].rank(pct=True) - 0.5
    raw = []
    for c in ['rev_1', 'rev_3', 'rev_5', 'rev_5_sc', 'mom_252_21', 'mom_63',
              'beta_mkt', 'dist_52w_high', 'idio_vol_63']:
        z = f"z_{c}"
        if z not in df.columns:
            continue
        ic = daily_ic(df, z, 'y_raw')
        m, t = newey_west_tstat(ic.to_numpy(), lags=args.lags)
        raw.append(dict(signal=c, ic_raw=m, t_raw=t))
    print("\n=== same signals vs forward 5d RAW return ===")
    print(pd.DataFrame(raw).sort_values('t_raw').to_string(index=False))
    print("  (near-zero t across the board is why the strategy must be traded"
          " factor-neutral: the edge exists only in residual space)")

    # --- qualification 2: does the signal ride the market? -------------------
    fwd_mkt = df.groupby('date')['mkt_ret'].first().rolling(5).sum().shift(-5)
    print("\n=== IC conditioned on the direction of the forward market move ===")
    cond = []
    for c in ['rev_5', 'rev_1', 'mom_252_21']:
        z = f"z_{c}"
        if z not in df.columns:
            continue
        ic = daily_ic(df, z, 'y')
        up = ic[fwd_mkt.reindex(ic.index) > 0]
        dn = ic[fwd_mkt.reindex(ic.index) <= 0]
        cond.append(dict(signal=c, ic_mkt_up=float(np.nanmean(up)),
                         ic_mkt_down=float(np.nanmean(dn)),
                         corr_with_fwd_mkt=float(pd.Series(ic).corr(fwd_mkt.reindex(ic.index)))))
    print(pd.DataFrame(cond).to_string(index=False))
    print("  (a signal that flips sign with the market is partly an unhedged"
          " beta bet, not clean cross-sectional selection)")
    print(f"\nsaved {args.out}  [{time.time()-t0:.0f}s]")
    return 0


if __name__ == '__main__':
    sys.exit(main())
