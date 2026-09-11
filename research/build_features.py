"""Panel -> residuals -> features -> target, for ANY universe.

Same logic as build_dataset.py, but parameterised so the primary (76-name) and
wide (503-name) universes run through identical code. build_dataset.py is left
untouched so the archived numbers stay reproducible by the documented command.

    python build_features.py --in data_proc      --out data_proc/dataset.parquet
    python build_features.py --in data_proc_wide --out data_proc_wide/dataset.parquet \
                             --min-names 100 --exclude MNST

--n-pc sets the dimensionality of the factor space the TARGET is defined in.
House Rule #2: if you change it, the book's neutrality constraints must change
to match (--n-pc 4 pairs with portfolio.FACTOR_COLS as shipped).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, '.')

import numpy as np
import pandas as pd

from src.factors import residualize
from src.features import (REGIME_FEATURES, STOCK_FEATURES, add_regime,
                          cross_sectional_rank, make_target,
                          per_ticker_features)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='indir', default='data_proc',
                    help='directory holding panel.parquet + vix.parquet')
    ap.add_argument('--out', default=None, help='output parquet (default <in>/dataset.parquet)')
    ap.add_argument('--n-pc', type=int, default=4)
    ap.add_argument('--lookback', type=int, default=252)
    ap.add_argument('--step', type=int, default=21)
    ap.add_argument('--min-names', type=int, default=30,
                    help='drop dates thinner than this before cross-sectional ranking')
    ap.add_argument('--exclude', nargs='*', default=[],
                    help='tickers to drop (e.g. failed data-quality checks)')
    args = ap.parse_args()

    t0 = time.time()
    indir = Path(args.indir)
    out_path = Path(args.out) if args.out else indir / 'dataset.parquet'

    panel = pd.read_parquet(indir / 'panel.parquet')
    vix = pd.read_parquet(indir / 'vix.parquet')
    if args.exclude:
        n0 = panel.ticker.nunique()
        panel = panel[~panel.ticker.isin(args.exclude)].copy()
        print(f"excluded {args.exclude}: {n0} -> {panel.ticker.nunique()} tickers")
    print(f"panel {panel.shape}  {panel.ticker.nunique()} tickers")

    panel = residualize(panel, lookback=args.lookback, step=args.step, n_pc=args.n_pc)
    print(f"residualised to beta + {args.n_pc} PCs  [{time.time()-t0:.0f}s]  "
          f"resid non-null: {panel['resid'].notna().mean():.3f}")

    w = panel.pivot(index='date', columns='ticker', values='resid')
    rr = panel.pivot(index='date', columns='ticker', values='close').pct_change()
    cw = w.iloc[-1500:].corr().to_numpy()
    cr = rr.iloc[-1500:].corr().to_numpy()
    iu = np.triu_indices_from(cw, 1)
    print(f"mean pairwise corr  raw={np.nanmean(cr[iu]):.3f}  residual={np.nanmean(cw[iu]):.3f}")

    feats = pd.concat([per_ticker_features(g) for _, g in panel.groupby('ticker', sort=True)],
                      ignore_index=True)
    print(f"features built [{time.time()-t0:.0f}s] {feats.shape}")

    feats = add_regime(feats, vix)
    feats = cross_sectional_rank(feats, STOCK_FEATURES, min_names=args.min_names)
    feats = make_target(feats, horizon=5)
    for h in (3, 10):
        feats[f'y_{h}'] = feats.groupby('date')[f'fwd_resid_{h}'].rank(pct=True) - 0.5
        feats.loc[feats[f'fwd_resid_{h}'].isna(), f'y_{h}'] = np.nan

    need = [f"z_{c}" for c in STOCK_FEATURES] + REGIME_FEATURES + ['y']
    before = len(feats)
    feats = feats.dropna(subset=need)
    print(f"dropna {before} -> {len(feats)}   dates {feats.date.min().date()}..{feats.date.max().date()}")
    per_date = feats.groupby('date').size()
    print(f"names/date: min {per_date.min()} med {int(per_date.median())} max {per_date.max()}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out_path)
    print(f"saved {out_path}  [{time.time()-t0:.0f}s]")
    return 0


if __name__ == '__main__':
    sys.exit(main())
