"""HANDOFF next-steps #1 and #3, run together on the 502-name universe.

#1 BREADTH. The 482-name validation run is the sole evidence that widening the
universe lifts net Sharpe from 0.74 to 1.22, but it is not a controlled
comparison: it used a 6-PC target against a 4-PC book (House Rule #2 violation),
2 seeds not 3, min_train 252 not 756, and blocks of 42 not 63. This run holds
EVERY protocol knob identical to the primary and changes only the universe, so
the difference is attributable to breadth alone.

#3 LEAN FEATURE SET -- PRE-REGISTERED. The ablation on the primary universe
found that dropping the 10 risk/vol features RAISES net Sharpe 0.739 -> 0.823.
D8 correctly refused to adopt that, because it was a choice made after seeing
out-of-sample results. This is the independent test it asked for.

    PRE-REGISTRATION, written before the wide universe was ever scored
    ----------------------------------------------------------------
    H1 (breadth):  full-feature model on 502 names beats the 76-name model's
                   OOS IC of +0.0240 on a t-stat basis, and lands annual vol
                   below 3% at the same gross exposure.
                   Acceptance (HANDOFF): OOS IC >= 0.020 with t >= 3.
    H2 (lean):     dropping the risk/vol block beats the full model on net
                   Sharpe on this universe, which neither variant has seen.
                   Adopt the lean set ONLY if H2 holds here. A win of any size
                   counts, but it is reported with its t-stat, not alone.

    Both arms are scored once. Whatever comes back is the number of record.

    python run_wide.py            # ~40 min
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, '.')

import numpy as np
import pandas as pd

from src.backtest import returns_matrix, simulate
from src.features import REGIME_FEATURES, STOCK_FEATURES
from src.model import walk_forward
from src.portfolio import FACTOR_COLS, positions_from_weights, signal_weights
from src.stats import daily_ic, newey_west_tstat, perf_stats

# Protocol frozen to match the primary run exactly (train_wf.py + run_backtest2.py).
PROTOCOL = dict(target='y', horizon=5, embargo=5, min_train_days=756,
                block_days=63, seeds=(7, 17, 27))
COST_BPS = 2.5
RISK_VOL_BLOCK = ['idio_vol_21', 'idio_vol_63', 'tot_vol_21', 'vol_ratio_5_63',
                  'park_vol_21', 'vov_63', 'semidev_21', 'skew_63', 'kurt_63', 'beta_mkt']

t0 = time.time()
df = pd.read_parquet('data_proc_wide/dataset.parquet')
R = returns_matrix(df)
print(f"universe: {df.ticker.nunique()} names, {len(df):,} rows, "
      f"{df.date.min().date()}..{df.date.max().date()}")
print(f"protocol (identical to primary): {PROTOCOL}, cost {COST_BPS}bp, "
      f"book neutral to {FACTOR_COLS}")

ALL_FEATS = [f"z_{c}" for c in STOCK_FEATURES] + REGIME_FEATURES
LEAN_FEATS = [f for f in ALL_FEATS if f not in {f"z_{c}" for c in RISK_VOL_BLOCK}]

rows = []
for feats, label in [(ALL_FEATS, 'full'), (LEAN_FEATS, 'lean (no risk/vol)')]:
    print(f"\n--- {label}: {len(feats)} features ---", flush=True)
    P = walk_forward(df, feats, **PROTOCOL)
    M = df.merge(P[['date', 'ticker', 'pred', 'pred_rank']], on=['date', 'ticker'])
    ic = daily_ic(M, 'pred_rank')
    ic_m, ic_t = newey_west_tstat(ic.to_numpy(), lags=20)

    W = signal_weights(M, 'pred_rank')
    res = simulate(positions_from_weights(W), R, cost_bps=COST_BPS)
    s = perf_stats(res['net'])
    gross = perf_stats(res['gross'])['sharpe']
    H = positions_from_weights(W)

    rows.append(dict(
        variant=label, n_feat=len(feats), oos_rows=len(M), blocks=P.block_start.nunique(),
        ic=ic_m, ic_t=ic_t, ic_ir=ic_m / np.nanstd(ic), ic_hit=float(np.nanmean(ic > 0)),
        gross_sharpe=gross, net_sharpe=s['sharpe'], ann_ret=s['ann_return'],
        ann_vol=s['ann_vol'], max_dd=s['max_dd'], t_nw=s['t_stat_nw'],
        ann_turn=res['turnover'].mean() * 252,
        gross_exp=H.abs().sum(axis=1).mean(),
        n_pos_50bp=(H.abs() > 0.005).sum(axis=1).mean(),
        max_pos=H.abs().max().max(),
    ))
    r = rows[-1]
    print(f"  OOS IC={r['ic']:+.4f} (NW t={r['ic_t']:+.2f}, IR={r['ic_ir']:+.3f}, "
          f"hit={r['ic_hit']:.3f})  blocks={r['blocks']}  rows={r['oos_rows']:,}")
    print(f"  net Sharpe={r['net_sharpe']:+.3f} (gross {r['gross_sharpe']:+.3f})  "
          f"ann={r['ann_ret']:+.2%}  vol={r['ann_vol']:.2%}  maxDD={r['max_dd']:+.2%}  "
          f"t={r['t_nw']:+.2f}")
    print(f"  turnover={r['ann_turn']:.1f}x  gross exp={r['gross_exp']:.3f}  "
          f"positions>0.5%={r['n_pos_50bp']:.1f}  largest={r['max_pos']:.3f}",
          flush=True)
    M[['date', 'ticker', 'pred_rank']].to_parquet(
        f"data_proc_wide/preds_{'full' if label == 'full' else 'lean'}.parquet")
    W.to_parquet(f"data_proc_wide/W_{'full' if label == 'full' else 'lean'}.parquet")

out = pd.DataFrame(rows)
out.to_csv('out/wide_universe.csv', index=False)

PRIMARY = dict(ic=0.0240, ic_t=3.86, net_sharpe=0.738, ann_vol=0.0418, ann_ret=0.0309)
full = out.iloc[0]
print("\n" + "=" * 72)
print("H1  BREADTH  (76 names -> 502 names, protocol held identical)")
print(f"  OOS IC       {PRIMARY['ic']:+.4f} (t {PRIMARY['ic_t']:+.2f})  ->  "
      f"{full['ic']:+.4f} (t {full['ic_t']:+.2f})")
print(f"  net Sharpe   {PRIMARY['net_sharpe']:+.3f}  ->  {full['net_sharpe']:+.3f}")
print(f"  annual vol   {PRIMARY['ann_vol']:.2%}  ->  {full['ann_vol']:.2%}")
print(f"  annual ret   {PRIMARY['ann_ret']:+.2%}  ->  {full['ann_ret']:+.2%}")
passed = (full['ic'] >= 0.020) and (full['ic_t'] >= 3.0)
print(f"  HANDOFF acceptance (IC >= 0.020 and t >= 3): "
      f"{'PASS' if passed else 'FAIL'}; vol below 3%: "
      f"{'PASS' if full['ann_vol'] < 0.03 else 'FAIL'}")

lean = out.iloc[1]
print("\nH2  LEAN FEATURE SET  (pre-registered; decides D8)")
print(f"  full  net Sharpe {full['net_sharpe']:+.3f}  IC {full['ic']:+.4f} (t {full['ic_t']:+.2f})")
print(f"  lean  net Sharpe {lean['net_sharpe']:+.3f}  IC {lean['ic']:+.4f} (t {lean['ic_t']:+.2f})")
print(f"  VERDICT: {'ADOPT lean' if lean['net_sharpe'] > full['net_sharpe'] else 'KEEP full'} "
      f"(primary-universe ablation had predicted lean wins: 0.739 -> 0.823)")
print("=" * 72)
print("saved out/wide_universe.csv")
