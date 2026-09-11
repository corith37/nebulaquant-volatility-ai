"""Does a target/portfolio factor-space MISMATCH cost what House Rule #2 claims?

Why this test exists
--------------------
`build_val.py` residualises the 482-name validation universe to **6** PCs
(`n_pc=6`), but `run_val.py` builds the book with the default
`FACTOR_COLS` = beta + **4** PCs. So the validation portfolio carries unhedged
PC5/PC6 exposure -- exactly the failure mode D6 identifies as worth 0.89 Sharpe
points. That matters because the validation's "net Sharpe 1.224" is the headline
evidence for HANDOFF next-step #1 (widen the universe).

The validation raw file is not in the repo, so the mismatch is measured here on
the PRIMARY universe instead, where the data is present. One walk-forward on a
6-PC target, then the same predictions traded two ways:

  matched   : target orthogonal to beta+PC1..6, book neutral to beta+PC1..6
  mismatched: target orthogonal to beta+PC1..6, book neutral to beta+PC1..4
              (what run_val.py actually did)

The gap between them is the cost of the mismatch, and tells us whether the
validation Sharpe is flattered or penalised by the bug.

    python test_npc_match.py            # ~6 min
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, '.')

import numpy as np
import pandas as pd

from src.backtest import returns_matrix, simulate
from src.factors import residualize
from src.features import (REGIME_FEATURES, STOCK_FEATURES, add_regime,
                          cross_sectional_rank, make_target,
                          per_ticker_features)
from src.model import walk_forward
from src.portfolio import positions_from_weights, signal_weights
from src.stats import daily_ic, newey_west_tstat, perf_stats

N_PC = 6

t0 = time.time()
panel = pd.read_parquet('data_proc/panel.parquet')
vix = pd.read_parquet('data_proc/vix.parquet')

panel = residualize(panel, lookback=252, step=21, n_pc=N_PC)
print(f"residualised to beta + {N_PC} PCs  [{time.time()-t0:.0f}s]")

feats = pd.concat([per_ticker_features(g) for _, g in panel.groupby('ticker', sort=True)],
                  ignore_index=True)
feats = add_regime(feats, vix)
feats = cross_sectional_rank(feats, STOCK_FEATURES, min_names=30)
feats = make_target(feats, horizon=5)
zcols = [f"z_{c}" for c in STOCK_FEATURES]
feats = feats.dropna(subset=zcols + REGIME_FEATURES + ['y'])
print(f"dataset {feats.shape}  {feats.date.min().date()}..{feats.date.max().date()}"
      f"  [{time.time()-t0:.0f}s]")

feat_cols = zcols + REGIME_FEATURES
P = walk_forward(feats, feat_cols, target='y', horizon=5, embargo=5,
                 min_train_days=756, block_days=63)
M = feats.merge(P[['date', 'ticker', 'pred', 'pred_rank']], on=['date', 'ticker'])
ic = daily_ic(M, 'pred_rank')
m, t = newey_west_tstat(ic.to_numpy(), lags=20)
print(f"walk-forward done [{time.time()-t0:.0f}s]  OOS IC={m:+.4f} NW t={t:+.2f}")

R = returns_matrix(feats)
BETA = ['beta_mkt']
ARMS = [
    (BETA + [f'load_pc{k}' for k in range(1, N_PC + 1)], f'matched      (book neutral to beta+{N_PC} PCs)'),
    (BETA + [f'load_pc{k}' for k in range(1, 5)],        'MISMATCHED   (book neutral to beta+4 PCs)  <- run_val.py'),
    (BETA + [f'load_pc{k}' for k in range(1, 3)],        'under-hedged (book neutral to beta+2 PCs)'),
    (BETA,                                               'beta only'),
    (['__none__'],                                       'dollar-neutral only'),
]

print(f"\n=== TARGET IS ORTHOGONAL TO beta + {N_PC} PCs; ONLY THE BOOK CHANGES ===")
rows = []
for fc, lab in ARMS:
    W = signal_weights(M, 'pred_rank', factor_cols=fc)
    r = simulate(positions_from_weights(W), R, cost_bps=2.5)
    s = perf_stats(r['net'])
    rows.append(dict(arm=lab, sharpe=s['sharpe'], ann=s['ann_return'],
                     vol=s['ann_vol'], t=s['t_stat_nw']))
    print(f"  {lab:<52} Sharpe={s['sharpe']:+.3f}  ann={s['ann_return']:+.2%}  "
          f"vol={s['ann_vol']:.2%}  t={s['t_stat_nw']:+.2f}")

out = pd.DataFrame(rows)
out.to_csv('out/npc_match.csv', index=False)
gap = out.sharpe.iloc[0] - out.sharpe.iloc[1]
print(f"\nCOST OF THE run_val.py MISMATCH: {gap:+.3f} Sharpe "
      f"({'matched is better -> validation number is UNDERSTATED' if gap > 0 else 'mismatched scored higher -> validation number is FLATTERED'})")
print(f"reference: archived 4-PC target / 4-PC book = +0.738")
print("saved out/npc_match.csv")
