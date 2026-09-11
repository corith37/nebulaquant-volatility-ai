import sys,time; sys.path.insert(0,'.')
import pandas as pd, numpy as np
from src.model import walk_forward
from src.portfolio import signal_weights, positions_from_weights
from src.backtest import simulate, returns_matrix
from src.stats import daily_ic, newey_west_tstat, perf_stats
from src.features import STOCK_FEATURES, REGIME_FEATURES
t0=time.time()
V=pd.read_parquet('data_proc/val_dataset.parquet')
feat=[f"z_{c}" for c in STOCK_FEATURES]+REGIME_FEATURES
print(f"VALIDATION UNIVERSE: {V.ticker.nunique()} names, {len(V):,} rows, "
      f"{V.date.min().date()}..{V.date.max().date()}")
P=walk_forward(V,feat,target='y',horizon=5,embargo=5,min_train_days=252,block_days=42,seeds=(7,17))
print(f"walk-forward done [{time.time()-t0:.0f}s] OOS rows={len(P):,} blocks={P.block_start.nunique()}")
MV=V.merge(P[['date','ticker','pred','pred_rank']],on=['date','ticker'])
ic=daily_ic(MV,'pred_rank'); m,t=newey_west_tstat(ic.to_numpy(),lags=20)
print(f"\nOOS IC vs fwd 5d residual: IC={m:+.4f} NW t={t:+.2f} IR={m/np.nanstd(ic):+.3f} hit={np.nanmean(ic>0):.3f}")
RV=returns_matrix(V)
for c in [0,2.5,5]:
    W=signal_weights(MV,'pred_rank'); r=simulate(positions_from_weights(W),RV,cost_bps=c)
    s=perf_stats(r['net'])
    print(f"  factor-neutral portfolio @{c:>4}bp: Sharpe={s['sharpe']:+.3f} ann={s['ann_return']:+.2%} "
          f"t={s['t_stat_nw']:+.2f} vol={s['ann_vol']:.2%} maxdd={s['max_dd']:+.2%}")
print("\n=== single-signal ICs in the 505-name universe (NW t, lag 20) ===")
rows=[]
for c in ['rev_1','rev_3','rev_5','rev_5_sc','mom_252_21','mom_63','beta_mkt','gap_abs_5',
          'clv_1','idio_vol_63','ln_dollar_vol','amihud_21','on_intra_spread_21','dist_52w_high']:
    ic=daily_ic(V,f"z_{c}"); m,t=newey_west_tstat(ic.to_numpy(),lags=20)
    rows.append(dict(signal=c,ic=m,t=t))
print(pd.DataFrame(rows).sort_values('t').to_string(index=False))
MV[['date','ticker','pred_rank']].to_parquet('data_proc/val_preds.parquet')
