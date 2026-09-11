import sys,time; sys.path.insert(0,'.')
import pandas as pd, numpy as np
from src.model import walk_forward
from src.stats import daily_ic, newey_west_tstat
from src.features import STOCK_FEATURES, REGIME_FEATURES

t0=time.time()
df = pd.read_parquet('data_proc/dataset.parquet')
feat = [f"z_{c}" for c in STOCK_FEATURES] + REGIME_FEATURES
print(f"{len(feat)} features, {len(df):,} rows")

P = walk_forward(df, feat, target='y', horizon=5, embargo=5,
                 min_train_days=756, block_days=63)
print(f"walk-forward done [{time.time()-t0:.0f}s]  OOS rows={len(P):,} "
      f"dates {P.date.min().date()}..{P.date.max().date()}  blocks={P.block_start.nunique()}")

M = df.merge(P[['date','ticker','pred','pred_rank']], on=['date','ticker'])
M.to_parquet('data_proc/oos_preds.parquet')

print("\n=== OUT-OF-SAMPLE IC (model prediction vs fwd 5d residual rank) ===")
for tgt,lab in [('y','fwd 5d residual'),('y_3','fwd 3d residual'),('y_10','fwd 10d residual')]:
    ic=daily_ic(M,'pred_rank',tgt); m,t=newey_west_tstat(ic.to_numpy(),lags=20)
    print(f"  {lab:>18}: IC={m:+.4f}  NW t={t:+.2f}  IR={m/np.nanstd(ic):+.3f}  hit={np.nanmean(ic>0):.3f}")
M['y_raw']=M.groupby('date')['fwd_ret_5'].rank(pct=True)-0.5
ic=daily_ic(M,'pred_rank','y_raw'); m,t=newey_west_tstat(ic.to_numpy(),lags=20)
print(f"  {'fwd 5d RAW':>18}: IC={m:+.4f}  NW t={t:+.2f}")

print("\n=== IC by year ===")
M['yr']=M.date.dt.year
for yr,g in M.groupby('yr'):
    ic=daily_ic(g,'pred_rank'); m,t=newey_west_tstat(ic.to_numpy(),lags=10)
    print(f"  {yr}: IC={m:+.4f} t={t:+.2f} n={len(g):,}")

print("\n=== IC by VIX tercile ===")
reg=M.groupby('date')['vix_pct252'].first()
M['vb']=pd.qcut(M.date.map(reg),3,labels=['lowVIX','mid','highVIX'])
for b,g in M.groupby('vb',observed=True):
    ic=daily_ic(g,'pred_rank'); m,t=newey_west_tstat(ic.to_numpy(),lags=20)
    print(f"  {b}: IC={m:+.4f} t={t:+.2f}")

print("\n=== top 20 feature importances ===")
print(P.attrs['importance'].head(20).to_string())
