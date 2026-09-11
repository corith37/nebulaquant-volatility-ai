import sys, time; sys.path.insert(0,'.')
import pandas as pd, numpy as np
from src.factors import residualize
from src.features import (per_ticker_features, add_regime, cross_sectional_rank,
                          make_target, STOCK_FEATURES, REGIME_FEATURES)

t0=time.time()
panel = pd.read_parquet('data_proc/panel.parquet')
vix   = pd.read_parquet('data_proc/vix.parquet')
print("panel", panel.shape)

panel = residualize(panel, lookback=252, step=21, n_pc=4)
print(f"residualized  [{time.time()-t0:.0f}s]  resid non-null: {panel['resid'].notna().mean():.3f}")

# sanity: residual should have far less common variation than raw returns
w = panel.pivot(index='date',columns='ticker',values='resid')
rr = panel.pivot(index='date',columns='ticker',values='close').pct_change()
cw = w.iloc[-1500:].corr().to_numpy(); cr = rr.iloc[-1500:].corr().to_numpy()
iu = np.triu_indices_from(cw,1)
print(f"mean pairwise corr  raw={np.nanmean(cr[iu]):.3f}  residual={np.nanmean(cw[iu]):.3f}")

feats = pd.concat([per_ticker_features(g) for _, g in panel.groupby('ticker', sort=True)], ignore_index=True)
print(f"features built [{time.time()-t0:.0f}s]", feats.shape)

feats = add_regime(feats, vix)
feats = cross_sectional_rank(feats, STOCK_FEATURES, min_names=30)
feats = make_target(feats, horizon=5)
for h in (3,10):
    feats[f'y_{h}'] = feats.groupby('date')[f'fwd_resid_{h}'].rank(pct=True)-0.5
    feats.loc[feats[f'fwd_resid_{h}'].isna(), f'y_{h}'] = np.nan

zcols = [f"z_{c}" for c in STOCK_FEATURES]
need = zcols + REGIME_FEATURES + ['y']
before=len(feats)
feats = feats.dropna(subset=need)
print(f"dropna {before} -> {len(feats)}   dates {feats.date.min().date()}..{feats.date.max().date()}")
print("names/date: min %d med %d max %d" % tuple(feats.groupby('date').size().describe()[['min','50%','max']]))
feats.to_parquet('data_proc/dataset.parquet')
print(f"saved [{time.time()-t0:.0f}s]")
