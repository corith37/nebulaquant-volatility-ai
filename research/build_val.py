"""Independent-universe validation: 505 S&P names, 2013-2018.
Non-overlapping in breadth (6.6x more names) and largely in time with the
primary 2017-2026 sample. Caveats recorded in the report: this file carries no
dividend adjustment and its membership is S&P 500 as of 2018 (survivorship)."""
import sys,time; sys.path.insert(0,'.')
import pandas as pd, numpy as np
from src.factors import residualize
from src.features import (per_ticker_features, add_regime, cross_sectional_rank,
                          make_target, STOCK_FEATURES, REGIME_FEATURES)
t0=time.time()
d = pd.read_csv('data_raw/all_stocks_5yr.csv')
d.columns=['date','open','high','low','close','volume','ticker']
d['date']=pd.to_datetime(d['date'])
d=d.dropna(subset=['open','high','low','close']).sort_values(['ticker','date'])
# drop names with short history
n=d.groupby('ticker')['date'].size(); d=d[d.ticker.isin(n[n>1100].index)]
d['raw_close']=d['close']
print("val panel",d.shape,d.ticker.nunique(),"tickers",d.date.min().date(),d.date.max().date())

# split check: huge return with no matching intraday range
d['r']=d.groupby('ticker')['close'].pct_change()
susp=((d.r.abs()>0.35)&(((d.high-d.low)/d.close)<0.15)).sum()
print("suspect unadjusted splits:",susp)

d = residualize(d[['date','ticker','open','high','low','close','raw_close','volume']],
                lookback=252, step=21, n_pc=6)
print(f"residualized [{time.time()-t0:.0f}s]")
f = pd.concat([per_ticker_features(g) for _,g in d.groupby('ticker',sort=True)], ignore_index=True)
vix = pd.read_csv('data_raw/vix-daily.csv'); vix.columns=['date','o','h','l','vix']
vix['date']=pd.to_datetime(vix['date']); vix=vix[['date','vix']]
f = add_regime(f, vix)
f = cross_sectional_rank(f, STOCK_FEATURES, min_names=100)
f = make_target(f, horizon=5)
zc=[f"z_{c}" for c in STOCK_FEATURES]
f = f.dropna(subset=zc+REGIME_FEATURES+['y'])
print(f"val dataset {f.shape} dates {f.date.min().date()}..{f.date.max().date()} "
      f"names/date med {int(f.groupby('date').size().median())}  [{time.time()-t0:.0f}s]")
f.to_parquet('data_proc/val_dataset.parquet')
