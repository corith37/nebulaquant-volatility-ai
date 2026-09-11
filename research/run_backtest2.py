import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np
from src.portfolio import build_positions, signal_weights, FACTOR_COLS
from src.backtest import simulate, returns_matrix
from src.stats import perf_stats, newey_west_tstat
np.random.seed(0)
pd.set_option('display.width',250); pd.set_option('display.float_format',lambda v:f"{v: .3f}")

M    = pd.read_parquet('data_proc/oos_preds.parquet')
full = pd.read_parquet('data_proc/dataset.parquet')
R    = returns_matrix(full)
mkt  = full.groupby('date')['mkt_ret'].first()

def run(sig, df=M, hold=5, cost=2.5, max_w=0.08, lo=False, fc=None):
    H = build_positions(df, sig, hold=hold, max_w=max_w, factor_cols=fc, long_only=lo)
    return H, simulate(H, R, cost_bps=cost)

def row(name,H,res):
    s=perf_stats(res['net']); s['name']=name
    s['gross_sharpe']=perf_stats(res['gross'])['sharpe']
    s['ann_turn']=res['turnover'].mean()*252
    s['n_names']=(H.abs()>1e-6).sum(axis=1).mean()
    s['beta_to_mkt']=np.corrcoef(res['net'].fillna(0), mkt.reindex(res.index).fillna(0))[0,1]
    return s

print("=== FACTOR-NEUTRAL LONG/SHORT (5d hold, 2.5bp one-way, 50bp/yr borrow) ===")
rows=[]
for sig,label in [('pred_rank','ML model (factor-neutral)'),
                  ('z_rev_5','reversal 5d'),('z_rev_5_sc','vol-scaled reversal 5d'),
                  ('z_rev_1','reversal 1d'),
                  ('z_mom_252_21','residual momentum 12-1'),('z_mom_63','residual momentum 3m'),
                  ('z_gap_abs_5','overnight gap (short high-gap)'),
                  ('z_on_intra_spread_21','overnight-intraday spread')]:
    if sig not in M.columns: continue
    H,res=run(sig); rows.append(row(label,H,res))
S=pd.DataFrame(rows).set_index('name')
print(S[['sharpe','gross_sharpe','ann_return','ann_vol','max_dd','t_stat_nw','ann_turn','n_names','beta_to_mkt']].to_string())

print("\n=== RANDOM-SIGNAL NULL (construction sanity check) ===")
nulls=[]
for i in range(12):
    M[f'_r{i}']=np.random.randn(len(M))
    _,r=run(f'_r{i}'); nulls.append(perf_stats(r['net'])['sharpe'])
nulls=np.array(nulls)
print(f"  Sharpe mean={nulls.mean():+.3f}  sd={nulls.std():.3f}  min={nulls.min():+.3f}  max={nulls.max():+.3f}")
print("  (a sound construction should centre near zero minus a small cost drag)")

print("\n=== ML model: cost sensitivity ===")
H,_=run('pred_rank'); out=[]
for c in [0,1,2.5,5,10,20]:
    r=simulate(H,R,cost_bps=c); s=perf_stats(r['net'])
    out.append(dict(cost_bps=c,sharpe=s['sharpe'],ann_ret=s['ann_return'],t=s['t_stat_nw']))
print(pd.DataFrame(out).to_string(index=False))

print("\n=== ML model: holding period ===")
out=[]
for h in [2,3,5,10,21]:
    Hh,r=run('pred_rank',hold=h); s=perf_stats(r['net'])
    out.append(dict(hold=h,sharpe=s['sharpe'],ann_ret=s['ann_return'],
                    ann_turn=r['turnover'].mean()*252,t=s['t_stat_nw']))
print(pd.DataFrame(out).to_string(index=False))

print("\n=== Does neutralisation matter? (ML model, 5d) ===")
for fc,lab in [(FACTOR_COLS,'market beta + 4 PCs'),(['beta_mkt'],'market beta only'),([],'dollar-neutral only')]:
    Hf,r=run('pred_rank',fc=fc); s=perf_stats(r['net'])
    b=np.corrcoef(r['net'].fillna(0),mkt.reindex(r.index).fillna(0))[0,1]
    print(f"  {lab:>22}: Sharpe={s['sharpe']:+.3f} ann={s['ann_return']:+.2%} t={s['t_stat_nw']:+.2f} corr_mkt={b:+.3f}")

print("\n=== ML model: year by year (net) ===")
H,res=run('pred_rank')
for yr,g in res.groupby(res.index.year):
    s=perf_stats(g['net']); print(f"  {yr}: ann={s['ann_return']:+7.2%} sharpe={s['sharpe']:+6.2f} maxdd={s['max_dd']:+.2%} n={s['n_days']}")

res.to_parquet('data_proc/pnl_ls2.parquet'); H.to_parquet('data_proc/holdings_ls2.parquet')
# robustness.py reads the model's TARGET weights (pre-tranche), not the holdings.
signal_weights(M, 'pred_rank').to_parquet('data_proc/W_model.parquet')
S.to_csv('out/strategy_summary2.csv')
print("\nsaved")
