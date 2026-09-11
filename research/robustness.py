import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np
from itertools import product
from src.portfolio import signal_weights, positions_from_weights, FACTOR_COLS
from src.backtest import simulate, returns_matrix
from src.stats import perf_stats, deflated_sharpe, pbo_cscv, newey_west_tstat
np.random.seed(1)
pd.set_option('display.width',250); pd.set_option('display.float_format',lambda v:f"{v: .3f}")

M=pd.read_parquet('data_proc/oos_preds.parquet')
full=pd.read_parquet('data_proc/dataset.parquet')
R=returns_matrix(full)
W=pd.read_parquet('data_proc/W_model.parquet')

print("=== 1. NEUTRALISATION LADDER (what each factor removal is worth) ===")
for fc,lab in [(None,'market beta + 4 PCs'),(['beta_mkt','load_pc1','load_pc2'],'beta + 2 PCs'),
               (['beta_mkt'],'market beta only'),(['__none__'],'dollar-neutral only')]:
    Wf=signal_weights(M,'pred_rank',factor_cols=fc)
    r=simulate(positions_from_weights(Wf),R,cost_bps=2.5); s=perf_stats(r['net'])
    print(f"  {lab:>22}: net Sharpe={s['sharpe']:+.3f}  ann={s['ann_return']:+.2%}  t={s['t_stat_nw']:+.2f}")

print("\n=== 2. NULL DISTRIBUTIONS ===")
def null_sharpes(kind,n=40,cost=2.5):
    out=[]
    for i in range(n):
        if kind=='iid':
            M['_s']=np.random.randn(len(M))
        else:  # block-permute the model's own signal across tickers within date
            M['_s']=M.groupby('date')['pred_rank'].transform(lambda s: s.sample(frac=1,random_state=i).to_numpy())
        Wn=signal_weights(M,'_s')
        out.append(perf_stats(simulate(positions_from_weights(Wn),R,cost_bps=cost)['net'])['sharpe'])
    return np.array(out)
n0=null_sharpes('iid',40,cost=0.0); n25=null_sharpes('iid',40,cost=2.5)
perm=null_sharpes('perm',40,cost=2.5)
print(f"  iid random, ZERO cost   : mean={n0.mean():+.3f} sd={n0.std():.3f} p95={np.percentile(n0,95):+.3f}")
print(f"  iid random, 2.5bp cost  : mean={n25.mean():+.3f} sd={n25.std():.3f} p95={np.percentile(n25,95):+.3f}")
print(f"  cross-sec permutation   : mean={perm.mean():+.3f} sd={perm.std():.3f} p95={np.percentile(perm,95):+.3f}")
r_model=simulate(positions_from_weights(W),R,cost_bps=2.5)['net']
s_model=perf_stats(r_model)['sharpe']
print(f"  MODEL Sharpe={s_model:+.3f}   empirical p-value vs permutation null = {(perm>=s_model).mean():.4f}")

print("\n=== 3. PROBABILITY OF BACKTEST OVERFITTING (CSCV) ===")
cfgs=list(product([2,3,5,10,21],[0.04,0.08,0.15]))
mat=[]
for h,mw in cfgs:
    Wc=signal_weights(M,'pred_rank',max_w=mw)
    mat.append(simulate(positions_from_weights(Wc,hold=h),R,cost_bps=2.5)['net'])
P=pd.concat(mat,axis=1).dropna().to_numpy()
print(f"  {len(cfgs)} configs (hold x max-weight), {P.shape[0]} days")
print(f"  PBO = {pbo_cscv(P, n_split=10):.3f}    (>0.5 means the selection rule is worse than random)")
print(f"  config Sharpe spread: min={min(perf_stats(pd.Series(P[:,i]))['sharpe'] for i in range(P.shape[1])):+.3f} "
      f"max={max(perf_stats(pd.Series(P[:,i]))['sharpe'] for i in range(P.shape[1])):+.3f}")

print("\n=== 4. DEFLATED SHARPE RATIO ===")
srs=[perf_stats(pd.Series(P[:,i]))['sharpe'] for i in range(P.shape[1])]
st=perf_stats(r_model)
sr_d=st['sharpe']/np.sqrt(252)                      # per-day
n_trials=len(cfgs)+8+6+4                            # configs + signals + costs + neutralisations
dsr=deflated_sharpe(sr_d,n=st['n_days'],skew=st['skew'],kurt=st['kurt']+3,
                    n_trials=n_trials,sr_trial_std=np.std(srs)/np.sqrt(252))
print(f"  observed annual Sharpe = {st['sharpe']:+.3f}   trials counted = {n_trials}")
print(f"  DSR = P(true Sharpe > 0 | selection) = {dsr:.4f}")
print(f"  skew={st['skew']:+.2f} exkurt={st['kurt']:+.2f} n={st['n_days']}")

print("\n=== 5. SUBPERIOD STABILITY (net, 2.5bp) ===")
h=len(r_model)//2
for lab,seg in [('first half',r_model.iloc[:h]),('second half',r_model.iloc[h:])]:
    s=perf_stats(seg); print(f"  {lab:>12}: Sharpe={s['sharpe']:+.3f} ann={s['ann_return']:+.2%} t={s['t_stat_nw']:+.2f} n={s['n_days']}")

print("\n=== 6. CAPACITY / IMPLEMENTATION ===")
H=positions_from_weights(W)
print(f"  mean gross exposure       {H.abs().sum(axis=1).mean():.3f}")
print(f"  mean # positions          {(H.abs()>1e-4).sum(axis=1).mean():.1f}")
print(f"  mean # positions >0.5%    {(H.abs()>0.005).sum(axis=1).mean():.1f}")
print(f"  largest position          {H.abs().max().max():.3f}")
print(f"  annual turnover           {simulate(H,R)['turnover'].mean()*252:.1f}x")
r_model.to_frame('net').to_parquet('data_proc/model_net.parquet')
np.save('out/null_perm.npy',perm); np.save('out/null_iid0.npy',n0)
