"""Export every number the report renders, as JSON, so the page has no magic constants."""
import sys, json; sys.path.insert(0, '.')
import pandas as pd, numpy as np
from src.portfolio import signal_weights, positions_from_weights, FACTOR_COLS
from src.backtest import simulate, returns_matrix
from src.stats import perf_stats, daily_ic, newey_west_tstat
from src.features import STOCK_FEATURES

out = {}
M = pd.read_parquet('data_proc/oos_preds.parquet')
full = pd.read_parquet('data_proc/dataset.parquet')
R = returns_matrix(full)
W = pd.read_parquet('data_proc/W_model.parquet')
H = positions_from_weights(W)
net = simulate(H, R, cost_bps=2.5)['net']

# ---- 1. neutralisation ladder ---------------------------------------------
ladder = []
for fc, lab in [(['__none__'], 'Dollar-neutral only'), (['beta_mkt'], '+ market beta'),
                (['beta_mkt', 'load_pc1', 'load_pc2'], '+ 2 PCs'), (None, '+ 4 PCs (full)')]:
    Wf = signal_weights(M, 'pred_rank', factor_cols=fc)
    s = perf_stats(simulate(positions_from_weights(Wf), R, cost_bps=2.5)['net'])
    ladder.append({'label': lab, 'sharpe': round(s['sharpe'], 3),
                   'ann': round(s['ann_return'], 4), 't': round(s['t_stat_nw'], 2)})
out['ladder'] = ladder

# ---- 2. equity curves ------------------------------------------------------
curves = {}
for sig, lab in [('pred_rank', 'ML model'), ('z_mom_252_21', 'Residual momentum 12-1'),
                 ('z_rev_5', 'Reversal 5d')]:
    Ws = signal_weights(M, sig)
    r = simulate(positions_from_weights(Ws), R, cost_bps=2.5)['net']
    curves[lab] = r
eq = pd.DataFrame(curves).fillna(0.0)
cum = (1 + eq).cumprod()
step = max(len(cum) // 400, 1)
out['equity'] = {
    'dates': [d.strftime('%Y-%m-%d') for d in cum.index[::step]],
    'series': {k: [round(v, 5) for v in cum[k].to_numpy()[::step]] for k in cum.columns},
}

# rolling 252d Sharpe of the model
roll = net.rolling(252)
rs = (roll.mean() / roll.std() * np.sqrt(252)).dropna()
out['rolling'] = {'dates': [d.strftime('%Y-%m-%d') for d in rs.index[::step]],
                  'values': [round(float(v), 3) for v in rs.to_numpy()[::step]]}

# ---- 3. cost sensitivity ---------------------------------------------------
costs = [0, 1, 2.5, 5, 7.5, 10, 15, 20]
prim = [round(perf_stats(simulate(H, R, cost_bps=c)['net'])['sharpe'], 3) for c in costs]
out['cost'] = {'bps': costs, 'primary': prim}
try:
    V = pd.read_parquet('data_proc/val_dataset.parquet')
    VP = pd.read_parquet('data_proc/val_preds.parquet')
    MV = V.merge(VP, on=['date', 'ticker'])
    RV = returns_matrix(V)
    Wv = signal_weights(MV, 'pred_rank')
    Hv = positions_from_weights(Wv)
    out['cost']['validation'] = [
        round(perf_stats(simulate(Hv, RV, cost_bps=c)['net'])['sharpe'], 3) for c in costs]
    icv = daily_ic(MV, 'pred_rank')
    MV['yr'] = MV.date.dt.year
    out['ic_year_val'] = {str(y): round(float(np.nanmean(daily_ic(g, 'pred_rank'))), 4)
                          for y, g in MV.groupby('yr') if len(g) > 5000}
except Exception as exc:
    print("validation export skipped:", exc)

# ---- 4. null distribution --------------------------------------------------
perm = np.load('out/null_perm.npy')
iid0 = np.load('out/null_iid0.npy')
out['null'] = {'perm': [round(float(x), 3) for x in perm],
               'iid0': [round(float(x), 3) for x in iid0],
               'model': round(perf_stats(net)['sharpe'], 3)}

# ---- 5. IC by year, primary ------------------------------------------------
M['yr'] = M.date.dt.year
out['ic_year'] = {str(y): round(float(np.nanmean(daily_ic(g, 'pred_rank'))), 4)
                  for y, g in M.groupby('yr')}

# ---- 6. univariate signal table -------------------------------------------
ic_full = pd.read_csv('out/ic_full.csv')
out['univariate'] = ic_full.sort_values('ic_t_nw', ascending=False).round(4).to_dict('records')

# ---- 7. headline stats -----------------------------------------------------
s = perf_stats(net)
ic = daily_ic(M, 'pred_rank')
m, t = newey_west_tstat(ic.to_numpy(), lags=20)
out['head'] = {k: round(float(v), 4) for k, v in s.items()}
out['head'].update({'ic': round(m, 4), 'ic_t': round(t, 2),
                    'turnover': round(simulate(H, R)['turnover'].mean() * 252, 1),
                    'gross_sharpe': round(perf_stats(simulate(H, R, cost_bps=0)['net'])['sharpe'], 3),
                    'n_positions': round(float((H.abs() > 0.005).sum(axis=1).mean()), 1),
                    'gross_exp': round(float(H.abs().sum(axis=1).mean()), 3)})

# ---- 8. ablation -----------------------------------------------------------
try:
    ab = pd.read_csv('out/ablation.csv')
    out['ablation'] = ab.round(4).to_dict('records')
except Exception as exc:
    print("ablation not ready:", exc)

# ---- 9. per-year net --------------------------------------------------------
out['year_net'] = {str(y): {'ann': round(perf_stats(g)['ann_return'], 4),
                            'sharpe': round(perf_stats(g)['sharpe'], 2)}
                   for y, g in net.groupby(net.index.year) if len(g) > 60}

json.dump(out, open('out/report_data.json', 'w'))
print("wrote out/report_data.json  keys:", list(out.keys()))
print("headline:", out['head'])
