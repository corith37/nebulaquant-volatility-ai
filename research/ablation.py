"""Leave-one-block-out ablation: which feature families actually carry the edge."""
import sys, time; sys.path.insert(0, '.')
import pandas as pd, numpy as np
from src.model import walk_forward
from src.portfolio import signal_weights, positions_from_weights
from src.backtest import simulate, returns_matrix
from src.stats import daily_ic, newey_west_tstat, perf_stats
from src.features import STOCK_FEATURES, REGIME_FEATURES

df = pd.read_parquet('data_proc/dataset.parquet')
R = returns_matrix(df)

BLOCKS = {
    'reversal': ['rev_1', 'rev_3', 'rev_5', 'rev_10', 'rev_1_sc', 'rev_5_sc', 'rev_10_sc'],
    'momentum': ['mom_21', 'mom_63', 'mom_252_21', 'mom_21_sc', 'dist_52w_high', 'dist_52w_low'],
    'risk/vol': ['idio_vol_21', 'idio_vol_63', 'tot_vol_21', 'vol_ratio_5_63', 'park_vol_21',
                 'vov_63', 'semidev_21', 'skew_63', 'kurt_63', 'beta_mkt'],
    'liquidity': ['ln_dollar_vol', 'amihud_21', 'vol_shock', 'vol_dryup', 'dv_trend'],
    'overnight/intraday': ['on_5', 'intra_5', 'on_21', 'intra_21',
                           'on_intra_spread_5', 'on_intra_spread_21'],
    'price shape': ['clv_1', 'clv_5', 'gap_abs_5', 'updays_10', 'max_abs_10',
                    'donch_pos_20', 'rsi_2', 'rsi_14'],
}
allf = [f"z_{c}" for c in STOCK_FEATURES] + REGIME_FEATURES


def evaluate(feat, label):
    P = walk_forward(df, feat, target='y', horizon=5, embargo=5,
                     min_train_days=756, block_days=63, seeds=(7, 17))
    M = df.merge(P[['date', 'ticker', 'pred_rank']], on=['date', 'ticker'])
    ic = daily_ic(M, 'pred_rank')
    m, t = newey_west_tstat(ic.to_numpy(), lags=20)
    W = signal_weights(M, 'pred_rank')
    s = perf_stats(simulate(positions_from_weights(W), R, cost_bps=2.5)['net'])
    print(f"  {label:<24} IC={m:+.4f}(t{t:+.2f})  netSharpe={s['sharpe']:+.3f}  "
          f"ann={s['ann_return']:+.2%}", flush=True)
    return dict(variant=label, ic=m, ic_t=t, sharpe=s['sharpe'], ann=s['ann_return'])


print("=== FEATURE-BLOCK ABLATION (leave-one-block-out) ===", flush=True)
rows = [evaluate(allf, 'FULL model')]
for name, cols in BLOCKS.items():
    drop = {f"z_{c}" for c in cols}
    rows.append(evaluate([f for f in allf if f not in drop], f"drop {name}"))
rows.append(evaluate(REGIME_FEATURES, 'regime context only'))
rows.append(evaluate([f"z_{c}" for c in STOCK_FEATURES], 'no regime context'))
pd.DataFrame(rows).to_csv('out/ablation.csv', index=False)
print("saved out/ablation.csv")
