"""Statistical machinery for judging whether an edge is real.

Overlapping forward-return labels make the daily IC series strongly
autocorrelated, so a naive t-stat overstates significance by roughly sqrt(h).
Everything here is Newey-West corrected.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps


def newey_west_tstat(x: np.ndarray, lags: int | None = None) -> tuple[float, float]:
    """Mean and NW t-stat of a serially-correlated series."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return float("nan"), float("nan")
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    mu = x.mean()
    e = x - mu
    gamma0 = float(e @ e) / n
    var = gamma0
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        cov = float(e[L:] @ e[:-L]) / n
        var += 2 * w * cov
    se = np.sqrt(max(var, 1e-24) / n)
    return mu, mu / se


def daily_ic(df: pd.DataFrame, sig: str, tgt: str = "y") -> pd.Series:
    """Cross-sectional Spearman IC per date."""
    def _f(g):
        a, b = g[sig].to_numpy(), g[tgt].to_numpy()
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 20:
            return np.nan
        return sps.spearmanr(a[m], b[m]).statistic
    return df.groupby("date")[[sig, tgt]].apply(_f)


def ic_summary(df: pd.DataFrame, sig: str, tgt: str = "y", lags: int = 10) -> dict:
    ic = daily_ic(df, sig, tgt)
    mu, t = newey_west_tstat(ic.to_numpy(), lags=lags)
    sd = np.nanstd(ic.to_numpy())
    return {
        "signal": sig,
        "ic_mean": mu,
        "ic_t_nw": t,
        "ic_std": sd,
        "ic_ir": mu / sd if sd > 0 else np.nan,
        "hit_rate": float(np.nanmean(ic.to_numpy() > 0)),
        "n_days": int(np.isfinite(ic.to_numpy()).sum()),
    }


def deflated_sharpe(sr: float, n: int, skew: float, kurt: float,
                    n_trials: int, sr_trial_std: float) -> float:
    """Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio.

    Answers: given that we looked at `n_trials` candidate strategies whose
    Sharpes varied with sd `sr_trial_std`, what is the probability the observed
    Sharpe reflects genuine skill rather than the maximum of a set of draws from
    a zero-skill distribution?  `sr` and the result are per-period (not annualised).
    """
    if n_trials < 2 or not np.isfinite(sr_trial_std) or sr_trial_std <= 0:
        sr0 = 0.0
    else:
        e = 0.5772156649
        z1 = sps.norm.ppf(1 - 1 / n_trials)
        z2 = sps.norm.ppf(1 - 1 / (n_trials * np.e))
        sr0 = sr_trial_std * ((1 - e) * z1 + e * z2)
    num = (sr - sr0) * np.sqrt(n - 1)
    den = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    if not np.isfinite(den) or den <= 0:
        return np.nan
    return float(sps.norm.cdf(num / den))


def pbo_cscv(perf: np.ndarray, n_split: int = 12) -> float:
    """Probability of Backtest Overfitting via Combinatorially Symmetric CV
    (Bailey, Borwein, Lopez de Prado & Zhu 2017).

    `perf` is a (T x N) matrix of per-period performance for N config variants.
    Splits time into `n_split` blocks, takes every half-sized combination as IS,
    the complement as OOS, and asks how often the IS-best config lands in the
    bottom half OOS. PBO > 0.5 means the selection procedure is worse than random.
    """
    from itertools import combinations

    T, N = perf.shape
    if N < 2:
        return np.nan
    blocks = np.array_split(np.arange(T), n_split)
    half = n_split // 2
    ranks = []
    for combo in combinations(range(n_split), half):
        is_idx = np.concatenate([blocks[i] for i in combo])
        oos_idx = np.concatenate([blocks[i] for i in range(n_split) if i not in combo])
        is_perf = perf[is_idx].mean(axis=0)
        oos_perf = perf[oos_idx].mean(axis=0)
        best = int(np.argmax(is_perf))
        # relative rank of the IS winner among OOS results, in (0,1)
        r = float(sps.rankdata(oos_perf)[best]) / (N + 1)
        ranks.append(r)
    ranks = np.array(ranks)
    logits = np.log(ranks / (1 - ranks))
    return float((logits <= 0).mean())


def perf_stats(rets: pd.Series, periods: int = 252) -> dict:
    r = rets.dropna()
    if len(r) < 20:
        return {}
    ann_ret = float((1 + r).prod() ** (periods / len(r)) - 1)
    ann_vol = float(r.std() * np.sqrt(periods))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    downside = r[r < 0].std() * np.sqrt(periods)
    _, t = newey_west_tstat(r.to_numpy(), lags=10)
    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": ann_ret / downside if downside > 0 else np.nan,
        "max_dd": dd,
        "calmar": ann_ret / abs(dd) if dd < 0 else np.nan,
        "t_stat_nw": t,
        "hit_rate": float((r > 0).mean()),
        "skew": float(r.skew()),
        "kurt": float(r.kurt()),
        "n_days": len(r),
    }
