"""Point-in-time statistical risk model.

Why this exists
---------------
Da, Liu & Schaumburg (NY Fed SR513) decompose short-term reversal profits and
find the *across-industry* component is worth **-0.295%/mo (t=-4.15)** while the
*within-industry* component is worth **+0.821%/mo (t=5.49)**. Naive close-to-close
reversal bolts a losing leg onto a winning one. To trade the winning leg you must
first strip out the systematic component of each stock's recent move.

We have no GICS codes, so instead of hand-mapping sectors we estimate a rolling
statistical factor model: an equal-weight market factor plus the top-K principal
components of the market-residual covariance. This subsumes industry structure
and, unlike a static sector map, it adapts as correlation structure changes.

Point-in-time discipline
------------------------
Betas for the window [t0, t0+step) are estimated **only** on returns strictly
before t0, using a trailing `lookback` window. They are then held fixed and
applied forward. No observation is ever residualised using its own future.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _fit_window(R: np.ndarray, n_pc: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit market betas + PCA loadings on a (T x N) return window.

    Returns (beta_mkt [N], W [N x K]) where W is orthonormal.
    """
    mkt = np.nanmean(R, axis=1)
    mkt_c = mkt - mkt.mean()
    denom = float(mkt_c @ mkt_c)
    Rc = R - R.mean(axis=0, keepdims=True)
    beta = (Rc.T @ mkt_c) / denom if denom > 0 else np.zeros(R.shape[1])

    E = Rc - np.outer(mkt_c, beta)          # market-residual returns
    cov = np.cov(E, rowvar=False)
    cov = np.nan_to_num(cov)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1][:n_pc]
    W = vecs[:, order]                       # N x K, orthonormal
    return beta, W


def residualize(
    panel: pd.DataFrame,
    lookback: int = 252,
    step: int = 21,
    n_pc: int = 4,
    min_obs: int = 200,
) -> pd.DataFrame:
    """Add `resid`, `beta_mkt`, `mkt_ret` columns to a long-format panel.

    `panel` needs columns: date, ticker, close.
    """
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    wide = panel.pivot(index="date", columns="ticker", values="close")
    rets = wide.pct_change()
    dates = rets.index.to_numpy()
    tickers = rets.columns.to_numpy()
    Rall = rets.to_numpy(dtype=float)
    T, N = Rall.shape

    resid = np.full((T, N), np.nan)
    beta_out = np.full((T, N), np.nan)
    load_out = np.full((T, N, n_pc), np.nan)
    mkt_out = np.full(T, np.nan)

    t0 = lookback + 1
    while t0 < T:
        t1 = min(t0 + step, T)
        win = Rall[t0 - lookback:t0]                      # strictly past
        good = np.isfinite(win).sum(axis=0) >= min_obs
        if good.sum() >= 10:
            Rw = np.nan_to_num(win[:, good])
            beta_g, W = _fit_window(Rw, n_pc)
            beta_full = np.full(N, np.nan)
            beta_full[good] = beta_g
            load_full = np.full((N, n_pc), np.nan)
            load_full[good] = W

            for t in range(t0, t1):
                r = Rall[t]
                obs = np.isfinite(r) & good
                if obs.sum() < 10:
                    continue
                mkt_t = float(np.nanmean(r[obs]))
                mkt_out[t] = mkt_t
                e = np.full(N, np.nan)
                e[obs] = r[obs] - beta_full[obs] * mkt_t
                # Project the market-residual onto the top-K PC subspace and
                # subtract it. W rows are aligned to `good` columns.
                e_g = np.nan_to_num(e[good])
                proj = W @ (W.T @ e_g)
                res_g = e_g - proj
                out_row = np.full(N, np.nan)
                out_row[good] = res_g
                out_row[~obs] = np.nan
                resid[t] = out_row
                beta_out[t] = beta_full
                load_out[t] = load_full
        t0 = t1

    res_df = pd.DataFrame(resid, index=dates, columns=tickers).stack(future_stack=True)
    bet_df = pd.DataFrame(beta_out, index=dates, columns=tickers).stack(future_stack=True)
    out = pd.DataFrame({"resid": res_df, "beta_mkt": bet_df})
    for k in range(n_pc):
        out[f"load_pc{k+1}"] = pd.DataFrame(
            load_out[:, :, k], index=dates, columns=tickers
        ).stack(future_stack=True)
    out.index.names = ["date", "ticker"]
    out = out.reset_index()
    mkt_df = pd.DataFrame({"date": dates, "mkt_ret": mkt_out})

    panel = panel.merge(out, on=["date", "ticker"], how="left")
    panel = panel.merge(mkt_df, on="date", how="left")
    return panel
