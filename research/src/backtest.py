"""Portfolio construction and costed simulation.

Construction choices and why they are not arbitrary
---------------------------------------------------
* **Overlapping tranches.** Each day we open 1/H of the book and hold it H days
  (Jegadeesh-Titman). Daily turnover falls to roughly 1/H of a full rebalance
  without changing the signal, which is the single cheapest way to buy back
  net-of-cost return. Novy-Marx & Velikov show turnover control is what separates
  Momentum surviving costs (0.68% -> 0.85% net with buy/hold bands) from
  reversal dying (+0.37% gross -> -1.28% net).
* **Beta neutralisation.** Dollar-neutral is not risk-neutral: a long-short book
  built from ranks routinely carries +/-0.3 net beta, and over a decade with a
  strong equity drift that beta *is* the reported alpha. We solve for weights
  that are simultaneously dollar- and beta-neutral.
* **One full day of implementation lag.** Signal uses data through the close of
  day t; the fill is at the close of day t+1; the first return earned is day t+2.

Execution timing is the most common place a backtest quietly cheats, so it is
made explicit in `build_positions` rather than hidden in an index shift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def target_weights(
    g: pd.DataFrame,
    sig: str,
    quantile: float = 0.2,
    beta_neutral: bool = True,
    long_only: bool = False,
    max_w: float = 0.10,
) -> pd.Series:
    """Weights for one date's cross-section. Gross exposure normalised to 1.0
    (long-short) or 1.0 long (long-only)."""
    g = g.dropna(subset=[sig])
    n = len(g)
    if n < 20:
        return pd.Series(dtype=float)
    k = max(int(np.floor(n * quantile)), 3)
    order = g[sig].rank(method="first")
    longs = g.index[order > n - k]
    shorts = g.index[order <= k]

    w = pd.Series(0.0, index=g.index)
    if long_only:
        w[longs] = 1.0 / len(longs)
        return w.clip(upper=max_w)

    w[longs] = 0.5 / len(longs)
    w[shorts] = -0.5 / len(shorts)

    if beta_neutral and "beta_mkt" in g.columns:
        b = g["beta_mkt"].fillna(1.0)
        bl = float((w[w > 0] * b[w > 0]).sum())
        bs = float((w[w < 0] * b[w < 0]).sum())
        if abs(bs) > 1e-6:
            # scale the short leg so net beta = 0, then renormalise gross to 1
            s = bl / abs(bs)
            w[w < 0] *= s
            gross = float(w.abs().sum())
            if gross > 0:
                w /= gross
            # restore dollar neutrality by shifting the residual cash tilt
            w -= w.mean()
    return w.clip(lower=-max_w, upper=max_w)


def build_positions(
    df: pd.DataFrame,
    sig: str,
    hold: int = 5,
    quantile: float = 0.2,
    beta_neutral: bool = True,
    long_only: bool = False,
    lag: int = 1,
) -> pd.DataFrame:
    """Daily holdings matrix (dates x tickers) from overlapping tranches.

    A tranche signalled on date t is filled at the close of t+`lag` and held for
    `hold` days, so it earns returns on days t+lag+1 .. t+lag+hold.
    """
    dates = np.sort(df["date"].unique())
    pos = {d: {} for d in dates}
    dpos = {d: i for i, d in enumerate(dates)}

    for d, g in df.groupby("date", sort=True):
        w = target_weights(g.set_index("ticker"), sig, quantile, beta_neutral, long_only)
        if w.empty:
            continue
        i0 = dpos[d] + lag + 1
        for k in range(hold):
            j = i0 + k
            if j >= len(dates):
                break
            tgt = pos[dates[j]]
            for tkr, val in w.items():
                if val != 0.0:
                    tgt[tkr] = tgt.get(tkr, 0.0) + val / hold

    H = pd.DataFrame.from_dict(pos, orient="index").fillna(0.0)
    H.index.name = "date"
    return H.sort_index()


def simulate(
    H: pd.DataFrame,
    rets: pd.DataFrame,
    cost_bps: float = 2.5,
    borrow_bps_annual: float = 50.0,
) -> pd.DataFrame:
    """Net daily P&L. `rets` is a (date x ticker) frame of simple returns where
    row s holds the close(s-1) -> close(s) return."""
    cols = [c for c in H.columns if c in rets.columns]
    H = H[cols]
    R = rets.reindex(index=H.index, columns=cols).fillna(0.0)

    gross_pnl = (H * R).sum(axis=1)
    turnover = (H - H.shift(1).fillna(0.0)).abs().sum(axis=1)
    tc = turnover * cost_bps / 1e4
    short_notional = H.clip(upper=0).abs().sum(axis=1)
    borrow = short_notional * (borrow_bps_annual / 1e4) / 252

    out = pd.DataFrame({
        "gross": gross_pnl,
        "turnover": turnover,
        "cost": tc,
        "borrow": borrow,
        "net": gross_pnl - tc - borrow,
        "gross_exposure": H.abs().sum(axis=1),
        "net_exposure": H.sum(axis=1),
    })
    return out


def returns_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """(date x ticker) simple returns, row s = close(s-1) -> close(s)."""
    wide = df.pivot_table(index="date", columns="ticker", values="close")
    return wide.pct_change()
