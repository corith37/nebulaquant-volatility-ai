"""Factor-neutral portfolio construction.

The diagnosis this module exists to fix
---------------------------------------
The model is trained to predict *residual* returns, where "residual" means
orthogonal to a 5-dimensional factor space (market beta + 4 principal
components). A portfolio that is merely dollar- and market-beta-neutral is still
loaded on PC2-PC4. Those loadings carry variance an order of magnitude larger
than the alpha, so the alpha is invisible in the P&L even when the IC is strongly
positive. That mismatch - predicting in one space, trading in another - is why
the first backtest returned a t-stat of -0.32 against an IC t-stat of +3.86.

The fix is to construct weights in the same space the prediction lives in:

    w  proportional to  (I - B (B'B)^-1 B') s

where `s` is the centred signal and `B` the point-in-time factor loadings. The
projection makes the book exactly neutral to every factor by construction, so
the realised P&L is the residual return the model actually forecasts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FACTOR_COLS = ["beta_mkt", "load_pc1", "load_pc2", "load_pc3", "load_pc4"]


def neutral_weights(
    g: pd.DataFrame,
    sig: str,
    factor_cols: list[str] | None = None,
    max_w: float = 0.08,
    gross: float = 1.0,
) -> pd.Series:
    """Factor-neutral, dollar-neutral weights for one date's cross-section."""
    if factor_cols is None:
        factor_cols = FACTOR_COLS
    factor_cols = [c for c in factor_cols if c in g.columns]
    g = g.dropna(subset=[sig])
    if len(g) < 20:
        return pd.Series(dtype=float)

    s = g[sig].to_numpy(float)
    s = s - s.mean()
    if s.std() == 0:
        return pd.Series(0.0, index=g.index)
    s = s / s.std()

    B = g[factor_cols].to_numpy(float) if factor_cols else np.zeros((len(g), 0))
    B = np.nan_to_num(B)
    ones = np.ones((len(g), 1))
    B = np.hstack([ones, B])                      # intercept => dollar neutrality

    # Orthogonal projection of the signal off the factor space.
    try:
        coef, *_ = np.linalg.lstsq(B, s, rcond=None)
        w = s - B @ coef
    except np.linalg.LinAlgError:
        w = s - s.mean()

    gsum = np.abs(w).sum()
    if gsum <= 0:
        return pd.Series(0.0, index=g.index)
    w = w / gsum * gross

    # Cap, then re-neutralise and re-scale (capping reintroduces a small tilt).
    for _ in range(3):
        w = np.clip(w, -max_w, max_w)
        coef, *_ = np.linalg.lstsq(B, w, rcond=None)
        w = w - B @ coef
        gsum = np.abs(w).sum()
        if gsum > 0:
            w = w / gsum * gross
    return pd.Series(w, index=g.index)


def signal_weights(
    df: pd.DataFrame,
    sig: str,
    max_w: float = 0.08,
    factor_cols: list[str] | None = None,
    long_only: bool = False,
    top_q: float = 0.2,
) -> pd.DataFrame:
    """(date x ticker) target weights, one row per signal date."""
    rows = {}
    for d, g in df.groupby("date", sort=True):
        gg = g.set_index("ticker")
        if long_only:
            n = len(gg)
            k = max(int(n * top_q), 3)
            top = gg[sig].rank(method="first") > n - k
            w = pd.Series(0.0, index=gg.index)
            if int(top.sum()):
                w[top] = 1.0 / int(top.sum())
        else:
            w = neutral_weights(gg, sig, factor_cols, max_w=max_w)
        if not w.empty:
            rows[d] = w
    W = pd.DataFrame.from_dict(rows, orient="index").fillna(0.0)
    W.index.name = "date"
    return W.sort_index()


def positions_from_weights(W: pd.DataFrame, hold: int = 5, lag: int = 1) -> pd.DataFrame:
    """Overlapping tranches, vectorised.

    A tranche signalled on date t is filled at the close of t+lag and earns
    returns on days t+lag+1 .. t+lag+hold, each tranche carrying 1/hold of the
    book. That is exactly a shifted rolling mean of the target-weight matrix.
    """
    return W.shift(lag + 1).rolling(hold, min_periods=1).mean().fillna(0.0)


def build_positions(
    df: pd.DataFrame,
    sig: str,
    hold: int = 5,
    lag: int = 1,
    max_w: float = 0.08,
    factor_cols: list[str] | None = None,
    long_only: bool = False,
) -> pd.DataFrame:
    W = signal_weights(df, sig, max_w=max_w, factor_cols=factor_cols, long_only=long_only)
    return positions_from_weights(W, hold=hold, lag=lag)
