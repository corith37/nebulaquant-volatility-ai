"""Feature construction.

Every stock-level feature is converted to a **within-date cross-sectional rank**
before it reaches the model. That is not cosmetic:

* It removes the date effect, so the model learns *relative* position rather than
  a market-timing rule disguised as stock selection.
* It kills the non-stationary-level failure mode. A raw `sma_200 = 312.4` is a
  near-unique fingerprint for one name in one era; a tree will happily memorise
  it and the rule silently expires when the price drifts. Ranks cannot be
  memorised that way.

Date-level regime variables are deliberately left un-ranked (they are constant
within a date, so ranking would destroy them). They enter as interaction context.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    out = out.where(dn != 0, 100.0)
    out[dn.isna()] = np.nan
    return out


def _safe_div(a, b):
    return a / b.replace(0, np.nan)


# ---------------------------------------------------------------------------
# per-ticker features
# ---------------------------------------------------------------------------

def per_ticker_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    c, h, l, o, v = g["close"], g["high"], g["low"], g["open"], g["volume"]
    r = c.pct_change()
    e = g["resid"]

    # --- overnight / intraday decomposition (Lou, Polk & Skouras 2019) -------
    # STR earns +0.93%/mo overnight and -1.05%/mo intraday; momentum is
    # essentially 100% overnight. Collapsing to close-to-close discards the sign.
    on = o / c.shift(1) - 1
    intr = c / o - 1
    g["on_ret"], g["intra_ret"] = on, intr

    # --- reversal block (on residual returns) --------------------------------
    for k in (1, 3, 5, 10):
        g[f"rev_{k}"] = -e.rolling(k).sum()
    iv21 = e.rolling(21).std()
    iv63 = e.rolling(63).std()
    iv5 = e.rolling(5).std()
    g["idio_vol_21"], g["idio_vol_63"] = iv21, iv63
    # Vol-scaled reversal: a -4% move in a quiet name is a far bigger dislocation
    # than the same move in a jumpy one.
    g["rev_1_sc"] = _safe_div(g["rev_1"], iv21)
    g["rev_5_sc"] = _safe_div(g["rev_5"], iv21)
    g["rev_10_sc"] = _safe_div(g["rev_10"], iv21)

    # --- momentum block ------------------------------------------------------
    g["mom_21"] = e.rolling(21).sum()
    g["mom_63"] = e.rolling(63).sum()
    g["mom_252_21"] = e.rolling(231).sum().shift(21)   # 12-1 momentum
    g["mom_21_sc"] = _safe_div(g["mom_21"], iv63)

    # --- overnight vs intraday accumulations ---------------------------------
    for k in (5, 21):
        g[f"on_{k}"] = on.rolling(k).sum()
        g[f"intra_{k}"] = intr.rolling(k).sum()
    g["on_intra_spread_5"] = g["on_5"] - g["intra_5"]
    g["on_intra_spread_21"] = g["on_21"] - g["intra_21"]

    # --- risk ----------------------------------------------------------------
    g["tot_vol_21"] = r.rolling(21).std()
    g["vol_ratio_5_63"] = _safe_div(iv5, iv63)
    # Parkinson high-low estimator: ~5x more efficient than close-to-close
    park = np.log(h / l) ** 2 / (4 * np.log(2))
    g["park_vol_21"] = np.sqrt(park.rolling(21).mean())
    g["vov_63"] = e.rolling(21).std().rolling(63).std()
    g["semidev_21"] = e.clip(upper=0).rolling(21).std()
    g["skew_63"] = e.rolling(63).skew()
    g["kurt_63"] = e.rolling(63).kurt()

    # --- liquidity -----------------------------------------------------------
    dv = c * v
    g["ln_dollar_vol"] = np.log1p(dv.rolling(21).mean())
    # Amihud (2002) illiquidity: price impact per dollar traded.
    g["amihud_21"] = (r.abs() / dv.replace(0, np.nan)).rolling(21).mean() * 1e9
    vma21 = v.rolling(21).mean()
    g["vol_shock"] = _safe_div(v, vma21)
    g["vol_dryup"] = _safe_div(v.rolling(5).mean(), vma21)
    g["dv_trend"] = _safe_div(dv.rolling(5).mean(), dv.rolling(63).mean())

    # --- price shape ---------------------------------------------------------
    clv = _safe_div(c - l, (h - l))
    g["clv_1"] = clv
    g["clv_5"] = clv.rolling(5).mean()
    g["gap_abs_5"] = on.abs().rolling(5).mean()
    g["updays_10"] = (e > 0).rolling(10).sum()
    g["max_abs_10"] = e.abs().rolling(10).max()
    hi252, lo252 = h.rolling(252).max(), l.rolling(252).min()
    g["dist_52w_high"] = c / hi252 - 1
    g["dist_52w_low"] = c / lo252 - 1
    hi20, lo20 = h.rolling(20).max(), l.rolling(20).min()
    g["donch_pos_20"] = _safe_div(c - lo20, (hi20 - lo20))
    g["rsi_2"] = _rsi(c, 2)
    g["rsi_14"] = _rsi(c, 14)

    # --- forward targets (labels only; never features) -----------------------
    for hzn in (3, 5, 10):
        g[f"fwd_resid_{hzn}"] = e.shift(-hzn).rolling(hzn).sum()   # t+1..t+h
        g[f"fwd_ret_{hzn}"] = c.shift(-hzn) / c - 1
    # Execution reference: we sign at close t, fill at close t+1.
    g["fill_close"] = c.shift(-1)
    return g


STOCK_FEATURES = [
    "rev_1", "rev_3", "rev_5", "rev_10", "rev_1_sc", "rev_5_sc", "rev_10_sc",
    "mom_21", "mom_63", "mom_252_21", "mom_21_sc",
    "on_5", "intra_5", "on_21", "intra_21", "on_intra_spread_5", "on_intra_spread_21",
    "idio_vol_21", "idio_vol_63", "tot_vol_21", "vol_ratio_5_63", "park_vol_21",
    "vov_63", "semidev_21", "skew_63", "kurt_63", "beta_mkt",
    "ln_dollar_vol", "amihud_21", "vol_shock", "vol_dryup", "dv_trend",
    "clv_1", "clv_5", "gap_abs_5", "updays_10", "max_abs_10",
    "dist_52w_high", "dist_52w_low", "donch_pos_20", "rsi_2", "rsi_14",
]

REGIME_FEATURES = [
    "vix", "vix_pct252", "vix_chg5", "mkt_vol_21", "xs_disp", "mkt_above_200",
    "corr_proxy", "mkt_ret_5",
]


# ---------------------------------------------------------------------------
# panel-level assembly
# ---------------------------------------------------------------------------

def add_regime(df: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    """Date-level regime context. Nagel (2012) shows normalised VIX predicts
    next-day reversal-strategy returns with a coefficient of 0.22 (t~11) and an
    adjusted R2 of 0.07 daily / 0.56 monthly - the single strongest conditioner
    for this family of strategies."""
    d = df.groupby("date").agg(
        mkt_ret=("mkt_ret", "first"),
        xs_disp=("resid", "std"),
        idio_var=("resid", lambda s: float(np.nanvar(s))),
    ).reset_index()
    tot = df.groupby("date")["close"].count().rename("n").reset_index()
    d = d.merge(tot, on="date")

    d = d.merge(vix, on="date", how="left")
    d["vix"] = d["vix"].ffill()
    d["vix_pct252"] = d["vix"].rolling(252).rank(pct=True)
    d["vix_chg5"] = d["vix"] / d["vix"].shift(5) - 1
    d["mkt_vol_21"] = d["mkt_ret"].rolling(21).std()
    d["mkt_ret_5"] = d["mkt_ret"].rolling(5).sum()
    mkt_idx = (1 + d["mkt_ret"].fillna(0)).cumprod()
    d["mkt_above_200"] = (mkt_idx > mkt_idx.rolling(200).mean()).astype(float)
    # Share of variance that is systematic: high = everything moving together,
    # which is when cross-sectional stock selection is hardest.
    tot_var = d["mkt_ret"].rolling(21).var()
    d["corr_proxy"] = 1 - (d["idio_var"].rolling(21).mean() /
                           (d["idio_var"].rolling(21).mean() + tot_var))
    keep = ["date"] + REGIME_FEATURES
    return df.merge(d[keep], on="date", how="left")


def cross_sectional_rank(df: pd.DataFrame, cols: list[str], min_names: int = 30) -> pd.DataFrame:
    """Within-date percentile rank, centred on zero. Dates thinner than
    `min_names` are dropped: a 'cross-sectional' rank over 8 names is noise."""
    df = df.copy()
    counts = df.groupby("date")["ticker"].transform("size")
    df = df[counts >= min_names].copy()
    g = df.groupby("date")
    for c in cols:
        df[f"z_{c}"] = g[c].rank(pct=True) - 0.5
    return df


def make_target(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Target = within-date rank of the forward residual return, centred.

    Ranking the target (rather than regressing raw returns) matters because raw
    forward returns are fat-tailed and heteroskedastic across time; a squared-loss
    model fit on them spends most of its capacity on a handful of crisis days.
    """
    df = df.copy()
    col = f"fwd_resid_{horizon}"
    df["y"] = df.groupby("date")[col].rank(pct=True) - 0.5
    df.loc[df[col].isna(), "y"] = np.nan
    return df
