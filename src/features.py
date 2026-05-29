"""Feature engineering.

Calculates volatility, momentum, trend, volume, and - importantly for this
project - *volatility-compression* features that tend to precede directional
price swings (squeeze -> expansion). All windows look BACKWARDS only; there is
no future-looking calculation here.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Helper indicators (pure pandas, no future leakage)
# ----------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # No losses in the (fully-formed) window => maximally overbought, RSI=100,
    # NOT undefined. Without this, short-period RSI (e.g. RSI-2) is NaN on every
    # up-streak and those rows get dropped by dropna, biasing the dataset toward
    # weak days. Warmup rows (loss is NaN) correctly stay NaN.
    rsi = rsi.where(loss != 0, 100.0)
    rsi[loss.isna()] = np.nan
    return rsi


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return _true_range(high, low, close).rolling(period).mean()


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist


def _bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0):
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std()
    upper = ma + num_std * sd
    lower = ma - num_std * sd
    return ma, upper, lower


def _keltner(high, low, close, period: int = 20, atr_mult: float = 1.5):
    """Keltner channel: EMA +/- atr_mult * ATR."""
    ema = close.ewm(span=period, adjust=False).mean()
    atr = _atr(high, low, close, period)
    upper = ema + atr_mult * atr
    lower = ema - atr_mult * atr
    return ema, upper, lower


def _rolling_percentile(s: pd.Series, window: int) -> pd.Series:
    """Rolling percentile rank (0..1) of the last value within the window."""
    return s.rolling(window).apply(
        lambda x: (x <= x[-1]).mean() if np.isfinite(x[-1]) else np.nan,
        raw=True,
    )


# ----------------------------------------------------------------------------
# Public feature builder
# ----------------------------------------------------------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all feature columns to a single-ticker OHLCV dataframe.

    Expects columns: Date, Open, High, Low, Close, Adj Close, Volume.
    Returns a new dataframe with feature columns appended. Does not drop NaN rows.
    """
    df = df.copy().sort_values("Date").reset_index(drop=True)

    high, low, close, openp, vol = (
        df["High"], df["Low"], df["Close"], df["Open"], df["Volume"],
    )
    ret = close.pct_change()

    # ===== Volatility =====
    atr14 = _atr(high, low, close, 14)
    df["atr_14"] = atr14
    df["atr_pct"] = atr14 / close
    df["daily_range_pct"] = (high - low) / close
    df["vol_5d"] = ret.rolling(5).std()
    df["vol_10d"] = ret.rolling(10).std()
    df["vol_20d"] = ret.rolling(20).std()
    df["gap_pct"] = (openp - close.shift(1)) / close.shift(1)

    # ===== Volatility COMPRESSION (pre-swing signals) =====
    bb_mid, bb_up, bb_low = _bollinger(close, 20, 2.0)
    bb_width = (bb_up - bb_low) / bb_mid
    df["bb_width_20"] = bb_width
    # Low percentile => historically tight band => squeeze.
    df["bb_width_pctile_126"] = _rolling_percentile(bb_width, 126)

    kc_mid, kc_up, kc_low = _keltner(high, low, close, 20, 1.5)
    kc_width = (kc_up - kc_low) / kc_mid
    df["kc_width_20"] = kc_width
    # TTM squeeze: Bollinger bands INSIDE Keltner channel => coiled spring.
    squeeze_on = ((bb_low > kc_low) & (bb_up < kc_up)).astype(int)
    df["ttm_squeeze_on"] = squeeze_on
    # Squeeze "fired" = it was on yesterday and turned off today (expansion start).
    df["ttm_squeeze_fired"] = ((squeeze_on.shift(1) == 1) & (squeeze_on == 0)).astype(int)
    # Number of consecutive bars the squeeze has been on.
    grp = (squeeze_on != squeeze_on.shift()).cumsum()
    df["squeeze_duration"] = squeeze_on.groupby(grp).cumsum()

    # ATR contraction.
    atr5 = _atr(high, low, close, 5)
    atr20 = _atr(high, low, close, 20)
    df["atr_ratio_5_20"] = atr5 / atr20
    df["atr_pctile_126"] = _rolling_percentile(atr14, 126)

    # Narrow-range bars (classic pre-breakout).
    rng = high - low
    df["nr7"] = (rng == rng.rolling(7).min()).astype(int)
    df["nr4"] = (rng == rng.rolling(4).min()).astype(int)

    # Consolidation duration: consecutive days range below its 20d median.
    tight = (rng < rng.rolling(20).median()).astype(int)
    grp_t = (tight != tight.shift()).cumsum()
    df["consolidation_days"] = tight.groupby(grp_t).cumsum()

    # Range-compression ratio: recent 5d range vs longer 20d range.
    range_5 = (high.rolling(5).max() - low.rolling(5).min())
    range_20 = (high.rolling(20).max() - low.rolling(20).min())
    df["range_compression"] = range_5 / range_20

    # Volatility of volatility (instability of the vol estimate).
    df["vol_of_vol"] = df["vol_20d"].rolling(20).std()

    # ===== Volume =====
    vol_ma20 = vol.rolling(20).mean()
    df["vol_ma_20"] = vol_ma20
    df["rel_volume"] = vol / vol_ma20
    df["vol_spike_ratio"] = vol / vol.rolling(5).mean()
    df["dollar_volume"] = close * vol
    # Volume dry-up: low recent volume relative to its average (pre-breakout).
    df["volume_dryup"] = vol.rolling(5).mean() / vol_ma20

    # ===== Momentum =====
    df["rsi_14"] = _rsi(close, 14)
    # RSI(2): Connors-style short-term oversold gauge (low = oversold pullback).
    df["rsi_2"] = _rsi(close, 2)
    macd, sig, hist = _macd(close)
    df["macd"] = macd
    df["macd_signal"] = sig
    df["macd_hist"] = hist
    # Short-horizon returns for mean-reversion ranking (most negative = oversold).
    df["ret_2d"] = close.pct_change(2)
    df["ret_3d"] = close.pct_change(3)
    df["ret_5d"] = close.pct_change(5)
    df["ret_10d"] = close.pct_change(10)
    df["ret_20d"] = close.pct_change(20)
    df["ret_42d"] = close.pct_change(42)
    df["ret_60d"] = close.pct_change(60)
    df["roc_10"] = (close / close.shift(10) - 1) * 100

    # ===== Trend =====
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    df["sma_20"] = sma20
    df["sma_50"] = sma50
    df["sma_200"] = sma200
    df["price_above_sma20"] = (close > sma20).astype(int)
    df["price_above_sma50"] = (close > sma50).astype(int)
    df["price_above_sma200"] = (close > sma200).astype(int)
    df["sma20_slope"] = sma20.diff(5) / sma20
    df["sma50_slope"] = sma50.diff(10) / sma50

    # ===== Donchian channels (breakout proximity) =====
    dc_high_20 = high.rolling(20).max()
    dc_low_20 = low.rolling(20).min()
    dc_high_55 = high.rolling(55).max()
    dc_low_55 = low.rolling(55).min()
    df["donchian_width_20"] = (dc_high_20 - dc_low_20) / close
    df["donchian_pos_20"] = (close - dc_low_20) / (dc_high_20 - dc_low_20)
    df["dist_from_20d_high"] = (close - dc_high_20) / dc_high_20
    df["dist_from_20d_low"] = (close - dc_low_20) / dc_low_20
    df["dist_from_55d_high"] = (close - dc_high_55) / dc_high_55

    high_52w = high.rolling(252).max()
    low_52w = low.rolling(252).min()
    df["dist_from_52w_high"] = (close - high_52w) / high_52w
    df["dist_from_52w_low"] = (close - low_52w) / low_52w

    # Replace +/-inf produced by zero-division with NaN.
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def add_market_context(
    df: pd.DataFrame,
    spy: pd.DataFrame,
    qqq: Optional[pd.DataFrame] = None,
    vix: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Merge market-regime features (SPY, optional QQQ and ^VIX) onto a ticker frame.

    Joined on Date. SPY/QQQ contribute trend + volatility context; VIX
    contributes a fear/volatility regime.
    """
    out = df.copy()

    spy = spy.sort_values("Date").reset_index(drop=True)
    spy_close = spy["Close"]
    spy_ctx = pd.DataFrame({
        "Date": spy["Date"],
        "spy_ret_5d": spy_close.pct_change(5),
        "spy_ret_20d": spy_close.pct_change(20),
        "spy_vol_20d": spy_close.pct_change().rolling(20).std(),
        "spy_above_sma50": (spy_close > spy_close.rolling(50).mean()).astype(int),
        # Broad-market regime gate: 200-day trend filter (long-only systems do
        # far better only taking signals while the index is in an uptrend).
        "spy_above_sma200": (spy_close > spy_close.rolling(200).mean()).astype(int),
    })
    out = out.merge(spy_ctx, on="Date", how="left")

    if qqq is not None and not qqq.empty:
        qqq = qqq.sort_values("Date").reset_index(drop=True)
        qc = qqq["Close"]
        qqq_ctx = pd.DataFrame({
            "Date": qqq["Date"],
            "qqq_ret_20d": qc.pct_change(20),
            "qqq_above_sma50": (qc > qc.rolling(50).mean()).astype(int),
        })
        out = out.merge(qqq_ctx, on="Date", how="left")

    if vix is not None and not vix.empty:
        vix = vix.sort_values("Date").reset_index(drop=True)
        vc = vix["Close"]
        vix_ctx = pd.DataFrame({
            "Date": vix["Date"],
            "vix_level": vc,
            "vix_pctile_252": _rolling_percentile(vc, 252),
            "vix_rising": (vc > vc.shift(5)).astype(int),
        })
        out = out.merge(vix_ctx, on="Date", how="left")

    return out


def add_cross_sectional(df: pd.DataFrame) -> pd.DataFrame:
    """Add cross-sectional (per-date) rank features across the whole universe.

    These encode the relative-strength / low-volatility thesis: on any given
    day, rank every ticker against its peers so the model can learn to favour
    the strongest, calmest names rather than judging each stock in isolation.

    Operates on the *combined* multi-ticker frame (must contain a ``Date``
    column with many tickers per date). Ranks are percentile ranks in [0, 1]
    computed independently within each date. Single-name dates get 0.5.

    Requires base features (``ret_20d``, ``ret_60d``, ``atr_pct``,
    ``dist_from_52w_high``) and, for the SPY-relative factor, ``spy_ret_20d``.
    """
    out = df.copy()
    if "Date" not in out.columns:
        return out

    grp = out.groupby("Date")

    def _rank(col: str) -> pd.Series:
        if col not in out.columns:
            return pd.Series(np.nan, index=out.index)
        # pct rank within each date; constant/single-member groups -> 0.5
        r = out.groupby("Date")[col].rank(pct=True)
        return r.fillna(0.5)

    # Higher = stronger / more desirable. ret_20d/42d/60d ~= 1/2/3-month
    # cross-sectional momentum (the short-swing sweep ranks against these).
    out["xs_ret20_rank"] = _rank("ret_20d")
    out["xs_ret42_rank"] = _rank("ret_42d")
    out["xs_ret60_rank"] = _rank("ret_60d")

    # Short-term cross-sectional REVERSAL: the most oversold names (biggest
    # recent loss) score highest. This is the opposite tilt to momentum and is
    # the documented short-horizon equity anomaly (buy the dip in an uptrend).
    out["xs_rev2_rank"] = 1.0 - _rank("ret_2d")
    out["xs_rev3_rank"] = 1.0 - _rank("ret_3d")
    out["xs_rev5_rank"] = 1.0 - _rank("ret_5d")
    # Low ATR% is desirable (quiet) -> invert the rank.
    out["xs_lowvol_rank"] = 1.0 - _rank("atr_pct")
    # Closer to 52w high (dist is negative, nearer 0 = closer) -> higher rank.
    out["xs_dist52w_rank"] = _rank("dist_from_52w_high")

    # Relative strength vs the broad market, then ranked cross-sectionally.
    if "spy_ret_20d" in out.columns and "ret_20d" in out.columns:
        out["xs_rs_vs_spy"] = out["ret_20d"] - out["spy_ret_20d"]
    else:
        out["xs_rs_vs_spy"] = np.nan
    out["xs_rs_rank"] = _rank("xs_rs_vs_spy")

    return out


# Columns that MUST NOT be passed to the model as features.
NON_FEATURE_COLS = {
    "Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume",
    # Legacy 3-class targets:
    "future_return", "label",
    # Triple-barrier / expansion targets + weights:
    "tb_label", "tb_return", "tb_exit_idx", "expansion_label",
    "sample_weight", "future_return_5d",
}


def feature_columns(df: pd.DataFrame) -> List[str]:
    """Return the list of model feature columns from a fully-built dataset."""
    return [c for c in df.columns if c not in NON_FEATURE_COLS]
