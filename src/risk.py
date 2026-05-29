"""Position sizing and trade-risk helpers (long-only).

Sizing is confidence- and volatility-aware: stronger model conviction sizes up,
higher-volatility names size down (vol targeting), all subject to per-position
and total-exposure caps.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RiskConfig:
    tp_atr_mult: float = 2.0
    sl_atr_mult: float = 1.0
    max_hold_days: int = 10
    commission_pct: float = 0.001
    slippage_pct: float = 0.001
    position_size_pct: float = 0.10
    max_position_pct: float = 0.20
    max_total_exposure: float = 1.0
    vol_target: bool = True
    prob_threshold: float = 0.55
    # Regime gate: only enter longs while the broad market is in an uptrend
    # (SPY > 200d SMA) and VIX is below `max_vix_pctile` of its 1y range.
    # Long-only edge concentrates in benign regimes; gating out bear/panic
    # windows removes the negative folds that crush risk-adjusted return.
    regime_filter: bool = True
    max_vix_pctile: float = 0.85
    # Per-name trend filters applied at entry. Momentum wants names already
    # trending up (above 20d); short-term mean-reversion wants oversold names
    # (below 20d) that are still in a long-term uptrend (above 200d), so these
    # are configurable rather than hard-coded.
    require_above_sma20: bool = True
    require_above_sma200: bool = False


# Width (in score units) above the threshold that maps confidence onto its full
# range. Calibrated meta-label scores compress into a narrow band just above the
# gate, so dividing by (1 - threshold) would collapse every position to a sliver
# of equity. A modest band keeps sizing meaningful while still tilting toward
# higher-conviction signals.
_CONF_BAND = 0.12


def confidence_fraction(
    score: float,
    threshold: float,
    atr_pct: float,
    cfg: RiskConfig,
    ref_atr_pct: float,
) -> float:
    """Return the target fraction of equity to allocate to a signal.

    - Above-threshold signals always receive a real allocation: the base size is
      tilted between 0.5x and 1.5x by confidence, where confidence maps the band
      ``[threshold, threshold + _CONF_BAND]`` onto [0, 1].
    - Vol targeting: scale by ref_atr_pct / atr_pct, clipped to [0.5, 2.0], so a
      twice-as-volatile name gets roughly half the size.
    - Capped at max_position_pct.
    """
    if score <= threshold:
        return 0.0
    conf = float(np.clip((score - threshold) / _CONF_BAND, 0.0, 1.0))
    frac = cfg.position_size_pct * (0.5 + conf)
    if cfg.vol_target and atr_pct and atr_pct > 0 and ref_atr_pct > 0:
        frac *= float(np.clip(ref_atr_pct / atr_pct, 0.5, 2.0))
    return float(min(frac, cfg.max_position_pct))


def stop_loss_price(entry: float, atr: float, cfg: RiskConfig) -> float:
    return entry - cfg.sl_atr_mult * atr


def take_profit_price(entry: float, atr: float, cfg: RiskConfig) -> float:
    return entry + cfg.tp_atr_mult * atr


def shares_for_fraction(capital: float, fraction: float, price: float) -> int:
    if price <= 0 or fraction <= 0:
        return 0
    return int((capital * fraction) // price)
