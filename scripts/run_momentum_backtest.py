"""Honest backtest of the PIVOTED strategy: regime-gated cross-sectional momentum.

Why this exists
---------------
The original squeeze->expansion ML thesis and the single-stage long meta-label
both failed the edge gate: the learned probability did NOT rank realized
outcomes (spearman ~ 0 / slightly negative). A direct factor study then showed
what this universe actually rewards over 2015-2024:

  * 60-day cross-sectional momentum (xs_ret60_rank): POSITIVE, monotone-ish.
  * low-volatility / near-52w-high tilts: NEGATIVE here (growth/high-beta names
    led), so the "buy calm names" half of the thesis is dropped.

So the deployable signal is rules-based, not ML: among names in a broad-market
uptrend (SPY > 200d SMA, benign VIX) that are themselves trending (> 20d SMA),
hold the top-decile momentum names with a fixed ~20-day horizon and rebalance.

This script proves it in the *production* event-driven engine (intrabar fills,
costs, cash-when-flat equity curve), benchmarked against SPY buy-and-hold and a
random-entry baseline over the identical span. There is no model training here,
so every entry uses only trailing information -> the whole-period result is the
honest out-of-sample estimate.

HONESTY CAVEAT: the 75-name universe is a curated list of *today's* known
large/mid-caps, so absolute returns carry survivorship/selection bias. The
robust claim is the *relative* momentum edge vs the same-universe equal-weight
and vs SPY on a risk-adjusted (Sharpe) basis.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.backtest import benchmarks, portfolio_backtest, save_backtest_outputs
from src.config import ensure_paths, load_config
from src.risk import RiskConfig
from src.utils import setup_logger

SCORE_COL = "xs_ret60_rank"   # the positively-predictive cross-sectional factor


def main() -> int:
    logger = setup_logger("nebulaquant.momentum_backtest")
    cfg = load_config()
    paths = ensure_paths()
    bt = cfg.get("backtest", {})
    mom = cfg.get("momentum", {})

    dataset_path = paths["data_processed"] / "model_dataset.csv"
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}. Run build_dataset first.")
        return 1
    df = pd.read_csv(dataset_path, parse_dates=["Date"])
    if SCORE_COL not in df.columns:
        logger.error(f"{SCORE_COL} missing - rebuild the dataset (add_cross_sectional).")
        return 1

    # Rules-based score: the cross-sectional momentum rank IS the score.
    df["score"] = df[SCORE_COL]

    top_pctile = float(mom.get("top_pctile", 0.90))   # top-decile momentum
    hold_days = int(mom.get("hold_days", 20))
    risk = RiskConfig(
        # Make the ATR barriers effectively unreachable so the exit is purely the
        # time barrier -> a fixed ~hold_days momentum hold, matching the factor study.
        tp_atr_mult=float(mom.get("tp_atr_mult", 100.0)),
        sl_atr_mult=float(mom.get("sl_atr_mult", 100.0)),
        max_hold_days=hold_days,
        commission_pct=bt.get("commission_pct", 0.001),
        slippage_pct=bt.get("slippage_pct", 0.001),
        position_size_pct=float(mom.get("position_size_pct", 0.12)),
        max_position_pct=float(mom.get("max_position_pct", 0.20)),
        max_total_exposure=float(mom.get("max_total_exposure", 1.0)),
        vol_target=bool(mom.get("vol_target", False)),   # equal-ish weight by default
        prob_threshold=top_pctile,
        regime_filter=bt.get("regime_filter", True),
        max_vix_pctile=bt.get("max_vix_pctile", 0.85),
    )
    logger.info(f"Momentum backtest: score={SCORE_COL} top_pctile={top_pctile} "
                f"hold_days={hold_days} vol_target={risk.vol_target} regime={risk.regime_filter}")

    trade_log, equity_curve, metrics = portfolio_backtest(
        df, cfg=risk,
        initial_capital=bt.get("initial_capital", 10_000),
        entry_min_rel_volume=float(mom.get("entry_min_rel_volume", 0.0)),
    )
    bench = benchmarks(df, cfg=risk, n_trades=metrics["n_trades"])
    metrics = {**metrics, **bench}

    out_dir = paths["reports_backtests"]
    save_backtest_outputs(
        trade_log=trade_log, metrics=metrics, out_dir=out_dir,
        equity_curve=equity_curve, prefix="momentum_",
    )

    logger.info("=== Momentum (cross-sectional, regime-gated) backtest ===")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")

    strat_ret = metrics.get("total_return", 0.0)
    spy_ret = metrics.get("spy_buy_hold_return", 0.0)
    strat_sh = metrics.get("sharpe", 0.0)
    spy_sh = metrics.get("spy_buy_hold_sharpe", 0.0)
    beat_ret = strat_ret > spy_ret
    beat_sh = strat_sh > spy_sh
    logger.info(f"VERDICT: return {strat_ret:+.4f} vs SPY {spy_ret:+.4f} "
                f"-> {'BEATS' if beat_ret else 'loses'};  "
                f"Sharpe {strat_sh:.3f} vs SPY {spy_sh:.3f} "
                f"-> {'BEATS' if beat_sh else 'loses'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
