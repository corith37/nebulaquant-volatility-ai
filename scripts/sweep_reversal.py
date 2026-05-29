"""Short-term mean-reversion sweep (a different signal family for short swings).

The momentum sweep showed momentum is a multi-week factor: shortening the hold to
3-10 days destroys its edge. The documented SHORT-horizon equity anomaly is the
opposite tilt -- short-term reversal: among names in a long-term uptrend, the ones
that just sold off the hardest tend to bounce over the next few days ("buy the dip
in an uptrend", Connors-style). This sweep tests that directly.

Signal (rules-based, trailing-only -> whole-period result is the honest OOS est.):
  * score = xs_rev{2,3,5}_rank  (most oversold over the last 2/3/5 days = highest)
  * enter the top decile (most oversold) ...
  * ... that are still in a long-term uptrend: price > 200d SMA (require_above_sma200)
  * ... and do NOT require price > 20d SMA (oversold names are below it by design)
  * broad-market regime gate ON (SPY > 200d SMA, benign VIX)
  * hold 2/3/5 days (reversion is fast), daily re-scan
  * two exit modes per cell: pure TIME exit, and a quick ATR PROFIT-TARGET exit

Runs in the production event-driven engine (costs, slippage, intrabar fills) and
benchmarks every cell against SPY buy-and-hold over the identical span. Writes
reports/backtests/reversal_sweep.csv. Does not touch the deployed momentum_* files.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.backtest import benchmarks, portfolio_backtest
from src.config import ensure_paths, load_config
from src.risk import RiskConfig
from src.utils import setup_logger

LOOKBACKS = {
    "rev2 (2d)": "xs_rev2_rank",
    "rev3 (3d)": "xs_rev3_rank",
    "rev5 (5d)": "xs_rev5_rank",
}
HOLDS = [2, 3, 5]
# exit_label -> (tp_atr_mult, sl_atr_mult). 100/100 == barriers disabled (time exit).
EXIT_MODES = {
    "time": (100.0, 100.0),
    "atr_tp": (1.5, 3.0),   # quick profit target, wide stop -> capture the bounce
}


def main() -> int:
    logger = setup_logger("nebulaquant.reversal_sweep")
    cfg = load_config()
    paths = ensure_paths()
    bt = cfg.get("backtest", {})
    mom = cfg.get("momentum", {})

    dataset_path = paths["data_processed"] / "model_dataset.csv"
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}. Run build_dataset first.")
        return 1
    df = pd.read_csv(dataset_path, parse_dates=["Date"])

    missing = [c for c in LOOKBACKS.values() if c not in df.columns]
    if missing:
        logger.error(f"Missing reversal columns {missing} - rebuild the dataset.")
        return 1

    top_pctile = float(mom.get("top_pctile", 0.90))
    rows = []

    for lb_label, score_col in LOOKBACKS.items():
        for hold in HOLDS:
            for ex_label, (tp, sl) in EXIT_MODES.items():
                d = df.copy()
                d["score"] = d[score_col]
                risk = RiskConfig(
                    tp_atr_mult=tp,
                    sl_atr_mult=sl,
                    max_hold_days=hold,
                    commission_pct=bt.get("commission_pct", 0.001),
                    slippage_pct=bt.get("slippage_pct", 0.001),
                    position_size_pct=float(mom.get("position_size_pct", 0.12)),
                    max_position_pct=float(mom.get("max_position_pct", 0.20)),
                    max_total_exposure=float(mom.get("max_total_exposure", 1.0)),
                    vol_target=False,
                    prob_threshold=top_pctile,
                    regime_filter=bt.get("regime_filter", True),
                    max_vix_pctile=bt.get("max_vix_pctile", 0.85),
                    # Mean-reversion trend filters: long-term uptrend, allow dip.
                    require_above_sma20=False,
                    require_above_sma200=True,
                )
                trade_log, equity_curve, metrics = portfolio_backtest(
                    d, cfg=risk,
                    initial_capital=bt.get("initial_capital", 10_000),
                    entry_min_rel_volume=float(mom.get("entry_min_rel_volume", 0.0)),
                )
                bench = benchmarks(d, cfg=risk, n_trades=metrics["n_trades"])
                spy_ret = bench.get("spy_buy_hold_return", 0.0)
                spy_sh = bench.get("spy_buy_hold_sharpe", 0.0)
                beats_r = metrics.get("total_return", 0.0) > spy_ret
                beats_s = metrics.get("sharpe", 0.0) > spy_sh
                rows.append({
                    "lookback": lb_label, "hold_days": hold, "exit": ex_label,
                    "total_return": metrics.get("total_return", 0.0),
                    "sharpe": metrics.get("sharpe", 0.0),
                    "max_drawdown": metrics.get("max_drawdown", 0.0),
                    "win_rate": metrics.get("win_rate", 0.0),
                    "profit_factor": metrics.get("profit_factor", 0.0),
                    "expectancy": metrics.get("expectancy", 0.0),
                    "avg_hold_days": metrics.get("avg_hold_days", 0.0),
                    "n_trades": metrics.get("n_trades", 0),
                    "spy_return": spy_ret, "spy_sharpe": spy_sh,
                    "beats_spy_return": beats_r, "beats_spy_sharpe": beats_s,
                })
                logger.info(
                    f"{lb_label:>10} | hold {hold}d | {ex_label:>6} | "
                    f"ret {metrics.get('total_return',0.0):+.3f} sharpe {metrics.get('sharpe',0.0):.2f} "
                    f"maxDD {metrics.get('max_drawdown',0.0):.3f} win {metrics.get('win_rate',0.0):.2f} "
                    f"PF {metrics.get('profit_factor',0.0):.2f} n {metrics.get('n_trades',0):>4} "
                    f"-> {'BEATS' if beats_r and beats_s else ('ret-only' if beats_r else ('sharpe-only' if beats_s else 'loses'))}"
                )

    res = pd.DataFrame(rows)
    out_path = paths["reports_backtests"] / "reversal_sweep.csv"
    res.to_csv(out_path, index=False)
    logger.info(f"Saved reversal sweep grid -> {out_path}")

    both = res[res["beats_spy_return"] & res["beats_spy_sharpe"]]
    print("\n=== SHORT-TERM REVERSAL SWEEP (oversold-in-uptrend, regime-gated) ===")
    cols = ["lookback", "hold_days", "exit", "total_return", "sharpe", "max_drawdown",
            "win_rate", "profit_factor", "avg_hold_days", "n_trades"]
    with pd.option_context("display.float_format", lambda x: f"{x:.3f}"):
        print(res[cols].to_string(index=False))
    print(f"\nSPY over span: return {res['spy_return'].iloc[0]:+.3f}, Sharpe {res['spy_sharpe'].iloc[0]:.2f}")
    if both.empty:
        print("\nVERDICT: NO reversal cell beats SPY on BOTH return and Sharpe. "
              "Short-term reversal does not demonstrate an edge over buy-and-hold "
              "net of costs in this universe.")
    else:
        best = both.sort_values("sharpe", ascending=False).iloc[0]
        print(f"\nVERDICT: {len(both)} cell(s) beat SPY on both metrics. Best by Sharpe: "
              f"{best['lookback']} / hold {int(best['hold_days'])}d / {best['exit']} -> "
              f"return {best['total_return']:+.3f}, Sharpe {best['sharpe']:.2f}, "
              f"maxDD {best['max_drawdown']:.3f}, {int(best['n_trades'])} trades.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
