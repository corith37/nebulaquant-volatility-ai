"""Short-swing day-to-day sweep: 1/2/3-month momentum x 3/5/10-day holds.

The user asked to refocus the tool on "day to day stock trading using samples of
1 to 3 months": signal from only the most recent 1-3 months of price action,
held for short swings (3-10 days), re-scanned daily. For a rules-based
cross-sectional momentum strategy that maps cleanly onto two knobs:

  * data/lookback window  -> which momentum rank we sort by:
        xs_ret20_rank (~1 month), xs_ret42_rank (~2 months), xs_ret60_rank (~3 months)
  * hold length           -> max_hold_days in {3, 5, 10}

Everything else matches the deployed momentum strategy (top-decile, regime gate
SPY>200d SMA + benign VIX, name trending > 20d & 200d SMA, daily entries, time
exit, NO ATR barriers) and runs in the SAME production event-driven engine with
costs + slippage + intrabar fills. Each cell is benchmarked against SPY
buy-and-hold over the identical span so we can honestly say which (if any) of the
shorter-horizon variants actually beats the benchmark NET OF the higher turnover.

This is a research sweep, not a deploy step: it prints a comparison grid and
writes reports/backtests/short_swing_sweep.csv. It does not overwrite the
deployed momentum_* outputs.
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

# lookback label -> cross-sectional momentum rank column
LOOKBACKS = {
    "1mo (20d)": "xs_ret20_rank",
    "2mo (42d)": "xs_ret42_rank",
    "3mo (60d)": "xs_ret60_rank",
}
HOLDS = [3, 5, 10]


def main() -> int:
    logger = setup_logger("nebulaquant.short_swing_sweep")
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
        logger.error(f"Missing rank columns {missing} - rebuild the dataset.")
        return 1

    top_pctile = float(mom.get("top_pctile", 0.90))
    rows = []

    for lb_label, score_col in LOOKBACKS.items():
        for hold in HOLDS:
            d = df.copy()
            d["score"] = d[score_col]
            risk = RiskConfig(
                # ATR barriers disabled -> pure time exit at `hold` days.
                tp_atr_mult=100.0,
                sl_atr_mult=100.0,
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
            )
            trade_log, equity_curve, metrics = portfolio_backtest(
                d, cfg=risk,
                initial_capital=bt.get("initial_capital", 10_000),
                entry_min_rel_volume=float(mom.get("entry_min_rel_volume", 0.0)),
            )
            bench = benchmarks(d, cfg=risk, n_trades=metrics["n_trades"])
            spy_ret = bench.get("spy_buy_hold_return", 0.0)
            spy_sh = bench.get("spy_buy_hold_sharpe", 0.0)
            row = {
                "lookback": lb_label,
                "hold_days": hold,
                "total_return": metrics.get("total_return", 0.0),
                "sharpe": metrics.get("sharpe", 0.0),
                "max_drawdown": metrics.get("max_drawdown", 0.0),
                "win_rate": metrics.get("win_rate", 0.0),
                "profit_factor": metrics.get("profit_factor", 0.0),
                "expectancy": metrics.get("expectancy", 0.0),
                "n_trades": metrics.get("n_trades", 0),
                "spy_return": spy_ret,
                "spy_sharpe": spy_sh,
                "beats_spy_return": metrics.get("total_return", 0.0) > spy_ret,
                "beats_spy_sharpe": metrics.get("sharpe", 0.0) > spy_sh,
            }
            rows.append(row)
            logger.info(
                f"{lb_label:>10} | hold {hold:>2}d | ret {row['total_return']:+.3f} "
                f"sharpe {row['sharpe']:.2f} maxDD {row['max_drawdown']:.3f} "
                f"win {row['win_rate']:.2f} PF {row['profit_factor']:.2f} "
                f"n {row['n_trades']:>4} | SPY ret {spy_ret:+.3f} sharpe {spy_sh:.2f} "
                f"-> {'BEATS' if row['beats_spy_return'] and row['beats_spy_sharpe'] else ('ret-only' if row['beats_spy_return'] else ('sharpe-only' if row['beats_spy_sharpe'] else 'loses'))}"
            )

    res = pd.DataFrame(rows)
    out_path = paths["reports_backtests"] / "short_swing_sweep.csv"
    res.to_csv(out_path, index=False)
    logger.info(f"Saved sweep grid -> {out_path}")

    # Honest verdict.
    both = res[res["beats_spy_return"] & res["beats_spy_sharpe"]]
    print("\n=== SHORT-SWING SWEEP (regime-gated top-decile momentum) ===")
    cols = ["lookback", "hold_days", "total_return", "sharpe", "max_drawdown",
            "win_rate", "profit_factor", "n_trades", "spy_return", "spy_sharpe"]
    with pd.option_context("display.float_format", lambda x: f"{x:.3f}"):
        print(res[cols].to_string(index=False))
    print(f"\nSPY over span: return {res['spy_return'].iloc[0]:+.3f}, "
          f"Sharpe {res['spy_sharpe'].iloc[0]:.2f}")
    if both.empty:
        print("\nVERDICT: NO short-swing cell beats SPY on BOTH return and Sharpe. "
              "Per the project's honesty rule, the short-horizon variant does not "
              "demonstrate an edge over buy-and-hold net of its higher turnover.")
    else:
        best = both.sort_values("sharpe", ascending=False).iloc[0]
        print(f"\nVERDICT: {len(both)} cell(s) beat SPY on both metrics. Best by Sharpe: "
              f"{best['lookback']} / hold {int(best['hold_days'])}d -> "
              f"return {best['total_return']:+.3f}, Sharpe {best['sharpe']:.2f}, "
              f"maxDD {best['max_drawdown']:.3f}, {int(best['n_trades'])} trades.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
