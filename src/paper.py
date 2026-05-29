"""Local paper-trading simulator (fake cash, no real orders).

Replays the deployed regime-gated cross-sectional momentum strategy over a recent
window using the SAME event-driven engine as the backtest
(``src.backtest.portfolio_backtest``). There is deliberately no second trading
code path: the paper account is byte-for-byte the validated strategy (identical
fills, costs, slippage, regime gate, sizing), so what you paper-trade is exactly
what was tested. "Replay recent history" makes this honest forward paper trading
on the most recent out-of-sample bars -- you get a full equity curve immediately
instead of waiting weeks for live days to accumulate.

This module places NO real orders and contacts no broker.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.backtest import benchmarks, portfolio_backtest
from src.risk import RiskConfig
from src.utils import setup_logger

logger = setup_logger("nebulaquant.paper")

# The deployed momentum signal: 60-day (3-month) cross-sectional momentum rank.
SCORE_COL = "xs_ret60_rank"


def momentum_risk_config(cfg: dict) -> RiskConfig:
    """Build the exact RiskConfig the deployed momentum backtest uses.

    Mirrors scripts/run_momentum_backtest.py: ATR barriers disabled (pure
    ~20-day time exit), equal-ish sizing, regime gate on. Kept here so the paper
    layer reproduces the validated strategy without copy-paste drift.
    """
    mom = cfg.get("momentum", {})
    bt = cfg.get("backtest", {})
    return RiskConfig(
        tp_atr_mult=float(mom.get("tp_atr_mult", 100.0)),
        sl_atr_mult=float(mom.get("sl_atr_mult", 100.0)),
        max_hold_days=int(mom.get("hold_days", 20)),
        commission_pct=bt.get("commission_pct", 0.001),
        slippage_pct=bt.get("slippage_pct", 0.001),
        position_size_pct=float(mom.get("position_size_pct", 0.12)),
        max_position_pct=float(mom.get("max_position_pct", 0.20)),
        max_total_exposure=float(mom.get("max_total_exposure", 1.0)),
        vol_target=bool(mom.get("vol_target", False)),
        prob_threshold=float(mom.get("top_pctile", 0.90)),
        regime_filter=bt.get("regime_filter", True),
        max_vix_pctile=bt.get("max_vix_pctile", 0.85),
        require_above_sma20=True,
        require_above_sma200=False,
    )


def slice_window(
    df: pd.DataFrame,
    lookback_months: Optional[int] = None,
    start=None,
    end=None,
):
    """Return the dataset restricted to the replay window plus its bounds."""
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    end = pd.to_datetime(end) if end is not None else d["Date"].max()
    if start is not None:
        start = pd.to_datetime(start)
    elif lookback_months is not None:
        start = end - pd.DateOffset(months=int(lookback_months))
    else:
        start = d["Date"].min()
    win = d[(d["Date"] >= start) & (d["Date"] <= end)].reset_index(drop=True)
    return win, start, end


def run_paper_session(
    df: pd.DataFrame,
    cfg: dict,
    capital: float,
    lookback_months: Optional[int] = None,
    start=None,
    end=None,
) -> dict:
    """Replay the momentum strategy over a window with fake cash.

    Returns a dict with trade_log, equity_curve, open_positions (current
    holdings, marked to market), combined metrics (incl. SPY benchmark), and a
    flat ``summary`` describing the account state.
    """
    risk = momentum_risk_config(cfg)
    entry_min_rel_volume = float(cfg.get("momentum", {}).get("entry_min_rel_volume", 0.0))

    window, start, end = slice_window(df, lookback_months, start, end)
    if window.empty or SCORE_COL not in window.columns:
        raise ValueError(
            f"Empty window or missing '{SCORE_COL}'. Rebuild the dataset or widen the window."
        )
    window["score"] = window[SCORE_COL]

    open_positions: list = []
    trade_log, equity_curve, metrics = portfolio_backtest(
        window,
        cfg=risk,
        initial_capital=capital,
        entry_min_rel_volume=entry_min_rel_volume,
        collect_open=open_positions,
    )
    bench = benchmarks(window, cfg=risk, n_trades=metrics["n_trades"])

    open_df = pd.DataFrame(open_positions)
    if not open_df.empty:
        open_df = open_df.sort_values("market_value", ascending=False).reset_index(drop=True)

    final_eq = float(equity_curve["equity"].iloc[-1]) if not equity_curve.empty else float(capital)
    invested = float(open_df["market_value"].sum()) if not open_df.empty else 0.0

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "strategy": "regime-gated 60d cross-sectional momentum (20d hold)",
        "window_start": str(pd.to_datetime(start).date()),
        "window_end": str(pd.to_datetime(end).date()),
        "trading_days": int(len(equity_curve)),
        "starting_capital": round(float(capital), 2),
        "final_equity": round(final_eq, 2),
        "total_return": metrics.get("total_return", 0.0),
        "sharpe": metrics.get("sharpe", 0.0),
        "max_drawdown": metrics.get("max_drawdown", 0.0),
        "n_closed_trades": metrics.get("n_trades", 0),
        "win_rate": metrics.get("win_rate", 0.0),
        "profit_factor": metrics.get("profit_factor", 0.0),
        "n_open_positions": int(len(open_df)),
        "open_market_value": round(invested, 2),
        "cash_estimate": round(final_eq - invested, 2),
        "spy_return": bench.get("spy_buy_hold_return", 0.0),
        "spy_sharpe": bench.get("spy_buy_hold_sharpe", 0.0),
        "beats_spy_return": bool(metrics.get("total_return", 0.0) > bench.get("spy_buy_hold_return", 0.0)),
        "beats_spy_sharpe": bool(metrics.get("sharpe", 0.0) > bench.get("spy_buy_hold_sharpe", 0.0)),
    }

    return {
        "trade_log": trade_log,
        "equity_curve": equity_curve,
        "open_positions": open_df,
        "metrics": {**metrics, **bench},
        "summary": summary,
    }


def save_paper_session(session: dict, out_dir: Path) -> None:
    """Persist the paper account state so it can be reviewed / shown in the app."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    session["equity_curve"].to_csv(out_dir / "paper_equity.csv", index=False)
    session["trade_log"].to_csv(out_dir / "paper_trades.csv", index=False)
    session["open_positions"].to_csv(out_dir / "paper_positions.csv", index=False)
    with open(out_dir / "paper_summary.json", "w", encoding="utf-8") as f:
        json.dump(session["summary"], f, indent=2)
    logger.info(f"Saved paper session -> {out_dir}")
