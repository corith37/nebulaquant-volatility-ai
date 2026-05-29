"""Paper-trade the deployed 20-day momentum strategy with fake cash.

Local simulator -- NO real orders, no broker. Replays the validated strategy over
a recent window (default: last 12 months) in the same event-driven engine used by
the backtest, with a fake-cash account. Prints the account summary, current
holdings, and how it compares to SPY over the same window; persists state under
data/paper/ so the dashboard can show it.

Examples:
    python scripts/paper_trade.py                      # last 12 months, $10k
    python scripts/paper_trade.py --months 6           # last 6 months
    python scripts/paper_trade.py --capital 25000      # $25k fake cash
    python scripts/paper_trade.py --start 2024-01-01 --end 2024-12-31
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import ensure_paths, load_config
from src.paper import run_paper_session, save_paper_session
from src.utils import setup_logger


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local fake-cash paper trader (momentum strategy).")
    p.add_argument("--months", type=int, default=None, help="Recent window length in months.")
    p.add_argument("--capital", type=float, default=None, help="Starting fake cash.")
    p.add_argument("--start", type=str, default=None, help="Window start date (YYYY-MM-DD).")
    p.add_argument("--end", type=str, default=None, help="Window end date (YYYY-MM-DD).")
    return p.parse_args()


def main() -> int:
    logger = setup_logger("nebulaquant.paper_trade")
    args = _parse_args()
    cfg = load_config()
    paths = ensure_paths()
    paper_cfg = cfg.get("paper", {})

    months = args.months if args.months is not None else (
        None if args.start else int(paper_cfg.get("lookback_months", 12))
    )
    capital = args.capital if args.capital is not None else float(paper_cfg.get("starting_capital", 10_000))

    dataset_path = paths["data_processed"] / "model_dataset.csv"
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}. Run build_dataset first.")
        return 1
    df = pd.read_csv(dataset_path, parse_dates=["Date"])

    try:
        session = run_paper_session(
            df, cfg, capital=capital, lookback_months=months, start=args.start, end=args.end,
        )
    except ValueError as e:
        logger.error(str(e))
        return 1

    save_paper_session(session, paths["data_paper"])
    s = session["summary"]

    print("\n=========== PAPER TRADING ACCOUNT (fake cash, no real orders) ===========")
    print(f"  Strategy        : {s['strategy']}")
    print(f"  Window          : {s['window_start']} -> {s['window_end']}  ({s['trading_days']} trading days)")
    print(f"  Starting cash   : ${s['starting_capital']:,.2f}")
    print(f"  Current equity  : ${s['final_equity']:,.2f}   ({s['total_return']:+.2%})")
    print(f"    of which cash : ${s['cash_estimate']:,.2f}")
    print(f"    in positions  : ${s['open_market_value']:,.2f}  ({s['n_open_positions']} open)")
    print(f"  Sharpe          : {s['sharpe']:.2f}      Max drawdown: {s['max_drawdown']:.2%}")
    print(f"  Closed trades   : {s['n_closed_trades']}   Win rate: {s['win_rate']:.1%}   PF: {s['profit_factor']:.2f}")
    print(f"  --- vs SPY buy & hold over the same window ---")
    print(f"  SPY return      : {s['spy_return']:+.2%}   (you: {s['total_return']:+.2%} -> "
          f"{'AHEAD' if s['beats_spy_return'] else 'behind'})")
    print(f"  SPY Sharpe      : {s['spy_sharpe']:.2f}   (you: {s['sharpe']:.2f} -> "
          f"{'AHEAD' if s['beats_spy_sharpe'] else 'behind'})")

    open_df = session["open_positions"]
    if not open_df.empty:
        print("\n  Current holdings (marked to market):")
        cols = ["ticker", "entry_date", "entry_price", "shares", "last_close",
                "market_value", "unrealized_pnl", "unrealized_pct", "bars_held"]
        show = open_df[cols].copy()
        show["entry_date"] = pd.to_datetime(show["entry_date"]).dt.date
        show["unrealized_pct"] = (show["unrealized_pct"] * 100).round(2)
        print(show.to_string(index=False))
    else:
        print("\n  No open positions at window end (strategy is in cash).")

    print(f"\n  Saved account state -> {paths['data_paper']}")
    print("  NOTE: research/paper only. No real orders were placed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
