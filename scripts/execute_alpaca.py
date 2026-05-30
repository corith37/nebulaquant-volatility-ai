"""Execute the momentum picks on an Alpaca account (PAPER by default, dry-run by default).

This is the automation layer: it reads the SAME regime-gated momentum picks the
dashboard shows (``src.scanner.today_picks``), checks what the account already
holds, sells anything past its ~20-day time exit, and buys the top free slots --
equal-weight, notional (fractional) orders so a small account works.

SAFETY BY DEFAULT:
  * PAPER account unless ``--live`` is passed (and ``--live`` demands ``--i-understand``).
  * DRY RUN unless ``--submit`` is passed: it prints the exact plan and stops.
  * Refuses to submit when the market is closed (notional orders need regular
    hours) unless ``--force``.
  * Never force-liquidates on a risk-off regime -- holds run to their time exit,
    matching the validated backtest. Risk-off only blocks NEW buys.

Keys come from env vars APCA_API_KEY_ID / APCA_API_SECRET_KEY, or a gitignored
``.env`` at the project root. See docs/ALPACA_PAPER_SETUP.md.

Examples:
    python scripts/execute_alpaca.py                  # dry run, paper: show the plan
    python scripts/execute_alpaca.py --submit         # actually trade the PAPER account
    python scripts/execute_alpaca.py --budget 200     # size as if equity were $200
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import ensure_paths, load_config
from src.execution import AlpacaBroker, load_alpaca_keys, plan_orders
from src.live_paper import add_picks, close_positions, load_ledger
from src.scanner import today_picks
from src.utils import setup_logger


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trade the momentum picks on Alpaca (paper by default).")
    p.add_argument("--submit", action="store_true", help="Actually place orders (default: dry run).")
    p.add_argument("--live", action="store_true", help="Use a LIVE account (default: paper).")
    p.add_argument("--i-understand", action="store_true", help="Required with --live: real money.")
    p.add_argument("--positions", type=int, default=None, help="Max concurrent positions.")
    p.add_argument("--deploy-fraction", type=float, default=None, help="Fraction of equity to deploy.")
    p.add_argument("--budget", type=float, default=None, help="Override equity used for sizing ($).")
    p.add_argument("--force", action="store_true", help="Submit even if the market is closed.")
    return p.parse_args()


def main() -> int:
    logger = setup_logger("nebulaquant.execute")
    args = _parse_args()
    cfg = load_config()
    paths = ensure_paths()
    ex = cfg.get("execution", {})

    paper = not args.live
    if args.live and not args.i_understand:
        logger.error("--live trades REAL money. Re-run with --live --i-understand to proceed.")
        return 2

    target_positions = int(args.positions if args.positions is not None else ex.get("target_positions", 3))
    deploy_fraction = float(args.deploy_fraction if args.deploy_fraction is not None else ex.get("deploy_fraction", 0.90))
    min_notional = float(ex.get("min_notional", 1.0))
    hold_days = int(ex.get("hold_days", cfg.get("momentum", {}).get("hold_days", 20)))

    # --- keys ---
    key, secret = load_alpaca_keys(paths["root"])
    if not key or not secret:
        logger.error(
            "Missing Alpaca keys. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY (env or .env). "
            "See docs/ALPACA_PAPER_SETUP.md."
        )
        return 1

    # --- today's gated picks (same source as the dashboard) ---
    picks_info = today_picks(cfg, paths["data_processed"], top_n=target_positions)
    if picks_info.get("empty"):
        logger.error("No picks: processed data missing. Run download_data + build_dataset.")
        return 1

    # --- broker ---
    try:
        broker = AlpacaBroker(key, secret, paper=paper)
        acct = broker.account()
        positions = broker.positions()
        market_open = broker.market_open()
    except RuntimeError as e:
        logger.error(str(e))
        return 1
    except Exception as e:  # noqa: BLE001  (network/auth)
        logger.error(f"Alpaca connection failed: {e}")
        return 1

    # --- ledger exit timing (ticker -> target_exit_date) ---
    ledger = load_ledger(paths["data_paper"])
    open_led = ledger[ledger["status"] == "open"] if not ledger.empty else ledger
    exit_map = dict(zip(open_led["ticker"].astype(str), open_led["target_exit_date"])) if not open_led.empty else {}

    today = date.today()
    plan = plan_orders(
        picks_info["picks"], positions.keys(), exit_map, today,
        target_positions=target_positions, regime_on=picks_info["regime_on"],
    )

    # --- size the buys (equal-weight, capped by available cash) ---
    equity = float(args.budget) if args.budget is not None else acct["equity"]
    per_slot = round((equity * deploy_fraction) / max(1, target_positions), 2)
    cash = acct["cash"]
    price_by_ticker = {p["ticker"]: p["price"] for p in picks_info["picks"]}
    buy_orders = []
    for tk in plan["buys"]:
        notional = round(min(per_slot, cash), 2)
        if notional < min_notional:
            continue
        buy_orders.append({"ticker": tk, "notional": notional, "price": price_by_ticker.get(tk)})
        cash -= notional

    # --- print the plan ---
    mode = "PAPER" if paper else "LIVE (real money)"
    print(f"\n========== ALPACA EXECUTION PLAN [{mode}] ==========")
    print(f"  Account   : status={acct['status']}  equity=${acct['equity']:,.2f}  "
          f"cash=${acct['cash']:,.2f}  market={'OPEN' if market_open else 'CLOSED'}")
    print(f"  Signal    : {picks_info['data_date']}  regime={'ON' if picks_info['regime_on'] else 'OFF (no new buys)'}")
    print(f"  Hold      : ~{hold_days} trading days -> time exit ~{picks_info['target_exit_date']}")
    print(f"  Sizing    : {target_positions} slots, {deploy_fraction:.0%} of "
          f"${equity:,.2f} -> ~${per_slot:,.2f}/slot")

    held = ", ".join(f"{t}(${v['market_value']:,.0f})" for t, v in positions.items()) or "(none)"
    print(f"  Holding   : {held}")
    if plan["sells"]:
        print("  SELL (past target exit):")
        for t in plan["sells"]:
            print(f"      - {t}  mktval ${positions.get(t, {}).get('market_value', 0):,.2f}")
    else:
        print("  SELL      : (none due)")
    if buy_orders:
        print("  BUY (notional):")
        for o in buy_orders:
            print(f"      - {o['ticker']:5s}  ${o['notional']:,.2f}  (~{o['price'] and o['notional']/o['price']:.4f} sh @ ${o['price']:,.2f})")
    else:
        why = "regime OFF" if not picks_info["regime_on"] else "no free slots / nothing qualifies"
        print(f"  BUY       : (none - {why})")

    if not args.submit:
        print("\n  DRY RUN - no orders sent. Re-run with --submit during market hours to execute.")
        print("  (Research/paper. No real orders placed.)\n")
        return 0

    if not market_open and not args.force:
        print("\n  Market is CLOSED - notional orders need regular hours. "
              "Re-run during market hours, or pass --force.\n")
        return 0

    # --- execute: sells first (free cash), then buys ---
    sold = []
    for t in plan["sells"]:
        try:
            broker.close(t)
            sold.append(t)
            print(f"  [sell] closed {t}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"sell {t} failed: {e}")
    if sold:
        close_positions(paths["data_paper"], sold, paths["data_processed"], today=today)

    exec_target = str((pd.Timestamp(today) + pd.offsets.BDay(hold_days)).date())
    bought_records = []
    for o in buy_orders:
        try:
            broker.buy_notional(o["ticker"], o["notional"])
            px = o["price"] or 0.0
            bought_records.append({
                "ticker": o["ticker"],
                "entry_date": str(today),
                "entry_price": px,
                "shares": (o["notional"] / px) if px else 0.0,
                "alloc": o["notional"],
                "hold_days": hold_days,
                "target_exit_date": exec_target,
            })
            print(f"  [buy ] {o['ticker']} ${o['notional']:,.2f}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"buy {o['ticker']} failed: {e}")
    if bought_records:
        add_picks(paths["data_paper"], bought_records)

    print(f"\n  Done. Sold {len(sold)}, bought {len(bought_records)}. "
          f"{'PAPER - no real money.' if paper else 'LIVE orders submitted.'}")
    print("  Track open positions in the dashboard's Today's Picks tab.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
