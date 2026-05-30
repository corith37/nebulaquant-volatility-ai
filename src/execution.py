"""Broker execution layer for paper (and, later, live) trading of the picks.

Two clearly separated parts:

* ``plan_orders(...)`` -- **pure** decision logic, fully testable with no network.
  Given today's regime-gated picks, what the account currently holds, and the
  live ledger's exit timing, it decides which positions to SELL (past their
  ~20-day time exit) and which new names to BUY (filling free slots, regime
  permitting). It deliberately does NOT size orders or touch a broker.

* ``AlpacaBroker`` -- a thin wrapper over ``alpaca-py`` for a **paper** (default)
  or live account. The SDK import is lazy/guarded so importing this module never
  requires ``alpaca-py`` to be installed.

Time-exit semantics match the validated backtest: a risk-off regime blocks new
buys but does NOT force-liquidate existing holds -- they run to their time exit.

Nothing here trades real money unless a caller explicitly builds a live broker
(``paper=False``) with funded live keys.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

ENV_KEY = "APCA_API_KEY_ID"
ENV_SECRET = "APCA_API_SECRET_KEY"


def load_alpaca_keys(root: Optional[Path] = None) -> tuple[Optional[str], Optional[str]]:
    """Read Alpaca keys from environment, falling back to a local ``.env`` file.

    The ``.env`` (lines ``KEY=VALUE``) is gitignored, so keys never enter the repo.
    Returns ``(key, secret)``; either may be ``None`` if not found.
    """
    key, secret = os.environ.get(ENV_KEY), os.environ.get(ENV_SECRET)
    if key and secret:
        return key, secret
    root = Path(root) if root else Path(__file__).resolve().parent.parent
    envf = root / ".env"
    if envf.exists():
        vals: dict[str, str] = {}
        for line in envf.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            vals[k.strip()] = v.strip().strip('"').strip("'")
        key = key or vals.get(ENV_KEY)
        secret = secret or vals.get(ENV_SECRET)
    return key, secret


def plan_orders(
    picks: list[dict],
    holdings: Iterable[str],
    open_ledger: dict,
    today,
    *,
    target_positions: int = 3,
    regime_on: bool = True,
) -> dict:
    """Decide which tickers to sell and buy. Pure: no sizing, no I/O.

    Parameters
    ----------
    picks : ranked best-first list of gated pick dicts (need a ``ticker`` key).
    holdings : tickers currently held in the (paper) account.
    open_ledger : ``{ticker: target_exit_date}`` for open ledger positions; used
        only to time SELLs.
    today : the decision date.
    target_positions : max concurrent positions (free slots = target - held).
    regime_on : when False, no new buys are proposed (holds still time-exit).

    Returns ``{"sells": [ticker, ...], "buys": [ticker, ...]}``.
    """
    today = pd.to_datetime(today).normalize()
    holdings = set(map(str, holdings))

    sells = []
    for t in sorted(holdings):
        tex = open_ledger.get(t)
        if tex is not None and not pd.isna(tex) and pd.to_datetime(tex).normalize() <= today:
            sells.append(t)

    held_after = holdings - set(sells)
    buys: list[str] = []
    if regime_on:
        slots = max(0, int(target_positions) - len(held_after))
        for p in picks:
            if slots <= 0:
                break
            tk = str(p["ticker"])
            if tk in held_after or tk in buys:
                continue
            buys.append(tk)
            slots -= 1
    return {"sells": sells, "buys": buys}


class AlpacaBroker:
    """Minimal Alpaca wrapper (paper by default). Lazy-imports ``alpaca-py``."""

    def __init__(self, key: str, secret: str, paper: bool = True):
        try:
            from alpaca.trading.client import TradingClient
        except ImportError as e:  # pragma: no cover - depends on optional dep
            raise RuntimeError(
                "alpaca-py is not installed. Run: pip install alpaca-py"
            ) from e
        self.paper = bool(paper)
        self.client = TradingClient(key, secret, paper=self.paper)

    # --- reads ---
    def account(self) -> dict:
        a = self.client.get_account()
        return {
            "status": str(a.status),
            "cash": float(a.cash),
            "equity": float(a.equity),
            "buying_power": float(a.buying_power),
            "portfolio_value": float(a.portfolio_value),
        }

    def positions(self) -> dict:
        out: dict[str, dict] = {}
        for p in self.client.get_all_positions():
            out[p.symbol] = {
                "qty": float(p.qty),
                "market_value": float(p.market_value),
                "avg_entry_price": float(p.avg_entry_price),
                "unrealized_pl": float(p.unrealized_pl),
            }
        return out

    def market_open(self) -> bool:
        return bool(self.client.get_clock().is_open)

    # --- writes ---
    def buy_notional(self, symbol: str, notional: float):
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        req = MarketOrderRequest(
            symbol=symbol,
            notional=round(float(notional), 2),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        return self.client.submit_order(order_data=req)

    def close(self, symbol: str):
        return self.client.close_position(symbol)
