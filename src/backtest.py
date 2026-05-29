"""Walk-forward, portfolio-level backtest for the long-only volatility strategy.

Two pieces:
  1. `walk_forward_predict` - retrain the two-stage model on each purged
     walk-forward fold and score only that fold's out-of-sample rows, then
     (optionally) append the final holdout scored by a model trained on all dev
     data. The result is a fully out-of-sample prediction frame.
  2. `portfolio_backtest` - an event-driven portfolio simulation over those OOS
     predictions: signals fire at a bar's close and ENTER at the next bar's
     close (no look-ahead); positions exit on ATR-scaled take-profit / stop-loss
     (intrabar high/low touch; if both touch in one bar the stop is assumed
     first) or a max-hold time barrier. Sizing is confidence- and
     volatility-aware with per-position and total-exposure caps.

Benchmarks (SPY buy-and-hold and a random-entry baseline) are reported so any
edge has to be demonstrated, not assumed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.risk import RiskConfig, confidence_fraction
from src.train_model import TwoStageModel, train_two_stage
from src.utils import setup_logger
from src.validation import (
    WalkForwardConfig,
    describe_splits,
    purged_walkforward_splits,
    train_holdout_split,
)

logger = setup_logger("nebulaquant.backtest")

# Columns the portfolio simulation needs from each prediction row.
_PRED_COLS = [
    "Date", "Ticker", "Close", "High", "Low", "atr_14", "atr_pct",
    "sma_20", "rel_volume", "price_above_sma20",
    # Regime-gate inputs (optional; gate is skipped if absent).
    "spy_above_sma200", "vix_pctile_252",
]


# ----------------------------------------------------------------------------
# Walk-forward prediction
# ----------------------------------------------------------------------------

def walk_forward_predict(
    df: pd.DataFrame,
    feat_cols: List[str],
    wf_cfg: WalkForwardConfig,
    params: Optional[Dict] = None,
    calibrate: bool = False,
    two_stage: bool = True,
    random_state: int = 42,
    include_holdout: bool = True,
) -> pd.DataFrame:
    """Score every fold's OOS rows with a model trained only on its past."""
    df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    dev, holdout = train_holdout_split(df, wf_cfg.holdout_frac)
    dev = dev.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    splits = purged_walkforward_splits(dev, wf_cfg)
    if not splits:
        raise RuntimeError("No walk-forward splits produced; need more history.")
    logger.info("Walk-forward folds:\n%s", describe_splits(dev, splits).to_string(index=False))

    preds: List[pd.DataFrame] = []
    for k, (tr_idx, te_idx) in enumerate(splits):
        train = dev.iloc[tr_idx]
        test = dev.iloc[te_idx]
        model = train_two_stage(train, feat_cols, params, calibrate, two_stage, random_state)
        scored = _score_rows(model, test, fold=k)
        preds.append(scored)

    # Final fold: train on ALL dev, score the untouched holdout.
    if include_holdout and not holdout.empty:
        model = train_two_stage(dev, feat_cols, params, calibrate, two_stage, random_state)
        preds.append(_score_rows(model, holdout, fold=len(splits)))

    out = pd.concat(preds, ignore_index=True)
    return out.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def _score_rows(model: TwoStageModel, rows: pd.DataFrame, fold: int) -> pd.DataFrame:
    scores = model.predict(rows)
    keep = [c for c in _PRED_COLS if c in rows.columns]
    out = rows[keep].copy()
    out["expansion_prob"] = scores["expansion_prob"].to_numpy()
    out["long_prob"] = scores["long_prob"].to_numpy()
    out["score"] = scores["score"].to_numpy()
    out["fold"] = fold
    return out


# ----------------------------------------------------------------------------
# Portfolio simulation
# ----------------------------------------------------------------------------

def portfolio_backtest(
    preds: pd.DataFrame,
    cfg: RiskConfig,
    initial_capital: float = 10_000.0,
    entry_min_rel_volume: float = 1.0,
    ref_atr_pct: Optional[float] = None,
    collect_open: Optional[list] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Event-driven long-only portfolio sim over OOS predictions.

    If ``collect_open`` (a list) is provided, positions still open at the final
    bar are appended to it, marked-to-market at that bar's close. This is how the
    paper-trading layer reads the account's current holdings; it does not change
    any trading behaviour or the returned trade_log/equity/metrics.
    """
    if preds.empty:
        return pd.DataFrame(), pd.DataFrame(columns=["date", "equity"]), _empty_metrics(initial_capital)

    preds = preds.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    if ref_atr_pct is None:
        ref_atr_pct = float(preds["atr_pct"].median())

    # Fast row lookup by (date, ticker) and a sorted global timeline.
    row_lookup = {(r.Date, r.Ticker): r for r in preds.itertuples(index=False)}
    dates = sorted(preds["Date"].unique())
    next_date = {d: dates[i + 1] for i, d in enumerate(dates[:-1])}

    cash = initial_capital
    positions: Dict[str, dict] = {}     # ticker -> position
    pending: Dict[str, dict] = {}       # ticker -> signal to enter next date
    trades: List[dict] = []
    equity_curve: List[dict] = []

    for d in dates:
        # ---- 1. Manage exits for open positions on this bar ----
        for ticker in list(positions.keys()):
            pos = positions[ticker]
            bar = row_lookup.get((d, ticker))
            if bar is None:
                continue
            exit_price = exit_reason = None
            if bar.Low <= pos["sl"]:
                exit_price, exit_reason = pos["sl"], "stop_loss"
            elif bar.High >= pos["tp"]:
                exit_price, exit_reason = pos["tp"], "take_profit"
            else:
                pos["bars_held"] += 1
                if pos["bars_held"] >= cfg.max_hold_days:
                    exit_price, exit_reason = float(bar.Close), "max_hold"
            if exit_price is not None:
                fill = exit_price * (1 - cfg.slippage_pct)
                proceeds = fill * pos["shares"]
                commission = proceeds * cfg.commission_pct
                cash += proceeds - commission
                pnl = (proceeds - commission) - pos["cost"]
                trades.append({
                    "ticker": ticker,
                    "entry_date": pos["entry_date"], "exit_date": d,
                    "entry_price": round(pos["entry_price"], 4),
                    "exit_price": round(fill, 4),
                    "shares": pos["shares"],
                    "pnl": round(pnl, 2),
                    "return_pct": round(pnl / pos["cost"], 6) if pos["cost"] else 0.0,
                    "hold_days": pos["bars_held"],
                    "exit_reason": exit_reason,
                    "score": round(pos["score"], 4),
                })
                del positions[ticker]

        # ---- 2. Execute pending entries scheduled for this date ----
        equity = cash + sum(p["shares"] * _close_or(row_lookup, d, t, p)
                            for t, p in positions.items())
        exposure = sum(p["shares"] * _close_or(row_lookup, d, t, p) for t, p in positions.items())
        for ticker, sig in list(pending.items()):
            if ticker in positions:
                continue
            bar = row_lookup.get((d, ticker))
            if bar is None:
                continue
            price = float(bar.Close) * (1 + cfg.slippage_pct)
            frac = confidence_fraction(sig["score"], cfg.prob_threshold, sig["atr_pct"], cfg, ref_atr_pct)
            if frac <= 0:
                continue
            if (exposure + frac * equity) > cfg.max_total_exposure * equity:
                continue
            shares = int((equity * frac) // price)
            cost = shares * price
            commission = cost * cfg.commission_pct
            if shares <= 0 or cost + commission > cash:
                continue
            cash -= cost + commission
            exposure += cost
            atr = sig["atr_14"]
            positions[ticker] = {
                "entry_date": d, "entry_price": price, "shares": shares,
                "cost": cost + commission,
                "tp": price + cfg.tp_atr_mult * atr,
                "sl": price - cfg.sl_atr_mult * atr,
                "bars_held": 0, "score": sig["score"],
            }
        pending = {}

        # ---- 3. Generate new signals from this bar (enter NEXT date) ----
        nd = next_date.get(d)
        if nd is not None:
            todays = preds[preds["Date"] == d]
            mask = (
                (todays["score"] >= cfg.prob_threshold)
                & (todays["rel_volume"] >= entry_min_rel_volume)
            )
            # Per-name trend filters (configurable). Momentum requires price
            # above the 20d SMA; mean-reversion instead requires a long-term
            # uptrend (above 200d) while allowing short-term weakness.
            if cfg.require_above_sma20 and "price_above_sma20" in todays.columns:
                mask &= (todays["price_above_sma20"] == 1)
            if cfg.require_above_sma200 and "price_above_sma200" in todays.columns:
                mask &= (todays["price_above_sma200"] == 1)
            # Regime gate: broad-market uptrend + non-panic VIX.
            if cfg.regime_filter:
                if "spy_above_sma200" in todays.columns:
                    mask &= (todays["spy_above_sma200"] == 1)
                if "vix_pctile_252" in todays.columns:
                    mask &= (todays["vix_pctile_252"] <= cfg.max_vix_pctile)
            cand = todays[mask].sort_values("score", ascending=False)
            for r in cand.itertuples(index=False):
                if r.Ticker in positions or r.Ticker in pending:
                    continue
                pending[r.Ticker] = {
                    "score": float(r.score), "atr_14": float(r.atr_14),
                    "atr_pct": float(r.atr_pct),
                }

        # ---- 4. Mark-to-market equity for the curve ----
        mtm = cash + sum(p["shares"] * _close_or(row_lookup, d, t, p) for t, p in positions.items())
        equity_curve.append({"date": d, "equity": round(mtm, 2)})

    # Report still-open positions (current holdings) for paper trading, marked
    # to market at the final available bar.
    if collect_open is not None and dates:
        last_d = dates[-1]
        for t, p in positions.items():
            bar = row_lookup.get((last_d, t))
            last_close = float(bar.Close) if bar is not None else p["entry_price"]
            mkt_value = p["shares"] * last_close
            collect_open.append({
                "ticker": t,
                "entry_date": p["entry_date"],
                "entry_price": round(p["entry_price"], 4),
                "shares": p["shares"],
                "cost_basis": round(p["cost"], 2),
                "last_close": round(last_close, 4),
                "market_value": round(mkt_value, 2),
                "unrealized_pnl": round(mkt_value - p["cost"], 2),
                "unrealized_pct": round(mkt_value / p["cost"] - 1, 6) if p["cost"] else 0.0,
                "bars_held": p["bars_held"],
                "score": round(p["score"], 4),
            })

    trade_log = pd.DataFrame(trades)
    eq_df = pd.DataFrame(equity_curve)
    metrics = compute_metrics(trade_log, eq_df, initial_capital)
    return trade_log, eq_df, metrics


def _close_or(lookup, date, ticker, pos) -> float:
    bar = lookup.get((date, ticker))
    return float(bar.Close) if bar is not None else pos["entry_price"]


# ----------------------------------------------------------------------------
# Metrics + benchmarks
# ----------------------------------------------------------------------------

def _empty_metrics(initial_capital: float) -> dict:
    return {
        "total_return": 0.0, "final_equity": initial_capital, "n_trades": 0,
        "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "profit_factor": 0.0,
        "expectancy": 0.0, "max_drawdown": 0.0, "sharpe": 0.0,
        "avg_hold_days": 0.0, "best_trade": 0.0, "worst_trade": 0.0,
    }


def compute_metrics(trade_log: pd.DataFrame, equity_curve: pd.DataFrame, initial_capital: float) -> dict:
    if trade_log.empty:
        return _empty_metrics(initial_capital)

    final_eq = float(equity_curve["equity"].iloc[-1]) if not equity_curve.empty else initial_capital
    total_return = final_eq / initial_capital - 1
    wins = trade_log[trade_log["pnl"] > 0]
    losses = trade_log[trade_log["pnl"] <= 0]
    win_rate = len(wins) / len(trade_log)
    avg_win = float(wins["pnl"].mean()) if len(wins) else 0.0
    avg_loss = float(losses["pnl"].mean()) if len(losses) else 0.0
    gross_win = float(wins["pnl"].sum())
    gross_loss = float(-losses["pnl"].sum())
    profit_factor = gross_win / gross_loss if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
    expectancy = float(trade_log["pnl"].mean())

    # Daily-equity Sharpe + drawdown.
    sharpe = max_dd = 0.0
    if not equity_curve.empty and len(equity_curve) > 2:
        eq = equity_curve["equity"].to_numpy()
        daily_ret = np.diff(eq) / eq[:-1]
        if daily_ret.std() > 0:
            sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))
        running_max = np.maximum.accumulate(eq)
        max_dd = float(((eq - running_max) / running_max).min())

    return {
        "total_return": round(float(total_return), 6),
        "final_equity": round(final_eq, 2),
        "n_trades": int(len(trade_log)),
        "win_rate": round(float(win_rate), 6),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(float(profit_factor), 4),
        "expectancy": round(expectancy, 4),
        "max_drawdown": round(max_dd, 6),
        "sharpe": round(sharpe, 4),
        "avg_hold_days": round(float(trade_log["hold_days"].mean()), 3),
        "best_trade": round(float(trade_log["pnl"].max()), 2),
        "worst_trade": round(float(trade_log["pnl"].min()), 2),
    }


def _simulate_long_outcome(g: pd.DataFrame, pos: int, cfg: RiskConfig) -> Optional[float]:
    """Return fractional PnL of a single long entered at bar pos+1's close."""
    highs, lows, closes, atrs = (
        g["High"].to_numpy(), g["Low"].to_numpy(), g["Close"].to_numpy(), g["atr_14"].to_numpy(),
    )
    n = len(g)
    entry_idx = pos + 1
    if entry_idx >= n or not np.isfinite(atrs[pos]) or atrs[pos] <= 0:
        return None
    entry = closes[entry_idx] * (1 + cfg.slippage_pct)
    tp = entry + cfg.tp_atr_mult * atrs[pos]
    sl = entry - cfg.sl_atr_mult * atrs[pos]
    for j in range(entry_idx, min(entry_idx + cfg.max_hold_days, n - 1) + 1):
        if lows[j] <= sl:
            return (sl * (1 - cfg.slippage_pct)) / entry - 1
        if highs[j] >= tp:
            return (tp * (1 - cfg.slippage_pct)) / entry - 1
    exit_idx = min(entry_idx + cfg.max_hold_days, n - 1)
    return (closes[exit_idx] * (1 - cfg.slippage_pct)) / entry - 1


def benchmarks(preds: pd.DataFrame, cfg: RiskConfig, n_trades: int, seed: int = 0) -> dict:
    """SPY buy-and-hold and random-entry baselines over the OOS window."""
    out = {}

    # SPY buy & hold.
    spy = preds[preds["Ticker"] == "SPY"].sort_values("Date")
    if len(spy) > 2:
        c = spy["Close"].to_numpy()
        out["spy_buy_hold_return"] = round(float(c[-1] / c[0] - 1), 6)
        dr = np.diff(c) / c[:-1]
        out["spy_buy_hold_sharpe"] = round(float(dr.mean() / dr.std() * np.sqrt(252)), 4) if dr.std() > 0 else 0.0
    else:
        out["spy_buy_hold_return"] = 0.0
        out["spy_buy_hold_sharpe"] = 0.0

    # Random-entry baseline: same trade count, random long entries, same exits.
    rng = np.random.default_rng(seed)
    pool = preds.reset_index(drop=True)
    n = max(min(n_trades, len(pool) - 1), 0)
    rets = []
    if n > 0:
        by_ticker = {t: g.sort_values("Date").reset_index(drop=True) for t, g in pool.groupby("Ticker")}
        idx = rng.choice(len(pool), size=min(n * 3, len(pool)), replace=False)
        for i in idx:
            ticker = pool.loc[i, "Ticker"]
            g = by_ticker[ticker]
            pos_matches = g.index[g["Date"] == pool.loc[i, "Date"]]
            if len(pos_matches) == 0:
                continue
            r = _simulate_long_outcome(g, int(pos_matches[0]), cfg)
            if r is not None:
                rets.append(r)
            if len(rets) >= n:
                break
    out["random_entry_avg_return"] = round(float(np.mean(rets)), 6) if rets else 0.0
    out["random_entry_n"] = len(rets)
    return out


# ----------------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------------

def save_backtest_outputs(
    trade_log: pd.DataFrame, metrics: dict, out_dir: Path,
    equity_curve: Optional[pd.DataFrame] = None,
    prefix: str = "",
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trade_log.to_csv(out_dir / f"{prefix}trade_log.csv", index=False)
    pd.DataFrame([metrics]).to_csv(out_dir / f"{prefix}backtest_metrics.csv", index=False)
    if equity_curve is not None and not equity_curve.empty:
        equity_curve.to_csv(out_dir / f"{prefix}equity_curve.csv", index=False)
    logger.info(f"Saved backtest outputs to {out_dir}")
