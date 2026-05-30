"""NebulaQuant Volatility AI - Streamlit dashboard.

Long-only research tool. The deployed strategy is **regime-gated cross-sectional
momentum** (it beats SPY on return and Sharpe; see the How-it-works tab for how the
earlier squeeze->expansion ML theses failed the edge gate). Tabs:

    1. Overview        - momentum strategy metrics vs SPY / random baseline + glossary
    2. Today's Picks   - the 3 best stocks to buy right now + a specific hold
                         timeline; refresh to live data; log + track forward paper
                         positions marked to market (fake cash, no real orders)
    3. Paper Trading   - run a fake-cash account over any window; equity vs SPY,
                         drawdown, current holdings, trade blotter, monthly returns
    4. Scanner         - today's ranked long candidates (latest_momentum.csv)
    5. Stock Detail    - per-ticker chart + its momentum rank
    6. Backtest        - equity vs SPY, drawdown, per-year, monthly returns, trades
    7. How it works    - the strategy, the failed ML history, honest caveats
    8. Settings        - read-only config view

Run with:
    streamlit run app/streamlit_app.py

Research / paper-trading only. NO real orders are ever placed. Not investment
advice. Headline returns carry survivorship bias (curated large-cap universe).
"""
from __future__ import annotations

import subprocess
import sys
from datetime import date
from pathlib import Path

# Make `src` importable when Streamlit launches this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from src.config import load_config, ensure_paths
from src.live_paper import add_picks, close_positions, load_ledger, mark_to_market
from src.paper import run_paper_session
from src.plotting import (
    candlestick_with_volume,
    drawdown_chart,
    equity_curve_chart,
    equity_vs_benchmark_chart,
    monthly_returns_heatmap,
)
from src.scanner import latest_per_ticker, scan_momentum


st.set_page_config(
    page_title="NebulaQuant Volatility AI",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)


# ----------------------------------------------------------------------------
# IO helpers
# ----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _read_csv(path_str: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path_str)


def _safe_read(path: Path) -> pd.DataFrame | None:
    try:
        if not path.exists():
            return None
        return _read_csv(str(path), path.stat().st_mtime)
    except Exception as e:  # noqa: BLE001
        st.warning(f"Failed to read {path.name}: {e}")
        return None


@st.cache_data(show_spinner="Running paper session (fake cash, replaying history)...")
def _run_paper_cached(
    dataset_path_str: str, dataset_mtime: float,
    months: int | None, capital: float, start: str | None, end: str | None,
) -> dict:
    """Cached wrapper so identical paper-session params return instantly."""
    cfg = load_config()
    df = pd.read_csv(dataset_path_str, parse_dates=["Date"])
    return run_paper_session(df, cfg, capital=capital, lookback_months=months, start=start, end=end)


def _processed_token(processed_dir: Path) -> float:
    """Max mtime across per-ticker feature files; used to bust the scan cache."""
    try:
        return max((p.stat().st_mtime for p in processed_dir.glob("*_features.csv")), default=0.0)
    except OSError:
        return 0.0


@st.cache_data(show_spinner="Ranking today's universe by momentum...")
def _live_scan_cached(
    processed_dir_str: str, token: float, top_pctile: float,
    max_vix_pctile: float, tickers: tuple,
) -> dict:
    """Rank the latest bar of each ticker with the deployed momentum scanner.

    Reuses ``latest_per_ticker`` + ``scan_momentum`` (the validated live
    counterpart to the backtest), so the picks page is the same logic, not a
    parallel one. ``token`` (file mtimes) invalidates the cache after a refresh.
    """
    snap = latest_per_ticker(Path(processed_dir_str), list(tickers))
    if snap.empty:
        return {"empty": True}
    scan = scan_momentum(snap, top_pctile=top_pctile, max_vix_pctile=max_vix_pctile)
    row0 = snap.iloc[0]
    regime = {
        "spy_above_sma200": int(row0.get("spy_above_sma200", 1) or 0),
        "vix_pctile": float(row0.get("vix_pctile_252", float("nan"))),
        "data_date": str(pd.to_datetime(snap["Date"]).max().date()),
    }
    return {"empty": False, "scan": scan, "regime": regime}


def _refresh_market_data() -> tuple[bool, str]:
    """Re-download prices and rebuild features via the venv python. (ok, log)."""
    root = Path(__file__).resolve().parent.parent
    logs = []
    for script in ("scripts/download_data.py", "scripts/build_dataset.py"):
        try:
            r = subprocess.run(
                [sys.executable, script], cwd=str(root),
                capture_output=True, text=True, timeout=1800,
            )
        except subprocess.TimeoutExpired:
            return False, "\n\n".join(logs + [f"$ {script}\n[timed out after 30 min]"])
        tail = (r.stdout or "")[-1200:] + (r.stderr or "")[-1800:]
        logs.append(f"$ {script}\n{tail}")
        if r.returncode != 0:
            return False, "\n\n".join(logs)
    return True, "\n\n".join(logs)


# ----------------------------------------------------------------------------
# Small formatting / analysis helpers
# ----------------------------------------------------------------------------

def _pct(x, digits=2) -> str:
    try:
        return f"{float(x) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "n/a"


def _money(x) -> str:
    try:
        return f"${float(x):,.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _spy_equity(spy_df: pd.DataFrame | None, equity_df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Rebase SPY buy-and-hold to the strategy's starting equity over its window."""
    if spy_df is None or spy_df.empty or equity_df is None or equity_df.empty:
        return None
    spy = spy_df.copy()
    spy["Date"] = pd.to_datetime(spy["Date"])
    eq_dates = pd.to_datetime(equity_df["date"])
    lo, hi = eq_dates.min(), eq_dates.max()
    spy = spy[(spy["Date"] >= lo) & (spy["Date"] <= hi)].sort_values("Date")
    if spy.empty:
        return None
    start_capital = float(equity_df["equity"].iloc[0])
    c0 = float(spy["Close"].iloc[0])
    if c0 <= 0:
        return None
    return pd.DataFrame({"date": spy["Date"].to_numpy(), "equity": start_capital * spy["Close"] / c0})


def _yearly_breakdown(equity_df: pd.DataFrame, spy_df: pd.DataFrame | None) -> pd.DataFrame:
    """Per-year strategy return vs SPY buy-and-hold over the same span."""
    eq = equity_df.copy()
    eq["date"] = pd.to_datetime(eq["date"])
    eq["year"] = eq["date"].dt.year
    rows = []
    spy = None
    if spy_df is not None and not spy_df.empty:
        spy = spy_df.copy()
        spy["Date"] = pd.to_datetime(spy["Date"])
        lo, hi = eq["date"].min(), eq["date"].max()
        spy = spy[(spy["Date"] >= lo) & (spy["Date"] <= hi)]
        spy["year"] = spy["Date"].dt.year
    for y, g in eq.groupby("year"):
        s0, s1 = g["equity"].iloc[0], g["equity"].iloc[-1]
        strat = s1 / s0 - 1 if s0 else float("nan")
        spy_ret = float("nan")
        if spy is not None:
            sg = spy[spy["year"] == y]["Close"]
            if len(sg) > 1:
                spy_ret = sg.iloc[-1] / sg.iloc[0] - 1
        dr = g["equity"].pct_change().fillna(0)
        invested = (dr.abs() > 1e-9).mean()
        rows.append({
            "year": int(y),
            "strategy": strat,
            "spy": spy_ret,
            "edge": (strat - spy_ret) if pd.notna(spy_ret) else float("nan"),
            "exposure": invested,
        })
    return pd.DataFrame(rows)


def _data_freshness(spy_df: pd.DataFrame | None) -> None:
    """Show how current the underlying data is, and warn if stale."""
    if spy_df is None or spy_df.empty or "Date" not in spy_df.columns:
        st.caption(":grey_question: Data status unknown - run `download_data` / `build_dataset`.")
        return
    last = pd.to_datetime(spy_df["Date"]).max().date()
    age = (date.today() - last).days
    if age <= 4:
        st.caption(f":green_circle: Data current through **{last}** ({age}d ago).")
    elif age <= 10:
        st.caption(f":yellow_circle: Data through **{last}** ({age}d ago) - consider re-running `download_data`.")
    else:
        st.caption(f":red_circle: Data is stale: **{last}** ({age}d ago). Re-run `download_data` then `build_dataset`.")


def _glossary() -> None:
    with st.expander(":book: How to read this dashboard (plain-English glossary)"):
        st.markdown(
            """
**The strategy in one sentence:** each day, buy the strongest-trending names
(top 10% by 3-month momentum) *only* when the market is healthy, hold ~20 trading
days, and sit in cash otherwise.

**Metrics, decoded:**
- **Total return** — how much the account grew over the whole period. *Inflated by
  survivorship bias here* (the universe is today's known winners), so trust it less
  than the risk-adjusted numbers below.
- **Sharpe** — return per unit of wobble. Higher = smoother ride for the same gain.
  Above ~1 is good; this is the *fairest* head-to-head vs SPY because it can't be
  inflated by a lucky universe.
- **Max drawdown** — the worst peak-to-trough drop. How much pain you'd have sat
  through. Smaller (closer to 0%) is better.
- **Win rate** — share of trades that made money. (A strategy can win <50% and still
  profit if winners are bigger than losers.)
- **Profit factor** — gross profit ÷ gross loss. Above 1.0 = profitable; 1.5+ is
  healthy.
- **Expectancy** — average $ made per trade.
- **vs SPY buy & hold** — the honesty bar. If the strategy doesn't beat simply
  holding SPY on **both** return and Sharpe, there's no demonstrated edge.
- **vs random entry** — sanity check that the *signal* matters, not just being in
  the market.

**Golden rule of this project:** beat SPY *and* random entry out-of-sample, or it
doesn't ship. Always weight the **Sharpe** and **same-universe** comparisons over the
raw return headline.
            """
        )


def _download_button(df: pd.DataFrame | None, label: str, filename: str) -> None:
    if df is None or df.empty:
        return
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=filename, mime="text/csv")


# Column formatting for the holdings / trades tables.
def _holdings_column_config():
    cc = st.column_config
    return {
        "ticker": cc.TextColumn("Ticker"),
        "entry_date": cc.DateColumn("Entry date"),
        "entry_price": cc.NumberColumn("Entry", format="$%.2f"),
        "shares": cc.NumberColumn("Shares"),
        "last_close": cc.NumberColumn("Last", format="$%.2f"),
        "cost_basis": cc.NumberColumn("Cost", format="$%.2f"),
        "market_value": cc.NumberColumn("Mkt value", format="$%.2f"),
        "unrealized_pnl": cc.NumberColumn("Unreal P&L", format="$%.2f"),
        "unrealized_pct_disp": cc.NumberColumn("Unreal %", format="%.2f%%"),
        "bars_held": cc.NumberColumn("Days held"),
        "score": cc.NumberColumn("Mom rank", format="%.3f"),
    }


# ----------------------------------------------------------------------------
# Tab renderers
# ----------------------------------------------------------------------------

def _render_overview(metrics_df, scan_df) -> None:
    st.subheader("Momentum strategy performance (full-span, out-of-sample)")
    _glossary()
    if metrics_df is not None and not metrics_df.empty:
        m = metrics_df.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Return", _pct(m.get("total_return", 0)),
                  help="Whole-period growth. Inflated by survivorship bias - trust Sharpe more.")
        c2.metric("Sharpe", f"{m.get('sharpe', 0):.2f}",
                  help="Return per unit of volatility. The fairest vs-SPY comparison.")
        c3.metric("Max Drawdown", _pct(m.get("max_drawdown", 0)),
                  help="Worst peak-to-trough drop. Smaller is better.")
        c4.metric("# Trades", int(m.get("n_trades", 0)),
                  help="Sample size. Hundreds = statistically meaningful.")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Win Rate", _pct(m.get("win_rate", 0)))
        c6.metric("Profit Factor", f"{m.get('profit_factor', 0):.2f}",
                  help="Gross profit / gross loss. >1 profitable, 1.5+ healthy.")
        c7.metric("Expectancy ($/trade)", f"{m.get('expectancy', 0):.2f}")
        c8.metric("Avg Hold (days)", f"{m.get('avg_hold_days', 0):.1f}")

        st.markdown("#### Edge vs benchmarks")
        b1, b2, b3 = st.columns(3)
        strat_ret = float(m.get("total_return", 0))
        spy_ret = float(m.get("spy_buy_hold_return", 0))
        strat_sh = float(m.get("sharpe", 0))
        spy_sh = float(m.get("spy_buy_hold_sharpe", 0))
        b1.metric("Strategy", _pct(strat_ret), help=f"Sharpe {strat_sh:.2f}")
        b2.metric("SPY Buy & Hold", _pct(spy_ret),
                  delta=_pct(strat_ret - spy_ret), help=f"Sharpe {spy_sh:.2f}")
        b3.metric("Random-entry avg/trade", _pct(m.get("random_entry_avg_return", 0)),
                  help=f"n={int(m.get('random_entry_n', 0))} simulated random long entries")
        if strat_ret > spy_ret and strat_sh > spy_sh:
            st.success(
                f"Strategy beats SPY on **return** ({_pct(strat_ret)} vs {_pct(spy_ret)}) "
                f"and **Sharpe** ({strat_sh:.2f} vs {spy_sh:.2f}) over the full span."
            )
        elif strat_ret > spy_ret or strat_sh > spy_sh:
            st.info("Strategy beats SPY on one of {return, Sharpe} but not both on this run.")
        else:
            st.warning("Strategy does not beat SPY here. Per the honesty rule, iterate before trusting it.")
        st.caption(
            ":warning: Absolute return is inflated by survivorship bias (curated large-cap "
            "universe). Weight the Sharpe and same-universe comparisons more."
        )
    else:
        st.info("No momentum backtest metrics yet. Run `python scripts/run_momentum_backtest.py`.")

    if scan_df is not None and not scan_df.empty:
        st.markdown("#### Today's top long candidates")
        longs = scan_df[scan_df.get("suggested_action") == "Long"]
        st.dataframe((longs if not longs.empty else scan_df).head(10), use_container_width=True, hide_index=True)


def _render_today_picks(cfg, paths) -> None:
    st.subheader(":dart: Today's Picks - the strategy's best buys right now")
    st.caption(
        "The strongest top-decile 60-day-momentum names that pass the regime/trend gate, "
        "with a specific hold timeline and suggested fake-cash sizing. Plan + track paper "
        "trades forward. **No real orders are placed.**"
    )

    processed = paths["data_processed"]
    mom = cfg.get("momentum", {})
    bt = cfg.get("backtest", {})
    paper_cfg = cfg.get("paper", {})
    hold_days = int(mom.get("hold_days", 20))
    top_pctile = float(mom.get("top_pctile", 0.90))
    max_vix = float(bt.get("max_vix_pctile", 0.85))
    per_pos = float(mom.get("position_size_pct", 0.12))
    tickers = tuple(cfg.get("data", {}).get("tickers", []))

    # --- Refresh-to-live-data control + freshness ---
    cbtn, cinfo = st.columns([1.1, 2.9])
    if cbtn.button("🔄 Refresh to latest prices",
                   help="Re-downloads prices and rebuilds features (~1-2 min). Then re-ranks."):
        with st.spinner("Downloading prices and rebuilding features (~1-2 min)..."):
            ok, log = _refresh_market_data()
        if ok:
            st.cache_data.clear()
            st.success("Data refreshed to the latest available close.")
        else:
            st.error("Refresh failed - showing the data already on disk.")
            with st.expander("Refresh log"):
                st.code(log)
    with cinfo:
        _data_freshness(_safe_read(processed / "SPY_features.csv"))

    res = _live_scan_cached(str(processed), _processed_token(processed),
                            top_pctile, max_vix, tickers)
    if res.get("empty"):
        st.info("No processed data yet. Click **Refresh** above, or run "
                "`python scripts/download_data.py` then `python scripts/build_dataset.py`.")
        _render_live_positions(paths)
        return

    scan = res["scan"]
    regime = res["regime"]
    data_date = pd.to_datetime(regime["data_date"])
    vix_ok = (pd.isna(regime["vix_pctile"])) or (regime["vix_pctile"] <= max_vix)
    regime_on = (regime["spy_above_sma200"] == 1) and vix_ok

    longs = scan[scan["suggested_action"] == "Long"].head(3) if not scan.empty else scan.iloc[0:0]

    capital = st.number_input(
        "Paper cash to deploy ($)", min_value=500,
        value=int(paper_cfg.get("starting_capital", 10000)), step=500,
        help=f"Each pick is sized at {per_pos:.0%} of this (the deployed position size).",
    )

    # --- Regime OFF or nothing qualifies: be honest, hold cash ---
    if not regime_on or longs.empty:
        why = []
        if regime["spy_above_sma200"] != 1:
            why.append("SPY is **below** its 200-day average")
        if not vix_ok:
            why.append(f"VIX is elevated (pctile {regime['vix_pctile']:.2f} > {max_vix:.2f})")
        if regime_on and longs.empty:
            why.append("no name currently clears the top-decile + trend filter")
        st.warning(
            ":lock: **No new buys today - the strategy is in cash.** "
            + ("Reason: " + "; ".join(why) + "." if why else "")
            + " By design it sits out weak/unconfirmed markets rather than forcing trades."
        )
        if not scan.empty:
            with st.expander("If the gate were open, these would lead (context only)"):
                lead = scan.sort_values("momentum_rank", ascending=False).head(5)
                st.dataframe(lead[["ticker", "close", "momentum_rank", "ret_60d",
                                   "rs_vs_spy", "suggested_action"]],
                             use_container_width=True, hide_index=True)
        _render_live_positions(paths)
        return

    # --- Regime ON: show the timeline + picks ---
    target_exit = (data_date + pd.offsets.BDay(hold_days)).normalize()
    cal_days = (target_exit - data_date).days
    st.success(
        f":white_check_mark: **Regime ON.** Buy the {len(longs)} name(s) below near the next "
        f"session, hold **~{hold_days} trading days** (~{cal_days} calendar days), and exit "
        f"on/around **{target_exit:%a %b %d, %Y}**."
    )
    st.caption(f"Signal date (last close): **{data_date:%b %d, %Y}**. The strategy uses a "
               f"time-based exit - it sells on the timeline regardless of price.")

    picks_records = []
    cols = st.columns(len(longs))
    for i, (_, r) in enumerate(longs.iterrows()):
        price = float(r["close"])
        alloc = capital * per_pos
        shares = int(alloc // price) if price > 0 else 0
        spend = shares * price
        with cols[i]:
            st.markdown(f"#### {i + 1}. {r['ticker']}")
            st.metric("Buy ~price", _money(price), help="Latest close (paper entry).")
            st.metric("Momentum rank", f"{r['momentum_rank']:.3f}",
                      help="Percentile of 60-day return across the universe (1.0 = strongest).")
            st.write(f"**60d return:** {_pct(r.get('ret_60d'))}")
            st.write(f"**RS vs SPY (20d):** {_pct(r.get('rs_vs_spy'))}")
            st.write(f"**Buy:** {shares} sh ≈ {_money(spend)}")
            st.write(f"**Sell ~{target_exit:%b %d}**")
        picks_records.append({
            "ticker": str(r["ticker"]),
            "entry_date": str(data_date.date()),
            "entry_price": price,
            "shares": shares,
            "alloc": alloc,
            "hold_days": hold_days,
            "target_exit_date": str(target_exit.date()),
        })

    plan = pd.DataFrame([{
        "Buy": p["ticker"], "Shares": p["shares"], "Entry ~$": round(p["entry_price"], 2),
        "Cost ~$": round(p["shares"] * p["entry_price"], 2),
        "Hold (trading days)": p["hold_days"],
        "Sell on/around": p["target_exit_date"],
    } for p in picks_records])
    st.markdown("##### The plan")
    st.dataframe(plan, use_container_width=True, hide_index=True)
    total_cost = float((plan["Cost ~$"]).sum())
    st.caption(f"Total deployed ≈ {_money(total_cost)} of {_money(capital)} "
               f"({total_cost / capital:.0%}); the rest stays in cash.")

    if st.button("➕ Add these picks to my paper portfolio", type="primary"):
        added, skipped = add_picks(paths["data_paper"], picks_records)
        msg = f"Logged {added} pick(s)."
        if skipped:
            msg += f" Skipped {skipped} already-open ticker(s)."
        st.success(msg)
        st.rerun()

    st.caption(
        ":warning: Survivorship caveat: this universe is today's known large-caps, so the live "
        "edge is likely smaller than the headline backtest. Sizing/timeline are suggestions, not "
        "advice. Research / paper only - **no real orders are placed**."
    )

    _render_live_positions(paths)


def _render_live_positions(paths) -> None:
    st.markdown("---")
    st.markdown("### :ledger: My live paper positions (forward test)")
    ledger = load_ledger(paths["data_paper"])
    if ledger is None or ledger.empty:
        st.caption("No paper positions yet. Use **Add these picks** above to start a forward test - "
                   "they'll be marked to market here on your next visit.")
        return

    mtm = mark_to_market(ledger, paths["data_processed"])
    if not mtm.empty:
        total_cost = float((mtm["entry_price"] * mtm["shares"]).sum())
        total_val = float(mtm["market_value"].sum())
        total_pnl = float(mtm["unrealized_pnl"].sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Open positions", int(len(mtm)))
        c2.metric("Market value", _money(total_val), delta=_money(total_pnl))
        c3.metric("Unrealized", _pct(total_pnl / total_cost if total_cost else 0.0))

        disp = mtm.copy()
        disp["unrealized_pct_disp"] = disp["unrealized_pct"] * 100
        disp["action"] = disp["due"].map(lambda d: "SELL (past target)" if d else "hold")
        order = ["ticker", "entry_date", "entry_price", "shares", "last_close",
                 "market_value", "unrealized_pnl", "unrealized_pct_disp",
                 "target_exit_date", "days_to_target", "action"]
        cc = st.column_config
        st.dataframe(
            disp[order], use_container_width=True, hide_index=True,
            column_config={
                "ticker": cc.TextColumn("Ticker"),
                "entry_date": cc.TextColumn("Entry"),
                "entry_price": cc.NumberColumn("Entry $", format="$%.2f"),
                "shares": cc.NumberColumn("Shares"),
                "last_close": cc.NumberColumn("Now $", format="$%.2f"),
                "market_value": cc.NumberColumn("Mkt value", format="$%.2f"),
                "unrealized_pnl": cc.NumberColumn("Unreal P&L", format="$%.2f"),
                "unrealized_pct_disp": cc.NumberColumn("Unreal %", format="%.2f%%"),
                "target_exit_date": cc.TextColumn("Target exit"),
                "days_to_target": cc.NumberColumn("Days left"),
                "action": cc.TextColumn("Action"),
            },
        )
        due = disp[disp["due"]]
        if not due.empty:
            st.warning(f":alarm_clock: {len(due)} position(s) reached the ~hold-window target exit: "
                       f"**{', '.join(due['ticker'].astype(str))}**. The strategy would sell these now.")

        to_close = st.multiselect("Close position(s) at last close", list(mtm["ticker"].astype(str)),
                                  default=list(due["ticker"].astype(str)))
        if st.button("Record exit for selected") and to_close:
            n = close_positions(paths["data_paper"], to_close, paths["data_processed"])
            st.success(f"Closed {n} position(s) at last close.")
            st.rerun()
    else:
        st.caption("No open positions right now.")

    closed = ledger[ledger["status"] == "closed"]
    if not closed.empty:
        realized = pd.to_numeric(closed["realized_pnl"], errors="coerce").fillna(0.0).sum()
        with st.expander(f"Closed paper trades ({len(closed)}) - realized P&L {_money(realized)}"):
            st.dataframe(
                closed[["ticker", "entry_date", "entry_price", "exit_date",
                        "exit_price", "shares", "realized_pnl"]],
                use_container_width=True, hide_index=True,
            )


def _render_paper(paths, spy_df) -> None:
    st.subheader(":money_with_wings: Paper Trading - fake cash, real prices, no real orders")
    st.caption(
        "Replays the deployed momentum strategy over the window you choose, in the **same "
        "engine** the backtest uses (identical fills, costs, regime gate, sizing). Nothing is "
        "ordered anywhere - this is a simulated account."
    )
    dataset_path = paths["data_processed"] / "model_dataset.csv"
    if not dataset_path.exists():
        st.info("No dataset found. Run `python scripts/build_dataset.py` first.")
        return

    with st.form("paper_form"):
        c1, c2, c3 = st.columns([1.2, 1, 1])
        mode = c1.radio("Window", ["Recent months", "Custom date range"], horizontal=True)
        capital = c2.number_input("Starting fake cash ($)", min_value=1000, value=10000, step=1000)
        months = start = end = None
        if mode == "Recent months":
            months = int(c3.slider("Months to replay", 1, 36, 12))
        else:
            d1, d2 = st.columns(2)
            start = str(d1.date_input("Start", value=date(2024, 1, 1)))
            end = str(d2.date_input("End", value=date.today()))
        submitted = st.form_submit_button("Run paper session", type="primary")

    # Compute on submit, or on first visit with defaults.
    if not submitted and "paper_done" not in st.session_state:
        months, start, end, capital = 12, None, None, 10000.0
    st.session_state["paper_done"] = True

    try:
        session = _run_paper_cached(
            str(dataset_path), dataset_path.stat().st_mtime,
            months, float(capital), start, end,
        )
    except Exception as e:  # noqa: BLE001
        st.error(f"Paper session failed: {e}")
        return

    s = session["summary"]
    equity_df = session["equity_curve"]
    open_df = session["open_positions"]
    trade_log = session["trade_log"]

    st.markdown(f"**Account** · {s['window_start']} → {s['window_end']} · {s['trading_days']} trading days")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Current equity", _money(s["final_equity"]),
              delta=_pct(s["total_return"]), help="Mark-to-market account value.")
    a2.metric("Cash", _money(s["cash_estimate"]))
    a3.metric("In positions", _money(s["open_market_value"]), help=f"{s['n_open_positions']} open")
    a4.metric("Started with", _money(s["starting_capital"]))

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Sharpe", f"{s['sharpe']:.2f}", help=f"SPY: {s['spy_sharpe']:.2f}")
    b2.metric("Max Drawdown", _pct(s["max_drawdown"]))
    b3.metric("Win Rate", _pct(s["win_rate"]), help=f"{s['n_closed_trades']} closed trades")
    b4.metric("Profit Factor", f"{s['profit_factor']:.2f}")

    if s["beats_spy_return"] and s["beats_spy_sharpe"]:
        st.success(f"Ahead of SPY on return ({_pct(s['total_return'])} vs {_pct(s['spy_return'])}) "
                   f"and Sharpe ({s['sharpe']:.2f} vs {s['spy_sharpe']:.2f}) this window.")
    elif s["beats_spy_return"] or s["beats_spy_sharpe"]:
        st.info(f"Mixed vs SPY: return {_pct(s['total_return'])} vs {_pct(s['spy_return'])}, "
                f"Sharpe {s['sharpe']:.2f} vs {s['spy_sharpe']:.2f}.")
    else:
        st.warning(f"Behind SPY this window: return {_pct(s['total_return'])} vs {_pct(s['spy_return'])}, "
                   f"Sharpe {s['sharpe']:.2f} vs {s['spy_sharpe']:.2f}.")

    spy_eq = _spy_equity(spy_df, equity_df)
    st.plotly_chart(equity_vs_benchmark_chart(equity_df, spy_eq), use_container_width=True)
    st.plotly_chart(drawdown_chart(equity_df), use_container_width=True)

    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("##### Current holdings (marked to market)")
        if open_df is not None and not open_df.empty:
            disp = open_df.copy()
            disp["unrealized_pct_disp"] = disp["unrealized_pct"] * 100
            order = ["ticker", "entry_date", "entry_price", "shares", "last_close",
                     "cost_basis", "market_value", "unrealized_pnl", "unrealized_pct_disp",
                     "bars_held", "score"]
            st.dataframe(disp[order], use_container_width=True, hide_index=True,
                         column_config=_holdings_column_config())
            _download_button(open_df, "Download holdings CSV", "paper_holdings.csv")
        else:
            st.info("Flat - no open positions at window end (strategy is in cash).")
    with right:
        st.markdown("##### Closed-trade stats")
        if trade_log is not None and not trade_log.empty:
            wins = trade_log[trade_log["pnl"] > 0]
            st.write({
                "closed trades": int(len(trade_log)),
                "winners": int(len(wins)),
                "avg win ($)": round(float(wins["pnl"].mean()) if len(wins) else 0.0, 2),
                "avg loss ($)": round(float(trade_log[trade_log["pnl"] <= 0]["pnl"].mean())
                                      if (trade_log["pnl"] <= 0).any() else 0.0, 2),
                "best ($)": round(float(trade_log["pnl"].max()), 2),
                "worst ($)": round(float(trade_log["pnl"].min()), 2),
            })
            if "exit_reason" in trade_log.columns:
                st.caption("Exit reasons:")
                st.dataframe(trade_log["exit_reason"].value_counts().rename("count"),
                             use_container_width=True)
        else:
            st.info("No closed trades in this window.")

    with st.expander("Monthly returns heatmap"):
        st.plotly_chart(monthly_returns_heatmap(equity_df), use_container_width=True)

    with st.expander("Full closed-trade blotter"):
        if trade_log is not None and not trade_log.empty:
            st.dataframe(trade_log, use_container_width=True, hide_index=True)
            _download_button(trade_log, "Download trades CSV", "paper_trades.csv")
        else:
            st.write("No trades.")

    st.caption(
        ":warning: One window can be lucky or unlucky - a single hot year (e.g. +98% over the "
        "last 12 months) is **not** the long-run expectation (full-period Sharpe ≈ 1.17). Compare "
        "several windows, and remember the survivorship caveat. Research/paper only - no real orders."
    )


def _render_scanner(scan_df) -> None:
    st.subheader("Scanner - today's ranked momentum candidates")
    if scan_df is None or scan_df.empty:
        st.info("No scan found. Run `python scripts/run_momentum_scanner.py`.")
        return
    ticker_q = st.text_input("Filter by ticker (substring)", "")
    min_rank = st.slider("Min momentum rank", 0.0, 1.0, 0.0, 0.05)
    actions = sorted(scan_df["suggested_action"].dropna().unique())
    action_sel = st.multiselect("Suggested action", actions, default=actions)

    view = scan_df.copy()
    if ticker_q:
        view = view[view["ticker"].str.contains(ticker_q, case=False, na=False)]
    view = view[view["momentum_rank"] >= min_rank]
    view = view[view["suggested_action"].isin(action_sel)]
    st.dataframe(view, use_container_width=True, hide_index=True)
    _download_button(view, "Download scan CSV", "scan_filtered.csv")
    st.caption(
        "`Long` = top-decile 60-day momentum **and** passes the regime/trend gate "
        "(SPY > 200d SMA, benign VIX, name > 20d & 200d SMA). `Watch` = near the cutoff. "
        "`No Trade` = momentum high but trend/regime gate fails."
    )


def _render_stock_detail(cfg, paths, scan_df) -> None:
    st.subheader("Stock Detail")
    tickers = cfg.get("data", {}).get("tickers", [])
    if not tickers:
        st.info("No tickers configured.")
        return
    ticker = st.selectbox("Select ticker", tickers)
    feat_path = paths["data_processed"] / f"{ticker}_features.csv"
    if not feat_path.exists():
        st.warning(f"No processed data found for {ticker}. Run build_dataset first.")
        return
    ticker_df = pd.read_csv(feat_path, parse_dates=["Date"]).dropna()
    tail = ticker_df.tail(250)
    st.plotly_chart(candlestick_with_volume(tail, title=f"{ticker} Price"), use_container_width=True)

    latest = ticker_df.tail(1).iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Close", f"{latest['Close']:.2f}")
    col2.metric("60d Return", _pct(latest.get("ret_60d", float("nan"))))
    col3.metric("ATR %", _pct(latest.get("atr_pct", float("nan"))))
    col4.metric("> 200d SMA", "yes" if latest.get("price_above_sma200", 0) == 1 else "no")

    if scan_df is not None and not scan_df.empty:
        row = scan_df[scan_df["ticker"] == ticker]
        if not row.empty:
            r = row.iloc[0]
            st.markdown("#### Latest momentum read")
            cc = st.columns(4)
            cc[0].metric("Momentum rank", f"{r.get('momentum_rank', 0):.3f}")
            cc[1].metric("RS vs SPY (20d)", _pct(r.get("rs_vs_spy", 0)))
            cc[2].metric("Regime OK", "yes" if r.get("market_regime_ok", 0) == 1 else "no")
            cc[3].metric("Suggested Action", str(r.get("suggested_action", "n/a")))


def _render_backtest(metrics_df, equity_df, trade_log_df, spy_df) -> None:
    st.subheader("Momentum backtest (full span)")
    if metrics_df is None or metrics_df.empty:
        st.info("No backtest metrics found. Run `python scripts/run_momentum_backtest.py`.")
    else:
        m = metrics_df.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Return", _pct(m.get("total_return", 0)))
        c2.metric("Sharpe", f"{m.get('sharpe', 0):.2f}", help=f"SPY: {float(m.get('spy_buy_hold_sharpe', 0)):.2f}")
        c3.metric("Max Drawdown", _pct(m.get("max_drawdown", 0)))
        c4.metric("# Trades", int(m.get("n_trades", 0)))

    if equity_df is not None and not equity_df.empty:
        st.markdown("##### Account value vs SPY (same starting cash)")
        spy_eq = _spy_equity(spy_df, equity_df)
        st.plotly_chart(equity_vs_benchmark_chart(equity_df, spy_eq), use_container_width=True)

        st.markdown("##### Drawdown")
        st.plotly_chart(drawdown_chart(equity_df), use_container_width=True)

        st.markdown("##### Per-year: strategy vs SPY")
        yb = _yearly_breakdown(equity_df, spy_df)
        if not yb.empty:
            show = yb.copy()
            for col in ("strategy", "spy", "edge", "exposure"):
                show[col] = show[col].map(lambda v: _pct(v) if pd.notna(v) else "n/a")
            st.dataframe(show, use_container_width=True, hide_index=True)
            wins = int((yb["edge"] > 0).sum())
            tot = int(yb["edge"].notna().sum())
            st.caption(f"Strategy beat SPY in **{wins} of {tot}** years on this run. "
                       "Lower exposure in down years reflects the regime gate sitting in cash.")

        with st.expander("Monthly returns heatmap"):
            st.plotly_chart(monthly_returns_heatmap(equity_df), use_container_width=True)

    if trade_log_df is not None and not trade_log_df.empty:
        st.markdown("##### Trade log")
        st.dataframe(trade_log_df, use_container_width=True, hide_index=True)
        _download_button(trade_log_df, "Download trade log CSV", "momentum_trade_log.csv")
        if "exit_reason" in trade_log_df.columns:
            st.markdown("###### Exit-reason breakdown")
            st.dataframe(trade_log_df["exit_reason"].value_counts().rename("count"),
                         use_container_width=True)
    else:
        st.info("No trade log found.")


def _render_how_it_works() -> None:
    st.subheader("How it works (and what the data rejected)")
    st.markdown(
        """
**Deployed strategy — regime-gated cross-sectional momentum.** Each day, among
names in a confirmed uptrend, hold the **top-decile 60-day momentum** names for
~20 trading days; sit in cash when the broad market is unhealthy.

- **Entry:** `SPY > 200d SMA` and VIX percentile ≤ 0.85 (regime ON); name
  `> 20d SMA` and `> 200d SMA`; name in the **top decile** of 60-day return ranked
  across the universe that day.
- **Exit:** time-based (~20 days). No ATR profit-target/stop — momentum pays out as
  multi-week *drift*, and tight barriers truncated it.
- **Rules-based, not ML.** Every input uses only trailing data, so the whole-period
  backtest is the honest out-of-sample estimate.

**Short-horizon variants were tested and rejected.** Shortening the hold to 3–10 days
(1/2/3-month momentum) and a short-term mean-reversion signal (buy oversold-in-uptrend)
both **lost to SPY net of costs** — momentum is a multi-week factor, and this growth
universe trends rather than mean-reverts. See `scripts/sweep_short_swing.py` /
`scripts/sweep_reversal.py`.

**How we got here.** The original squeeze→expansion ML thesis was held to one rule:
beat SPY *and* random entry out-of-sample, or don't ship.

- **Two-stage model** (`score = expansion_prob × long_prob`): an edge diagnostic
  showed the top score decile was *anti-predictive* (Spearman ≈ −0.02 to −0.07).
- **Single-stage long meta-label**: same — flat win rate, near-zero return spread.
- **Direct factor study** revealed why: in this universe/era **60-day momentum is
  positively predictive**, while **low-vol and near-52w-high tilts were inverted**.

**Honest caveats.**

- *Survivorship / selection bias*: the 75-name universe is today's known
  large/mid-caps, so absolute return is inflated. The robust claims are the
  risk-adjusted Sharpe edge and the same-universe outperformance vs equal-weight.
- *Factor chosen after looking at the data*, but 60-day momentum is a
  decades-documented anomaly that generalized across every year here.

The legacy ML pipeline is retained in the repo for transparency; it did not produce a
deployable edge.
        """
    )


def main() -> None:
    cfg = load_config()
    paths = ensure_paths()
    bt_dir = paths["reports_backtests"]

    scan_df = _safe_read(paths["data_predictions"] / "latest_momentum.csv")
    metrics_df = _safe_read(bt_dir / "momentum_backtest_metrics.csv")
    trade_log_df = _safe_read(bt_dir / "momentum_trade_log.csv")
    equity_df = _safe_read(bt_dir / "momentum_equity_curve.csv")
    spy_df = _safe_read(paths["data_processed"] / "SPY_features.csv")

    st.title(":sparkles: NebulaQuant Volatility AI")
    st.caption(
        "Long-only **regime-gated cross-sectional momentum**. Beats SPY on return and "
        "Sharpe out-of-sample. Research / paper-trading only - **no real orders**. Headline "
        "returns carry survivorship bias (see *How it works*)."
    )
    _data_freshness(spy_df)

    tabs = st.tabs([
        "Overview", "Today's Picks", "Paper Trading", "Scanner", "Stock Detail",
        "Backtest", "How it works", "Settings",
    ])

    with tabs[0]:
        _render_overview(metrics_df, scan_df)
    with tabs[1]:
        _render_today_picks(cfg, paths)
    with tabs[2]:
        _render_paper(paths, spy_df)
    with tabs[3]:
        _render_scanner(scan_df)
    with tabs[4]:
        _render_stock_detail(cfg, paths, scan_df)
    with tabs[5]:
        _render_backtest(metrics_df, equity_df, trade_log_df, spy_df)
    with tabs[6]:
        _render_how_it_works()
    with tabs[7]:
        st.subheader("Settings (read-only view of config.yaml)")
        st.json(cfg)
        st.caption(
            "To change settings, edit `config.yaml` at the project root and re-run the "
            "relevant scripts. The `momentum:` block drives the deployed strategy; `paper:` "
            "sets the paper-trading defaults."
        )


if __name__ == "__main__":
    main()
