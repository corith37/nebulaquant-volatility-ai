"""Tests for the forward live-paper ledger (src/live_paper.py)."""
from __future__ import annotations

import pandas as pd

from src.live_paper import (
    add_picks,
    close_positions,
    load_ledger,
    mark_to_market,
)


def _write_feature(processed_dir, ticker, last_close):
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    closes = [10.0, 11.0, 12.0, 13.0, last_close]
    pd.DataFrame({"Date": dates, "Close": closes}).to_csv(
        processed_dir / f"{ticker}_features.csv", index=False
    )


def _pick(ticker, price, shares=10):
    return {
        "ticker": ticker, "entry_date": "2024-01-05", "entry_price": price,
        "shares": shares, "alloc": price * shares, "hold_days": 20,
        "target_exit_date": "2024-02-02",
    }


def test_add_picks_and_no_double_buy(tmp_path):
    paper = tmp_path / "paper"
    added, skipped = add_picks(paper, [_pick("AAA", 100.0), _pick("BBB", 50.0)])
    assert (added, skipped) == (2, 0)

    # Re-adding an already-open ticker is skipped; a new one is added.
    added, skipped = add_picks(paper, [_pick("AAA", 100.0), _pick("CCC", 25.0)])
    assert (added, skipped) == (1, 1)

    led = load_ledger(paper)
    assert sorted(led["ticker"]) == ["AAA", "BBB", "CCC"]
    assert (led["status"] == "open").all()


def test_mark_to_market_pnl_and_due_flag(tmp_path):
    paper = tmp_path / "paper"
    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    _write_feature(processed, "AAA", last_close=110.0)  # +10 / share

    add_picks(paper, [_pick("AAA", 100.0, shares=10)])
    led = load_ledger(paper)

    # As of a date before the target exit -> not due, P&L = (110-100)*10 = 100.
    mtm = mark_to_market(led, processed, today="2024-01-15")
    row = mtm.iloc[0]
    assert row["last_close"] == 110.0
    assert round(row["unrealized_pnl"], 2) == 100.0
    assert round(row["unrealized_pct"], 4) == 0.10
    assert not bool(row["due"])
    assert row["days_to_target"] > 0

    # After the target exit date -> flagged due.
    mtm_late = mark_to_market(led, processed, today="2024-03-01")
    assert bool(mtm_late.iloc[0]["due"])
    assert mtm_late.iloc[0]["days_to_target"] < 0


def test_close_positions_records_realized_pnl(tmp_path):
    paper = tmp_path / "paper"
    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    _write_feature(processed, "AAA", last_close=120.0)

    add_picks(paper, [_pick("AAA", 100.0, shares=10)])
    n = close_positions(paper, ["AAA"], processed, today="2024-02-05")
    assert n == 1

    led = load_ledger(paper)
    closed = led[led["status"] == "closed"]
    assert len(closed) == 1
    assert float(closed.iloc[0]["exit_price"]) == 120.0
    assert float(closed.iloc[0]["realized_pnl"]) == 200.0  # (120-100)*10

    # Now flat -> nothing open to mark.
    assert mark_to_market(led, processed).empty
