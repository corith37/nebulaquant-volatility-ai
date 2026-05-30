"""Tests for the pure execution-planning logic (src/execution.py).

The Alpaca network wrapper is not exercised here (no keys / no network); only the
deterministic decision logic and key loading are tested.
"""
from __future__ import annotations

from src.execution import load_alpaca_keys, plan_orders


def _picks(*tickers):
    return [{"ticker": t, "price": 100.0} for t in tickers]


def test_buys_fill_free_slots_when_regime_on():
    plan = plan_orders(
        _picks("AAA", "BBB", "CCC", "DDD"),
        holdings=[], open_ledger={}, today="2024-06-01",
        target_positions=3, regime_on=True,
    )
    assert plan["sells"] == []
    assert plan["buys"] == ["AAA", "BBB", "CCC"]  # top 3, capped by slots


def test_regime_off_blocks_new_buys_but_holds_run():
    # Held name not yet at its exit -> no sell; regime off -> no buys.
    plan = plan_orders(
        _picks("AAA", "BBB"),
        holdings=["ZZZ"],
        open_ledger={"ZZZ": "2024-12-31"},
        today="2024-06-01",
        target_positions=3, regime_on=False,
    )
    assert plan["sells"] == []
    assert plan["buys"] == []


def test_time_exit_sells_and_frees_a_slot():
    # ZZZ is past its target exit -> sell it; that frees a slot so we can add AAA.
    plan = plan_orders(
        _picks("AAA", "BBB"),
        holdings=["ZZZ", "YYY"],
        open_ledger={"ZZZ": "2024-05-01", "YYY": "2024-12-31"},
        today="2024-06-01",
        target_positions=2, regime_on=True,
    )
    assert plan["sells"] == ["ZZZ"]
    # held_after = {YYY}; 2 slots - 1 held = 1 free -> buy top new pick.
    assert plan["buys"] == ["AAA"]


def test_already_held_pick_is_not_rebought():
    plan = plan_orders(
        _picks("AAA", "BBB", "CCC"),
        holdings=["AAA"],
        open_ledger={"AAA": "2024-12-31"},
        today="2024-06-01",
        target_positions=3, regime_on=True,
    )
    assert plan["sells"] == []
    assert "AAA" not in plan["buys"]
    assert plan["buys"] == ["BBB", "CCC"]  # 2 free slots


def test_load_keys_from_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    (tmp_path / ".env").write_text(
        'APCA_API_KEY_ID=PKTEST123\nAPCA_API_SECRET_KEY="shh-secret"\n# comment\n',
        encoding="utf-8",
    )
    key, secret = load_alpaca_keys(root=tmp_path)
    assert key == "PKTEST123"
    assert secret == "shh-secret"


def test_env_vars_take_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "ENVKEY")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "ENVSECRET")
    (tmp_path / ".env").write_text("APCA_API_KEY_ID=FILEKEY\n", encoding="utf-8")
    key, secret = load_alpaca_keys(root=tmp_path)
    assert key == "ENVKEY" and secret == "ENVSECRET"
