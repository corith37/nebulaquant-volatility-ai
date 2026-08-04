"""Guards against the two ways this pipeline silently serves stale signals:

1. An unsettled intraday bar leaking into the daily dataset.
2. The scanner skipping the newest bars because forward-looking *label* columns
   are NaN there by construction.
"""
import pandas as pd

from src.data_loader import drop_partial_bar
from src.scanner import LABEL_COLS, latest_per_ticker


def _bars(dates):
    return pd.DataFrame({
        "Date": pd.to_datetime(dates),
        "Open": 1.0, "High": 2.0, "Low": 0.5, "Close": 1.5, "Volume": 100,
    })


def test_drops_todays_bar_while_market_open(monkeypatch):
    import src.data_loader as dl

    now = pd.Timestamp("2026-08-04 11:30", tz=dl._EASTERN)
    monkeypatch.setattr(dl, "datetime", type("D", (), {"now": staticmethod(lambda tz: now)}))

    df = _bars(["2026-08-03", "2026-08-04"])
    out = drop_partial_bar(df)
    assert len(out) == 1
    assert out["Date"].iloc[-1] == pd.Timestamp("2026-08-03")


def test_keeps_todays_bar_after_the_close(monkeypatch):
    import src.data_loader as dl

    now = pd.Timestamp("2026-08-04 16:45", tz=dl._EASTERN)
    monkeypatch.setattr(dl, "datetime", type("D", (), {"now": staticmethod(lambda tz: now)}))

    df = _bars(["2026-08-03", "2026-08-04"])
    assert len(drop_partial_bar(df)) == 2


def test_stale_prior_session_bar_is_never_dropped(monkeypatch):
    import src.data_loader as dl

    now = pd.Timestamp("2026-08-04 11:30", tz=dl._EASTERN)
    monkeypatch.setattr(dl, "datetime", type("D", (), {"now": staticmethod(lambda tz: now)}))

    df = _bars(["2026-07-31", "2026-08-03"])
    assert len(drop_partial_bar(df)) == 2


def test_scanner_uses_newest_bar_despite_unfilled_labels(tmp_path):
    """Labels are NaN for the most recent horizon; the scanner must still see today."""
    df = _bars(pd.bdate_range("2026-07-01", periods=20))
    df["Ticker"] = "TEST"
    df["atr_pct"] = 0.05
    for col in LABEL_COLS:
        df[col] = 1.0
    df.loc[df.index[-10:], list(LABEL_COLS)] = float("nan")
    df.to_csv(tmp_path / "TEST_features.csv", index=False)

    snap = latest_per_ticker(tmp_path, ["TEST"])
    assert len(snap) == 1
    assert pd.to_datetime(snap["Date"].iloc[0]) == df["Date"].iloc[-1]


def test_scanner_still_skips_bars_with_missing_features(tmp_path):
    """A genuinely incomplete feature row must not be served as today's signal."""
    df = _bars(pd.bdate_range("2026-07-01", periods=20))
    df["Ticker"] = "TEST"
    df["atr_pct"] = 0.05
    df.loc[df.index[-1], "atr_pct"] = float("nan")
    df.to_csv(tmp_path / "TEST_features.csv", index=False)

    snap = latest_per_ticker(tmp_path, ["TEST"])
    assert len(snap) == 1
    assert pd.to_datetime(snap["Date"].iloc[0]) == df["Date"].iloc[-2]
