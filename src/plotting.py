"""Plotly chart helpers for the Streamlit dashboard."""
from __future__ import annotations

import calendar

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def candlestick_with_volume(df: pd.DataFrame, title: str = "") -> go.Figure:
    """Render candlestick + volume in a stacked plot."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, row_heights=[0.7, 0.3],
        subplot_titles=("Price", "Volume"),
    )
    fig.add_trace(
        go.Candlestick(
            x=df["Date"], open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name="OHLC",
        ),
        row=1, col=1,
    )
    if "sma_20" in df:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["sma_20"], name="SMA20", line=dict(width=1)), row=1, col=1)
    if "sma_50" in df:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["sma_50"], name="SMA50", line=dict(width=1)), row=1, col=1)
    fig.add_trace(
        go.Bar(x=df["Date"], y=df["Volume"], name="Volume", marker=dict(opacity=0.6)),
        row=2, col=1,
    )
    fig.update_layout(
        title=title, xaxis_rangeslider_visible=False,
        template="plotly_dark", height=600,
    )
    return fig


def equity_curve_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        return fig
    fig.add_trace(go.Scatter(x=df["date"], y=df["equity"], mode="lines", name="Equity"))
    fig.update_layout(
        title="Equity Curve", template="plotly_dark",
        xaxis_title="Date", yaxis_title="Account Equity ($)", height=400,
    )
    return fig


def equity_vs_benchmark_chart(
    strat_df: pd.DataFrame,
    bench_df: pd.DataFrame | None = None,
    title: str = "Account value vs SPY (same starting cash)",
    strat_name: str = "Strategy",
    bench_name: str = "SPY buy & hold",
) -> go.Figure:
    """Overlay the strategy equity curve and a rebased benchmark equity curve.

    Both frames use columns ``date`` and ``equity``; the benchmark should already
    be rebased to the same starting capital. This is the single most important
    "is it actually beating buy-and-hold?" view.
    """
    fig = go.Figure()
    if strat_df is None or strat_df.empty:
        return fig
    fig.add_trace(go.Scatter(
        x=strat_df["date"], y=strat_df["equity"], mode="lines",
        name=strat_name, line=dict(width=2.5, color="#2ecc71"),
    ))
    if bench_df is not None and not bench_df.empty:
        fig.add_trace(go.Scatter(
            x=bench_df["date"], y=bench_df["equity"], mode="lines",
            name=bench_name, line=dict(width=2, color="#888", dash="dot"),
        ))
    fig.update_layout(
        title=title, template="plotly_dark", height=420,
        xaxis_title="Date", yaxis_title="Account value ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )
    return fig


def drawdown_chart(equity_df: pd.DataFrame, title: str = "Drawdown (peak-to-trough)") -> go.Figure:
    """Underwater plot: % below the running peak. Quick read of pain/risk."""
    fig = go.Figure()
    if equity_df is None or equity_df.empty:
        return fig
    eq = equity_df["equity"].to_numpy(dtype=float)
    run_max = np.maximum.accumulate(eq)
    dd = np.divide(eq - run_max, run_max, out=np.zeros_like(eq), where=run_max > 0)
    fig.add_trace(go.Scatter(
        x=equity_df["date"], y=dd, mode="lines", fill="tozeroy",
        name="Drawdown", line=dict(color="#e74c3c"),
    ))
    fig.update_layout(
        title=title, template="plotly_dark", height=280,
        xaxis_title="Date", yaxis_title="Drawdown",
        yaxis=dict(tickformat=".0%"), showlegend=False,
    )
    return fig


def monthly_returns_heatmap(equity_df: pd.DataFrame, title: str = "Monthly returns (%)") -> go.Figure:
    """Year x month grid of equity returns, green/red centred at 0."""
    fig = go.Figure()
    if equity_df is None or equity_df.empty:
        return fig
    s = equity_df.copy()
    s["date"] = pd.to_datetime(s["date"])
    s["ym"] = s["date"].dt.to_period("M")
    monthly = s.groupby("ym")["equity"].last()
    rets = monthly.pct_change().dropna()
    if rets.empty:
        return fig
    grid = pd.DataFrame({
        "year": rets.index.year, "month": rets.index.month, "ret": rets.to_numpy(),
    })
    pivot = grid.pivot(index="year", columns="month", values="ret").reindex(columns=range(1, 13))
    z = pivot.to_numpy(dtype=float) * 100
    text = np.where(np.isnan(z), "", np.round(z, 1).astype(str))
    fig.add_trace(go.Heatmap(
        z=z, x=[calendar.month_abbr[m] for m in range(1, 13)],
        y=[str(y) for y in pivot.index],
        colorscale="RdYlGn", zmid=0, text=text, texttemplate="%{text}",
        hovertemplate="%{y} %{x}: %{z:.1f}%<extra></extra>", colorbar=dict(title="%"),
    ))
    fig.update_layout(
        title=title, template="plotly_dark", height=120 + 34 * max(1, len(pivot.index)),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def confusion_matrix_heatmap(cm: pd.DataFrame) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=cm.values, x=list(cm.columns), y=list(cm.index),
        colorscale="Blues", showscale=True,
    ))
    fig.update_layout(title="Confusion Matrix", template="plotly_dark", height=400)
    return fig


def calibration_curve_chart(df: pd.DataFrame) -> go.Figure:
    """Reliability diagram: mean predicted prob vs observed frequency per bin."""
    fig = go.Figure()
    if df.empty or "mean_pred" not in df or "observed_freq" not in df:
        return fig
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="grey"), name="Perfect calibration",
    ))
    fig.add_trace(go.Scatter(
        x=df["mean_pred"], y=df["observed_freq"], mode="lines+markers",
        name="Model", marker=dict(size=8),
    ))
    fig.update_layout(
        title="Calibration Curve (Stage B)", template="plotly_dark",
        xaxis_title="Mean predicted probability", yaxis_title="Observed hit rate",
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]), height=420,
    )
    return fig


def feature_importance_bar(df: pd.DataFrame, top_n: int = 25) -> go.Figure:
    df = df.head(top_n)
    fig = go.Figure(go.Bar(
        x=df["importance"], y=df["feature"], orientation="h",
    ))
    fig.update_layout(
        title=f"Top {top_n} Features", template="plotly_dark",
        yaxis=dict(autorange="reversed"), height=600,
    )
    return fig
