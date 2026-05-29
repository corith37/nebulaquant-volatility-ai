# NebulaQuant Volatility AI

A personal, long-only quant research tool. It started as a hunt for
**volatility-compression setups that precede a directional swing** ("squeeze →
expansion"), but the research was held to one rule: **prove an out-of-sample edge
versus SPY buy-and-hold and a random-entry baseline, or don't ship it.** The
squeeze→expansion ML thesis did not clear that bar. Following the data instead of
the hypothesis led to a simpler signal that *does*: **regime-gated cross-sectional
momentum.**

> **Research / paper-trading tool. It does not place real orders.** Backtest
> performance does not guarantee future returns, and the headline numbers below
> carry real survivorship bias (see caveats). Do not risk capital until you have
> independently validated and paper-traded any signal.

---

## What actually works: regime-gated cross-sectional momentum

Each trading day, among names that are in a confirmed uptrend, hold the
**top-decile 60-day momentum** names with a fixed ~20-day horizon; sit in cash
when the broad market is not healthy.

**Entry (all must hold):**
- Broad-market regime ON: `SPY > 200d SMA` **and** VIX percentile ≤ 0.85.
- Name trending: `Close > 20d SMA` **and** `Close > 200d SMA`.
- Name in the **top decile** of 60-day return *ranked across the universe that day*.

**Exit:** time-based, ~20 trading days (no ATR profit-target/stop truncation —
that was what strangled the edge; momentum pays out as multi-week *drift*).

**Sizing:** equal-ish per name, capped per-position and by total exposure.

This is **rules-based, not ML.** Because every input uses only trailing data and
there is no model to fit, the whole-period backtest *is* the honest out-of-sample
estimate.

### Results (2015–2026, production engine: costs, slippage, intrabar fills)

| Metric | Momentum strategy | SPY buy & hold |
|---|---:|---:|
| Total return | **+1006%** | +264% |
| **Sharpe** | **1.18** | 0.79 |
| Win rate | 57% | — |
| Profit factor | 1.55 | — |
| Expectancy | +$88 / trade (988 trades) | — |
| Max drawdown | −29% | — |

- **Beats SPY in 9 of 11 full years.** The two misses were *underperformance in up
  years* (2021, 2023), not losses.
- **Defends in bear markets:** 2022 was −18.7% vs SPY −19.9%, with the regime gate
  cutting exposure to ~36% (mostly cash) that year.
- The edge is **not** a 2020–21 tech-mania artifact — it underperformed in 2021.

### Honest caveats (read these)

- **Survivorship / selection bias.** The 75-name universe is *today's* known
  large/mid-caps (NVDA, TSLA, PLTR, …). A momentum strategy on a curated winners'
  list inflates absolute returns. The **robust** claims are the risk-adjusted
  Sharpe edge (1.18 vs 0.79) and the *same-universe* outperformance vs equal-weight
  (top-decile momentum beat an equal-weight basket of the same gated names by
  ~+2%/rebalance) — both far less bias-sensitive than the +1006% headline.
- **Factor chosen after looking at the data**, but 60-day momentum is a
  decades-documented anomaly that generalized across every year here — confirmation,
  not data-mining.

---

## How we got here (what the data rejected)

The repo still contains the full two-stage ML pipeline. It is retained for
transparency; it did **not** produce a deployable edge. The findings that drove
the pivot:

- **Stage A (expansion detector) + Stage B (long meta-label), `score = P_A·P_B`.**
  An edge diagnostic bucketed the calibrated score into deciles vs realized
  triple-barrier outcomes: the top decile was *anti-predictive* (Spearman ≈ −0.02
  to −0.07). The model did not rank winners.
- **Single-stage long meta-label** (`score = long_prob`): same verdict — flat win
  rate across deciles, near-zero return spread.
- **Direct factor study** (no ML, no triple-barrier) revealed why: in this
  universe/era, **60-day momentum is positively predictive**, while **low-vol and
  near-52w-high tilts were *inverted*** (calm/defensive names lagged the growth
  leaders). So the relative-strength half of the original thesis was kept and the
  low-volatility half was dropped. The ATR triple-barrier label was also actively
  harmful — it truncated the multi-week drift the momentum edge depends on.

---

## What the pipeline does

1. Downloads daily OHLCV for ~75 liquid names + SPY/QQQ/^VIX context (`yfinance`).
2. Builds features: volatility-compression (Bollinger-width percentile, TTM squeeze,
   ATR ratio/percentile, NR7/NR4, consolidation, range compression, vol-of-vol,
   volume dry-up, Donchian), momentum/trend, SPY/QQQ/VIX regime context, and
   **cross-sectional per-date ranks** (`xs_ret20/60_rank`, `xs_lowvol_rank`,
   `xs_dist52w_rank`, `xs_rs_vs_spy`/`xs_rs_rank`) — the inputs to the momentum
   strategy.
3. **Momentum backtest** (`run_momentum_backtest.py`): the deployable strategy,
   benchmarked vs SPY buy-and-hold and random entry, over the full span.
4. **Momentum scanner** (`run_momentum_scanner.py`): today's ranked long candidates.
5. *(Legacy ML, retained)* triple-barrier/expansion labels, purged + embargoed
   walk-forward validation, two-stage calibrated model, walk-forward OOS backtest,
   and a risk-adjusted hyperparameter search.

---

## Folder structure

```text
nebulaquant-volatility-ai/
├── config.yaml          # universe, features, momentum, walk-forward, backtest, optimization
├── requirements.txt
├── data/
│   ├── raw/             # downloaded OHLCV per ticker (^VIX saved as _VIX.csv)
│   ├── processed/       # per-ticker features + model_dataset.csv
│   └── predictions/     # latest_momentum.csv (deployed) + latest_predictions.csv (legacy ML)
├── models/saved/        # two_stage_model.joblib, feature_columns.json (legacy ML)
├── reports/
│   ├── metrics/         # legacy ML: model_metrics, calibration_curve, best_params.json
│   ├── backtests/       # momentum_*.csv (deployed) + trade_log/equity_curve/oos (legacy ML)
│   └── charts/
├── src/                 # features, labels, validation, train_model, backtest, risk, optimize, scanner
├── app/streamlit_app.py
├── scripts/             # CLI entrypoints
└── tests/               # pytest suite (incl. purged-splitter leakage test)
```

---

## Setup

```bash
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# macOS/Linux:         source .venv/bin/activate
pip install -r requirements.txt   # optuna is optional; random-search fallback otherwise
```

---

## Workflow

**Deployed (momentum) path:**

```bash
python scripts/download_data.py          # OHLCV + SPY/QQQ/^VIX -> data/raw/
python scripts/build_dataset.py          # features (incl. cross-sectional ranks) -> data/processed/
python scripts/run_momentum_backtest.py  # strategy vs SPY + random -> reports/backtests/momentum_*
python scripts/run_momentum_scanner.py   # today's ranked longs -> data/predictions/latest_momentum.csv
streamlit run app/streamlit_app.py       # dashboard
```

**Legacy ML path (retained for reference; did not beat SPY):**

```bash
python scripts/optimize.py     # risk-adjusted search -> reports/metrics/best_params.json
python scripts/train.py        # two-stage calibrated model + holdout eval
python scripts/run_backtest.py # walk-forward OOS backtest + benchmarks
python scripts/run_scanner.py  # legacy ML scanner -> data/predictions/latest_predictions.csv
```

---

## Configuration (`config.yaml`)

- `data.tickers` / `market_ticker` / `qqq_ticker` / `vix_ticker` / `context_tickers`.
- **`momentum.*`** — the deployed strategy: `score_col` (`xs_ret60_rank`),
  `top_pctile` (0.90 → top decile), `hold_days` (~20), `tp/sl_atr_mult` (set high
  to disable barriers → pure time exit), `vol_target`, position-size caps,
  `entry_min_rel_volume`.
- `backtest.*` — capital, costs, `regime_filter`, `max_vix_pctile` (the regime gate
  shared by both paths).
- *(legacy ML)* `triple_barrier.*`, `model.*` (`two_stage` is now `false`),
  `walkforward.*`, `optimization.*`.

---

## Reading the results honestly

The project's rule: **if a strategy does not beat SPY buy-and-hold AND random entry
out-of-sample, there is no demonstrated edge — iterate before trusting it.** The
momentum strategy clears both, on a statistically meaningful 988 trades. When you
read the dashboard, keep the survivorship caveat in mind and weight the
**risk-adjusted (Sharpe) and same-universe** comparisons over the raw return.

---

## Tests

```bash
pytest -q
```

Covers compression-feature sanity, triple-barrier / expansion / uniqueness label
correctness, the **purged-splitter leakage guarantee**, vol-targeted sizing math,
and backtest metrics.

---

## Risk disclaimer

For research, education, and paper trading only. Backtests overfit, markets change,
slippage is real, fills are not guaranteed, and the universe here carries
survivorship bias. Always paper-trade before risking capital. The authors accept no
liability for any loss arising from use of this code.

---

## Roadmap / next iterations

- **Address survivorship bias**: re-test on a point-in-time / delisted-inclusive
  universe (or index-constituent history) to see how much absolute return survives.
- Sub-period and parameter-sensitivity sweeps (lookback, hold, breadth) to confirm
  robustness rather than a lucky parameter pocket.
- Sector caps / correlation-aware sizing so the top decile isn't all one theme.
- Volatility-scaled position sizing and an explicit cash-yield assumption when flat.
- Alpaca **paper** trading with daily-loss and max-position controls.
```
