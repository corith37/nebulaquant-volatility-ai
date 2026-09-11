# Factor-neutral swing research pipeline

A rebuild of the NebulaQuant modelling stack around one finding: **the portfolio
must be neutral to the same factor space the prediction is defined in.** With
identical signals and costs, a dollar-neutral book scores −0.16 net Sharpe and a
fully factor-neutral book scores +0.74.

Result summary (76 US large caps, out-of-sample 2020-01 → 2026-08, 1,666 days):

| Metric | Value |
|---|---|
| OOS information coefficient | +0.0240 (Newey-West t = 3.86) |
| Net Sharpe @ 2.5 bp one-way | 0.738 |
| Annual return / vol | +3.09% / 4.18% |
| Max drawdown | −5.1% |
| Annual turnover | 39.8× |
| Break-even cost | ~10 bp one-way |

Replicated on an independent 482-name universe over 2016–2018: IC +0.0255
(t = 4.11), net Sharpe 1.22 @ 2.5 bp.

## Why this differs from the original pipeline

| Original | Here | Reason |
|---|---|---|
| Raw price levels as features (`sma_200`, `atr_14`, `macd`, `dollar_volume`) | Within-date cross-sectional ranks | Raw levels are name-and-era fingerprints a tree memorises; the rules expire silently as prices drift |
| Absolute per-stock triple-barrier label | Rank of forward 5d **residual** return | Raw-return targets are dominated by beta (mean pairwise corr 0.267 → −0.011 after residualising) |
| Long-only, 3 concurrent positions | Factor-neutral long/short, ~57 positions | 3 positions ≈ 13 independent bets/year — too few to measure skill |
| Hyperparameters tuned on the walk-forward folds | Fixed a priori, never tuned | Tuning on the reported folds makes them in-sample |
| Full rebalance | 5 overlapping daily tranches | Cuts turnover ~5× at identical signal |

## Modules

    src/panel.py      raw OHLCV → total-return basis + data-integrity QA
    src/factors.py    rolling point-in-time market + PCA risk model
    src/features.py   42 ranked stock features, 8 raw regime features, targets
    src/model.py      purged + embargoed expanding walk-forward
    src/portfolio.py  factor-neutral weights + vectorised overlapping tranches
    src/stats.py      Newey-West t-stats, deflated Sharpe, PBO/CSCV

## Running it

    pip install pandas numpy scipy scikit-learn lightgbm pyarrow

    python3 build_dataset.py    # panel → residuals → features   (~1 min)
    python3 train_wf.py         # walk-forward, reports OOS IC   (~4 min)
    python3 run_backtest2.py    # costed portfolio + baselines   (~3 min)
    python3 robustness.py       # nulls, deflated Sharpe, PBO    (~12 min)
    python3 run_val.py          # independent 482-name check     (~5 min)
    python3 ablation.py         # leave-one-block-out            (~25 min)

`build_dataset.py` reads `data/raw/*.csv` in the existing project layout.

## The two files worth reading first

`src/portfolio.py` holds the orthogonal projection that makes the strategy work:

    w  ∝  (I − B (B'B)⁻¹ B') s

`src/stats.py` holds the tests that try to prove it doesn't.

## Known limits

- Both universes are survivorship-biased (present-day membership). The
  market-neutral construction mutes this but does not remove it.
- Deflated Sharpe 0.918 — below the conventional 0.95 bar.
- PBO 0.690, but across a config Sharpe spread of only 0.518–0.779. Read it as
  "do not optimise hold period or weight cap", not "the edge is spurious".
- Unlevered return is 3.1%/yr. Meaningful returns need ~3× gross leverage.
- No earnings-date, index-event or corporate-action model.

Full write-up, including the literature the design is grounded in, is in the
research note published alongside this code.
