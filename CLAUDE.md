# NebulaQuant — project context

Orientation for anyone (human or AI) picking this repo up. Read this first.

## What this repo is

Two things live here:

1. **`src/`, `scripts/`, `config.yaml`, `app/`** — the original NebulaQuant
   volatility/swing model. A two-stage XGBoost triple-barrier system that was
   later pivoted to a rules-based, regime-gated 60-day momentum screen, with
   Alpaca paper execution. Still runnable; unchanged by the research below.

2. **`research/`** — a 2026 rebuild of the modelling approach as a
   cross-sectional, factor-neutral pipeline. Self-contained, reads the same
   `data/raw/*.csv`. Nothing in it has traded real money.

**Start with `research/HANDOFF.md`.** It is the complete written record:
findings, numbers, method, limits, and ranked next steps.
`research/DECISIONS.md` records what was tried and rejected — read it before
proposing anything, because several obvious ideas have already been tested and
failed for recorded reasons.

## The single most important finding

The model's predictions were never the bottleneck; the portfolio was.

The target is defined as returns orthogonal to a 5-dimensional factor space
(market beta + 4 principal components). A book that is only *dollar*-neutral
still carries PC2–PC4 exposure whose variance dwarfs the alpha. Same
predictions, same costs:

| Neutralisation | Net Sharpe |
|---|---:|
| Dollar-neutral only | **−0.16** |
| + market beta | +0.46 |
| + 2 PCs | +0.67 |
| + 4 PCs | **+0.74** |

Headline: out-of-sample IC **+0.0240 (t = 3.86)**, net Sharpe **0.738** after
2.5 bp, 3.09%/yr on 4.18% vol, 39.8× turnover. Replicates on an independent
482-name universe at IC **+0.0255 (t = 4.11)**.

## House rules for this codebase

These are not style preferences — each one is a bug that already happened or a
result that already failed. See `research/DECISIONS.md` for the evidence.

1. **Never feed raw price or volume levels to a model.** `sma_200`, `atr_14`,
   `macd`, `dollar_volume` in dollars are name-and-era fingerprints a tree will
   memorise; the rules expire silently as prices drift. Every stock feature must
   be a within-date cross-sectional rank.
2. **The target's factor space and the portfolio's neutrality constraints must
   match.** If you change `n_pc` in `research/src/factors.py`, change
   `FACTOR_COLS` in `research/src/portfolio.py` to match. This is the finding
   above; breaking it silently destroys the strategy.
3. **Do not tune hyperparameters on the walk-forward folds.** That is what
   compromised the original model's reported numbers. `research/src/model.py`
   fixes them a priori and PBO (0.690 across a config spread of 0.518–0.779)
   confirms the surface is flat anyway. Do not add an Optuna loop.
4. **Do not optimise holding period or weight cap.** Same reason. 5 days, 8%.
5. **Report the pre-registered number, not the best one found.** The ablation
   suggests a lean feature set scores 0.823 vs 0.739 — that is a post-hoc
   selection and must be pre-registered and re-tested before adoption.
6. **Rescale OHL onto the dividend-adjusted basis** when loading yfinance-style
   files. Skipping it injects a fake −0.5% overnight gap on every ex-dividend
   date. Handled in `research/src/panel.py`.
7. **Newey-West every t-stat.** Overlapping 5-day labels make daily IC series
   strongly autocorrelated; a naive t-stat overstates significance by ~√h.
   `research/src/stats.py` has the machinery.
8. **Track IC, not P&L, when validating.** On a 0.74 Sharpe strategy, 60 days of
   P&L is noise. Cross-sectional IC converges far faster.

## Layout

```
config.yaml, src/, scripts/, app/     original model (untouched)
data/raw/*.csv                        77 tickers OHLCV 2015→2026, + ^VIX
research/
  HANDOFF.md                          ← full findings, read first
  DECISIONS.md                        ← what was tried and rejected
  README_RESEARCH.md                  short orientation
  src/panel.py                        load, dividend-rescale, QA
  src/factors.py                      rolling point-in-time market + PCA model
  src/features.py                     42 ranked + 8 regime features, targets
  src/model.py                        purged/embargoed walk-forward
  src/portfolio.py                    factor-neutral projection + tranches
  src/backtest.py                     costed simulation
  src/stats.py                        Newey-West, deflated Sharpe, PBO/CSCV
  build_dataset.py → ablation.py      6 runnable scripts, in order
  report/the-neutralization-gap.html  standalone research note (open locally)
  results/                            every number as produced, + raw logs
```

## Running the research pipeline

```bash
pip install pandas numpy scipy scikit-learn lightgbm pyarrow
cd research
python3 build_dataset.py   # ~1 min      python3 robustness.py  # ~12 min
python3 train_wf.py        # ~4 min      python3 run_val.py     # ~5 min
python3 run_backtest2.py   # ~3 min      python3 ablation.py    # ~25 min
```

Requires pandas 3.0-compatible code: `groupby.apply(include_groups=True)` is
removed, and `DataFrame.stack()` needs `future_stack=True`.

## Current priorities

1. **Widen the universe to 400–500 names.** Highest-value change available and
   the evidence is in hand: same pipeline on 482 names gave less than half the
   volatility for the same IC, Sharpe 0.74 → 1.22. IR ≈ IC × √breadth is the one
   relationship here with theory behind it.
2. **Buy point-in-time data with delisted names** (Sharadar/Norgate, ~$50–100/mo).
   The only fix for the one bias no modelling choice can correct.
3. **Pre-register and test the lean feature set** on the wider universe.
4. **Paper-trade a quarter, tracking IC.**

## What this is not

A 0.74 Sharpe market-neutral strategy earning 3.1%/yr unlevered is real but
modest. It needs ~3× leverage to matter, carries 57 positions with a short
locate on each, and its statistical case is suggestive rather than settled
(deflated Sharpe 0.918, below the 0.95 bar; neither subperiod individually
significant). Both universes are survivorship-biased. Nothing here is investment
advice, and every figure is a backtest.
