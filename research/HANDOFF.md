# NebulaQuant research handoff

**Date:** 31 August 2026
**Scope:** audit of the existing model, and a replacement cross-sectional pipeline
**Status:** research complete, not deployed. Nothing here has traded real money.

This document is the complete written record. The HTML note in `report/` is the
same material with charts; this file is the version you can grep, diff and paste
into a ticket.

---

## 1. The one thing to take away

The model's predictions were never the bottleneck. **The portfolio was.**

The model predicts returns that are orthogonal to a 5-dimensional factor space
(market beta + 4 principal components). A book that is only *dollar*-neutral
still carries loadings on PC2–PC4, and the variance of those loadings is an
order of magnitude larger than the alpha. The forecast is real but invisible in
the P&L.

Identical predictions, identical 2.5 bp costs, identical 5-day overlapping
tranches — only the neutralisation changes:

| Neutralisation | Net Sharpe | Annual | NW t |
|---|---:|---:|---:|
| Dollar-neutral only | **−0.156** | −0.93% | −0.34 |
| + market beta | +0.460 | +2.25% | 1.37 |
| + 2 principal components | +0.674 | +2.93% | 1.98 |
| + 4 principal components | **+0.738** | +3.09% | 2.18 |

That is 0.89 Sharpe points from construction alone — more than any feature,
model choice, or hyperparameter examined in this study.

**The rule this implies:** whatever space you define the target in, you must
neutralise the book to that same space. `src/portfolio.py` does this with an
orthogonal projection:

```
w  ∝  (I − B (B'B)⁻¹ B') s
```

where `s` is the centred signal and `B` the point-in-time factor loadings
(intercept + `beta_mkt` + `load_pc1..4`). The intercept column is what enforces
dollar neutrality; the rest enforce factor neutrality.

---

## 2. Headline results

Primary universe: 76 US large caps, 177,709 stock-days, features from 2017-01.
Out-of-sample window 2020-01 → 2026-08 (1,666 trading days, 27 walk-forward
blocks, 124,264 scored stock-days).

| Metric | Value |
|---|---:|
| OOS information coefficient (5d residual) | **+0.0240** |
| IC Newey-West t-stat (lag 20) | **+3.86** |
| IC information ratio | +0.167 |
| Daily IC hit rate | 57.1% |
| Gross Sharpe (0 bp) | 0.984 |
| **Net Sharpe (2.5 bp one-way)** | **0.738** |
| Annual return / volatility | +3.09% / 4.18% |
| Max drawdown | −5.1% |
| Sortino / Calmar | 1.155 / 0.605 |
| Portfolio NW t-stat | +2.18 |
| Annual turnover (two-way) | 39.8× |
| Positions above 0.5% weight | 57 |
| Mean gross exposure | 0.89 |
| Largest single position | 5.9% |
| Break-even cost | ~10 bp one-way |

Costs charged: 2.5 bp per side on turnover, plus 50 bp/yr borrow on short
notional. Execution assumes a full day of implementation lag — signal at the
close of day *t*, fill at the close of *t+1*, first return earned *t+2*.

### Year by year (net)

| Year | Net return | Sharpe | OOS IC |
|---|---:|---:|---:|
| 2020 | +6.59% | 1.27 | 0.0437 |
| 2021 | −0.08% | −0.02 | 0.0101 |
| 2022 | +4.04% | 1.03 | 0.0100 |
| 2023 | +3.26% | 0.98 | 0.0227 |
| 2024 | +0.91% | 0.22 | 0.0237 |
| 2025 | +8.05% | 2.00 | 0.0410 |
| 2026 (partial, 160d) | −2.93% | −0.57 | 0.0128 |

### Single-signal baselines, same construction and costs

| Signal | Gross Sharpe | Net Sharpe | Annual turnover |
|---|---:|---:|---:|
| **ML model** | **1.040** | **0.738** | 39.8× |
| Residual momentum 12-1 | 0.717 | 0.616 | 11.2× |
| Reversal 5d | 0.401 | 0.031 | 67.4× |
| Vol-scaled reversal 5d | 0.415 | 0.024 | 67.9× |
| Reversal 1d | 0.480 | −0.169 | 67.0× |
| Residual momentum 3m | 0.112 | −0.002 | 18.9× |
| Overnight−intraday spread | 0.170 | −0.062 | 32.1× |
| Overnight gap (short high-gap) | −0.079 | −0.425 | 59.5× |

Reversal has genuine *gross* alpha and loses all of it to 67× turnover. That is
the Novy-Marx & Velikov result (+0.37%/mo gross → −1.28%/mo net) reproduced in
miniature. Momentum survives because it barely trades.

---

## 3. Independent replication

The whole pipeline was rebuilt from scratch on **482 S&P 500 names, 2015–2018**
(6.3× the breadth, a period that barely overlaps the primary OOS window).
Nothing was re-tuned.

| Measure | Primary (76 names) | Validation (482 names) |
|---|---:|---:|
| OOS period | 2020-01 → 2026-08 | 2016-02 → 2018-01 |
| OOS stock-days | 124,264 | 219,223 |
| **OOS IC** | **+0.0240** | **+0.0255** |
| **IC NW t-stat** | **+3.86** | **+4.11** |
| IC information ratio | +0.167 | +0.309 |
| Daily IC hit rate | 57.1% | 62.3% |
| Net Sharpe @ 2.5 bp | +0.738 | +1.224 |
| Annual volatility | 4.18% | 1.90% |

Two independent universes agreeing on IC to within 0.0015 is considerably harder
to produce by chance than either result alone. This is the strongest evidence in
the study.

The validation universe also **recovers the reversal effect** that mega caps
could not resolve — 1-day reversal reaches t = 2.61 and 5-day t = 2.25 there,
against 1.6–1.9 in the 76-name set. Reversal is a small- and mid-cap phenomenon
that thins out in mega caps, exactly as the literature says.

Caveats on the validation file: no dividend adjustment (its overnight component
is contaminated on ex-div dates) and S&P 500 membership as of 2018. It is a real
test of *breadth and period*, not of survivorship.

---

## 4. Robustness — the tests designed to kill it

**Null distributions (40 draws each).** The permutation null shuffles the
model's own predictions across tickers within each date, preserving every
distributional property and destroying only the alignment with returns.

| Null | Mean | SD | Best draw | Model | p |
|---|---:|---:|---:|---:|---:|
| Permutation, 2.5 bp | −0.450 | 0.349 | +0.226 | +0.738 | 0.0000 |
| i.i.d. random, 2.5 bp | −1.101 | 0.357 | −0.451 | +0.738 | 0.0000 |
| i.i.d. random, 0 bp | −0.045 | 0.347 | +0.735 | +0.984 (0 bp) | 0.0000 |

Note the third row honestly: the best *cost-free* random draw (+0.735) lands
within a rounding error of the model's *costed* result (+0.738). The
like-for-like comparison is the model's own zero-cost Sharpe of 0.984, which
still beats all 40 draws. A lucky random book at zero cost is not a low bar.

**Deflated Sharpe ratio: 0.918.** Charging for 33 configurations examined
(15 hold × weight-cap combinations, 8 candidate signals, 6 cost levels,
4 neutralisation variants). Suggestive, but short of the conventional 0.95 bar.
Reported as-is rather than by quietly lowering the trial count.

**Probability of backtest overfitting: 0.690** (CSCV, 15 configs, 10 blocks).
Above 0.5 nominally means the selection procedure is worse than random. The
honest reading is narrower: **all 15 configurations score between +0.518 and
+0.779.** When every variant performs about equally well, which one ranks first
in any given sample is close to arbitrary and PBO climbs by construction. This
says *do not optimise the holding period or the weight cap* — not *the strategy
is spurious*. A spurious strategy would show a wide spread with negatives.

**Subperiod stability.** First half (833d): Sharpe 0.934, +3.93%, t = 1.80.
Second half (833d): Sharpe 0.540, +2.25%, t = 1.24. Both positive, neither
individually significant. The weaker second half is consistent with documented
post-publication decay of liquidity-provision premia — or with ordinary noise.
Six years cannot separate the two.

**Parameter sensitivity.** Flat, which is the good kind of boring:

| Hold (days) | 2 | 3 | 5 | 10 | 21 |
|---|---:|---:|---:|---:|---:|
| Net Sharpe | 0.520 | 0.630 | 0.738 | 0.779 | 0.735 |
| Annual turnover | 76× | 57× | 40× | 23× | 13× |

| Cost (bp one-way) | 0 | 1 | 2.5 | 5 | 7.5 | 10 | 15 | 20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Net Sharpe | 0.984 | 0.885 | 0.738 | 0.494 | 0.252 | 0.013 | −0.459 | −0.921 |

---

## 5. Feature-block ablation

Each block removed in turn, full walk-forward re-run from scratch (~2.5 min each).

| Variant | OOS IC | IC t | Net Sharpe | Δ vs full |
|---|---:|---:|---:|---:|
| **FULL model** | +0.0230 | 3.70 | **0.739** | — |
| drop momentum | +0.0151 | 2.46 | 0.341 | **−0.398** |
| no regime context | +0.0230 | 3.68 | 0.650 | −0.089 |
| drop reversal | +0.0240 | 3.75 | 0.695 | −0.043 |
| drop overnight/intraday | +0.0254 | 3.89 | 0.698 | −0.041 |
| drop liquidity | +0.0235 | 3.57 | 0.711 | −0.027 |
| drop price shape | +0.0244 | 3.83 | 0.768 | +0.029 |
| drop risk/vol | +0.0259 | 4.12 | **0.823** | **+0.085** |
| regime context only | — | — | — | degenerate¹ |

¹ Regime features are constant within a date, so they carry no cross-sectional
information and cannot rank stocks. This is a sanity check that passed — the
variant is correctly undefined rather than accidentally predictive.

**Reading:** momentum is load-bearing and everything else is a rounding error
next to it. Regime context is worth +0.089, which confirms the model is
exploiting the conditional structure Nagel documents. The risk/volatility block
is **net harmful** — its ten features (beta, idio vol, skew, kurtosis, semidev,
Parkinson vol, vol-of-vol …) largely duplicate exposures the portfolio
construction already neutralises, so they contribute estimation noise without
information.

> **Do not simply adopt the 0.823.** Acting on that ablation is a selection
> decision made after seeing out-of-sample results. The pre-registered full model
> scores 0.739 and that is the number to plan against. The lean variant is a
> hypothesis for the next independent test.

---

## 6. What the raw signals say

Univariate daily cross-sectional Spearman IC against forward 5-day *residual*
return, Newey-West at lag 10. Full table in `results/ic_full.csv` (42 signals).

| Signal | IC | NW t | Reading |
|---|---:|---:|---|
| mom_252_21 (12-1 residual momentum) | +0.0244 | +3.53 | strongest positive |
| dist_52w_high | +0.0126 | +1.83 | |
| rev_1 | +0.0065 | +1.89 | reversal, weak |
| rev_3 | +0.0081 | +1.66 | |
| rev_5 | +0.0092 | +1.61 | |
| rev_5_sc (vol-scaled) | +0.0088 | +1.57 | |
| beta_mkt | −0.0234 | −3.55 | strongest negative |
| gap_abs_5 | −0.0179 | −3.20 | |
| clv_1 (close location in range) | −0.0088 | −3.11 | |
| vov_63 | −0.0134 | −2.79 | |
| ln_dollar_vol | −0.0138 | −2.64 | size tilt |
| idio_vol_63 | −0.0128 | −2.46 | low-vol effect |

Two important qualifications, both established by explicit tests:

**Reversal carries residual market exposure.** Splitting the daily IC series on
the direction of the forward market return, `rev_5` gives **+0.026 when the
market rises and −0.019 when it falls** (corr of daily IC with forward market
return = +0.18). Momentum does not flip (+0.025 / +0.023). So the reversal
signal in this universe is partly an unhedged beta bet, not clean selection.

**Almost nothing predicts raw returns.** Measured against forward *raw* (not
residual) 5-day returns, every reversal variant falls below t = 1.0. Only
`mom_252_21` (t = 2.51) and `beta_mkt` (t = 2.00, which is just beta earning the
equity risk premium) survive. This is the observation that reframed the whole
project into section 1.

**Conditional structure.** 5-day reversal IC rises from **+0.0047** in the
lowest VIX tercile to **+0.0183** in the highest — roughly Nagel's result, in the
predicted direction, but not enough to carry a strategy on its own.

---

## 7. Method, precisely

| Component | Choice |
|---|---|
| Universe | 76 US large caps, 68–76 names per date |
| Sample | features 2017-01-03 → 2026-08-21, 177,709 stock-days |
| Risk model | equal-weight market factor + top-4 PCs of market-residual covariance |
| Beta estimation | trailing 252 days, refit every 21 days, held fixed and applied forward |
| Features | 42 stock-level (all within-date percentile ranks) + 8 raw date-level regime |
| Target | within-date rank of forward 5-day residual return |
| Model | LightGBM regression, 15 leaves, depth 4, lr 0.03, 300 trees, min_child_samples 200, feature_fraction 0.6, bagging 0.7, L2 = 10 |
| Seeds | 3, averaged (variance reduction, not an accuracy trick) |
| Hyperparameters | **fixed a priori, never tuned** |
| Validation | expanding walk-forward, 63-day OOS blocks, 3-year min train |
| Purge | 5-day label horizon + 5-day embargo = 10 days |
| Portfolio | orthogonal projection, gross 1.0, position cap 8% |
| Tranches | 5 overlapping (1/5 of the book opened daily, held 5 days) |
| Execution | signal close *t*, fill close *t+1*, returns *t+2* … *t+6* |

### Point-in-time discipline

- Betas and PC loadings for window `[t0, t0+21)` are estimated only on returns
  strictly before `t0`. No observation is residualised using its own future.
- Every stock feature is a within-date rank, so the model sees relative position
  and cannot memorise a name-and-era price level.
- The purge removes the label's own 5-day look-ahead plus a 5-day embargo at the
  left edge of each test block. Because the window is expanding, training data
  always precedes the block, so a one-sided purge is sufficient.

### Data quality — checked before any model was fit

- All 77 raw files: no NaN closes, no duplicate dates, no OHLC violations, no
  unadjusted splits. The two flagged single-day moves above 35% (ORCL, SNOW) are
  real earnings reactions, verified against the intraday range.
- **Open/high/low are rescaled onto the dividend-adjusted basis.** yfinance
  files carry `Close` (split-adjusted) and `Adj Close` (split + dividend).
  Skipping the rescale injects a spurious −0.5% overnight gap on every
  ex-dividend date, which would have poisoned the overnight/intraday split. See
  `src/panel.py`.
- Factor model works as claimed: mean pairwise return correlation falls from
  **0.267** (raw) to **−0.011** (residual).

---

## 8. Reproducing it

```bash
pip install pandas numpy scipy scikit-learn lightgbm pyarrow

python3 build_dataset.py    # panel → residuals → features    (~1 min)
python3 train_wf.py         # walk-forward, reports OOS IC     (~4 min)
python3 run_backtest2.py    # costed portfolio + baselines     (~3 min)
python3 robustness.py       # nulls, deflated Sharpe, PBO      (~12 min)
python3 run_val.py          # independent 482-name check       (~5 min)
python3 ablation.py         # leave-one-block-out              (~25 min)
python3 export_charts.py    # regenerate report/report_data.json
```

`build_dataset.py` reads `../data/raw/*.csv` from the parent project. The
482-name validation set is fetched by `build_val.py` from a public GitHub
dataset; it is not in this repo.

Every number in this document and in the HTML report comes from these scripts.
Raw run logs are in `results/`.

### Environment notes that cost real time

- **pandas 3.0** removed `groupby.apply(..., include_groups=True)`. Use an
  explicit `pd.concat([f(g) for _, g in df.groupby(...)])`.
- `DataFrame.stack()` needs `future_stack=True` on pandas 3.0.
- `pyarrow` is required for the parquet intermediates.
- The cloud research environment had no market-data egress; all price data came
  from the local `data/raw` files.

---

## 9. Honest limits

**It needs leverage to matter.** 3.1%/yr unlevered on 4.2% vol is below cash for
much of the sample. A 10% return needs roughly 3× gross leverage: a margin
account, financing costs the 50 bp/yr borrow assumption only partly captures,
and the discipline to hold through a ~15% levered drawdown.

**It is operationally heavy.** 57 positions above 0.5%, 0.89 gross exposure,
40× annual turnover — roughly 20 trades a day, long and short, needing
fractional shares and a short locate on every name. Not a hand-run strategy.

**The statistical case is suggestive, not settled.** t = 2.18 over 1,666 days.
DSR 0.918, below 0.95. Neither subperiod significant alone. The replication
carries most of the weight.

**Survivorship bias is present in both samples.** Both universes are defined by
present-day membership. Market-neutral construction mutes this substantially but
does not remove it. Until a point-in-time constituent file with delisted names
exists, every number here has an unquantified upward tilt.

**No corporate-action model.** Prices are total-return adjusted, but earnings
dates, index adds/deletes, secondary offerings and M&A are unmodelled. At a
5-day horizon these are exactly the events most likely to produce large adverse
residuals the model cannot see coming.

**Capacity is untested.** No market-impact model beyond a flat spread
assumption. Fine at retail size; unknown above it.

---

## 10. Next steps, ranked by expected value per hour

**1. Widen the universe to 400–500 names.** Highest return available, and the
evidence is already in hand: the same pipeline on 482 names produced less than
half the volatility for the same IC and lifted net Sharpe from 0.74 to 1.22.
Breadth is the one lever with a theoretical guarantee (IR ≈ IC × √breadth). It
also moves the book toward the mid-caps where reversal actually lives.
*Acceptance:* OOS IC ≥ 0.020 with t ≥ 3 on the wider universe; annual vol below
3% at the same gross exposure.

**2. Buy point-in-time data with delisted names.** Sharadar or Norgate,
~$50–100/month. The only way to remove the one bias no modelling choice can
correct. Converts every result here from "suggestive" to "measured."
*Acceptance:* rebuilt universe includes names that delisted mid-sample; compare
IC and Sharpe against the survivor-only run and report the gap.

**3. Test the lean feature set out-of-sample.** The ablation says the risk/vol
block hurts. Confirm or reject on the wider universe — and register the decision
before looking, so it is a test rather than another selection.
*Acceptance:* pre-registered; adopt only if it beats the full model on data
neither variant has seen.

**4. Paper-trade a quarter and track IC, not P&L.** Sixty days of P&L on a 0.74
Sharpe strategy tells you nothing; noise dwarfs signal. Daily cross-sectional IC
converges far faster.
*Acceptance:* live IC within noise of 0.024. If materially below, the difference
is implementation — check fill prices against the assumed close, and realised
turnover against the assumed 40×.

**5. Do not optimise the holding period or the weight cap.** PBO 0.690 across a
config spread of 0.518–0.779 is a direct instruction. Leave them at 5 days and
8%.

### Not worth doing

- More features. The ablation shows six of seven blocks contribute ≤ 0.04 Sharpe
  each and one is negative. Feature count is not the constraint; breadth is.
- Hyperparameter search. It is what compromised the original model's reported
  numbers, and PBO says the response surface is flat anyway.
- A long-only version. It underperformed: model long-only Sharpe 1.066 vs 0.984
  for an equal-weight universe, excess t = 0.37. On a survivorship-biased
  universe of names that all went up, long-only measures the universe, not the
  model.

---

## 11. Literature the design rests on

| Paper | Finding | What it forced |
|---|---|---|
| Nagel, *Evaporating Liquidity* (NBER w17653) | Normalised VIX predicts next-day reversal returns, coef 0.22 (t≈11), adj R² 0.07 daily / 0.56 monthly | Regime features as interaction context |
| Novy-Marx & Velikov, *A Taxonomy of Anomalies and Their Trading Costs* (NBER w20721) | Short-term reversal +0.37%/mo gross → −1.28%/mo net (t=−6.02); buy/hold bands cut turnover 41% | Large-cap universe; overlapping tranches |
| Da, Liu & Schaumburg, *Decomposing Short-Term Return Reversal* (NY Fed SR513) | Across-industry component −0.295%/mo (t=−4.15); within-industry +0.821%/mo (t=5.49) | Residualise before ranking |
| Lou, Polk & Skouras, *A Tug of War* (JFE 2019) | Momentum 100% overnight (0.98%/mo, t=3.84); reversal +0.93% overnight, −1.05% intraday | Overnight/intraday decomposition |
| Bailey & López de Prado | Deflated Sharpe ratio; PBO via CSCV | The honesty tests in `src/stats.py` |

---

*Research note, not investment advice. Every figure is a backtest on
survivorship-biased data; realised results will differ.*
