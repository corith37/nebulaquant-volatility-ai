# Decision log — what was tried, what happened, what to not repeat

Chronological. The point of this file is to stop the next person (human or
model) from rediscovering the same dead ends. Every entry records the outcome,
not just the intent.

---

## D1 — Use the existing triple-barrier / two-stage model as the base
**Rejected.** The architecture, not the tuning, was the problem. Four issues,
each independently sufficient:

1. `feature_columns()` returns every column not on a deny-list, sweeping in
   `sma_20/50/200`, `atr_14`, `macd`, `macd_signal`, `vol_ma_20`,
   `dollar_volume` — all raw dollar levels. Not future leakage (it survives a
   purged split), but a tree memorises `sma_200 ∈ [310,315]` as a fingerprint for
   one name in one era, and the rule expires silently as prices drift.
2. `tb_label` is absolute per-stock. With mean pairwise correlation 0.267 across
   the universe, most explainable variance is beta — so the model becomes a
   market timer wearing a stock-picker's clothes. The eventual pivot to
   "regime-gated momentum" in `config.yaml` formalised exactly that.
3. `execution.target_positions: 3` with a 20-day hold ≈ 13 independent bets/year.
   IR ≈ IC × √breadth; at that breadth an annual result is a coin-flip sequence.
4. `optimize.py` selects hyperparameters on the median Sharpe *across the
   walk-forward test folds*, which makes those folds in-sample. Only the 15%
   holdout stays clean and 440 days cannot carry a 20-trial search.

**Kept from it:** the purged walk-forward concept, the uniqueness-weight idea,
and the general project layout. The engineering was sound; the framing wasn't.

---

## D2 — Predict absolute returns
**Rejected in favour of factor-residual returns.** Residualising drops mean
pairwise correlation from 0.267 to −0.011, which is what makes genuine
cross-sectional selection possible instead of disguised beta timing.

---

## D3 — Use GICS sectors for industry-relative reversal
**Not possible; replaced with rolling PCA.** No sector map available. A rolling
statistical factor model (equal-weight market + top-4 PCs of the market-residual
covariance, 252d window refit every 21d) subsumes industry structure and adapts
as correlations change, which a static sector map cannot. Verified working:
residual correlation −0.011.

---

## D4 — Short-term reversal as the primary signal
**Demoted.** The literature predicted this and the data confirmed it: in a
76-name mega-cap cross-section, `rev_5` reaches only t = 1.61 and `rev_1`
t = 1.89. Two further tests:

- Reversal IC **flips sign with market direction** (+0.026 market-up, −0.019
  market-down; corr with forward market return +0.18). It is partly an unhedged
  beta bet.
- Standalone, reversal nets 0.031 Sharpe from 0.401 gross — 67× turnover eats
  it, exactly Novy-Marx & Velikov.

Reversal stays in the feature set (ablation: worth +0.043) but it is not the
thesis. In the **482-name** universe it is materially stronger (rev_1 t = 2.61,
rev_5 t = 2.25), confirming it is a small/mid-cap effect. That is an argument
for widening the universe, not for weighting reversal harder here.

---

## D5 — First portfolio construction: quantile buckets + beta scaling
**Rejected — it was buggy and it hid the real result.** The construction picked
top/bottom quantiles, then scaled the short leg so `Σ w·β = 0`, renormalised
gross, and finally did `w -= w.mean()` to restore dollar neutrality.

Two failures:
- `w -= w.mean()` on a mostly-zero vector makes **every** name non-zero, smearing
  tiny weights across all 76 and inflating turnover. `avg_names` came out as
  exactly 74.587 for *every* quantile setting — the tell that something was
  wrong.
- You cannot be both dollar- and beta-neutral with only two free scalars (leg
  scalings) unless the legs happen to have equal beta. The two constraints
  fight, and the code silently satisfied neither.

Result: net Sharpe **−0.159**, t = −0.32, against an IC t-stat of +3.86. The
contradiction is what led to D6.

**Replaced by:** orthogonal projection (`src/portfolio.py`), which satisfies all
constraints simultaneously by construction and produces continuous weights.

---

## D6 — THE finding: neutralise to the prediction's own factor space
**Adopted.** The model predicts returns orthogonal to {market beta, PC1..PC4}.
Hedging only market beta leaves PC2–PC4 exposure whose variance swamps the
alpha. Projecting the signal orthogonal to the full loading matrix recovers it:

| Neutralisation | Net Sharpe |
|---|---:|
| Dollar-neutral only | −0.156 |
| + market beta | +0.460 |
| + 2 PCs | +0.674 |
| + 4 PCs | +0.738 |

**General rule for this codebase:** the target's definition and the portfolio's
neutrality constraints must reference the same factor set. If you change
`n_pc` in `factors.py`, you must change `FACTOR_COLS` in `portfolio.py` to match.

---

## D7 — Hyperparameter search
**Deliberately not done.** Hyperparameters are fixed a priori in
`src/model.py::PARAMS`. Tuning on the walk-forward folds is what compromised the
original model's reported numbers. The PBO result (0.690 across a config spread
of only 0.518–0.779) independently confirms the response surface is flat — there
is nothing to win and the apparent wins will not persist.

**Do not add an Optuna loop.** If you must explore, do it on a period you then
never report.

---

## D8 — Adopt the lean feature set (drop the risk/vol block)
**Deferred, not adopted.** Ablation says dropping the 10 risk/vol features
*raises* net Sharpe from 0.739 to 0.823. Plausible mechanism: those features
(beta, idio vol, skew, kurtosis, semidev, Parkinson vol, vol-of-vol) largely
duplicate exposures the portfolio construction already neutralises, so they add
estimation noise without information.

But acting on it is a selection decision made after seeing OOS results. The
pre-registered full model scores 0.739 and that is the number of record. Treat
the lean variant as a **hypothesis for the next independent test**, pre-register
it, and adopt only if it wins on data neither variant has seen.

---

## D9 — Long-only variant
**Rejected.** Model long-only Sharpe 1.066 vs 0.984 for an equal-weight
universe; excess return +0.8%/yr with t = 0.37. On a survivorship-biased
universe of names that all went up, a long-only backtest measures the universe,
not the model. The market-neutral construction is what makes the survivorship
bias tolerable.

---

## D10 — Optimise holding period / weight cap
**Rejected on evidence.** Net Sharpe across hold periods: 2d → 0.520, 3d →
0.630, 5d → 0.738, 10d → 0.779, 21d → 0.735. Flat, which is reassuring but also
means picking the max (10d) is fitting noise. PBO 0.690 says the same thing
formally. Left at 5 days / 8% cap.

---

## D11 — Python triple-loop for overlapping tranches
**Rejected on performance.** The original `build_positions` looped
dates × tickers × tranches: ~90 s per configuration, which made the robustness
suite (40 nulls × 3 + 15 configs) take hours.

**Replaced by** `signal_weights()` (one small `lstsq` per date) plus
`positions_from_weights()`, which is just a shifted rolling mean of the target
weight matrix — a tranche signalled on *t*, filled at *t+lag*, held `hold` days
*is* `W.shift(lag+1).rolling(hold).mean()`. **90 s → 5 s.**

---

## D12 — `factor_cols=[]` to mean "no factors"
**Bug, fixed.** `factor_cols or FACTOR_COLS` treats an empty list as falsy, so
passing `[]` silently applied the *full* factor set. It made the
"dollar-neutral only" arm of the ladder produce results identical to the full
arm (0.738 vs 0.738) — which is how it was caught. Now an explicit
`if factor_cols is None`. Pass `['__none__']` for a genuinely factor-free run.

**Lesson for this codebase:** never use `x or DEFAULT` where `x` may legitimately
be an empty collection.

---

## D13 — matplotlib PNG charts in the report
**Rejected.** PNGs bake in one theme's colours and cannot adapt to a reader's
light/dark setting, and they lose hover inspection. Replaced with hand-written
inline SVG driven by embedded JSON and CSS custom properties, so the report is
theme-aware and interactive, and every rendered number traces to
`report/report_data.json`.

---

## D14 — Which null distribution to quote
**Refined after looking at the chart.** The i.i.d. random null was drawn at
*zero* cost while the model is quoted at 2.5 bp, so the naive comparison is not
like-for-like — and the best cost-free random draw (+0.735) lands within a
rounding error of the model's costed +0.738. The report now quotes the
cost-matched permutation null (best draw +0.226, p = 0.0000) as the primary test
and states the zero-cost comparison against the model's own zero-cost Sharpe
(0.984). Both give p = 0.0000, but only one of them is an honest comparison.

---

## Environment gotchas (cost real time, will recur)

- **pandas 3.0** removed `groupby.apply(..., include_groups=True)` — use an
  explicit `pd.concat([f(g) for _, g in df.groupby(...)])`.
- **pandas 3.0** `DataFrame.stack()` requires `future_stack=True`.
- **pyarrow** is needed for the parquet intermediates; it is not preinstalled.
- yfinance files carry both `Close` (split-adjusted) and `Adj Close`
  (split + dividend). Open/high/low are split-adjusted only and **must** be
  rescaled by the dividend factor, or every ex-dividend date shows a fake −0.5%
  overnight gap. Handled in `src/panel.py`.
- The research environment had **no market-data network egress** — no Yahoo, no
  Stooq. All prices came from the local `data/raw` files. Plan any future data
  work around that.

---

## D15 — Reproduce the archive before building on it
**Done; the archive is faithful.** Every documented number was regenerated from
the shipped code on 1 Sep 2026 and matches to the printed precision:

| Check | Archive | Re-run |
|---|---:|---:|
| dataset rows / span | 177,709 · 2017-01-03→2026-08-21 | identical |
| mean pairwise corr raw → residual | 0.267 → −0.011 | identical |
| OOS IC (NW t) | +0.0240 (3.86) | +0.0240 (3.86) |
| net Sharpe @2.5bp | +0.738 | +0.738 |
| neutralisation ladder | −0.156 / +0.460 / +0.738 | identical |
| all 42 univariate ICs | — | max |Δ| = 0.00e+00 |

Conclusion: `results/` can be trusted as a reference. Diff against it after any
change rather than re-deriving from scratch.

**One documentation slip found:** HANDOFF §6 quotes `mom_252_21` at NW t = +3.53.
Its own `results/ic_full.csv` says **+3.2997**. The signal is still the strongest
positive and no conclusion changes, but the table should be corrected.

---

## D16 — The pipeline was not runnable from a clean checkout
**Fixed.** Three producers were missing, so `python3 build_dataset.py` failed on
line 8 and the documented reproduction sequence could not be executed at all:

1. **No stage 0.** `build_dataset.py` opens `data_proc/panel.parquet`, but
   nothing called `src/panel.py::build_panel`. HANDOFF said "build_dataset.py
   reads ../data/raw/*.csv"; the code did not. → added `build_panel.py`.
2. **`W_model.parquet` had no writer.** `robustness.py` reads it on line 13.
   → `run_backtest2.py` now saves the model's target weights.
3. **`ic_full.csv` had no writer.** HANDOFF §6 cites it for 42 signals and
   `export_charts.py` reads it. → added `run_ic_study.py`, which also
   regenerates the raw-return and market-direction qualifications in §6.

Also: `data_proc/` and `out/` are not created by any script, and the scripts
write to `out/` while the archive lives in `results/`. Left as-is deliberately —
re-runs land in `out/` and can be diffed against the immutable `results/`.

**Lesson:** a findings archive is not a pipeline. Anything cited in HANDOFF must
have a script that regenerates it, or the number is unfalsifiable.

---

## D17 — `build_val.py` violates House Rule #2
**Confirmed bug; the validation Sharpe is not what it claims to be.**
`build_val.py` residualises to `n_pc=6`, but `run_val.py` builds the book with
the default `FACTOR_COLS` = beta + **4** PCs. The 482-name portfolio therefore
carried unhedged PC5/PC6 exposure — precisely the D6 failure mode.

Measured on the primary universe (`test_npc_match.py`), target orthogonal to
beta + 6 PCs, only the book changing:

| Book neutral to | Net Sharpe | ann | t |
|---|---:|---:|---:|
| beta + 6 PCs (matched) | **+0.622** | +2.52% | 1.71 |
| beta + 4 PCs (what run_val.py did) | +0.417 | +1.74% | 1.18 |
| beta + 2 PCs | +0.421 | +1.88% | 1.18 |
| beta only | +0.245 | +1.20% | 0.72 |
| dollar-neutral only | −0.069 | −0.37% | −0.11 |

The mismatch costs **0.205 Sharpe**, so the validation's +1.224 is *understated*,
not flattered. Good news for the breadth thesis — but fix it before quoting it.

**Second, less comfortable finding: `n_pc` is itself consequential.** On the
primary universe a 6-PC target scores +0.622 against the archived 4-PC target's
+0.738, and OOS IC falls from +0.0240 (t 3.86) to +0.0223 (t 3.43). One run, so
not settled — but it means the primary and validation runs were not measuring the
same target. **Do not treat `n_pc` as a free knob**; it belongs with hold period
and weight cap under D10 (do not optimise), and any universe compared against the
primary must use `n_pc=4`.

---

## D18 — "Sharpe 0.74 → 1.22" is not a controlled comparison
**Downgraded as evidence.** It is quoted in HANDOFF §10 and CLAUDE.md as the
justification for next-step #1. The two runs differ in five ways at once:

| | primary | validation |
|---|---|---|
| target factor space | beta + 4 PC | beta + **6** PC |
| book neutral to | beta + 4 PC | beta + 4 PC (**mismatched**, D17) |
| seeds | 3 | **2** |
| min_train_days | 756 | **252** |
| block_days | 63 | **42** |

The **IC** replication is robust to all of this (+0.0240 vs +0.0255) and remains
the strongest evidence in the study — ranking quality is protocol-insensitive.
The **Sharpe** comparison is not, and the validation portfolio's own t-stat is
**+1.52** (falling to +0.92 at 5 bp), which HANDOFF §3 omits while quoting
t-stats everywhere else. Breadth is still the right priority — it just needs a
clean test, which is what `run_wide.py` runs.

---

## D19 — Wide universes need a corruption gate, not just a split check
**Added to `src/panel.py::qa_report`.** The 503-name S&P pull contained one
badly broken series: **MNST** alternated between ~47 and ~93 for weeks in
Jul–Aug 2026 (a 2:1 split applied inconsistently by the vendor), including a
+95.6% day with a 1.5% intraday range. The existing `suspect_unadj_split` check
counted those bars but could not distinguish them from a genuine +40% M&A day
(BIIB, ECHO and FLEX all legitimately trip it).

Two new columns separate the cases:
- `n_oscillate` — bars >30% from their own **centred** 5-day median. A one-off
  jump barely registers; a flip-flop puts every other bar far from the centre.
- `round_trips` — a >35% move reversed by a >30% opposite move within 3 bars.

`corrupt = n_oscillate >= 3 or round_trips >= 2` flags MNST and **nothing else**
across 503 names, and flags nothing in the original 76. `build_panel.py` prints
the exclusion command. This matters more as breadth grows: one oscillating name
corrupts the equal-weight market factor for *every* other name.

---

## D20 — Implementation lag costs less than feared
**Measured, no action needed.** The target is the residual over t+1..t+5, but the
book earns t+2..t+6 (sign at close t, fill at close t+1). The same OOS
predictions scored against the window actually traded:

| Scored against | IC | NW t |
|---|---:|---:|
| trained window t+1..t+5 (the reported number) | +0.0240 | +3.86 |
| traded window t+2..t+6 (what the P&L earns) | +0.0229 | +3.61 |

~5% relative decay. The signal is not fragile to the lag, and retraining on the
traded window is not worth the complexity. **But when paper-trading (next step
#4), benchmark live IC against +0.0229, not the headline +0.0240.**

---

## D21 — SPY and QQQ sit in the tradeable panel
**Noted; low priority.** `build_panel(exclude=("VIX",))` keeps both ETFs, so they
enter the equal-weight market factor, the PCA covariance, and the book. The
projection largely defends itself — they receive the two *smallest* mean absolute
weights of all 76 names (0.0067 and 0.0070 vs a 0.0129 median), together 1.37% of
gross. Worth removing for cleanliness, and the "76 names" breadth figure is really
74 stocks, but it is not moving any result. The wide universe has no ETFs.
