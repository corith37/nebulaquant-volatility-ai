"""Purged, embargoed, expanding walk-forward training.

Hyperparameters are FIXED a priori, not searched. That is a deliberate design
choice. The most common way a strategy like this dies is that the researcher
tunes depth/learning-rate/threshold against the same walk-forward folds they then
report as "out-of-sample" - at which point the folds are in-sample and the
reported Sharpe is a selection statistic, not an expectation. Capacity is set
low (few leaves, heavy L2, high min_child_samples) because with ~76 names the
effective cross-sectional breadth is small and a deep tree will memorise names.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import lightgbm as lgb

PARAMS = dict(
    objective="regression",
    metric="l2",
    num_leaves=15,
    max_depth=4,
    learning_rate=0.03,
    n_estimators=300,
    min_child_samples=200,
    feature_fraction=0.6,
    bagging_fraction=0.7,
    bagging_freq=1,
    lambda_l2=10.0,
    verbose=-1,
    n_jobs=2,
    seed=7,
)


def walk_forward(
    df: pd.DataFrame,
    feat_cols: list[str],
    target: str = "y",
    horizon: int = 5,
    embargo: int = 5,
    min_train_days: int = 756,
    block_days: int = 63,
    params: dict | None = None,
    seeds: tuple[int, ...] = (7, 17, 27),
) -> pd.DataFrame:
    """Score every out-of-sample block with a model trained only on its past.

    Purge = `horizon` (the label's own look-ahead) + `embargo`. Because the
    window is expanding, training data always precedes the test block, so a
    one-sided purge at the left edge of the block is sufficient.

    Averaging over several seeds is not an accuracy trick - it reduces the
    variance that LightGBM's row/column subsampling injects, so that the reported
    OOS number reflects the strategy rather than one lucky random state.
    """
    params = {**PARAMS, **(params or {})}
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    dates = np.sort(df["date"].unique())
    n = len(dates)
    purge = horizon + embargo

    preds, importances = [], []
    start = min_train_days
    while start < n:
        stop = min(start + block_days, n)
        train_end = start - purge
        if train_end <= 252:
            start = stop
            continue
        tr_dates = dates[:train_end]
        te_dates = dates[start:stop]
        tr = df[df["date"].isin(tr_dates)]
        te = df[df["date"].isin(te_dates)]
        if len(tr) < 5000 or te.empty:
            start = stop
            continue

        Xtr, ytr = tr[feat_cols].to_numpy(np.float32), tr[target].to_numpy(np.float32)
        Xte = te[feat_cols].to_numpy(np.float32)
        pr = np.zeros(len(te))
        for s in seeds:
            m = lgb.LGBMRegressor(**{**params, "seed": s})
            m.fit(Xtr, ytr)
            pr += m.predict(Xte) / len(seeds)
        importances.append(pd.Series(m.feature_importances_, index=feat_cols))

        out = te[["date", "ticker"]].copy()
        out["pred"] = pr
        out["block_start"] = dates[start]
        preds.append(out)
        start = stop

    P = pd.concat(preds, ignore_index=True)
    # Rank predictions within date: only relative order matters downstream, and
    # ranking makes the signal robust to drift in the model's output scale.
    P["pred_rank"] = P.groupby("date")["pred"].rank(pct=True) - 0.5
    imp = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
    P.attrs["importance"] = imp
    return P
