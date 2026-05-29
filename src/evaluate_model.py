"""Evaluation for the two-stage model.

Produces, for the long meta-label (Stage B) on a held-out set:
  - threshold-based classification metrics (accuracy/precision/recall/F1)
  - a calibration (reliability) table - predicted prob vs realized hit rate
  - confusion matrix
  - feature importance (from the underlying XGB booster)
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.train_model import TwoStageModel, _pos_proba
from src.utils import setup_logger

logger = setup_logger("nebulaquant.evaluate")


def _underlying_xgb(stage):
    """Return the XGBClassifier inside a (possibly calibrated) stage, or None."""
    if stage is None:
        return None
    if hasattr(stage, "feature_importances_"):
        return stage
    # CalibratedClassifierCV(prefit) -> .estimator ; older -> .calibrated_classifiers_
    est = getattr(stage, "estimator", None)
    if est is not None and hasattr(est, "feature_importances_"):
        return est
    cc = getattr(stage, "calibrated_classifiers_", None)
    if cc:
        inner = getattr(cc[0], "estimator", None)
        if inner is not None and hasattr(inner, "feature_importances_"):
            return inner
    return None


def calibration_table(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Reliability table: mean predicted prob vs observed frequency per bin."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        rows.append({
            "bin": f"{bins[b]:.1f}-{bins[b+1]:.1f}",
            "mean_pred": float(p[mask].mean()),
            "observed_freq": float(y_true[mask].mean()),
            "count": int(mask.sum()),
        })
    return pd.DataFrame(rows)


def evaluate(
    model: TwoStageModel,
    holdout_df: pd.DataFrame,
    out_dir: Path,
    threshold: float = 0.5,
) -> dict:
    """Evaluate Stage B on the holdout and persist metric files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = holdout_df[model.feat_cols]
    y_true = holdout_df["tb_label"].astype(int).to_numpy()
    p_long = _pos_proba(model.stage_b, X)
    preds = (p_long >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "base_rate": float(y_true.mean()),
        "n_holdout": int(len(y_true)),
        "threshold": float(threshold),
    }
    pd.DataFrame([metrics]).to_csv(out_dir / "model_metrics.csv", index=False)

    with open(out_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(
            y_true, preds, target_names=["no_win", "long_win"], zero_division=0,
        ))

    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    pd.DataFrame(
        cm, index=["true_no_win", "true_long_win"],
        columns=["pred_no_win", "pred_long_win"],
    ).to_csv(out_dir / "confusion_matrix.csv")

    calibration_table(y_true, p_long).to_csv(out_dir / "calibration_curve.csv", index=False)

    xgb = _underlying_xgb(model.stage_b)
    if xgb is not None:
        imp = pd.DataFrame({
            "feature": model.feat_cols,
            "importance": xgb.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        imp.to_csv(out_dir / "feature_importance.csv", index=False)

    logger.info(f"Evaluation written to {out_dir} | metrics={metrics}")
    return metrics
