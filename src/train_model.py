"""Two-stage, probability-calibrated XGBoost model.

Stage A  - expansion detector: P(volatility expands over the next N bars).
Stage B  - long meta-label: P(a long triple-barrier trade hits TP before SL).
Combined score = P_A * P_B  (the spec's volatility x direction formula).

Each stage is an XGBoost classifier wrapped in isotonic calibration so the
emitted probabilities are meaningful (a 0.6 means ~60% empirically). Calibration
is fit on a *later* chronological slice of the training window than the base
model, so it never peeks at the future.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

try:  # sklearn >= 1.6 removed cv="prefit" in favour of FrozenEstimator.
    from sklearn.frozen import FrozenEstimator
    _HAS_FROZEN = True
except ImportError:  # pragma: no cover - older sklearn
    FrozenEstimator = None
    _HAS_FROZEN = False

from src.features import feature_columns
from src.utils import setup_logger

logger = setup_logger("nebulaquant.train")


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------

class _ConstantProb:
    """Fallback predictor when a training slice has a single class."""

    def __init__(self, p: float):
        self.p = float(np.clip(p, 1e-6, 1 - 1e-6))

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1 - self.p), np.full(n, self.p)])


def _pos_proba(model, X) -> np.ndarray:
    """Probability of the positive class (column for label==1)."""
    if model is None:
        return np.ones(len(X))
    proba = model.predict_proba(X)
    # Find the column for class 1 if classes_ available, else assume col 1.
    classes = getattr(model, "classes_", None)
    if classes is not None and 1 in list(classes):
        idx = list(classes).index(1)
        return proba[:, idx]
    return proba[:, -1]


def _make_xgb(params: Dict, y: pd.Series, random_state: int) -> XGBClassifier:
    pos = max(int((y == 1).sum()), 1)
    neg = max(int((y == 0).sum()), 1)
    return XGBClassifier(
        n_estimators=int(params.get("n_estimators", 400)),
        max_depth=int(params.get("max_depth", 5)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        subsample=float(params.get("subsample", 0.9)),
        colsample_bytree=float(params.get("colsample_bytree", 0.9)),
        min_child_weight=float(params.get("min_child_weight", 1.0)),
        gamma=float(params.get("gamma", 0.0)),
        scale_pos_weight=neg / pos,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )


def _train_calibrated(
    X: pd.DataFrame,
    y: pd.Series,
    params: Dict,
    sample_weight: Optional[np.ndarray] = None,
    calibrate: bool = True,
    random_state: int = 42,
):
    """Fit an XGB classifier with optional leakage-aware isotonic calibration."""
    y = y.astype(int)
    if y.nunique() < 2:
        return _ConstantProb(float(y.mean()) if len(y) else 0.0)

    can_calibrate = calibrate and len(X) > 300
    if can_calibrate:
        cut = int(len(X) * 0.8)
        Xf, yf = X.iloc[:cut], y.iloc[:cut]
        Xc, yc = X.iloc[cut:], y.iloc[cut:]
        wf = sample_weight[:cut] if sample_weight is not None else None
        if yf.nunique() < 2 or yc.nunique() < 2:
            can_calibrate = False

    if not can_calibrate:
        model = _make_xgb(params, y, random_state)
        model.fit(X, y, sample_weight=sample_weight)
        return model

    base = _make_xgb(params, yf, random_state)
    base.fit(Xf, yf, sample_weight=wf)
    # Fit isotonic calibration on the later (held-out) slice without refitting
    # the base estimator. sklearn >= 1.6 uses FrozenEstimator; older uses prefit.
    if _HAS_FROZEN:
        calib = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
    else:  # pragma: no cover - older sklearn
        calib = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
    calib.fit(Xc, yc)
    return calib


# ----------------------------------------------------------------------------
# Two-stage model container
# ----------------------------------------------------------------------------

@dataclass
class TwoStageModel:
    feat_cols: List[str]
    stage_a: object          # expansion detector (or None)
    stage_b: object          # long meta-label
    two_stage: bool = True

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return expansion_prob, long_prob, and combined score for each row."""
        X = df[self.feat_cols]
        pb = _pos_proba(self.stage_b, X)
        pa = _pos_proba(self.stage_a, X) if (self.two_stage and self.stage_a) else np.ones(len(X))
        return pd.DataFrame({
            "expansion_prob": pa,
            "long_prob": pb,
            "score": pa * pb,
        }, index=df.index)


def train_two_stage(
    train_df: pd.DataFrame,
    feat_cols: List[str],
    params: Optional[Dict] = None,
    calibrate: bool = True,
    two_stage: bool = True,
    random_state: int = 42,
) -> TwoStageModel:
    """Train Stage A (expansion) and Stage B (long meta-label) on `train_df`.

    `train_df` must be sorted by Date and contain `expansion_label`, `tb_label`,
    and `sample_weight` columns.
    """
    train_df = train_df.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    params = params or {}
    X = train_df[feat_cols]
    w = train_df["sample_weight"].to_numpy() if "sample_weight" in train_df else None

    stage_b = _train_calibrated(
        X, train_df["tb_label"], params, sample_weight=w,
        calibrate=calibrate, random_state=random_state,
    )
    stage_a = None
    if two_stage and "expansion_label" in train_df:
        stage_a = _train_calibrated(
            X, train_df["expansion_label"], params, sample_weight=w,
            calibrate=calibrate, random_state=random_state,
        )
    return TwoStageModel(feat_cols=feat_cols, stage_a=stage_a, stage_b=stage_b, two_stage=two_stage)


# ----------------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------------

def save_model(model: TwoStageModel, model_dir: Path) -> None:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "two_stage_model.joblib")
    with open(model_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(model.feat_cols, f, indent=2)
    logger.info(f"Saved two-stage model -> {model_dir / 'two_stage_model.joblib'}")


def load_model(model_dir: Path) -> Tuple[TwoStageModel, List[str]]:
    model_dir = Path(model_dir)
    path = model_dir / "two_stage_model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    model: TwoStageModel = joblib.load(path)
    return model, model.feat_cols
