"""Backtest-driven hyperparameter optimization.

The optimizer searches XGBoost hyperparameters, triple-barrier multiples, and
the probability threshold to maximise a *risk-adjusted, out-of-sample* objective:

    score = AGG(fold Sharpe) - penalty(worst-fold max drawdown breaching the cap)

where AGG is the configured fold aggregation (median by default, which resists
overfitting to a single lucky fold). Every evaluation is fully walk-forward and
the final untouched holdout is never seen (``include_holdout=False``).

Optuna is used when installed; otherwise a reproducible random search over the
same space is used so there is no hard new dependency.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.backtest import portfolio_backtest, walk_forward_predict
from src.risk import RiskConfig
from src.utils import setup_logger
from src.validation import WalkForwardConfig

logger = setup_logger("nebulaquant.optimize")


# ----------------------------------------------------------------------------
# Search space
# ----------------------------------------------------------------------------

# (name, kind) where kind is "int" or "float". Ranges come from config.
_INT_PARAMS = {"max_depth", "n_estimators", "min_child_weight"}
_MODEL_PARAMS = {
    "max_depth", "learning_rate", "n_estimators",
    "min_child_weight", "subsample", "gamma",
}
_RISK_PARAMS = {"probability_threshold", "tp_atr_mult", "sl_atr_mult"}


def _suggest_optuna(trial, space: Dict[str, list]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, rng in space.items():
        lo, hi = rng[0], rng[1]
        if name in _INT_PARAMS:
            out[name] = trial.suggest_int(name, int(lo), int(hi))
        else:
            out[name] = trial.suggest_float(name, float(lo), float(hi))
    return out


def _suggest_random(rng: np.random.Generator, space: Dict[str, list]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, r in space.items():
        lo, hi = r[0], r[1]
        if name in _INT_PARAMS:
            out[name] = int(rng.integers(int(lo), int(hi) + 1))
        else:
            out[name] = float(rng.uniform(float(lo), float(hi)))
    return out


def _split_params(sampled: Dict[str, float]) -> Tuple[Dict, float, float, float]:
    """Split a sampled config into (model_params, threshold, tp_mult, sl_mult)."""
    model_params = {k: v for k, v in sampled.items() if k in _MODEL_PARAMS}
    threshold = float(sampled.get("probability_threshold", 0.55))
    tp_mult = float(sampled.get("tp_atr_mult", 2.0))
    sl_mult = float(sampled.get("sl_atr_mult", 1.0))
    return model_params, threshold, tp_mult, sl_mult


# ----------------------------------------------------------------------------
# Objective
# ----------------------------------------------------------------------------

@dataclass
class OptimizeConfig:
    objective: str = "sharpe"
    max_drawdown_cap: float = 0.20
    fold_aggregation: str = "median"      # median | mean | min
    n_trials: int = 40
    use_optuna: bool = True
    penalty_weight: float = 5.0           # Sharpe units removed per 100% DD over cap
    calibrate: bool = True
    initial_capital: float = 10_000.0


def _aggregate(values: List[float], how: str) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=float)
    if how == "mean":
        return float(arr.mean())
    if how == "min":
        return float(arr.min())
    return float(np.median(arr))


def evaluate_config(
    df: pd.DataFrame,
    feat_cols: List[str],
    wf_cfg: WalkForwardConfig,
    bt_cfg: dict,
    opt_cfg: OptimizeConfig,
    sampled: Dict[str, float],
    two_stage: bool = True,
    random_state: int = 42,
) -> Tuple[float, dict]:
    """Run a walk-forward backtest for one sampled config; return (score, info)."""
    model_params, threshold, tp_mult, sl_mult = _split_params(sampled)

    preds = walk_forward_predict(
        df,
        feat_cols=feat_cols,
        wf_cfg=wf_cfg,
        params=model_params,
        calibrate=opt_cfg.calibrate,
        two_stage=two_stage,
        random_state=random_state,
        include_holdout=False,   # NEVER touch the holdout during optimization
    )
    if preds.empty or "fold" not in preds.columns:
        return -1e9, {"reason": "no_predictions"}

    risk = RiskConfig(
        tp_atr_mult=tp_mult,
        sl_atr_mult=sl_mult,
        max_hold_days=bt_cfg.get("max_hold_days", 10),
        commission_pct=bt_cfg.get("commission_pct", 0.001),
        slippage_pct=bt_cfg.get("slippage_pct", 0.001),
        position_size_pct=bt_cfg.get("position_size_pct", 0.10),
        max_position_pct=bt_cfg.get("max_position_pct", 0.20),
        max_total_exposure=bt_cfg.get("max_total_exposure", 1.0),
        vol_target=bt_cfg.get("vol_target", True),
        prob_threshold=threshold,
        regime_filter=bt_cfg.get("regime_filter", True),
        max_vix_pctile=bt_cfg.get("max_vix_pctile", 0.85),
    )

    fold_sharpes: List[float] = []
    fold_dds: List[float] = []
    total_trades = 0
    for fold, grp in preds.groupby("fold"):
        _, eq, m = portfolio_backtest(grp, risk, initial_capital=opt_cfg.initial_capital)
        fold_sharpes.append(m["sharpe"])
        fold_dds.append(abs(m["max_drawdown"]))
        total_trades += m["n_trades"]

    agg_sharpe = _aggregate(fold_sharpes, opt_cfg.fold_aggregation)
    worst_dd = max(fold_dds) if fold_dds else 0.0

    # Drawdown-cap penalty (risk-adjusted objective per the user's choice).
    breach = max(worst_dd - opt_cfg.max_drawdown_cap, 0.0)
    penalty = opt_cfg.penalty_weight * breach
    # Discourage degenerate configs that almost never trade.
    if total_trades < max(2 * wf_cfg.n_splits, 10):
        penalty += 1.0

    score = agg_sharpe - penalty
    info = {
        "agg_sharpe": round(agg_sharpe, 4),
        "worst_fold_dd": round(worst_dd, 4),
        "penalty": round(penalty, 4),
        "total_trades": int(total_trades),
        "fold_sharpes": [round(s, 4) for s in fold_sharpes],
    }
    return float(score), info


# ----------------------------------------------------------------------------
# Search drivers
# ----------------------------------------------------------------------------

def optimize(
    df: pd.DataFrame,
    feat_cols: List[str],
    wf_cfg: WalkForwardConfig,
    bt_cfg: dict,
    opt_cfg: OptimizeConfig,
    search_space: Dict[str, list],
    two_stage: bool = True,
    random_state: int = 42,
) -> dict:
    """Run the search and return the best config + bookkeeping."""
    eval_fn = lambda sampled: evaluate_config(
        df, feat_cols, wf_cfg, bt_cfg, opt_cfg, sampled,
        two_stage=two_stage, random_state=random_state,
    )

    if opt_cfg.use_optuna:
        try:
            import optuna  # noqa: F401
            return _optimize_optuna(eval_fn, search_space, opt_cfg, random_state)
        except ImportError:
            logger.warning("optuna not installed; falling back to random search.")

    return _optimize_random(eval_fn, search_space, opt_cfg, random_state)


def _optimize_optuna(eval_fn: Callable, search_space, opt_cfg: OptimizeConfig, random_state: int) -> dict:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    trial_info: Dict[int, dict] = {}

    def objective(trial):
        sampled = _suggest_optuna(trial, search_space)
        score, info = eval_fn(sampled)
        trial_info[trial.number] = info
        trial.set_user_attr("info", info)
        logger.info(f"[optuna trial {trial.number}] score={score:.4f} {info}")
        return score

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=opt_cfg.n_trials)

    best = study.best_trial
    sampled = best.params
    model_params, threshold, tp_mult, sl_mult = _split_params(sampled)
    return {
        "method": "optuna",
        "best_score": float(best.value),
        "best_info": best.user_attrs.get("info", {}),
        "model_params": model_params,
        "probability_threshold": threshold,
        "tp_atr_mult": tp_mult,
        "sl_atr_mult": sl_mult,
        "n_trials": opt_cfg.n_trials,
        "raw_params": sampled,
    }


def _optimize_random(eval_fn: Callable, search_space, opt_cfg: OptimizeConfig, random_state: int) -> dict:
    rng = np.random.default_rng(random_state)
    best_score = -np.inf
    best = None
    for i in range(opt_cfg.n_trials):
        sampled = _suggest_random(rng, search_space)
        score, info = eval_fn(sampled)
        logger.info(f"[random trial {i}] score={score:.4f} {info}")
        if score > best_score:
            best_score = score
            best = (sampled, info)

    sampled, info = best if best else ({}, {})
    model_params, threshold, tp_mult, sl_mult = _split_params(sampled)
    return {
        "method": "random_search",
        "best_score": float(best_score),
        "best_info": info,
        "model_params": model_params,
        "probability_threshold": threshold,
        "tp_atr_mult": tp_mult,
        "sl_atr_mult": sl_mult,
        "n_trials": opt_cfg.n_trials,
        "raw_params": sampled,
    }


def save_best_params(result: dict, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "best_params.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info(f"Saved best params -> {path}")
    return path
