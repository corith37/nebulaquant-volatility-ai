"""Backtest-driven hyperparameter optimization.

Searches model + barrier + threshold params to maximise a risk-adjusted,
walk-forward out-of-sample objective (median-fold Sharpe penalized when the
worst fold breaches the drawdown cap). The final untouched holdout is never
seen here. Writes the winner to reports/metrics/best_params.json, which both
train and run_backtest pick up automatically.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import load_config, ensure_paths
from src.features import feature_columns
from src.optimize import OptimizeConfig, optimize, save_best_params
from src.utils import setup_logger
from src.validation import WalkForwardConfig


def main() -> int:
    logger = setup_logger("nebulaquant.optimize_script")
    cfg = load_config()
    paths = ensure_paths()

    dataset_path = paths["data_processed"] / "model_dataset.csv"
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}. Run build_dataset first.")
        return 1

    df = pd.read_csv(dataset_path, parse_dates=["Date"])
    if df.empty:
        logger.error("Dataset is empty.")
        return 1

    feat_cols = feature_columns(df)
    model_cfg = cfg.get("model", {})
    wf_raw = cfg.get("walkforward", {})
    tb_cfg = cfg.get("triple_barrier", {})
    bt_cfg = cfg.get("backtest", {})
    opt_raw = cfg.get("optimization", {})

    wf_cfg = WalkForwardConfig(
        n_splits=wf_raw.get("n_splits", 5),
        horizon=tb_cfg.get("horizon", 10),
        embargo=wf_raw.get("embargo", 5),
        holdout_frac=wf_raw.get("holdout_frac", 0.15),
        min_train_dates=wf_raw.get("min_train_dates", 504),
    )
    opt_cfg = OptimizeConfig(
        objective=opt_raw.get("objective", "sharpe"),
        max_drawdown_cap=opt_raw.get("max_drawdown_cap", 0.20),
        fold_aggregation=opt_raw.get("fold_aggregation", "median"),
        n_trials=opt_raw.get("n_trials", 40),
        use_optuna=opt_raw.get("use_optuna", True),
        calibrate=model_cfg.get("calibrate", True),
        initial_capital=bt_cfg.get("initial_capital", 10_000),
    )
    search_space = opt_raw.get("search_space", {})
    if not search_space:
        logger.error("No optimization.search_space configured.")
        return 1

    logger.info(
        f"Optimizing {opt_cfg.n_trials} trials | objective={opt_cfg.objective} "
        f"| dd_cap={opt_cfg.max_drawdown_cap} | agg={opt_cfg.fold_aggregation}"
    )
    result = optimize(
        df,
        feat_cols=feat_cols,
        wf_cfg=wf_cfg,
        bt_cfg=bt_cfg,
        opt_cfg=opt_cfg,
        search_space=search_space,
        two_stage=model_cfg.get("two_stage", True),
        random_state=model_cfg.get("random_state", 42),
    )
    save_best_params(result, paths["reports_metrics"])

    logger.info("=== Best config ===")
    logger.info(f"  method: {result['method']}  best_score: {result['best_score']:.4f}")
    logger.info(f"  info: {result['best_info']}")
    logger.info(f"  model_params: {result['model_params']}")
    logger.info(
        f"  threshold={result['probability_threshold']:.4f} "
        f"tp={result['tp_atr_mult']:.3f} sl={result['sl_atr_mult']:.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
