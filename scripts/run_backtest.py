"""Walk-forward, out-of-sample backtest for the long-only volatility strategy.

This is the honest profitability estimate: the two-stage model is retrained on
each purged walk-forward fold and only ever trades rows it never saw in
training (plus a final untouched holdout). Strategy metrics are reported
alongside SPY buy-and-hold and a random-entry baseline so any edge has to be
demonstrated, not assumed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.backtest import (
    benchmarks,
    portfolio_backtest,
    save_backtest_outputs,
    walk_forward_predict,
)
from src.config import load_config, ensure_paths
from src.features import feature_columns
from src.risk import RiskConfig
from src.utils import setup_logger
from src.validation import WalkForwardConfig


def main() -> int:
    logger = setup_logger("nebulaquant.backtest_script")
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
    wf_cfg_raw = cfg.get("walkforward", {})
    tb_cfg = cfg.get("triple_barrier", {})
    bt_cfg = cfg.get("backtest", {})

    # Optimizer-tuned params (model hyperparams + barrier multiples + threshold).
    best_params_path = paths["reports_metrics"] / "best_params.json"
    tuned = {}
    if best_params_path.exists():
        tuned = json.loads(best_params_path.read_text(encoding="utf-8"))
        logger.info(f"Loaded tuned params: {tuned}")
    model_params = tuned.get("model_params", {})

    wf_cfg = WalkForwardConfig(
        n_splits=wf_cfg_raw.get("n_splits", 5),
        horizon=tb_cfg.get("horizon", 10),
        embargo=wf_cfg_raw.get("embargo", 5),
        holdout_frac=wf_cfg_raw.get("holdout_frac", 0.15),
        min_train_dates=wf_cfg_raw.get("min_train_dates", 504),
    )

    logger.info("Running walk-forward OOS predictions (retrain per fold)...")
    preds = walk_forward_predict(
        df,
        feat_cols=feat_cols,
        wf_cfg=wf_cfg,
        params=model_params,
        calibrate=model_cfg.get("calibrate", True),
        two_stage=model_cfg.get("two_stage", True),
        random_state=model_cfg.get("random_state", 42),
        include_holdout=True,
    )

    threshold = tuned.get("probability_threshold", model_cfg.get("probability_threshold", 0.55))
    risk = RiskConfig(
        tp_atr_mult=tuned.get("tp_atr_mult", bt_cfg.get("take_profit_atr_mult", 2.0)),
        sl_atr_mult=tuned.get("sl_atr_mult", bt_cfg.get("stop_loss_atr_mult", 1.0)),
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

    trade_log, equity_curve, metrics = portfolio_backtest(
        preds,
        cfg=risk,
        initial_capital=bt_cfg.get("initial_capital", 10_000),
    )

    bench = benchmarks(preds, cfg=risk, n_trades=metrics["n_trades"])
    metrics = {**metrics, **bench}

    save_backtest_outputs(
        trade_log=trade_log,
        metrics=metrics,
        out_dir=paths["reports_backtests"],
        equity_curve=equity_curve,
    )
    preds.to_csv(paths["reports_backtests"] / "oos_predictions.csv", index=False)

    logger.info("=== Walk-forward backtest summary ===")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
