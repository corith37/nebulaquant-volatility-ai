"""Train the two-stage calibrated model and evaluate it on the final holdout.

The model is trained on the development set (everything before the untouched
holdout) so the saved artifact reflects all usable history, while metrics are
reported on the holdout the model never saw.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import load_config, ensure_paths
from src.evaluate_model import evaluate
from src.features import feature_columns
from src.train_model import save_model, train_two_stage
from src.utils import setup_logger
from src.validation import train_holdout_split


def main() -> int:
    logger = setup_logger("nebulaquant.train_script")
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
    wf_cfg = cfg.get("walkforward", {})

    # Apply optimizer-tuned params if present.
    best_params_path = paths["reports_metrics"] / "best_params.json"
    params = {}
    if best_params_path.exists():
        params = json.loads(best_params_path.read_text(encoding="utf-8")).get("model_params", {})
        logger.info(f"Loaded tuned params: {params}")

    dev, holdout = train_holdout_split(df, holdout_frac=wf_cfg.get("holdout_frac", 0.15))
    logger.info(f"Dev rows: {len(dev)} | Holdout rows: {len(holdout)} | Features: {len(feat_cols)}")

    model = train_two_stage(
        train_df=dev,
        feat_cols=feat_cols,
        params=params,
        calibrate=model_cfg.get("calibrate", True),
        two_stage=model_cfg.get("two_stage", True),
        random_state=model_cfg.get("random_state", 42),
    )
    save_model(model, paths["models"])

    if not holdout.empty:
        evaluate(model, holdout, out_dir=paths["reports_metrics"])
    else:
        logger.warning("Empty holdout; skipping evaluation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
