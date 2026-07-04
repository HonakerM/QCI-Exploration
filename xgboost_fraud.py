"""
xgboost_fraud.py
----------------
Credit card fraud detection using XGBoost.

Loads the Kaggle Playground Series S3E4 dataset, balances it, trains an
XGBClassifier, evaluates it, and plots the ROC curve. Runs entirely on
CPU with no external API calls.

Usage
-----
    python xgboost_fraud.py
    python xgboost_fraud.py --save-plots
    python xgboost_fraud.py --results-file xgboost_results.json
    python xgboost_fraud.py --load-results --results-file xgboost_results.json
    python xgboost_fraud.py --train-file /data/train.csv --test-file /data/test.csv

Expected results
----------------
    AUC      ~0.887
    Training ~1–2 seconds
"""

import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import typer
from dataclasses import dataclass
from sklearn.metrics import log_loss, roc_auc_score, roc_curve
from xgboost import XGBClassifier

from common import (
    DataConfig,
    DataSplit,
    ModelResults,
    compute_metrics,
    load_data,
    plot_metric_comparison,
    plot_roc_curves,
    prep_data,
    print_results,
)
from common.logging import get_logger, setup_logging


LOGGER = get_logger(__name__)


# ---------------------------------------------------------------------------
# XGBoost-specific config
# ---------------------------------------------------------------------------


@dataclass
class XGBoostConfig:
    """Hyperparameters for the XGBoost classifier."""

    n_estimators: int = 3093
    min_child_weight: int = 96
    max_depth: int = 12
    learning_rate: float = 0.07516
    subsample: float = 0.95
    colsample_bytree: float = 0.95
    reg_lambda: float = 1.50
    reg_alpha: float = 1.50
    gamma: float = 1.50
    max_bin: int = 512
    random_state: int = 228
    objective: str = "binary:logistic"
    tree_method: str = "auto"
    eval_metric: str = "auc"

    def as_dict(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "min_child_weight": self.min_child_weight,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "reg_alpha": self.reg_alpha,
            "gamma": self.gamma,
            "max_bin": self.max_bin,
            "random_state": self.random_state,
            "objective": self.objective,
            "tree_method": self.tree_method,
            "eval_metric": self.eval_metric,
        }


# ---------------------------------------------------------------------------
# Label remapping
# ---------------------------------------------------------------------------


def remap_labels(split: DataSplit) -> DataSplit:
    """
    Return a copy of *split* with labels re-scaled from {-1, +1} → {0, 1}.

    XGBoost's 'binary:logistic' objective requires non-negative labels.
    Applies: label = 0.5 * (label + 1). The original DataSplit is unchanged.
    """
    remap = np.vectorize(lambda v: 0.5 * (v + 1))
    return DataSplit(
        X_train=split.X_train,
        y_train=remap(split.y_train),
        X_test=split.X_test,
        y_test=remap(split.y_test),
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MODEL_NAME = "XGBoost"
_LABELS = [0, 1]  # XGBoost uses non-negative integer labels
_POS_LABEL = 1


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(split: DataSplit, cfg: XGBoostConfig) -> ModelResults:
    """Fit XGBClassifier and return fully-populated ModelResults."""
    model = XGBClassifier(**cfg.as_dict())
    LOGGER.info("Training %s...", _MODEL_NAME)
    t0 = time.time()
    model.fit(split.X_train, split.y_train)
    elapsed = time.time() - t0
    LOGGER.info("Done in %.2fs", elapsed)

    y_train_pred = model.predict(split.X_train)
    y_test_pred = model.predict(split.X_test)

    train_metrics = compute_metrics(
        split.y_train,
        y_train_pred,
        split="train",
        labels=_LABELS,
        pos_label=_POS_LABEL,
    )
    test_metrics = compute_metrics(
        split.y_test,
        y_test_pred,
        split="test",
        labels=_LABELS,
        pos_label=_POS_LABEL,
    )

    y_test_probs = model.predict_proba(split.X_test)[:, 1]
    auc = float(roc_auc_score(split.y_test, y_test_probs))
    logloss = float(log_loss(split.y_test, y_test_probs))
    fpr, tpr, _ = roc_curve(split.y_test, y_test_probs)

    return ModelResults(
        model_name=_MODEL_NAME,
        training_time_seconds=elapsed,
        fpr=fpr,
        tpr=tpr,
        auc=auc,
        log_loss=logloss,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(
    train_file: Path = typer.Option(Path("train.csv"), help="Path to Kaggle train.csv (default: train.csv)"),
    test_file: Path = typer.Option(Path("test.csv"), help="Path to Kaggle test.csv (default: test.csv)"),
    save_plots: bool = typer.Option(False, "--save-plots", help="Save ROC and metric plots to PNG files instead of showing them"),
    results_file: Path = typer.Option(Path("xgboost_results.json"), "--results-file", help="Path to save or load serialized model results"),
    load_results: bool = typer.Option(False, "--load-results", help="Load existing results from --results-file instead of retraining"),
) -> None:
    """Run XGBoost fraud training and evaluation."""
    setup_logging()

    data_cfg = DataConfig(
        train_file=Path(train_file),
        test_file=Path(test_file),
    )
    xgb_cfg = XGBoostConfig()

    overall_start = time.time()
    LOGGER.info("xgboost_fraud start")

    # 1. Load & engineer features
    df = load_data(data_cfg)

    # 2. Balance + split -> labels {-1, +1}
    split = prep_data(df, data_cfg)
    LOGGER.info("  %s train rows | %s test rows | %s features", split.n_train, split.n_test, split.n_features)

    # 3. Load saved results or train a fresh model
    if load_results:
        if not results_file.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")
        LOGGER.info("Loading saved results from %s", results_file)
        results = ModelResults.load(results_file)
    else:
        xgb_split = remap_labels(split)
        results = train(xgb_split, xgb_cfg)
        results.save(results_file)
        LOGGER.info("Saved results to %s", results_file)

    print_results(results)

    # 4. Visualize
    roc_path = Path("xgboost_roc.png") if save_plots else None
    metric_path = Path("xgboost_metrics.png") if save_plots else None
    plot_roc_curves([results], save_path=roc_path)
    plot_metric_comparison([results], save_path=metric_path)

    LOGGER.info("done (%.1fs total)", time.time() - overall_start)


if __name__ == "__main__":
    typer.run(main)
