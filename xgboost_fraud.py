"""Trains and evaluates an XGBoost classifier for fraud detection."""

import time
from pathlib import Path
import numpy as np
import typer
from dataclasses import dataclass
from sklearn.metrics import log_loss, roc_auc_score, roc_curve
from xgboost import XGBClassifier
from common.binary_classification.data_types import (
    DataConfig,
    DataSplit,
    ModelResults,
)
from common.binary_classification.data_loader import get_data_split
from common.binary_classification.evaluation import compute_metrics, print_results
from common.binary_classification.visualization import (
    plot_metric_comparison,
    plot_roc_curves,
)
from common.logging import get_logger, setup_logging


LOGGER = get_logger(__name__)


# ---------------------------------------------------------------------------
# XGBoost-specific config
# ---------------------------------------------------------------------------


@dataclass
class XGBoostConfig:
    """Hyperparameters for the XGBoost classifier.

    Attributes:
        n_estimators: Number of boosting rounds.
        min_child_weight: Minimum sum of instance weight needed in a child.
        max_depth: Maximum tree depth.
        learning_rate: Boosting learning rate (eta).
        subsample: Fraction of rows sampled per tree.
        colsample_bytree: Fraction of columns sampled per tree.
        reg_lambda: L2 regularization term on weights.
        reg_alpha: L1 regularization term on weights.
        gamma: Minimum loss reduction required to make a further split.
        max_bin: Maximum number of bins used for histogram-based splits.
        random_state: Seed used for reproducibility.
        objective: XGBoost learning objective.
        tree_method: XGBoost tree construction algorithm.
        eval_metric: Metric used for evaluation during training.
    """

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
        """Converts the config into keyword arguments for XGBClassifier.

        Returns:
            A dictionary of hyperparameters suitable for XGBClassifier(**kwargs).
        """
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
    """Returns a copy of split with labels re-scaled from {-1, +1} to {0, 1}.

    XGBoost's 'binary:logistic' objective requires non-negative labels.
    Applies: label = 0.5 * (label + 1). The original DataSplit is unchanged.

    Args:
        split: The data split whose labels should be remapped.

    Returns:
        A new DataSplit with y_train and y_test remapped to {0, 1}.
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
    """Fits XGBClassifier and returns fully-populated ModelResults.

    Args:
        split: Training and test data with labels in {0, 1}.
        cfg: Hyperparameters for the XGBoost classifier.

    Returns:
        ModelResults: The trained model's metrics, ROC curve data, and timing.
    """
    model = XGBClassifier(**cfg.as_dict(), enable_categorical=True)
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
    train_file: Path | None = None,
    test_file: Path | None = None,
    save_plots: bool = False,
    results_file: Path = Path("xgboost_results.json"),
    load_results: bool = False,
    class_override: str | None = None,
    additional_feature_names: list[str] = typer.Option(
        default_factory=lambda: ["Amount", "Time"]
    ),
    no_additional_features: bool = False,
):
    """Runs XGBoost fraud training and evaluation.

    Args:
        train_file (Path | None): Path to the training CSV file, if any.
        test_file (Path | None): Path to the test CSV file, if any.
        save_plots (bool): If True, save ROC and metric plots as PNGs instead of
            showing them.
        results_file (Path): Path to save (or load) the ModelResults JSON.
        load_results (bool): If True, load previously saved results instead of
            training a new model.
        class_override (str): If provided, overrides the default label column
            name.
        additional_feature_names (list[str]): Extra raw feature columns to include
            alongside the engineered features.
        no_additional_features (bool): If True, ignore additional_feature_names
            and train on engineered features only.
    """
    setup_logging()

    data_cfg = DataConfig(
        train_file=Path(train_file) if train_file is not None else None,
        test_file=Path(test_file) if test_file is not None else None,
        additional_feature_names=additional_feature_names,
    )
    if no_additional_features:
        data_cfg.additional_feature_names = []

    if class_override:
        data_cfg.class_name = class_override

    xgb_cfg = XGBoostConfig()

    overall_start = time.time()
    LOGGER.info("xgboost_fraud start")

    split = get_data_split(data_cfg)
    LOGGER.info(
        "  %s train rows | %s test rows | %s features",
        split.n_train,
        split.n_test,
        split.n_features,
    )

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
