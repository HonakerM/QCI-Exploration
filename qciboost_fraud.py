"""Trains and evaluates a QBoostClassifier on QCi Dirac-3 for fraud detection."""

import time
from pathlib import Path
import joblib
import numpy as np
from dataclasses import dataclass
from dotenv import load_dotenv
from sklearn.metrics import log_loss, roc_auc_score, roc_curve
import typer
from common.binary_classification.data_types import DataConfig, DataSplit, ModelResults
from common.binary_classification.data_loader import get_data_split
from common.binary_classification.evaluation import compute_metrics, print_results
from common.binary_classification.visualization import (
    plot_metric_comparison,
    plot_roc_curves,
)
from common.logging import get_logger, setup_logging
from common.qci import get_time_remaining

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MODEL_NAME = "CVQBoost"
_LABELS = [-1, 1]  # QBoostClassifier requires {-1, +1}
_POS_LABEL = 1
LOGGER = get_logger(__name__)

# ---------------------------------------------------------------------------
# CVQBoost-specific config
# ---------------------------------------------------------------------------


@dataclass
class CVQBoostConfig:
    """Hyperparameters for the QBoostClassifier running on QCi Dirac-3.

    Attributes:
        relaxation_schedule (int): Relaxation schedule index used by the Dirac-3
            solver.
        num_samples (int): Number of samples requested from the solver.
        lambda_coef (float): Regularization coefficient applied during training.
        weak_cls_strategy (str): Strategy used to fit weak classifiers;
            'sequential' is required on Windows (single-threaded weak
            classifiers).
    """

    relaxation_schedule: int = 2
    num_samples: int = 1
    lambda_coef: float = 0.0

    weak_cls_strategy: str = "sequential"
    weak_cls_type: str = "knn"
    weak_cls_schedule: int = 1
    include_smu_params: bool = True

    def to_qboost_config(self) -> dict:
        """Converts the config into keyword arguments for QBoostClassifier.

        Returns:
            dict: A dictionary of hyperparameters suitable for QBoostClassifier(**kwargs).
        """
        weak_cls_params = {}
        if self.include_smu_params:
            if self.weak_cls_type == "knn":
                weak_cls_params = {
                    "weights": "uniform",
                    "n_neighbors": 15,
                    "metric": "minkowski",
                }
            elif self.weak_cls_type == "lda":
                weak_cls_params = {"solver": "lsqr", "shrinkage": "auto"}
            elif self.weak_cls_type == "lg":
                weak_cls_params = {
                    "penalty": "l2",
                    "solver": "lbfgs",
                    "C": 10,
                }
            elif self.weak_cls_type == "xgb":
                weak_cls_params = {
                    "n_estimators": 100,
                    "max_depth": 3,
                    "learning_rate": 0.1,
                    "subsample": 1.0,
                    "colsample_bytree": 0.8,
                    "min_child_weight": 1,
                    "reg_lambda": 1.0,
                    "reg_alpha": 0.0,
                }
        return {
            "relaxation_schedule": self.relaxation_schedule,
            "num_samples": self.num_samples,
            "lambda_coef": self.lambda_coef,
            "weak_cls_strategy": self.weak_cls_strategy,
            "weak_cls_type": self.weak_cls_type,
            "weak_cls_schedule": self.weak_cls_schedule,
            "weak_cls_params": weak_cls_params,
        }


# ---------------------------------------------------------------------------
# Label validation
# ---------------------------------------------------------------------------


def validate_labels(split: DataSplit):
    """Asserts that labels are exactly {-1, +1} as QBoostClassifier requires.

    Args:
        split (DataSplit): The data split whose test labels should be validated.
    """
    bad = split.y_test[~np.isin(split.y_test, [-1, 1])]
    if len(bad):
        raise AssertionError(f"QBoost requires labels in {{-1, 1}}.  Found: {bad}")


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def _save_model(model: "QBoostClassifier", cfg: CVQBoostConfig, path: Path) -> None:  # noqa
    bundle = {
        "h_list": model.h_list,
        "ind_list": model.ind_list,
        "params": model.params,
        "classes_": model.classes_,
        "weak_cls_type": cfg.weak_cls_type,
        "weak_cls_schedule": cfg.weak_cls_schedule,
        "relaxation_schedule": cfg.relaxation_schedule,
    }
    joblib.dump(bundle, path)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(split: DataSplit, cfg: CVQBoostConfig, data_cfg: DataConfig) -> ModelResults:
    """Fits QBoostClassifier on Dirac-3 and returns fully-populated ModelResults.

    Labels in split must be {-1, +1} - call validate_labels() before this
    function if you are unsure.

    Args:
        split (DataSplit): Training and test data with labels in {-1, +1}.
        cfg (CVQBoostConfig): Hyperparameters for the QBoostClassifier.

    Returns:
        ModelResults: The trained model's metrics, ROC curve data, and timing.
    """
    # Import here so the rest of the script loads without quantum libs installed
    from eqc_models.ml import QBoostClassifier

    model = QBoostClassifier(**cfg.to_qboost_config())

    LOGGER.info("Submitting %s job to QCi Dirac-3...", _MODEL_NAME)
    t0 = time.time()
    model.fit(split.X_train, split.y_train)
    elapsed = time.time() - t0
    LOGGER.info("  Done in %.2fs", elapsed)

    if data_cfg.model_file:
        _save_model(model, cfg, data_cfg.model_file)
        LOGGER.info("  Saved model to %s", data_cfg.model_file)

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

    # Raw scores in [-1, +1]  ->  probabilities in [0, 1]
    raw_scores = model.predict_raw(split.X_test)
    y_test_probs = np.clip(0.5 * (raw_scores + 1.0), 0.0, 1.0)
    auc = float(roc_auc_score(split.y_test, y_test_probs))
    logloss = float(log_loss(split.y_test, y_test_probs))
    fpr, tpr, _ = roc_curve(split.y_test, y_test_probs)

    model_name = f"{_MODEL_NAME} ({cfg.weak_cls_type} {'Oversampled' if data_cfg.should_over_sample else ''})"
    if data_cfg.model_name_override is not None:
        model_name = data_cfg.model_name_override

    return ModelResults(
        model_name=model_name,
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
    model_file: Path | None = None,
    dry_run: bool = False,
    save_plots: bool = False,
    results_file: Path = Path("qciboost_results.json"),
    load_results: bool = False,
    class_override: str | None = None,
    additional_feature_names: list[str] = typer.Option(
        default_factory=lambda: ["Amount", "Time"]
    ),
    no_additional_features: bool = False,
    weak_cls_type: str | None = None,
    should_over_sample: bool = False,
    over_sample_percentage: float = 1.0,
    model_name_override: str | None = None,
    non_fraud_sample_size: int | None = None,
    enforce_equal_samples: bool = False,
):
    """Runs QCi Dirac-3 QBoost fraud training and evaluation.

    Args:
        train_file (Path | None): Path to the training CSV file, if any.
        test_file (Path | None): Path to the test CSV file, if any.
        dry_run (bool): If True, load and validate data without submitting a job
            to Dirac-3.
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
        should_over_sample (bool): If we should oversample the training data
    """
    load_dotenv()  # pull QCI_TOKEN / QCI_API_URL from .env if present
    setup_logging()

    data_cfg = DataConfig(
        train_file=Path(train_file) if train_file is not None else None,
        test_file=Path(test_file) if test_file is not None else None,
        additional_feature_names=additional_feature_names,
        should_over_sample=should_over_sample,
        over_sample_percentage=over_sample_percentage,
        model_name_override=model_name_override,
        enforce_equal_samples=enforce_equal_samples,
        model_file=model_file,
    )
    if non_fraud_sample_size:
        data_cfg.non_fraud_sample_size = non_fraud_sample_size
    if no_additional_features:
        data_cfg.additional_feature_names = []

    if class_override:
        data_cfg.class_name = class_override

    cvq_cfg = CVQBoostConfig()
    if weak_cls_type:
        cvq_cfg.weak_cls_type = weak_cls_type

    overall_start = time.time()
    LOGGER.info("qciboost_fraud start")

    if load_results:
        if not results_file.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")
        LOGGER.info("Loading saved results from %s", results_file)
        results = ModelResults.load(results_file)
        print_results(results)

        roc_path = Path("qciboost_roc.png") if save_plots else None
        metric_path = Path("qciboost_metrics.png") if save_plots else None
        plot_roc_curves([results], save_path=roc_path)
        plot_metric_comparison([results], save_path=metric_path)

        LOGGER.info("done (%.1fs total)", time.time() - overall_start)
        return

    if dry_run:
        LOGGER.info("--dry-run: credentials OK, loading data and prepping split...")

    # 2. Load & engineer features
    split = get_data_split(data_cfg)

    validate_labels(split)
    LOGGER.info(
        "  %s train rows | %s test rows | %s features",
        split.n_train,
        split.n_test,
        split.n_features,
    )

    if dry_run:
        LOGGER.info(
            "--dry-run complete. Everything looks good - remove --dry-run to submit to Dirac-3."
        )
        LOGGER.info("done (%.1fs total)", time.time() - overall_start)
        return

    # 4. Train on Dirac-3
    LOGGER.error(
        f"CONTINUING WILL CAUSE CHARGES TO QCI ACCOUNT! YOU HAVE {get_time_remaining()}s REMAINING"
    )
    LOGGER.error("TYPE `start` AND PRESS <enter> TO CONTINUE")
    LOGGER.error("ANY OTHER INPUT OR <ctrl>+c WILL EXIT")
    required_input = input()
    if required_input.lower() != "start":
        LOGGER.error("EXITING")
        return

    results = train(split, cvq_cfg, data_cfg)
    results.save(results_file)
    LOGGER.info("Saved results to %s", results_file)
    print_results(results)

    # 5. Visualize
    roc_path = Path("qciboost_roc.png") if save_plots else None
    metric_path = Path("qciboost_metrics.png") if save_plots else None
    plot_roc_curves([results], save_path=roc_path)
    plot_metric_comparison([results], save_path=metric_path)

    LOGGER.info("done (%.1fs total)", time.time() - overall_start)


if __name__ == "__main__":
    typer.run(main)
