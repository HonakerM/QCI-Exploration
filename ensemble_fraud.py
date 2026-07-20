"""Trains and evaluates pluggable ensemble classifiers for fraud detection."""

from enum import StrEnum
import time
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import log_loss, roc_auc_score, roc_curve
import typer
from common.binary_classification.data_types import DataConfig, DataSplit, ModelResults
from common.binary_classification.data_loader import get_data_split
from common.binary_classification.ensemble_classifiers.qboost import CVQBoostConfig
from common.binary_classification.evaluation import compute_metrics, print_results
from common.binary_classification.visualization import (
    plot_metric_comparison,
    plot_roc_curves,
)
from common.binary_classification.ensemble_classifiers.base import (
    ClassifierAdapter,
    available_algorithms,
    get_adapter_cls,
)
from common.logging import get_logger, setup_logging

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_LABELS = [-1, 1]  # every registered ensemble classifier requires {-1, +1} labels
_POS_LABEL = 1
LOGGER = get_logger(__name__)


# ---------------------------------------------------------------------------
# Label validation
# ---------------------------------------------------------------------------


def validate_labels(split: DataSplit):
    """Asserts that labels are exactly {-1, +1}, as every registered ensemble
    classifier requires.

    Args:
        split (DataSplit): The data split whose test labels should be validated.
    """
    bad = split.y_test[~np.isin(split.y_test, [-1, 1])]
    if len(bad):
        raise AssertionError(
            f"Ensemble classifiers here require labels in {{-1, 1}}. Found: {bad}"
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    split: DataSplit, adapter: ClassifierAdapter, data_cfg: DataConfig
) -> ModelResults:
    """Fits `adapter` and returns fully-populated ModelResults.

    Written entirely against the ClassifierAdapter interface, so it works
    unchanged for any registered ensemble algorithm.

    Labels in split must be {-1, +1} - call validate_labels() before this
    function if you are unsure.

    Args:
        split (DataSplit): Training and test data with labels in {-1, +1}.
        adapter (ClassifierAdapter): The ensemble classifier to fit and evaluate.
        data_cfg (DataConfig): Data/run configuration (oversampling, naming, etc).

    Returns:
        ModelResults: The trained model's metrics, ROC curve data, and timing.
    """
    LOGGER.info("Submitting %s job...", adapter.config.display_name)
    t0 = time.time()
    adapter.fit(split.X_train, split.y_train)
    elapsed = time.time() - t0
    LOGGER.info("  Done in %.2fs", elapsed)

    if data_cfg.model_file:
        adapter.save(data_cfg.model_file)
        LOGGER.info("  Saved model to %s", data_cfg.model_file)

    y_train_pred = adapter.predict(split.X_train)
    y_test_pred = adapter.predict(split.X_test)

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

    y_test_probs = adapter.predict_proba(split.X_test)
    auc = float(roc_auc_score(split.y_test, y_test_probs))
    logloss = float(log_loss(split.y_test, y_test_probs))
    fpr, tpr, _ = roc_curve(split.y_test, y_test_probs)

    model_name = adapter.config.display_name
    if data_cfg.should_over_sample:
        model_name = f"{model_name} (Oversampled)"
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
# Classifier input list. Done here to allow for imports
ClassifierAlgorithms = StrEnum("ClassifierAlgorithms", available_algorithms())


def main(
    train_file: Path | None = None,
    test_file: Path | None = None,
    model_file: Path | None = None,
    dry_run: bool = False,
    save_plots: bool = False,
    results_file: Path = Path("ensemble_results.json"),
    load_results: bool = False,
    class_override: str | None = None,
    additional_feature_names: list[str] = typer.Option(
        default_factory=lambda: ["Amount", "Time"]
    ),
    no_additional_features: bool = False,
    algorithm: ClassifierAlgorithms = ClassifierAlgorithms("cvqboost"),
    weak_cls_type: str | None = None,
    should_over_sample: bool = False,
    over_sample_percentage: float = 1.0,
    model_name_override: str | None = None,
    non_fraud_sample_size: int | None = None,
    enforce_equal_samples: bool = False,
):
    """Runs fraud training and evaluation for a chosen ensemble classifier algorithm.

    Args:
        train_file (Path | None): Path to the training CSV file, if any.
        test_file (Path | None): Path to the test CSV file, if any.
        dry_run (bool): If True, load and validate data without submitting a job.
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
        algorithm (str): Which registered ensemble classifier to train, e.g.
            "cvqboost". See
            common.binary_classification.ensemble_classifiers.base.available_algorithms().
        weak_cls_type (str | None): CVQBoost-specific weak classifier override;
            ignored for other ensemble algorithms.
        should_over_sample (bool): If we should oversample the training data.
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

    adapter_cls = get_adapter_cls(algorithm)
    classifier_cfg = adapter_cls.config_cls()()
    # Algorithm-specific CLI overrides live here, gated to the ensemble
    # algorithms they apply to; each new algorithm adds its own guarded block.
    if isinstance(classifier_cfg, CVQBoostConfig) and weak_cls_type:
        classifier_cfg.weak_cls_type = weak_cls_type

    overall_start = time.time()
    LOGGER.info("ensemble fraud start (algorithm=%s)", algorithm)

    if load_results:
        if not results_file.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")
        LOGGER.info("Loading saved results from %s", results_file)
        results = ModelResults.load(results_file)
        print_results(results)

        roc_path = Path("ensemble_roc.png") if save_plots else None
        metric_path = Path("ensemble_metrics.png") if save_plots else None
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
            "--dry-run complete. Everything looks good - remove --dry-run to submit."
        )
        LOGGER.info("done (%.1fs total)", time.time() - overall_start)
        return

    # 4. Build the ensemble adapter and train
    adapter = adapter_cls(classifier_cfg)

    warning = adapter.submission_warning()
    if warning is not None:
        LOGGER.error(warning)
        LOGGER.error("TYPE `start` AND PRESS <enter> TO CONTINUE")
        LOGGER.error("ANY OTHER INPUT OR <ctrl>+c WILL EXIT")
        required_input = input()
        if required_input.lower() != "start":
            LOGGER.error("EXITING")
            return

    results = train(split, adapter, data_cfg)
    results.save(results_file)
    LOGGER.info("Saved results to %s", results_file)
    print_results(results)

    # 5. Visualize
    roc_path = Path("ensemble_roc.png") if save_plots else None
    metric_path = Path("ensemble_metrics.png") if save_plots else None
    plot_roc_curves([results], save_path=roc_path)
    plot_metric_comparison([results], save_path=metric_path)

    LOGGER.info("done (%.1fs total)", time.time() - overall_start)


if __name__ == "__main__":
    typer.run(main)
