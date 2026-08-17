"""Trains and evaluates pluggable binary classifiers for fraud detection.

This runner unifies the previous `ensemble_fraud.py` and the newer
`classifier_fraud.py` behavior: it accepts YAML test files that specify an
`algorithm`, `data`, and `classifier` section, builds the requested adapter
from the shared registry, runs training/evaluation, and saves results to the
corresponding `results/...` JSON path.
"""

from enum import StrEnum
import time
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import auc, log_loss, precision_recall_curve, roc_auc_score, roc_curve
import typer

from common.binary_classification.ensemble_classifiers import *
from common.binary_classification.classifiers import *
from common.binary_classification.base import (
    ClassifierAdapter,
    available_algorithms,
    get_adapter_cls,
)
from common.binary_classification.data_loader import get_data_split
from common.binary_classification.data_types import (
    DataConfig,
    DataSplit,
    ModelResults,
    TimingInfo,
)
from common.binary_classification.evaluation import compute_metrics, print_results
from common.binary_classification.visualization import (
    plot_metric_comparison,
    plot_pr_curves,
    plot_roc_curves,
    plot_timing_comparison,
)
from common.data_files import convert_path_to_results, load_data_dict, load_yaml
from common.logging import get_logger, setup_logging


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_LABELS = [-1, 1]
_POS_LABEL = 1
LOGGER = get_logger(__name__)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    split: DataSplit,
    adapter: ClassifierAdapter,
    data_cfg: DataConfig,
    data_prep_seconds: float = 0.0,
) -> ModelResults:
    """Fits `adapter` and returns fully-populated ModelResults.

    The runner always works with labels in {-1, +1}; adapters that require a
    different label encoding (e.g. XGBoost) are expected to perform any
    internal remapping themselves.
    """
    LOGGER.info("Submitting %s job...", adapter.config.display_name)
    t0 = time.perf_counter()
    adapter.fit(split.X_train, split.y_train)
    fit_seconds = time.perf_counter() - t0
    LOGGER.info("  Fit done in %.2fs", fit_seconds)

    if data_cfg.model_file:
        adapter.save(data_cfg.model_file)
        LOGGER.info("  Saved model to %s", data_cfg.model_file)

    t0 = time.perf_counter()
    y_train_pred = adapter.predict(split.X_train)
    y_test_pred = adapter.predict(split.X_test)
    y_test_probs = adapter.predict_proba(split.X_test)
    predict_seconds = time.perf_counter() - t0
    LOGGER.info("  Predict done in %.2fs", predict_seconds)

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

    auc_score = float(roc_auc_score(split.y_test, y_test_probs))
    logloss = float(log_loss(split.y_test, y_test_probs))
    fpr, tpr, _ = roc_curve(split.y_test, y_test_probs)

    # Precision-Recall: compute and guard against degenerate cases
    pr_precision, pr_recall, _ = precision_recall_curve(split.y_test, y_test_probs)
    auc_pr = float(auc(pr_recall, pr_precision)) if pr_precision.size and pr_recall.size else 0.0

    # Log diagnostic info if PR curve is empty
    if pr_precision.size == 0 or pr_recall.size == 0:
        LOGGER.warning("PR arrays empty: y_test_bin unique=%s, y_test_probs unique head=%s", np.unique(y_test_bin), np.unique(y_test_probs)[:5])


    adapter_timing = adapter.get_train_timing()
    if adapter_timing is not None:
        fit_seconds = None

    model_name = adapter.config.display_name
    if data_cfg.should_over_sample:
        model_name = f"{model_name} (Oversampled)"
    if data_cfg.model_name_override is not None:
        model_name = data_cfg.model_name_override

    return ModelResults(
        model_name=model_name,
        timing=TimingInfo(
            data_prep=data_prep_seconds,
            fit=fit_seconds,
            predict=predict_seconds,
            adapter=adapter_timing or {},
        ),
        fpr=fpr,
        tpr=tpr,
        auc=auc_score,
        log_loss=logloss,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        pr_precision=pr_precision,
        pr_recall=pr_recall,
        auc_pr=auc_pr,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

APP = typer.Typer()

# Classifier input list. Done here to allow for imports
ClassifierAlgorithms = StrEnum("ClassifierAlgorithms", available_algorithms())
SUPPRESS_WARNINGS = False


@APP.command("test-folder")
def test_folder(
    test_folder: Path,
    dry_run: bool = False,
    display_plots: bool = False,
    suppress_warnings: bool = False,
    rerun: bool = False
):
    """Runs fraud training and evaluation for a chosen classifier algorithm.

    Args:
        test_folder (Path): Path to a folder containing one or more YAML test files.
        dry_run (bool): If True, only loads the test files and validates them without
            running any training. Defaults to False.
    """
    load_dotenv()  # pull QCI_TOKEN / QCI_API_URL from .env if present
    setup_logging()
    if suppress_warnings:
        global SUPPRESS_WARNINGS
        SUPPRESS_WARNINGS = True

    for file in test_folder.glob("**/*.yaml"):
        LOGGER.info("Running test file: %s", file)
        results = convert_path_to_results(file)
        if results.exists() and not rerun:
            LOGGER.warning(
                "Results file %s already exists. Skipping test file %s.",
                results,
                file,
            )
            continue
        test_file(file, dry_run=dry_run, display_plots=display_plots, _from_folder=True)


@APP.command("test-file")
def test_file(
    test_file: Path,
    dry_run: bool = False,
    display_plots: bool = True,
    save_plots: bool = False,
    _from_folder: bool = typer.Option(default=False, expose_value=False, hidden=True),
):
    """Runs fraud training and evaluation for a chosen classifier algorithm.

    Args:
        test_file (Path): Path to a YAML test file containing algorithm, data, and
            classifier configuration.
        dry_run (bool): If True, only loads the test file and validates it without
            running any training. Defaults to False.
        display_plots (bool): If True, displays ROC and metric comparison plots.
            Defaults to False.
        save_plots (bool): If True, saves ROC and metric comparison plots to the
            current working directory. Defaults to False.
    """
    if not _from_folder:
        load_dotenv()  # pull QCI_TOKEN / QCI_API_URL from .env if present
        setup_logging()

    data = load_yaml(test_file)

    algorithm = data.get("algorithm")
    if not algorithm:
        raise ValueError(
            "Missing 'algorithm' key in test file. Must be one of: "
            f"{available_algorithms()}"
        )

    data_config_raw = data.get("data")
    if not data_config_raw:
        raise ValueError("Missing 'data' key in test file. Must be a dict.")

    classifier_config_raw = data.get("classifier")
    if classifier_config_raw is None:
        raise ValueError("Missing 'classifier' key in test file. Must be a dict.")

    data_cfg = load_data_dict(data_config_raw, DataConfig)
    adapter_cls = get_adapter_cls(algorithm)
    classifier_cfg = load_data_dict(classifier_config_raw, adapter_cls.config_cls())

    overall_start = time.time()
    LOGGER.info("ensemble fraud start (algorithm=%s)", algorithm)

    if dry_run:
        LOGGER.info("--dry-run: credentials OK, loading data and prepping split...")

    # 2. Load & engineer features
    data_prep_start = time.perf_counter()
    split = get_data_split(data_cfg)
    data_prep_seconds = time.perf_counter() - data_prep_start

    # validate that labels coming out of the loader are {-1, +1}
    bad = split.y_test[~np.isin(split.y_test, [-1, 1])]
    if len(bad):
        raise AssertionError(
            f"Classifiers here expect labels in {{-1, 1}}. Found: {bad}"
        )

    LOGGER.info(
        "  %s train rows | %s test rows | %s features",
        split.n_train,
        split.n_test,
        split.n_features,
    )

    if dry_run:
        LOGGER.info("--dry-run complete. Everything looks good - remove --dry-run to submit.")
        LOGGER.info("done (%.1fs total)", time.time() - overall_start)
        return

    # 4. Build the adapter and train
    adapter = adapter_cls(classifier_cfg)

    warning = adapter.submission_warning()
    if not SUPPRESS_WARNINGS and warning is not None:
        LOGGER.error(warning)
        LOGGER.error("TYPE `start` AND PRESS <enter> TO CONTINUE")
        LOGGER.error("ANY OTHER INPUT OR <ctrl>+c WILL EXIT")
        required_input = input()
        if required_input.lower() != "start":
            LOGGER.error("EXITING")
            return

    results = train(split, adapter, data_cfg, data_prep_seconds=data_prep_seconds)

    results_file = convert_path_to_results(test_file)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results.save(results_file)
    LOGGER.info("Saved results to %s", results_file)
    print_results(results)

    # 5. Visualize
    if display_plots:
        plot_roc_curves([results])
        plot_pr_curves([results])
        plot_metric_comparison([results])
        plot_timing_comparison([results])
    if save_plots:
        roc_path = Path("ensemble_roc.png") if save_plots else None
        metric_path = Path("ensemble_metrics.png") if save_plots else None
        timing_path = Path("ensemble_timing.png") if save_plots else None
        plot_roc_curves([results], save_path=roc_path)
        plot_metric_comparison([results], save_path=metric_path)
        plot_timing_comparison([results], save_path=timing_path)

    LOGGER.info("done (%.1fs total)", time.time() - overall_start)


if __name__ == "__main__":
    APP()
