"""Run binary-classification experiments defined by YAML config files."""

import copy
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    auc,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from common.binary_classification.ensemble_classifiers import *  # noqa: F403
from common.binary_classification.classifiers import *  # noqa: F403
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
    RepetitionConfig,
    TimingInfo,
)
from common.binary_classification.evaluation import compute_metrics, print_results
from common.binary_classification.visualization import (
    plot_metric_comparison,
    plot_pr_curves,
    plot_roc_curves,
    plot_timing_comparison,
)
from common.data_files import (
    convert_path_to_result_run,
    convert_path_to_results,
    load_data_dict,
    load_yaml,
)
from common.logging import get_logger


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
    random_state: int | None = None,
) -> ModelResults:
    """Train an adapter and package the resulting metrics and timing.

    Args:
        split (DataSplit): Prepared training and test arrays for the dataset.
        adapter (ClassifierAdapter): Model adapter to fit and score.
        data_cfg (DataConfig): Data settings used to name the run and save artifacts.
        data_prep_seconds (float): Time spent preparing the dataset before fitting.

    Returns:
        ModelResults: Metrics, ROC/PR arrays, and timing for the trained model.
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
    auc_pr = (
        float(auc(pr_recall, pr_precision))
        if pr_precision.size and pr_recall.size
        else 0.0
    )

    # Log diagnostic info if PR curve is empty
    if pr_precision.size == 0 or pr_recall.size == 0:
        LOGGER.warning(
            "PR arrays empty: y_test unique=%s, y_test_probs unique head=%s",
            np.unique(split.y_test),
            np.unique(y_test_probs)[:5],
        )

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
        random_state=random_state,
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


def _load_repetition_config(data: dict) -> RepetitionConfig | None:
    """Load the optional repetition block from a parsed YAML document."""
    repetition_data = data.get("repetition")
    if repetition_data is None:
        return None
    return load_data_dict(repetition_data, RepetitionConfig)


def _generate_run_random_states(repetition: RepetitionConfig) -> list[int]:
    """Generate deterministic per-run seeds from the repetition seed."""
    rng = np.random.default_rng(repetition.random_state)
    return [
        int(value)
        for value in rng.integers(
            0,
            np.iinfo(np.uint32).max,
            size=repetition.num,
            dtype=np.uint32,
        )
    ]


def test_file(
    test_file: Path,
    dry_run: bool = False,
    display_plots: bool = True,
    save_plots: bool = False,
    suppress_warnings: bool = False,
):
    """Run a single experiment from a YAML file.

    Args:
        test_file (Path): YAML file that defines the data and model configuration.
        dry_run (bool): Validate the config and data pipeline without training.
        display_plots (bool): Display training diagnostics and comparison plots.
        save_plots (bool): Save comparison plots to disk when enabled.
        suppress_warnings (bool): Suppress warnings during the run.
    """
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
    repetition_cfg = _load_repetition_config(data)
    adapter_cls = get_adapter_cls(algorithm)
    classifier_cfg = load_data_dict(classifier_config_raw, adapter_cls.config_cls())

    if dry_run:
        LOGGER.info("--dry-run: credentials OK, loading data and prepping split...")

    run_random_states = (
        _generate_run_random_states(repetition_cfg)
        if repetition_cfg is not None
        else [data_cfg.random_state]
    )

    results = []
    for run_random_state in run_random_states:
        overall_start = time.time()
        LOGGER.info("ensemble fraud start (algorithm=%s)", algorithm)

        data_cfg_run = copy.deepcopy(data_cfg)
        classifier_cfg_run = copy.deepcopy(classifier_cfg)

        if repetition_cfg is not None:
            data_cfg_run.random_state = run_random_state
            if hasattr(classifier_cfg_run, "random_state"):
                setattr(classifier_cfg_run, "random_state", run_random_state)

        # Skip if we've already done it
        if repetition_cfg is None:
            results_file = convert_path_to_results(test_file)
        else:
            results_file = convert_path_to_result_run(test_file, run_random_state)

        if results_file.exists():
            LOGGER.info("Skipping existing results at %s", results_file)
            continue

        # 2. Load & engineer features
        data_prep_start = time.perf_counter()
        split = get_data_split(data_cfg_run)
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
            LOGGER.info(
                "--dry-run complete. Everything looks good - remove --dry-run to continue."
            )
            LOGGER.info("done (%.1fs total)", time.time() - overall_start)
            return

        # 4. Build the adapter and train
        adapter = adapter_cls(classifier_cfg_run)

        warning = adapter.submission_warning()
        if not suppress_warnings and warning is not None:
            LOGGER.error(warning)
            LOGGER.error("TYPE `start` AND PRESS <enter> TO CONTINUE")
            LOGGER.error("ANY OTHER INPUT OR <ctrl>+c WILL EXIT")
            required_input = input()
            if required_input.lower() != "start":
                LOGGER.error("EXITING")
                return

        result = train(
            split,
            adapter,
            data_cfg_run,
            data_prep_seconds=data_prep_seconds,
            random_state=data_cfg_run.random_state,
        )

        if repetition_cfg is None:
            results_file = convert_path_to_results(test_file)
        else:
            results_file = convert_path_to_result_run(test_file, run_random_state)

        if results_file.exists():
            LOGGER.info("Skipping existing results at %s", results_file)
            continue

        results_file.parent.mkdir(parents=True, exist_ok=True)
        result.save(results_file)
        LOGGER.info("Saved results to %s", results_file)
        print_results(result)
        results.append(result)

    # 5. Visualize
    if display_plots:
        plot_roc_curves(results)
        plot_pr_curves(results)
        plot_metric_comparison(results)
        plot_timing_comparison(results)
    if save_plots:
        roc_path = Path("ensemble_roc.png") if save_plots else None
        metric_path = Path("ensemble_metrics.png") if save_plots else None
        timing_path = Path("ensemble_timing.png") if save_plots else None
        plot_roc_curves(results, save_path=roc_path)
        plot_metric_comparison(results, save_path=metric_path)
        plot_timing_comparison(results, save_path=timing_path)
