import time
from pathlib import Path
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

LOGGER = get_logger(__name__)

# ---------------------------------------------------------------------------
# CVQBoost-specific config
# ---------------------------------------------------------------------------


@dataclass
class CVQBoostConfig:
    """
    Hyperparameters for the QBoostClassifier running on QCi Dirac-3.

    ⚠️  Each run consumes paid QPU allocation (~1 QPU second, ~$0.22/run).
    """

    relaxation_schedule: int = 1
    num_samples: int = 1
    lambda_coef: float = 0.0

    # 'sequential' is required on Windows (single-threaded weak classifiers)
    weak_cls_strategy: str = "sequential"

    def to_qboost_config(self) -> dict:
        return {
            "relaxation_schedule": self.relaxation_schedule,
            "num_samples": self.num_samples,
            "lambda_coef": self.lambda_coef,
            "weak_cls_strategy": self.weak_cls_strategy,
        }


# ---------------------------------------------------------------------------
# Label validation
# ---------------------------------------------------------------------------


def validate_labels(split: DataSplit) -> None:
    """
    Assert that labels are exactly {-1, +1} as QBoostClassifier requires.

    Raises
    ------
    AssertionError if any unexpected values are found.
    """
    bad = split.y_test[~np.isin(split.y_test, [-1, 1])]
    if len(bad):
        raise AssertionError(f"QBoost requires labels in {{-1, 1}}.  Found: {bad}")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MODEL_NAME = "CVQBoost"
_LABELS = [-1, 1]  # QBoostClassifier requires {-1, +1}
_POS_LABEL = 1


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(split: DataSplit, cfg: CVQBoostConfig) -> ModelResults:
    """
    Fit QBoostClassifier on Dirac-3 and return fully-populated ModelResults.

    Labels in *split* must be {-1, +1} - call validate_labels()
    before this function if you are unsure.
    """
    # Import here so the rest of the script loads without quantum libs installed
    from eqc_models.ml import QBoostClassifier

    model = QBoostClassifier(**cfg.to_qboost_config())

    LOGGER.info("Submitting %s job to QCi Dirac-3...", _MODEL_NAME)
    t0 = time.time()
    model.fit(split.X_train, split.y_train)
    elapsed = time.time() - t0
    LOGGER.info("  Done in %.2fs", elapsed)

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
    y_test_probs = np.array([min(0.5 * (s + 1.0), 1.0) for s in raw_scores])
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
    dry_run: bool = False,
    save_plots: bool = False,
    results_file: Path = Path("qciboost_results.json"),
    load_results: bool = False,
    class_override: str | None = None,
    additional_feature_names: list[str] = typer.Option(
        default_factory=lambda: ["Amount", "Time"]
    ),
    no_additional_features: bool = False,
) -> None:
    load_dotenv()  # pull QCI_TOKEN / QCI_API_URL from .env if present
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

    cvq_cfg = CVQBoostConfig()

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
    results = train(split, cvq_cfg)
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
