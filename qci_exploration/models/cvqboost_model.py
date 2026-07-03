"""
models/cvqboost_model.py
------------------------
Trains and evaluates a CVQBoost (QBoostClassifier) on QCi Dirac-3.

⚠️  QUANTUM EXECUTION WARNING
Calling train_and_evaluate() submits a job to QCi Dirac-3 quantum
hardware, which consumes paid QPU allocation time (~1 QPU second per
run, quoted at ~$0.22/run at the time of writing).

main.py guards this call with a safety flag (--enable-quantum).
Do not remove that guard.
"""

from dataclasses import dataclass
import os
import time

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score, roc_curve

from eqc_models.ml import QBoostClassifier

from qci_exploration.data_loader import validate_qboost_labels
from evaluation import compute_metrics, print_results
from qci_exploration.data_types import PreppedData, ModelResults


# ---------------------------------------------------------------------------
# CVQBoost (Quantum)
# ---------------------------------------------------------------------------

@dataclass
class CVQBoostConfig:
    """
    Hyperparameters for the QBoostClassifier running on QCi Dirac-3.

    WARNING: Submitting a job to Dirac-3 consumes paid QPU allocation.
    See README for cost details before enabling quantum execution.
    """
    relaxation_schedule: int = 1
    num_samples: int = 1
    lambda_coef: float = 0.0

    # 'sequential' is required on Windows (single-threaded weak classifiers)
    weak_cls_strategy: str = "sequential"

    # Environment variable names for QCi credentials (loaded from .env)
    token_env_var: str = "QCI_TOKEN"
    api_url_env_var: str = "QCI_API_URL"

    def as_dict(self) -> dict:
        """Return params for QBoostClassifier(**...)."""
        return {
            "relaxation_schedule": self.relaxation_schedule,
            "num_samples": self.num_samples,
            "lambda_coef": self.lambda_coef,
            "weak_cls_strategy": self.weak_cls_strategy,
        }

_MODEL_NAME = "CVQBoost"
_LABELS = [-1, 1]
_POS_LABEL = 1

def xgboost_labels(split: PreppedData) -> PreppedData:
    return split  # CVQBoost labels are already in {-1, +1}, so no conversion is needed

def check_credentials(cfg: CVQBoostConfig) -> None:
    """
    Verify that the required QCi environment variables are set.

    Raises
    ------
    EnvironmentError if either variable is missing.
    """
    missing = [
        var for var in (cfg.token_env_var, cfg.api_url_env_var)
        if var not in os.environ
    ]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {missing}\n"
            "Set them in a .env file or export them before running.\n"
            "Example .env:\n"
            "  QCI_TOKEN=your_token_here\n"
            "  QCI_API_URL=https://api.qci-prod.com"
        )
    token = os.environ[cfg.token_env_var]
    url   = os.environ[cfg.api_url_env_var]
    print(f"  {cfg.token_env_var}: {'*' * len(token)}")
    print(f"  {cfg.api_url_env_var}: {url}")


def _normalize_probs(raw_scores: list[float]) -> np.ndarray:
    """
    Map raw QBoost scores from [-1, +1] range into [0, 1] probabilities.

    Formula:  p = 0.5 * (score + 1)
    """
    return np.array([0.5 * (s + 1.0) for s in raw_scores])


def train_and_evaluate(split: PreppedData, cfg: CVQBoostConfig) -> ModelResults:
    """
    Fit a QBoostClassifier on split.X_train / y_train, then score against
    both splits.

    Parameters
    ----------
    split : PreppedData  — labels must be in {-1, +1}
    cfg   : CVQBoostConfig

    Returns
    -------
    ModelResults populated with ROC curve data and per-split metrics.
    """
    validate_qboost_labels(split)

    model = QBoostClassifier(**cfg.as_dict())

    print(f"Training {_MODEL_NAME} on QCi Dirac-3...")
    t0 = time.time()
    model.fit(split.X_train, split.y_train)
    training_time = time.time() - t0
    print(f"  Training took {training_time:.2f}s")

    y_train_pred = model.predict(split.X_train)
    y_test_pred  = model.predict(split.X_test)

    train_metrics = compute_metrics(
        split.y_train, y_train_pred, split="train",
        labels=_LABELS, pos_label=_POS_LABEL,
    )
    test_metrics = compute_metrics(
        split.y_test, y_test_pred, split="test",
        labels=_LABELS, pos_label=_POS_LABEL,
    )

    # Raw scores → probabilities for AUC / log-loss / ROC curve
    y_test_probs = _normalize_probs(model.predict_raw(split.X_test))
    auc      = roc_auc_score(split.y_test, y_test_probs)
    logloss  = log_loss(split.y_test, y_test_probs)
    fpr, tpr, _ = roc_curve(split.y_test, y_test_probs)

    results = ModelResults(
        model_name=_MODEL_NAME,
        training_time_seconds=training_time,
        fpr=fpr,
        tpr=tpr,
        auc=auc,
        log_loss=logloss,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
    )

    print_results(results)
    return results
