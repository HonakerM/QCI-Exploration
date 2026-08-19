"""Wrap the QBoost model behind the shared classifier adapter API."""

from dataclasses import dataclass
from pathlib import Path
import time

import joblib
import numpy as np

from common.binary_classification.base import (
    ClassifierAdapter,
    ClassifierConfig,
    register_classifier,
)
from common.qci import get_time_remaining


@dataclass
class CVQBoostConfig(ClassifierConfig):
    """Store the config for the QCi QBoost model.

    Attributes:
        algorithm_name (str): Registry key used to resolve this adapter.
        relaxation_schedule (int): Relaxation schedule index sent to the Dirac-3 solver.
        num_samples (int): Number of samples requested from the solver.
        lambda_coef (float): Regularization weight applied during training.
        weak_cls_strategy (str): Weak-classifier training strategy.
        weak_cls_type (str): Weak learner model type used in the ensemble.
        weak_cls_schedule (int): Schedule index for the weak classifiers.
        weak_cls_params (dict | None): Extra parameters forwarded to the weak learner.
    """

    algorithm_name = "cvqboost"

    relaxation_schedule: int = 2
    num_samples: int = 1
    lambda_coef: float = 0.0

    weak_cls_strategy: str = "sequential"
    weak_cls_type: str = "knn"
    weak_cls_schedule: int = 1
    weak_cls_params: dict | None = None

    def to_classifier_config(self) -> dict:
        """Convert the config into keyword arguments for the QBoost backend.

        Returns:
            dict: Keyword arguments for the QBoost model constructor.
        """
        weak_cls_params = self.weak_cls_params or {}
        return {
            "relaxation_schedule": self.relaxation_schedule,
            "num_samples": self.num_samples,
            "lambda_coef": self.lambda_coef,
            "weak_cls_strategy": self.weak_cls_strategy,
            "weak_cls_type": self.weak_cls_type,
            "weak_cls_schedule": self.weak_cls_schedule,
            "weak_cls_params": weak_cls_params,
        }

    @property
    def display_name(self) -> str:
        """Return the user-facing label for this QBoost variant.

        Returns:
            str: Display name used in reporting.
        """
        return f"CVQBoost ({self.weak_cls_type})"


# ---------------------------------------------------------------------------
# CVQBoost: adapter
# ---------------------------------------------------------------------------


@register_classifier
class CVQBoostAdapter(ClassifierAdapter[CVQBoostConfig]):
    """Wrap the QCi Dirac-3 QBoost backend in the shared adapter interface."""

    def __init__(self, config: CVQBoostConfig):
        """Create the QBoost adapter and wrap the backend model.

        Args:
            config (CVQBoostConfig): QBoost hyperparameters and solver settings.
        """
        super().__init__(config)
        # Import here so the rest of the script loads without quantum libs installed
        from eqc_models.ml import QBoostClassifier

        self.model = QBoostClassifier(**config.to_classifier_config())
        self.result: object | None = None

        og_hamiltonian = self.model.get_hamiltonian

        def timed_get_hamiltonian(*args, **kwargs):
            t0 = time.perf_counter()
            result = og_hamiltonian(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            self.model.get_hamiltonian_elapsed = elapsed
            return result

        self.model.get_hamiltonian = timed_get_hamiltonian

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Submit the training job to QCi and store the response payload.

        Args:
            X_train (np.ndarray): Training feature matrix.
            y_train (np.ndarray): Training labels.
        """
        self.result = self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard class predictions in the shared {-1, 1} format.

        Args:
            X (np.ndarray): Feature rows to score.

        Returns:
            np.ndarray: Predicted labels.
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return the positive-class probability for each row.

        Args:
            X (np.ndarray): Feature rows to score.

        Returns:
            np.ndarray: Probability of the positive class.
        """
        raw_scores = self.model.predict_raw(X)
        return np.clip(0.5 * (raw_scores + 1.0), 0.0, 1.0)

    def save(self, path: Path) -> None:
        """Persist the fitted QBoost model bundle to a joblib file.

        Args:
            path (Path): Output location for the saved model state.
        """
        bundle = {
            "h_list": self.model.h_list,
            "ind_list": self.model.ind_list,
            "params": self.model.params,
            "classes_": self.model.classes_,
            "weak_cls_type": self.config.weak_cls_type,
            "weak_cls_schedule": self.config.weak_cls_schedule,
            "relaxation_schedule": self.config.relaxation_schedule,
        }
        joblib.dump(bundle, path)

    def get_train_timing(self) -> dict[str, float] | None:
        """Return the timing breakdown captured from the QBoost training job.

        Returns:
            dict[str, float] | None: Timing values for the QBoost stages.
        """
        solve_resp = self.result

        NS_TO_S = 1e-9

        times = {
            "get_hamiltonian": float(self.model.get_hamiltonian_elapsed),
            "preprocessing": float(solve_resp.preprocessing_time) * NS_TO_S,
        }

        if len(solve_resp.run_time) == 1:
            times["run"] = float(solve_resp.run_time[0]) * NS_TO_S
        else:
            times.update(
                {
                    f"run_{i}": float(t) * NS_TO_S
                    for i, t in enumerate(solve_resp.run_time)
                }
            )

        if len(solve_resp.postprocessing_time) == 1:
            times["postprocessing"] = float(solve_resp.postprocessing_time[0]) * NS_TO_S
        else:
            times.update(
                {
                    f"postprocessing_{i}": float(t) * NS_TO_S
                    for i, t in enumerate(solve_resp.postprocessing_time)
                }
            )
        return times

    def submission_warning(self) -> str | None:
        """Return the charge warning before submitting a QCi training job.

        Returns:
            str | None: Warning message describing the remaining account time.
        """
        return (
            f"CONTINUING WILL CAUSE CHARGES TO QCI ACCOUNT! "
            f"YOU HAVE {get_time_remaining()}s REMAINING"
        )
