# ---------------------------------------------------------------------------
# CVQBoost: config
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# CVQBoost: config
# ---------------------------------------------------------------------------


@dataclass
class CVQBoostConfig(ClassifierConfig):
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

    algorithm_name = "cvqboost"

    relaxation_schedule: int = 2
    num_samples: int = 1
    lambda_coef: float = 0.0

    weak_cls_strategy: str = "sequential"
    weak_cls_type: str = "knn"
    weak_cls_schedule: int = 1
    weak_cls_params: dict | None = None

    def to_classifier_config(self) -> dict:
        """Converts the config into keyword arguments for QBoostClassifier.

        Returns:
            dict: A dictionary of hyperparameters suitable for QBoostClassifier(**kwargs).
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
        """Short name identifying this classifier variant."""
        return f"CVQBoost ({self.weak_cls_type})"


# ---------------------------------------------------------------------------
# CVQBoost: adapter
# ---------------------------------------------------------------------------


@register_classifier
class CVQBoostAdapter(ClassifierAdapter[CVQBoostConfig]):
    """Adapts eqc_models' QBoostClassifier (QCi Dirac-3) to ClassifierAdapter."""

    def __init__(self, config: CVQBoostConfig):
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
        """Submits a training job to QCi Dirac-3 and fits in place."""
        self.result = self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns hard {-1, +1} predictions."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Maps QBoost's raw scores in [-1, +1] to probabilities in [0, 1]."""
        raw_scores = self.model.predict_raw(X)
        return np.clip(0.5 * (raw_scores + 1.0), 0.0, 1.0)

    def save(self, path: Path) -> None:
        """Persists the fitted model's boosting state as a joblib bundle."""
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
        """Returns adapter-specific timing measurements."""
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
        """Warns that training will incur charges on the QCi account."""
        return (
            f"CONTINUING WILL CAUSE CHARGES TO QCI ACCOUNT! "
            f"YOU HAVE {get_time_remaining()}s REMAINING"
        )
