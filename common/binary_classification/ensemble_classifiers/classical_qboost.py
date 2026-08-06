from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from common.binary_classification.base import (
    ClassifierAdapter,
    ClassifierConfig,
    register_classifier,
)

import time

from scipy.optimize import minimize

from eqc_models.ml.classifierqboost import QBoostClassifier


class ClassicalQBoostClassifier(QBoostClassifier):
    """QBoost with the Dirac-3 solve step replaced by scipy.optimize (L-BFGS-B).

    `get_hamiltonian()` and `set_model()` are inherited unchanged, so
    `fit()` (also inherited, unmodified) builds the exact same bounded QUBO
    relaxation `(J, C, upper_bound)` it always would. The only difference is
    `solve()`: where QBoostClassifier submits that problem to Dirac-3, this
    class minimizes it directly with scipy's L-BFGS-B, which natively
    supports the box constraints `0 <= x <= upper_bound` the problem is
    already defined with.

    Parameters
    ----------
    max_iter : int, default 1000
        Maximum L-BFGS-B iterations.

    tol : float, default 1e-8
        Function-value convergence tolerance (`ftol`) for L-BFGS-B.

    **kwargs
        Forwarded to QBoostClassifier.__init__ (weak_cls_type,
        weak_cls_schedule, weak_cls_strategy, lambda_coef, etc.). Dirac-3-
        specific arguments (solver_access, api_url, api_token, ip_addr,
        port, relaxation_schedule, num_samples) are accepted for signature
        compatibility but unused by solve().

    Examples
    -----------

    >>> from sklearn import datasets
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> from sklearn.model_selection import train_test_split
    >>> iris = datasets.load_iris()
    >>> X = iris.data
    >>> y = iris.target
    >>> scaler = MinMaxScaler()
    >>> X = scaler.fit_transform(X)
    >>> for i in range(len(y)):
    ...     if y[i] == 0:
    ...         y[i] = -1
    ...     elif y[i] == 2:
    ...         y[i] = 1
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X,
    ...     y,
    ...     test_size=0.2,
    ...     random_state=42,
    ... )
    >>> from eqc_models.ml.classical_qboost import ClassicalQBoostClassifier
    >>> obj = ClassicalQBoostClassifier(weak_cls_strategy="sequential")
    >>> from contextlib import redirect_stdout
    >>> import io
    >>> f = io.StringIO()
    >>> with redirect_stdout(f):
    ...    obj.fit(X_train, y_train)
    ...    y_train_prd = obj.predict(X_train)
    ...    y_test_prd = obj.predict(X_test)

    """

    def __init__(
        self,
        max_iter=1000,
        tol=1e-8,
        method: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_iter = max_iter
        self.tol = tol
        self.method = method
        self.minimize_elapsed = None

    def solve(self):  # type: ignore[override]
        """Classically solves the bounded QUBO relaxation via L-BFGS-B.

        Reads `self._J`, `self._C`, and `self.upper_bound` exactly as
        `set_model()` (inherited, unmodified) left them, and minimizes
        `x'Jx + C'x` subject to `0 <= x <= upper_bound` — the same problem
        `ClassifierBase.solve()` would otherwise ship to Dirac-3.

        Returns
        -------
        sol : ndarray of shape (n_classifiers,)
            The optimized per-weak-classifier weights.
        response : dict
            A small JSON-serializable summary of the classical solve,
            standing in for the response QBoostClassifier.fit() would
            otherwise get back from Dirac-3.
        """
        J = self._J
        C = np.asarray(self._C, dtype=np.float64).reshape(-1)
        J.shape[0]

        bounds = [(0.0, float(ub)) for ub in np.asarray(self.upper_bound).reshape(-1)]

        def objective(x):
            # x'Jx + C'x, matching the QUBO relaxation Dirac-3 would anneal.
            Jx = J @ x
            value = x @ Jx + C @ x
            grad = 2.0 * Jx + C
            return value, grad

        # Start at the midpoint of each variable's box constraint.
        x0 = np.asarray(self.upper_bound, dtype=np.float64).reshape(-1) / 2.0

        t0 = time.perf_counter()
        result = minimize(
            objective,
            x0,
            method=self.method,
            jac=True,
            bounds=bounds,
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )
        elapsed = time.perf_counter() - t0

        sol = result.x

        self.minimize_elapsed = elapsed

        response = {
            "solver": "scipy.optimize.minimize (L-BFGS-B)",
            "success": bool(result.success),
            "message": str(result.message),
            "n_iter": int(result.nit),
            "final_objective": float(result.fun),
            "solve_time_seconds": elapsed,
        }

        return sol, response


@dataclass
class ClassicalQBoostConfig(ClassifierConfig):
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

    algorithm_name = "classical_qboost"
    optimization_method: str = "L-BFGS-B"

    relaxation_schedule: int = 2
    num_samples: int = 1
    lambda_coef: float = 0.0

    weak_cls_strategy: str = "sequential"
    weak_cls_type: str = "knn"
    weak_cls_schedule: int = 1
    include_smu_params: bool = True

    def to_classifier_config(self) -> dict:
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
            # Does notthing for classical
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
        return f"Classical QBoost ({self.weak_cls_type} - {self.optimization_method})"


# ---------------------------------------------------------------------------
# CVQBoost: adapter
# ---------------------------------------------------------------------------


@register_classifier
class ClassicalQBoostAdapter(ClassifierAdapter[ClassicalQBoostConfig]):
    """Adapts eqc_models' QBoostClassifier (QCi Dirac-3) to ClassifierAdapter."""

    def __init__(self, config: ClassicalQBoostConfig):
        super().__init__(config)
        self.model = ClassicalQBoostClassifier(**config.to_classifier_config())

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
        self.model.fit(X_train, y_train)

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
        if self.model.minimize_elapsed is None:
            return None
        return {
            "solve": float(self.model.minimize_elapsed),
            "get_hamiltonian": float(self.model.get_hamiltonian_elapsed),
        }

    def submission_warning(self) -> str | None:
        """Warns that training will incur charges on the QCi account."""
        return None
