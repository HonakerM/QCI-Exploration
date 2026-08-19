"""Wrap the classical QBoost solver behind the shared classifier adapter API."""

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
        """Solve the bounded QUBO relaxation with L-BFGS-B.

        Returns:
            tuple[np.ndarray, dict]: Optimized weights and the solve metadata.
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
            "solver": f"scipy.optimize.minimize ({self.method})",
            "success": bool(result.success),
            "message": str(result.message),
            "n_iter": int(result.nit),
            "final_objective": float(result.fun),
            "solve_time_seconds": elapsed,
        }

        return sol, response


@dataclass
class ClassicalQBoostConfig(ClassifierConfig):
    """Store the configuration for the classical QBoost solver.

    Attributes:
        algorithm_name (str): Registry key used to resolve this adapter.
        optimization_method (str): SciPy optimizer method used for the solve step.
        relaxation_schedule (int): Relaxation schedule index sent to the solver.
        num_samples (int): Number of samples requested from the backend.
        lambda_coef (float): Regularization coefficient used in the objective.
        weak_cls_strategy (str): Strategy for fitting the weak learners.
        weak_cls_type (str): Type of weak learner used by the ensemble.
        weak_cls_schedule (int): Schedule index for weak learners.
        weak_cls_params (dict | None): Extra parameters passed to the weak learner.
    """

    algorithm_name = "classical_qboost"
    optimization_method: str = "L-BFGS-B"

    relaxation_schedule: int = 2
    num_samples: int = 1
    lambda_coef: float = 0.0

    weak_cls_strategy: str = "sequential"
    weak_cls_type: str = "knn"
    weak_cls_schedule: int = 1
    weak_cls_params: dict | None = None

    def to_classifier_config(self) -> dict:
        """Convert the config into keyword arguments for the backend model.

        Returns:
            dict: Keyword arguments for the QBoost constructor.
        """
        weak_cls_params = self.weak_cls_params or {}
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
        """Return the user-facing name for this classical solver variant.

        Returns:
            str: Display label for the model.
        """
        return f"Classical QBoost ({self.weak_cls_type} - {self.optimization_method})"


# ---------------------------------------------------------------------------
# CVQBoost: adapter
# ---------------------------------------------------------------------------


@register_classifier
class ClassicalQBoostAdapter(ClassifierAdapter[ClassicalQBoostConfig]):
    """Wrap the classical QBoost implementation in the shared adapter interface."""

    def __init__(self, config: ClassicalQBoostConfig):
        """Create the classical QBoost adapter.

        Args:
            config (ClassicalQBoostConfig): Classical QBoost configuration.
        """
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
        """Train the classical QBoost model in place.

        Args:
            X_train (np.ndarray): Training feature matrix.
            y_train (np.ndarray): Training labels.
        """
        self.model.fit(X_train, y_train)

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
        """Persist the fitted model state to a joblib bundle.

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
        """Return the timing recorded for the classical solve step.

        Returns:
            dict[str, float] | None: Timing values captured during optimization.
        """
        if self.model.minimize_elapsed is None:
            return None
        return {
            "solve": float(self.model.minimize_elapsed),
            "get_hamiltonian": float(self.model.get_hamiltonian_elapsed),
        }

    def submission_warning(self) -> str | None:
        """Warns that training will incur charges on the QCi account."""
        return None
