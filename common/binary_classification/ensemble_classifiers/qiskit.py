from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from qiskit.primitives import StatevectorSampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.minimum_eigensolvers import QAOA, NumPyMinimumEigensolver
from qiskit_optimization.optimizers import COBYLA
from qiskit_aer.primitives import SamplerV2

from common.binary_classification.base import (
    ClassifierAdapter,
    ClassifierConfig,
    register_classifier,
)

import time

from qiskit_optimization import QuadraticProgram

from eqc_models.ml.classifierqboost import QBoostClassifier


class QiskitQBoostClassifier(QBoostClassifier):
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
        classical: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.minimize_elapsed = None
        self.classical = classical

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

        qp = QuadraticProgram()

        n = J.shape[0]

        for i in range(n):
            qp.binary_var(name=f"x_{i}")

        qp.minimize(
            quadratic=J,
            linear=C,
        )

        
        if self.classical:
            mes = NumPyMinimumEigensolver()
        else:
            mes = QAOA(
                        sampler=StatevectorSampler(seed=123),
                        optimizer=COBYLA(),
                        reps=2,
                    )
            
        optimizer = MinimumEigenOptimizer(mes)

        result = optimizer.solve(qp)

        x = np.asarray(result.x, dtype=int)

        response = {
            "solver": f"qiskit qubo",
            "success": True,
            "n_iter": 1,
            "solve_time_seconds": 1,
        }

        return x, response


@dataclass
class QiskitQBoostConfig(ClassifierConfig):
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

    algorithm_name = "qiskit"
    classical: bool = False

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
            "classical": self.classical,
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
        return f"Qiskit QBoost"


# ---------------------------------------------------------------------------
# CVQBoost: adapter
# ---------------------------------------------------------------------------


@register_classifier
class QiskitQBoostAdapter(ClassifierAdapter[QiskitQBoostConfig]):
    """Adapts eqc_models' QBoostClassifier (QCi Dirac-3) to ClassifierAdapter."""

    def __init__(self, config: QiskitQBoostConfig):
        super().__init__(config)
        self.model = QiskitQBoostClassifier(**config.to_classifier_config())

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
