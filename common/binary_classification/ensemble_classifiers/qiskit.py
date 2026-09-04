"""Wrap the Qiskit QBoost implementation behind the shared adapter interface."""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from qiskit import generate_preset_pass_manager
from qiskit.primitives import StatevectorSampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.minimum_eigensolvers import QAOA, NumPyMinimumEigensolver
from qiskit_optimization.optimizers import COBYLA
from qiskit_ibm_runtime import QiskitRuntimeService, Session

from common.qiskit import get_client as get_qiskit_client
from common.binary_classification.base import (
    ClassifierAdapter,
    ClassifierConfig,
    register_classifier,
)
from qiskit_aer.primitives import EstimatorV2

import time

from qiskit_optimization import QuadraticProgram

from eqc_models.ml.classifierqboost import QBoostClassifier

import numpy as np
from scipy.optimize import minimize
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer.primitives import EstimatorV2, SamplerV2
from qiskit_ibm_runtime import EstimatorV2 as RuntimeEstimator
from qiskit_ibm_runtime import SamplerV2 as RuntimeSampler

 
def qubo_to_ising_paulis(J: np.ndarray, C: np.ndarray):
    """Convert a QUBO  minimize x^T J x + C^T x, x in {0,1}^n  into an
    Ising Hamiltonian using the substitution x_i = (1 - z_i) / 2.
 
    Assumes J is symmetric (symmetrize first if it isn't -- see note below).
 
    Returns
    -------
    pauli_list : list[tuple[str, list[int], float]]
        Sparse pauli terms, in the same ("ZZ"/"Z", qubits, coeff) format
        used by `SparsePauliOp.from_sparse_list` / `build_max_cut_paulis`.
    offset : float
        Constant energy shift. The QUBO objective for a bitstring x equals
        <H>_x + offset, where <H>_x is the Hamiltonian expectation value
        on the corresponding computational basis state.
    """
    n = J.shape[0]
    L = np.diag(J).copy() + C  # combined linear coefficients (diag(J) + C)
 
    h = -L / 2.0
    offset = np.sum(L) / 2.0
    zz_terms = {}
 
    for i in range(n):
        for j in range(i + 1, n):
            k_ij = J[i, j] + J[j, i]  # = 2*J[i, j] for symmetric J
            if np.isclose(k_ij, 0.0):
                continue
            h[i] -= k_ij / 4.0
            h[j] -= k_ij / 4.0
            offset += k_ij / 4.0
            zz_terms[(i, j)] = k_ij / 4.0
 
    pauli_list = [("ZZ", [i, j], coeff) for (i, j), coeff in zz_terms.items()]
    pauli_list += [("Z", [i], h[i]) for i in range(n) if not np.isclose(h[i], 0.0)]
 
    return pauli_list, offset


class QiskitQBoostClassifier(QBoostClassifier):
    def __init__(
        self,
        classical: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.minimize_elapsed = None
        self.classical = classical

    def solve(self):  # type: ignore[override]
        """Solve the QUBO with a Qiskit optimizer or a classical fallback.

        Returns:
            tuple[np.ndarray, dict]: Optimized variables and metadata about the solve.
        """
        J = self._J
        C = np.asarray(self._C, dtype=np.float64).reshape(-1)
        n = J.shape[0]

        pauli_list, offset = qubo_to_ising_paulis(J, C)
        cost_hamiltonian = SparsePauliOp.from_sparse_list(pauli_list, n)
        print("Cost Hamiltonian:", cost_hamiltonian)
        print("Offset:", offset)

        circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=2)
        circuit.measure_all()
        
        initial_gamma = np.pi
        initial_beta = np.pi / 2
        init_params = [initial_beta, initial_beta, initial_gamma, initial_gamma]
        
        
        def cost_func_estimator(params, ansatz, hamiltonian, estimator):
            pub = (ansatz, hamiltonian, params)
            job = estimator.run([pub])
            cost = job.result()[0].data.evs
            objective_func_vals.append(cost)
            return cost
        
        
        objective_func_vals = []
        aer_estimator = EstimatorV2()

        
        result = minimize(
            cost_func_estimator,
            init_params,
            args=(circuit.decompose(reps=10), cost_hamiltonian, aer_estimator),
            method="COBYLA",
            tol=1e-2,
        )
        print(result)

        response = {
            "solver": "qiskit qubo",
            "success": True,

            "n_iter": 1,
            "solve_time_seconds": 1,
        }

        return None, response


@dataclass
class QiskitQBoostConfig(ClassifierConfig):
    """Store the configuration for the Qiskit-backed QBoost solver.

    Attributes:
        algorithm_name (str): Registry key used to resolve this adapter.
        classical (bool): Whether to use the classical NumPy eigensolver instead of QAOA.
        relaxation_schedule (int): Relaxation schedule index sent to the backend.
        num_samples (int): Number of samples requested from the solver.
        lambda_coef (float): Regularization coefficient applied during training.
        weak_cls_strategy (str): Strategy used to fit the weak learners.
        weak_cls_type (str): Type of weak learner used by the ensemble.
        weak_cls_schedule (int): Schedule index for the weak learners.
        weak_cls_params (dict | None): Extra parameters passed to the weak learner.
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
        """Convert the config into keyword arguments for the backend model.

        Returns:
            dict: Keyword arguments for the QBoost constructor.
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
        """Return the user-facing name for this Qiskit-backed variant.

        Returns:
            str: Display label for the model.
        """
        return "Qiskit QBoost"


# ---------------------------------------------------------------------------
# CVQBoost: adapter
# ---------------------------------------------------------------------------


@register_classifier
class QiskitQBoostAdapter(ClassifierAdapter[QiskitQBoostConfig]):
    """Wrap the Qiskit QBoost implementation in the shared adapter interface."""

    def __init__(self, config: QiskitQBoostConfig):
        """Create the adapter and its Qiskit-backed model.

        Args:
            config (QiskitQBoostConfig): Qiskit QBoost configuration.
        """
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
        """Train the Qiskit-backed model in place.

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
        """Return the timing recorded for the Qiskit optimization step.

        Returns:
            dict[str, float] | None: Timing values recorded by the adapter.
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
