"""Create and cache the IBM Qiskit runtime client used by the project."""

import os
from functools import lru_cache

from qiskit_ibm_runtime import QiskitRuntimeService


@lru_cache(maxsize=1)
def get_client() -> QiskitRuntimeService:
    """Create a cached IBM Quantum client from the configured token.

    Returns:
        QiskitRuntimeService: Runtime service configured for the project token.
    """
    return QiskitRuntimeService(
        token=os.getenv("IBM_TOKEN", "TEST"),
    )
