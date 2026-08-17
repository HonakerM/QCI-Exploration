IBM_TOKEN

from qiskit_ibm_runtime import QiskitRuntimeService


"""Shared client factory for connecting to the QCi API."""

import os
import sys

from qci_client import QciClient
from functools import lru_cache


@lru_cache(maxsize=1)
def get_client() -> QiskitRuntimeService:
    """Builds and caches a QciClient configured from the environment.

    Returns:
        QciClient: A QciClient authenticated using the QCI_API_URL and QCI_TOKEN
        environment variables (falling back to defaults if unset).
    """
    return QiskitRuntimeService(
        # For `token`, use the 44-character API_KEY you created
        # and saved from the IBM Quantum Platform Home dashboard
        token=os.getenv("IBM_TOKEN", "TEST"),
    )
