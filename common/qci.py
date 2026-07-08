"""Shared client factory for connecting to the QCi API."""

import os

from qci_client import QciClient
from functools import lru_cache


@lru_cache(maxsize=1)
def get_client() -> QciClient:
    """Builds and caches a QciClient configured from the environment.

    Returns:
        QciClient: A QciClient authenticated using the QCI_API_URL and QCI_TOKEN
        environment variables (falling back to defaults if unset).
    """
    return QciClient(
        url=os.getenv("QCI_API_URL", "https://api.qci-prod.com"),
        api_token=os.getenv("QCI_TOKEN", "1vxYZxvDRomg5rKwRuIn_JYH"),
    )
