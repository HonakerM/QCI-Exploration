"""Create and cache the QCI API client used across the project."""

import os
import sys

from qci_client import QciClient
from functools import lru_cache


@lru_cache(maxsize=1)
def get_client() -> QciClient:
    """Create a cached QCI client from the environment configuration.

    Returns:
        QciClient: Authenticated client instance for the configured QCI endpoint.
    """
    return QciClient(
        url=os.getenv("QCI_API_URL", "https://api.qci-prod.com"),
        api_token=os.getenv("QCI_TOKEN", "TEST"),
    )


def get_time_remaining(machine: str = "dirac") -> int | None:
    """Return the remaining allocation time for a named QCI machine.

    Args:
        machine (str): Machine name to inspect. Defaults to "dirac".

    Returns:
        int | None: Remaining seconds for the machine, or None if unavailable.
    """
    mach: dict = get_client().get_allocations()["allocations"].get(machine, {})
    if mach.get("paid") and not mach.get("metered"):
        return sys.maxsize
    else:
        return mach.get("seconds", None)
