"""Shared client factory for connecting to the QCi API."""

import os
import sys

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


def get_time_remaining(machine: str = "dirac") -> int | None:
    """Get the time remaining for the client for a given machine

    Args:
        machine (str, optional): The machine to check time on. Defaults to "dirac".

    Returns:
        int | None: Int if we have time remaining or not
    """
    mach: dict = get_client().get_allocations()["allocations"].get(machine, {})
    if mach.get("paid") and not mach.get("metered"):
        return sys.maxsize
    else:
        return mach.get("seconds", None)
