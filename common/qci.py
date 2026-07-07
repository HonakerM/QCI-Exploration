import os

from qci_client import QciClient
from functools import lru_cache


@lru_cache(maxsize=1)
def get_client() -> QciClient:
    return QciClient(
        url=os.getenv("QCI_API_URL", "https://api.qci-prod.com"),
        api_token=os.getenv("QCI_TOKEN", "1vxYZxvDRomg5rKwRuIn_JYH"),
    )
