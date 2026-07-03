
from qci_client import JobStatus, QciClient

from qci_exploration.config import CONFIG


CLIENT: QciClient = QciClient(url=CONFIG.qci.url, api_token=CONFIG.qci.api_key)