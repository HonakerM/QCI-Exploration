from dataclasses import dataclass, field
import os

from dotenv import load_dotenv

# Load dotenv
load_dotenv()

@dataclass
class QCIConfig:
    url: str = os.getenv("QCI_URL", "https://api.qci-prod.com")
    api_key: str |None = os.getenv("QCI_API_KEY", None)

@dataclass
class Config:
    qci: QCIConfig = field(default_factory=QCIConfig)

CONFIG: Config = Config()