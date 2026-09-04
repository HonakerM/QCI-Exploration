# Import to force registration of all classifiers in this package.
from .xgboost import XGBoostAdapter, XGBoostConfig
from .lg import LogisticRegressionAdapter, LogisticRegressionConfig
from .lda import LDAAdapter, LDAConfig
from .knn import KNNAdapter, KNNConfig
from .cad import CADAdapter, CADConfig

__all__ = [
    "CADAdapter",
    "CADConfig",
    "XGBoostAdapter",
    "XGBoostConfig",
    "LogisticRegressionAdapter",
    "LogisticRegressionConfig",
    "LDAAdapter",
    "LDAConfig",
    "KNNAdapter",
    "KNNConfig",
]
