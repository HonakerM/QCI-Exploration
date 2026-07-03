"""
models/xgboost_model.py
-----------------------
Trains and evaluates an XGBoost classifier.
"""

from dataclasses import dataclass
import time

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score, roc_curve
from xgboost import XGBClassifier

from evaluation import compute_metrics, print_results
from qci_exploration.data_types import PreppedData, ModelResults



# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

@dataclass
class XGBoostConfig:
    """Hyperparameters for the XGBoost classifier."""
    n_estimators: int = 3093
    min_child_weight: int = 96
    max_depth: int = 12
    learning_rate: float = 0.07516
    subsample: float = 0.95
    colsample_bytree: float = 0.95
    reg_lambda: float = 1.50
    reg_alpha: float = 1.50
    gamma: float = 1.50
    max_bin: int = 512
    random_state: int = 228
    objective: str = "binary:logistic"
    tree_method: str = "auto"
    eval_metric: str = "auc"

    def as_dict(self) -> dict:
        """Return params as a plain dict for XGBClassifier(**...)."""
        return {
            "n_estimators": self.n_estimators,
            "min_child_weight": self.min_child_weight,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "reg_alpha": self.reg_alpha,
            "gamma": self.gamma,
            "max_bin": self.max_bin,
            "random_state": self.random_state,
            "objective": self.objective,
            "tree_method": self.tree_method,
            "eval_metric": self.eval_metric,
        }

def xgboost_labels(split: PreppedData) -> PreppedData:
    """
    Return a new PreppedData with labels re-scaled from {-1, +1} → {0, 1}.

    XGBoost's 'binary:logistic' objective requires non-negative integer
    labels, so we apply:  label_xgb = 0.5 * (label + 1)

    The original PreppedData is not modified.
    """
    def remap(y: np.ndarray) -> np.ndarray:
        return np.array([0.5 * (v + 1) for v in y])

    return PreppedData(
        X_train=split.X_train,
        y_train=remap(split.y_train),
        X_test=split.X_test,
        y_test=remap(split.y_test),
    )

_MODEL_NAME = "XGBoost"
_LABELS = [0, 1]
_POS_LABEL = 1


def train_and_evaluate(split: PreppedData, cfg: XGBoostConfig) -> ModelResults:
    """
    Fit an XGBClassifier on split.X_train / y_train, then score against
    both splits.

    Parameters
    ----------
    split : DataSplit  — labels must be in {0, 1}
    cfg   : XGBoostConfig

    Returns
    -------
    ModelResults populated with ROC curve data and per-split metrics.
    """
    split = xgboost_labels(split)
    model = XGBClassifier(**cfg.as_dict())

    print(f"Training {_MODEL_NAME}...")
    t0 = time.time()
    model.fit(split.X_train, split.y_train)
    training_time = time.time() - t0
    print(f"  Training took {training_time:.2f}s")

    y_train_pred = model.predict(split.X_train)
    y_test_pred  = model.predict(split.X_test)

    train_metrics = compute_metrics(
        split.y_train, y_train_pred, split="train",
        labels=_LABELS, pos_label=_POS_LABEL,
    )
    test_metrics = compute_metrics(
        split.y_test, y_test_pred, split="test",
        labels=_LABELS, pos_label=_POS_LABEL,
    )

    # Probabilities for AUC / log-loss / ROC curve
    y_test_probs = model.predict_proba(split.X_test)[:, 1]
    auc      = roc_auc_score(split.y_test, y_test_probs)
    logloss  = log_loss(split.y_test, y_test_probs)
    fpr, tpr, _ = roc_curve(split.y_test, y_test_probs)

    results = ModelResults(
        model_name=_MODEL_NAME,
        training_time_seconds=training_time,
        fpr=fpr,
        tpr=tpr,
        auc=auc,
        log_loss=logloss,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
    )

    print_results(results)
    return results
