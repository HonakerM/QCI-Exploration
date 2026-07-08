"""Shared metric computation used by both fraud scripts."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

from .data_types import ClassificationMetrics, ModelResults


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split: str,
    labels: list[int],
    pos_label: int,
) -> ClassificationMetrics:
    """Computes binary classification metrics for one data split.

    Args:
        y_true (np.ndarray): Ground-truth labels.
        y_pred (np.ndarray): Hard predicted labels (not probabilities).
        split (str): Name of the split these metrics belong to, e.g. "train" or
            "test", stored on the returned dataclass.
        labels (list[int]): Ordered label list, e.g. [-1, 1] or [0, 1].
        pos_label (int): The label treated as the positive (fraud) class.

    Returns:
        ClassificationMetrics: The computed precision, recall, F1, accuracy, and confusion matrix.
    """
    precision = precision_score(y_true, y_pred, labels=labels, pos_label=pos_label)
    recall = recall_score(y_true, y_pred, labels=labels, pos_label=pos_label)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return ClassificationMetrics(
        split=split,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        confusion_matrix=cm,
    )


def print_results(results: ModelResults):
    """Pretty-prints a ModelResults summary to stdout.

    Args:
        results (ModelResults): The model results to print.

    Returns:
        None.
    """
    print(results.summary())
