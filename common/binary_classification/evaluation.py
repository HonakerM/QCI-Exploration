"""Compute classification metrics for each model output."""

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
    """Compute the summary metrics for one split of the label predictions.

    Args:
        y_true (np.ndarray): Ground-truth labels.
        y_pred (np.ndarray): Predicted labels for the same rows.
        split (str): Split name, such as "train" or "test".
        labels (list[int]): Ordered class labels to score.
        pos_label (int): Positive-class label used for precision and recall.

    Returns:
        ClassificationMetrics: Precision, recall, F1, accuracy, and confusion matrix.
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
    """Print a readable summary of a model's training and evaluation output.

    Args:
        results (ModelResults): Result bundle to display.
    """
    print(results.summary())
