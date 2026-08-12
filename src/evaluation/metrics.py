"""
metrics.py
----------
Computes confusion matrices and per-class precision/recall/F1, using
scikit-learn's battle-tested implementations under the hood (we don't
hand-roll these formulas in production code -- too easy to introduce
subtle bugs -- but the theory above explains exactly what these
functions compute).
"""

from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


@dataclass
class ClassificationReport:
    """Structured evaluation results for one model on one dataset split."""

    confusion: np.ndarray            # shape (num_classes, num_classes)
    class_names: List[str]
    precision: np.ndarray            # shape (num_classes,)
    recall: np.ndarray
    f1: np.ndarray
    support: np.ndarray              # number of true examples per class
    macro_f1: float
    weighted_f1: float
    accuracy: float


def evaluate_predictions(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
) -> ClassificationReport:
    """
    Computes a full classification report from predicted vs true labels.

    Args:
        y_true: Ground-truth integer class labels.
        y_pred: Model-predicted integer class labels (same order/length).
        class_names: Human-readable names in index order,
                     e.g. ["N", "S", "V", "F", "Q"].

    Returns:
        A ClassificationReport with confusion matrix and all metrics.
    """
    num_classes = len(class_names)
    labels = list(range(num_classes))

    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    _, _, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )

    accuracy = float(np.mean(np.array(y_true) == np.array(y_pred)))

    return ClassificationReport(
        confusion=conf_matrix,
        class_names=class_names,
        precision=precision,
        recall=recall,
        f1=f1,
        support=support,
        macro_f1=float(macro_f1),
        weighted_f1=float(weighted_f1),
        accuracy=accuracy,
    )


def format_report(report: ClassificationReport) -> str:
    """Renders a ClassificationReport as a readable text table."""
    lines = [
        f"{'Class':<8}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Support':>10}"
    ]
    for i, name in enumerate(report.class_names):
        lines.append(
            f"{name:<8}{report.precision[i]:>10.3f}{report.recall[i]:>10.3f}"
            f"{report.f1[i]:>10.3f}{report.support[i]:>10d}"
        )
    lines.append("-" * 48)
    lines.append(f"{'Accuracy':<8}{'':>30}{report.accuracy:>10.3f}")
    lines.append(f"{'Macro F1':<8}{'':>30}{report.macro_f1:>10.3f}")
    lines.append(f"{'Weighted F1':<8}{'':>27}{report.weighted_f1:>10.3f}")
    return "\n".join(lines)
