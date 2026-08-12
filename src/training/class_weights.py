"""
class_weights.py
------------------
Computes inverse-frequency class weights for use with a weighted loss
function, to counteract severe class imbalance (e.g. 98.5% Normal beats).
"""

from collections import Counter
from typing import List

import torch


def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    """
    Computes inverse-frequency weights: w_c = N / (K * n_c)

    Args:
        labels: List of integer class labels (e.g. from dataset.labels).
        num_classes: Total number of classes (K).

    Returns:
        A tensor of shape (num_classes,) with one weight per class,
        ready to pass directly to nn.CrossEntropyLoss(weight=...).
    """
    counts = Counter(labels)
    n_total = len(labels)

    weights = torch.ones(num_classes)
    for class_idx in range(num_classes):
        n_c = counts.get(class_idx, 0)
        if n_c > 0:
            weights[class_idx] = n_total / (num_classes * n_c)
        else:
            # Class not present at all in this data -- weight is
            # meaningless (no gradient signal for it anyway), keep at
            # a neutral default rather than dividing by zero.
            weights[class_idx] = 1.0

    return weights
