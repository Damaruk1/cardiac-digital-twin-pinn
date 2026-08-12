"""
confusion_matrix_plot.py
---------------------------
Renders a confusion matrix as an annotated heatmap.
"""

from typing import List

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(
    confusion: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> plt.Figure:
    """
    Plots a confusion matrix as a heatmap with cell values annotated.

    Args:
        confusion: (num_classes, num_classes) confusion matrix, rows=true, cols=predicted.
        class_names: Names in index order, e.g. ["N", "S", "V", "F", "Q"].
        title: Plot title.
        normalize: If True, show each row as a percentage of that row's
                   total (easier to read recall directly off the diagonal).
                   If False, show raw counts.

    Returns:
        The matplotlib Figure.
    """
    if normalize:
        row_sums = confusion.sum(axis=1, keepdims=True)
        # Avoid division by zero for classes with 0 true examples in this split
        display_matrix = np.divide(
            confusion, row_sums, out=np.zeros_like(confusion, dtype=float), where=row_sums != 0
        )
        value_format = ".2f"
    else:
        display_matrix = confusion
        value_format = "d"

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(display_matrix, cmap="Blues", vmin=0)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)

    # Annotate each cell with its value, using dark/light text depending
    # on that cell's brightness for readability.
    threshold = display_matrix.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            value = display_matrix[i, j]
            color = "white" if value > threshold else "black"
            ax.text(j, i, format(value, value_format), ha="center", va="center", color=color)

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig
