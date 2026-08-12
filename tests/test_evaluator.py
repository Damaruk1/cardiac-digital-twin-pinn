"""
test_evaluator.py
-------------------
Phase 9 tests: verify Evaluator correctly runs inference and collects
predictions, and that the confusion matrix plot renders without error.

Run with:
    pytest tests/test_evaluator.py -v
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch
from matplotlib.figure import Figure
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.evaluator import Evaluator
from src.models.cnn import ECG1DCNN
from src.visualization.confusion_matrix_plot import plot_confusion_matrix


def test_evaluator_collects_correct_number_of_predictions():
    model = ECG1DCNN(in_channels=1, window_length=64, num_classes=5)
    signals = torch.randn(20, 1, 64)
    labels = torch.randint(0, 5, (20,))
    loader = DataLoader(TensorDataset(signals, labels), batch_size=4)

    evaluator = Evaluator(model, class_names=["N", "S", "V", "F", "Q"])
    y_true, y_pred = evaluator.collect_predictions(loader)

    assert len(y_true) == 20
    assert len(y_pred) == 20
    assert all(0 <= p < 5 for p in y_pred)


def test_evaluator_produces_valid_classification_report():
    model = ECG1DCNN(in_channels=1, window_length=64, num_classes=5)
    signals = torch.randn(20, 1, 64)
    labels = torch.randint(0, 5, (20,))
    loader = DataLoader(TensorDataset(signals, labels), batch_size=4)

    evaluator = Evaluator(model, class_names=["N", "S", "V", "F", "Q"])
    report = evaluator.evaluate(loader)

    assert report.confusion.shape == (5, 5)
    assert 0.0 <= report.accuracy <= 1.0
    assert 0.0 <= report.macro_f1 <= 1.0


def test_confusion_matrix_plot_returns_figure():
    confusion = np.array([[8, 2], [1, 9]])
    fig = plot_confusion_matrix(confusion, class_names=["N", "V"])

    assert isinstance(fig, Figure)


def test_confusion_matrix_plot_handles_zero_row():
    """A class with zero true examples in this split (row sums to 0)
    shouldn't cause a division-by-zero crash when normalizing."""
    confusion = np.array([[10, 0], [0, 0]])
    fig = plot_confusion_matrix(confusion, class_names=["N", "V"], normalize=True)

    assert isinstance(fig, Figure)
