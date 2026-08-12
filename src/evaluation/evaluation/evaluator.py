"""
evaluator.py
------------
Runs a trained model over a DataLoader in inference mode, collecting
predictions and true labels for scoring.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import ClassificationReport, evaluate_predictions


class Evaluator:
    """Runs inference and produces a ClassificationReport."""

    def __init__(self, model: nn.Module, class_names: List[str], device: str = "cpu"):
        self.model = model.to(device)
        self.class_names = class_names
        self.device = device

    def collect_predictions(self, dataloader: DataLoader) -> Tuple[List[int], List[int]]:
        """
        Runs the model over every batch in the dataloader (no gradient
        updates) and collects true vs predicted labels.

        Returns:
            (y_true, y_pred) as plain Python lists.
        """
        self.model.eval()
        y_true: List[int] = []
        y_pred: List[int] = []

        with torch.no_grad():
            for signals, labels in dataloader:
                signals = signals.to(self.device)
                logits = self.model(signals)
                predictions = logits.argmax(dim=1).cpu()

                y_true.extend(labels.tolist())
                y_pred.extend(predictions.tolist())

        return y_true, y_pred

    def evaluate(self, dataloader: DataLoader) -> ClassificationReport:
        """
        Full evaluation: runs inference, then computes the classification
        report against the true labels.
        """
        y_true, y_pred = self.collect_predictions(dataloader)
        return evaluate_predictions(y_true, y_pred, self.class_names)
