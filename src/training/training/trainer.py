"""
trainer.py
----------
Handles the training loop: epoch iteration, weighted loss, validation,
per-class recall tracking, and best-checkpoint saving.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class EpochMetrics:
    """Metrics recorded after one training or validation epoch."""

    loss: float
    accuracy: float
    per_class_recall: List[float] = field(default_factory=list)


class Trainer:
    """Runs the training loop for any model matching our (batch, channels,
    length) -> (batch, num_classes) interface -- works unchanged for
    both ECG1DCNN and ECGTransformer."""

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        class_weights: torch.Tensor,
        learning_rate: float = 1e-3,
        checkpoint_dir: str = "checkpoints",
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float("inf")

    def _run_epoch(self, dataloader: DataLoader, train: bool) -> EpochMetrics:
        self.model.train() if train else self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        class_correct = [0] * self.num_classes
        class_total = [0] * self.num_classes

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for signals, labels in dataloader:
                signals, labels = signals.to(self.device), labels.to(self.device)

                if train:
                    self.optimizer.zero_grad()

                logits = self.model(signals)
                loss = self.criterion(logits, labels)

                if train:
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * signals.size(0)
                predictions = logits.argmax(dim=1)
                total_correct += (predictions == labels).sum().item()
                total_samples += signals.size(0)

                for true_label, pred_label in zip(labels.tolist(), predictions.tolist()):
                    class_total[true_label] += 1
                    if true_label == pred_label:
                        class_correct[true_label] += 1

        per_class_recall = [
            (class_correct[c] / class_total[c]) if class_total[c] > 0 else float("nan")
            for c in range(self.num_classes)
        ]

        return EpochMetrics(
            loss=total_loss / total_samples,
            accuracy=total_correct / total_samples,
            per_class_recall=per_class_recall,
        )

    def train_epoch(self, dataloader: DataLoader) -> EpochMetrics:
        """Runs one training epoch (with gradient updates)."""
        return self._run_epoch(dataloader, train=True)

    def validate_epoch(self, dataloader: DataLoader) -> EpochMetrics:
        """Runs one validation epoch (no gradient updates)."""
        return self._run_epoch(dataloader, train=False)

    def save_checkpoint_if_best(self, val_metrics: EpochMetrics, model_name: str) -> bool:
        """
        Saves the model's weights if this epoch's validation loss is
        the best seen so far.

        Returns:
            True if a checkpoint was saved (i.e. this was a new best).
        """
        if val_metrics.loss < self.best_val_loss:
            self.best_val_loss = val_metrics.loss
            checkpoint_path = self.checkpoint_dir / f"{model_name}_best.pt"
            torch.save(self.model.state_dict(), checkpoint_path)
            return True
        return False

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, model_name: str, logger=None):
        """
        Runs the full training loop for the given number of epochs,
        logging progress and saving the best checkpoint.
        """
        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate_epoch(val_loader)
            improved = self.save_checkpoint_if_best(val_metrics, model_name)

            if logger is not None:
                marker = " *" if improved else ""
                logger.info(
                    f"[{model_name}] Epoch {epoch}/{epochs} -- "
                    f"train_loss={train_metrics.loss:.4f} train_acc={train_metrics.accuracy:.3f} | "
                    f"val_loss={val_metrics.loss:.4f} val_acc={val_metrics.accuracy:.3f}{marker}"
                )
