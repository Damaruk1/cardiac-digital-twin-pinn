"""
test_trainer.py
-----------------
Phase 8 test: the classic "can it overfit a tiny dataset" sanity check.
If a training loop is correctly wired (forward, loss, backward,
optimizer step all connected properly), it should be able to drive
loss on a handful of examples down to near zero within a few epochs.
If this test fails, something in the training loop's plumbing is
broken -- this catches bugs that shape-only tests would miss.

Run with:
    pytest tests/test_trainer.py -v
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.cnn import ECG1DCNN
from src.training.class_weights import compute_class_weights
from src.training.trainer import Trainer


def test_trainer_can_overfit_tiny_dataset(tmp_path):
    torch.manual_seed(0)

    # 8 tiny, fixed examples -- a model with correctly wired gradients
    # should be able to memorize these easily within a few epochs.
    signals = torch.randn(8, 1, 64)
    labels = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])

    dataset = TensorDataset(signals, labels)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = ECG1DCNN(in_channels=1, window_length=64, num_classes=5)
    class_weights = compute_class_weights(labels.tolist(), num_classes=5)

    trainer = Trainer(
        model=model,
        num_classes=5,
        class_weights=class_weights,
        learning_rate=1e-2,
        checkpoint_dir=str(tmp_path),
    )

    first_epoch_loss = trainer.train_epoch(loader).loss
    for _ in range(30):
        final_metrics = trainer.train_epoch(loader)

    assert final_metrics.loss < first_epoch_loss, (
        "Loss did not decrease -- training loop may be broken "
        "(check gradient flow, optimizer step, or loss computation)."
    )
    assert final_metrics.loss < 0.5, (
        f"Model failed to overfit 8 tiny examples (final loss={final_metrics.loss:.3f}). "
        "A correctly wired training loop should easily memorize this."
    )


def test_checkpoint_saved_only_on_improvement(tmp_path):
    torch.manual_seed(0)
    signals = torch.randn(8, 1, 64)
    labels = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])
    dataset = TensorDataset(signals, labels)
    loader = DataLoader(dataset, batch_size=4)

    model = ECG1DCNN(in_channels=1, window_length=64, num_classes=5)
    class_weights = compute_class_weights(labels.tolist(), num_classes=5)
    trainer = Trainer(
        model=model, num_classes=5, class_weights=class_weights,
        checkpoint_dir=str(tmp_path),
    )

    val_metrics = trainer.validate_epoch(loader)
    first_save = trainer.save_checkpoint_if_best(val_metrics, "test_model")
    assert first_save is True  # first epoch is always an improvement over infinity

    checkpoint_file = tmp_path / "test_model_best.pt"
    assert checkpoint_file.exists()
