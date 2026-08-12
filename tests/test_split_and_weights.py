"""
test_split_and_weights.py
----------------------------
Phase 8 tests: verify stratified splitting preserves class balance
(or falls back gracefully), and class weight calculation is correct.

Run with:
    pytest tests/test_split_and_weights.py -v
"""

import numpy as np
import torch

from src.data.beat_dataset import BeatDataset
from src.data.dataset_loader import BeatAnnotations, ECGRecord
from src.data.split import stratified_split
from src.training.class_weights import compute_class_weights


def _make_balanced_dataset(n_per_class=20) -> BeatDataset:
    """Builds a small dataset with enough examples per class to stratify."""
    n_samples = 20000
    signal = np.random.randn(n_samples, 1).astype(np.float64)
    record = ECGRecord(
        record_name="fake", signal=signal, sampling_rate=360,
        lead_names=["MLII"], duration_sec=n_samples / 360,
    )

    classes = ["N", "S", "V", "F", "Q"]
    sample_indices = []
    aami_classes = []
    positions = np.linspace(200, n_samples - 200, n_per_class * len(classes)).astype(int)
    pos_iter = iter(positions)
    for cls in classes:
        for _ in range(n_per_class):
            sample_indices.append(next(pos_iter))
            aami_classes.append(cls)

    annotations = BeatAnnotations(
        record_name="fake",
        sample_indices=np.array(sample_indices),
        raw_symbols=aami_classes,
        aami_classes=aami_classes,
    )
    return BeatDataset(record, annotations, window_before=90, window_after=90)


def test_stratified_split_preserves_class_presence():
    dataset = _make_balanced_dataset(n_per_class=20)
    splits = stratified_split(dataset, train_frac=0.70, val_frac=0.15)

    train_labels = set(dataset.labels[i] for i in splits.train.indices)
    val_labels = set(dataset.labels[i] for i in splits.val.indices)
    test_labels = set(dataset.labels[i] for i in splits.test.indices)

    # With 20 examples per class, all 5 classes should appear in all 3 splits
    assert train_labels == {0, 1, 2, 3, 4}
    assert val_labels == {0, 1, 2, 3, 4}
    assert test_labels == {0, 1, 2, 3, 4}


def test_stratified_split_sizes_roughly_match_fractions():
    dataset = _make_balanced_dataset(n_per_class=20)  # 100 total
    splits = stratified_split(dataset, train_frac=0.70, val_frac=0.15)

    total = len(dataset)
    assert abs(len(splits.train) / total - 0.70) < 0.05
    assert abs(len(splits.val) / total - 0.15) < 0.05
    assert abs(len(splits.test) / total - 0.15) < 0.05


def test_split_has_no_overlapping_indices():
    dataset = _make_balanced_dataset(n_per_class=20)
    splits = stratified_split(dataset, train_frac=0.70, val_frac=0.15)

    train_set = set(splits.train.indices)
    val_set = set(splits.val.indices)
    test_set = set(splits.test.indices)

    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)


def test_split_falls_back_gracefully_with_rare_class():
    """A class with only 1 example should not crash the split -- it
    should trigger the random-split fallback instead."""
    n_samples = 5000
    signal = np.random.randn(n_samples, 1).astype(np.float64)
    record = ECGRecord(
        record_name="fake", signal=signal, sampling_rate=360,
        lead_names=["MLII"], duration_sec=n_samples / 360,
    )
    # 50 Normal beats, but only 1 rare V beat
    sample_indices = list(np.linspace(200, n_samples - 200, 50).astype(int)) + [2500]
    aami_classes = ["N"] * 50 + ["V"]
    annotations = BeatAnnotations(
        record_name="fake", sample_indices=np.array(sample_indices),
        raw_symbols=aami_classes, aami_classes=aami_classes,
    )
    dataset = BeatDataset(record, annotations, window_before=90, window_after=90)

    # Must not raise, even though stratification is impossible for class V
    splits = stratified_split(dataset, train_frac=0.70, val_frac=0.15)
    assert len(splits.train) + len(splits.val) + len(splits.test) == len(dataset)


def test_compute_class_weights_inverse_frequency():
    # 80 of class 0, 20 of class 1 -- class 1 should get a higher weight
    labels = [0] * 80 + [1] * 20
    weights = compute_class_weights(labels, num_classes=2)

    assert weights[1] > weights[0]
    # w_0 = 100 / (2 * 80) = 0.625 ; w_1 = 100 / (2 * 20) = 2.5
    assert torch.isclose(weights[0], torch.tensor(0.625), atol=1e-3)
    assert torch.isclose(weights[1], torch.tensor(2.5), atol=1e-3)


def test_compute_class_weights_handles_missing_class():
    """A class with zero examples shouldn't cause division by zero."""
    labels = [0] * 50  # class 1 never appears
    weights = compute_class_weights(labels, num_classes=2)

    assert weights[1] == 1.0  # neutral default, no crash
