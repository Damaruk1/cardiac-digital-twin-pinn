"""
test_beat_dataset.py
----------------------
Phase 6 tests: verify BeatDataset correctly windows beats around
R-peaks, maps AAMI labels to indices, and skips edge beats -- using
directly-constructed fake ECGRecord/BeatAnnotations (no file I/O
or network needed).

Run with:
    pytest tests/test_beat_dataset.py -v
"""

import numpy as np
import torch

from src.data.beat_dataset import AAMI_CLASS_TO_INDEX, BeatDataset
from src.data.dataset_loader import BeatAnnotations, ECGRecord


def _make_fake_record(n_samples=1000, n_leads=2) -> ECGRecord:
    signal = np.random.randn(n_samples, n_leads).astype(np.float64)
    return ECGRecord(
        record_name="fake",
        signal=signal,
        sampling_rate=360,
        lead_names=["MLII", "V5"],
        duration_sec=n_samples / 360,
    )


def test_windows_extracted_with_correct_shape():
    record = _make_fake_record(n_samples=1000, n_leads=2)
    annotations = BeatAnnotations(
        record_name="fake",
        sample_indices=np.array([200, 400, 600]),
        raw_symbols=["N", "V", "N"],
        aami_classes=["N", "V", "N"],
    )
    dataset = BeatDataset(record, annotations, window_before=90, window_after=90)

    assert len(dataset) == 3
    signal_tensor, label_tensor = dataset[0]
    assert signal_tensor.shape == (2, 180)  # (n_leads, window_length)
    assert isinstance(label_tensor.item(), int)


def test_beats_too_close_to_edges_are_skipped():
    record = _make_fake_record(n_samples=1000, n_leads=1)
    annotations = BeatAnnotations(
        record_name="fake",
        # First beat too close to start, last too close to end, middle is fine
        sample_indices=np.array([10, 500, 995]),
        raw_symbols=["N", "N", "N"],
        aami_classes=["N", "N", "N"],
    )
    dataset = BeatDataset(record, annotations, window_before=90, window_after=90)

    assert len(dataset) == 1  # only the middle beat survives


def test_labels_correctly_mapped_to_aami_indices():
    record = _make_fake_record(n_samples=1000, n_leads=1)
    annotations = BeatAnnotations(
        record_name="fake",
        sample_indices=np.array([200, 400, 600, 800]),
        raw_symbols=["N", "S", "V", "F"],
        aami_classes=["N", "S", "V", "F"],
    )
    dataset = BeatDataset(record, annotations, window_before=90, window_after=90)

    expected_labels = [
        AAMI_CLASS_TO_INDEX["N"],
        AAMI_CLASS_TO_INDEX["S"],
        AAMI_CLASS_TO_INDEX["V"],
        AAMI_CLASS_TO_INDEX["F"],
    ]
    assert dataset.labels == expected_labels


def test_getitem_returns_correct_tensor_types():
    record = _make_fake_record(n_samples=1000, n_leads=2)
    annotations = BeatAnnotations(
        record_name="fake",
        sample_indices=np.array([500]),
        raw_symbols=["N"],
        aami_classes=["N"],
    )
    dataset = BeatDataset(record, annotations, window_before=90, window_after=90)

    signal_tensor, label_tensor = dataset[0]
    assert signal_tensor.dtype == torch.float32
    assert label_tensor.dtype == torch.long


def test_window_length_property():
    record = _make_fake_record()
    annotations = BeatAnnotations(
        record_name="fake", sample_indices=np.array([]), raw_symbols=[], aami_classes=[]
    )
    dataset = BeatDataset(record, annotations, window_before=64, window_after=96)

    assert dataset.window_length == 160
