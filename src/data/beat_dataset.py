"""
beat_dataset.py
----------------
Converts a full ECG record + its beat annotations (from Phase 5) into
individual, fixed-length, labeled windows suitable for CNN training.

Each training example = one heartbeat = a short window of the signal
centered on that beat's annotated R-peak location, paired with its
AAMI class label.
"""

from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.dataset_loader import BeatAnnotations, ECGRecord

# Fixed mapping from AAMI class letters to integer labels, in a
# consistent order used everywhere in the project (model output
# indices, confusion matrices, etc. all follow this order).
AAMI_CLASS_TO_INDEX = {"N": 0, "S": 1, "V": 2, "F": 3, "Q": 4}
INDEX_TO_AAMI_CLASS = {v: k for k, v in AAMI_CLASS_TO_INDEX.items()}


class BeatDataset(Dataset):
    """PyTorch Dataset yielding (beat_window, label) pairs."""

    def __init__(
        self,
        record: ECGRecord,
        annotations: BeatAnnotations,
        window_before: int = 90,
        window_after: int = 90,
    ):
        """
        Args:
            record: A loaded ECGRecord (from ECGDatasetLoader).
            annotations: The matching BeatAnnotations for that record.
            window_before: Samples to include before the R-peak.
            window_after: Samples to include after the R-peak.
                          Total window length = window_before + window_after.
        """
        self.record = record
        self.window_before = window_before
        self.window_after = window_after

        self.windows, self.labels = self._extract_windows(annotations)

    def _extract_windows(
        self, annotations: BeatAnnotations
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Slices out a fixed-length window around every annotated beat.
        Beats too close to the start/end of the recording (where a
        full window wouldn't fit) are skipped.
        """
        windows = []
        labels = []
        n_samples = self.record.signal.shape[0]

        for peak_idx, aami_class in zip(annotations.sample_indices, annotations.aami_classes):
            start = peak_idx - self.window_before
            end = peak_idx + self.window_after

            if start < 0 or end > n_samples:
                continue  # skip beats too close to the recording edges

            window = self.record.signal[start:end, :]  # shape: (window_length, n_leads)
            windows.append(window)
            labels.append(AAMI_CLASS_TO_INDEX[aami_class])

        return windows, labels

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            signal_tensor: shape (n_leads, window_length) -- PyTorch
                            Conv1d expects channels BEFORE length,
                            which is why we transpose here.
            label_tensor: scalar long tensor, the class index.
        """
        window = self.windows[idx]  # (window_length, n_leads)
        signal_tensor = torch.tensor(window.T, dtype=torch.float32)  # -> (n_leads, window_length)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return signal_tensor, label_tensor

    @property
    def window_length(self) -> int:
        return self.window_before + self.window_after
