"""
dataset_loader.py
------------------
Loads real ECG records from the MIT-BIH Arrhythmia Database (via
PhysioNet, using the WFDB format) and maps raw cardiologist beat
annotations onto the standard AAMI EC57 5-class scheme.

WFDB format primer:
    Every "record" is 3 files sharing a base name, e.g. record "100":
        100.hea  -- header: sampling rate, lead names, duration
        100.dat  -- the raw signal (binary-encoded voltages)
        100.atr  -- annotations: sample index + label for every beat
"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import wfdb

# AAMI EC57 standard: groups MIT-BIH's ~15 raw beat symbols into
# 5 clinically meaningful superclasses. This mapping is used across
# essentially all published ECG arrhythmia classification research,
# so we standardize on it now rather than inventing our own scheme.
AAMI_BEAT_MAPPING: Dict[str, str] = {
    # Normal
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    # Supraventricular ectopic
    "A": "S", "a": "S", "J": "S", "S": "S",
    # Ventricular ectopic
    "V": "V", "E": "V",
    # Fusion
    "F": "F",
    # Unknown / paced / unclassifiable
    "/": "Q", "f": "Q", "Q": "Q",
}


@dataclass
class ECGRecord:
    """A single loaded ECG record: signal + metadata."""

    record_name: str
    signal: np.ndarray          # shape: (n_samples, n_leads)
    sampling_rate: int
    lead_names: List[str]
    duration_sec: float


@dataclass
class BeatAnnotations:
    """Beat-level annotations for a record."""

    record_name: str
    sample_indices: np.ndarray   # where each beat occurs, in samples
    raw_symbols: List[str]       # original MIT-BIH symbols (N, V, A, ...)
    aami_classes: List[str]      # mapped to N/S/V/F/Q


class ECGDatasetLoader:
    """
    Downloads and loads records from the MIT-BIH Arrhythmia Database.
    """

    def __init__(self, data_dir: Path, database: str = "mitdb"):
        """
        Args:
            data_dir: Local directory to store/read downloaded records.
            database: PhysioNet database short-code. "mitdb" is the
                      MIT-BIH Arrhythmia Database.
        """
        self.data_dir = Path(data_dir)
        self.database = database
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_record(self, record_name: str) -> None:
        """
        Download one record's .hea/.dat/.atr files from PhysioNet,
        if not already present locally.

        Args:
            record_name: e.g. "100" (MIT-BIH records are named 100-234).
        """
        local_header = self.data_dir / f"{record_name}.hea"
        if local_header.exists():
            return  # already downloaded, skip

        wfdb.dl_database(
            self.database,
            dl_dir=str(self.data_dir),
            records=[record_name],
            annotators=["atr"],
        )

    def load_record(self, record_name: str) -> ECGRecord:
        """
        Load a record's signal data from local files (must be
        downloaded first via download_record).

        Args:
            record_name: e.g. "100".

        Returns:
            ECGRecord with the signal array and metadata.
        """
        record = wfdb.rdrecord(str(self.data_dir / record_name))
        return ECGRecord(
            record_name=record_name,
            signal=record.p_signal,
            sampling_rate=record.fs,
            lead_names=record.sig_name,
            duration_sec=record.sig_len / record.fs,
        )

    def load_annotations(self, record_name: str) -> BeatAnnotations:
        """
        Load beat-by-beat cardiologist annotations for a record, and
        map raw symbols to AAMI superclasses.

        Args:
            record_name: e.g. "100".

        Returns:
            BeatAnnotations with sample locations and both raw and
            AAMI-mapped labels.
        """
        annotation = wfdb.rdann(str(self.data_dir / record_name), "atr")

        raw_symbols = list(annotation.symbol)
        aami_classes = [
            AAMI_BEAT_MAPPING.get(sym, "Q")  # unknown symbols -> "Q"
            for sym in raw_symbols
        ]

        return BeatAnnotations(
            record_name=record_name,
            sample_indices=np.array(annotation.sample),
            raw_symbols=raw_symbols,
            aami_classes=aami_classes,
        )

    @staticmethod
    def get_class_distribution(annotations: BeatAnnotations) -> Counter:
        """
        Count how many beats fall into each AAMI class -- makes the
        class imbalance immediately visible.

        Args:
            annotations: Output of load_annotations().

        Returns:
            Counter mapping AAMI class -> count.
        """
        return Counter(annotations.aami_classes)
