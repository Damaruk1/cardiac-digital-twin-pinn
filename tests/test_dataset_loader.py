"""
test_dataset_loader.py
------------------------
Phase 5 tests: verify ECGDatasetLoader parsing and AAMI mapping logic
WITHOUT requiring network access. We generate a small, fake WFDB
record locally (using wfdb's own writer functions) and point the
loader at it -- this tests our parsing code, not PhysioNet's servers.

Run with:
    pytest tests/test_dataset_loader.py -v
"""

import shutil
from pathlib import Path

import numpy as np
import pytest
import wfdb

from src.data.dataset_loader import AAMI_BEAT_MAPPING, ECGDatasetLoader

TEST_DATA_DIR = Path("tests/_tmp_wfdb_fixture")
RECORD_NAME = "test001"


@pytest.fixture(scope="module", autouse=True)
def fake_wfdb_record():
    """Creates a small synthetic WFDB record + annotations on disk,
    then cleans up after the tests in this file finish."""
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

    sampling_rate = 360
    n_samples = sampling_rate * 5  # 5 seconds
    fake_signal = np.random.normal(0, 0.1, size=(n_samples, 2)).astype(np.float64)

    wfdb.wrsamp(
        record_name=RECORD_NAME,
        fs=sampling_rate,
        units=["mV", "mV"],
        sig_name=["MLII", "V5"],
        p_signal=fake_signal,
        fmt=["16", "16"],
        write_dir=str(TEST_DATA_DIR),
    )

    # 5 fake beats with a mix of raw symbols to test AAMI mapping
    beat_samples = np.array([100, 500, 900, 1300, 1700])
    beat_symbols = ["N", "V", "A", "N", "F"]

    wfdb.wrann(
        record_name=RECORD_NAME,
        extension="atr",
        sample=beat_samples,
        symbol=beat_symbols,
        write_dir=str(TEST_DATA_DIR),
    )

    yield

    shutil.rmtree(TEST_DATA_DIR, ignore_errors=True)


def test_load_record_reads_correct_metadata():
    loader = ECGDatasetLoader(data_dir=TEST_DATA_DIR)
    record = loader.load_record(RECORD_NAME)

    assert record.sampling_rate == 360
    assert record.lead_names == ["MLII", "V5"]
    assert record.signal.shape[1] == 2
    assert abs(record.duration_sec - 5.0) < 0.1


def test_load_annotations_maps_symbols_to_aami_classes():
    loader = ECGDatasetLoader(data_dir=TEST_DATA_DIR)
    annotations = loader.load_annotations(RECORD_NAME)

    assert annotations.raw_symbols == ["N", "V", "A", "N", "F"]
    assert annotations.aami_classes == ["N", "V", "S", "N", "F"]


def test_class_distribution_counts_correctly():
    loader = ECGDatasetLoader(data_dir=TEST_DATA_DIR)
    annotations = loader.load_annotations(RECORD_NAME)
    distribution = loader.get_class_distribution(annotations)

    assert distribution["N"] == 2
    assert distribution["V"] == 1
    assert distribution["S"] == 1
    assert distribution["F"] == 1


def test_aami_mapping_covers_all_common_mitdb_symbols():
    """Sanity check: our mapping table should cover the symbols that
    actually appear in MIT-BIH, so nothing silently falls through to 'Q'."""
    common_symbols = ["N", "L", "R", "A", "V", "F", "/", "j", "e", "J", "a", "E", "f", "Q"]
    for symbol in common_symbols:
        assert symbol in AAMI_BEAT_MAPPING, f"Missing mapping for symbol: {symbol}"


def test_download_record_skips_if_already_present():
    """download_record should not error or re-download if files exist."""
    loader = ECGDatasetLoader(data_dir=TEST_DATA_DIR)
    # Should return immediately without attempting network access,
    # since test001.hea already exists in TEST_DATA_DIR.
    loader.download_record(RECORD_NAME)  # must not raise
