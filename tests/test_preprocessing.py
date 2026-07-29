"""
test_preprocessing.py
----------------------
Phase 3 tests: verify filtering and R-peak detection behave correctly
using synthetic ECG signals with known ground-truth heart rate.

Run with:
    pytest tests/test_preprocessing.py -v
"""

import numpy as np

from src.data.synthetic import add_noise, generate_clean_ecg
from src.preprocessing.filters import ECGFilter
from src.preprocessing.peak_detection import detect_r_peaks

SAMPLING_RATE = 250


def test_bandpass_filter_removes_baseline_wander():
    """A pure low-frequency (0.1 Hz) signal should be heavily attenuated
    by a bandpass filter with a 0.5 Hz low cutoff."""
    t = np.arange(0, 10, 1 / SAMPLING_RATE)
    baseline_wander_only = 1.0 * np.sin(2 * np.pi * 0.1 * t)

    ecg_filter = ECGFilter(sampling_rate=SAMPLING_RATE)
    filtered = ecg_filter.bandpass_filter(baseline_wander_only)

    # Amplitude should be drastically reduced (allow filter edge effects)
    assert np.std(filtered) < 0.1 * np.std(baseline_wander_only)


def test_notch_filter_removes_powerline_frequency():
    """A pure 50Hz sine wave should be heavily attenuated by the notch filter."""
    t = np.arange(0, 5, 1 / SAMPLING_RATE)
    powerline_only = 1.0 * np.sin(2 * np.pi * 50.0 * t)

    ecg_filter = ECGFilter(sampling_rate=SAMPLING_RATE)
    filtered = ecg_filter.notch_filter(powerline_only, notch_freq_hz=50.0)

    assert np.std(filtered) < 0.1 * np.std(powerline_only)


def test_bandpass_filter_rejects_invalid_cutoffs():
    """Cutoffs above Nyquist frequency should raise a clear error."""
    ecg_filter = ECGFilter(sampling_rate=SAMPLING_RATE)
    dummy_signal = np.zeros(1000)
    try:
        ecg_filter.bandpass_filter(dummy_signal, low_cut_hz=0.5, high_cut_hz=200.0)
        assert False, "Expected ValueError for cutoff above Nyquist"
    except ValueError:
        pass


def test_r_peak_detection_matches_known_heart_rate():
    """End-to-end: a synthetic 75 bpm ECG, cleaned and peak-detected,
    should report a heart rate within 5 bpm of the true value."""
    clean = generate_clean_ecg(duration_sec=10, sampling_rate=SAMPLING_RATE)
    noisy = add_noise(clean, sampling_rate=SAMPLING_RATE, random_seed=1)

    ecg_filter = ECGFilter(sampling_rate=SAMPLING_RATE)
    cleaned = ecg_filter.clean(noisy)

    result = detect_r_peaks(cleaned, sampling_rate=SAMPLING_RATE)

    assert len(result.peak_indices) > 5
    assert abs(result.heart_rate_bpm - 75.0) < 5.0


def test_r_peak_detection_handles_flat_signal_gracefully():
    """A flat (no-peak) signal should return an empty result, not crash."""
    flat_signal = np.zeros(1000)
    result = detect_r_peaks(flat_signal, sampling_rate=SAMPLING_RATE)

    assert len(result.peak_indices) == 0
    assert result.heart_rate_bpm == 0.0
