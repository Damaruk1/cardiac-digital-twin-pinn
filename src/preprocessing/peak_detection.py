"""
peak_detection.py
------------------
Detects R-peaks (the sharp spike of the QRS complex) in a cleaned
ECG signal, and derives heart rate from the intervals between them.

This is a simplified peak detector (threshold + minimum-distance
constraint). A full Pan-Tompkins algorithm (derivative + squaring +
moving-window integration) is more robust and is a natural upgrade
for Phase 5 once we're working with noisier real-world data.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks


@dataclass
class RPeakResult:
    """Container for R-peak detection output."""

    peak_indices: np.ndarray      # sample indices where R-peaks occur
    peak_times_sec: np.ndarray    # R-peak times in seconds
    rr_intervals_sec: np.ndarray  # time between consecutive R-peaks
    heart_rate_bpm: float         # average heart rate across the signal


def detect_r_peaks(
    signal: np.ndarray,
    sampling_rate: int,
    max_heart_rate_bpm: int = 220,
) -> RPeakResult:
    """
    Detect R-peaks in a cleaned ECG signal and compute heart rate.

    Args:
        signal: Cleaned (filtered) 1D ECG signal.
        sampling_rate: Samples per second (Hz).
        max_heart_rate_bpm: Physiological upper bound on heart rate,
                             used to set a minimum distance between
                             peaks (prevents detecting noise spikes
                             as extra beats).

    Returns:
        RPeakResult with peak locations, RR intervals, and heart rate.
    """
    # Convert "max heart rate" into "minimum samples between beats".
    # E.g. 220 bpm -> 220/60 beats per second -> min gap in samples:
    min_distance_samples = int((60.0 / max_heart_rate_bpm) * sampling_rate)

    # Dynamic threshold: peaks must be notably above the signal's own
    # noise floor, not an arbitrary fixed number (which would break on
    # signals with different amplitude scales).
    threshold = np.mean(signal) + 0.5 * np.std(signal)

    peak_indices, _ = find_peaks(
        signal,
        height=threshold,
        distance=min_distance_samples,
    )

    peak_times_sec = peak_indices / sampling_rate

    if len(peak_indices) < 2:
        # Not enough peaks to compute intervals -- return empty result
        # rather than crashing, so callers can handle this gracefully.
        return RPeakResult(
            peak_indices=peak_indices,
            peak_times_sec=peak_times_sec,
            rr_intervals_sec=np.array([]),
            heart_rate_bpm=0.0,
        )

    rr_intervals_sec = np.diff(peak_times_sec)
    heart_rate_bpm = 60.0 / np.mean(rr_intervals_sec)

    return RPeakResult(
        peak_indices=peak_indices,
        peak_times_sec=peak_times_sec,
        rr_intervals_sec=rr_intervals_sec,
        heart_rate_bpm=float(heart_rate_bpm),
    )
