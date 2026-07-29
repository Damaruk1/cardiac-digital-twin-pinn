"""
filters.py
----------
Digital filters for cleaning raw ECG signals.

Uses Butterworth filters applied with `filtfilt` (zero-phase, forward-
backward filtering) so that R-peaks and interval timings are NOT
shifted -- a single-pass filter would introduce a phase delay that
corrupts every downstream interval measurement (PR, QRS, QT).
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


class ECGFilter:
    """
    Bundles the standard ECG cleaning filters: bandpass (removes
    baseline wander + high-frequency noise) and notch (removes
    powerline interference).
    """

    def __init__(self, sampling_rate: int):
        """
        Args:
            sampling_rate: Samples per second (Hz) of the input signal.
                            Must match the signal's true sampling rate,
                            or every cutoff frequency below is wrong.
        """
        self.sampling_rate = sampling_rate
        self.nyquist = sampling_rate / 2.0  # Nyquist frequency

    def bandpass_filter(
        self,
        signal: np.ndarray,
        low_cut_hz: float = 0.5,
        high_cut_hz: float = 40.0,
        order: int = 4,
    ) -> np.ndarray:
        """
        Apply a Butterworth bandpass filter.

        Args:
            signal: Raw 1D ECG signal.
            low_cut_hz: High-pass cutoff -- removes baseline wander below this.
            high_cut_hz: Low-pass cutoff -- removes noise above this.
            order: Filter order (steepness of roll-off). Higher = sharper
                   cutoff but more risk of ringing artifacts. 4 is a safe default.

        Returns:
            Filtered 1D ECG signal, same length as input.
        """
        low = low_cut_hz / self.nyquist
        high = high_cut_hz / self.nyquist

        if not (0 < low < high < 1):
            raise ValueError(
                f"Invalid cutoff frequencies for sampling_rate={self.sampling_rate}: "
                f"low={low_cut_hz}Hz, high={high_cut_hz}Hz, nyquist={self.nyquist}Hz"
            )

        b, a = butter(order, [low, high], btype="band")
        return filtfilt(b, a, signal)

    def notch_filter(
        self,
        signal: np.ndarray,
        notch_freq_hz: float = 50.0,
        quality_factor: float = 30.0,
    ) -> np.ndarray:
        """
        Apply a notch filter to remove a narrow frequency band --
        specifically the powerline frequency (50Hz in most countries,
        60Hz in the US and parts of the Americas).

        Args:
            signal: 1D ECG signal (ideally already bandpass-filtered).
            notch_freq_hz: The exact frequency to remove.
            quality_factor: How narrow the notch is. Higher = narrower
                            (removes less of the surrounding signal).

        Returns:
            Filtered 1D ECG signal, same length as input.
        """
        w0 = notch_freq_hz / self.nyquist
        b, a = iirnotch(w0, quality_factor)
        return filtfilt(b, a, signal)

    def clean(
        self,
        signal: np.ndarray,
        powerline_freq_hz: float = 50.0,
    ) -> np.ndarray:
        """
        Convenience method: runs the full standard cleaning pipeline
        (bandpass, then notch) in one call.

        Args:
            signal: Raw 1D ECG signal.
            powerline_freq_hz: 50.0 or 60.0 depending on the region
                                the data was recorded in.

        Returns:
            Fully cleaned 1D ECG signal.
        """
        bandpassed = self.bandpass_filter(signal)
        cleaned = self.notch_filter(bandpassed, notch_freq_hz=powerline_freq_hz)
        return cleaned
