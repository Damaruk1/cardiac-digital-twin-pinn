"""
synthetic.py
------------
Generates a realistic synthetic ECG signal, with optional injected
noise (baseline wander + powerline interference), for testing the
preprocessing pipeline BEFORE real datasets are wired up in Phase 5.

This is a temporary stand-in. Nothing here is used past Phase 5 —
once we load real data, this module stays only for unit-test fixtures.
"""

import numpy as np
import neurokit2 as nk


def generate_clean_ecg(duration_sec: int = 10, sampling_rate: int = 250) -> np.ndarray:
    """
    Generate a clean synthetic ECG signal using neurokit2's
    physiologically-realistic ECG simulator.

    Args:
        duration_sec: Length of the signal in seconds.
        sampling_rate: Samples per second (Hz).

    Returns:
        1D numpy array of the clean ECG signal.
    """
    ecg = nk.ecg_simulate(
        duration=duration_sec,
        sampling_rate=sampling_rate,
        heart_rate=75,
        noise=0.0,  # no built-in noise -- we add our own, controlled noise below
    )
    return np.asarray(ecg)


def add_noise(
    clean_signal: np.ndarray,
    sampling_rate: int = 250,
    baseline_wander_amplitude: float = 0.3,
    powerline_freq_hz: float = 50.0,
    powerline_amplitude: float = 0.1,
    random_seed: int = 42,
) -> np.ndarray:
    """
    Inject realistic noise into a clean ECG signal: baseline wander
    (slow drift, simulates breathing/electrode movement) and powerline
    interference (simulates AC mains pickup).

    Args:
        clean_signal: The clean 1D ECG array.
        sampling_rate: Samples per second (Hz).
        baseline_wander_amplitude: Amplitude of the slow drift (mV-ish).
        powerline_freq_hz: 50 Hz (most of the world) or 60 Hz (US/parts of Americas).
        powerline_amplitude: Amplitude of the powerline sine wave.
        random_seed: For reproducibility.

    Returns:
        Noisy 1D ECG array, same length as input.
    """
    rng = np.random.default_rng(random_seed)
    n_samples = len(clean_signal)
    t = np.arange(n_samples) / sampling_rate

    # Baseline wander: a very low frequency sine wave (~0.2 Hz, like breathing)
    baseline_wander = baseline_wander_amplitude * np.sin(2 * np.pi * 0.2 * t)

    # Powerline interference: a sine wave at exactly the mains frequency
    powerline_noise = powerline_amplitude * np.sin(2 * np.pi * powerline_freq_hz * t)

    # Small amount of random (EMG-like) noise
    random_noise = rng.normal(0, 0.02, size=n_samples)

    return clean_signal + baseline_wander + powerline_noise + random_noise
