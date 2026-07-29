"""
demo_phase3_preprocessing.py
-----------------------------
Demonstrates the full Phase 3 preprocessing pipeline end-to-end:

    1. Generate a synthetic ECG (stand-in for real data, Phase 5).
    2. Inject realistic noise (baseline wander + powerline interference).
    3. Clean it using ECGFilter (bandpass + notch).
    4. Detect R-peaks and compute heart rate.
    5. Plot raw vs cleaned signal with detected peaks marked.

Run with:
    python -m scripts.demo_phase3_preprocessing
"""

import matplotlib.pyplot as plt

from src.data.synthetic import add_noise, generate_clean_ecg
from src.preprocessing.filters import ECGFilter
from src.preprocessing.peak_detection import detect_r_peaks
from src.utils.config_loader import load_config
from src.utils.logger import get_logger


def main() -> None:
    config = load_config("configs/config.yaml")
    logger = get_logger(
        name=__name__,
        log_dir=str(config.logs),
        log_filename=config.log_filename,
        level=config.log_level,
        log_to_file=config.log_to_file,
    )

    sr = config.sampling_rate_hz

    # Step 1 + 2: synthetic signal + injected noise
    logger.info("Generating synthetic ECG (true heart rate = 75 bpm)...")
    clean = generate_clean_ecg(duration_sec=10, sampling_rate=sr)
    noisy = add_noise(clean, sampling_rate=sr, powerline_freq_hz=config.powerline_freq_hz)

    # Step 3: filter
    logger.info("Applying bandpass + notch filters...")
    ecg_filter = ECGFilter(sampling_rate=sr)
    cleaned = ecg_filter.clean(noisy, powerline_freq_hz=config.powerline_freq_hz)

    # Step 4: R-peak detection + heart rate
    logger.info("Detecting R-peaks...")
    result = detect_r_peaks(
        cleaned, sampling_rate=sr, max_heart_rate_bpm=config.max_heart_rate_bpm
    )
    logger.info(f"Detected {len(result.peak_indices)} R-peaks.")
    logger.info(f"Estimated heart rate: {result.heart_rate_bpm:.1f} bpm (true = 75 bpm)")

    # Step 5: plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    time_axis = [i / sr for i in range(len(noisy))]

    axes[0].plot(time_axis, noisy, color="tab:red", linewidth=0.8)
    axes[0].set_title("Raw (noisy) synthetic ECG")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(time_axis, cleaned, color="tab:blue", linewidth=0.8, label="Cleaned signal")
    axes[1].scatter(
        result.peak_times_sec,
        cleaned[result.peak_indices],
        color="black",
        marker="x",
        label="Detected R-peaks",
        zorder=5,
    )
    axes[1].set_title(f"Cleaned ECG with R-peaks (HR = {result.heart_rate_bpm:.1f} bpm)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    output_path = "logs/phase3_preprocessing_demo.png"
    plt.savefig(output_path, dpi=150)
    logger.info(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main()
