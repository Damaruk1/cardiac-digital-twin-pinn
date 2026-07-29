"""
demo_phase4_visualization.py
------------------------------
Demonstrates clinical-grid ECG plotting:
    1. Single-lead plot with R-peaks marked, on the standard grid.
    2. Multi-lead plot using PLACEHOLDER leads (simple transforms of
       one synthetic signal) -- NOT physiologically real multi-lead
       data. Real multi-lead data arrives in Phase 5.

Run with:
    python -m scripts.demo_phase4_visualization
"""

import numpy as np
import matplotlib.pyplot as plt

from src.data.synthetic import add_noise, generate_clean_ecg
from src.preprocessing.filters import ECGFilter
from src.preprocessing.peak_detection import detect_r_peaks
from src.visualization.ecg_plotter import ECGPlotter
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

    # --- Reuse Phase 3 pipeline to get a clean signal + R-peaks ---
    clean = generate_clean_ecg(duration_sec=6, sampling_rate=sr)
    noisy = add_noise(clean, sampling_rate=sr, powerline_freq_hz=config.powerline_freq_hz)
    ecg_filter = ECGFilter(sampling_rate=sr)
    cleaned = ecg_filter.clean(noisy, powerline_freq_hz=config.powerline_freq_hz)
    result = detect_r_peaks(cleaned, sampling_rate=sr, max_heart_rate_bpm=config.max_heart_rate_bpm)

    plotter = ECGPlotter()

    # --- 1. Single-lead clinical grid plot ---
    logger.info("Rendering single-lead clinical grid plot...")
    ax = plotter.plot_single_lead(
        cleaned, sr, title="Lead II (synthetic, cleaned)", peak_indices=result.peak_indices
    )
    ax.figure.savefig("logs/phase4_single_lead_clinical.png", dpi=150)
    plt.close(ax.figure)
    logger.info("Saved logs/phase4_single_lead_clinical.png")

    # --- 2. Multi-lead plot using PLACEHOLDER leads ---
    # These are NOT real leads -- just amplitude-scaled/inverted copies
    # of one synthetic signal, to demonstrate the multi-lead plotting
    # utility works correctly. Real 12-lead data arrives in Phase 5.
    logger.warning(
        "Multi-lead plot below uses PLACEHOLDER leads (scaled/inverted "
        "copies of one signal) -- not physiologically real. Real "
        "multi-lead data is loaded in Phase 5."
    )
    placeholder_leads = {
        "I": cleaned,
        "II (inverted)": -cleaned,
        "III (half amplitude)": cleaned * 0.5,
    }
    fig = plotter.plot_multi_lead(placeholder_leads, sr, peak_indices=result.peak_indices)
    fig.savefig("logs/phase4_multi_lead_placeholder.png", dpi=150)
    plt.close(fig)
    logger.info("Saved logs/phase4_multi_lead_placeholder.png")


if __name__ == "__main__":
    main()
