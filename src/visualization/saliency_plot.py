"""
saliency_plot.py
-----------------
Overlays a Grad-CAM importance curve on top of the raw ECG signal,
using a color-coded background so it's immediately visible which
timesteps drove the model's prediction.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_saliency_overlay(
    signal: np.ndarray,
    cam: np.ndarray,
    title: str = "Grad-CAM",
) -> plt.Figure:
    """
    Args:
        signal: 1D array, the raw signal for ONE lead (window_length,).
        cam: 1D array, same length, values in [0, 1] from GradCAM1D.generate().
        title: Plot title.

    Returns:
        The matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    time_axis = np.arange(len(signal))

    # Background heatmap: use pcolormesh-style shading via imshow stretched
    # across the full plot, so the color intensity shows importance.
    ax.imshow(
        cam[np.newaxis, :],
        aspect="auto",
        cmap="Reds",
        alpha=0.5,
        extent=[time_axis[0], time_axis[-1], signal.min() - 0.1, signal.max() + 0.1],
    )

    ax.plot(time_axis, signal, color="black", linewidth=1.2, label="ECG signal")
    ax.set_title(title)
    ax.set_xlabel("Sample index within window")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right")
    plt.tight_layout()
    return fig
