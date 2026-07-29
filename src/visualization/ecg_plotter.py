"""
ecg_plotter.py
--------------
Renders ECG signals on a standard clinical-style grid (the pink/red
graph paper look), matching the international ECG paper standard:
    - Small box:  0.04s x 0.1mV
    - Big box:    0.20s x 0.5mV  (5 small boxes)

This lets intervals (PR, QRS, QT) be estimated visually by counting
boxes, exactly like a clinician reading a printed ECG strip.
"""

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import MultipleLocator


class ECGPlotter:
    """Draws ECG signals on a clinical-standard grid."""

    SMALL_BOX_TIME_SEC = 0.04
    SMALL_BOX_VOLTAGE_MV = 0.1
    BIG_BOX_TIME_SEC = 0.20
    BIG_BOX_VOLTAGE_MV = 0.5

    MINOR_GRID_COLOR = "#ffb3c1"  # light pink
    MAJOR_GRID_COLOR = "#ff4d6d"  # deeper red
    SIGNAL_COLOR = "#000000"

    def _draw_grid(self, ax: Axes) -> None:
        """Apply the standard small/big box grid to a matplotlib Axes."""
        ax.xaxis.set_minor_locator(MultipleLocator(self.SMALL_BOX_TIME_SEC))
        ax.yaxis.set_minor_locator(MultipleLocator(self.SMALL_BOX_VOLTAGE_MV))
        ax.xaxis.set_major_locator(MultipleLocator(self.BIG_BOX_TIME_SEC))
        ax.yaxis.set_major_locator(MultipleLocator(self.BIG_BOX_VOLTAGE_MV))

        ax.grid(which="minor", color=self.MINOR_GRID_COLOR, linewidth=0.5)
        ax.grid(which="major", color=self.MAJOR_GRID_COLOR, linewidth=1.0)
        ax.set_facecolor("#fff5f7")  # faint pink paper background

    def plot_single_lead(
        self,
        signal: np.ndarray,
        sampling_rate: int,
        title: str = "ECG",
        peak_indices: Optional[np.ndarray] = None,
        ax: Optional[Axes] = None,
    ) -> Axes:
        """
        Plot one lead on a clinical grid.

        Args:
            signal: 1D ECG signal array.
            sampling_rate: Samples per second (Hz).
            title: Plot title (e.g. "Lead II").
            peak_indices: Optional array of R-peak sample indices to mark.
            ax: Optional existing Axes to draw on (for multi-lead grids).
                If None, a new figure+axes is created.

        Returns:
            The matplotlib Axes the signal was drawn on.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 3))

        time_axis = np.arange(len(signal)) / sampling_rate

        self._draw_grid(ax)
        ax.plot(time_axis, signal, color=self.SIGNAL_COLOR, linewidth=1.0)

        if peak_indices is not None and len(peak_indices) > 0:
            ax.scatter(
                peak_indices / sampling_rate,
                signal[peak_indices],
                color="tab:blue",
                marker="o",
                s=20,
                zorder=5,
                label="R-peak",
            )
            ax.legend(loc="upper right", fontsize=8)

        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlim(time_axis[0], time_axis[-1])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("mV")
        return ax

    def plot_multi_lead(
        self,
        leads: dict,
        sampling_rate: int,
        peak_indices: Optional[np.ndarray] = None,
        figsize_per_lead: float = 1.8,
    ) -> plt.Figure:
        """
        Plot multiple leads stacked vertically, each on its own clinical grid.

        Args:
            leads: Dict mapping lead name -> 1D signal array,
                   e.g. {"I": arr1, "II": arr2, "V1": arr3}.
            sampling_rate: Samples per second (Hz), shared across leads.
            peak_indices: Optional R-peaks to mark on every lead (assumes
                          leads are time-aligned, which is true for real
                          multi-lead recordings).
            figsize_per_lead: Vertical inches allocated per lead subplot.

        Returns:
            The matplotlib Figure containing all lead subplots.
        """
        n_leads = len(leads)
        fig, axes = plt.subplots(
            n_leads, 1, figsize=(12, figsize_per_lead * n_leads), sharex=True
        )
        if n_leads == 1:
            axes = [axes]

        for ax, (lead_name, signal) in zip(axes, leads.items()):
            self.plot_single_lead(
                signal,
                sampling_rate,
                title=f"Lead {lead_name}",
                peak_indices=peak_indices,
                ax=ax,
            )

        plt.tight_layout()
        return fig
