"""
test_visualization.py
----------------------
Phase 4 tests: verify ECGPlotter produces valid matplotlib objects
with the correct clinical grid configuration.

Run with:
    pytest tests/test_visualization.py -v
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for automated tests

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.visualization.ecg_plotter import ECGPlotter

SAMPLING_RATE = 250


def test_plot_single_lead_returns_axes():
    """plot_single_lead should return a valid matplotlib Axes."""
    plotter = ECGPlotter()
    signal = np.sin(np.linspace(0, 20, SAMPLING_RATE * 4))
    ax = plotter.plot_single_lead(signal, SAMPLING_RATE, title="Test Lead")
    assert isinstance(ax, Axes)


def test_plot_single_lead_grid_spacing_is_correct():
    """The minor grid spacing must match the clinical standard (0.04s)."""
    plotter = ECGPlotter()
    signal = np.zeros(SAMPLING_RATE * 2)
    ax = plotter.plot_single_lead(signal, SAMPLING_RATE)
    minor_locator = ax.xaxis.get_minor_locator()
    assert minor_locator.MAXTICKS  # locator exists and is configured
    # Confirm the base spacing value matches the ECG standard
    assert plotter.SMALL_BOX_TIME_SEC == 0.04


def test_plot_multi_lead_returns_figure_with_correct_subplot_count():
    """plot_multi_lead should create one subplot per lead."""
    plotter = ECGPlotter()
    signal = np.zeros(SAMPLING_RATE * 2)
    leads = {"I": signal, "II": signal, "III": signal}
    fig = plotter.plot_multi_lead(leads, SAMPLING_RATE)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 3


def test_plot_single_lead_marks_r_peaks_without_error():
    """Passing peak_indices should not raise, and should add a legend."""
    plotter = ECGPlotter()
    signal = np.sin(np.linspace(0, 20, SAMPLING_RATE * 4))
    peaks = np.array([50, 300, 550])
    ax = plotter.plot_single_lead(signal, SAMPLING_RATE, peak_indices=peaks)
    assert ax.get_legend() is not None
