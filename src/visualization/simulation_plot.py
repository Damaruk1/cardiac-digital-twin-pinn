"""
simulation_plot.py
--------------------
Visualizes a SimulationResult as a grid of 3D snapshots, one per
requested time point, colored by transmembrane potential -- showing
how activation spreads across the anatomical heart mesh over time.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from src.simulation.heart_simulator import SimulationResult


def plot_simulation_snapshots(result: SimulationResult, n_cols: int = 3) -> plt.Figure:
    """
    Plots one 3D scatter subplot per time point, all sharing the same
    color scale, so activation spread is visually comparable across
    panels.
    """
    n_times = len(result.time_points)
    n_rows = (n_times + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(5 * n_cols, 4.5 * n_rows))

    vmin, vmax = result.activation.min(), result.activation.max()

    for i, t_val in enumerate(result.time_points):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection="3d")
        points = result.mesh.vertices
        colors = result.activation[i]

        scatter = ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            c=colors, cmap="hot", vmin=vmin, vmax=vmax, s=8,
        )
        ax.set_title(f"t = {t_val:.2f}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    fig.colorbar(scatter, ax=fig.axes, label="u (transmembrane potential)", shrink=0.6)
    fig.suptitle("Simulated Cardiac Activation Over Time", fontsize=14)
    return fig
