"""
heart_mesh_plot.py
--------------------
Renders the HeartMesh as a 3D scatter plot, colored by anatomical
region, with optional highlighting of specific implicated regions
(e.g. the regions a particular ECG lead points to).
"""

from typing import List, Optional

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers 3D projection

from src.anatomy.heart_mesh import HeartMesh

REGION_COLORS = {
    "anterior": "#e63946",
    "inferior": "#457b9d",
    "lateral": "#2a9d8f",
    "septal": "#f4a261",
    "posterior": "#cccccc",  # neutral gray -- not clinically localizable from standard leads
}


def plot_heart_mesh(
    mesh: HeartMesh,
    highlighted_regions: Optional[List[str]] = None,
    title: str = "Heart Anatomical Regions",
) -> plt.Figure:
    """
    Plots the heart mesh as a colored 3D scatter, optionally dimming
    all regions except a highlighted subset (e.g. the regions
    implicated by an abnormal beat's source lead).

    Args:
        mesh: A HeartMesh from generate_heart_mesh().
        highlighted_regions: If given, only these regions are shown at
                               full opacity; all others are faded, to
                               draw the eye to the implicated area(s).
        title: Plot title.

    Returns:
        The matplotlib Figure.
    """
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    for region, color in REGION_COLORS.items():
        mask = mesh.region_labels == region
        if not mask.any():
            continue

        if highlighted_regions is not None:
            alpha = 0.95 if region in highlighted_regions else 0.08
        else:
            alpha = 0.7

        points = mesh.vertices[mask]
        ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            color=color, alpha=alpha, s=8, label=region,
        )

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z (apex to base)")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    return fig
