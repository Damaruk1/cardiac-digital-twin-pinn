"""
heart_mesh.py
-------------
Generates a SIMPLIFIED parametric 3D heart-like surface, with each
point labeled by anatomical wall region (anterior/inferior/lateral/
septal/posterior).

IMPORTANT: this is a generic geometric surrogate, NOT a patient-specific
anatomical model. Real patient-specific geometry would come from
segmenting an MRI or CT scan -- well outside this project's scope. This
mesh exists to give the PINN (Phase 12-13) a spatial domain to solve
its physics over, and to give us something to visualize anatomical
findings on.

Shape: a tapered prolate spheroid (egg-like), apex pointing down (-z),
base at the top (+z) -- a common simplified stand-in for the left
ventricle's overall shape in introductory cardiac modeling.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class HeartMesh:
    """A labeled point cloud approximating the heart's surface."""

    vertices: np.ndarray       # shape (N, 3): x, y, z coordinates
    region_labels: np.ndarray  # shape (N,): region name per vertex
    n_theta: int                # grid resolution, kept for reshaping if needed
    n_phi: int


def _assign_region(theta: float, phi: float, inferior_theta_threshold: float) -> str:
    """
    Assigns an anatomical region label based on a point's angular
    position on the parametric surface.

    Args:
        theta: Polar angle from the apex (0) to the base (pi).
        phi: Azimuthal angle around the long axis, in [-pi, pi].
        inferior_theta_threshold: Points with theta below this (i.e.
                                    close to the apex) are labeled
                                    "inferior" regardless of azimuth,
                                    approximating the diaphragmatic
                                    surface at the bottom of the heart.
    """
    if theta < inferior_theta_threshold:
        return "inferior"

    # Azimuthal quadrants for the remaining (non-apex) surface
    if -np.pi / 4 <= phi < np.pi / 4:
        return "anterior"
    elif np.pi / 4 <= phi < 3 * np.pi / 4:
        return "lateral"
    elif -3 * np.pi / 4 <= phi < -np.pi / 4:
        return "septal"
    else:
        return "posterior"  # not clinically localizable from standard leads


def generate_heart_mesh(
    n_theta: int = 30,
    n_phi: int = 40,
    a: float = 1.0,
    b: float = 1.0,
    c: float = 1.6,
    inferior_fraction: float = 0.25,
) -> HeartMesh:
    """
    Generates the parametric heart-shaped point cloud.

    Args:
        n_theta: Number of polar angle samples (apex to base).
        n_phi: Number of azimuthal angle samples (around the long axis).
        a, b: Radii controlling the width (x, y directions).
        c: Radius controlling the length (z direction, apex to base).
        inferior_fraction: Fraction of the theta range (from the apex)
                             labeled as the inferior wall.

    Returns:
        A HeartMesh with vertices and per-vertex region labels.
    """
    theta_vals = np.linspace(0.01, np.pi - 0.01, n_theta)  # avoid poles exactly
    phi_vals = np.linspace(-np.pi, np.pi, n_phi, endpoint=False)
    inferior_threshold = inferior_fraction * np.pi

    vertices = []
    labels = []

    for theta in theta_vals:
        for phi in phi_vals:
            x = a * np.sin(theta) * np.cos(phi)
            y = b * np.sin(theta) * np.sin(phi)
            z = -c * np.cos(theta)  # apex (theta near 0) -> z near -c (points down)

            vertices.append([x, y, z])
            labels.append(_assign_region(theta, phi, inferior_threshold))

    return HeartMesh(
        vertices=np.array(vertices),
        region_labels=np.array(labels),
        n_theta=n_theta,
        n_phi=n_phi,
    )
