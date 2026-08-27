"""
heart_simulator.py
--------------------
Projects the trained 1D PINN solution u(x,t) onto the 3D heart mesh
(Phase 11), producing an activation value at every mesh vertex, for
any requested set of time points.

How the projection works:
    The PINN's spatial coordinate x represents position along the
    apex-to-base fiber, x in [0,1]. The heart mesh's z-coordinate
    spans [-c, c] (apex to base, see heart_mesh.py). We linearly
    remap each vertex's z to the PINN's x range, then evaluate the
    trained network there. Vertices at similar heights along the
    heart get similar activation values -- a reasonable simplified
    stand-in for genuine 3D wave propagation.
"""

from dataclasses import dataclass

import numpy as np
import torch

from src.anatomy.heart_mesh import HeartMesh
from src.pinn.collocation import Domain
from src.pinn.pinn_model import PINNNet


@dataclass
class SimulationResult:
    """Activation values at every mesh vertex, for a set of time points."""

    time_points: np.ndarray       # shape (n_times,)
    activation: np.ndarray        # shape (n_times, n_vertices) -- u values
    mesh: HeartMesh


def _remap_to_domain(z_values: np.ndarray, domain: Domain) -> np.ndarray:
    """Linearly maps mesh z-coordinates to the PINN's x-domain range."""
    z_min, z_max = z_values.min(), z_values.max()
    normalized = (z_values - z_min) / (z_max - z_min)  # -> [0, 1]
    return normalized * (domain.x_max - domain.x_min) + domain.x_min


def simulate_activation(
    model: PINNNet,
    mesh: HeartMesh,
    domain: Domain,
    time_points: np.ndarray,
) -> SimulationResult:
    """
    Evaluates the trained PINN at every mesh vertex, for each requested
    time point, producing a full spatiotemporal activation map over
    the anatomical geometry.

    Args:
        model: A trained PINNNet (from Phase 12-13).
        mesh: A HeartMesh (from Phase 11).
        domain: The Domain the PINN was trained on (for correct x-range remapping).
        time_points: 1D array of times to simulate, e.g. np.linspace(0, 1, 5).

    Returns:
        SimulationResult with activation values at every (time, vertex) pair.
    """
    model.eval()

    x_mapped = _remap_to_domain(mesh.vertices[:, 2], domain)  # z -> x
    n_vertices = len(x_mapped)

    all_activations = []
    with torch.no_grad():
        for t_val in time_points:
            x_tensor = torch.tensor(x_mapped, dtype=torch.float32).unsqueeze(1)
            t_tensor = torch.full((n_vertices, 1), float(t_val), dtype=torch.float32)

            output = model(x_tensor, t_tensor)
            u_values = output[:, 0].numpy()  # transmembrane potential column
            all_activations.append(u_values)

    return SimulationResult(
        time_points=np.array(time_points),
        activation=np.array(all_activations),  # shape (n_times, n_vertices)
        mesh=mesh,
    )
