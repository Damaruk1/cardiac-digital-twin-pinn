"""
test_simulation.py
--------------------
Phase 14 tests: verify the PINN-to-mesh projection produces correctly
shaped, finite activation values, and that the z-to-x remapping is
mathematically correct.

Run with:
    pytest tests/test_simulation.py -v
"""

import numpy as np
import torch

from src.anatomy.heart_mesh import generate_heart_mesh
from src.pinn.collocation import Domain
from src.pinn.pinn_model import PINNNet
from src.simulation.heart_simulator import _remap_to_domain, simulate_activation


def test_remap_to_domain_produces_correct_range():
    z_values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    domain = Domain(x_min=0.0, x_max=1.0)

    remapped = _remap_to_domain(z_values, domain)

    assert np.isclose(remapped.min(), 0.0)
    assert np.isclose(remapped.max(), 1.0)


def test_remap_preserves_order():
    """A larger z should map to a larger (or equal) x -- the mapping
    must not scramble the spatial ordering."""
    z_values = np.array([-2.0, -1.0, 0.5, 1.5])
    domain = Domain(x_min=0.0, x_max=1.0)

    remapped = _remap_to_domain(z_values, domain)

    assert np.all(np.diff(remapped) >= 0)  # monotonically non-decreasing


def test_simulate_activation_output_shape():
    model = PINNNet(hidden_dim=16, n_hidden_layers=2)
    mesh = generate_heart_mesh(n_theta=10, n_phi=10)
    domain = Domain()
    time_points = np.linspace(0, 1, 4)

    result = simulate_activation(model, mesh, domain, time_points)

    assert result.activation.shape == (4, len(mesh.vertices))
    assert len(result.time_points) == 4


def test_simulate_activation_values_are_finite():
    model = PINNNet(hidden_dim=16, n_hidden_layers=2)
    mesh = generate_heart_mesh(n_theta=10, n_phi=10)
    domain = Domain()
    time_points = np.linspace(0, 1, 3)

    result = simulate_activation(model, mesh, domain, time_points)

    assert np.all(np.isfinite(result.activation))


def test_simulate_activation_does_not_modify_model_gradients():
    """simulate_activation should run in inference mode -- it must not
    leave the model in training mode or accumulate unwanted gradients."""
    model = PINNNet(hidden_dim=16, n_hidden_layers=2)
    mesh = generate_heart_mesh(n_theta=10, n_phi=10)
    domain = Domain()

    simulate_activation(model, mesh, domain, np.array([0.5]))

    assert not model.training  # should be left in eval mode
    for param in model.parameters():
        assert param.grad is None  # no gradients should have been computed
