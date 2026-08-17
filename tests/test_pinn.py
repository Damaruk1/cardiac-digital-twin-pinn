"""
test_pinn.py
-------------
Phase 12-13 tests: verify the PINN architecture, collocation point
sampling, and PDE residual computation all behave correctly.

Run with:
    pytest tests/test_pinn.py -v
"""

import torch

from src.pinn.collocation import Domain, sample_boundary_points, sample_collocation_points, sample_initial_condition
from src.pinn.fitzhugh_nagumo import FHNParameters, compute_fhn_residuals
from src.pinn.pinn_model import PINNNet
from src.pinn.pinn_trainer import PINNTrainer


def test_pinn_model_output_shape():
    model = PINNNet(hidden_dim=32, n_hidden_layers=3)
    x = torch.rand(10, 1, requires_grad=True)
    t = torch.rand(10, 1, requires_grad=True)

    output = model(x, t)

    assert output.shape == (10, 2)  # columns: u, w


def test_collocation_points_have_gradients_enabled():
    domain = Domain()
    x, t = sample_collocation_points(50, domain)

    assert x.requires_grad
    assert t.requires_grad
    assert x.shape == (50, 1)


def test_collocation_points_within_domain_bounds():
    domain = Domain(x_min=0.0, x_max=2.0, t_min=0.0, t_max=3.0)
    x, t = sample_collocation_points(200, domain)

    assert x.min() >= 0.0 and x.max() <= 2.0
    assert t.min() >= 0.0 and t.max() <= 3.0


def test_initial_condition_stimulus_is_localized():
    """The stimulus should be 1 near x_min and 0 elsewhere -- not
    spread uniformly across the whole domain."""
    domain = Domain(x_min=0.0, x_max=1.0)
    x, t, u0, w0 = sample_initial_condition(500, domain, stimulus_width=0.1)

    assert torch.all(t == 0.0)  # all at t=0 by definition
    near_start = x < 0.1
    assert torch.all(u0[near_start] == 1.0)
    assert torch.all(u0[~near_start] == 0.0)
    assert torch.all(w0 == 0.0)  # recovery variable starts at rest


def test_boundary_points_only_at_edges():
    domain = Domain(x_min=0.0, x_max=1.0)
    x, t = sample_boundary_points(50, domain)

    assert x.shape == (100, 1)  # n_points at EACH of 2 boundaries
    unique_x = torch.unique(x)
    assert torch.allclose(unique_x, torch.tensor([0.0, 1.0]))


def test_fhn_residual_output_shapes():
    model = PINNNet(hidden_dim=16, n_hidden_layers=2)
    domain = Domain()
    x, t = sample_collocation_points(20, domain)
    params = FHNParameters()

    r_u, r_w = compute_fhn_residuals(model, x, t, params)

    assert r_u.shape == (20, 1)
    assert r_w.shape == (20, 1)


def test_fhn_residual_is_finite():
    """Residuals should never be NaN/Inf for reasonable random inputs --
    catches numerical instability in the derivative chain."""
    model = PINNNet(hidden_dim=16, n_hidden_layers=2)
    domain = Domain()
    x, t = sample_collocation_points(50, domain)
    params = FHNParameters()

    r_u, r_w = compute_fhn_residuals(model, x, t, params)

    assert torch.all(torch.isfinite(r_u))
    assert torch.all(torch.isfinite(r_w))


def test_pinn_trainer_reduces_total_loss():
    """Sanity check: training for a modest number of steps should
    reduce the total loss from its random-initialization value.
    This is the PINN equivalent of Phase 8's 'can it overfit' test --
    it verifies the physics-based gradient flow is correctly wired."""
    torch.manual_seed(0)
    model = PINNNet(hidden_dim=32, n_hidden_layers=3)
    domain = Domain()
    params = FHNParameters()

    trainer = PINNTrainer(
        model=model, domain=domain, fhn_params=params,
        learning_rate=1e-3, n_collocation=200, n_ic=50, n_bc=20,
    )

    history = trainer.fit(epochs=100)

    assert history.total_loss[-1] < history.total_loss[0], (
        "Total loss did not decrease -- PINN training loop may be broken "
        "(check autograd graph construction, create_graph=True usage)."
    )
