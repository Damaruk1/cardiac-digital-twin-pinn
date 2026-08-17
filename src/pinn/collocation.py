"""
collocation.py
---------------
Samples the three kinds of points a PINN needs during training:

1. COLLOCATION points -- random (x,t) pairs throughout the domain
   interior, where we enforce the PDE residual == 0.
2. INITIAL CONDITION points -- points at t=0, where we enforce the
   solution matches a known starting state (here: a localized
   electrical stimulus at one end of the fiber, mimicking a pacing
   signal).
3. BOUNDARY CONDITION points -- points at the domain's spatial edges
   (x=0 and x=L), where we enforce zero-flux (Neumann) boundaries --
   physically, "no current escapes the ends of the fiber."
"""

from dataclasses import dataclass

import torch


@dataclass
class Domain:
    """The spatial and temporal extent of the simulation."""

    x_min: float = 0.0
    x_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0


def sample_collocation_points(n_points: int, domain: Domain) -> tuple:
    """
    Randomly samples interior (x, t) points where the PDE residual
    will be evaluated and minimized.

    Returns:
        (x, t): each shape (n_points, 1), with requires_grad=True
                 (required so autograd can differentiate through them).
    """
    x = torch.rand(n_points, 1) * (domain.x_max - domain.x_min) + domain.x_min
    t = torch.rand(n_points, 1) * (domain.t_max - domain.t_min) + domain.t_min
    x.requires_grad_(True)
    t.requires_grad_(True)
    return x, t


def sample_initial_condition(n_points: int, domain: Domain, stimulus_width: float = 0.1) -> tuple:
    """
    Samples points at t=0 with target (u0, w0) values representing a
    localized electrical stimulus near x=x_min (mimicking a pacing
    electrode triggering a wave at one end of the fiber).

    Returns:
        (x, t, u0, w0): x,t are the input coordinates (t is all zeros);
                         u0,w0 are the target values the network should
                         match at these points. All shape (n_points, 1).
    """
    x = torch.rand(n_points, 1) * (domain.x_max - domain.x_min) + domain.x_min
    t = torch.zeros(n_points, 1)

    # Localized stimulus: u0=1 near x_min, decaying to 0 -- triggers
    # a traveling wave that should propagate toward x_max over time.
    u0 = torch.where(x < domain.x_min + stimulus_width, torch.ones_like(x), torch.zeros_like(x))
    w0 = torch.zeros_like(x)  # recovery variable starts at rest everywhere

    x.requires_grad_(True)
    t.requires_grad_(True)
    return x, t, u0, w0


def sample_boundary_points(n_points: int, domain: Domain) -> tuple:
    """
    Samples points at both spatial boundaries (x=x_min and x=x_max),
    at random times, for enforcing the zero-flux Neumann condition
    (du/dx = 0 at the fiber's ends).

    Returns:
        (x, t): each shape (2*n_points, 1) -- n_points at each boundary,
                 with requires_grad=True.
    """
    t = torch.rand(2 * n_points, 1) * (domain.t_max - domain.t_min) + domain.t_min
    x_left = torch.full((n_points, 1), domain.x_min)
    x_right = torch.full((n_points, 1), domain.x_max)
    x = torch.cat([x_left, x_right], dim=0)

    x.requires_grad_(True)
    t.requires_grad_(True)
    return x, t
