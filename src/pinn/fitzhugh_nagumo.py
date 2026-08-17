"""
fitzhugh_nagumo.py
--------------------
Computes the FitzHugh-Nagumo PDE residuals for a PINN, using PyTorch
autograd to get the exact derivatives of the network's output with
respect to its inputs -- no finite-difference approximation, no mesh.

Governing equations:
    du/dt = D * d2u/dx2 + u(1-u)(u-a) - w      (reaction-diffusion)
    dw/dt = eps * (u - gamma * w)               (recovery dynamics)

A PERFECT solution would make both residuals exactly zero everywhere.
Training minimizes the residuals (not against labeled data -- against
the equation itself) at randomly sampled "collocation points".
"""

from dataclasses import dataclass

import torch

from src.pinn.pinn_model import PINNNet


@dataclass
class FHNParameters:
    """Physical parameters of the FitzHugh-Nagumo model."""

    D: float = 0.1       # diffusion coefficient (conduction speed control)
    a: float = 0.1       # excitation threshold
    epsilon: float = 0.01  # recovery time-scale (small = slow recovery)
    gamma: float = 0.5     # recovery coupling strength


def compute_fhn_residuals(
    model: PINNNet,
    x: torch.Tensor,
    t: torch.Tensor,
    params: FHNParameters,
) -> tuple:
    """
    Computes the PDE residuals r_u and r_w at the given (x, t) points.

    Args:
        model: The PINNNet.
        x: Spatial coordinates, shape (N, 1), MUST have requires_grad=True.
        t: Time coordinates, shape (N, 1), MUST have requires_grad=True.
        params: FHN physical parameters.

    Returns:
        (r_u, r_w): residual tensors, each shape (N, 1). Both should
                     approach zero as training converges.
    """
    output = model(x, t)
    u = output[:, 0:1]
    w = output[:, 1:2]

    # First derivatives, via autograd. create_graph=True is essential --
    # it keeps these derivatives part of the computational graph, so we
    # can differentiate them AGAIN (for d2u/dx2) and so gradients flow
    # back through them during the loss's own .backward() call.
    grad_outputs = torch.ones_like(u)

    du_dt = torch.autograd.grad(u, t, grad_outputs=grad_outputs, create_graph=True)[0]
    du_dx = torch.autograd.grad(u, x, grad_outputs=grad_outputs, create_graph=True)[0]
    d2u_dx2 = torch.autograd.grad(du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0]

    dw_dt = torch.autograd.grad(w, t, grad_outputs=grad_outputs, create_graph=True)[0]

    # The PDE residuals -- rearranged so a perfect solution gives r=0
    reaction_term = u * (1 - u) * (u - params.a)
    r_u = du_dt - params.D * d2u_dx2 - reaction_term + w
    r_w = dw_dt - params.epsilon * (u - params.gamma * w)

    return r_u, r_w
