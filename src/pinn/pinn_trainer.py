"""
pinn_trainer.py
-----------------
Trains a PINNNet by minimizing a composite loss:
    Loss = Loss_physics + Loss_IC + Loss_BC

Unlike Phase 8's Trainer, there is no "training data" in the labeled
sense -- every loss term is either the PDE residual (physics) or a
known constraint (initial/boundary conditions). This is the defining
characteristic of a PINN: the physics equation itself supervises training.
"""

from dataclasses import dataclass, field
from typing import List

import torch

from src.pinn.collocation import Domain, sample_boundary_points, sample_collocation_points, sample_initial_condition
from src.pinn.fitzhugh_nagumo import FHNParameters, compute_fhn_residuals
from src.pinn.pinn_model import PINNNet


@dataclass
class PINNTrainingHistory:
    """Tracks loss components across training for later inspection/plotting."""

    total_loss: List[float] = field(default_factory=list)
    physics_loss: List[float] = field(default_factory=list)
    ic_loss: List[float] = field(default_factory=list)
    bc_loss: List[float] = field(default_factory=list)


class PINNTrainer:
    """Trains a PINNNet against the FitzHugh-Nagumo equations."""

    def __init__(
        self,
        model: PINNNet,
        domain: Domain,
        fhn_params: FHNParameters,
        learning_rate: float = 1e-3,
        n_collocation: int = 2000,
        n_ic: int = 200,
        n_bc: int = 100,
    ):
        self.model = model
        self.domain = domain
        self.fhn_params = fhn_params
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.n_collocation = n_collocation
        self.n_ic = n_ic
        self.n_bc = n_bc

    def _compute_losses(self) -> tuple:
        """Draws fresh random collocation/IC/BC points and computes all
        three loss components. Resampling every step (rather than using
        a fixed point set) acts like a form of data augmentation -- the
        network can't just memorize a fixed grid."""

        # --- Physics loss: PDE residual should be zero everywhere ---
        x_col, t_col = sample_collocation_points(self.n_collocation, self.domain)
        r_u, r_w = compute_fhn_residuals(self.model, x_col, t_col, self.fhn_params)
        physics_loss = torch.mean(r_u**2) + torch.mean(r_w**2)

        # --- Initial condition loss: match the known starting state ---
        x_ic, t_ic, u0_target, w0_target = sample_initial_condition(self.n_ic, self.domain)
        output_ic = self.model(x_ic, t_ic)
        ic_loss = torch.mean((output_ic[:, 0:1] - u0_target) ** 2) + \
                  torch.mean((output_ic[:, 1:2] - w0_target) ** 2)

        # --- Boundary condition loss: zero-flux at both fiber ends ---
        x_bc, t_bc = sample_boundary_points(self.n_bc, self.domain)
        output_bc = self.model(x_bc, t_bc)
        u_bc = output_bc[:, 0:1]
        du_dx_bc = torch.autograd.grad(
            u_bc, x_bc, grad_outputs=torch.ones_like(u_bc), create_graph=True
        )[0]
        bc_loss = torch.mean(du_dx_bc**2)

        return physics_loss, ic_loss, bc_loss

    def fit(self, epochs: int, log_every: int = 100, logger=None) -> PINNTrainingHistory:
        """
        Runs the PINN training loop.

        Args:
            epochs: Number of optimization steps (there's no concept
                    of "one epoch through a dataset" here -- each step
                    draws fresh random points).
            log_every: How often to log progress.
            logger: Optional logger.

        Returns:
            PINNTrainingHistory with per-step loss values.
        """
        history = PINNTrainingHistory()

        for epoch in range(1, epochs + 1):
            self.optimizer.zero_grad()

            physics_loss, ic_loss, bc_loss = self._compute_losses()
            total_loss = physics_loss + ic_loss + bc_loss

            total_loss.backward()
            self.optimizer.step()

            history.total_loss.append(total_loss.item())
            history.physics_loss.append(physics_loss.item())
            history.ic_loss.append(ic_loss.item())
            history.bc_loss.append(bc_loss.item())

            if logger is not None and (epoch % log_every == 0 or epoch == 1):
                logger.info(
                    f"Epoch {epoch}/{epochs} -- total={total_loss.item():.5f} "
                    f"physics={physics_loss.item():.5f} ic={ic_loss.item():.5f} "
                    f"bc={bc_loss.item():.5f}"
                )

        return history
