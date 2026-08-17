"""
pinn_model.py
-------------
The neural network at the heart of the PINN: a simple MLP mapping
(x, t) -> (u, w), the transmembrane potential and recovery variable
of the FitzHugh-Nagumo cardiac model.

Unlike the CNN/Transformer earlier, this network is not "trained on
labeled data" in the usual sense -- it's trained so that ITS OWN
DERIVATIVES (computed via autograd) satisfy a differential equation.
The architecture itself is simple; all the interesting work happens
in how the loss is computed (see fitzhugh_nagumo.py).
"""

import torch
import torch.nn as nn


class PINNNet(nn.Module):
    """Fully-connected network mapping (x, t) -> (u, w)."""

    def __init__(self, hidden_dim: int = 64, n_hidden_layers: int = 4):
        """
        Args:
            hidden_dim: Width of each hidden layer.
            n_hidden_layers: Number of hidden layers.
        """
        super().__init__()

        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]  # input: (x, t)
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 2))  # output: (u, w)

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (N, 1) -- spatial coordinate.
            t: Tensor of shape (N, 1) -- time coordinate.

        Returns:
            Tensor of shape (N, 2): columns are [u, w].

        Note: Tanh activation (not ReLU) is standard for PINNs -- it's
        smooth and infinitely differentiable, which matters because we
        take SECOND derivatives of the network's output with respect
        to its inputs. ReLU's derivative is discontinuous at 0, which
        would make second derivatives zero almost everywhere -- fatal
        for a method that depends on those derivatives being meaningful.
        """
        inputs = torch.cat([x, t], dim=1)
        return self.network(inputs)
