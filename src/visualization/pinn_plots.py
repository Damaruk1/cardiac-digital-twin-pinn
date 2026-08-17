"""
pinn_plots.py
-------------
Visualizes a trained PINN's learned solution field u(x,t) as a
heatmap -- space on one axis, time on the other -- so a propagating
electrical wave is visible as a diagonal band of high u values
spreading from the stimulus site over time.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.pinn.collocation import Domain
from src.pinn.pinn_model import PINNNet
from src.pinn.pinn_trainer import PINNTrainingHistory


def plot_solution_heatmap(
    model: PINNNet,
    domain: Domain,
    resolution: int = 100,
    title: str = "PINN Solution: u(x,t)",
) -> plt.Figure:
    """
    Evaluates the trained model on a dense (x,t) grid and plots u as
    a heatmap.
    """
    x_vals = np.linspace(domain.x_min, domain.x_max, resolution)
    t_vals = np.linspace(domain.t_min, domain.t_max, resolution)
    X, T = np.meshgrid(x_vals, t_vals)

    x_flat = torch.tensor(X.flatten(), dtype=torch.float32).unsqueeze(1)
    t_flat = torch.tensor(T.flatten(), dtype=torch.float32).unsqueeze(1)

    model.eval()
    with torch.no_grad():
        output = model(x_flat, t_flat)
    u_grid = output[:, 0].numpy().reshape(resolution, resolution)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.pcolormesh(X, T, u_grid, cmap="hot", shading="auto")
    ax.set_xlabel("x (position along fiber)")
    ax.set_ylabel("t (time)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="u (transmembrane potential)")
    plt.tight_layout()
    return fig


def plot_training_history(history: PINNTrainingHistory) -> plt.Figure:
    """Plots how each loss component evolved during training, on a log scale
    (loss values often span several orders of magnitude for PINNs)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history.total_loss) + 1)

    ax.plot(epochs, history.total_loss, label="Total loss")
    ax.plot(epochs, history.physics_loss, label="Physics (PDE residual)")
    ax.plot(epochs, history.ic_loss, label="Initial condition")
    ax.plot(epochs, history.bc_loss, label="Boundary condition")

    ax.set_yscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("PINN Training Loss Components")
    ax.legend()
    plt.tight_layout()
    return fig
