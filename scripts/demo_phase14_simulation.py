"""
demo_phase14_simulation.py
------------------------------
Loads the trained PINN (Phase 12-13) and the anatomical heart mesh
(Phase 11), then simulates and visualizes cardiac activation
propagating across the 3D anatomy over time.

Run with:
    python -m scripts.demo_phase14_simulation

NOTE: if no PINN checkpoint exists yet, this will train one fresh
(same settings as Phase 12-13's demo) rather than fail.
"""

from pathlib import Path

import numpy as np
import torch

from src.anatomy.heart_mesh import generate_heart_mesh
from src.pinn.collocation import Domain
from src.pinn.fitzhugh_nagumo import FHNParameters
from src.pinn.pinn_model import PINNNet
from src.pinn.pinn_trainer import PINNTrainer
from src.simulation.heart_simulator import simulate_activation
from src.visualization.simulation_plot import plot_simulation_snapshots
from src.utils.config_loader import load_config
from src.utils.logger import get_logger


def main() -> None:
    config = load_config("configs/config.yaml")
    logger = get_logger(
        name=__name__,
        log_dir=str(config.logs),
        log_filename=config.log_filename,
        level=config.log_level,
        log_to_file=config.log_to_file,
    )

    domain = Domain(x_min=0.0, x_max=1.0, t_min=0.0, t_max=1.0)
    fhn_params = FHNParameters(D=0.1, a=0.1, epsilon=0.01, gamma=0.5)
    model = PINNNet(hidden_dim=64, n_hidden_layers=4)

    checkpoint_path = Path(config.checkpoint_dir) / "pinn_fhn_best.pt"
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        logger.info(f"Loaded PINN checkpoint from {checkpoint_path}")
    else:
        logger.info("No PINN checkpoint found -- training a fresh one now "
                     "(same as Phase 12-13's demo)...")
        trainer = PINNTrainer(model=model, domain=domain, fhn_params=fhn_params)
        trainer.fit(epochs=3000, log_every=1000, logger=logger)
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"Saved new PINN checkpoint to {checkpoint_path}")

    logger.info("Generating anatomical heart mesh...")
    mesh = generate_heart_mesh()

    logger.info("Simulating activation across the 3D heart mesh over time...")
    time_points = np.linspace(domain.t_min, domain.t_max, 6)
    result = simulate_activation(model, mesh, domain, time_points)

    logger.info(f"Activation range across simulation: "
                f"[{result.activation.min():.3f}, {result.activation.max():.3f}]")

    fig = plot_simulation_snapshots(result, n_cols=3)
    output_path = "logs/phase14_simulation_snapshots.png"
    fig.savefig(output_path, dpi=150)
    logger.info(f"Saved {output_path}")


if __name__ == "__main__":
    main()
