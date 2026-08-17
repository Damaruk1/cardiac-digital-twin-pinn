"""
demo_phase12_13_pinn.py
--------------------------
Trains a Physics-Informed Neural Network to solve the FitzHugh-Nagumo
cardiac electrical propagation equation in 1D space + time, with NO
labeled training data -- only the PDE itself, an initial stimulus, and
insulated boundary conditions.

Run with:
    python -m scripts.demo_phase12_13_pinn
"""

from src.pinn.collocation import Domain
from src.pinn.fitzhugh_nagumo import FHNParameters
from src.pinn.pinn_model import PINNNet
from src.pinn.pinn_trainer import PINNTrainer
from src.visualization.pinn_plots import plot_solution_heatmap, plot_training_history
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
    trainer = PINNTrainer(
        model=model,
        domain=domain,
        fhn_params=fhn_params,
        learning_rate=1e-3,
        n_collocation=2000,
        n_ic=200,
        n_bc=100,
    )

    logger.info("Training PINN on FitzHugh-Nagumo equations (no labeled data -- physics only)...")
    logger.info(f"Domain: x in [{domain.x_min}, {domain.x_max}], t in [{domain.t_min}, {domain.t_max}]")

    history = trainer.fit(epochs=3000, log_every=500, logger=logger)

    logger.info(f"Final losses -- physics={history.physics_loss[-1]:.5f}, "
                f"ic={history.ic_loss[-1]:.5f}, bc={history.bc_loss[-1]:.5f}")
    logger.info(f"Physics loss reduced by {history.physics_loss[0] / max(history.physics_loss[-1], 1e-10):.1f}x "
                f"from initial random-init value.")

    fig_solution = plot_solution_heatmap(model, domain, title="Learned Cardiac Wave Propagation u(x,t)")
    fig_solution.savefig("logs/phase12_13_pinn_solution.png", dpi=150)
    logger.info("Saved logs/phase12_13_pinn_solution.png")

    fig_history = plot_training_history(history)
    fig_history.savefig("logs/phase12_13_pinn_training_history.png", dpi=150)
    logger.info("Saved logs/phase12_13_pinn_training_history.png")


if __name__ == "__main__":
    main()
