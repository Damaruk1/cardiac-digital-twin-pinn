"""
demo_phase8_training.py
--------------------------
Trains the CNN (Phase 6) on real MIT-BIH beat data, using stratified
splitting and class-weighted loss to address the imbalance problem
surfaced back in Phase 5.

Run with:
    python -m scripts.demo_phase8_training

NOTE: requires Phase 5's demo to have been run first.
"""

from torch.utils.data import DataLoader

from src.data.beat_dataset import BeatDataset
from src.data.dataset_loader import ECGDatasetLoader
from src.data.split import stratified_split
from src.models.cnn import ECG1DCNN
from src.training.class_weights import compute_class_weights
from src.training.trainer import Trainer
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

    loader = ECGDatasetLoader(data_dir=config.data_raw, database=config.database)
    record_name = config.sample_records[0]

    try:
        record = loader.load_record(record_name)
        annotations = loader.load_annotations(record_name)
    except FileNotFoundError:
        logger.error(
            f"Record '{record_name}' not found. Run "
            "'python -m scripts.demo_phase5_dataset_loading' first."
        )
        return

    dataset = BeatDataset(
        record, annotations,
        window_before=config.window_before,
        window_after=config.window_after,
    )
    logger.info(f"Total beats: {len(dataset)}")

    # --- Stratified split ---
    splits = stratified_split(
        dataset, train_frac=config.train_frac, val_frac=config.val_frac, logger=logger
    )
    logger.info(f"Split sizes -- train: {len(splits.train)}, val: {len(splits.val)}, "
                f"test: {len(splits.test)}")

    train_loader = DataLoader(splits.train, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(splits.val, batch_size=config.batch_size, shuffle=False)

    # --- Class weights, computed from the TRAINING split only ---
    train_labels = [dataset.labels[i] for i in splits.train.indices]
    class_weights = compute_class_weights(train_labels, num_classes=5)
    logger.info(f"Class weights (N,S,V,F,Q): {[round(w, 2) for w in class_weights.tolist()]}")

    # --- Build model and trainer ---
    model = ECG1DCNN(
        in_channels=len(record.lead_names),
        window_length=dataset.window_length,
        num_classes=5,
    )
    trainer = Trainer(
        model=model,
        num_classes=5,
        class_weights=class_weights,
        learning_rate=config.learning_rate,
        checkpoint_dir=config.checkpoint_dir,
    )

    logger.info(f"Training CNN for {config.epochs} epochs...")
    trainer.fit(train_loader, val_loader, epochs=config.epochs, model_name="cnn", logger=logger)
    logger.info(f"Training complete. Best checkpoint saved to {config.checkpoint_dir}/cnn_best.pt")


if __name__ == "__main__":
    main()
