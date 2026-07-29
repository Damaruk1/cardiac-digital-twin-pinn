"""
demo_phase7_transformer.py
-----------------------------
Runs the same real MIT-BIH beat data through BOTH the CNN (Phase 6)
and the Transformer (Phase 7), side by side -- comparing parameter
counts and forward-pass timing. Full accuracy comparison comes later
in Phase 9 (Evaluation), once both models are actually trained
(Phase 8).

Run with:
    python -m scripts.demo_phase7_transformer

NOTE: requires Phase 5's demo to have been run first (real data
downloaded into data/raw/).
"""

import time

import torch
from torch.utils.data import DataLoader

from src.data.beat_dataset import BeatDataset
from src.data.dataset_loader import ECGDatasetLoader
from src.models.cnn import ECG1DCNN
from src.models.transformer import ECGTransformer
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

    dataset = BeatDataset(record, annotations, window_before=90, window_after=90)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    signals_batch, labels_batch = next(iter(dataloader))
    logger.info(f"Using real batch: {tuple(signals_batch.shape)}")

    models = {
        "CNN": ECG1DCNN(
            in_channels=len(record.lead_names),
            window_length=dataset.window_length,
            num_classes=5,
        ),
        "Transformer": ECGTransformer(
            in_channels=len(record.lead_names),
            window_length=dataset.window_length,
            num_classes=5,
        ),
    }

    logger.info(f"{'Model':<12} {'Parameters':>12} {'Forward pass time':>20}")
    logger.info("-" * 46)

    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            start = time.perf_counter()
            logits = model(signals_batch)
            elapsed_ms = (time.perf_counter() - start) * 1000

        assert logits.shape == (signals_batch.shape[0], 5), f"{name} produced wrong output shape!"

        logger.info(f"{name:<12} {model.count_parameters():>12,} {elapsed_ms:>17.2f}ms")

    logger.info("Both architectures verified working on identical real data batches.")
    logger.info("Training + fair accuracy comparison happens in Phase 8-9.")


if __name__ == "__main__":
    main()
