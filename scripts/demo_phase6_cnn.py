"""
demo_phase6_cnn.py
--------------------
Connects everything built so far into one pipeline:
    Phase 5's real MIT-BIH record + annotations
        -> Phase 6's BeatDataset (windowed, labeled beats)
        -> Phase 6's ECG1DCNN (forward pass only -- no training yet,
           that's Phase 8)

This proves the architecture is wired correctly against REAL data
shapes before we invest time in a training loop.

Run with:
    python -m scripts.demo_phase6_cnn

NOTE: requires that Phase 5's demo has already been run at least once
(so the record is downloaded and sitting in data/raw/).
"""

from collections import Counter

import torch
from torch.utils.data import DataLoader

from src.data.beat_dataset import BeatDataset, INDEX_TO_AAMI_CLASS
from src.data.dataset_loader import ECGDatasetLoader
from src.models.cnn import ECG1DCNN
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
            f"Record '{record_name}' not found in {config.data_raw}. "
            "Run 'python -m scripts.demo_phase5_dataset_loading' first "
            "to download it."
        )
        return

    logger.info(f"Loaded record '{record_name}': {record.signal.shape[0]} samples, "
                f"{len(record.lead_names)} leads")

    # --- Build the beat-windowed dataset ---
    dataset = BeatDataset(record, annotations, window_before=90, window_after=90)
    logger.info(f"Extracted {len(dataset)} beat windows "
                f"(window_length={dataset.window_length} samples)")

    label_counts = Counter(dataset.labels)
    logger.info("Windowed dataset class distribution:")
    for idx, count in sorted(label_counts.items()):
        logger.info(f"  {INDEX_TO_AAMI_CLASS[idx]}: {count}")

    # --- Build a DataLoader and pull one batch ---
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    signals_batch, labels_batch = next(iter(dataloader))
    logger.info(f"Batch shapes -- signals: {tuple(signals_batch.shape)}, "
                f"labels: {tuple(labels_batch.shape)}")

    # --- Build the model and run one forward pass ---
    model = ECG1DCNN(
        in_channels=len(record.lead_names),
        window_length=dataset.window_length,
        num_classes=5,
    )
    logger.info(f"Model created with {model.count_parameters():,} trainable parameters")

    model.eval()
    with torch.no_grad():
        logits = model(signals_batch)

    logger.info(f"Forward pass output shape: {tuple(logits.shape)} "
                f"(expected: (batch_size={signals_batch.shape[0]}, num_classes=5))")
    logger.info("Forward pass successful -- architecture is correctly wired to real data.")


if __name__ == "__main__":
    main()
